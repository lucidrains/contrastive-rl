# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout",
#   "fire",
#   "gymnasium[box2d]",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.3.0",
#   "tqdm"
# ]
# ///

from __future__ import annotations

import os
from fire import Fire
from shutil import rmtree
from collections import deque
from functools import partial

import torch
from torch import from_numpy, cat, tensor
import torch.nn.functional as F

import numpy as np
from einops import rearrange

from tqdm import tqdm
import gymnasium as gym
from accelerate import Accelerator

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
    sample_random_state
)

from x_mlps_pytorch import ResidualNormedMLP, AttnResidualNormedMLP
from discrete_continuous_embed_readout import Readout

from dashboard import Dashboard

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def module_device(m):
    return next(m.parameters()).device

# main

def main(
    num_episodes = 50_000,
    max_timesteps = 500,
    num_episodes_before_learn = 128,
    buffer_size = 512,
    video_folder = './recordings',
    render_every_eps = None,
    dim_contrastive_embed = 64,
    cl_train_steps = 2_500,
    cl_batch_size = 64,
    actor_batch_size = 128,
    actor_num_train_steps = 1000,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    actor_dim = 64,
    actor_depth = 4,
    critic_dim = 64,
    critic_depth = 8,
    goal_dim = 64,
    goal_depth = 8,
    weight_decay = 1e-4,
    max_grad_norm = 0.5,
    train_critic_soft_one_hot = False,
    repetition_factor = 2,
    use_sigmoid_contrastive_learning = True,
    sigmoid_bias = -5.,
    cl_l2norm_embed = True,
    exploration_random_goal_prob = 0.025,
    exploration_sample_from_buffer_prob = 0.5,
    reward_part_of_goal = False,
    reward_norm = 100.,
    reward_fourier_encode = False,
    reward_fourier_dim = 16,
    use_attn_residual_mlp = True,
    use_wandb = False,
    cpu = False
):
    # clear video folder

    rmtree(video_folder, ignore_errors = True)
    os.makedirs(video_folder, exist_ok = True)

    # accelerator

    accelerator = Accelerator(
        log_with = 'wandb' if use_wandb else None,
        cpu = cpu
    )

    if use_wandb:
        accelerator.init_trackers(
            project_name = 'contrastive-rl',
            config = locals()
        )

    # env

    env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

    # recording

    render_every_eps = default(render_every_eps, num_episodes_before_learn)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'lunar',
        episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
        disable_logger = True
    )

    dim_state = 8
    dim_goal = 8
    if reward_part_of_goal:
        dim_goal += reward_fourier_dim if reward_fourier_encode else 1
    dim_action = 4

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-lunar-discrete',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            reward = ('float', 1),
            action_hard_one_hot = ('float', dim_action),
            action_soft_one_hot = ('float', dim_action)
        ),
        circular = True,
        overwrite = True
    )

    # model

    device = accelerator.device

    if use_attn_residual_mlp:
        MLP = AttnResidualNormedMLP
    else:
        MLP = partial(ResidualNormedMLP, residual_every = 4, keel_post_ln = True)

    actor_encoder = MLP(
        dim_in = dim_state + dim_goal, # state and goal
        dim = actor_dim,
        depth = actor_depth,
        dim_out = dim_action
    ).to(device)

    actor_readout = Readout(num_discrete = dim_action, dim = 0)

    critic_encoder = MLP(
        dim_in = dim_state + dim_action,
        dim = critic_dim,
        dim_out = dim_contrastive_embed,
        depth = critic_depth
    ).to(device)

    goal_encoder = MLP(
        dim_in = dim_goal,
        dim = goal_dim,
        dim_out = dim_contrastive_embed,
        depth = goal_depth
    ).to(device)

    # contrastive learning module

    if use_sigmoid_contrastive_learning:
        contrastive_learn = SigmoidContrastiveLearning(bias = sigmoid_bias, l2norm_embed = cl_l2norm_embed)
    else:
        contrastive_learn = ContrastiveLearning(l2norm_embed = True, learned_temp = True)

    critic_trainer = ContrastiveRLTrainer(
        critic_encoder,
        goal_encoder,
        batch_size = cl_batch_size,
        learning_rate = critic_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        repetition_factor = repetition_factor,
        reward_part_of_goal = reward_part_of_goal,
        reward_norm = reward_norm,
        reward_fourier_encode = reward_fourier_encode,
        reward_fourier_dim = reward_fourier_dim,
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    # assertions

    assert num_episodes_before_learn > cl_batch_size

    actor_trainer = ActorTrainer(
        actor_encoder,
        critic_encoder,
        goal_encoder,
        batch_size = actor_batch_size,
        learning_rate = actor_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        softmax_actor_output = True,
        reward_part_of_goal = reward_part_of_goal,
        reward_norm = reward_norm,
        reward_fourier_encode = reward_fourier_encode,
        reward_fourier_dim = reward_fourier_dim,
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    base_actor_goal = tensor([0., 0., 0., 0., 0., 0., 1., 1.], device = device)
    max_step_reward = 1.0

    # episodes

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)

    dashboard = Dashboard(
        num_episodes,
        title = "Contrastive RL - Lunar Lander",
        env_name = "LunarLander-v3",
        hyperparams = dict(
            critic_learning_rate = critic_learning_rate,
            actor_learning_rate = actor_learning_rate,
            cl_batch_size = cl_batch_size,
            actor_batch_size = actor_batch_size,
            buffer_size = f"{buffer_size}",
            max_timesteps = f"{max_timesteps}",
            weight_decay = f"{weight_decay}",
            max_grad_norm = f"{max_grad_norm}",
            train_critic_soft_one_hot = f"{train_critic_soft_one_hot}",
            repetition_factor = repetition_factor,
            use_sigmoid_contrastive_learning = use_sigmoid_contrastive_learning,
            exploration_random_goal_prob = exploration_random_goal_prob,
            exploration_sample_from_buffer_prob = exploration_sample_from_buffer_prob,
            reward_part_of_goal = reward_part_of_goal,
            reward_norm = reward_norm,
            reward_fourier_encode = reward_fourier_encode,
            reward_fourier_dim = reward_fourier_dim,
            use_attn_residual_mlp = use_attn_residual_mlp
        )
    )

    with dashboard.create_renderable() as live:
        for eps in range(num_episodes):

            state, *_ = env.reset()

            cum_reward = 0.
            eps_steps = 0
            cl_loss = 0.
            actor_loss = 0.

            # decide on goal for the episode

            is_exploring = torch.rand((), device = device) < exploration_random_goal_prob

            eps_goal = base_actor_goal

            if reward_part_of_goal:
                max_reward_tensor = tensor([max_step_reward / reward_norm], device = device, dtype = torch.float32)
                if reward_fourier_encode:
                    actor_trainer.reward_fourier_encode = actor_trainer.reward_fourier_encode.to(device)
                    max_reward_tensor = actor_trainer.reward_fourier_encode(max_reward_tensor)
                    max_reward_tensor = rearrange(max_reward_tensor, '1 d -> d')
                eps_goal = cat((eps_goal, max_reward_tensor), dim = -1)

            if is_exploring:
                eps_goal = sample_random_state(
                    replay_buffer,
                    env,
                    exploration_sample_from_buffer_prob
                ).to(device)

                if reward_part_of_goal and eps_goal.shape[-1] == dim_state:
                    rand_reward = torch.rand((1,), device = device, dtype = torch.float32)

                    if reward_fourier_encode:
                        actor_trainer.reward_fourier_encode = actor_trainer.reward_fourier_encode.to(device)
                        rand_reward = actor_trainer.reward_fourier_encode(rand_reward)
                        rand_reward = rearrange(rand_reward, '1 d -> d')

                    eps_goal = cat((eps_goal, rand_reward), dim = -1)

            states = []
            rewards = []
            hard_one_hots = []
            soft_one_hots = []

            for _ in range(max_timesteps):

                actor_encoder.eval()

                curr_state = from_numpy(state).to(device)

                action_logits = actor_encoder(cat((curr_state, eps_goal), dim = -1))

                action = actor_readout.sample(action_logits)

                next_state, reward, terminated, truncated, *_ = env.step(action.cpu().numpy())

                # store transition data

                states.append(state)
                rewards.append(reward)
                hard_one_hots.append(F.one_hot(action.long(), num_classes = dim_action).float().detach().cpu())
                soft_one_hots.append(action_logits.softmax(dim = -1).detach().cpu())

                cum_reward += reward
                eps_steps += 1

                done = truncated or terminated

                if done:
                    break

                state = next_state

            if len(rewards) > 0:
                max_step_reward = max(max_step_reward, max(rewards))

            # store episode if length >= 2

            if len(states) >= 2:
                replay_buffer.store_episode(
                    state = states,
                    reward = rewards,
                    action_hard_one_hot = hard_one_hots,
                    action_soft_one_hot = soft_one_hots
                )

            if not is_exploring:
                rolling_reward.append(cum_reward)
                rolling_steps.append(eps_steps)

                dashboard.update_metrics(
                    last_eps_reward = f"{cum_reward:.2f}",
                    last_eps_steps = eps_steps
                )

            live.update(dashboard.render())

            # train the critic and actor

            if (eps + 1) >= num_episodes_before_learn and divisible_by(eps + 1, num_episodes_before_learn):

                data = replay_buffer.get_all_data(
                    fields = ['state', 'reward', 'action_hard_one_hot', 'action_soft_one_hot'],
                    meta_fields = ['episode_lens']
                )

                trajectories = data['state']
                rewards_for_critic = data['reward']
                episode_lens = data['episode_lens']

                if train_critic_soft_one_hot:
                    actions_for_critic = data['action_soft_one_hot']
                else:
                    actions_for_critic = data['action_hard_one_hot']

                cl_loss = critic_trainer(
                    trajectories,
                    cl_train_steps,
                    lens = episode_lens,
                    actions = actions_for_critic,
                    rewards = rewards_for_critic,
                    pbar = dashboard.critic_pbar
                )

                actor_loss = actor_trainer(
                    trajectories,
                    actor_num_train_steps,
                    lens = episode_lens,
                    rewards = rewards_for_critic,
                    sample_fn = actor_readout.sample,
                    pbar = dashboard.actor_pbar
                )

                dashboard.update_metrics(
                    critic_loss = f"{cl_loss:.4f}",
                    actor_loss = f"{actor_loss:.4f}"
                )

            dashboard.advance_progress()

            if not is_exploring:
                avg_reward = sum(rolling_reward) / len(rolling_reward) if len(rolling_reward) > 0 else 0.
                avg_steps = sum(rolling_steps) / len(rolling_steps) if len(rolling_steps) > 0 else 0.

                dashboard.update_metrics(
                    avg_cum_reward_100 = f"{avg_reward:.2f}",
                    avg_steps_100 = f"{avg_steps:.1f}"
                )

                if use_wandb:
                    accelerator.log({
                        "avg_cum_reward_100": avg_reward,
                        "avg_steps_100": avg_steps,
                        "last_eps_reward": cum_reward,
                        "critic_loss": cl_loss,
                        "actor_loss": actor_loss
                    })

            live.update(dashboard.render())

    if use_wandb:
        accelerator.end_training()

# fire

if __name__ == '__main__':
    Fire(main)
