# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout>=0.2.0",
#   "fire",
#   "gymnasium[box2d]",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.1.32",
#   "tqdm"
# ]
# ///

from __future__ import annotations

import os
from fire import Fire
from shutil import rmtree
from collections import deque

import torch
from torch import nn, from_numpy, cat, tensor
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

from einops.layers.torch import Rearrange
from x_mlps_pytorch import ResidualNormedMLP
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
    num_episodes_before_learn = 512,
    buffer_size = 1536,
    video_folder = './recordings',
    render_every_eps = None,
    dim_contrastive_embed = 64,
    cl_train_steps = 2500,
    cl_batch_size = 256,
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
    repetition_factor = 2,
    use_sigmoid_contrastive_learning = True,
    sigmoid_bias = -5.,
    cl_l2norm_embed = True,
    exploration_random_goal_prob = 0.025,
    exploration_sample_from_buffer_prob = 0.5,
    reward_part_of_goal = False,
    reward_norm = 100.,
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

    env = gym.make('LunarLander-v3', continuous = True, render_mode = 'rgb_array')

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
    dim_goal = 8 + (1 if reward_part_of_goal else 0)
    dim_action = 2

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-lunar-continuous',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            action = ('float', dim_action),
            reward = ('float', 1),
        ),
        circular = True,
        overwrite = True
    )

    # model

    device = accelerator.device

    actor_encoder = nn.Sequential(
        ResidualNormedMLP(
            dim_in = dim_state + dim_goal, # state and goal
            dim = actor_dim,
            depth = actor_depth,
            residual_every = 2,
            dim_out = dim_action * 2, # for kumaraswamy alpha and beta
            keel_post_ln = True
        ),
        Rearrange('... (action mu_logvar) -> ... action mu_logvar', mu_logvar = 2)
    ).to(device)

    actor_readout = Readout(
        num_continuous = dim_action,
        continuous_dist_type = 'kumaraswamy',
        continuous_dist_kwargs = dict(unimodal = True),
        dim = 0
    )

    critic_encoder = ResidualNormedMLP(
        dim_in = dim_state + dim_action,
        dim = critic_dim,
        dim_out = dim_contrastive_embed,
        depth = critic_depth,
        residual_every = 4,
        keel_post_ln = True
    ).to(device)

    goal_encoder = ResidualNormedMLP(
        dim_in = dim_goal,
        dim = goal_dim,
        dim_out = dim_contrastive_embed,
        depth = goal_depth,
        residual_every = 4,
        keel_post_ln = True
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
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    # assertions

    assert num_episodes_before_learn > cl_batch_size

    def sample_fn(logits, differentiable = False):
        action = actor_readout.sample(logits, differentiable = differentiable)
        return action * 2 - 1

    actor_trainer = ActorTrainer(
        actor_encoder,
        critic_encoder,
        goal_encoder,
        batch_size = actor_batch_size,
        learning_rate = actor_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        softmax_actor_output = False,
        reward_part_of_goal = reward_part_of_goal,
        reward_norm = reward_norm,
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    actor_goal = tensor([0., 0., 0., 0., 0., 0., 1., 1.], device = device)

    if reward_part_of_goal:
        max_reward = tensor([1.], device = device, dtype = torch.float32)
        actor_goal = cat((actor_goal, max_reward), dim = -1)

    # episodes

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)

    dashboard = Dashboard(
        num_episodes,
        title = "Contrastive RL - Lunar Lander (Continuous)",
        env_name = "LunarLanderContinuous-v3",
        hyperparams = dict(
            critic_learning_rate = critic_learning_rate,
            actor_learning_rate = actor_learning_rate,
            cl_batch_size = cl_batch_size,
            actor_batch_size = actor_batch_size,
            buffer_size = f"{buffer_size}",
            max_timesteps = f"{max_timesteps}",
            weight_decay = f"{weight_decay}",
            max_grad_norm = f"{max_grad_norm}",
            repetition_factor = f"{repetition_factor}",
            use_sigmoid_contrastive_learning = use_sigmoid_contrastive_learning,
            exploration_random_goal_prob = exploration_random_goal_prob,
            exploration_sample_from_buffer_prob = exploration_sample_from_buffer_prob
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

            is_exploring = torch.rand(()) < exploration_random_goal_prob

            eps_goal = actor_goal

            if is_exploring:
                eps_goal = sample_random_state(
                    replay_buffer,
                    env,
                    exploration_sample_from_buffer_prob
                ).to(device)

                if reward_part_of_goal and eps_goal.shape[-1] == dim_state:
                    rand_reward = torch.rand((1,), device = device, dtype = torch.float32)
                    eps_goal = cat((eps_goal, rand_reward), dim = -1)

            states = []
            actions = []
            rewards = []

            for _ in range(max_timesteps):

                actor_encoder.eval()

                curr_state = from_numpy(state).to(device)

                action_logits = actor_encoder(cat((curr_state, eps_goal), dim = -1))

                action = sample_fn(action_logits)

                next_state, reward, terminated, truncated, *_ = env.step(action.detach().cpu().numpy())

                # store transition data

                states.append(state)
                actions.append(action.detach().cpu())
                rewards.append(reward)

                cum_reward += reward
                eps_steps += 1

                done = truncated or terminated

                if done:
                    break

                state = next_state

            # store episode if length >= 2

            if len(states) >= 2:
                replay_buffer.store_episode(
                    state = states,
                    action = actions,
                    reward = rewards
                )

            if not is_exploring:
                rolling_reward.append(cum_reward)
                rolling_steps.append(eps_steps)

                dashboard.update_diagnostics(
                    last_eps_reward = f"{cum_reward:.2f}",
                    last_eps_steps = eps_steps
                )

            live.update(dashboard.render())

            # train the critic and actor

            if (eps + 1) >= num_episodes_before_learn and divisible_by(eps + 1, num_episodes_before_learn):

                data = replay_buffer.get_all_data(
                    fields = ['state', 'action', 'reward'],
                    meta_fields = ['episode_lens']
                )

                trajectories = data['state']
                episode_lens = data['episode_lens']
                actions_for_critic = data['action']
                rewards_for_critic = data['reward']

                cl_loss = critic_trainer(
                    trajectories,
                    cl_train_steps,
                    lens = episode_lens,
                    actions = actions_for_critic,
                    rewards = rewards_for_critic
                )

                actor_loss = actor_trainer(
                    trajectories,
                    actor_num_train_steps,
                    lens = episode_lens,
                    rewards = rewards_for_critic,
                    sample_fn = lambda logits: sample_fn(logits, differentiable = True)
                )

                dashboard.update_diagnostics(
                    critic_loss = f"{cl_loss:.4f}",
                    actor_loss = f"{actor_loss:.4f}"
                )

            dashboard.advance_progress()

            if not is_exploring:
                avg_reward = sum(rolling_reward) / len(rolling_reward) if len(rolling_reward) > 0 else 0.
                avg_steps = sum(rolling_steps) / len(rolling_steps) if len(rolling_steps) > 0 else 0.

                dashboard.update_episode_info(
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
