# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout",
#   "fire",
#   "gymnasium[box2d]",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.1.32",
#   "tqdm"
# ]
# ///

from __future__ import annotations

from fire import Fire
from shutil import rmtree
import os

import torch
from torch import from_numpy, cat, tensor
import torch.nn.functional as F

from tqdm import tqdm

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning
)

from x_mlps_pytorch import ResidualNormedMLP

from discrete_continuous_embed_readout import Readout

from dashboard import Dashboard

from accelerate import Accelerator
from collections import deque

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
    buffer_size = 512,
    video_folder = './recordings',
    render_every_eps = None,
    dim_contrastive_embed = 32,
    cl_train_steps = 5000,
    cl_batch_size = 256,
    actor_batch_size = 128,
    actor_num_train_steps = 1000,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    train_critic_soft_one_hot = False,
    repetition_factor = 1,
    use_sigmoid_contrastive_learning = True,
    sigmoid_bias = -5.,
    cl_l2norm_embed = True,
    use_wandb = False,
    cpu = True
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

    # create env

    env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

    # recording

    rmtree(video_folder, ignore_errors = True)

    render_every_eps = default(render_every_eps, num_episodes_before_learn)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'lunar',
        episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
        disable_logger = True
    )

    dim_state = 8
    dim_action = 4

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-lunar-discrete',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            action = 'int',
            action_soft_one_hot = ('float', dim_action)
        ),
        circular = True,
        overwrite = True
    )

    # model

    actor_encoder = ResidualNormedMLP(
        dim_in = dim_state + dim_state, # state and goal
        dim = 64,
        depth = 4,
        residual_every = 2,
        dim_out = dim_action,
        keel_post_ln = True
    )

    actor_readout = Readout(num_discrete = 4, dim = 0)

    critic_encoder = ResidualNormedMLP(
        dim_in = dim_state + dim_action,
        dim = 64,
        dim_out = dim_contrastive_embed,
        depth = 8,
        residual_every = 4,
        keel_post_ln = True
    )

    goal_encoder = ResidualNormedMLP(
        dim_in = dim_state,
        dim = 64,
        dim_out = dim_contrastive_embed,
        depth = 8,
        residual_every = 4,
        keel_post_ln = True
    )

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
        repetition_factor = repetition_factor,
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
        softmax_actor_output = True,
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    actor_goal = tensor([0., 0., 0., 0., 0., 0., 1., 1.], device = module_device(actor_encoder))

    # episodes

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)

    dashboard = Dashboard(num_episodes, title = "Contrastive RL - Lunar Lander", env_name = "LunarLander-v3", hyperparams = dict(
        critic_learning_rate = critic_learning_rate,
        actor_learning_rate = actor_learning_rate,
        cl_batch_size = cl_batch_size,
        actor_batch_size = actor_batch_size,
        buffer_size = buffer_size,
        max_timesteps = max_timesteps,
        train_critic_soft_one_hot = train_critic_soft_one_hot,
        repetition_factor = repetition_factor,
        use_sigmoid_contrastive_learning = use_sigmoid_contrastive_learning
    ))

    with dashboard.create_renderable() as live:
        for eps in range(num_episodes):

            state, *_ = env.reset()

            cum_reward = 0.
            eps_steps = 0
            cl_loss = 0.
            actor_loss = 0.

            with replay_buffer.one_episode():

                for _ in range(max_timesteps):

                    actor_encoder.eval()
                    action_logits = actor_encoder(cat((from_numpy(state).to(module_device(actor_encoder)), actor_goal), dim = -1))

                    action = actor_readout.sample(action_logits)

                    next_state, reward, terminated, truncated, *_ = env.step(action.cpu().numpy())

                    cum_reward += reward
                    eps_steps += 1

                    done = truncated or terminated

                    replay_buffer.store(
                        state = state,
                        action = action,
                        action_soft_one_hot = action_logits.softmax(dim = -1)
                    )

                    if done:
                        break

                    state = next_state

                rolling_reward.append(cum_reward)
                rolling_steps.append(eps_steps)

                dashboard.update_diagnostics(
                    last_eps_reward = f"{cum_reward:.2f}",
                    last_eps_steps = eps_steps
                )

                live.update(dashboard.render())

            # train the critic with contrastive learning

            # train the critic with contrastive learning

            if (eps + 1) >= num_episodes_before_learn and divisible_by(eps + 1, num_episodes_before_learn):

                data = replay_buffer.get_all_data(
                    fields = ['state', 'action', 'action_soft_one_hot'],
                    meta_fields = ['episode_lens']
                )

                trajectories = data['state']
                episode_lens = data['episode_lens']

                if train_critic_soft_one_hot:
                    actions_for_critic = data['action_soft_one_hot']
                else:
                    actions_for_critic = F.one_hot(data['action'].long(), num_classes = 4)

                # filter out episodes that are too short
                keep_mask = episode_lens >= 2
                trajectories = trajectories[keep_mask]
                episode_lens = episode_lens[keep_mask]
                actions_for_critic = actions_for_critic[keep_mask]

                if episode_lens.shape[0] > 0:
                    cl_loss = critic_trainer(
                        trajectories,
                        cl_train_steps,
                        lens = episode_lens,
                        actions = actions_for_critic
                    )

                    actor_loss = actor_trainer(
                        trajectories,
                        actor_num_train_steps,
                        lens = episode_lens,
                        sample_fn = actor_readout.sample
                    )

                dashboard.update_diagnostics(
                    critic_loss = f"{cl_loss:.4f}",
                    actor_loss = f"{actor_loss:.4f}"
                )
            dashboard.advance_progress()

            avg_reward = sum(rolling_reward) / len(rolling_reward)
            avg_steps = sum(rolling_steps) / len(rolling_steps)

            dashboard.update_episode_info(
                avg_cum_reward_100 = f"{avg_reward:.2f}",
                avg_steps_100 = f"{avg_steps:.1f}"
            )

            if use_wandb:
                accelerator.log({
                    "avg_cum_reward_100": avg_reward,
                    "avg_steps_100": avg_steps,
                    "last_eps_reward": cum_reward,
                    "critic_loss": cl_loss if 'cl_loss' in locals() else 0.,
                    "actor_loss": actor_loss if 'actor_loss' in locals() else 0.
                })

            live.update(dashboard.render())

    if use_wandb:
        accelerator.end_training()

# fire

if __name__ == '__main__':
    Fire(main)
