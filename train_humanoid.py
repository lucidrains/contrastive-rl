# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout",
#   "fire",
#   "gymnasium[mujoco]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.1.32",
#   "tqdm"
# ]
# ///

from __future__ import annotations

import os
import numpy as np
from fire import Fire
from shutil import rmtree
from pathlib import Path
from collections import deque

import torch
from torch import nn, from_numpy, cat, tensor, zeros_like
import torch.nn.functional as F

from tqdm import tqdm
import gymnasium as gym
from accelerate import Accelerator

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning
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
    num_episodes = 2_000_000,
    max_timesteps = 1000,
    num_episodes_before_learn = 512,
    buffer_size = 512,
    video_folder = './recordings_humanoid',
    render_every_eps = None,
    dim_contrastive_embed = 64,
    cl_train_steps = 10000,
    cl_batch_size = 256,
    actor_batch_size = 128,
    actor_num_train_steps = 1000,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    repetition_factor = 2,
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

    # env

    env = gym.make('Humanoid-v5', render_mode = 'rgb_array')

    # recording

    render_every_eps = default(render_every_eps, num_episodes_before_learn)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'humanoid',
        episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
        disable_logger = True
    )

    # dims

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-humanoid',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', obs_dim),
            action = ('float', action_dim),
        ),
        circular = True,
        overwrite = True
    )

    # model

    device = torch.device('cpu') if cpu else torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    actor_encoder = nn.Sequential(
        ResidualNormedMLP(
            dim_in = obs_dim * 2,
            dim = 256,
            depth = 16,
            dim_out = action_dim * 2,
            keel_post_ln = True
        ),
        Rearrange('... (action mu_logvar) -> ... action mu_logvar', mu_logvar = 2)
    ).to(device)

    actor_readout = Readout(num_continuous = action_dim, continuous_squashed = False, dim = 0)

    critic_encoder = ResidualNormedMLP(
        dim_in = obs_dim + action_dim,
        dim = 256,
        dim_out = dim_contrastive_embed,
        depth = 16,
        residual_every = 4,
        keel_post_ln = True
    ).to(device)

    goal_encoder = ResidualNormedMLP(
        dim_in = obs_dim,
        dim = 256,
        dim_out = dim_contrastive_embed,
        depth = 16,
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
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    # setup actor goal
    # for Humanoid-v5 (default):
    # obs[0] is torso height (qpos[2])
    # obs[1:5] is torso orientation (qpos[3:7])
    # obs[22] is x-velocity (qvel[0])

    actor_goal = torch.zeros(obs_dim, device = device)
    actor_goal[0] = 1.3  # target height
    actor_goal[1] = 1.0  # target orientation
    actor_goal[22] = 1.0 # target x-velocity

    # episodes

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)

    dashboard = Dashboard(
        num_episodes,
        title = "Contrastive RL - Humanoid",
        env_name = "Humanoid-v5",
        hyperparams = dict(
            critic_learning_rate = critic_learning_rate,
            actor_learning_rate = actor_learning_rate,
            cl_batch_size = cl_batch_size,
            actor_batch_size = actor_batch_size,
            buffer_size = buffer_size,
            max_timesteps = max_timesteps,
            repetition_factor = repetition_factor,
            use_sigmoid_contrastive_learning = use_sigmoid_contrastive_learning
        )
    )

    with dashboard.create_renderable() as live:
        for eps in range(num_episodes):

            state, *_ = env.reset()

            cum_reward = 0.
            eps_steps = 0
            cl_loss = 0.
            actor_loss = 0.

            states = []
            actions = []

            for _ in range(max_timesteps):

                state = state.astype(np.float32)

                actor_encoder.eval()

                action_logits = actor_encoder(cat((from_numpy(state).to(device), actor_goal), dim = -1))

                action = actor_readout.sample(action_logits)

                next_state, reward, terminated, truncated, *_ = env.step(action.detach().cpu().numpy())

                # store transition data

                states.append(state)
                actions.append(action.detach().cpu())

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
                    action = actions
                )

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
                    fields = ['state', 'action'],
                    meta_fields = ['episode_lens']
                )

                trajectories = data['state']
                episode_lens = data['episode_lens']
                actions_for_critic = data['action']

                cl_loss = critic_trainer(
                    trajectories,
                    cl_train_steps,
                    lens = episode_lens,
                    actions = actions_for_critic
                )

                actor_loss = actor_trainer(
                    trajectories,
                    num_train_steps = actor_num_train_steps,
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
                    "critic_loss": cl_loss,
                    "actor_loss": actor_loss
                })

            live.update(dashboard.render())

    if use_wandb:
        accelerator.end_training()

# fire

if __name__ == '__main__':
    Fire(main)
