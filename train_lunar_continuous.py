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

import torch
from torch import nn
from torch import from_numpy, tensor
import torch.nn.functional as F

from tqdm import tqdm

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveWrapper,
    ContrastiveRLTrainer,
    ActorTrainer
)

from einops.layers.torch import Rearrange

from x_mlps_pytorch import ResidualNormedMLP

from discrete_continuous_embed_readout import Readout

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
    num_episodes = 100_000,
    max_timesteps = 500,
    num_episodes_before_learn = 250,
    buffer_size = 5_000,
    video_folder = './recordings',
    render_every_eps = None,
    dim_contrastive_embed = 32,
    cl_train_steps = 2_000,
    cl_batch_size = 256,
    actor_batch_size = 16,
    actor_num_train_steps = 500,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    cpu = True
):

    # create env

    env = gym.make('LunarLander-v3', continuous = True, render_mode = 'rgb_array')

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

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-lunar',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', 8),
            action = ('float', 2),
        ),
        circular = True,
        overwrite = True
    )

    # model

    actor_encoder = nn.Sequential(
        ResidualNormedMLP(
            dim_in = 8,
            dim = 32,
            depth = 8,
            dim_out = 4
        ),
        Rearrange('... (action mu_logvar) -> ... action mu_logvar', mu_logvar = 2)
    )

    actor_readout = Readout(num_continuous = 2, continuous_squashed = True, dim = 0)

    critic_encoder = ResidualNormedMLP(
        dim_in = 8 + 2,
        dim = 64,
        dim_out = dim_contrastive_embed,
        depth = 16,
        residual_every = 4,
    )

    goal_encoder = ResidualNormedMLP(
        dim_in = 8,
        dim = 64,
        dim_out = dim_contrastive_embed,
        depth = 16,
        residual_every = 4
    )

    critic_trainer = ContrastiveRLTrainer(
        critic_encoder,
        goal_encoder,
        batch_size = cl_batch_size,
        learning_rate = critic_learning_rate,
        cpu = cpu
    )

    actor_trainer = ActorTrainer(
        actor_encoder,
        critic_encoder,
        goal_encoder,
        batch_size = actor_batch_size,
        learning_rate = actor_learning_rate,
        cpu = cpu,
    )

    actor_goal = tensor([0., 0., 0., 0., 0., 0., 1., 1.], device = module_device(actor_encoder))

    # episodes

    pbar = tqdm(range(num_episodes), desc = 'episodes')
    for eps in pbar:

        state, *_ = env.reset()

        cum_reward = 0.

        with replay_buffer.one_episode():

            for _ in range(max_timesteps):

                action_logits = actor_encoder((from_numpy(state), actor_goal))

                action = actor_readout.sample(action_logits)

                next_state, reward, terminated, truncated, *_ = env.step(action.detach().cpu().numpy())

                cum_reward += reward

                done = truncated or terminated

                replay_buffer.store(
                    state = state,
                    action = action
                )

                if done:
                    break

                state = next_state

            pbar.set_description(f'cumulative reward: {cum_reward.item():.1f}')

        # train the critic with contrastive learning

        if divisible_by(eps + 1, num_episodes_before_learn):

            data = replay_buffer.get_all_data(
                fields = ['state', 'action'],
                meta_fields = ['episode_lens']
            )

            critic_trainer(
                data['state'],
                cl_train_steps,
                lens = data['episode_lens'],
                actions = data['action']
            )

            actor_trainer(
                data['state'],
                actor_num_train_steps,
                lens = data['episode_lens'],
                sample_fn = actor_readout.sample
            )

# fire

if __name__ == '__main__':
    Fire(main)
