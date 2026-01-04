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
from torch import from_numpy
import torch.nn.functional as F

from tqdm import tqdm

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveWrapper,
    ContrastiveRLTrainer,
    ActorTrainer
)

from x_mlps_pytorch import ResidualNormedMLP

from discrete_continuous_embed_readout import Readout

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main

def main(
    num_episodes = 10_000,
    max_timesteps = 500,
    num_episodes_before_learn = 500,
    buffer_size = 1_000,
    video_folder = './recordings',
    render_every_eps = None,
    dim_contrastive_embed = 32,
    cl_train_steps = 500
):

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

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-lunar',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', 8),
            action = 'int',
        ),
        circular = True,
        overwrite = True
    )

    # model

    actor_encoder = ResidualNormedMLP(
        dim_in = 8,
        dim = 32,
        depth = 8,
        dim_out = 4
    )

    actor_readout = Readout(num_discrete = 4, dim = 0)

    critic_encoder = ResidualNormedMLP(
        dim_in = 8 + 4,
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
        cpu = True
    )

    # episodes

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        state, *_ = env.reset()

        with replay_buffer.one_episode():

            for _ in range(max_timesteps):

                action_logits = actor_encoder(from_numpy(state))

                action = actor_readout.sample(action_logits)

                next_state, reward, terminated, truncated, *_ = env.step(action.cpu().numpy())

                done = truncated or terminated

                replay_buffer.store(
                    state = state,
                    action = action
                )

                if done:
                    break

                state = next_state

        # train the critic with contrastive learning

        if divisible_by(eps + 1, num_episodes_before_learn):

            data = replay_buffer.get_all_data(
                fields = ['state', 'action'],
                meta_fields = ['episode_lens']
            )

            one_hot_actions = F.one_hot(data['action'].long(), num_classes = 4)

            critic_trainer(
                data['state'],
                cl_train_steps,
                lens = data['episode_lens'],
                actions = one_hot_actions
            )

# fire

if __name__ == '__main__':
    Fire(main)
