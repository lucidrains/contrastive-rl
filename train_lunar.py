# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "fire",
#   "gymnasium[box2d]",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.10",
#   "tqdm"
# ]
# ///

from __future__ import annotations

from fire import Fire
from shutil import rmtree

import torch

from tqdm import tqdm

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveWrapper,
    ContrastiveRLTrainer,
    ActorTrainer
)

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
    render_every_eps = None
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
            actions = 'int',
        ),
        circular = True,
        overwrite = True
    )

    # episodes

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        state, *_ = env.reset()

        with replay_buffer.one_episode():

            for _ in range(max_timesteps):

                action = torch.randint(0, 4, ()).numpy()

                next_state, reward, terminated, truncated, *_ = env.step(action)

                done = truncated or terminated

                replay_buffer.store(
                    state = state,
                    action = action
                )

                if done:
                    break

                state = next_state

        if divisible_by(eps + 1, num_episodes_before_learn):
            break

# fire

if __name__ == '__main__':
    Fire(main)
