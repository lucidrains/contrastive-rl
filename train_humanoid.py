# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout>=0.2.1",
#   "fire",
#   "gymnasium[mujoco,other]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.1.32",
#   "hl-gauss-pytorch"
# ]
# ///

from __future__ import annotations

import os
import numpy as np
from fire import Fire
from shutil import rmtree
from collections import deque

import torch
from torch import nn, from_numpy, cat
import torch.nn.functional as F

from einops import rearrange, repeat

import gymnasium as gym
from accelerate import Accelerator

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
)

from einops.layers.torch import Rearrange
from x_mlps_pytorch import ResidualNormedMLP
from discrete_continuous_embed_readout import Readout

from hl_gauss_pytorch import HLGaussLoss

from dashboard import Dashboard

# classes

class CriticWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hl_gauss: nn.Module | None,
        dim_action: int
    ):
        super().__init__()
        self.encoder = encoder
        self.hl_gauss = hl_gauss
        self.dim_action = dim_action

    def forward(self, state_and_action):
        if not exists(self.hl_gauss):
            return self.encoder(state_and_action)

        dim_action = self.dim_action
        state, action = state_and_action[..., :-dim_action], state_and_action[..., -dim_action:]

        action_probs = self.hl_gauss.transform_to_probs(action)
        action_probs = rearrange(action_probs, '... a bins -> ... (a bins)')

        state_and_action = cat((state, action_probs), dim = -1)
        return self.encoder(state_and_action)

# helpers

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
    num_envs = 8,
    max_timesteps = 1000,
    num_episodes_before_learn = 512,
    buffer_size = 1536,
    video_folder = './recordings_humanoid',
    render_every_eps = None,
    dim_contrastive_embed = 64,
    cl_train_steps = 4_000,
    cl_batch_size = 256,
    actor_batch_size = 128,
    actor_num_train_steps = 1_500,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    weight_decay = 1e-4,
    max_grad_norm = 0.5,
    repetition_factor = 2,
    use_sigmoid_contrastive_learning = True,
    sigmoid_bias = -5.,
    cl_l2norm_embed = True,
    exploration_random_goal_prob = 0.025,
    exploration_sample_from_buffer_prob = 0.5,
    use_hl_gauss_critic_actions = True,
    hl_gauss_num_bins = 16,
    hl_gauss_sigma = 0.05,
    actor_dist_type = 'squashed_gaussian',
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

    env = gym.make_vec('Humanoid-v5', num_envs = num_envs, render_mode = 'rgb_array')

    # recording

    render_every_eps = default(render_every_eps, num_episodes_before_learn // num_envs)

    env = gym.wrappers.vector.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'humanoid',
        episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
        disable_logger = True
    )

    # dims

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

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

    device = accelerator.device

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

    if actor_dist_type == 'squashed_gaussian':
        continuous_dist_type = 'gaussian'
        continuous_dist_kwargs = dict()
        continuous_squashed = True
        from_range = (-1., 1.)
    elif actor_dist_type == 'kumaraswamy':
        continuous_dist_type = 'kumaraswamy'
        continuous_dist_kwargs = dict(unimodal = True)
        continuous_squashed = False
        from_range = (0., 1.)
    else:
        raise ValueError(f'invalid actor_dist_type {actor_dist_type}. must be "squashed_gaussian" or "kumaraswamy"')

    actor_readout = Readout(
        num_continuous = action_dim,
        continuous_dist_type = continuous_dist_type,
        continuous_dist_kwargs = continuous_dist_kwargs,
        continuous_squashed = continuous_squashed,
        dim = 0
    )

    hl_gauss = None
    critic_dim_action = action_dim

    if use_hl_gauss_critic_actions:
        hl_gauss = HLGaussLoss(
            min_value = -0.4,
            max_value = 0.4,
            num_bins = hl_gauss_num_bins,
            sigma = hl_gauss_sigma,
        ).to(device)

        critic_dim_action = action_dim * hl_gauss_num_bins

    critic_encoder = ResidualNormedMLP(
        dim_in = obs_dim + critic_dim_action,
        dim = 256,
        dim_out = dim_contrastive_embed,
        depth = 16,
        residual_every = 4,
        keel_post_ln = True
    ).to(device)

    critic_encoder = CriticWrapper(critic_encoder, hl_gauss, action_dim)

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

    # trainers

    critic_trainer = ContrastiveRLTrainer(
        critic_encoder,
        goal_encoder,
        batch_size = cl_batch_size,
        learning_rate = critic_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        repetition_factor = repetition_factor,
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    assert num_episodes_before_learn > cl_batch_size

    actor_trainer = ActorTrainer(
        actor_encoder,
        critic_encoder,
        goal_encoder,
        batch_size = actor_batch_size,
        learning_rate = actor_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        softmax_actor_output = False,
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    # actor goal
    # Humanoid-v5: obs[0] = torso height, obs[1] = torso orientation, obs[22] = x-velocity

    actor_goal = torch.zeros(obs_dim, device = device)
    actor_goal[0] = 1.3
    actor_goal[1] = 1.0
    actor_goal[22] = 1.0

    # action rescaling

    action_low = torch.from_numpy(env.single_action_space.low).to(device)
    action_high = torch.from_numpy(env.single_action_space.high).to(device)

    def sample_fn(logits, differentiable = False):
        return actor_readout.sample(logits, differentiable = differentiable, rescale_range = (action_low, action_high))

    # goal sampling for exploration

    def sample_single_goal():
        if replay_buffer.num_episodes > 0 and torch.rand(()) < exploration_sample_from_buffer_prob:
            all_states = replay_buffer.get_all_data(fields = ['state'])['state']
            flat = rearrange(all_states, '... d -> (...) d')
            idx = torch.randint(0, flat.shape[0], (1,)).item()
            return flat[idx].to(device)

        return from_numpy(env.single_observation_space.sample()).float().to(device)

    # dashboard

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)

    dashboard = Dashboard(
        num_episodes,
        title = 'Contrastive RL - Humanoid',
        env_name = 'Humanoid-v5',
        hyperparams = dict(
            critic_learning_rate = critic_learning_rate,
            actor_learning_rate = actor_learning_rate,
            cl_batch_size = cl_batch_size,
            actor_batch_size = actor_batch_size,
            buffer_size = buffer_size,
            max_timesteps = max_timesteps,
            weight_decay = weight_decay,
            max_grad_norm = max_grad_norm,
            repetition_factor = repetition_factor,
            use_sigmoid_contrastive_learning = use_sigmoid_contrastive_learning,
            exploration_random_goal_prob = exploration_random_goal_prob,
            exploration_sample_from_buffer_prob = exploration_sample_from_buffer_prob,
            use_hl_gauss_critic_actions = use_hl_gauss_critic_actions,
            hl_gauss_num_bins = hl_gauss_num_bins,
            hl_gauss_sigma = f'{hl_gauss_sigma}' if hl_gauss_sigma else 'None',
            actor_dist_type = actor_dist_type
        )
    )

    # training loop

    with dashboard.create_renderable() as live:
        state, *_ = env.reset()

        cum_reward = np.zeros(num_envs)
        eps_steps = np.zeros(num_envs, dtype = int)

        is_exploring = torch.rand(num_envs) < exploration_random_goal_prob
        eps_goal = repeat(actor_goal, 'd -> n d', n = num_envs)

        for i in range(num_envs):
            if is_exploring[i]:
                eps_goal[i] = sample_single_goal()

        states = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]

        total_episodes = 0
        episodes_since_last_learn = 0

        while total_episodes < num_episodes:
            state = state.astype(np.float32)

            # actor inference

            actor_encoder.eval()

            with torch.no_grad():
                action_logits = actor_encoder(cat((from_numpy(state).to(device), eps_goal), dim = -1))
                action = sample_fn(action_logits, differentiable = False)

            # environment step

            next_state, reward, terminated, truncated, infos = env.step(action.cpu().numpy())

            for i in range(num_envs):
                states[i].append(state[i])
                actions[i].append(action[i].cpu())

            cum_reward += reward
            eps_steps += 1

            # handle completed episodes

            done = terminated | truncated

            for i in range(num_envs):
                if not done[i]:
                    continue

                total_episodes += 1
                episodes_since_last_learn += 1
                dashboard.advance_progress()

                if len(actions[i]) >= 2:
                    replay_buffer.store_episode(
                        state = states[i],
                        action = actions[i]
                    )

                if not is_exploring[i]:
                    rolling_reward.append(cum_reward[i])
                    rolling_steps.append(int(eps_steps[i]))

                # reset accumulators and resample goal

                states[i] = []
                actions[i] = []
                cum_reward[i] = 0.
                eps_steps[i] = 0

                is_exploring[i] = torch.rand(()) < exploration_random_goal_prob
                eps_goal[i] = sample_single_goal() if is_exploring[i] else actor_goal

            # training

            if episodes_since_last_learn >= num_episodes_before_learn:
                episodes_since_last_learn = 0

                data = replay_buffer.get_all_data(
                    fields = ['state', 'action'],
                    meta_fields = ['episode_lens']
                )

                trajectories = torch.as_tensor(data['state']).to(device)
                episode_lens = torch.as_tensor(data['episode_lens']).to(device)
                actions_for_critic = torch.as_tensor(data['action']).to(device)

                cl_loss = critic_trainer(
                    trajectories,
                    cl_train_steps,
                    lens = episode_lens,
                    actions = actions_for_critic,
                    pbar = dashboard.critic_pbar
                )

                actor_loss = actor_trainer(
                    trajectories,
                    num_train_steps = actor_num_train_steps,
                    lens = episode_lens,
                    sample_fn = lambda logits: sample_fn(logits, differentiable = True),
                    pbar = dashboard.actor_pbar
                )

                dashboard.update_metrics(
                    critic_loss = f'{cl_loss:.4f}',
                    actor_loss = f'{actor_loss:.4f}'
                )

            # update dashboard

            if len(rolling_reward) > 0:
                avg_reward = sum(rolling_reward) / len(rolling_reward)
                avg_steps = sum(rolling_steps) / len(rolling_steps)

                dashboard.update_metrics(
                    avg_cum_reward_100 = f'{avg_reward:.2f}',
                    avg_steps_100 = f'{avg_steps:.1f}',
                    last_eps_reward = f'{rolling_reward[-1]:.2f}',
                    last_eps_steps = rolling_steps[-1]
                )

                if use_wandb:
                    accelerator.log({
                        'avg_cum_reward_100': avg_reward,
                        'avg_steps_100': avg_steps,
                        'last_eps_reward': rolling_reward[-1],
                        'critic_loss': cl_loss if 'cl_loss' in locals() else 0.,
                        'actor_loss': actor_loss if 'actor_loss' in locals() else 0.
                    })

            dashboard.refresh()
            state = next_state

    if use_wandb:
        accelerator.end_training()

# fire

if __name__ == '__main__':
    Fire(main)
