# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "fire",
#   "gymnasium[box2d]",
#   "gymnasium[mujoco]",
#   "imageio",
#   "imageio-ffmpeg",
#   "mean-conc-beta>=0.2.1",
#   "memmap-replay-buffer>=0.0.17",
#   "x-mlps-pytorch>=0.6.1",
#   "x-ppo>=0.0.21",
#   "numpy",
#   "torch",
# ]
#
# [tool.uv.sources]
# contrastive-rl-pytorch = { path = "." }
# ///

# rewards are the exponential of the negative distance
# from a state-goal CRL critic, optimized with PPO and GAE

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, arange, cat, is_tensor, nn
from torch.nn import Module
from torch.distributions import Categorical, Distribution
from torch.optim import AdamW
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from einops import rearrange, repeat
from torch_einops_utils.shape import divisible_by, shape

from fire import Fire
import imageio.v2 as imageio

from x_mlps_pytorch import MLP
from x_ppo import calc_gae, ppo_actor_loss
from mean_conc_beta import Beta

from contrastive_rl_pytorch import (
    ContrastiveLearning,
    ContrastiveWrapper,
    sample_discount,
    sample_truncated_geometric
)


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def mlp(dim_in, dim, dim_out, depth = 2):
    return MLP(dim_in, *((dim,) * depth), dim_out, activation = nn.SiLU())

def broadcast_goal(state, goal):
    if state.ndim == 1:
        state = rearrange(state, 'd -> 1 d')

    if goal.ndim == 1:
        goal = rearrange(goal, 'd -> 1 d')

    state_batch = shape(state, 'b ...').dims['b']
    goal_batch = shape(goal, 'b ...').dims['b']

    if goal_batch == 1 and state_batch > 1:
        goal = repeat(goal, '1 d -> b d', b = state_batch)

    return state, goal


def transform_pendulum(obs):
    x, theta, x_dot, theta_dot = obs
    return np.array([x, np.sin(theta), np.cos(theta), x_dot, theta_dot], dtype = np.float32)

def transform_lunar(obs):
    x, y, vx, vy, angle, vangle, leg1, leg2 = obs
    return np.array([x, y, vx, vy, np.sin(angle), np.cos(angle), vangle, leg1, leg2], dtype = np.float32)

def identity_obs(obs):
    return np.asarray(obs, dtype = np.float32)

def transform_batch(fn, obs):
    return np.stack([fn(o) for o in obs])

def normalize_goal(rms, goal, dim_goal):
    out = (goal - rms.mean[:dim_goal]) / np.sqrt(rms.var[:dim_goal] + 1e-8)
    return np.clip(out, -rms.clip, rms.clip).astype(np.float32)

class RunningMeanStd:
    def __init__(self, shape = (), clip = 10., epsilon = 1e-4):
        self.mean = np.zeros(shape, dtype = np.float64)
        self.var = np.ones(shape, dtype = np.float64)
        self.count = epsilon
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, dtype = np.float64)
        batch_mean, batch_var = x.mean(axis = 0), x.var(axis = 0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        x = np.asarray(x, dtype = np.float32)
        out = (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return dict(mean = self.mean, var = self.var, count = self.count)

    def load_state_dict(self, state):
        self.mean = np.asarray(state['mean'], dtype = np.float64)
        self.var = np.asarray(state['var'], dtype = np.float64)
        self.count = float(state['count'])

# per-env vector wrapper that keeps final_obs and reset goals

class VecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._goals = [None] * self.num_envs

    def reset(self, seed = None):
        obs, infos = [], []

        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            o, info = env.reset(seed = env_seed)
            info = dict(info)
            self._goals[i] = info.get('goal')
            info['goal'] = self._goals[i]
            obs.append(o)
            infos.append(info)

        return np.stack(obs), infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = [], [], [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            o, reward, term, trunc, info = env.step(action)
            info = dict(info)

            done = bool(term or trunc)
            info['final_obs'] = o if done else None

            if done:
                o, reset_info = env.reset()
                reset_info = dict(reset_info)
                self._goals[i] = reset_info.get('goal')

            info['goal'] = self._goals[i]

            obs.append(o)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)

        return (
            np.stack(obs),
            np.asarray(rewards, dtype = np.float32),
            np.asarray(terminated, dtype = np.bool_),
            np.asarray(truncated, dtype = np.bool_),
            infos
        )

    def close(self):
        for env in self.envs:
            env.close()

# running observation normalization

# eval replay buffer helpers

def make_eval_buffer(folder, max_episodes, max_timesteps, dim_state, dim_action, dim_goal = None):
    from memmap_replay_buffer import ReplayBuffer

    fields = dict(
        state = ('float', dim_state),
        reward = ('float', 1),
        action = ('float', dim_action)
    )

    if dim_goal is not None:
        fields['goal'] = ('float', dim_goal)

    return ReplayBuffer(
        folder,
        max_episodes = max_episodes,
        max_timesteps = max_timesteps + 1,
        fields = fields,
        circular = True,
        overwrite = True
    )

def store_eval_episode(replay_buffer, states, rewards, actions, goals = None):
    if len(states) < 2:
        return

    data = dict(
        state = states,
        reward = rewards,
        action = actions
    )

    if goals is not None:
        data['goal'] = goals

    replay_buffer.store_episode(**data)

# eval loops

def evaluate_continuous(
    trainer,
    env_fn,
    goal,
    obs_transform,
    rms = None,
    action_scale = 1.,
    episodes = 5,
    max_steps = 1000,
    seed = 0,
    deterministic = True,
    replay_buffer = None,
    success_threshold = None,
    goal_normalize = None
):
    env = env_fn()

    returns, lengths, successes = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed = seed + ep)

        ep_states, ep_actions, ep_rewards = [], [], []
        total = 0.
        steps = 0

        while steps < max_steps:
            transformed = obs_transform(obs)
            normed = rms.normalize(transformed) if exists(rms) else np.asarray(transformed, dtype = np.float32)
            goal_norm = goal_normalize(rms, goal) if exists(goal_normalize) else np.asarray(goal, dtype = np.float32)

            state_t = torch.from_numpy(normed).float().to(trainer.device)
            goal_t = torch.from_numpy(goal_norm).float().to(trainer.device)

            action, _ = trainer.act(state_t, goal_t, deterministic = deterministic)
            action_np = action.detach().cpu().numpy().reshape(-1) * action_scale

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            ep_states.append(normed)
            ep_actions.append(action.detach().cpu().numpy().reshape(-1))
            ep_rewards.append(float(reward))

            total += float(reward)
            steps += 1
            obs = next_obs

            if done:
                break

        returns.append(total)
        lengths.append(steps)
        successes.append(bool(success_threshold is not None and total >= success_threshold))

        if replay_buffer is not None and len(ep_states) >= 2:
            goals = np.repeat(np.asarray(goal_norm, dtype = np.float32)[None], len(ep_states), axis = 0)
            store_eval_episode(
                replay_buffer,
                np.asarray(ep_states, dtype = np.float32),
                np.asarray(ep_rewards, dtype = np.float32),
                np.asarray(ep_actions, dtype = np.float32),
                goals
            )

    env.close()

    return dict(
        return_mean = float(np.mean(returns)),
        length_mean = float(np.mean(lengths)),
        success_rate = float(np.mean(successes)) if success_threshold is not None else None
    )

# json logging

def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent = 2)

# four rooms

class FourRoomsEnv(gym.Env):
    metadata = dict(render_modes = ('rgb_array',))

    def __init__(self, size = 9, max_steps = 128, seed = 0, start_radius = 100, exploration_goal_prob = 0., local_radius = 1, goal_margin = 2):
        super().__init__()

        assert size >= 5 and not divisible_by(size, 2)

        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.walls = np.zeros((size, size), dtype = np.int8)
        mid = size // 2

        self.walls[mid, :] = 1
        self.walls[mid, 2] = 0
        self.walls[mid, size - 3] = 0

        self.walls[:, mid] = 1
        self.walls[2, mid] = 0
        self.walls[size - 3, mid] = 0

        self.free = np.argwhere(self.walls == 0)
        self.fixed_goal = np.array([size - goal_margin, size - goal_margin])
        self.start_radius = start_radius
        self.exploration_goal_prob = exploration_goal_prob
        self.local_radius = local_radius

        obs_dim = 2 + (2 * local_radius + 1) ** 2

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0., 1., shape = (obs_dim,), dtype = np.float32)

        self.agent = None
        self.goal = None
        self.steps = 0

    def _local_walls(self):
        row, col = self.agent
        bits = []

        for dr in range(-self.local_radius, self.local_radius + 1):
            for dc in range(-self.local_radius, self.local_radius + 1):
                r, c = row + dr, col + dc
                is_wall = r < 0 or r >= self.size or c < 0 or c >= self.size or self.walls[r, c] == 1
                bits.append(float(is_wall))

        return bits

    def _obs(self):
        row, col = self.agent
        pos = [row / (self.size - 1), col / (self.size - 1)]
        return np.array(pos + self._local_walls(), dtype = np.float32)

    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.rng.random() < self.exploration_goal_prob:
            self.goal = self.free[self.rng.integers(0, len(self.free))].copy()
        else:
            self.goal = self.fixed_goal.copy()

        cells = self.free

        if self.start_radius < self.size * 2:
            dist = np.abs(cells - self.goal).sum(axis = 1)
            cells = cells[dist <= self.start_radius]

        cells = cells[np.any(cells != self.goal, axis = 1)]
        self.agent = cells[self.rng.integers(0, len(cells))].copy()

        self.steps = 0
        return self._obs(), dict(goal = self.goal / (self.size - 1))

    def step(self, action):
        dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[int(action)]
        row, col = self.agent
        nr, nc = row + dr, col + dc

        if 0 <= nr < self.size and 0 <= nc < self.size and self.walls[nr, nc] == 0:
            self.agent = np.array([nr, nc])

        self.steps += 1

        success = bool(np.array_equal(self.agent, self.goal))
        truncated = self.steps >= self.max_steps

        return self._obs(), float(success), success, truncated, dict(goal = self.goal / (self.size - 1), success = success)

    def render(self, cell = 32):
        img = np.ones((self.size * cell, self.size * cell, 3), dtype = np.uint8) * 255

        for row in range(self.size):
            for col in range(self.size):
                if self.walls[row, col]:
                    img[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = (40, 40, 40)

        gr, gc = self.goal
        img[gr * cell:(gr + 1) * cell, gc * cell:(gc + 1) * cell] = (40, 190, 80)

        ar, ac = self.agent
        pad = cell // 5
        img[ar * cell + pad:(ar + 1) * cell - pad, ac * cell + pad:(ac + 1) * cell - pad] = (40, 90, 230)

        return img

def set_curriculum(envs, update, start_radius = 2, step = 4, max_radius = 20):
    radius = min(start_radius + update // step, max_radius)

    for env in envs.envs:
        env.start_radius = radius

# env presets

def env_config(name):
    if name == 'inverted_pendulum':
        return dict(
            make = lambda: gym.make('InvertedPendulum-v5'),
            transform = transform_pendulum,
            target_goal = np.array([0., 0., 1., 0., 0.], dtype = np.float32),
            dim_state = 5,
            dim_goal = 5,
            dim_action = 1,
            action_type = 'continuous',
            action_scale = 3.,
            max_steps = 1000,
            success = 900.,
            num_envs = 16,
            rollout_len = 128,
            num_updates = 200,
            actor_hidden = 128,
            actor_depth = 2,
            critic_hidden = 256,
            critic_depth = 2,
            critic_distance_scale = 1.,
            critic_lse_reg = 0.1,
            reward_temperature = 5.,
            reward_scale = 1.,
            critic_discount = 0.9,
            entropy_coef = 0.,
            exploration_goal_prob = 0.25,
            success_bonus = 0.,
            use_rms = True,
            goals_from_env = False
        )

    if name == 'lunar_continuous':
        return dict(
            make = lambda: gym.make('LunarLander-v3', continuous = True),
            transform = transform_lunar,
            target_goal = np.array([0., 0., 0., 0., 0., 1., 0.], dtype = np.float32),
            dim_state = 9,
            dim_goal = 7,
            dim_action = 2,
            action_type = 'continuous',
            action_scale = 1.,
            max_steps = 1000,
            success = 200.,
            num_envs = 16,
            rollout_len = 128,
            num_updates = 400,
            actor_hidden = 256,
            actor_depth = 3,
            critic_hidden = 256,
            critic_depth = 3,
            critic_distance_scale = 1.,
            critic_lse_reg = 0.1,
            reward_temperature = 10.,
            reward_scale = 1.,
            critic_discount = 0.95,
            entropy_coef = 0.,
            exploration_goal_prob = 0.25,
            success_bonus = 0.,
            use_rms = True,
            goals_from_env = False
        )

    assert name == 'fourrooms'

    return dict(
        transform = identity_obs,
        target_goal = np.array([7., 7.], dtype = np.float32) / 8.,
        dim_state = 11,
        dim_goal = 2,
        dim_action = 4,
        action_type = 'discrete',
        action_scale = 1.,
        max_steps = 128,
        success = 0.9,
        num_envs = 32,
        rollout_len = 64,
        num_updates = 300,
        actor_hidden = 128,
        actor_depth = 2,
        critic_hidden = 128,
        critic_depth = 2,
        critic_distance_scale = 5.,
        critic_lse_reg = 0.01,
        reward_temperature = 5.,
        reward_scale = 0.01,
        critic_discount = 0.95,
        entropy_coef = 0.01,
        exploration_goal_prob = 0.5,
        success_bonus = 1.,
        use_rms = False,
        goals_from_env = True
    )

# four rooms eval / video

def evaluate_fourrooms(trainer, env_fn, episodes = 20, max_steps = 128, seed = 0, replay_buffer = None):
    env = env_fn()

    successes, returns, lengths = [], [], []

    for ep in range(episodes):
        obs, info = env.reset(seed = seed + ep)
        goal = info['goal']

        ep_states, ep_actions, ep_rewards, ep_goals = [], [], [], []
        total = 0.
        steps = 0

        while steps < max_steps:
            state_t = torch.from_numpy(obs).float().to(trainer.device)
            goal_t = torch.from_numpy(goal).float().to(trainer.device)

            action, _ = trainer.act(state_t, goal_t, deterministic = True)
            action_np = int(action.item())

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            ep_states.append(obs)
            ep_actions.append(action_np)
            ep_rewards.append(float(reward))
            ep_goals.append(goal)

            total += float(reward)
            steps += 1
            obs = next_obs

            if done:
                break

        successes.append(bool(info.get('success', False)))
        returns.append(total)
        lengths.append(steps)

        if replay_buffer is not None and len(ep_states) >= 2:
            store_eval_episode(
                replay_buffer,
                np.asarray(ep_states, dtype = np.float32),
                np.asarray(ep_rewards, dtype = np.float32),
                np.asarray(ep_actions, dtype = np.float32),
                np.asarray(ep_goals, dtype = np.float32)
            )

    env.close()

    return dict(
        success_rate = float(np.mean(successes)),
        return_mean = float(np.mean(returns)),
        length_mean = float(np.mean(lengths))
    )

def record_winning_video(trainer, env_fn, path, max_steps = 128, seed = 0, episodes = 200, fps = 8, contact_sheet_frames = 6):
    env = env_fn()
    candidates = []

    def room_of(cell):
        mid = env.size // 2
        return int(cell[0] > mid) * 2 + int(cell[1] > mid)

    for ep in range(episodes):
        obs, info = env.reset(seed = seed + ep)
        goal = info['goal']
        start = env.agent.copy()
        frames = [env.render()]
        won = False

        for _ in range(max_steps):
            state_t = torch.from_numpy(obs).float().to(trainer.device)
            goal_t = torch.from_numpy(goal).float().to(trainer.device)

            action, _ = trainer.act(state_t, goal_t, deterministic = True)
            obs, _, terminated, truncated, _ = env.step(int(action.item()))
            frames.append(env.render())

            if terminated:
                won = True
                break

            if truncated:
                break

        if won:
            candidates.append(dict(frames = frames, start_room = room_of(start), goal_room = room_of(env.goal)))

    env.close()

    assert len(candidates) > 0, 'no winning episode found'

    preferred = [c for c in candidates if c['start_room'] != c['goal_room'] and c['start_room'] == 0]
    preferred = preferred or [c for c in candidates if c['start_room'] != c['goal_room']]
    best = max(preferred or candidates, key = lambda c: len(c['frames']))
    frames = best['frames']

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    imageio.mimsave(path, frames, fps = fps)
    imageio.mimsave(path.with_suffix('.gif'), frames, fps = fps, loop = 0)

    picks = np.linspace(0, len(frames) - 1, min(contact_sheet_frames, len(frames))).astype(int)
    contact = np.concatenate([frames[i] for i in picks], axis = 1)
    imageio.imwrite(path.with_name('winning-contact-sheet.png'), contact)

    return path

# actors

class BetaActor(Module):
    """mean-conc-beta actor for bounded continuous actions."""

    def __init__(
        self,
        dim_state,
        dim_goal,
        dim_action,
        hidden_dim = 256,
        depth = 2,
        beta: Beta | None = None,
        **beta_kwargs
    ):
        super().__init__()

        self.net = mlp(dim_state + dim_goal, hidden_dim, dim_action * 2, depth)
        self.beta = default(beta, Beta(**beta_kwargs))

    def forward(self, state, goal):
        state, goal = broadcast_goal(state, goal)
        params = self.net(cat((state, goal), dim = -1))
        params = rearrange(params, '... (a d) -> ... a d', d = 2)
        return self.beta(params)

    def log_prob(self, distr: Distribution, action: Tensor):
        return distr.log_prob(action).sum(dim = -1)

    def entropy(self, distr: Distribution):
        return distr.entropy().sum(dim = -1)

class CategoricalActor(Module):
    def __init__(
        self,
        dim_state,
        dim_goal,
        num_actions,
        hidden_dim = 256,
        depth = 2
    ):
        super().__init__()

        self.net = mlp(dim_state + dim_goal, hidden_dim, num_actions, depth)

    def forward(self, state, goal):
        state, goal = broadcast_goal(state, goal)
        return Categorical(logits = self.net(cat((state, goal), dim = -1)))

    def log_prob(self, distr: Distribution, action: Tensor):
        return distr.log_prob(action.long())

    def entropy(self, distr: Distribution):
        return distr.entropy()

# ppo with distance rewards

class DistanceRewardPPO(Module):
    # ppo with state-goal crl distance rewards
    # reward = exp(-||phi(s') - psi(g*)|| / tau)
    # advantage = standard GAE with a separate value net

    def __init__(
        self,
        actor: Module,
        critic_wrapper: ContrastiveWrapper,
        action_type = 'continuous',
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        weight_decay = 0.,
        max_grad_norm = 0.5,
        discount = 0.99,
        lam = 0.95,
        clip_eps = 0.2,
        entropy_coef = 1e-3,
        ppo_epochs = 4,
        minibatch_size = 256,
        critic_steps = 8,
        critic_batch_size = 256,
        critic_discount = 0.9,
        critic_negative_buffer_size = 8192,
        reward_temperature = 5.,
        reward_scale = 1.,
        value: Module | None = None,
        value_lr: float | None = None,
        value_epochs = 4,
        value_minibatch_size: int | None = None,
        normalize_advantages = True,
        state_to_goal_fn: Callable | None = None,
        ppo_kwargs: dict | None = None,
        device: str | torch.device | None = None
    ):
        super().__init__()

        assert action_type in ('continuous', 'discrete')

        self.actor = actor
        self.critic_wrapper = critic_wrapper
        self.value = value
        self.action_type = action_type

        self.discount = discount
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.critic_steps = critic_steps
        self.critic_batch_size = critic_batch_size
        self.critic_discount = critic_discount
        self.critic_negative_buffer_size = critic_negative_buffer_size
        self.reward_temperature = reward_temperature
        self.reward_scale = reward_scale
        self.value_epochs = value_epochs
        self.value_minibatch_size = default(value_minibatch_size, minibatch_size)
        self.normalize_advantages = normalize_advantages
        self.state_to_goal_fn = default(state_to_goal_fn, lambda t: t)
        self.ppo_kwargs = default(ppo_kwargs, dict())
        self.max_grad_norm = max_grad_norm

        self.actor_opt = AdamW(self.actor.parameters(), lr = actor_lr, weight_decay = weight_decay)
        self.critic_opt = AdamW(self.critic_wrapper.parameters(), lr = critic_lr, weight_decay = weight_decay)

        self.value_opt = None
        if exists(self.value):
            self.value_opt = AdamW(self.value.parameters(), lr = default(value_lr, critic_lr), weight_decay = weight_decay)

        self.negative_buffer = deque(maxlen = critic_negative_buffer_size)

        self.device = torch.device(device) if exists(device) else torch.device('cpu')
        self.to(self.device)

    # tensor helpers

    def _to_tensor(self, t, dtype = None):
        if is_tensor(t):
            return t.to(device = self.device, dtype = dtype) if exists(dtype) else t.to(self.device)

        return torch.as_tensor(t, device = self.device, dtype = dtype)

    def _broadcast_goals(self, goals, num_trajs, num_steps):
        goals = self._to_tensor(goals, torch.float32)

        if goals.ndim == 1:
            return repeat(goals, 'g -> n t g', n = num_trajs, t = num_steps)

        if goals.ndim == 2:
            goal_count = shape(goals, 'b ...').dims['b']

            if goal_count == 1:
                return repeat(goals, '1 g -> n t g', n = num_trajs, t = num_steps)

            if goal_count == num_trajs:
                return repeat(goals, 'n g -> n t g', t = num_steps)

            if goal_count == num_trajs * num_steps:
                return rearrange(goals, '(n t) g -> n t g', n = num_trajs)

            raise ValueError(f'cannot broadcast goals with shape {tuple(goals.shape)}')

        assert goals.ndim == 3
        return goals

    # acting

    @torch.no_grad()
    def act(self, states, goals, deterministic = False):
        states = self._to_tensor(states, torch.float32)
        goals = self._to_tensor(goals, torch.float32)

        single = states.ndim == 1
        states, goals = broadcast_goal(states, goals)

        distr = self.actor(states, goals)

        if deterministic:
            if self.action_type == 'discrete':
                action = distr.logits.argmax(dim = -1)
            else:
                action = distr.mean
        else:
            action = distr.sample()

        log_prob = self.actor.log_prob(distr, action)
        return (action.squeeze(0), log_prob.squeeze(0)) if single else (action, log_prob)

    # critic

    def _episode_ends(self, dones, num_steps):
        if not exists(dones):
            return torch.full((1, num_steps), num_steps - 1, device = self.device, dtype = torch.long)

        dones = dones.bool()
        num_trajs = shape(dones, 'n ...').dims['n']
        step_indices = repeat(arange(num_steps, device = self.device), 't -> n t', n = num_trajs)

        # nearest done at or after each timestep; default to the final rollout step

        next_done = torch.where(dones, step_indices, torch.full_like(step_indices, num_steps - 1))
        return torch.flip(torch.cummin(torch.flip(next_done, dims = (1,)), dim = 1).values, dims = (1,))

    def _sample_critic_pairs(self, states, dones, batch_size):
        dims = shape(states, 'n t ...').dims
        num_trajs, num_steps = dims['n'], dims['t']

        ends = self._episode_ends(dones, num_steps)

        if shape(ends, 'n ...').dims['n'] == 1 and num_trajs > 1:
            ends = repeat(ends, '1 t -> n t', n = num_trajs)

        times = repeat(arange(num_steps, device = self.device), 't -> n t', n = num_trajs)
        valid = ends > times
        traj_indices, time_indices = torch.nonzero(valid, as_tuple = True)

        assert traj_indices.numel() > 0, 'no valid past/future pairs - each episode needs at least two steps'

        rand_indices = torch.randint(0, traj_indices.numel(), (batch_size,), device = self.device)

        traj_indices = traj_indices[rand_indices]
        time_indices = time_indices[rand_indices]

        max_steps = ends[traj_indices, time_indices] - time_indices
        discount = rearrange(sample_discount(self.critic_discount, batch_size, device = self.device), 'b 1 -> b')
        delta_times = sample_truncated_geometric(max_steps, discount)

        future_times = time_indices + delta_times

        past_states = states[traj_indices, time_indices]
        future_goals = states[traj_indices, future_times]

        return past_states, future_goals

    def _sample_negative_goals(self, count):
        if len(self.negative_buffer) == 0:
            return None

        count = min(count, len(self.negative_buffer))
        indices = torch.randint(0, len(self.negative_buffer), (count,), device = 'cpu')
        goals = torch.stack([self.negative_buffer[int(i)] for i in indices])
        return goals.to(self.device)

    def update_critic(self, states, dones = None, steps: int | None = None, batch_size: int | None = None):
        steps = default(steps, self.critic_steps)
        batch_size = default(batch_size, self.critic_batch_size)

        self.critic_wrapper.train()

        losses = []
        for _ in range(steps):
            past_states, future_goals = self._sample_critic_pairs(states, dones, batch_size)
            future_goals = self.state_to_goal_fn(future_goals)

            extra_negatives = self._sample_negative_goals(batch_size)
            if exists(extra_negatives):
                extra_negatives = self.state_to_goal_fn(extra_negatives)

            loss = self.critic_wrapper(past_states, future_goals, extra_negatives = extra_negatives)

            self.critic_opt.zero_grad()
            loss.backward()

            if exists(self.max_grad_norm):
                torch.nn.utils.clip_grad_norm_(self.critic_wrapper.parameters(), self.max_grad_norm)

            self.critic_opt.step()
            losses.append(loss.item())

        # keep a short FIFO of states for harder / more diverse negatives

        flat_states = rearrange(states.detach(), 'n t d -> (n t) d').cpu()
        for state in flat_states:
            self.negative_buffer.append(state)

        return sum(losses) / len(losses) if losses else 0.

    # ppo

    def _ppo_update(self, states, goals, actions, old_log_probs, advantages):
        flat_states = rearrange(states, 'n t d -> (n t) d')
        flat_goals = rearrange(goals, 'n t g -> (n t) g')
        flat_actions = rearrange(actions, 'n t ... -> (n t) ...')
        flat_old_log_probs = rearrange(old_log_probs, 'n t -> (n t)')
        flat_advantages = rearrange(advantages, 'n t -> (n t)')

        batch_size = shape(flat_states, 'b ...').dims['b']
        minibatch_size = min(self.minibatch_size, batch_size)

        policy_losses = []
        entropies = []
        approx_kls = []
        clip_fracs = []

        self.actor.train()

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(batch_size, device = self.device)

            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start:(start + minibatch_size)]

                distr = self.actor(flat_states[indices], flat_goals[indices])
                new_log_probs = self.actor.log_prob(distr, flat_actions[indices])

                policy_loss = ppo_actor_loss(
                    new_log_probs,
                    flat_old_log_probs[indices],
                    flat_advantages[indices],
                    eps_clip = self.clip_eps,
                    normalize_advantages = self.normalize_advantages,
                    **self.ppo_kwargs
                ).mean()

                entropy = self.actor.entropy(distr)
                if exists(entropy) and self.entropy_coef > 0.:
                    policy_loss = policy_loss - entropy.mean() * self.entropy_coef

                self.actor_opt.zero_grad()
                policy_loss.backward()

                if exists(self.max_grad_norm):
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                self.actor_opt.step()

                policy_losses.append(policy_loss.item())

                if exists(entropy):
                    entropies.append(entropy.mean().item())

                log_ratio = new_log_probs - flat_old_log_probs[indices]
                ratio = log_ratio.exp()
                approx_kls.append(((ratio - 1.) - log_ratio).mean().item())
                clip_fracs.append(((ratio - 1.).abs() > self.clip_eps).float().mean().item())

        return dict(
            policy_loss = sum(policy_losses) / len(policy_losses),
            entropy = sum(entropies) / len(entropies) if entropies else 0.,
            approx_kl = sum(approx_kls) / len(approx_kls),
            clip_frac = sum(clip_fracs) / len(clip_fracs)
        )

    def _value_update(self, flat_states, flat_goals, flat_returns):
        if not exists(self.value) or self.value_epochs <= 0:
            return 0.

        batch_size = shape(flat_states, 'b ...').dims['b']
        minibatch_size = min(self.value_minibatch_size, batch_size)

        self.value.train()
        losses = []

        for _ in range(self.value_epochs):
            permutation = torch.randperm(batch_size, device = self.device)

            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start:(start + minibatch_size)]

                inputs = cat((flat_states[indices], flat_goals[indices]), dim = -1)
                pred = self.value(inputs).squeeze(-1)
                loss = F.mse_loss(pred, flat_returns[indices])

                self.value_opt.zero_grad()
                loss.backward()

                if exists(self.max_grad_norm):
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)

                self.value_opt.step()
                losses.append(loss.item())

        return sum(losses) / len(losses) if losses else 0.

    # public update

    def update(
        self,
        states,
        goals,
        actions,
        old_log_probs,
        dones = None,
        next_states = None,
        terminated = None,
        aux_rewards = None,
        critic_steps: int | None = None,
        ppo_epochs: int | None = None
    ):
        states = self._to_tensor(states, torch.float32)
        actions = self._to_tensor(actions)
        old_log_probs = self._to_tensor(old_log_probs, torch.float32)
        dones = self._to_tensor(dones, torch.bool) if exists(dones) else None

        dims = shape(states, 'n t ...').dims
        num_trajs, num_steps = dims['n'], dims['t']

        goals = self._broadcast_goals(goals, num_trajs, num_steps)

        if not exists(next_states):
            next_states = cat((states[:, 1:], states[:, -1:]), dim = 1)
        else:
            next_states = self._to_tensor(next_states, torch.float32)

        if not exists(terminated):
            terminated = dones if exists(dones) else torch.zeros((num_trajs, num_steps), device = self.device, dtype = torch.bool)

        terminated = self._to_tensor(terminated, torch.bool)

        # critic update on the fresh on-policy batch

        critic_steps = default(critic_steps, self.critic_steps)
        critic_loss = self.update_critic(states, dones, steps = critic_steps) if critic_steps > 0 else 0.

        # flatten for reward / value / ppo

        flat_states = rearrange(states, 'n t d -> (n t) d')
        flat_next_states = rearrange(next_states, 'n t d -> (n t) d')
        flat_goals = rearrange(goals, 'n t g -> (n t) g')

        # reward

        with torch.no_grad():
            distances = self.critic_wrapper.distance_pairs(flat_next_states, flat_goals)
            rewards = torch.exp(-distances / self.reward_temperature) * self.reward_scale

            rewards = rearrange(rewards, '(n t) -> n t', n = num_trajs)
            distances = rearrange(distances, '(n t) -> n t', n = num_trajs)

            if exists(self.value):
                flat_values = self.value(cat((flat_states, flat_goals), dim = -1)).squeeze(-1)
                flat_next_values = self.value(cat((flat_next_states, flat_goals), dim = -1)).squeeze(-1)
                values = rearrange(flat_values, '(n t) -> n t', n = num_trajs)
                next_values = rearrange(flat_next_values, '(n t) -> n t', n = num_trajs)
            else:
                values = torch.zeros_like(rewards)
                next_values = values

        if exists(aux_rewards):
            rewards = rewards + self._to_tensor(aux_rewards, torch.float32)

        masks = (~terminated).float()

        gae = calc_gae(
            rewards,
            values,
            masks = masks,
            gamma = self.discount,
            lam = self.lam,
            next_values = next_values,
            return_advantages = True
        )

        advantages = gae.advantages

        # value regression to the GAE returns

        value_loss = 0.
        if exists(self.value):
            flat_returns = rearrange(gae.returns, 'n t -> (n t)')
            value_loss = self._value_update(flat_states, flat_goals, flat_returns)

        ppo_epochs = default(ppo_epochs, self.ppo_epochs)

        ppo_metrics = self._ppo_update(
            states,
            goals,
            actions,
            old_log_probs,
            advantages
        )

        return dict(
            critic_loss = critic_loss,
            value_loss = value_loss,
            distance_mean = distances.mean().item(),
            value_mean = values.mean().item(),
            value_std = values.std().item(),
            reward_mean = rewards.mean().item(),
            advantage_mean = advantages.mean().item(),
            advantage_std = advantages.std().item(),
            **ppo_metrics
        )

# train

def main(
    env = 'lunar_continuous',
    num_envs = None,
    rollout_len = None,
    num_updates = None,
    max_steps = None,
    actor_lr = 3e-4,
    critic_lr = 3e-4,
    value_lr = None,
    weight_decay = 1e-4,
    discount = 0.99,
    lam = 0.97,
    clip_eps = 0.2,
    entropy_coef = None,
    ppo_epochs = 4,
    minibatch_size = 256,
    critic_steps = 16,
    critic_batch_size = 256,
    critic_discount = None,
    critic_negative_buffer_size = 8192,
    value_hidden = 256,
    value_depth = 2,
    value_epochs = 4,
    critic_embed_dim = 64,
    reward_temperature = None,
    reward_scale = None,
    success_bonus = None,
    exploration_goal_prob = None,
    goal_buffer_size = 8192,
    init_conc = 2.,
    random_warmup_updates = 10,
    eval_every = 10,
    log_every = 5,
    num_eval_episodes = 20,
    eval_buffer_size = 5000,
    eval_seed_offset = 10_000,
    video_seed_offset = 20_000,
    video_fps = 8,
    contact_sheet_frames = 6,
    curriculum_start_radius = 2,
    curriculum_step = 4,
    curriculum_max_radius = 20,
    checkpoint_folder = None,
    replay_folder = None,
    video_folder = None,
    log_path = None,
    seed = 0,
    device = None
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = env_config(env)

    num_envs = default(num_envs, cfg['num_envs'])
    rollout_len = default(rollout_len, cfg['rollout_len'])
    num_updates = default(num_updates, cfg['num_updates'])
    max_steps = default(max_steps, cfg['max_steps'])
    entropy_coef = default(entropy_coef, cfg['entropy_coef'])
    critic_discount = default(critic_discount, cfg['critic_discount'])
    reward_temperature = default(reward_temperature, cfg['reward_temperature'])
    reward_scale = default(reward_scale, cfg['reward_scale'])
    success_bonus = default(success_bonus, cfg['success_bonus'])
    exploration_goal_prob = default(exploration_goal_prob, cfg['exploration_goal_prob'])

    device = device or 'cpu'

    checkpoint_folder = checkpoint_folder or f'./checkpoints-crl-distance-{env}'
    replay_folder = replay_folder or f'./replay-crl-distance-{env}'
    video_folder = video_folder or f'./videos-crl-distance-{env}'
    log_path = log_path or f'./logs-crl-distance-{env}.jsonl'

    Path(checkpoint_folder).mkdir(parents = True, exist_ok = True)

    dim_state = cfg['dim_state']
    dim_goal = cfg['dim_goal']
    dim_action = cfg['dim_action']
    transform = cfg['transform']

    if cfg['goals_from_env']:
        def train_env_fn():
            return FourRoomsEnv(max_steps = max_steps, seed = seed, exploration_goal_prob = exploration_goal_prob)

        def eval_env_fn():
            return FourRoomsEnv(max_steps = max_steps, seed = seed, exploration_goal_prob = 0.)
    else:
        train_env_fn = cfg['make']
        eval_env_fn = cfg['make']

    envs = VecEnv([train_env_fn for _ in range(num_envs)])

    if cfg['goals_from_env']:
        set_curriculum(envs, 1, curriculum_start_radius, curriculum_step, curriculum_max_radius)

    obs, infos = envs.reset(seed = seed)
    obs = transform_batch(transform, obs)

    rms = RunningMeanStd(shape = (dim_state,)) if cfg['use_rms'] else None

    if exists(rms):
        rms.update(obs)

    if cfg['goals_from_env']:
        current_goals = np.stack([info['goal'] for info in infos])
    else:
        goal_buffer = deque(maxlen = goal_buffer_size)

        def sample_goal():
            if len(goal_buffer) > 0 and np.random.rand() < exploration_goal_prob:
                state = goal_buffer[np.random.randint(len(goal_buffer))]
                return state[:dim_goal].copy()

            return normalize_goal(rms, cfg['target_goal'], dim_goal)

        current_goals = np.stack([sample_goal() for _ in range(num_envs)])

    actor = BetaActor(
        dim_state = dim_state,
        dim_goal = dim_goal,
        dim_action = dim_action,
        hidden_dim = cfg['actor_hidden'],
        depth = cfg['actor_depth'],
        init_conc = init_conc,
        pos_fn = 'softplus',
        max_unimodal_floor = 50.,
        squash_fn = 'leaky_tanh'
    ) if cfg['action_type'] == 'continuous' else CategoricalActor(
        dim_state = dim_state,
        dim_goal = dim_goal,
        num_actions = dim_action,
        hidden_dim = cfg['actor_hidden'],
        depth = cfg['actor_depth']
    )

    state_encoder = MLP(
        dim_state,
        *((cfg['critic_hidden'],) * cfg['critic_depth']),
        critic_embed_dim,
        activation = nn.SiLU()
    )

    goal_encoder = MLP(
        dim_goal,
        *((cfg['critic_hidden'],) * cfg['critic_depth']),
        critic_embed_dim,
        activation = nn.SiLU()
    )

    contrastive_learn = ContrastiveLearning(
        use_euclidean = True,
        l2norm_embed = False,
        learned_temp = False,
        scale = cfg['critic_distance_scale'],
        symmetric = False,
        lse_reg_weight = cfg['critic_lse_reg']
    )

    critic_wrapper = ContrastiveWrapper(
        state_encoder,
        contrastive_learn,
        future_encoder = goal_encoder
    )

    value = MLP(
        dim_state + dim_goal,
        *((value_hidden,) * value_depth),
        1,
        activation = nn.SiLU()
    )

    state_to_goal_fn = (lambda t: t[..., :dim_goal]) if dim_goal < dim_state else None

    trainer = DistanceRewardPPO(
        actor,
        critic_wrapper,
        value = value,
        action_type = cfg['action_type'],
        actor_lr = actor_lr,
        critic_lr = critic_lr,
        value_lr = value_lr,
        value_epochs = value_epochs,
        weight_decay = weight_decay,
        discount = discount,
        lam = lam,
        clip_eps = clip_eps,
        entropy_coef = entropy_coef,
        ppo_epochs = ppo_epochs,
        minibatch_size = minibatch_size,
        critic_steps = critic_steps,
        critic_batch_size = critic_batch_size,
        critic_discount = critic_discount,
        critic_negative_buffer_size = critic_negative_buffer_size,
        reward_temperature = reward_temperature,
        reward_scale = reward_scale,
        state_to_goal_fn = state_to_goal_fn,
        device = device
    )

    eval_buffer = make_eval_buffer(
        replay_folder,
        max_episodes = eval_buffer_size,
        max_timesteps = max_steps,
        dim_state = dim_state,
        dim_action = dim_action if cfg['action_type'] == 'continuous' else 1,
        dim_goal = dim_goal
    )

    best_metric = float('-inf')
    solved_count = 0
    history = []

    for update in range(1, num_updates + 1):
        if cfg['goals_from_env']:
            set_curriculum(envs, update, curriculum_start_radius, curriculum_step, curriculum_max_radius)

        raw_states = np.zeros((num_envs, rollout_len, dim_state), dtype = np.float32)
        raw_next_states = np.zeros_like(raw_states)
        goals = np.zeros((num_envs, rollout_len, dim_goal), dtype = np.float32)
        actions = np.zeros((num_envs, rollout_len, dim_action), dtype = np.float32) if cfg['action_type'] == 'continuous' else np.zeros((num_envs, rollout_len), dtype = np.int64)
        log_probs = np.zeros((num_envs, rollout_len), dtype = np.float32)
        env_rewards = np.zeros((num_envs, rollout_len), dtype = np.float32)
        terminateds = np.zeros((num_envs, rollout_len), dtype = np.bool_)
        truncateds = np.zeros_like(terminateds)

        for step in range(rollout_len):
            normed = rms.normalize(obs) if exists(rms) else obs
            state_t = torch.from_numpy(normed).float().to(device)
            goals_norm = current_goals
            goal_t = torch.from_numpy(goals_norm).float().to(device)

            if update <= random_warmup_updates:
                if cfg['action_type'] == 'continuous':
                    action_np = np.random.uniform(-1., 1., size = (num_envs, dim_action)).astype(np.float32)
                else:
                    action_np = np.random.randint(0, dim_action, size = (num_envs,)).astype(np.int64)

                log_prob = torch.zeros(num_envs, device = device)
            else:
                with torch.no_grad():
                    action, log_prob = trainer.act(state_t, goal_t)

                action_np = action.cpu().numpy()

            exec_action = action_np * cfg['action_scale'] if cfg['action_type'] == 'continuous' else action_np
            next_obs_raw, step_rewards, terminated, truncated, infos = envs.step(exec_action)

            next_obs = transform_batch(transform, next_obs_raw)
            final_obs = np.stack([
                transform(info['final_obs']) if info['final_obs'] is not None else transform(next_obs_raw[i])
                for i, info in enumerate(infos)
            ])

            raw_states[:, step] = obs
            raw_next_states[:, step] = final_obs
            goals[:, step] = goals_norm
            actions[:, step] = action_np
            log_probs[:, step] = log_prob.cpu().numpy()
            env_rewards[:, step] = step_rewards
            terminateds[:, step] = terminated
            truncateds[:, step] = truncated

            if cfg['goals_from_env']:
                current_goals = np.stack([info['goal'] for info in infos])
            else:
                for i in np.where(terminated | truncated)[0]:
                    current_goals[i] = sample_goal()

            obs = next_obs

        if exists(rms):
            states = rms.normalize(raw_states)
            next_states = rms.normalize(raw_next_states)
            rms.update(raw_states.reshape(-1, dim_state))

            for state in states.reshape(-1, dim_state):
                goal_buffer.append(state.copy())
        else:
            states = raw_states
            next_states = raw_next_states

        dones = terminateds | truncateds

        if update <= random_warmup_updates:
            critic_loss = trainer.update_critic(
                torch.from_numpy(states).float().to(device),
                torch.from_numpy(dones).to(device)
            )
            metrics = dict(
                critic_loss = critic_loss,
                value_loss = 0.,
                distance_mean = 0.,
                value_mean = 0.,
                reward_mean = 0.,
                advantage_mean = 0.,
                advantage_std = 0.,
                policy_loss = 0.,
                entropy = 0.
            )
        else:
            aux_rewards = success_bonus * env_rewards if success_bonus > 0. else None

            metrics = trainer.update(
                states = states,
                goals = goals,
                actions = actions,
                old_log_probs = log_probs,
                dones = dones,
                next_states = next_states,
                terminated = terminateds,
                aux_rewards = aux_rewards
            )

        row = dict(
            update = update,
            env_steps = update * num_envs * rollout_len,
            **metrics
        )

        if divisible_by(update, eval_every) or update == 1:
            if cfg['goals_from_env']:
                result = evaluate_fourrooms(
                    trainer,
                    eval_env_fn,
                    episodes = num_eval_episodes,
                    max_steps = max_steps,
                    seed = eval_seed_offset + seed,
                    replay_buffer = eval_buffer
                )

                metric = result['success_rate']
                solved = metric >= cfg['success']

                row.update(
                    eval_success = metric,
                    eval_return = result['return_mean'],
                    eval_length = result['length_mean']
                )

                print(
                    f"update {update:4d} | steps {row['env_steps']:8d} | "
                    f"critic {metrics['critic_loss']:7.3f} | "
                    f"vloss {metrics['value_loss']:7.3f} | "
                    f"dist {metrics['distance_mean']:6.2f} | "
                    f"reward {metrics['reward_mean']:+.4f} | "
                    f"success {metric:6.0%} | len {result['length_mean']:6.1f}",
                    flush = True
                )
            else:
                result = evaluate_continuous(
                    trainer,
                    eval_env_fn,
                    cfg['target_goal'],
                    transform,
                    rms = rms,
                    action_scale = cfg['action_scale'],
                    episodes = num_eval_episodes,
                    max_steps = max_steps,
                    seed = eval_seed_offset + seed,
                    deterministic = True,
                    replay_buffer = eval_buffer,
                    success_threshold = cfg['success'],
                    goal_normalize = lambda r, g: normalize_goal(r, g, dim_goal)
                )

                metric = result['return_mean']
                solved = metric >= cfg['success']

                row.update(
                    eval_return = metric,
                    eval_length = result['length_mean'],
                    eval_success = result['success_rate']
                )

                print(
                    f"update {update:4d} | steps {row['env_steps']:8d} | "
                    f"critic {metrics['critic_loss']:7.3f} | "
                    f"vloss {metrics['value_loss']:7.3f} | "
                    f"dist {metrics['distance_mean']:6.2f} | "
                    f"value {metrics['value_mean']:7.3f} | "
                    f"reward {metrics['reward_mean']:+.4f} | "
                    f"eval {metric:7.1f} | len {result['length_mean']:6.1f} | "
                    f"success {result['success_rate']:.0%}",
                    flush = True
                )

            if metric > best_metric:
                best_metric = metric
                torch.save(actor.state_dict(), f'{checkpoint_folder}/actor-best.pt')
                torch.save(critic_wrapper.state_dict(), f'{checkpoint_folder}/critic-best.pt')
                torch.save(value.state_dict(), f'{checkpoint_folder}/value-best.pt')

                if exists(rms):
                    torch.save(rms.state_dict(), f'{checkpoint_folder}/rms-best.pt')

            solved_count = solved_count + 1 if solved else 0

            if solved_count >= 2:
                print(f'solved at update {update}', flush = True)

                if cfg['goals_from_env']:
                    video_path = record_winning_video(
                        trainer,
                        eval_env_fn,
                        Path(video_folder) / 'winning.mp4',
                        max_steps = max_steps,
                        seed = video_seed_offset + seed,
                        fps = video_fps,
                        contact_sheet_frames = contact_sheet_frames
                    )

                    print(f'video saved to {video_path}', flush = True)

                history.append(row)
                append_jsonl(log_path, row)
                break
        elif divisible_by(update, log_every):
            print(
                f"update {update:4d} | steps {row['env_steps']:8d} | "
                f"critic {metrics['critic_loss']:7.3f} | "
                f"vloss {metrics['value_loss']:7.3f} | "
                f"dist {metrics['distance_mean']:6.2f} | "
                f"reward {metrics['reward_mean']:+.4f}",
                flush = True
            )

        history.append(row)
        append_jsonl(log_path, row)

    torch.save(actor.state_dict(), f'{checkpoint_folder}/actor-final.pt')
    torch.save(critic_wrapper.state_dict(), f'{checkpoint_folder}/critic-final.pt')
    torch.save(value.state_dict(), f'{checkpoint_folder}/value-final.pt')

    if exists(rms):
        torch.save(rms.state_dict(), f'{checkpoint_folder}/rms-final.pt')
    save_json(f'{checkpoint_folder}/history.json', dict(history = history, best_metric = best_metric))

    envs.close()

if __name__ == '__main__':
    Fire(main)
