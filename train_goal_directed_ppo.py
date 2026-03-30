# /// script
# dependencies = [
#   "adam-atan2-pytorch",
#   "assoc-scan",
#   "contrastive-rl-pytorch",
#   "einops",
#   "ema-pytorch",
#   "fire",
#   "gymnasium[box2d,other]",
#   "hl-gauss-pytorch>=0.1.7",
#   "numpy",
#   "torch",
#   "torchaudio",
#   "torchvision",
#   "tqdm",
#   "hyper-connections",
#   "memmap-replay-buffer",
#   "x-mlps-pytorch>=0.3.0"
# ]
# ///

"""
Goal-directed PPO with contrastive RL reward bonus.

Uses a simplified PPO augmented by a learned
goal-similarity bonus: sigma(sim(critic_encoder(state, action), goal_encoder(goal)))
is added to every environment reward, where the encoders are trained via
contrastive RL from the same replay buffer.
"""

from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.distributions import Categorical

from einops import rearrange

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from hyper_connections import ManifoldConstrainedHyperConnections

from assoc_scan import AssocScan

import gymnasium as gym

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    SigmoidContrastiveLearning,
    sample_random_state
)

from x_mlps_pytorch import ResidualNormedMLP

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def maybe(fn):
    return lambda v: fn(v) if exists(v) else v

def divisible_by(num, den):
    return (num % den) == 0

def module_device(module):
    if torch.is_tensor(module): return module.device
    return next(module.parameters()).device

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer, max_grad_norm = 0.5):
    optimizer.zero_grad()
    loss.mean().backward()

    if exists(max_grad_norm):
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm)

    optimizer.step()

def l2norm(t):
    return F.normalize(t, dim = -1)


# SimBa MLP

class ReluSquared(Module):
    def forward(self, x):
        return x.sign() * F.relu(x) ** 2

class SimBa(Module):
    def __init__(self, dim, dim_hidden = None, depth = 3, dropout = 0., expansion_factor = 2, num_residual_streams = 4):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * expansion_factor)
        self.proj_in = nn.Linear(dim, dim_hidden)
        dim_inner = dim_hidden * expansion_factor

        init_hyper_conn, self.expand_stream, self.reduce_stream = ManifoldConstrainedHyperConnections.get_init_and_expand_reduce_stream_functions(1, num_fracs = num_residual_streams, sinkhorn_iters = 2)

        layers = []
        for ind in range(depth):
            layer = nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.Dropout(dropout),
            )
            layer = init_hyper_conn(dim = dim_hidden, layer_index = ind, branch = layer)
            layers.append(layer)

        self.layers = ModuleList(layers)
        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        no_batch = x.ndim == 1
        if no_batch:
            x = rearrange(x, '... -> 1 ...')
        x = self.proj_in(x)
        x = self.expand_stream(x)
        for layer in self.layers:
            x = layer(x)
        x = self.reduce_stream(x)
        out = self.final_norm(x)
        if no_batch:
            out = rearrange(out, '1 ... -> ...')
        return out

# actor and critic

class Actor(Module):
    def __init__(self, state_dim, hidden_dim, num_actions, goal_dim = 8, mlp_depth = 2, dropout = 0.1):
        super().__init__()
        self.net = SimBa(state_dim + goal_dim, dim_hidden = hidden_dim * 2, depth = mlp_depth, dropout = dropout)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x, goal):
        x = cat((x, goal), dim = -1)
        hidden = self.net(x)
        return self.action_head(hidden)

class Critic(Module):
    def __init__(self, state_dim, hidden_dim, num_actions, goal_dim = 8, dim_pred = 1, mlp_depth = 4, dropout = 0.1):
        super().__init__()
        self.net = SimBa(state_dim + goal_dim + num_actions, dim_hidden = hidden_dim, depth = mlp_depth, dropout = dropout)
        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x, goal, past_action):
        x = cat((x, goal, past_action), dim = -1)
        hidden = self.net(x)
        return self.value_head(hidden)

# GAE via associative scan

def calc_gae(rewards, values, masks, gamma = 0.99, lam = 0.95, use_accelerated = None):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)
    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]
    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks
    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)
    return scan(gates, delta) + values

# PPO agent

class PPO(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        reward_range,
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        ema_decay,
        use_delight_gating = True,
        actor_mlp_depth = 2,
        critic_mlp_depth = 4,
        goal_dim = 8,
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.actor = Actor(state_dim, actor_hidden_dim, num_actions, goal_dim = goal_dim, mlp_depth = actor_mlp_depth)
        self.critic = Critic(state_dim, critic_hidden_dim, num_actions, goal_dim = goal_dim, dim_pred = critic_pred_num_bins, mlp_depth = critic_mlp_depth)

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, update_model_with_ema_every = 1000)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, update_model_with_ema_every = 1000)

        self.opt_actor = AdoptAtan2(self.actor.parameters(), lr = lr, betas = betas)
        self.opt_critic = AdoptAtan2(self.critic.parameters(), lr = lr, betas = betas)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.use_delight_gating = use_delight_gating
        self.save_path = Path(save_path)

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return
        data = torch.load(str(self.save_path), weights_only = True)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories: ReplayBuffer, device = None):
        hl_gauss = self.critic_hl_gauss_loss

        # GAE pass
        dl = memories.dataloader(
            batch_size = 4,
            return_indices = True,
            to_named_tuple = ('_index', 'is_boundary', 'value', 'reward'),
            device = device
        )

        for indices, is_boundaries, values, rewards in dl:
            with torch.no_grad():
                masks = (1. - is_boundaries.float())
                scalar_values = hl_gauss(values)
                returns = calc_gae(
                    rewards = rewards, masks = masks,
                    lam = self.lam, gamma = self.gamma,
                    values = scalar_values, use_accelerated = False
                )
                memories.data['returns'][indices, :returns.shape[-1]] = returns.cpu().numpy()
                memories.flush()

        # minibatch training
        dl = memories.dataloader(
            batch_size = self.minibatch_size,
            shuffle = True,
            filter_fields = dict(learnable = True),
            to_named_tuple = ('state', 'action', 'action_log_prob', 'returns', 'value', 'past_action', 'target_goal'),
            timestep_level = True,
            device = device
        )

        self.actor.train()
        self.critic.train()

        for _ in range(self.epochs):
            for _, (states, actions, old_log_probs, returns, old_values, past_action, target_goals) in enumerate(dl):

                action_logits = self.actor(states, target_goals)
                dist = Categorical(logits = action_logits)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                scalar_old_values = hl_gauss(old_values)

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(returns - scalar_old_values.detach())

                maybe_gated_advantages = advantages

                if self.use_delight_gating:
                    delight_gate = (-action_log_probs * advantages).sigmoid().detach()
                    maybe_gated_advantages = advantages * delight_gate

                surr1 = ratios * maybe_gated_advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * maybe_gated_advantages
                policy_loss = -torch.min(surr1, surr2) - self.beta_s * entropy

                update_network_(policy_loss, self.opt_actor)

                # critic update
                clip = self.value_clip
                values = self.critic(states, target_goals, past_action)
                scalar_values = hl_gauss(values)

                clipped_returns = returns.clamp(scalar_old_values - clip, scalar_old_values + clip)
                clipped_loss = hl_gauss(values, clipped_returns, reduction = 'none')
                loss = hl_gauss(values, returns, reduction = 'none')

                old_lo = scalar_old_values - clip
                old_hi = scalar_old_values + clip

                def is_between(mid, lo, hi):
                    return (lo < mid) & (mid < hi)

                value_loss = torch.where(
                    is_between(scalar_values, returns, old_lo) | is_between(scalar_values, old_hi, returns),
                    0., torch.min(loss, clipped_loss)
                ).mean()

                update_network_(value_loss, self.opt_critic)

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 2000,
    max_timesteps = 500,
    # ppo
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    actor_mlp_depth = 2,
    critic_mlp_depth = 4,
    update_timesteps = 5000,
    buffer_episodes = 40,
    critic_pred_num_bins = 250,
    reward_range = (-400., 400.),
    minibatch_size = 64,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    ema_decay = 0.9,
    epochs = 2,
    use_delight_gating = True,
    goal_bonus_weight = 1.0,
    contrastive_embed_dim = 64,
    contrastive_encoder_dim = 64,
    contrastive_encoder_depth = 4,
    goal_encoder_dim = 64,
    goal_encoder_depth = 4,
    cl_train_steps = 500,
    cl_batch_size = 128,
    cl_repetition_factor = 2,
    cl_learning_rate = 3e-4,
    cl_bonus_loss_threshold = 0.04,
    cl_bonus_loss_temperature = 0.01,
    sigmoid_bias = -5.,
    exploration_random_goal_prob = 0.05,
    exploration_sample_from_buffer_prob = 0.5,
    # misc
    seed = None,
    render = True,
    render_every_eps = 50,
    save_every = 1000,
    video_folder = './lunar-recording-goal-ppo',
    load = False,
    cpu = True
):
    device = torch.device('cpu') if cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if render:
        rmtree(video_folder, ignore_errors = True)

    env = gym.make(env_name, render_mode = 'rgb_array')

    if render:
        env = gym.wrappers.RecordVideo(
            env = env, video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    state_dim = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)

    # replay buffer for PPO

    memories = ReplayBuffer(
        './lunar-memories-goal-ppo/data',
        max_episodes = buffer_episodes,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            learnable = 'bool',
            state = ('float', state_dim),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            is_boundary = 'bool',
            value = ('float', critic_pred_num_bins),
            returns = 'float',
            past_action = ('int', num_actions),
            target_goal = ('float', state_dim),
        ),
        circular = True,
        overwrite = True
    )

    # replay buffer for contrastive RL (stores raw env rewards for contrastive training)

    cl_replay_buffer = ReplayBuffer(
        './lunar-memories-goal-ppo/contrastive',
        max_episodes = cl_batch_size * 2,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', state_dim),
            action_one_hot = ('float', num_actions),
        ),
        circular = True,
        overwrite = True
    )

    # PPO agent

    agent = PPO(
        state_dim, num_actions, actor_hidden_dim, critic_hidden_dim,
        critic_pred_num_bins, reward_range, epochs, minibatch_size,
        lr, betas, lam, gamma, beta_s, eps_clip, value_clip, ema_decay,
        use_delight_gating = use_delight_gating,
        actor_mlp_depth = actor_mlp_depth,
        critic_mlp_depth = critic_mlp_depth,
        goal_dim = state_dim,
        save_path = './goal_directed_ppo.pt'
    ).to(device)

    if load:
        agent.load()

    # contrastive RL encoders (critic_encoder: state+action -> embed, goal_encoder: state -> embed)

    contrastive_learn = SigmoidContrastiveLearning(bias = sigmoid_bias, l2norm_embed = True)

    critic_encoder = ResidualNormedMLP(
        dim_in = state_dim + num_actions,
        dim = contrastive_encoder_dim,
        dim_out = contrastive_embed_dim,
        depth = contrastive_encoder_depth,
        residual_every = 4,
        keel_post_ln = True
    ).to(device)

    goal_encoder = ResidualNormedMLP(
        dim_in = state_dim,
        dim = goal_encoder_dim,
        dim_out = contrastive_embed_dim,
        depth = goal_encoder_depth,
        residual_every = 4,
        keel_post_ln = True
    ).to(device)

    cl_trainer = ContrastiveRLTrainer(
        critic_encoder,
        goal_encoder,
        batch_size = cl_batch_size,
        repetition_factor = cl_repetition_factor,
        learning_rate = cl_learning_rate,
        contrastive_learn = contrastive_learn,
        cpu = not torch.cuda.is_available()
    )

    # goal state for LunarLander: x=0, y=0, vx=0, vy=0, angle=0, angular_vel=0, left_leg=1, right_leg=1
    goal_state = tensor([0., 0., 0., 0., 0., 0., 1., 1.], device = device)

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    reward_window = deque(maxlen = 100)
    bonus_window = deque(maxlen = 100)
    steps_window = deque(maxlen = 100)
    last_cl_loss = float('inf')

    pbar = tqdm(range(num_episodes), desc = 'episodes')
    for eps in pbar:

        episode_reward = 0.
        episode_bonus = 0.
        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).float().to(device)
        past_action = torch.zeros(num_actions).to(device)

        is_exploring = torch.rand((), device = device) < exploration_random_goal_prob
        eps_goal = goal_state

        if is_exploring:
            eps_goal = sample_random_state(
                cl_replay_buffer,
                env,
                exploration_sample_from_buffer_prob
            ).to(device)

        ep_states = []
        ep_action_one_hots = []

        with memories.one_episode():
            for timestep in range(max_timesteps):
                time += 1

                action_logits = agent.ema_actor.forward_eval(state, eps_goal)
                value = agent.ema_critic.forward_eval(state, eps_goal, past_action)

                dist = Categorical(logits = action_logits)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                action_item = action.item()

                next_state, reward, terminated, truncated, _ = env.step(action_item)
                next_state = torch.from_numpy(next_state).float().to(device)

                action_one_hot = F.one_hot(action, num_classes = num_actions).float()

                if timestep == max_timesteps - 1:
                    truncated = True

                reward = float(reward)
                env_reward = reward

                # contrastive RL goal-similarity bonus
                # sigmoid(sim(critic_encoder(state, action), goal_encoder(goal)))

                with torch.no_grad():
                    critic_encoder.eval()
                    goal_encoder.eval()

                    state_action = cat((state, action_one_hot), dim = -1)
                    critic_embed = l2norm(critic_encoder(state_action))
                    goal_embed = l2norm(goal_encoder(eps_goal))

                    sim = (critic_embed * goal_embed).sum(dim = -1)

                    cl_loss_weight = 0.
                    if last_cl_loss < float('inf'):
                        cl_loss_weight = torch.tensor((cl_bonus_loss_threshold - last_cl_loss) / cl_bonus_loss_temperature).sigmoid().item()

                    bonus = sim.sigmoid().item() * goal_bonus_weight * cl_loss_weight

                reward += bonus
                episode_bonus += bonus

                # bootstrap for non-terminal truncation
                updating_agent = divisible_by(time, update_timesteps)
                done = terminated or truncated or updating_agent

                if done and not terminated:
                    next_value = agent.ema_critic.forward_eval(next_state, eps_goal, action_one_hot)
                    scalar_next_value = agent.critic_hl_gauss_loss(next_value).item()
                    reward += agent.gamma * scalar_next_value

                episode_reward += env_reward

                # store for contrastive training
                ep_states.append(state.cpu())
                ep_action_one_hots.append(action_one_hot.cpu())

                memories.store(
                    learnable = True,
                    state = state,
                    action = action,
                    action_log_prob = action_log_prob,
                    reward = reward,
                    is_boundary = done,
                    value = value,
                    past_action = past_action,
                    target_goal = eps_goal,
                )

                state = next_state
                past_action = F.one_hot(action, num_classes = num_actions).float()

                if updating_agent:
                    agent.learn(memories, device)

                if done:
                    break

        # store episode for contrastive learning
        if len(ep_states) >= 2:
            cl_replay_buffer.store_episode(
                state = ep_states,
                action_one_hot = ep_action_one_hots,
            )

        if not is_exploring:
            reward_window.append(episode_reward)
            bonus_window.append(episode_bonus)
            steps_window.append(timestep + 1)

        # periodically train contrastive encoders
        if cl_replay_buffer.num_episodes >= cl_batch_size and divisible_by(eps + 1, cl_batch_size):
            data = cl_replay_buffer.get_all_data(
                fields = ['state', 'action_one_hot'],
                meta_fields = ['episode_lens']
            )
            last_cl_loss, _ = cl_trainer(
                data['state'],
                cl_train_steps,
                lens = data['episode_lens'],
                actions = data['action_one_hot'],
                pbar = partial(tqdm, leave = False)
            )

        pbar.set_postfix(
            reward = f"{sum(reward_window) / len(reward_window):.2f}" if len(reward_window) > 0 else "0.00",
            bonus = f"{sum(bonus_window) / len(bonus_window):.3f}" if len(bonus_window) > 0 else "0.000",
            steps = f"{sum(steps_window) / len(steps_window):.1f}" if len(steps_window) > 0 else "0.0",
            cl_loss = f"{last_cl_loss:.3f}"
        )

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
