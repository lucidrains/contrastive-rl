# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout>=0.2.1",
#   "env-ssl-wrapper>=0.4.0",
#   "fire",
#   "gymnasium[mujoco,other]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.6.1",
#   "hl-gauss-pytorch"
# ]
# ///

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from collections import deque
from functools import partial

import numpy as np
import torch
from torch import nn, from_numpy, cat
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import gymnasium as gym
from fire import Fire
from accelerate import Accelerator

from memmap_replay_buffer import ReplayBuffer
from env_ssl_wrapper import StandardizeEnv

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
)

from x_mlps_pytorch import ResidualNormedMLP, AttnResidualNormedMLP
from discrete_continuous_embed_readout import Readout
from hl_gauss_pytorch import HLGaussLoss
from dashboard import Dashboard

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def state_to_kinematic_goal(s):
    # Torso z (height, index 0:1), torso quaternion w, x, y, z (indices 1:5), torso forward velocity vx (index 22:23)
    return cat((s[..., 0:1], s[..., 1:5], s[..., 22:23]), dim = -1)

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

# main

def main(
    max_env_steps: int = 10_000_000,
    num_envs: int = 16,
    max_timesteps: int = 1000,
    learn_every_env_steps: int = 16_000,
    eval_every_env_steps: int = 500_000,
    buffer_size: int = 2048,
    video_folder: str = './recordings_humanoid',
    checkpoint_folder: str = './checkpoints-humanoid',
    reward_json_path: str = './humanoid_rewards.json',
    dim_contrastive_embed: int = 64,
    cl_train_steps: int = 256,
    cl_batch_size: int = 128,
    actor_batch_size: int = 128,
    actor_num_train_steps: int = 128,
    critic_learning_rate: float = 3e-4,
    actor_learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 0.5,
    repetition_factor: int = 2,
    use_sigmoid: bool = True,
    sigmoid_bias: float = -5.,
    cl_l2norm_embed: bool = True,
    action_entropy_loss_weight: float = 1e-3,
    target_goal_prob: float = 0.8,
    target_forward_velocity: float = 1.5,
    exploration_random_goal_prob: float = 0.05,
    exploration_sample_from_buffer_prob: float = 0.5,
    use_hl_gauss_critic_actions: bool = False,
    hl_gauss_num_bins: int = 16,
    hl_gauss_sigma: float = 0.05,
    actor_dist_type: str = 'beta',
    mlp_depth: int = 4,
    use_attn_residual_mlp: bool = True,
    use_rmsnorm: bool = True,
    use_wandb: bool = False,
    cpu: bool = False,
    resume: bool = False,
    clear_recordings: bool = True
):
    # directories

    video_path = Path(video_folder)
    video_path.mkdir(parents = True, exist_ok = True)
    checkpoint_path = Path(checkpoint_folder)
    checkpoint_path.mkdir(parents = True, exist_ok = True)

    if clear_recordings:
        for video_file in video_path.glob('*.mp4'):
            try:
                video_file.unlink()
            except OSError:
                pass

    # accelerator

    accelerator = Accelerator(
        log_with = 'wandb' if use_wandb else None,
        cpu = cpu
    )

    if use_wandb:
        accelerator.init_trackers(
            project_name = 'contrastive-rl-humanoid',
            config = locals()
        )

    device = accelerator.device

    # environment (headless for maximum throughput)

    env = gym.make_vec(
        'Humanoid-v5',
        num_envs = num_envs,
        vector_kwargs = dict(autoreset_mode = 'SameStep')
    )
    env = StandardizeEnv(env)

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]
    goal_dim = 6  # z (1), quaternion (4), vx (1)

    dt = 0.015
    if hasattr(env.unwrapped, 'envs') and len(env.unwrapped.envs) > 0:
        dt = getattr(env.unwrapped.envs[0].unwrapped, 'dt', 0.015)

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
        overwrite = not resume
    )

    # models

    MLP = partial(AttnResidualNormedMLP, use_rmsnorm = use_rmsnorm) if use_attn_residual_mlp else partial(ResidualNormedMLP, residual_every = 2, keel_post_ln = True)

    actor_encoder = nn.Sequential(
        MLP(
            dim_in = obs_dim + goal_dim,
            dim = 256,
            depth = mlp_depth,
            dim_out = action_dim * 2
        ),
        Rearrange('... (action mu_logvar) -> ... action mu_logvar', mu_logvar = 2)
    ).to(device)

    # small weights near 0 for stable initial posture

    if hasattr(actor_encoder[0], 'proj_out'):
        nn.init.zeros_(actor_encoder[0].proj_out.weight)
        nn.init.zeros_(actor_encoder[0].proj_out.bias)

    latest_ckpt = checkpoint_path / 'actor-latest.pt'
    if resume and latest_ckpt.exists():
        actor_encoder.load_state_dict(torch.load(latest_ckpt, map_location = device))

    if actor_dist_type == 'beta':
        actor_readout = Readout(
            num_continuous = action_dim,
            continuous_dist_type = 'beta',
            continuous_dist_kwargs = dict(unimodal = True),
            continuous_squashed = False,
            dim = 0
        )
    else:
        actor_readout = Readout(
            num_continuous = action_dim,
            continuous_dist_type = 'gaussian',
            continuous_dist_kwargs = dict(log_var_clamp_range = (-5.0, 0.0)),
            continuous_squashed = True,
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

    critic_encoder = MLP(
        dim_in = obs_dim + critic_dim_action,
        dim = 256,
        dim_out = dim_contrastive_embed,
        depth = mlp_depth
    ).to(device)

    critic_encoder = CriticWrapper(critic_encoder, hl_gauss, action_dim)

    goal_encoder = MLP(
        dim_in = goal_dim,
        dim = 256,
        dim_out = dim_contrastive_embed,
        depth = mlp_depth
    ).to(device)

    # contrastive learning

    if use_sigmoid:
        contrastive_learn = SigmoidContrastiveLearning(
            bias = sigmoid_bias,
            l2norm_embed = cl_l2norm_embed,
            learned_scale = True
        )
    else:
        contrastive_learn = ContrastiveLearning(
            l2norm_embed = cl_l2norm_embed,
            learned_temp = True
        )

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
        contrastive_learn = contrastive_learn,
        state_to_goal_fn = state_to_kinematic_goal
    )

    actor_trainer = ActorTrainer(
        actor_encoder,
        critic_encoder,
        goal_encoder,
        batch_size = actor_batch_size,
        learning_rate = actor_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        action_entropy_loss_weight = action_entropy_loss_weight,
        normalize_q_values = True,
        target_goal_prob = target_goal_prob,
        cpu = cpu,
        contrastive_learn = contrastive_learn,
        state_to_goal_fn = state_to_kinematic_goal
    )

    # target walking goal: torso height 1.35m, upright quaternion, forward velocity 1.5 m/s

    target_state = torch.zeros(obs_dim, device = device)
    target_state[0] = 1.35
    target_state[1] = 1.0
    target_state[22] = target_forward_velocity
    actor_goal = state_to_kinematic_goal(target_state)

    # sampling helpers

    action_low = from_numpy(env.action_space.low).to(device)
    action_high = from_numpy(env.action_space.high).to(device)

    def sample_fn(logits, differentiable = False):
        return actor_readout.sample(logits, differentiable = differentiable, rescale_range = (action_low, action_high))

    def sample_single_goal():
        if replay_buffer.num_episodes > 0 and torch.rand(()) < exploration_sample_from_buffer_prob:
            all_states = replay_buffer.get_all_data(fields = ['state'])['state']
            flat = rearrange(all_states, '... d -> (...) d')
            idx = torch.randint(0, flat.shape[0], (1,)).item()
            return state_to_kinematic_goal(flat[idx].to(device))

        rand_s = from_numpy(env.single_observation_space.sample()).float().to(device)
        return state_to_kinematic_goal(rand_s)

    # single environment evaluation video recording

    def record_eval_episode(step_count):
        eval_env = gym.make('Humanoid-v5', render_mode = 'rgb_array')
        step_m_str = f'{step_count / 1e6:.1f}M'
        eval_env = gym.wrappers.RecordVideo(
            eval_env,
            video_folder = str(video_path),
            name_prefix = f'humanoid-step-{step_m_str}',
            episode_trigger = lambda _: True,
            disable_logger = True
        )

        eval_state, eval_info = eval_env.reset()
        init_pos_x = eval_info.get('x_position', 0.0) if isinstance(eval_info, dict) else 0.0
        cum_eval_reward = 0.0
        eval_step = 0

        actor_encoder.eval()

        for _ in range(max_timesteps):
            st_tensor = torch.from_numpy(eval_state.astype(np.float32)).to(device)
            net_in = cat((st_tensor, actor_goal), dim = -1).unsqueeze(0)

            with torch.no_grad():
                logits = actor_encoder(net_in)
                dist = actor_readout.continuous_dist.dist(logits)
                if actor_dist_type == 'beta':
                    eval_action = action_low + dist.mean * (action_high - action_low)
                else:
                    squashed = torch.tanh(dist.mean)
                    eval_action = action_low + (squashed + 1.0) * 0.5 * (action_high - action_low)

            next_st, r, term, trunc, s_info = eval_env.step(eval_action.squeeze(0).cpu().numpy())
            cum_eval_reward += r
            eval_step += 1
            eval_state = next_st

            if term or trunc:
                break

        final_pos_x = s_info.get('x_position', 0.0) if isinstance(s_info, dict) else 0.0
        eval_dist = final_pos_x - init_pos_x
        eval_avg_vel = eval_dist / (eval_step * eval_env.unwrapped.dt) if eval_step > 0 else 0.0
        eval_env.close()

        return eval_step, cum_eval_reward, eval_dist, eval_avg_vel

    # tracking & metrics

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)
    rolling_distance = deque(maxlen = 100)
    rolling_velocity = deque(maxlen = 100)
    best_eval_dist = -float('inf')
    last_eval_dist = 0.0

    total_env_steps = 0
    if Path(reward_json_path).exists() and resume:
        try:
            with open(reward_json_path, 'r') as f:
                prev_data = json.load(f)
                rolling_reward.extend(prev_data.get('rewards', [])[-100:])
                rolling_steps.extend(prev_data.get('lengths', [])[-100:])
                rolling_distance.extend(prev_data.get('distances', [])[-100:])
                rolling_velocity.extend(prev_data.get('velocities', [])[-100:])
                total_env_steps = prev_data.get('total_env_steps', 0)
        except Exception:
            pass

    def save_rewards():
        if len(rolling_reward) == 0:
            return

        with open(reward_json_path, 'w') as f:
            json.dump(dict(
                total_env_steps = total_env_steps,
                env_steps_M = round(total_env_steps / 1e6, 3),
                rewards = list(rolling_reward),
                lengths = list(rolling_steps),
                distances = list(rolling_distance),
                velocities = list(rolling_velocity),
                avg_reward = sum(rolling_reward) / len(rolling_reward),
                avg_distance = sum(rolling_distance) / len(rolling_distance) if rolling_distance else 0.0
            ), f)

    # dashboard

    dashboard = Dashboard(
        num_timesteps = max_env_steps,
        title = 'Contrastive RL - Humanoid (10M Steps)',
        env_name = 'Humanoid-v5',
        hyperparams = dict(
            max_env_steps = f'{max_env_steps / 1e6:.1f}M',
            num_envs = num_envs,
            cl_batch_size = cl_batch_size,
            actor_batch_size = actor_batch_size,
            buffer_size = buffer_size,
            learn_every_steps = learn_every_env_steps,
            target_velocity = target_forward_velocity,
            actor_dist_type = actor_dist_type
        )
    )

    # training loop

    start_time = time.time()
    next_eval_step = eval_every_env_steps

    with dashboard.create_renderable():

        state, info = env.reset()

        init_x = np.array(info.get('x_position', np.zeros(num_envs)), dtype = np.float32) if isinstance(info, dict) else np.zeros(num_envs, dtype = np.float32)

        cum_reward = np.zeros(num_envs)
        eps_steps = np.zeros(num_envs, dtype = int)

        is_exploring = torch.rand(num_envs) < exploration_random_goal_prob
        eps_goal = repeat(actor_goal, 'd -> n d', n = num_envs).clone()

        for i in range(num_envs):
            if is_exploring[i]:
                eps_goal[i] = sample_single_goal()

        states = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]

        total_episodes = 0
        steps_since_last_learn = 0
        min_buffer_episodes = max(cl_batch_size, actor_batch_size)

        while total_env_steps < max_env_steps:

            state = state.float()

            # actor inference

            actor_encoder.eval()

            with torch.no_grad():
                net_in = cat((state.to(device), eps_goal), dim = -1)
                action_logits = actor_encoder(net_in)
                action = sample_fn(action_logits)

            # step environment

            next_state, reward, terminated, truncated, step_info = env.step(action)

            total_env_steps += num_envs
            steps_since_last_learn += num_envs
            dashboard.advance_steps(num_envs)

            step_rewards = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
            curr_x_positions = step_info.get('x_position', None) if isinstance(step_info, dict) else None

            for i in range(num_envs):
                states[i].append(state[i].cpu().numpy())
                actions[i].append(action[i].cpu().numpy())

            cum_reward += step_rewards
            eps_steps += 1

            # handle completed episodes

            dones = (terminated | truncated).cpu().numpy() if isinstance(terminated, torch.Tensor) else (terminated | truncated)
            done = dones | (eps_steps >= max_timesteps)

            for i in range(num_envs):
                if not done[i]:
                    continue

                total_episodes += 1
                dashboard.advance_progress()

                if len(actions[i]) >= 2:
                    replay_buffer.store_episode(
                        state = states[i][:max_timesteps],
                        action = actions[i][:max_timesteps]
                    )

                if curr_x_positions is not None:
                    final_x = float(curr_x_positions[i])
                    dist = final_x - init_x[i]
                    dur = eps_steps[i] * dt
                    vel = (dist / dur) if dur > 0 else 0.0
                    rolling_distance.append(float(dist))
                    rolling_velocity.append(float(vel))
                    init_x[i] = final_x

                if not is_exploring[i]:
                    rolling_reward.append(float(cum_reward[i]))
                    rolling_steps.append(int(eps_steps[i]))
                    save_rewards()

                states[i] = []
                actions[i] = []
                cum_reward[i] = 0.
                eps_steps[i] = 0

                is_exploring[i] = torch.rand(()) < exploration_random_goal_prob
                eps_goal[i] = sample_single_goal() if is_exploring[i] else actor_goal

            # training step

            can_train = (replay_buffer.num_episodes >= min_buffer_episodes) and (steps_since_last_learn >= learn_every_env_steps)

            if can_train:
                steps_since_last_learn = 0

                data = replay_buffer.get_all_data(
                    fields = ['state', 'action'],
                    meta_fields = ['episode_lens']
                )

                trajectories = torch.as_tensor(data['state'])
                episode_lens = torch.as_tensor(data['episode_lens'])
                actions_for_critic = torch.as_tensor(data['action'])

                cl_loss = critic_trainer(
                    trajectories,
                    num_train_steps = cl_train_steps,
                    lens = episode_lens,
                    actions = actions_for_critic,
                    pbar = dashboard.critic_pbar
                )

                actor_loss = actor_trainer(
                    trajectories,
                    num_train_steps = actor_num_train_steps,
                    lens = episode_lens,
                    sample_fn = lambda logits: sample_fn(logits, differentiable = True),
                    entropy_fn = actor_readout.entropy if action_entropy_loss_weight > 0. else None,
                    target_goals = target_state,
                    pbar = dashboard.actor_pbar
                )

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                dashboard.update_metrics(
                    critic_loss = f'{cl_loss:.4f}',
                    actor_loss = f'{actor_loss:.4f}'
                )

                torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-latest.pt')

            # periodic evaluation and video recording

            if total_env_steps >= next_eval_step:
                next_eval_step += eval_every_env_steps
                eval_st, eval_rw, eval_d, eval_v = record_eval_episode(total_env_steps)
                last_eval_dist = eval_d
                step_m_tag = f'{total_env_steps / 1e6:.1f}M'
                torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-{step_m_tag}.pt')

                if eval_d > best_eval_dist:
                    best_eval_dist = eval_d
                    torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-best.pt')

                save_rewards()

            # update metrics

            elapsed = max(time.time() - start_time, 1e-4)
            fps = int(total_env_steps / elapsed)

            if len(rolling_reward) > 0:
                avg_reward = sum(rolling_reward) / len(rolling_reward)
                avg_steps = sum(rolling_steps) / len(rolling_steps)
                avg_dist = sum(rolling_distance) / len(rolling_distance) if rolling_distance else 0.0
                avg_vel = sum(rolling_velocity) / len(rolling_velocity) if rolling_velocity else 0.0

                dashboard.update_metrics(
                    env_steps_M = f'{total_env_steps / 1e6:.2f}M / {max_env_steps / 1e6:.1f}M',
                    fps = f'{fps:,}',
                    avg_cum_reward_100 = f'{avg_reward:.2f}',
                    avg_steps_100 = f'{avg_steps:.1f}',
                    avg_distance = f'{avg_dist:.2f}m',
                    avg_velocity = f'{avg_vel:.2f}m/s',
                    eval_distance = f'{last_eval_dist:.2f}m (best: {best_eval_dist:.2f}m)'
                )

                if use_wandb:
                    accelerator.log({
                        'env_steps': total_env_steps,
                        'avg_cum_reward_100': avg_reward,
                        'avg_steps_100': avg_steps,
                        'avg_distance': avg_dist,
                        'avg_velocity': avg_vel,
                        'last_eval_dist': last_eval_dist,
                        'best_eval_dist': best_eval_dist,
                        'fps': fps,
                        'critic_loss': cl_loss if 'cl_loss' in locals() else 0.,
                        'actor_loss': actor_loss if 'actor_loss' in locals() else 0.
                    })

            dashboard.refresh()
            state = next_state

    # final evaluation and save

    final_st, final_rw, final_d, final_v = record_eval_episode(total_env_steps)
    torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-final.pt')
    torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-latest.pt')
    save_rewards()

    if use_wandb:
        accelerator.end_training()

if __name__ == '__main__':
    Fire(main)
