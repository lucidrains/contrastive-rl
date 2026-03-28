# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout>=0.2.1",
#   "fire",
#   "gymnasium[mujoco]>=1.0.0",
#   "gymnasium-robotics",
#   "gymnasium[other]",
#   "hl-gauss-pytorch>=0.1.2",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.3.0",
#   "tqdm"
# ]
# ///

from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from fire import Fire
from shutil import rmtree
from collections import deque
from functools import partial

import torch
from torch import nn, from_numpy, cat, is_tensor
import torch.nn.functional as F

from einops import rearrange, repeat

import gymnasium as gym
import gymnasium_robotics

from accelerate import Accelerator

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
)

from einops.layers.torch import Rearrange
from x_mlps_pytorch import ResidualNormedMLP, AttnResidualNormedMLP
from discrete_continuous_embed_readout import Readout
from hl_gauss_pytorch import HLGaussLoss

from dashboard import Dashboard

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def maybe(fn):
    return lambda v: fn(v) if exists(v) else v

def module_device(module):
    if is_tensor(module): return module.device
    return next(module.parameters()).device

def divisible_by(num, den):
    return (num % den) == 0

def concat_goal_obs(state_dict):
    return np.concatenate([state_dict['observation'], state_dict['achieved_goal']], axis = -1)

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
    num_episodes = 10_000,
    num_envs = 8,
    max_timesteps = 1000,
    num_episodes_before_learn = 128,
    buffer_size = 512,
    video_folder = './recordings_antmaze',
    dim_contrastive_embed = 128,
    cl_train_steps = 2_000,
    cl_batch_size = 128,
    actor_batch_size = 128,
    actor_num_train_steps = 1_000,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    actor_dim = 128,
    actor_depth = 16,
    critic_dim = 128,
    critic_depth = 12,
    goal_dim = 128,
    goal_depth = 8,
    weight_decay = 1e-4,
    max_grad_norm = 0.5,
    repetition_factor = 2,
    use_sigmoid_contrastive_learning = True,
    sigmoid_bias = -5.,
    cl_l2norm_embed = True,
    exploration_random_goal_prob = 0.1,
    use_attn_residual_mlp = True,
    env_name = 'AntMaze_UMaze-v5',
    use_wandb = False,
    cpu = False,
    reward_part_of_goal = False,
    reward_norm = 1.0,
    reward_fourier_encode = False,
    reward_fourier_dim = 16,
    use_hl_gauss_critic_actions = True,
    hl_gauss_num_bins = 16,
    hl_gauss_sigma = None,
    sigreg_loss_weight = 0.1,
    goal_includes_obs = False
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
            project_name = 'contrastive-rl-antmaze',
            config = locals()
        )

    # env - antmaze provides dict obs with 'observation', 'achieved_goal', 'desired_goal'

    env = gym.make_vec(env_name, num_envs = num_envs, vectorization_mode = 'async')

    # dims - using single spaces since it's a vector env

    obs_dim = env.single_observation_space['observation'].shape[0]
    goal_dim_env = env.single_observation_space['desired_goal'].shape[0]
    action_dim = env.single_action_space.shape[0]

    dim_state = obs_dim + goal_dim_env  # full state = obs + achieved_goal
    dim_goal = dim_state if goal_includes_obs else goal_dim_env
    dim_action = action_dim

    # target observation if goal_includes_obs
    UPRIGHT_OBS = np.zeros(obs_dim, dtype=np.float32)
    UPRIGHT_OBS[0] = 0.75 # z height
    UPRIGHT_OBS[1] = 1.0  # w component of quaternion (identity)

    dim_reward = reward_fourier_dim if reward_fourier_encode else 1
    dim_goal_encoder_in = dim_goal + (dim_reward if reward_part_of_goal else 0)

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-antmaze',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            action = ('float', dim_action),
            reward = 'float',
        ),
        circular = True,
        overwrite = True
    )

    # model

    device = accelerator.device
    env_device = accelerator.device

    if use_attn_residual_mlp:
        MLP = AttnResidualNormedMLP
    else:
        MLP = partial(ResidualNormedMLP, residual_every = 4, keel_post_ln = True)

    actor_encoder = nn.Sequential(
        MLP(
            dim_in = dim_state + dim_goal_encoder_in,  # state + goal (with optional reward)
            dim = actor_dim,
            depth = actor_depth,
            dim_out = dim_action * 2        # mu + logvar
        ),
        Rearrange('... (action mu_logvar) -> ... action mu_logvar', mu_logvar = 2)
    ).to(device)

    actor_readout = Readout(
        num_continuous = dim_action,
        continuous_dist_type = 'beta',
        continuous_dist_kwargs = dict(unimodal = True),
        continuous_squashed = False,
        dim = 0
    )

    hl_gauss = None
    critic_dim_action = dim_action

    if use_hl_gauss_critic_actions:
        hl_gauss = HLGaussLoss(
            min_value = -1.,
            max_value = 1.,
            num_bins = hl_gauss_num_bins,
            sigma = hl_gauss_sigma,
            clamp_to_range = True
        ).to(device)
        critic_dim_action = dim_action * hl_gauss_num_bins

    critic_encoder = MLP(
        dim_in = dim_state + critic_dim_action,
        dim = critic_dim,
        dim_out = dim_contrastive_embed,
        depth = critic_depth
    ).to(device)

    critic_encoder = CriticWrapper(critic_encoder, hl_gauss, dim_action)

    goal_encoder = MLP(
        dim_in = dim_goal_encoder_in,
        dim = goal_dim,
        dim_out = dim_contrastive_embed,
        depth = goal_depth
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
        cpu = cpu,
        contrastive_learn = contrastive_learn,
        state_to_goal_fn = lambda t: t[..., -dim_goal:],
        reward_part_of_goal = reward_part_of_goal,
        reward_norm = reward_norm,
        reward_fourier_encode = reward_fourier_encode,
        reward_fourier_dim = reward_fourier_dim,
        sigreg_loss_weight = sigreg_loss_weight
    )

    assert num_episodes_before_learn >= cl_batch_size, "num_episodes_before_learn must be >= cl_batch_size"

    # action rescaling

    def sample_fn(logits, differentiable = False, is_rollout = False):
        readout = actor_readout_rollout if is_rollout else actor_readout
        return readout.sample(logits, differentiable = differentiable, rescale_range = (-1., 1.))

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
        contrastive_learn = contrastive_learn,
        action_entropy_loss_weight = 5e-2,
        state_to_goal_fn = lambda t: t[..., -dim_goal:],
        reward_part_of_goal = reward_part_of_goal,
        reward_norm = reward_norm,
        reward_fourier_encode = reward_fourier_encode,
        reward_fourier_dim = reward_fourier_dim
    )

    # goal sampling for exploration

    actor_encoder_rollout = actor_encoder
    actor_readout_rollout = actor_readout

    def sync_rollout_model_():
        pass

    # goal sampling for exploration

    def sample_random_goal():
        if replay_buffer.num_episodes > 0 and torch.rand(()) < 0.5:
            all_states = replay_buffer.get_all_data(fields = ['state'])['state']
            flat = rearrange(all_states, '... d -> (...) d')
            idx = torch.randint(0, flat.shape[0], (1,)).item()
            state_slice = flat[idx]
            goal_slice = state_slice[..., -goal_dim_env:]
            
            if goal_includes_obs:
                goal_slice = np.concatenate([UPRIGHT_OBS, goal_slice], axis=-1)
                
            return from_numpy(goal_slice).float().to(env_device) if isinstance(goal_slice, np.ndarray) else goal_slice.float().to(env_device)

        temp_env = gym.make(env_name)
        state_dict, _ = temp_env.reset()
        temp_env.close()
        
        target_goal = state_dict['desired_goal'].astype(np.float32)
        if goal_includes_obs:
            target_goal = np.concatenate([UPRIGHT_OBS, target_goal], axis=-1)
            
        return from_numpy(target_goal).to(env_device)

    # evaluation video recording setup

    def record_eval_episode(eps_id):
        eval_env = gym.make(env_name, render_mode = 'rgb_array')
        eval_env = gym.wrappers.RecordVideo(
            eval_env,
            video_folder = video_folder,
            name_prefix = f'antmaze_eval_eps_{eps_id}',
            episode_trigger = lambda _: True,
            disable_logger = True
        )
        
        state_dict, _ = eval_env.reset()
        state = concat_goal_obs(state_dict).astype(np.float32)

        desired_goal = state_dict['desired_goal'].astype(np.float32)
        if goal_includes_obs:
            desired_goal = np.concatenate([UPRIGHT_OBS, desired_goal], axis=-1)
            
        eps_goal = from_numpy(desired_goal).to(env_device)

        if reward_part_of_goal:
            desired_reward = torch.ones((1,), device=env_device) / reward_norm
            if reward_fourier_encode:
                actor_trainer.reward_fourier_encode = actor_trainer.reward_fourier_encode.to(env_device)
                desired_reward = actor_trainer.reward_fourier_encode(desired_reward).squeeze(0)
            eps_goal = torch.cat((eps_goal, desired_reward), dim=-1)

        actor_encoder_rollout.eval()

        for _ in range(max_timesteps):
            curr_state = from_numpy(state).to(env_device)
            with torch.no_grad():
                action_logits = actor_encoder_rollout((curr_state, eps_goal))
                action = sample_fn(action_logits, is_rollout=True)

            next_state_dict, reward, terminated, truncated, _ = eval_env.step(action.detach().cpu().numpy())
            
            if terminated or truncated:
                break
                
            state = concat_goal_obs(next_state_dict).astype(np.float32)

        eval_env.close()

    # dashboard

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)
    rolling_success = deque(maxlen = 100)

    dashboard = Dashboard(
        num_episodes,
        title = f'Contrastive RL - {env_name}',
        env_name = env_name,
        hyperparams = dict(
            num_envs = num_envs,
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
            use_attn_residual_mlp = use_attn_residual_mlp,
            sigreg_loss_weight = sigreg_loss_weight,
        )
    )

    # training loop

    cl_loss = 0.
    critic_sigreg_loss = 0.
    actor_loss = 0.

    with dashboard.create_renderable() as live:
        state_dict, _ = env.reset()
        state = concat_goal_obs(state_dict).astype(np.float32)

        cum_reward = np.zeros(num_envs)
        eps_steps = np.zeros(num_envs, dtype = int)

        is_exploring = torch.rand(num_envs) < exploration_random_goal_prob
        eps_goal = torch.zeros((num_envs, dim_goal_encoder_in), device = env_device)
        
        if reward_part_of_goal:
            desired_rewards = torch.ones((num_envs, 1), device=env_device) / reward_norm
            if reward_fourier_encode:
                actor_trainer.reward_fourier_encode = actor_trainer.reward_fourier_encode.to(env_device)
                desired_rewards = actor_trainer.reward_fourier_encode(desired_rewards)
            eps_goal[:, dim_goal:] = desired_rewards

        def reset_eps_goal(i, target_goal_dict_i):
            if is_exploring[i]:
                eps_goal[i, :dim_goal] = sample_random_goal()
            else:
                target_xy = target_goal_dict_i.astype(np.float32)
                if goal_includes_obs:
                    target_goal = np.concatenate([UPRIGHT_OBS, target_xy], axis=-1)
                else:
                    target_goal = target_xy
                eps_goal[i, :dim_goal] = from_numpy(target_goal).to(env_device)

        for i in range(num_envs):
            reset_eps_goal(i, state_dict['desired_goal'][i])

        states = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]

        total_episodes = 0
        episodes_since_last_learn = 0

        sync_rollout_model_()

        while total_episodes < num_episodes:

            actor_encoder_rollout.eval()

            curr_state = from_numpy(state).to(env_device)

            with torch.no_grad():
                action_logits = actor_encoder_rollout((curr_state, eps_goal))
                action = sample_fn(action_logits, is_rollout=True)

            next_state_dict, reward, terminated, truncated, infos = env.step(action.detach().cpu().numpy())

            for i in range(num_envs):
                states[i].append(state[i])
                actions[i].append(action[i].cpu())
                rewards[i].append(reward[i])

            cum_reward += reward
            eps_steps += 1

            done = truncated | terminated

            for i in range(num_envs):
                if not done[i]:
                    continue

                total_episodes += 1
                episodes_since_last_learn += 1
                dashboard.advance_progress()

                # fetch success from final_info returned by gym wrapper
                has_success = False
                if "final_info" in infos and infos["_final_info"][i]:
                    if "is_success" in infos["final_info"][i]:
                        has_success = bool(infos["final_info"][i]["is_success"])
                
                rolling_success.append(1.0 if has_success else 0.0)

                if len(states[i]) >= 2:
                    replay_buffer.store_episode(
                        state = states[i],
                        action = actions[i],
                        reward = np.array(rewards[i], dtype = np.float32)
                    )

                if not is_exploring[i]:
                    rolling_reward.append(cum_reward[i])
                    rolling_steps.append(int(eps_steps[i]))

                states[i] = []
                actions[i] = []
                rewards[i] = []
                cum_reward[i] = 0.
                eps_steps[i] = 0

                is_exploring[i] = torch.rand(()) < exploration_random_goal_prob
                reset_eps_goal(i, next_state_dict['desired_goal'][i])

            # train critic and actor

            if episodes_since_last_learn >= num_episodes_before_learn:
                episodes_since_last_learn = 0

                data = replay_buffer.get_all_data(
                    fields = ['state', 'action', 'reward'],
                    meta_fields = ['episode_lens']
                )

                trajectories = data['state']
                episode_lens = data['episode_lens']
                actions_for_critic = data['action']
                rewards_for_trainers = data['reward']

                cl_loss, critic_sigreg_loss = critic_trainer(
                    trajectories,
                    cl_train_steps,
                    lens = episode_lens,
                    actions = actions_for_critic,
                    rewards = rewards_for_trainers,
                    pbar = dashboard.critic_pbar
                )

                actor_loss = actor_trainer(
                    trajectories,
                    actor_num_train_steps,
                    lens = episode_lens,
                    rewards = rewards_for_trainers,
                    pbar = dashboard.actor_pbar,
                    sample_fn = lambda logits: sample_fn(logits, differentiable = True),
                    entropy_fn = actor_readout.entropy
                )

                dashboard.update_metrics(
                    critic_loss = f'{cl_loss:.4f}',
                    critic_sigreg_loss = f'{critic_sigreg_loss:.4f}',
                    actor_loss = f'{actor_loss:.4f}'
                )

                sync_rollout_model_()

                # Record an evaluation episode precisely after learning step
                record_eval_episode(total_episodes)

            # update dashboard metrics
            if len(rolling_reward) > 0:
                avg_reward = sum(rolling_reward) / len(rolling_reward)
                avg_steps = sum(rolling_steps) / len(rolling_steps)
                avg_success = sum(rolling_success) / len(rolling_success)

                dashboard.update_metrics(
                    avg_cum_reward_100 = f'{avg_reward:.2f}',
                    avg_steps_100 = f'{avg_steps:.1f}',
                    success_rate_100 = f'{avg_success:.2%}'
                )

                if use_wandb:
                    accelerator.log({
                        'avg_cum_reward_100': avg_reward,
                        'avg_steps_100': avg_steps,
                        'last_eps_reward': cum_reward[0], # Using the first env as a sample proxy
                        'critic_loss': cl_loss if isinstance(cl_loss, float) else cl_loss,
                        'critic_sigreg_loss': critic_sigreg_loss,
                        'actor_loss': actor_loss
                    })

            live.update(dashboard.render())
            state = concat_goal_obs(next_state_dict).astype(np.float32)

    if use_wandb:
        accelerator.end_training()

# fire

if __name__ == '__main__':
    Fire(main)
