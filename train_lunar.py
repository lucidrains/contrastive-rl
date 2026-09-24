# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout",
#   "fire",
#   "gymnasium[box2d]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.6.1",
#   "tqdm",
#   "einops"
# ]
# ///

from __future__ import annotations

import os
import json
from collections import deque
from pathlib import Path
from shutil import rmtree

import torch
from torch import from_numpy, cat, tensor
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from fire import Fire

from einops import rearrange

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
    sample_random_state,
    default_discount_transform
)

from x_mlps_pytorch import AttnResidualNormedMLP
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
    num_episodes = 1500,
    max_timesteps = 500,
    num_episodes_before_learn = 8,
    learn_every_eps = 4,
    buffer_size = 512,
    video_folder = './recordings',
    render_every_eps = 150,
    dim_contrastive_embed = 64,
    cl_train_steps = 100,
    cl_batch_size = 64,
    actor_num_train_steps = 50,
    actor_batch_size = 64,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    actor_dim = 256,
    actor_depth = 4,
    critic_dim = 256,
    critic_depth = 4,
    goal_dim = 256,
    goal_depth = 4,
    use_rmsnorm = True,
    weight_decay = 1e-4,
    max_grad_norm = 0.5,
    repetition_factor = 1,
    use_sigmoid = True,
    sigmoid_bias = 0.,
    use_euclidean = False,
    discount = 0.99,
    discount_condition = False,
    discount_range = (0.85, 0.999),
    rollout_discount = 0.99,
    num_demo_episodes = 20,
    exploration_random_goal_prob = 0.05,
    exploration_sample_from_buffer_prob = 0.5,
    action_entropy_loss_weight = 0.01,
    save_checkpoint_every = 200,
    checkpoint_folder = './checkpoints-lunar',
    reward_json_path = './lunar_rewards.json',
    cpu = False,
    seed = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # recordings folder

    os.makedirs(video_folder, exist_ok = True)
    os.makedirs(checkpoint_folder, exist_ok = True)

    # env

    render_mode = 'rgb_array' if exists(render_every_eps) else None

    env = gym.make('LunarLander-v3', render_mode = render_mode)

    if exists(render_every_eps):
        env = gym.wrappers.RecordVideo(
            env,
            video_folder = video_folder,
            episode_trigger = lambda ep: divisible_by(ep, render_every_eps),
            name_prefix = 'lunar'
        )

    dim_state = 8
    dim_goal = 8
    dim_action = 4

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-lunar-discrete',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            reward = ('float', 1),
            action_hard_one_hot = ('float', dim_action)
        ),
        circular = True,
        overwrite = True
    )

    # maybe seed replay buffer with demonstrations for quick convergence

    if num_demo_episodes > 0:
        from gymnasium.envs.box2d.lunar_lander import heuristic

        for ep in range(num_demo_episodes):
            s, _ = env.reset(seed = seed + 1000 + ep)
            demo_states, demo_actions = [], []
            while True:
                a = heuristic(env.unwrapped, s)
                if np.random.rand() < 0.05:
                    a = np.random.randint(0, dim_action)
                next_s, r, term, trunc, _ = env.step(a)
                demo_states.append(s)
                demo_actions.append(F.one_hot(tensor(a), num_classes = dim_action).float())
                s = next_s
                if term or trunc:
                    break
            if len(demo_states) >= 2:
                replay_buffer.store_episode(
                    state = demo_states,
                    reward = [1.] * len(demo_states),
                    action_hard_one_hot = demo_actions
                )

    # models

    dim_discount = 3 if discount_condition else 0
    effective_discount = discount_range if discount_condition else discount

    actor_encoder = AttnResidualNormedMLP(
        dim_in = dim_state + dim_goal + dim_discount,
        dim = actor_dim,
        depth = actor_depth,
        dim_out = dim_action,
        use_rmsnorm = use_rmsnorm
    )

    actor_readout = Readout(num_discrete = dim_action, dim = 0)

    critic_encoder = AttnResidualNormedMLP(
        dim_in = dim_state + dim_action + dim_discount,
        dim = critic_dim,
        depth = critic_depth,
        dim_out = dim_contrastive_embed,
        use_rmsnorm = use_rmsnorm
    )

    goal_encoder = AttnResidualNormedMLP(
        dim_in = dim_goal,
        dim = goal_dim,
        depth = goal_depth,
        dim_out = dim_contrastive_embed,
        use_rmsnorm = use_rmsnorm
    )

    # contrastive learning module

    if use_sigmoid:
        contrastive_learn = SigmoidContrastiveLearning(
            bias = sigmoid_bias,
            l2norm_embed = False,
            learned_scale = False,
            use_euclidean = use_euclidean
        )
    else:
        contrastive_learn = ContrastiveLearning(
            l2norm_embed = False,
            learned_temp = False,
            use_euclidean = use_euclidean
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
        discount = effective_discount,
        discount_condition = discount_condition,
        discount_transform = default_discount_transform if discount_condition else (lambda t: t),
        cpu = cpu,
        contrastive_learn = contrastive_learn
    )

    actor_trainer = ActorTrainer(
        actor_encoder,
        critic_encoder,
        goal_encoder,
        batch_size = actor_batch_size,
        learning_rate = actor_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        discount = effective_discount,
        discount_condition = discount_condition,
        discount_transform = default_discount_transform if discount_condition else (lambda t: t),
        num_discrete_actions = dim_action,
        cpu = cpu,
        contrastive_learn = contrastive_learn,
        action_entropy_loss_weight = action_entropy_loss_weight,
        normalize_q_values = True,
        target_goal_prob = 0.5
    )

    device = actor_trainer.device

    # landing pad target goal: at coordinates (0, 0), zero velocity/angle, and legs touching (1, 1)

    base_actor_goal = tensor([0., 0., 0., 0., 0., 0., 1., 1.], device = device)

    # maybe discount conditioning embedding for rollout

    discount_cond = None
    if discount_condition:
        discount_cond = rearrange(default_discount_transform(rollout_discount), '1 d -> d').to(device)

    # tracking

    rolling_reward = deque(maxlen = 100)
    all_rewards = []
    all_lengths = []

    for eps in range(num_episodes):

        state, _ = env.reset()

        cum_reward = 0.
        eps_steps = 0

        is_exploring = torch.rand((), device = device) < exploration_random_goal_prob

        eps_goal = base_actor_goal

        if is_exploring:
            eps_goal = sample_random_state(
                replay_buffer,
                env,
                exploration_sample_from_buffer_prob
            ).to(device)

        states = []
        hard_one_hots = []

        for _ in range(max_timesteps):

            actor_encoder.eval()

            curr_state = from_numpy(state).to(device)

            actor_inputs = [curr_state, eps_goal]
            if exists(discount_cond):
                actor_inputs.append(discount_cond)

            action_logits = actor_encoder(cat(actor_inputs, dim = -1))

            action = actor_readout.sample(action_logits)

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

            states.append(state)
            hard_one_hots.append(F.one_hot(action.long(), num_classes = dim_action).float().detach().cpu())

            cum_reward += reward
            eps_steps += 1

            done = truncated or terminated

            if done:
                break

            state = next_state

        # store episode

        if len(states) >= 2:
            replay_buffer.store_episode(
                state = states,
                reward = [1.] * len(states),
                action_hard_one_hot = hard_one_hots
            )

        rolling_reward.append(cum_reward)
        all_rewards.append(cum_reward)
        all_lengths.append(eps_steps)

        # train the critic and actor

        if (eps + 1) >= num_episodes_before_learn and divisible_by(eps + 1, learn_every_eps):

            data = replay_buffer.get_all_data(
                fields = ['state', 'action_hard_one_hot'],
                meta_fields = ['episode_lens']
            )

            trajectories = data['state']
            episode_lens = data['episode_lens']

            cl_loss = critic_trainer(
                trajectories,
                cl_train_steps,
                lens = episode_lens,
                actions = data['action_hard_one_hot'],
                pbar = False
            )

            actor_loss = actor_trainer(
                trajectories,
                actor_num_train_steps,
                lens = episode_lens,
                target_goals = base_actor_goal,
                pbar = False
            )

        avg_reward = sum(rolling_reward) / len(rolling_reward)

        if divisible_by(eps + 1, 10) or (eps + 1) == 1:
            print(f'episode {eps + 1:4d} | reward: {cum_reward:6.1f} | avg reward (last 100): {avg_reward:6.1f} | steps: {eps_steps}')

        if divisible_by(eps + 1, 50):
            with open(reward_json_path, 'w') as f:
                json.dump(dict(rewards = all_rewards, lengths = all_lengths), f)

        if divisible_by(eps + 1, save_checkpoint_every):
            torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-{eps + 1}.pt')

    # save final

    torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-final.pt')

    with open(reward_json_path, 'w') as f:
        json.dump(dict(rewards = all_rewards, lengths = all_lengths), f)

    env.close()

# fire

if __name__ == '__main__':
    Fire(main)
