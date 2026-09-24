# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "fire",
#   "gymnasium[box2d]",
#   "mean-conc-beta>=0.2.1",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.6.1",
#   "einops"
# ]
# ///

from __future__ import annotations

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import json
from collections import deque

import torch
from torch import from_numpy, cat, tensor

import numpy as np
import gymnasium as gym
from fire import Fire
from einops import rearrange

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
    sample_random_state
)

from x_mlps_pytorch import AttnResidualNormedMLP
from mean_conc_beta import Beta

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main

def main(
    num_episodes = 2000,
    max_timesteps = 500,
    num_episodes_before_learn = 8,
    learn_every_eps = 4,
    buffer_size = 512,
    video_folder = './recordings_continuous',
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
    num_streams = 1,
    use_rmsnorm = True,
    weight_decay = 1e-4,
    max_grad_norm = 0.5,
    repetition_factor = 1,
    use_sigmoid = True,
    sigmoid_bias = 0.,
    use_euclidean = False,
    discount = 0.99,
    exploration_random_goal_prob = 0.05,
    exploration_sample_from_buffer_prob = 0.5,
    action_entropy_loss_weight = 0.005,
    action_chunk_size = 1,
    beta_pos_fn = 'softplus',
    beta_max_unimodal_floor = 50.,
    beta_squash_fn = 'leaky_tanh',
    beta_detach_entropy_mean = True,
    save_checkpoint_every = 200,
    checkpoint_folder = './checkpoints-lunar-continuous',
    reward_json_path = './lunar_continuous_rewards.json',
    replay_buffer_folder = None,
    cpu = False,
    seed = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    replay_buffer_folder = default(replay_buffer_folder, f'./replay-lunar-continuous-chunk{action_chunk_size}')

    # folders

    os.makedirs(video_folder, exist_ok = True)
    os.makedirs(checkpoint_folder, exist_ok = True)

    # env

    record_video = exists(render_every_eps) and int(render_every_eps) > 0
    render_mode = 'rgb_array' if record_video else None

    env = gym.make('LunarLander-v3', continuous = True, render_mode = render_mode)

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder = video_folder,
            episode_trigger = lambda ep: divisible_by(ep, int(render_every_eps)),
            name_prefix = f'lunar-cont-c{action_chunk_size}'
        )

    dim_state = 8
    dim_goal = 8
    dim_action = 2

    # replay buffer

    replay_buffer = ReplayBuffer(
        replay_buffer_folder,
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            reward = ('float', 1),
            action = ('float', dim_action)
        ),
        circular = True,
        overwrite = True
    )

    # models

    actor_distr = Beta(
        pos_fn = beta_pos_fn,
        max_unimodal_floor = beta_max_unimodal_floor,
        squash_fn = beta_squash_fn,
        detach_entropy_mean = beta_detach_entropy_mean
    )

    actor_encoder = AttnResidualNormedMLP(
        dim_in = dim_state + dim_goal,
        dim = actor_dim,
        depth = actor_depth,
        dim_out = action_chunk_size * dim_action * 2, # raw mean and raw conc for each action in chunk
        use_rmsnorm = use_rmsnorm,
        num_streams = num_streams
    )

    contrastive_embed_dim = dim_contrastive_embed

    critic_encoder = AttnResidualNormedMLP(
        dim_in = dim_state + action_chunk_size * dim_action,
        dim = critic_dim,
        depth = critic_depth,
        dim_out = contrastive_embed_dim,
        use_rmsnorm = use_rmsnorm,
        num_streams = num_streams
    )

    goal_encoder = AttnResidualNormedMLP(
        dim_in = dim_goal,
        dim = goal_dim,
        depth = goal_depth,
        dim_out = contrastive_embed_dim,
        use_rmsnorm = use_rmsnorm,
        num_streams = num_streams
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
        discount = discount,
        action_chunk_size = action_chunk_size,
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
        cpu = cpu,
        contrastive_learn = contrastive_learn,
        action_entropy_loss_weight = action_entropy_loss_weight,
        normalize_q_values = True,
        target_goal_prob = 0.5
    )

    device = actor_trainer.device

    # landing pad target goal: at coordinates (0, 0), zero velocity/angle, and legs touching (1, 1)

    base_actor_goal = tensor([0., 0., 0., 0., 0., 0., 1., 1.], device = device)

    # action distribution helpers

    def to_dist_params(logits):
        # rearrange and constitute the action chunk dimension - using c for chunk
        return rearrange(logits, '... (c a d) -> ... c a d', c = action_chunk_size, a = dim_action, d = 2)

    def sample_fn(logits, differentiable = False):
        dist = actor_distr(to_dist_params(logits))
        return dist.rsample() if differentiable else dist.sample()

    def entropy_fn(logits):
        return actor_distr.entropy(to_dist_params(logits), sum_action_dim = False)

    # tracking

    def state_to_actor_state(s):
        return from_numpy(s).to(device)

    rolling_reward = deque(maxlen = 100)
    all_rewards = []
    all_lengths = []
    best_reward = float('-inf')

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
        actions = []

        while eps_steps < max_timesteps:

            actor_state = state_to_actor_state(state)

            with torch.no_grad():
                action_logits = actor_encoder(torch.cat((actor_state, eps_goal), dim = -1))
                action_chunk = sample_fn(action_logits, differentiable = False)

            action_chunk = action_chunk.cpu().numpy()

            for a in range(action_chunk_size):
                action = action_chunk[a]

                next_state, reward, terminated, truncated, _ = env.step(action)

                done = terminated or truncated

                cum_reward += reward
                eps_steps += 1

                states.append(state)
                actions.append(action)

                state = next_state

                if done or eps_steps >= max_timesteps:
                    break

            if done:
                break

        # rename video if recorded

        if record_video and env.recording:
            old_vid = os.path.join(video_folder, f'lunar-cont-c{action_chunk_size}-episode-{env.episode_id}.mp4')
            env.stop_recording()
            if os.path.exists(old_vid):
                new_vid = os.path.join(video_folder, f'lunar-cont-c{action_chunk_size}-ep{eps + 1:04d}-rew{cum_reward:+.1f}.mp4')
                os.rename(old_vid, new_vid)

        # store episode

        if len(states) >= 2:
            replay_buffer.store_episode(
                state = states,
                reward = [1.] * len(states),
                action = actions
            )

        rolling_reward.append(cum_reward)
        all_rewards.append(cum_reward)
        all_lengths.append(eps_steps)

        # train the critic and actor

        if (eps + 1) >= num_episodes_before_learn and divisible_by(eps + 1, learn_every_eps):

            data = replay_buffer.get_all_data(
                fields = ['state', 'action'],
                meta_fields = ['episode_lens']
            )

            trajectories = data['state']
            episode_lens = data['episode_lens']

            cl_loss = critic_trainer(
                trajectories,
                cl_train_steps,
                lens = episode_lens,
                actions = data['action'],
                pbar = False
            )

            actor_loss = actor_trainer(
                trajectories,
                actor_num_train_steps,
                lens = episode_lens,
                sample_fn = lambda logits: sample_fn(logits, differentiable = True),
                entropy_fn = entropy_fn,
                target_goals = base_actor_goal,
                pbar = False
            )

        avg_reward = sum(rolling_reward) / len(rolling_reward)

        if divisible_by(eps + 1, 10) or (eps + 1) == 1:
            print(f'episode {eps + 1:4d} | reward: {cum_reward:6.1f} | avg reward (last 100): {avg_reward:6.1f} | steps: {eps_steps}', flush = True)

        if len(rolling_reward) >= 20 and avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-best.pt')

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
