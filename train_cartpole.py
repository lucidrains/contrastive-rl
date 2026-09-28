# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout",
#   "env-ssl-wrapper>=0.4.7",
#   "fire",
#   "gymnasium[classic-control]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.6.1",
#   "tqdm",
#   "einops",
# ]
# ///

from __future__ import annotations

import os
from collections import deque

import torch
from torch import nn, from_numpy, cat, tensor
import torch.nn.functional as F

import numpy as np

import gymnasium as gym

from einops.layers.torch import Rearrange

from fire import Fire

from memmap_replay_buffer import ReplayBuffer

from env_ssl_wrapper import ActionChunkWrapper

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
    CategoricalActionAdapter,
    sample_random_state
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
    num_episodes = 2000,
    max_timesteps = 500,
    num_episodes_before_learn = 8,
    learn_every_eps = 4,
    buffer_size = 512,
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
    use_sigmoid_contrastive_learning = True,
    sigmoid_bias = 0.,
    cl_l2norm_embed = False,
    learned_scale = False,
    learned_temp = False,
    use_euclidean = False,
    critic_action_repr = None,   # None | 'raw_logits' | 'softmax_probs' - condition the critic on the action distribution parameters
    chunk_len = 1,               # number of actions executed per environment step - https://arxiv.org/abs/2608.30640
    discount = 0.99,
    target_goal_prob = 0.5,
    early_stop_reward = 475.0,
    temperature_init = 1.0,
    temperature_final = 0.1,
    temperature_anneal_eps = 35,
    eval_every_eps = 5,
    num_eval_episodes = 3,
    exploration_random_goal_prob = 0.05,
    exploration_sample_from_buffer_prob = 0.5,
    action_entropy_loss_weight = 0.005,
    normalize_q_values = True,
    save_checkpoint_every = 100,
    checkpoint_folder = './checkpoints-cartpole',
    replay_buffer_folder = './replay-cartpole',
    cpu = False,
    seed = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert critic_action_repr in (None, 'raw_logits', 'softmax_probs')

    # env

    env = ActionChunkWrapper(gym.make('CartPole-v1'), chunk_len = chunk_len)

    dim_state = 4
    dim_goal = 4
    dim_action = 2

    max_chunks = (max_timesteps + chunk_len - 1) // chunk_len

    # action adapter - condition the critic on the categorical distribution parameters, as proposed in
    # https://arxiv.org/abs/2506.16608 - He et al., distributions as actions

    use_action_adapter = exists(critic_action_repr) or chunk_len > 1

    action_adapter = None
    if use_action_adapter:
        action_adapter = CategoricalActionAdapter(
            dim_action,
            action_repr = default(critic_action_repr, 'softmax_probs')
        )

    # replay buffer

    action_field = 'action_repr' if exists(action_adapter) else 'action_hard_one_hot'
    dim_action_field = dim_action * chunk_len if exists(action_adapter) else dim_action

    replay_buffer = ReplayBuffer(
        replay_buffer_folder,
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            reward = ('float', 1),
            action_hard_one_hot = ('float', dim_action_field),
            action_repr = ('float', dim_action_field)
        ),
        circular = True,
        overwrite = True
    )

    # models

    actor_net = AttnResidualNormedMLP(
        dim_in = dim_state + dim_goal,
        dim = actor_dim,
        depth = actor_depth,
        dim_out = dim_action_field,
        use_rmsnorm = use_rmsnorm
    )

    actor_encoder = nn.Sequential(
        actor_net,
        Rearrange('... (c a) -> ... c a', c = chunk_len, a = dim_action)
    ) if chunk_len > 1 else actor_net

    actor_readout = Readout(num_discrete = dim_action, dim = 0)

    contrastive_embed_dim = dim_contrastive_embed

    critic_encoder = AttnResidualNormedMLP(
        dim_in = dim_state + dim_action_field,
        dim = critic_dim,
        depth = critic_depth,
        dim_out = contrastive_embed_dim,
        use_rmsnorm = use_rmsnorm
    )

    goal_encoder = AttnResidualNormedMLP(
        dim_in = dim_goal,
        dim = goal_dim,
        depth = goal_depth,
        dim_out = contrastive_embed_dim,
        use_rmsnorm = use_rmsnorm
    )

    # contrastive learning module

    if use_sigmoid_contrastive_learning:
        contrastive_learn = SigmoidContrastiveLearning(
            bias = sigmoid_bias,
            l2norm_embed = cl_l2norm_embed,
            learned_scale = learned_scale,
            use_euclidean = use_euclidean
        )
    else:
        contrastive_learn = ContrastiveLearning(
            l2norm_embed = cl_l2norm_embed,
            learned_temp = learned_temp,
            use_euclidean = use_euclidean
        )

    # trainer

    critic_trainer = ContrastiveRLTrainer(
        critic_encoder,
        goal_encoder,
        batch_size = cl_batch_size,
        learning_rate = critic_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        repetition_factor = repetition_factor,
        discount = discount ** chunk_len,
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
        num_discrete_actions = dim_action if not exists(action_adapter) else None,
        action_adapter = action_adapter,
        cpu = cpu,
        contrastive_learn = contrastive_learn,
        action_entropy_loss_weight = action_entropy_loss_weight,
        normalize_q_values = normalize_q_values,
        target_goal_prob = target_goal_prob
    )

    device = actor_trainer.device

    # goal - the upright balanced state

    base_actor_goal = tensor([0., 0., 0., 0.], device = device)

    # training

    rolling_reward = deque(maxlen = 100)

    os.makedirs(checkpoint_folder, exist_ok = True)

    best_eval_reward = 0.0

    for eps in range(num_episodes):

        state, _ = env.reset()

        cum_reward = 0.
        eps_steps = 0

        # temperature anneals smoothly
        anneal_ratio = min(1.0, eps / max(1, temperature_anneal_eps))
        temp = temperature_init - anneal_ratio * (temperature_init - temperature_final)

        is_exploring = torch.rand((), device = device) < exploration_random_goal_prob

        eps_goal = base_actor_goal

        if is_exploring:
            eps_goal = sample_random_state(
                replay_buffer,
                env,
                exploration_sample_from_buffer_prob
            ).to(device)

        states = []
        actions_stored = []

        for _ in range(max_chunks):

            actor_encoder.eval()

            curr_state = from_numpy(state).to(device)

            action_logits = actor_encoder(cat((curr_state, eps_goal), dim = -1))

            action = actor_readout.sample(action_logits, temperature = temp)

            action_stored = action_adapter.to_critic(action_logits) if exists(action_adapter) \
                else F.one_hot(action.long(), num_classes = dim_action).float()

            action_chunk = action.reshape(1, chunk_len).cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action_chunk)

            states.append(state)
            actions_stored.append(action_stored.detach().cpu().flatten())

            cum_reward += float(np.sum(reward))
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
                **{action_field: actions_stored}
            )

        rolling_reward.append(cum_reward)

        # train the critic and actor

        if (eps + 1) >= num_episodes_before_learn and divisible_by(eps + 1, learn_every_eps):

            data = replay_buffer.get_all_data(
                fields = ['state', action_field],
                meta_fields = ['episode_lens']
            )

            trajectories = data['state']
            episode_lens = data['episode_lens']

            critic_trainer(
                trajectories,
                cl_train_steps,
                lens = episode_lens,
                actions = data[action_field],
                pbar = False
            )

            actor_trainer(
                trajectories,
                actor_num_train_steps,
                lens = episode_lens,
                target_goals = base_actor_goal,
                pbar = False
            )

        avg_reward = sum(rolling_reward) / len(rolling_reward)

        # periodic evaluation with greedy deterministic policy
        if divisible_by(eps + 1, eval_every_eps):
            actor_encoder.eval()
            eval_scores = []
            for _ in range(num_eval_episodes):
                eval_s, _ = env.reset()
                eval_tot = 0.0
                for _ in range(max_chunks):
                    eval_st = from_numpy(eval_s).to(device)
                    eval_logits = actor_encoder(cat((eval_st, base_actor_goal), dim = -1))
                    eval_act = eval_logits.argmax(dim = -1)
                    eval_chunk = eval_act.reshape(1, chunk_len).cpu().numpy()
                    eval_s, eval_r, eval_term, eval_trunc, _ = env.step(eval_chunk)
                    eval_tot += float(np.sum(eval_r))
                    if eval_term or eval_trunc:
                        break
                eval_scores.append(eval_tot)

            eval_mean = sum(eval_scores) / len(eval_scores)
            print(f'episode {eps + 1:3d} (temp: {temp:.2f}) | train: {cum_reward:5.0f} | eval (greedy): {eval_mean:5.1f} {eval_scores} | avg train: {avg_reward:5.1f}')

            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-best.pt')

            if eval_mean >= early_stop_reward:
                print(f'\nCartPole solved at episode {eps + 1} with evaluation score {eval_mean:.1f}!')
                break
        elif (eps + 1) % 10 == 0:
            print(f'episode {eps + 1:3d} | reward: {cum_reward:5.0f} | avg reward: {avg_reward:5.1f}')

        if (eps + 1) % save_checkpoint_every == 0:
            torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-{eps + 1}.pt')

    # final save

    torch.save(actor_encoder.state_dict(), f'{checkpoint_folder}/actor-final.pt')

# fire

if __name__ == '__main__':
    Fire(main)
