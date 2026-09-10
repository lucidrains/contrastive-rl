from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Callable

import torch
from torch import cat, arange, tensor, from_numpy, is_tensor, Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader

from einops import einsum, rearrange, repeat
from torch_einops_utils import z_score, lens_to_mask

from accelerate import Accelerator
from tqdm import tqdm

from contrastive_rl_pytorch.distributed import is_distributed, AllGather

# ein

# b - batch
# d - feature dimension (observation of embed)
# t - time
# n - num trajectories
# na - num actions

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def compact(arr):
    return [v for v in arr if exists(v)]

def compact_with_inverse(arr):
    indices = [i for i, v in enumerate(arr) if exists(v)]
    compacted = [arr[i] for i in indices]

    def inverse(values):
        out = [None] * len(arr)
        for i, val in zip(indices, values):
            out[i] = val
        return out

    return compacted, inverse

def divisible_by(num, den):
    return (num % den) == 0

def identity(t):
    return t

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def l2norm(t):
    return F.normalize(t, dim = -1)

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def arange_from_tensor_dim(t, dim = 0):
    length = t.shape[dim]
    return arange(length, device = t.device)

# similarity helper

def calc_similarity(
    embeds1,
    embeds2,
    *,
    use_euclidean = False,
    all_pairs = False
):
    if use_euclidean:
        if all_pairs:
            return -torch.cdist(embeds1, embeds2)
        return -(embeds1 - embeds2).norm(dim = -1)

    if all_pairs:
        return einsum(embeds1, embeds2, 'i d, j d -> i j')

    return einsum(embeds1, embeds2, '... d, ... d -> ...')

# sample random state

def sample_random_state(
    replay_buffer,
    env,
    exploration_sample_from_buffer_prob = 0.5,
):
    if replay_buffer.num_episodes > 0 and torch.rand(()) < exploration_sample_from_buffer_prob:
        # sample from buffer

        all_states = replay_buffer.get_all_data(fields = ['state'])['state']
        states = rearrange(all_states, '... d -> (...) d')

        num_states = states.shape[0]
        rand_id = torch.randint(0, num_states, (1,), device = states.device)

        random_state = states[rand_id]

        return rearrange(random_state, '1 d -> d')

    # sample from env

    state = env.observation_space.sample()
    return from_numpy(state).float()

# truncated geometric time sampling

def sample_truncated_geometric(
    max_steps: Tensor | int,
    discount: float | Tensor,
    *,
    rand_uniform: Tensor | None = None,
    eps = 1e-20,
    device: torch.device | None = None
) -> Tensor:
    # truncated geometric in [1, max_steps]

    if not is_tensor(max_steps):
        max_steps = tensor(max_steps, device = device)

    if not is_tensor(discount):
        discount = tensor(discount, device = max_steps.device, dtype = torch.float32)
    else:
        discount = discount.to(device = max_steps.device, dtype = torch.float32)

    if not exists(rand_uniform):
        rand_uniform = torch.rand_like(max_steps, dtype = torch.float32)

    is_undiscounted = discount >= (1. - eps)

    safe_discount = torch.where(is_undiscounted, tensor(0.5, device = discount.device), discount)
    prob_reach = 1. - safe_discount ** max_steps

    delta_undiscounted = rand_uniform * max_steps
    delta_discounted = log(1. - rand_uniform * prob_reach, eps = eps) / log(safe_discount, eps = eps)

    delta = torch.where(is_undiscounted, delta_undiscounted, delta_discounted)
    delta = delta.floor().long() + 1

    return delta.clamp(min = 1).minimum(max_steps)

sample_truncated_geometric_time = sample_truncated_geometric

# contrastive wrapper module

class ContrastiveLearning(Module):
    def __init__(
        self,
        l2norm_embed = True,
        learned_temp = True,
        use_euclidean = False
    ):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.use_euclidean = use_euclidean

        self.learned_log_temp = None
        if learned_temp:
            self.learned_log_temp = Parameter(tensor(1.))

    @property
    def scale(self):
        return self.learned_log_temp.exp() if exists(self.learned_log_temp) else 1.

    def forward(
        self,
        embeds1,
        embeds2,
        return_contrastive_score = False
    ):
        if self.l2norm_embed:
            embeds1, embeds2 = map(l2norm, (embeds1, embeds2))

        sim = calc_similarity(
            embeds1,
            embeds2,
            use_euclidean = self.use_euclidean,
            all_pairs = not return_contrastive_score
        )

        sim = sim * self.scale

        if return_contrastive_score:
            return sim

        # labels, which is 1 across diagonal

        labels = arange_from_tensor_dim(embeds1, dim = 0)

        # transpose

        sim_transpose = rearrange(sim, 'i j -> j i')

        loss = (
            F.cross_entropy(sim, labels) +
            F.cross_entropy(sim_transpose, labels)
        ) * 0.5

        return loss

class SigmoidContrastiveLearning(Module):
    def __init__(
        self,
        bias = 0.,
        l2norm_embed = False,
        learned_scale = False,
        use_euclidean = False
    ):
        super().__init__()
        self.bias = bias
        self.l2norm_embed = l2norm_embed
        self.use_euclidean = use_euclidean

        self.learned_log_scale = None
        if learned_scale:
            self.learned_log_scale = Parameter(tensor(1.)) # starts at ~2.7

    @property
    def scale(self):
        return self.learned_log_scale.exp() if exists(self.learned_log_scale) else 1.

    def forward(
        self,
        embeds1,
        embeds2,
        return_contrastive_score = False
    ):
        if self.l2norm_embed:
            embeds1, embeds2 = map(l2norm, (embeds1, embeds2))

        sim = calc_similarity(
            embeds1,
            embeds2,
            use_euclidean = self.use_euclidean,
            all_pairs = not return_contrastive_score
        )

        sim = sim * self.scale + self.bias

        if return_contrastive_score:
            return sim

        # labels

        labels = torch.eye(sim.shape[0], device = sim.device)

        # binary cross entropy

        loss = F.binary_cross_entropy_with_logits(sim, labels)

        return loss

# contrastive wrapper module

class ContrastiveWrapper(Module):
    def __init__(
        self,
        encoder: Module,
        contrastive_learn: Module,
        future_encoder: Module | None = None
    ):
        super().__init__()

        self.encode = encoder
        self.encode_future = default(future_encoder, encoder)

        self.contrastive_learn = contrastive_learn
        self.all_gather = AllGather()

    @property
    def scale(self):
        return self.contrastive_learn.scale

    def forward(
        self,
        past,     # (b d)
        future,   # (b d)
        past_action = None # (b na)
    ):
        if exists(past_action):
            if past_action.ndim > 2:
                past_action = rearrange(past_action, 'b c ... -> b (c ...)')

            past = cat((past, past_action), dim = -1)

        encoded_past = self.encode(past)
        encoded_future = self.encode_future(future)

        if is_distributed():
            encoded_past, _ = self.all_gather(encoded_past)
            encoded_future, _ = self.all_gather(encoded_future)

        return self.contrastive_learn(encoded_past, encoded_future)

# contrastive RL trainer

class ContrastiveRLTrainer(Module):
    def __init__(
        self,
        encoder: Module,
        future_encoder: Module | None = None,
        batch_size = 256,
        repetition_factor = 2,
        learning_rate = 3e-4,
        weight_decay = 0.,
        max_grad_norm = 0.5,
        discount = 0.99,
        action_chunk_size = 1,
        contrastive_learn: Module | None = None,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        cpu = False,
        state_to_goal_fn: Callable = identity,
        state_to_critic_state_fn: Callable = identity
    ):
        super().__init__()

        self.state_to_goal_fn = state_to_goal_fn
        self.state_to_critic_state_fn = state_to_critic_state_fn

        self.accelerator = Accelerator(cpu = cpu, **accelerate_kwargs)

        if not exists(contrastive_learn):
            contrastive_learn = ContrastiveLearning()

        contrast_wrapper = ContrastiveWrapper(
            encoder = encoder,
            future_encoder = future_encoder,
            contrastive_learn = contrastive_learn
        )

        assert divisible_by(batch_size, repetition_factor)
        self.batch_size = batch_size // repetition_factor   # effective batch size is smaller and then repeated
        self.repetition_factor = repetition_factor          # the in-trajectory repetition factor - basically having the network learn to distinguish negative features from within the same trajectory
        self.max_grad_norm = max_grad_norm
        self.discount = discount
        self.action_chunk_size = action_chunk_size

        optimizer = AdamW(contrast_wrapper.parameters(), lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)

        (
            self.contrast_wrapper,
            self.optimizer,
        ) = self.accelerator.prepare(
            contrast_wrapper,
            optimizer,
        )

    @property
    def unwrapped_contrast_wrapper(self):
        return self.accelerator.unwrap_model(self.contrast_wrapper)

    @property
    def scale(self):
        return self.unwrapped_contrast_wrapper.scale

    @property
    def use_sigmoid(self):
        return self.contrast_wrapper.use_sigmoid

    @property
    def sigmoid_bias(self):
        return self.contrast_wrapper.sigmoid_bias

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def forward(
        self,
        trajectories,       # (n t d)
        num_train_steps,
        *,
        lens = None,        # (n)
        actions = None,     # (n na)
        goal_state = None,  # (n t dg)
        pbar = None
    ):
        traj_var_lens = exists(lens)

        max_traj_len = trajectories.shape[1]

        assert max_traj_len >= 2
        assert not exists(lens) or (lens >= 2).all()

        # dataset and dataloader

        all_data = dict(states = trajectories, lens = lens, actions = actions, goal_states = goal_state)

        keys = list(all_data.keys())
        values = list(all_data.values())

        values_exist, inverse_compact = compact_with_inverse(values)

        dataset = TensorDataset(*values_exist)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, drop_last = False)

        # prepare

        dataloader = self.accelerator.prepare(dataloader)

        iter_dataloader = cycle(dataloader)

        # training steps

        loss_item = 0.

        if pbar is False:
            pbar = partial(tqdm, disable = True)
        elif not exists(pbar):
            pbar = tqdm

        pbar_instance = pbar(range(num_train_steps), disable = not self.accelerator.is_main_process)

        for _ in pbar_instance:

            data = next(iter_dataloader)

            data_dict = dict(zip(keys, inverse_compact(data)))

            trajs = data_dict['states']

            trajs = repeat(trajs, 'b ... -> (b r) ...', r = self.repetition_factor)

            # handle goal

            goal = trajs

            if exists(data_dict['goal_states']):
                goal = data_dict['goal_states']
                goal = repeat(goal, 'b ... -> (b r) ...', r = self.repetition_factor)

            # handle trajectory lens

            if exists(data_dict['lens']):
                traj_lens = data_dict['lens']
                traj_lens = repeat(traj_lens, 'b ... -> (b r) ...', r = self.repetition_factor)

            # batch arange for indexing out past future observations

            batch_size = trajs.shape[0]
            batch_arange = arange_from_tensor_dim(trajs, dim = 0)

            # get past times

            if traj_var_lens:
                past_times = torch.rand((batch_size, 1), device = self.device).mul(traj_lens[:, None] - 1).floor().long()
            else:
                past_times = torch.randint(0, max_traj_len - 1, (batch_size, 1), device = self.device)

            clamp_traj_len = (max_traj_len - 1) if not traj_var_lens else rearrange(traj_lens - 1, 'b -> b 1')

            # future times drawn from geometric distribution truncated to remaining length

            remainder_steps = clamp_traj_len - past_times
            delta_times = sample_truncated_geometric(remainder_steps, self.discount)
            future_times = past_times + delta_times

            # pick out the past and future observations as positive pairs

            batch_arange = rearrange(batch_arange, '... -> ... 1')

            past_obs = trajs[batch_arange, past_times]
            future_obs = goal[batch_arange, future_times]

            past_obs, future_obs = tuple(rearrange(t, 'b 1 ... -> b ...') for t in (past_obs, future_obs))

            past_obs = self.state_to_critic_state_fn(past_obs)
            future_obs = self.state_to_goal_fn(future_obs)

            # handle maybe action

            past_action = None

            if exists(data_dict['actions']):
                actions = data_dict['actions']
                actions = repeat(actions, 'b ... -> (b r) ...', r = self.repetition_factor)

                if self.action_chunk_size > 1:
                    # gather the action chunk starting at each past time, clamped to the trajectory end

                    max_action_idx = (actions.shape[1] - 1) if not traj_var_lens else rearrange(traj_lens - 1, 'b -> b 1')
                    action_offsets = arange(self.action_chunk_size, device = self.device)
                    action_times = (past_times + action_offsets).clamp(max = max_action_idx)
                    past_action = actions[batch_arange, action_times]
                else:
                    past_action = actions[batch_arange, past_times]
                    past_action = rearrange(past_action, 'b 1 ... -> b ...')

            # contrastive learning

            loss = self.contrast_wrapper(past_obs, future_obs, past_action)

            loss_item = loss.item()

            pbar_instance.set_description(f'loss: {loss_item:.3f}')

            # backwards and optimizer step

            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.contrast_wrapper.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_item

# training the actor

class ActorTrainer(Module):
    def __init__(
        self,
        actor: Module,
        encoder: Module,
        goal_encoder: Module,
        batch_size = 32,
        learning_rate = 3e-4,
        weight_decay = 0.,
        max_grad_norm = 0.5,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        softmax_actor_output = False,
        contrastive_learn: Module | None = None,
        cpu = False,
        action_entropy_loss_weight = 0.,
        state_to_goal_fn: Callable = identity,
        state_to_actor_state_fn: Callable = identity,
        state_to_critic_state_fn: Callable = identity,
        num_discrete_actions: int | None = None,
        normalize_q_values = True,
        target_goal_prob = 0.5
    ):
        super().__init__()

        self.num_discrete_actions = num_discrete_actions
        self.normalize_q_values = normalize_q_values
        self.target_goal_prob = target_goal_prob

        self.state_to_goal_fn = state_to_goal_fn
        self.state_to_actor_state_fn = state_to_actor_state_fn
        self.state_to_critic_state_fn = state_to_critic_state_fn

        self.accelerator = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.max_grad_norm = max_grad_norm
        self.action_entropy_loss_weight = action_entropy_loss_weight

        optimizer = AdamW(actor.parameters(), lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)
        self.actor = actor

        self.softmax_actor_output = softmax_actor_output

        if not exists(contrastive_learn):
            contrastive_learn = ContrastiveLearning()

        self.contrastive_learn = contrastive_learn

        (
            self.actor,
            self.optimizer,
        ) = self.accelerator.prepare(
            actor,
            optimizer
        )

        self.batch_size = batch_size

        self.goal_encoder = goal_encoder
        self.encoder = encoder

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def forward(
        self,
        trajectories,
        num_train_steps,
        *,
        lens = None,
        sample_fn = None,
        entropy_fn = None,
        pbar = None,
        target_goals = None
    ):
        device = self.device

        # setup models

        goal_encoder = deepcopy(self.goal_encoder).to(device)
        encoder = deepcopy(self.encoder).to(device)

        goal_encoder.requires_grad_(False)
        encoder.requires_grad_(False)

        goal_encoder.eval()
        encoder.eval()

        if exists(target_goals):
            if not is_tensor(target_goals):
                target_goals = tensor(target_goals, device = device, dtype = torch.float32)
            else:
                target_goals = target_goals.to(device)

            if target_goals.ndim == 1:
                target_goals = rearrange(target_goals, 'd -> 1 d')

            num_target_goals = target_goals.shape[0]

        if not is_tensor(trajectories):
            trajectories = from_numpy(trajectories)

        if exists(lens):
            lens = tensor(lens) if not is_tensor(lens) else lens
            traj_len = trajectories.shape[-2]
            mask = lens_to_mask(lens, max_len = traj_len)
            states = trajectories[mask]
        else:
            states = rearrange(trajectories, '... d -> (...) d')

        dataset = TensorDataset(states)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        goal_dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        dataloader, goal_dataloader = self.accelerator.prepare(dataloader, goal_dataloader)

        iter_dataloader = cycle(dataloader)
        iter_goal_dataloader = cycle(goal_dataloader)

        # training loop

        self.actor.train()

        if pbar is False:
            pbar = partial(tqdm, disable = True)
        elif not exists(pbar):
            pbar = tqdm

        pbar_instance = pbar(range(num_train_steps), disable = not self.accelerator.is_main_process)

        for _ in pbar_instance:

            (state,) = next(iter_dataloader)
            (goal,) = next(iter_goal_dataloader)

            batch = state.shape[0]

            if exists(target_goals):
                rand_indices = torch.randint(0, num_target_goals, (batch,), device = device)
                batch_target_goals = target_goals[rand_indices]
                mix_mask = rearrange(torch.rand(batch, device = device) < self.target_goal_prob, 'b -> b 1')
                goal = torch.where(mix_mask, batch_target_goals, goal)

            # forward state and goal

            actor_state = self.state_to_actor_state_fn(state)
            actor_goal = self.state_to_goal_fn(goal)

            action_logits = self.actor(cat((actor_state, actor_goal), dim = -1))

            if exists(self.num_discrete_actions):
                num_actions = self.num_discrete_actions
                action_probs = action_logits.softmax(dim = -1)
                critic_state = self.state_to_critic_state_fn(state)

                # evaluate all discrete action candidates at once

                with torch.no_grad():
                    all_actions = torch.eye(num_actions, device = device)
                    all_actions = repeat(all_actions, 'a na -> b a na', b = batch)
                    repeated_states = repeat(critic_state, 'b d -> b a d', a = num_actions)

                    state_actions = cat((repeated_states, all_actions), dim = -1)
                    encoded_state_actions = encoder(rearrange(state_actions, 'b a d -> (b a) d'))
                    encoded_state_actions = rearrange(encoded_state_actions, '(b a) d -> b a d', a = num_actions)

                    encoded_goal = goal_encoder(actor_goal)
                    repeated_goals = repeat(encoded_goal, 'b d -> b a d', a = num_actions)

                    q_values = self.contrastive_learn(
                        encoded_state_actions,
                        repeated_goals,
                        return_contrastive_score = True
                    )

                    if self.normalize_q_values:
                        q_values = z_score(q_values, dim = -1)

                # closed-form policy expectation

                expected_q = einsum(action_probs, q_values, 'b a, b a -> b')
                loss = -expected_q.mean()

                if self.action_entropy_loss_weight > 0.:
                    entropy = -einsum(action_probs, log(action_probs), 'b a, b a -> b')
                    loss = loss - entropy.mean() * self.action_entropy_loss_weight

            else:
                if self.softmax_actor_output:
                    action = action_logits.softmax(dim = -1)
                elif exists(sample_fn):
                    action = sample_fn(action_logits)
                else:
                    action = action_logits

                # encode state

                if action.ndim > 2:
                    action = rearrange(action, 'b c ... -> b (c ...)')

                critic_state = self.state_to_critic_state_fn(state)
                encoded_state_action = encoder(cat((critic_state, action), dim = -1))

                with torch.no_grad():
                    encoded_goal = goal_encoder(actor_goal)

                sim = self.contrastive_learn(
                    encoded_state_action,
                    encoded_goal,
                    return_contrastive_score = True
                )

                loss = -sim.mean()

                if self.action_entropy_loss_weight > 0. and exists(entropy_fn):
                    entropy = entropy_fn(action_logits)
                    loss = loss - entropy.mean() * self.action_entropy_loss_weight

            self.accelerator.backward(loss)

            pbar_instance.set_description(f'actor loss: {loss.item():.3f}')

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()
