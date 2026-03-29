from __future__ import annotations

from copy import deepcopy

import torch
from torch import cat, arange, tensor, from_numpy, is_tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F

from accelerate import Accelerator

from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Any

import einx
from einops import einsum, rearrange, repeat

from x_mlps_pytorch import ResidualNormedMLP, AttnResidualNormedMLP

from contrastive_rl_pytorch.distributed import is_distributed, AllGather

from tqdm import tqdm

# ein

# b - batch
# d - feature dimension (observation of embed)
# t - time
# n - num trajectories
# na - num actions

# helper functions

def exists(v):
    return v is not None

def compact(arr):
    return [*filter(exists, arr)]

def compact_with_inverse(arr):
    indices = [i for i, el in enumerate(arr) if exists(el)]

    def inverse(out):
        nones = [None] * len(arr)

        for i, out_el in zip(indices, out):
            nones[i] = out_el

        return nones

    return compact(arr), inverse

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t):
    return t

def l2norm(t):
    return F.normalize(t, dim = -1)

def arange_from_tensor_dim(t, dim):
    device = t.device
    return torch.arange(t.shape[dim], device = device)

def cycle(dl):
    assert len(dl) > 0

    while True:
        for batch in dl:
            yield batch

# tensor functions

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


def sigreg_loss(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17
):
    # Randall Balestriero - https://arxiv.org/abs/2511.08544

    dim, device = x.shape[-1], x.device

    # slice sampling

    rand_projs = torch.randn((num_slices, dim), device = device)
    rand_projs = l2norm(rand_projs)

    # integration points

    t = torch.linspace(*domain, num_knots, device = device)

    # theoretical CF for N(0, 1) and Gauss. window

    exp_f = (-0.5 * t.square()).exp()

    # empirical CF

    x_t = einsum(x, rand_projs, '... d, m d -> ... m')
    x_t = rearrange(x_t, '... m -> (...) m')

    x_t = rearrange(x_t, 'n m -> n m 1') * t
    ecf = (1j * x_t).exp().mean(dim = 0)

    # weighted L2 distance

    err = ecf.sub(exp_f).abs().square().mul(exp_f)

    return torch.trapz(err, t, dim = -1).mean()

# fourier encode

class FourierEncode(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        assert divisible_by(dim, 2)
        self.dim = dim
        self.register_buffer('weight', torch.randn(1, dim // 2))

    def forward(self, x):
        if x.ndim == 1:
            x = rearrange(x, '... -> ... 1')
        x = x @ self.weight
        return cat((x.sin(), x.cos()), dim = -1)

# contrastive wrapper module

class ContrastiveLearning(Module):
    def __init__(
        self,
        l2norm_embed = True,
        learned_temp = True
    ):
        super().__init__()
        self.l2norm_embed = l2norm_embed

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

        if return_contrastive_score:
            sim = (embeds1 * embeds2).sum(dim = -1)
        else:
            sim = einsum(embeds1, embeds2, 'i d, j d -> i j')

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
        bias = -10.,
        l2norm_embed = True,
        learned_scale = True
    ):
        super().__init__()
        self.bias = bias
        self.l2norm_embed = l2norm_embed

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

        if return_contrastive_score:
            sim = (embeds1 * embeds2).sum(dim = -1)
        else:
            sim = einsum(embeds1, embeds2, 'i d, j d -> i j')

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
        future_encoder: Module | None = None,
        sigreg_loss_weight = 0.
    ):
        super().__init__()

        self.encode = encoder
        self.encode_future = default(future_encoder, encoder)

        self.contrastive_learn = contrastive_learn
        self.all_gather = AllGather()

        self.sigreg_loss_weight = sigreg_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        past,     # (b d)
        future,   # (b d)
        past_action = None # (b na)
    ):

        if exists(past_action):
            past = cat((past, past_action), dim = -1)

        encoded_past = self.encode(past)
        encoded_future = self.encode_future(future)

        if is_distributed():
            encoded_past, _ = self.all_gather(encoded_past)
            encoded_future, _ = self.all_gather(encoded_future)

        loss = self.contrastive_learn(encoded_past, encoded_future)

        sigreg_loss_val = self.zero
        if self.sigreg_loss_weight > 0.:
            sigreg_loss_val = (sigreg_loss(encoded_past) + sigreg_loss(encoded_future)) * 0.5
            loss = loss + sigreg_loss_val * self.sigreg_loss_weight

        return loss, sigreg_loss_val

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
        reward_part_of_goal = False,
        reward_norm = 1.0,
        reward_fourier_encode = False,
        reward_fourier_dim = 16,
        contrastive_learn: Module | None = None,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        cpu = False,
        sigreg_loss_weight = 0.,
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
            contrastive_learn = contrastive_learn,
            sigreg_loss_weight = sigreg_loss_weight
        )

        assert divisible_by(batch_size, repetition_factor)
        self.batch_size = batch_size // repetition_factor   # effective batch size is smaller and then repeated
        self.repetition_factor = repetition_factor          # the in-trajectory repetition factor - basically having the network learn to distinguish negative features from within the same trajectory
        self.max_grad_norm = max_grad_norm
        self.discount = discount

        self.reward_part_of_goal = reward_part_of_goal
        self.reward_norm = reward_norm
        self.reward_fourier_encode = None

        if reward_part_of_goal and reward_fourier_encode:
            self.reward_fourier_encode = FourierEncode(reward_fourier_dim)

        optimizer = AdamW(contrast_wrapper.parameters(), lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)

        (
            self.contrast_wrapper,
            self.optimizer,
        ) = self.accelerator.prepare(
            contrast_wrapper,
            optimizer,
        )

    @property
    def scale(self):
        learned_log_temp = getattr(self.contrast_wrapper.contrastive_learn, 'learned_log_temp', None)
        return learned_log_temp.exp().item() if exists(learned_log_temp) else 1.

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
        rewards = None,     # (n t)
        goal_state = None,   # (n t dg)
        pbar = None
    ):
        traj_var_lens = exists(lens)

        max_traj_len = trajectories.shape[1]

        assert max_traj_len >= 2
        assert not exists(lens) or (lens >= 2).all()

        # dataset and dataloader

        all_data = dict(states = trajectories, lens = lens, actions = actions, rewards = rewards, goal_states = goal_state)

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

        if not exists(pbar):
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

            # future times, using delta time drawn from geometric distribution

            delta_times = torch.empty(past_times.shape, dtype = torch.long, device = 'cpu').geometric_(1. - self.discount).to(self.device)
            future_times = past_times + delta_times.clamp(min = 1)

            # clamping future times by max_traj_len if not variable lengths else prepare variable length

            clamp_traj_len = (max_traj_len - 1) if not traj_var_lens else rearrange(traj_lens - 1, 'b -> b 1')

            # clamp

            future_times.clamp_(max = clamp_traj_len)

            # pick out the past and future observations as positive pairs

            batch_arange = rearrange(batch_arange, '... -> ... 1')

            past_obs = trajs[batch_arange, past_times]
            future_obs = goal[batch_arange, future_times]

            past_obs, future_obs = tuple(rearrange(t, 'b 1 ... -> b ...') for t in (past_obs, future_obs))

            past_obs = self.state_to_critic_state_fn(past_obs)
            future_obs = self.state_to_goal_fn(future_obs)

            if self.reward_part_of_goal:
                rewards = data_dict['rewards']
                assert exists(rewards), 'rewards must be passed if reward_part_of_goal is True'

                rewards = repeat(rewards, 'b ... -> (b r) ...', r = self.repetition_factor)
                picked_rewards = rearrange(rewards[batch_arange, future_times], 'b 1 ... -> b ...')

                future_obs_rewards = picked_rewards / self.reward_norm

                if exists(self.reward_fourier_encode):
                    self.reward_fourier_encode = self.reward_fourier_encode.to(self.device)
                    future_obs_rewards = self.reward_fourier_encode(future_obs_rewards)
                elif future_obs_rewards.ndim == 1:
                    future_obs_rewards = rearrange(future_obs_rewards, '... -> ... 1')

                future_obs = cat((future_obs, future_obs_rewards), dim = -1)

            # handle maybe action

            past_action = None

            if exists(data_dict['actions']):
                actions = data_dict['actions']
                actions = repeat(actions, 'b ... -> (b r) ...', r = self.repetition_factor)

                past_action = actions[batch_arange, past_times]
                past_action = rearrange(past_action, 'b 1 ... -> b ...')

            # contrastive learning

            loss, sigreg_loss_val = self.contrast_wrapper(past_obs, future_obs, past_action)

            loss_item = loss.item()
            sigreg_loss_item = sigreg_loss_val.item() if is_tensor(sigreg_loss_val) else sigreg_loss_val

            desc = f'loss: {loss_item:.3f}'

            if self.contrast_wrapper.sigreg_loss_weight > 0.:
                desc += f' | sigreg: {sigreg_loss_item:.3f}'

            pbar_instance.set_description(desc)

            # backwards and optimizer step

            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.contrast_wrapper.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_item, sigreg_loss_item

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
        reward_part_of_goal = False,
        reward_norm = 1.0,
        reward_fourier_encode = False,
        reward_fourier_dim = 16,
        contrastive_learn: Module | None = None,
        cpu = False,
        action_entropy_loss_weight = 0.,
        state_to_goal_fn: Callable = identity,
        state_to_actor_state_fn: Callable = identity,
        state_to_critic_state_fn: Callable = identity
    ):
        super().__init__()

        self.state_to_goal_fn = state_to_goal_fn
        self.state_to_actor_state_fn = state_to_actor_state_fn
        self.state_to_critic_state_fn = state_to_critic_state_fn

        self.accelerator = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.max_grad_norm = max_grad_norm
        self.action_entropy_loss_weight = action_entropy_loss_weight

        optimizer = AdamW(actor.parameters(), lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)
        self.actor = actor

        # in a recent CRL paper, they made the discovery that passing softmax output directly to critic (without any hard one-hot straight-through) can work

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

        self.reward_part_of_goal = reward_part_of_goal
        self.reward_norm = reward_norm
        self.reward_fourier_encode = None

        if reward_part_of_goal and reward_fourier_encode:
            self.reward_fourier_encode = FourierEncode(reward_fourier_dim)

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
        rewards = None,
        sample_fn = None,
        entropy_fn = None,
        q_critic = None,
        q_loss_weight = 0.,
        pbar = None
    ):

        device = self.device

        # setup models

        goal_encoder = deepcopy(self.goal_encoder).to(device)
        encoder = deepcopy(self.encoder).to(device)

        goal_encoder.requires_grad_(False)
        encoder.requires_grad_(False)

        goal_encoder.eval()
        encoder.eval()

        if not is_tensor(trajectories):
            trajectories = from_numpy(trajectories)
            
        if exists(lens):
            lens = tensor(lens) if not is_tensor(lens) else lens
            traj_len = trajectories.shape[-2]
            mask = einx.less('t, b -> b t', arange(traj_len, device = lens.device), lens)
            states = trajectories[mask]
        else:
            states = rearrange(trajectories, '... d -> (...) d')

        dataset = TensorDataset(states)

        goal_data = [states]

        if self.reward_part_of_goal:
            assert exists(rewards), 'rewards must be passed to actor trainer if reward_part_of_goal is True'

            if not is_tensor(rewards):
                rewards = from_numpy(rewards)

            goal_rewards = rewards[mask] if exists(lens) else rearrange(rewards, '... -> (...)')

            if goal_rewards.ndim == 1:
                goal_rewards = rearrange(goal_rewards, 'b -> b 1')

            goal_data.append(goal_rewards)

        goal_dataset = TensorDataset(*goal_data)

        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        goal_dataloader = DataLoader(goal_dataset, batch_size = self.batch_size, shuffle = True)

        dataloader, goal_dataloader = self.accelerator.prepare(dataloader, goal_dataloader)

        iter_dataloader = cycle(dataloader)
        iter_goal_dataloader = cycle(goal_dataloader)

        # training loop

        self.actor.train()

        if not exists(pbar):
            pbar = tqdm

        pbar_instance = pbar(range(num_train_steps), disable = not self.accelerator.is_main_process)

        for _ in pbar_instance:

            state, = next(iter_dataloader)
            goal, *maybe_goal_rewards = next(iter_goal_dataloader)

            # forward state and goal

            actor_state = self.state_to_actor_state_fn(state)
            actor_goal = self.state_to_goal_fn(goal)

            if self.reward_part_of_goal:
                goal_rewards, = maybe_goal_rewards

                goal_rewards = goal_rewards / self.reward_norm

                if exists(self.reward_fourier_encode):
                    self.reward_fourier_encode = self.reward_fourier_encode.to(self.device)
                    goal_rewards = self.reward_fourier_encode(goal_rewards)
                elif goal_rewards.ndim == 1:
                    goal_rewards = rearrange(goal_rewards, '... -> ... 1')

                actor_goal = cat((actor_goal, goal_rewards), dim = -1)

            action_logits = self.actor((actor_state, actor_goal))

            actor_output = sample_fn(action_logits) if exists(sample_fn) else action_logits

            if self.softmax_actor_output:
                action = actor_output.softmax(dim = -1)
            else:
                action = actor_output

            # encode state

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

            if q_loss_weight > 0. and exists(q_critic):
                with torch.no_grad():
                    q_values = q_critic(actor_state)

                if self.softmax_actor_output:
                    action_probs = action_logits.softmax(dim = -1)
                else:
                    action_probs = action_logits

                expected_q = (action_probs * q_values).sum(dim = -1)
                loss = loss - expected_q.mean() * q_loss_weight

            self.accelerator.backward(loss)

            pbar_instance.set_description(f'actor loss: {loss.item():.3f}')

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

# TD Trainer

class TDTrainer(Module):
    def __init__(
        self,
        q_critic,
        ema_q_critic,
        batch_size = 128,
        learning_rate = 3e-4,
        weight_decay = 0.,
        td_gamma = 0.9,
        max_grad_norm = 0.5,
        cpu = False,
        **adam_kwargs
    ):
        super().__init__()
        self.accelerator = Accelerator(cpu = cpu)
        self.device = self.accelerator.device

        self.q_critic = q_critic.to(self.device)
        self.ema_q_critic = ema_q_critic.to(self.device)

        self.batch_size = batch_size
        self.td_gamma = td_gamma
        self.max_grad_norm = max_grad_norm

        optimizer = AdamW(self.q_critic.parameters(), lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)

        self.q_critic, self.optimizer = self.accelerator.prepare(self.q_critic, optimizer)

    def forward(
        self,
        trajectories,
        num_train_steps,
        *,
        actions,
        rewards,
        lens,
        pbar = None
    ):
        device = self.device

        if not is_tensor(trajectories): trajectories = from_numpy(trajectories)
        if not is_tensor(actions): actions = from_numpy(actions)
        if not is_tensor(rewards): rewards = from_numpy(rewards)
        if not is_tensor(lens): lens = tensor(lens)
        
        trajectories, actions, rewards, lens = trajectories.to(device), actions.to(device), rewards.to(device), lens.to(device)

        batch, seq_len, _ = trajectories.shape

        state = trajectories[:, :-1]
        next_state = trajectories[:, 1:]
        action = actions[:, :-1]
        reward = rewards[:, :-1]

        if reward.ndim == 3:
            reward = rearrange(reward, 'b t 1 -> b t')

        lens_clamped = lens.clamp(max = seq_len).long()
        lens_clamped = rearrange(lens_clamped, 'b -> b 1')

        seq = torch.arange(seq_len - 1, device = device)
        seq = rearrange(seq, 'n -> 1 n')

        is_valid = seq < (lens_clamped - 1)
        is_done = seq == (lens_clamped - 2)

        state = state[is_valid]
        next_state = next_state[is_valid]
        action = action[is_valid]
        reward = reward[is_valid]
        is_done = is_done[is_valid]

        dataset = TensorDataset(state, next_state, action, reward, is_done)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        dataloader = self.accelerator.prepare(dataloader)

        td_loss_float = 0.
        
        self.q_critic.train()

        if not exists(pbar):
            pbar = tqdm

        pbar_instance = pbar(range(num_train_steps), disable = not self.accelerator.is_main_process)

        for _, (state, next_state, action, reward, is_done) in zip(pbar_instance, cycle(dataloader)):
            q = self.q_critic(state)
            q = q.gather(-1, rearrange(action, 'b -> b 1')).squeeze(-1)

            with torch.no_grad():
                next_q = self.ema_q_critic(next_state).max(dim = -1).values
                target_q = reward + self.td_gamma * next_q * (~is_done).float()

            td_loss = F.mse_loss(q, target_q)

            self.optimizer.zero_grad(set_to_none = True)
            self.accelerator.backward(td_loss)

            if self.max_grad_norm > 0.:
                self.accelerator.clip_grad_norm_(self.q_critic.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.ema_q_critic.update()

            td_loss_float = td_loss.item()
            
            pbar_instance.set_description(f'td loss: {td_loss_float:.3f}')

        return td_loss_float
