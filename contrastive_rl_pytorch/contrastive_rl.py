from __future__ import annotations

from copy import deepcopy

import torch
from torch import cat, arange, tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F

from accelerate import Accelerator

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import einx
from einops import einsum, rearrange, repeat

from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

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
            if embeds1.ndim == 2:
                sim = einsum(embeds1, embeds2, 'b d, b d -> b')
            else:
                sim = (embeds1 * embeds2).sum(dim = -1)

            return sim * self.scale

        # similarity

        sim = einsum(embeds1, embeds2, 'i d, j d -> i j')
        sim = sim * self.scale

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
        bias = -10.
    ):
        super().__init__()
        self.bias = bias

    def forward(
        self,
        embeds1,
        embeds2,
        return_contrastive_score = False
    ):
        if return_contrastive_score:
            if embeds1.ndim == 2:
                sim = einsum(embeds1, embeds2, 'b d, b d -> b')
            else:
                sim = (embeds1 * embeds2).sum(dim = -1)

            return (sim + self.bias).sigmoid()

        # similarity

        sim = einsum(embeds1, embeds2, 'i d, j d -> i j')
        sim = sim + self.bias

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
        discount = 0.99,
        contrastive_learn: Module | None = None,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        cpu = False
    ):
        super().__init__()

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

        self.discount = discount

        optimizer = Adam(contrast_wrapper.parameters(), lr = learning_rate, **adam_kwargs)

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
        goal_state = None   # (n t dg)
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

        pbar = tqdm(range(num_train_steps), disable = not self.accelerator.is_main_process)

        for _ in pbar:

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

            future_times = past_times + torch.empty_like(past_times).geometric_(1. - self.discount).clamp(min = 1)

            # clamping future times by max_traj_len if not variable lengths else prepare variable length

            clamp_traj_len = (max_traj_len - 1) if not traj_var_lens else rearrange(traj_lens - 1, 'b -> b 1')

            # clamp

            future_times.clamp_(max = clamp_traj_len)

            # pick out the past and future observations as positive pairs

            batch_arange = rearrange(batch_arange, '... -> ... 1')

            past_obs = trajs[batch_arange, past_times]
            future_obs = goal[batch_arange, future_times]

            past_obs, future_obs = tuple(rearrange(t, 'b 1 ... -> b ...') for t in (past_obs, future_obs))

            # handle maybe action

            past_action = None

            if exists(data_dict['actions']):
                actions = data_dict['actions']
                actions = repeat(actions, 'b ... -> (b r) ...', r = self.repetition_factor)

                past_action = actions[batch_arange, past_times]
                past_action = rearrange(past_action, 'b 1 ... -> b ...')

            # contrastive learning

            loss = self.contrast_wrapper(past_obs, future_obs, past_action)

            pbar.set_description(f'loss: {loss.item():.3f}')

            # backwards and optimizer step

            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

# training the actor

class ActorTrainer(Module):
    def __init__(
        self,
        actor: Module,
        encoder: Module,
        goal_encoder: Module,
        batch_size = 32,
        learning_rate = 3e-4,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        softmax_actor_output = False,
        contrastive_learn: Module | None = None,
        cpu = False,
    ):
        super().__init__()

        self.accelerator = Accelerator(cpu = cpu, **accelerate_kwargs)

        optimizer = Adam(actor.parameters(), lr = learning_rate, **adam_kwargs)

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
        sample_fn = None
    ):

        device = self.device

        # setup models

        goal_encoder = self.goal_encoder.to(device)
        encoder = deepcopy(self.encoder).to(device)
        encoder.requires_grad_(False)

        # data

        if exists(lens):
            traj_len = trajectories.shape[-2]
            mask = einx.less('t, b -> b t', arange(traj_len, device = self.device), lens)
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

        pbar = tqdm(range(num_train_steps), disable = not self.accelerator.is_main_process)

        for _ in pbar:

            state, = next(iter_dataloader)
            goal, = next(iter_goal_dataloader)

            # forward state and goal

            action = self.actor((state, goal))

            if self.softmax_actor_output:
                action = action.softmax(dim = -1)
            elif exists(sample_fn):
                action = sample_fn(action)

            # encode state

            encoder.eval()
            encoded_state_action = encoder(cat((state, action), dim = -1))

            with torch.no_grad():
                goal_encoder.eval()
                encoded_goal = goal_encoder(goal)

            sim = self.contrastive_learn(
                encoded_state_action,
                encoded_goal,
                return_contrastive_score = True
            )

            # maximize the similarity between the encoded state action trained from contrastive RL and the encoded goal

            loss = -sim.mean()

            self.accelerator.backward(loss)

            pbar.set_description(f'actor loss: {loss.item():.3f}')

            self.optimizer.step()
            self.optimizer.zero_grad()
