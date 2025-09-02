from __future__ import annotations

import torch
from torch.nn import Module
import torch.nn.functional as F

from accelerate import Accelerator

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from einops import einsum, rearrange, repeat

from x_mlps_pytorch import MLP

# ein

# b - batch
# d - feature dimension (observation of embed)
# t - time
# n - num trajectories

# helper functions

def exists(v):
    return v is not None

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
    while True:
        for batch in dl:
            yield batch

# tensor functions

def contrastive_loss(
    embeds1,          # (b d)
    embeds2,          # (b d)
    norm = False,     # not needed as original paper had a very nice negative results section at the end, but we'll allow for it
    temperature = 1.,
    eps = 1e-4
):
    assert embeds1.shape == embeds2.shape

    # maybe norm

    if norm:
        embeds1, embeds2 = map(l2norm, (embeds1, embeds2))

    # similarity

    sim = einsum(embeds1, embeds2, 'i d, j d -> i j')

    if temperature != 1.:
        sim = sim / max(temperature, eps)

    # labels, which is 1 across diagonal

    labels = arange_from_tensor_dim(embeds1, dim = 0)

    # transpose

    sim_transpose = rearrange(sim, 'i j -> j i')

    contrastive_loss = (
        F.cross_entropy(sim, labels) +
        F.cross_entropy(sim_transpose, labels)
    ) * 0.5

    return contrastive_loss

# contrastive wrapper module

class ContrastiveWrapper(Module):
    def __init__(
        self,
        encoder: Module,
        future_encoder: Module | None = None, # in negative section, they claim no benefit of separate encoder, but will allow for it
        contrastive_kwargs: dict = dict()
    ):
        super().__init__()

        self.encode = encoder
        self.encode_future = default(future_encoder, encoder)

        self.contrastive_loss_kwargs = contrastive_kwargs

    def forward(
        self,
        past,     # (b d)
        future,   # (b d)
    ):
        encoded_past = self.encode(past)
        encoded_future = self.encode_future(future)

        loss = contrastive_loss(encoded_past, encoded_future, **self.contrastive_loss_kwargs)
        return loss

# contrastive RL trainer

class ContrastiveRLTrainer(Module):
    def __init__(
        self,
        encoder: Module,
        future_encoder: Module | None = None,
        batch_size = 32,
        repetition_factor = 2,
        learning_rate = 3e-4,
        discount = 0.99,
        adam_kwargs: dict = dict(),
        constrast_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs)

        contrast_wrapper = ContrastiveWrapper(encoder = encoder, future_encoder = future_encoder, **constrast_kwargs)

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
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def forward(
        self,
        trajectories, # (n t d) - assume not variable length for starters
        num_train_steps
    ):
        traj_len = trajectories.shape[1]

        # dataset and dataloader

        dataset = TensorDataset(trajectories)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, drop_last = True)

        # prepare

        dataloader = self.accelerator.prepare(dataloader)

        iter_dataloader = cycle(dataloader)

        # training steps

        for _ in range(num_train_steps):

            trajs, *_ = next(iter_dataloader)
            trajs = repeat(trajs, 'b ... -> (b r) ...', r = self.repetition_factor)

            batch_size = trajs.shape[0]
            batch_arange = arange_from_tensor_dim(trajs, dim = 0)

            past_times = torch.randint(0, traj_len - 1, (batch_size, 1)) # feels like max past time should be dynamically adjusted base on the trajectory length, deal with that later

            future_times = past_times + torch.empty_like(past_times).geometric_(1. - self.discount)
            future_times.clamp_(max = traj_len - 1)

            batch_arange = rearrange(batch_arange, '... -> ... 1')

            past_obs = trajs[batch_arange, past_times]
            future_obs = trajs[batch_arange, future_times]

            past_obs, future_obs = tuple(rearrange(t, 'b 1 d -> b d') for t in (past_obs, future_obs))

            loss = self.contrast_wrapper(past_obs, future_obs)

            self.print(f'loss: {loss.item():.3f}')

            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.print('training complete')