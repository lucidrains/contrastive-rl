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
from torch_einops_utils import lens_to_mask, masked_mean, safe_cat, z_score
from torch_einops_utils.device import fallback_from
from torch_einops_utils.shape import shape

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

def default_discount_transform(discount):
    if not is_tensor(discount):
        discount = tensor(discount)

    if discount.ndim == 0:
        discount = rearrange(discount, '-> 1 1')
    elif discount.ndim == 1:
        discount = rearrange(discount, 'b -> b 1')

    return cat((
        discount,
        1. - discount,
        -log(1. - discount)
    ), dim = -1)

def l2norm(t):
    return F.normalize(t, dim = -1)

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def arange_from_tensor_dim(t, dim = 0):
    length = t.shape[dim]
    return arange(length, device = t.device)

# distance helper

# temporal distances: euclidean for unnormalized critics, squared chord on the sphere
# geodesic, chord, and log-cosine were dropped after squared chord converged more reliably

TEMPORAL_DISTANCE_TYPES = ('euclidean', 'squared_chord')

@fallback_from('mps')
def pairwise_euclidean_distance(embeds1, embeds2):
    return torch.cdist(embeds1, embeds2)

def calc_distance(
    embeds1,
    embeds2,
    *,
    use_euclidean = False,
    distance_type: str | None = None,
    all_pairs = False
):
    distance_type = default(distance_type, 'euclidean' if use_euclidean else 'squared_chord')

    if distance_type == 'euclidean':
        if all_pairs:
            return pairwise_euclidean_distance(embeds1, embeds2)

        return (embeds1 - embeds2).norm(dim = -1)

    assert distance_type == 'squared_chord', f'unknown distance type {distance_type}'

    embeds1, embeds2 = map(l2norm, (embeds1, embeds2))

    if all_pairs:
        cos = einsum(embeds1, embeds2, 'i d, j d -> i j')
    else:
        cos = einsum(embeds1, embeds2, '... d, ... d -> ...')

    return (2. - 2. * cos).clamp(min = 0.)

# similarity helper

def calc_similarity(
    embeds1,
    embeds2,
    *,
    use_euclidean = False,
    all_pairs = False
):
    if use_euclidean:
        return -calc_distance(embeds1, embeds2, use_euclidean = True, all_pairs = all_pairs)

    if all_pairs:
        return einsum(embeds1, embeds2, 'i d, j d -> i j')

    return einsum(embeds1, embeds2, '... d, ... d -> ...')

# temporal distance regularization

def calc_temporal_distance_loss(
    dist,
    delta_times,
    raw_discount,
    *,
    goal_states = None,
    time_weight = 1.,
    sep_weight = 1.,
    goal_eps = 0.25,
    max_distance = None,
    eps = 1e-6
):
    has_time_loss = time_weight > 0.
    has_sep_loss = sep_weight > 0.
    has_goal_eps = goal_eps > 0.

    if not (has_time_loss or has_sep_loss):
        return dist.new_zeros(())

    loss = dist.new_zeros(())

    delta = rearrange(delta_times, 'b 1 -> b').float()
    discount = rearrange(raw_discount, 'b 1 -> b').float().clamp(min = eps, max = 1. - eps)
    kappa = -discount.log()

    if has_time_loss:
        target = kappa * delta
        if exists(max_distance):
            target = target.clamp(max = max_distance)
        loss = loss + time_weight * F.smooth_l1_loss(dist.diagonal(), target)

    if has_sep_loss:
        margin = kappa / (1. - discount)
        if exists(max_distance):
            margin = margin.clamp(max = max_distance)

        mask = ~torch.eye(dist.shape[0], dtype = torch.bool, device = dist.device)

        if exists(goal_states) and has_goal_eps:
            goal_states = rearrange(goal_states, 'b ... -> b (...)')
            is_distinct_goal = pairwise_euclidean_distance(goal_states, goal_states) > goal_eps
            mask = mask & is_distinct_goal

        hinge = F.relu(rearrange(margin, 'b -> b 1') - dist)
        loss = loss + sep_weight * masked_mean(hinge, mask)

    return loss

def calc_embedding_temporal_loss(
    embeds1,
    embeds2,
    *,
    distance_type,
    l2norm_embed,
    delta_times,
    raw_discount,
    goal_states,
    time_weight,
    sep_weight,
    goal_eps
):
    dist = calc_distance(embeds1, embeds2, distance_type = distance_type, all_pairs = True)

    max_distance = dict(
        euclidean = 2. if l2norm_embed else None,
        squared_chord = 4.
    )[distance_type]

    return calc_temporal_distance_loss(
        dist,
        delta_times,
        raw_discount,
        goal_states = goal_states,
        time_weight = time_weight,
        sep_weight = sep_weight,
        goal_eps = goal_eps,
        max_distance = max_distance
    )

# sample random state

def sample_random_state(
    replay_buffer,
    env,
    exploration_sample_from_buffer_prob = 0.5,
):
    has_episodes = replay_buffer.num_episodes > 0

    if has_episodes and torch.rand(()) < exploration_sample_from_buffer_prob:
        # sample from buffer, masking out the padding past each episode length

        data = replay_buffer.get_all_data(fields = ['state'], meta_fields = ['episode_lens'])
        states, lens = data['state'], data['episode_lens']

        mask = lens_to_mask(lens, max_len = states.shape[-2])
        states = states[mask]

        rand_id = torch.randint(states.shape[0], (1,), device = states.device)
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

# sample discount

def sample_discount(
    discount: float | tuple[float, ...] | list[float] | Tensor = 0.99,
    batch_size: int = 1,
    *,
    device: torch.device | None = None
) -> Tensor:
    if is_tensor(discount):
        d = discount.to(device = device)
        return rearrange(d, 'b -> b 1') if d.ndim == 1 else d

    if isinstance(discount, (int, float)):
        return torch.full((batch_size, 1), float(discount), device = device)

    if len(discount) == 2 and isinstance(discount, tuple):
        min_discount, max_discount = discount
        return torch.empty((batch_size, 1), device = device).uniform_(min_discount, max_discount)

    discount_tensor = tensor(discount, device = device, dtype = torch.float32)
    indices = torch.randint(0, len(discount), (batch_size,), device = device)
    return rearrange(discount_tensor[indices], 'b -> b 1')

# cayley transform -> orthogonal rotation matrix

@fallback_from('mps')
def cayley_rotation(a, eye):
    omega = a - a.T
    return torch.linalg.solve(eye + omega, eye - omega)

# contrastive learning

class ContrastiveLearning(Module):
    def __init__(
        self,
        l2norm_embed = True,
        learned_temp = False,
        use_euclidean = False,
        symmetric = True,
        lse_reg_weight = 0.,
        rotation_dim: int | None = None,
        scale = 10.,
        time_weight = 0.,
        sep_weight = 0.,
        goal_eps = 0.25,
        temporal_distance_type: str | None = None
    ):
        super().__init__()

        has_time_loss = time_weight > 0.
        has_sep_loss = sep_weight > 0.
        has_temporal = has_time_loss or has_sep_loss
        has_lse_reg = lse_reg_weight > 0.
        has_rotation = exists(rotation_dim)

        temporal_distance_type = default(temporal_distance_type, 'euclidean' if use_euclidean else 'squared_chord')

        assert learned_temp or scale > 0.
        assert lse_reg_weight >= 0.
        assert time_weight >= 0. and sep_weight >= 0., 'temporal weights must be non-negative'
        assert goal_eps >= 0., 'goal_eps must be non-negative'
        assert temporal_distance_type in TEMPORAL_DISTANCE_TYPES
        assert not (has_temporal and temporal_distance_type in TEMPORAL_DISTANCE_TYPES[1:] and not l2norm_embed), 'hyperspherical temporal distance requires l2norm_embed'
        assert not has_rotation or rotation_dim >= 1
        assert not (symmetric and has_lse_reg), 'lse_reg_weight requires symmetric = False'
        assert not (has_rotation and (use_euclidean or not l2norm_embed)), 'rotation requires l2norm and cosine similarity'

        self.l2norm_embed = l2norm_embed
        self.use_euclidean = use_euclidean
        self.symmetric = symmetric
        self.has_lse_reg = has_lse_reg
        self.lse_reg_weight = lse_reg_weight
        self.scale_value = scale
        self.time_weight = time_weight
        self.sep_weight = sep_weight
        self.goal_eps = goal_eps
        self.temporal_distance_type = temporal_distance_type
        self.accepts_temporal_targets = has_temporal
        self.has_rotation = has_rotation
        self.rotation_param = Parameter(torch.zeros(rotation_dim, rotation_dim)) if has_rotation else None
        self.register_buffer('eye', torch.eye(rotation_dim) if has_rotation else None, persistent = False)

        self.learned_log_temp = None
        if learned_temp:
            self.learned_log_temp = Parameter(tensor(1.))

    @property
    def scale(self):
        return self.learned_log_temp.exp() if exists(self.learned_log_temp) else self.scale_value

    @property
    def rotation(self):
        return cayley_rotation(self.rotation_param, self.eye)

    def forward(
        self,
        embeds1,
        embeds2,
        return_contrastive_score = False,
        delta_times = None,
        raw_discount = None,
        goal_states = None
    ):
        if self.l2norm_embed:
            embeds1, embeds2 = map(l2norm, (embeds1, embeds2))

        if self.has_rotation and not self.use_euclidean:
            assert embeds2.shape[-1] == self.rotation_param.shape[0], f'rotation dim {self.rotation_param.shape[0]} must match embed dim {embeds2.shape[-1]}'
            embeds2 = einsum(self.rotation, embeds2, 'i j, ... j -> ... i')  # rotate goal

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

        loss = F.cross_entropy(sim, labels)

        if self.symmetric:
            loss = (loss + F.cross_entropy(rearrange(sim, 'i j -> j i'), labels)) * 0.5

        if self.has_lse_reg:
            loss = loss + self.lse_reg_weight * sim.logsumexp(dim = -1).square().mean()

        if self.accepts_temporal_targets:
            assert exists(delta_times) and exists(raw_discount)

            loss = loss + calc_embedding_temporal_loss(
                embeds1,
                embeds2,
                distance_type = self.temporal_distance_type,
                l2norm_embed = self.l2norm_embed,
                delta_times = delta_times,
                raw_discount = raw_discount,
                goal_states = goal_states,
                time_weight = self.time_weight,
                sep_weight = self.sep_weight,
                goal_eps = self.goal_eps
            )

        return loss

class SigmoidContrastiveLearning(Module):
    def __init__(
        self,
        bias = 0.,
        l2norm_embed = False,
        learned_scale = False,
        use_euclidean = False,
        rotation_dim: int | None = None,
        time_weight = 0.,
        sep_weight = 0.,
        goal_eps = 0.25,
        temporal_distance_type: str | None = None
    ):
        super().__init__()

        has_time_loss = time_weight > 0.
        has_sep_loss = sep_weight > 0.
        has_temporal = has_time_loss or has_sep_loss
        has_rotation = exists(rotation_dim)

        temporal_distance_type = default(temporal_distance_type, 'euclidean' if use_euclidean else 'squared_chord')

        assert time_weight >= 0. and sep_weight >= 0., 'temporal weights must be non-negative'
        assert goal_eps >= 0., 'goal_eps must be non-negative'
        assert temporal_distance_type in TEMPORAL_DISTANCE_TYPES
        assert not (has_temporal and temporal_distance_type in TEMPORAL_DISTANCE_TYPES[1:] and not l2norm_embed), 'hyperspherical temporal distance requires l2norm_embed'
        assert not has_rotation or rotation_dim >= 1
        assert not (has_rotation and (use_euclidean or not l2norm_embed)), 'rotation requires l2norm and cosine similarity'

        self.bias = bias
        self.l2norm_embed = l2norm_embed
        self.use_euclidean = use_euclidean
        self.time_weight = time_weight
        self.sep_weight = sep_weight
        self.goal_eps = goal_eps
        self.temporal_distance_type = temporal_distance_type
        self.accepts_temporal_targets = has_temporal
        self.has_rotation = has_rotation
        self.rotation_param = Parameter(torch.zeros(rotation_dim, rotation_dim)) if has_rotation else None
        self.register_buffer('eye', torch.eye(rotation_dim) if has_rotation else None, persistent = False)

        self.learned_log_scale = None
        if learned_scale:
            self.learned_log_scale = Parameter(tensor(1.)) # starts at ~2.7

    @property
    def scale(self):
        return self.learned_log_scale.exp() if exists(self.learned_log_scale) else 1.

    @property
    def rotation(self):
        return cayley_rotation(self.rotation_param, self.eye)

    def forward(
        self,
        embeds1,
        embeds2,
        return_contrastive_score = False,
        delta_times = None,
        raw_discount = None,
        goal_states = None
    ):
        if self.l2norm_embed:
            embeds1, embeds2 = map(l2norm, (embeds1, embeds2))

        if self.has_rotation and not self.use_euclidean:
            assert embeds2.shape[-1] == self.rotation_param.shape[0], f'rotation dim {self.rotation_param.shape[0]} must match embed dim {embeds2.shape[-1]}'
            embeds2 = einsum(self.rotation, embeds2, 'i j, ... j -> ... i')  # rotate goal

        sim = calc_similarity(
            embeds1,
            embeds2,
            use_euclidean = self.use_euclidean,
            all_pairs = not return_contrastive_score
        )

        sim = sim * self.scale + self.bias

        batch, *_ = sim.shape

        if return_contrastive_score:
            return sim

        # labels

        labels = torch.eye(batch, device = sim.device)

        # binary cross entropy

        loss = F.binary_cross_entropy_with_logits(sim, labels)

        if self.accepts_temporal_targets:
            assert exists(delta_times) and exists(raw_discount)

            loss = loss + calc_embedding_temporal_loss(
                embeds1,
                embeds2,
                distance_type = self.temporal_distance_type,
                l2norm_embed = self.l2norm_embed,
                delta_times = delta_times,
                raw_discount = raw_discount,
                goal_states = goal_states,
                time_weight = self.time_weight,
                sep_weight = self.sep_weight,
                goal_eps = self.goal_eps
            )

        return loss

# compatibility alias - temporal regularization lives on the sigmoid critic

class TemporalDistanceLearning(SigmoidContrastiveLearning):
    def __init__(
        self,
        time_weight = 1.,
        sep_weight = 1.,
        goal_eps = 0.25,
        **kwargs
    ):
        kwargs.setdefault('use_euclidean', True)
        kwargs.setdefault('l2norm_embed', False)

        super().__init__(
            time_weight = time_weight,
            sep_weight = sep_weight,
            goal_eps = goal_eps,
            **kwargs
        )

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

    def distance_pairs(self, past, future):
        past_embeds, future_embeds = self.encode(past), self.encode_future(future)

        if self.contrastive_learn.l2norm_embed:
            past_embeds, future_embeds = map(l2norm, (past_embeds, future_embeds))

        if self.contrastive_learn.has_rotation and not self.contrastive_learn.use_euclidean:
            future_embeds = einsum(self.contrastive_learn.rotation, future_embeds, 'i j, ... j -> ... i')

        return calc_distance(past_embeds, future_embeds, use_euclidean = self.contrastive_learn.use_euclidean)

    def forward(
        self,
        past,     # (b d)
        future,   # (b d)
        past_action = None, # (b na)
        discount = None,    # (b 1)
        delta_times = None, # (b 1)
        raw_discount = None, # (b 1)
        extra_negatives = None # (b d)
    ):
        if exists(extra_negatives):
            future = safe_cat((future, extra_negatives), dim = 0)

        if exists(past_action):
            if past_action.ndim > 2:
                past_action = rearrange(past_action, 'b c ... -> b (c ...)')

            past = cat((past, past_action), dim = -1)

        if exists(discount):
            past = cat((past, discount), dim = -1)

        encoded_past = self.encode(past)
        encoded_future = self.encode_future(future)

        if is_distributed():
            encoded_past, _ = self.all_gather(encoded_past)
            encoded_future, _ = self.all_gather(encoded_future)

        has_temporal_targets = getattr(self.contrastive_learn, 'accepts_temporal_targets', False)

        if has_temporal_targets:
            loss = self.contrastive_learn(
                encoded_past,
                encoded_future,
                delta_times = delta_times,
                raw_discount = raw_discount,
                goal_states = future
            )
        else:
            loss = self.contrastive_learn(encoded_past, encoded_future)

        return loss

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
        discount: float | tuple[float, ...] | list[float] = 0.99,
        goal_discount: float | tuple[float, ...] | list[float] | None = None,
        discount_condition = False,
        discount_transform: Callable = identity,
        condition_on_discount = None,
        action_chunk_size = 1,   # action sequences as representations - https://arxiv.org/abs/2608.30640
        contrastive_learn: Module | None = None,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        cpu = False,
        state_to_goal_fn: Callable = identity,
        state_to_critic_state_fn: Callable = identity
    ):
        super().__init__()

        assert batch_size >= 1
        assert repetition_factor >= 1
        assert action_chunk_size >= 1

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
        self.goal_discount = default(goal_discount, discount)
        self.discount_condition = default(condition_on_discount, discount_condition)
        self.discount_transform = discount_transform
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
            pbar_instance = pbar(range(num_train_steps))
        else:
            pbar = default(pbar, tqdm)
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
                past_times = torch.rand((batch_size, 1), device = self.device).mul(rearrange(traj_lens - 1, 'b -> b 1')).floor().long()
            else:
                past_times = torch.randint(0, max_traj_len - 1, (batch_size, 1), device = self.device)

            clamp_traj_len = (max_traj_len - 1) if not traj_var_lens else rearrange(traj_lens - 1, 'b -> b 1')

            # future times drawn from geometric distribution truncated to remaining length

            remainder_steps = clamp_traj_len - past_times

            discount = sample_discount(
                self.discount,
                batch_size,
                device = self.device
            )

            goal_discount = sample_discount(
                self.goal_discount,
                batch_size,
                device = self.device
            )

            delta_times = sample_truncated_geometric(remainder_steps, goal_discount)
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

            discount_cond = self.discount_transform(discount) if self.discount_condition else None

            loss = self.contrast_wrapper(
                past_obs,
                future_obs,
                past_action,
                discount = discount_cond,
                delta_times = delta_times,
                raw_discount = discount
            )

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
        discount: float | tuple[float, ...] | list[float] = 0.99,
        discount_condition = False,
        discount_transform: Callable = identity,
        condition_on_discount = None,
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
        target_goal_prob = 0.5,
        action_adapter = None
    ):
        super().__init__()

        assert batch_size >= 1
        assert not exists(num_discrete_actions) or num_discrete_actions >= 1
        assert 0. <= target_goal_prob <= 1.
        assert action_entropy_loss_weight >= 0.

        has_action_entropy_loss = action_entropy_loss_weight > 0.

        self.num_discrete_actions = num_discrete_actions
        self.normalize_q_values = normalize_q_values
        self.target_goal_prob = target_goal_prob
        self.discount = discount
        self.discount_condition = default(condition_on_discount, discount_condition)
        self.discount_transform = discount_transform

        self.state_to_goal_fn = state_to_goal_fn
        self.state_to_actor_state_fn = state_to_actor_state_fn
        self.state_to_critic_state_fn = state_to_critic_state_fn

        self.accelerator = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.max_grad_norm = max_grad_norm
        self.has_action_entropy_loss = has_action_entropy_loss
        self.action_entropy_loss_weight = action_entropy_loss_weight

        optimizer = AdamW(actor.parameters(), lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)
        self.actor = actor

        self.softmax_actor_output = softmax_actor_output

        assert not (exists(action_adapter) and exists(num_discrete_actions)), 'action adapter is used in place of the discrete action branch'
        assert not (exists(action_adapter) and softmax_actor_output), 'action adapter is used in place of softmax_actor_output'

        self.action_adapter = action_adapter

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
        target_goals = None,
        discount = None
    ):
        device = self.device

        has_train_steps = num_train_steps > 0

        if not has_train_steps:
            return 0.

        # action adapter handles the mapping from raw actor outputs to what the critic consumes

        if exists(self.action_adapter):
            sample_fn = default(sample_fn, self.action_adapter.to_critic)
            entropy_fn = default(entropy_fn, self.action_adapter.entropy)

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
            pbar_instance = pbar(range(num_train_steps))
        else:
            pbar = default(pbar, tqdm)
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

            # maybe discount conditioning

            batch_discount = None
            if self.discount_condition:
                raw_discount = default(
                    discount,
                    sample_discount(self.discount, batch, device = device)
                )
                if not is_tensor(raw_discount):
                    raw_discount = torch.full((batch, 1), float(raw_discount), device = device)
                elif raw_discount.ndim == 1:
                    raw_discount = rearrange(raw_discount, 'b -> b 1')

                discount_batch, *_ = raw_discount.shape

                if discount_batch != batch:
                    raw_discount = repeat(raw_discount, '1 ... -> b ...', b = batch)

                batch_discount = self.discount_transform(raw_discount)

            # forward state and goal

            actor_state = self.state_to_actor_state_fn(state)
            actor_goal = self.state_to_goal_fn(goal)

            actor_input = [actor_state, actor_goal]
            if exists(batch_discount):
                actor_input.append(batch_discount)

            action_logits = self.actor(cat(actor_input, dim = -1))

            if exists(self.num_discrete_actions):
                num_actions = self.num_discrete_actions
                action_probs = action_logits.softmax(dim = -1)
                critic_state = self.state_to_critic_state_fn(state)

                # evaluate all discrete action candidates at once

                with torch.no_grad():
                    all_actions = torch.eye(num_actions, device = device)
                    all_actions = repeat(all_actions, 'a na -> b a na', b = batch)
                    repeated_states = repeat(critic_state, 'b d -> b a d', a = num_actions)

                    critic_input = [repeated_states, all_actions]
                    if exists(batch_discount):
                        repeated_discount = repeat(batch_discount, 'b d -> b a d', a = num_actions)
                        critic_input.append(repeated_discount)

                    state_actions = cat(critic_input, dim = -1)
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

                if self.has_action_entropy_loss:
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

                critic_input = [critic_state, action]
                if exists(batch_discount):
                    critic_input.append(batch_discount)

                encoded_state_action = encoder(cat(critic_input, dim = -1))

                with torch.no_grad():
                    encoded_goal = goal_encoder(actor_goal)

                sim = self.contrastive_learn(
                    encoded_state_action,
                    encoded_goal,
                    return_contrastive_score = True
                )

                loss = -sim.mean()

                if self.has_action_entropy_loss and exists(entropy_fn):
                    entropy = entropy_fn(action_logits)

                    if exists(entropy):
                        loss = loss - entropy.mean() * self.action_entropy_loss_weight

            self.accelerator.backward(loss)

            pbar_instance.set_description(f'actor loss: {loss.item():.3f}')

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()
