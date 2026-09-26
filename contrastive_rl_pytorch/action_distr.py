from __future__ import annotations

from typing import Callable

from torch import Tensor, cat
from torch.nn import Module
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution

from einops import rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# action adapter
#
# maps raw actor outputs to the action representation the contrastive critic consumes,
# following "distributions as actions" - https://arxiv.org/abs/2506.16608

class ActionAdapter(Module):
    def __init__(
        self,
        dim,                # dimension of the representation fed to the critic
        actor_dim = None,   # dimension of the raw params coming from the actor
        distr_fn: Callable[..., Distribution] | None = None,
        sample_fn: Callable[..., Tensor] | None = None,
        to_critic_fn: Callable[..., Tensor] | None = None,
        entropy_fn: Callable[..., Tensor] | None = None
    ):
        super().__init__()
        self.dim = dim
        self.actor_dim = default(actor_dim, dim)
        self.distr_fn = distr_fn
        self.sample_fn = sample_fn
        self.to_critic_fn = to_critic_fn
        self.entropy_fn = entropy_fn

    def to_dist(self, params):
        assert exists(self.distr_fn), 'distr_fn must be supplied to build a distribution'
        return self.distr_fn(params)

    def sample(self, params, differentiable = False):
        if exists(self.sample_fn):
            return self.sample_fn(params, differentiable)

        dist = self.to_dist(params)

        assert not differentiable or dist.has_rsample, 'distribution must have rsample to be sampled differentiably'

        return dist.rsample() if differentiable else dist.sample()

    def to_critic(self, params):
        return self.to_critic_fn(params) if exists(self.to_critic_fn) else params

    def entropy(self, params):
        return self.entropy_fn(params) if exists(self.entropy_fn) else None

# categorical actions - raw params are logits

class CategoricalActionAdapter(ActionAdapter):
    def __init__(
        self,
        num_actions,
        action_repr = 'softmax_probs',   # 'raw_logits' | 'softmax_probs' | 'sampled'
        temperature = 1.,
        hard = False
    ):
        assert action_repr in ('raw_logits', 'softmax_probs', 'sampled')

        super().__init__(dim = num_actions, actor_dim = num_actions)

        self.action_repr = action_repr
        self.temperature = temperature
        self.hard = hard

    def to_dist(self, logits):
        return Categorical(logits = logits / self.temperature)

    def gumbel_sample(self, logits):
        return F.gumbel_softmax(logits, tau = self.temperature, hard = self.hard)

    def sample(self, logits, differentiable = False):
        return self.gumbel_sample(logits) if differentiable else self.to_dist(logits).sample()

    def to_critic(self, logits):
        if self.action_repr == 'raw_logits':
            return logits

        if self.action_repr == 'softmax_probs':
            return logits.softmax(dim = -1)

        return self.gumbel_sample(logits)

    def entropy(self, logits):
        return self.to_dist(logits).entropy()

# mean-conc-beta actions - raw params are the mean and concentration

class MeanConcBetaActionAdapter(ActionAdapter):
    def __init__(
        self,
        distr,
        dim_action,
        action_repr = 'transformed',   # 'sampled' | 'raw_params' | 'transformed'
        chunk_size = 1,
        param_dim = 2
    ):
        assert action_repr in ('sampled', 'raw_params', 'transformed')
        assert action_repr != 'transformed' or hasattr(distr, 'to_transformed')

        dim = dict(
            sampled = chunk_size * dim_action,
            raw_params = chunk_size * dim_action * param_dim,
            transformed = chunk_size * dim_action * 2   # unit mean and concentration
        )[action_repr]

        super().__init__(dim = dim, actor_dim = chunk_size * dim_action * param_dim)

        self.distr = distr
        self.dim_action = dim_action
        self.action_repr = action_repr
        self.chunk_size = chunk_size
        self.param_dim = param_dim

    def to_dist_params(self, params):
        return rearrange(
            params,
            '... (c a d) -> ... c a d',
            c = self.chunk_size,
            a = self.dim_action,
            d = self.param_dim
        )

    def to_dist(self, params):
        return self.distr(self.to_dist_params(params))

    def to_critic(self, params):
        if self.action_repr == 'raw_params':
            return params

        if self.action_repr == 'sampled':
            return rearrange(self.sample(params, differentiable = True), '... c a -> ... (c a)')

        transformed = self.distr.to_transformed(self.to_dist_params(params))
        return cat([rearrange(t, '... c a -> ... (c a)') for t in transformed], dim = -1)

    def entropy(self, params):
        if not hasattr(self.distr, 'entropy'):
            return None

        return self.distr.entropy(self.to_dist_params(params), sum_action_dim = False)
