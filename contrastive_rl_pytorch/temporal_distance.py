from __future__ import annotations

import torch
from torch.nn import Module
import torch.nn.functional as F

import einx
from einops import rearrange

from torch_einops_utils.device import fallback_from

def exists(v):
    return v is not None

@fallback_from('mps')
def pairwise_distance(x, y):
    return torch.cdist(x, y)

# l_time + l_sep from chronosrl - https://arxiv.org/abs/2609.36238

class TemporalDistanceLearning(Module):
    accepts_temporal_targets = True

    def __init__(
        self,
        time_weight = 1.,
        sep_weight = 1.,
        classify_weight = 1.,
        goal_eps = 0.25,
        eps = 1e-6
    ):
        super().__init__()
        self.time_weight = time_weight
        self.sep_weight = sep_weight
        self.classify_weight = classify_weight
        self.goal_eps = goal_eps
        self.eps = eps

    def distance(self, embeds1, embeds2, all_pairs = False):
        if all_pairs:
            return pairwise_distance(embeds1, embeds2)

        return (embeds1 - embeds2).square().sum(dim = -1).clamp(min = 1e-8).sqrt()

    def forward(
        self,
        embeds1,
        embeds2,
        delta_times = None,
        raw_discount = None,
        goal_states = None,
        return_contrastive_score = False
    ):
        if return_contrastive_score:
            return -self.distance(embeds1, embeds2)

        dist = self.distance(embeds1, embeds2, all_pairs = True)
        batch = dist.shape[0]

        loss = dist.new_zeros(())

        if self.classify_weight > 0.:
            labels = torch.eye(batch, device = dist.device)
            loss = loss + self.classify_weight * F.binary_cross_entropy_with_logits(-dist, labels)

        assert exists(delta_times) and exists(raw_discount)

        delta = rearrange(delta_times, 'b 1 -> b').float()
        discount = rearrange(raw_discount, 'b 1 -> b').float().clamp(min = self.eps, max = 1. - self.eps)
        kappa = -discount.log()

        if self.time_weight > 0.:
            loss = loss + self.time_weight * F.smooth_l1_loss(dist.diagonal(), kappa * delta)

        if self.sep_weight > 0.:
            margin = kappa / (1. - discount)

            valid = ~torch.eye(batch, dtype = torch.bool, device = dist.device)

            if exists(goal_states) and self.goal_eps > 0.:
                goal_states = rearrange(goal_states, 'b ... -> b (...)')
                valid = valid & (pairwise_distance(goal_states, goal_states) > self.goal_eps)

            hinge = F.relu(einx.subtract('b, b c -> b c', margin, dist))
            loss = loss + self.sep_weight * (hinge * valid).sum() / valid.sum().clamp(min = 1)

        return loss
