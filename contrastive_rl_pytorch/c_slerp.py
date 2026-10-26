from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, cat, stack
import torch.nn.functional as F

from einops import einsum, rearrange
from torch_einops_utils import slerp

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

class CSlerpPlan(NamedTuple):
    waypoints: Tensor
    num_waypoints: int
    selected_index: int | None
    subgoal: Tensor | None

# C-Slerp: closed-form adaptive-waypoint planning
# Park et al. - https://arxiv.org/abs/2610.20409

def c_slerp_plan(
    state_embed: Tensor,
    goal_embed: Tensor,
    rotation = None,
    *,
    num_waypoints = None,               # K, None -> infer
    max_num_waypoints = 4,              # K_max
    k_temperature = 0.1,                # eta, 0. -> argmax
    alpha = 1.,                         # entropy weight, 0. -> deterministic
    threshold = 0.4,                    # epsilon
    eps = 1e-7,
    score_fn = None,                    # (state, waypoints) -> phi(s, pi(s, z)) . z
    generator = None
) -> CSlerpPlan:

    """C-Slerp planning - Park et al., https://arxiv.org/abs/2610.20409"""

    assert state_embed.ndim == 1 and goal_embed.shape == state_embed.shape
    assert 0 <= default(num_waypoints, max_num_waypoints) and alpha >= 0.

    device, dtype = state_embed.device, state_embed.dtype
    state, goal = map(l2norm, (state_embed, goal_embed))
    rotation = default(rotation, torch.eye(state.shape[-1], device = device, dtype = dtype))
    num_candidates = int(default(num_waypoints, max_num_waypoints)) + 1

    # K-selection, eq. 12 - 14

    rotation_powers = stack([torch.linalg.matrix_power(rotation, i) for i in range(1, num_candidates + 1)])
    rotated_goals = rotation_powers @ goal
    alignments = rotated_goals @ state

    if exists(num_waypoints):
        k = int(num_waypoints)
    elif k_temperature > 0:
        logits = (alignments / k_temperature).softmax(dim = -1)
        k = torch.multinomial(logits, 1, generator = generator).item()
    else:
        k = alignments.argmax().item()

    # closed-form waypoint means, eq. 11

    t = torch.arange(1, k + 1, device = device, dtype = dtype) / (k + 1)
    means = einsum(rotation_powers[:k], slerp(state, rotated_goals[k], t, eps = eps), 'n j i, n j -> n i')

    if alpha > 0.:
        theta = (state @ rotated_goals[k]).clamp(-1. + eps, 1. - eps).acos()
        std = (alpha / (2. * (theta / (k + 1)).cos().clamp(min = eps))).sqrt()
        means = l2norm(means + std * torch.empty_like(means).normal_(generator = generator))

    waypoints = cat((means, rearrange(goal, 'd -> 1 d')), dim = 0)

    # furthest reachable waypoint, eq. 15

    scores = score_fn(state, waypoints) if exists(score_fn) else None
    positions = torch.arange(k + 1, device = device)
    selected_index = max(torch.where(scores >= threshold, positions, -1).max().item(), 0) if exists(scores) else None

    return CSlerpPlan(waypoints, k, selected_index, waypoints[selected_index] if exists(selected_index) else None)
