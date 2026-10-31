# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "env-ssl-wrapper>=0.6.5",
#   "fire",
#   "gym-pusht",
#   "gymnasium",
#   "hl-gauss-pytorch>=0.1.12",
#   "imageio",
#   "imageio-ffmpeg",
#   "mean-conc-beta>=0.2.1",
#   "numpy",
#   "torch",
#   "torch-einops-utils",
#   "x-mlps-pytorch>=0.6.1",
#   "x-ppo>=0.0.21",
# ]
#
# [tool.uv.sources]
# contrastive-rl-pytorch = { path = "." }
# ///

# Success-Guided Sampling (SGS)
# Octi Zhang et al. — https://arxiv.org/abs/2610.12465

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import shutil
import time

import gymnasium as gym
from gymnasium import spaces
import gym_pusht  # noqa: F401
from gym_pusht.envs.pusht import PushTEnv
import imageio.v2 as imageio
import numpy as np
import pymunk

import torch
from torch import Tensor, arange, cat, nn, stack
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from fire import Fire
from hl_gauss_pytorch import HLGaussLoss
from mean_conc_beta import Beta
from torch_einops_utils import maybe
from torch_einops_utils.shape import default, divisible_by, exists, shape
from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP
from x_ppo import calc_gae, combine_grouped_advantages, hpo_actor_loss

from contrastive_rl_pytorch import (
    ContrastiveLearning,
    ContrastiveWrapper,
    sample_discount,
    sample_truncated_geometric
)
from train_ppo_with_crl_distance_rewards import (
    DistanceRewardPPO,
    VecEnv,
    append_jsonl,
    broadcast_goal,
    save_json
)
from env_ssl_wrapper import (
    SGSSampler,
    SuccessGuidedSamplingWrapper,
    ladder_frontier_index,
    ladder_score_fn
)

# constants

GOAL_POSE = np.array([256., 256., np.pi / 4], dtype = np.float32)
GOAL_TH = float(np.pi / 4)
GOAL_COM = np.array([256. - 45. * np.sin(GOAL_TH), 256. + 45. * np.cos(GOAL_TH)], dtype = np.float32)

DIM_SINGLE_OBS = 6
DIM_ACTION = 2
DIM_STATE = DIM_SINGLE_OBS * 2 + DIM_ACTION  # 14D: [cur_obs (6D), prev_obs (6D), prev_action (2D)]
DIM_GOAL = 4                                 # 4D:  [(gx - 256) / 256, (gy - 256) / 256, sin(gth), cos(gth)]

PUSHT_TARGET_GOAL = np.array([
    (GOAL_POSE[0] - 256.) / 256.,
    (GOAL_POSE[1] - 256.) / 256.,
    np.sin(GOAL_TH),
    np.cos(GOAL_TH)
], dtype = np.float32)

NUM_EASY = 10
NUM_FRONTIER = 6
LADDER_SIZE = 120
NATIVE_BANK_SIZE = 36
SUPER_NATIVE_SIZE = 64
NUM_HELDOUT_TIERS = 24
NUM_HELDOUT_NATIVE = 16

_T_LOCAL_QUADS = (
    np.array([[-60., 0.], [60., 0.], [60., 30.], [-60., 30.]], dtype = np.float64),
    np.array([[-15., 30.], [15., 30.], [15., 120.], [-15., 120.]], dtype = np.float64)
)
_T_GOAL_AREA = 6300.0

# helpers

def wrap_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

# fast analytical polygon intersection & sparse binary step for PushTEnv

def _clip_poly_convex(subject: np.ndarray, clip_quad: np.ndarray) -> np.ndarray:
    out = subject
    for i in range(len(clip_quad)):
        if len(out) == 0:
            return out
        a, b = clip_quad[i], clip_quad[(i + 1) % len(clip_quad)]
        edge = b - a
        inside = (edge[0] * (out[:, 1] - a[1]) - edge[1] * (out[:, 0] - a[0])) >= -1e-9
        pts = []
        for j in range(len(out)):
            cur_pt, prev_pt = out[j], out[j - 1]
            cur_in, prev_in = inside[j], inside[j - 1]
            if cur_in != prev_in:
                d = cur_pt - prev_pt
                denom = edge[0] * d[1] - edge[1] * d[0]
                t = 0.0 if abs(denom) < 1e-12 else (edge[0] * (a[1] - prev_pt[1]) - edge[1] * (a[0] - prev_pt[0])) / denom
                pts.append(prev_pt + np.clip(t, 0.0, 1.0) * d)
            if cur_in:
                pts.append(cur_pt)
        out = np.asarray(pts, dtype = np.float64) if pts else np.empty((0, 2), dtype = np.float64)
    return out

def _poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    return 0.5 * float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))

def fast_t_coverage(block_pos, block_angle: float, goal_pos, goal_angle: float) -> float:
    cb, sb = math.cos(block_angle), math.sin(block_angle)
    cg, sg = math.cos(goal_angle), math.sin(goal_angle)
    rb = np.array([[cb, sb], [-sb, cb]], dtype = np.float64)
    rg = np.array([[cg, sg], [-sg, cg]], dtype = np.float64)
    tb = np.array([float(block_pos[0]), float(block_pos[1])], dtype = np.float64)
    tg = np.array([float(goal_pos[0]), float(goal_pos[1])], dtype = np.float64)

    b_quads = [q @ rb + tb for q in _T_LOCAL_QUADS]
    g_quads = [q @ rg + tg for q in _T_LOCAL_QUADS]

    inter_area = sum(
        _poly_area(poly)
        for bq in b_quads
        for gq in g_quads
        if len(poly := _clip_poly_convex(bq, gq)) >= 3
    )
    return float(min(max(inter_area / _T_GOAL_AREA, 0.0), 1.0))

def patch_pusht_env():
    if getattr(PushTEnv, '_fast_sparse_patched', False):
        return

    def fast_reset(self, seed = None, options = None):
        super(PushTEnv, self).reset(seed = seed)
        if exists(seed):
            self._seed = seed
        self.rng = np.random.default_rng(seed = self._seed)
        self._setup()
        if exists(options) and exists(options.get('reset_to_state')):
            state = np.asarray(options.get('reset_to_state'), dtype = np.float32)
            self.goal_pose = state[5:8].astype(np.float64).copy() if state.shape[0] >= 8 else GOAL_POSE.astype(np.float64).copy()
            self._set_state(state[:5])
        else:
            self.goal_pose = GOAL_POSE.astype(np.float64).copy()
            self._set_state(np.array([
                self.rng.integers(50, 450), self.rng.integers(50, 450),
                self.rng.integers(100, 400), self.rng.integers(100, 400),
                self.rng.uniform(-np.pi, np.pi)
            ], dtype = np.float32))
        self.latest_action = np.array(self.agent.position, dtype = np.float64)
        return self.get_obs(), self._get_info()

    def fast_get_obs(self):
        return np.array(
            tuple(self.agent.position) + tuple(self.block.position) + (self.block.angle % (2 * np.pi),) + tuple(self.goal_pose),
            dtype = np.float64
        )

    def fast_step_sparse(self, action):
        self.n_contact_points = 0
        self.latest_action = action
        for _ in range(int(1.0 / (self.dt * self.control_hz))):
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (pymunk.Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity += acceleration * self.dt
            self.space.step(self.dt)

        coverage = fast_t_coverage(self.block.position, float(self.block.angle), self.goal_pose[:2], float(self.goal_pose[2]))
        terminated = bool(coverage > self.success_threshold)
        info = dict(
            pos_agent = np.array(self.agent.position),
            vel_agent = np.array(self.agent.velocity),
            block_pose = np.array(list(self.block.position) + [self.block.angle]),
            goal_pose = self.goal_pose,
            n_contacts = self.n_contact_points,
            coverage = coverage,
            success = terminated,
            is_success = terminated
        )
        return self.get_obs(), float(terminated), terminated, False, info

    PushTEnv.reset = fast_reset
    PushTEnv.get_obs = fast_get_obs
    PushTEnv.step = fast_step_sparse
    PushTEnv._fast_sparse_patched = True

patch_pusht_env()

# networks

class ResidualBetaActor(Module):
    def __init__(
        self,
        dim_state: int = DIM_STATE,
        dim_goal: int = DIM_GOAL,
        dim_action: int = DIM_ACTION,
        hidden_dim: int = 256,
        depth: int = 12,
        residual_every: int = 2,
        beta: Beta | None = None,
        **beta_kwargs
    ):
        super().__init__()
        self.net = ResidualNormedMLP(
            dim_in = dim_state + dim_goal,
            dim = hidden_dim,
            depth = depth,
            dim_out = dim_action * 2,
            residual_every = residual_every,
            activation = nn.SiLU(),
            final_norm = True
        )
        self.beta = default(beta, lambda: Beta(**beta_kwargs))

    def log_prob(self, distr, action: Tensor) -> Tensor:
        return reduce(distr.log_prob(action), '... a -> ...', 'sum')

    def entropy(self, distr) -> Tensor:
        return reduce(distr.entropy(), '... a -> ...', 'sum')

    def forward(self, state: Tensor, goal: Tensor | None = None):
        if exists(goal):
            state, goal = broadcast_goal(state, goal)
            state = cat((state, goal), dim = -1)
        params = rearrange(self.net(state), '... (a d) -> ... a d', d = 2)
        return self.beta(params)

class ResidualEncoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int = 256,
        depth: int = 16,
        dim_out: int = 64,
        residual_every: int = 4
    ):
        super().__init__()
        self.net = ResidualNormedMLP(
            dim_in = dim_in,
            dim = dim_hidden,
            depth = depth,
            dim_out = dim_out,
            residual_every = residual_every,
            activation = nn.SiLU(),
            final_norm = True
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

# state & goal representation (pure world coordinates)

def goal_pose_from_com(goal_com: np.ndarray, goal_th: float) -> np.ndarray:
    return np.array([
        float(goal_com[0] + 45.0 * np.sin(goal_th)),
        float(goal_com[1] - 45.0 * np.cos(goal_th)),
        float(goal_th)
    ], dtype = np.float32)

def config_to_obs(cfg: np.ndarray) -> np.ndarray:
    c = np.asarray(cfg, dtype = np.float32)
    th = c[..., 4]
    bx = c[..., 2] + 45.0 * np.sin(th)
    by = c[..., 3] + 45.0 * (1.0 - np.cos(th))
    gx, gy, gth = (c[..., 5], c[..., 6], c[..., 7]) if c.shape[-1] >= 8 else (np.full_like(th, 256.0), np.full_like(th, 256.0), np.full_like(th, GOAL_TH))
    return np.stack([c[..., 0], c[..., 1], bx, by, th % (2.0 * np.pi), gx, gy, gth], axis = -1).astype(np.float32)

def state_from_com(agent: np.ndarray, block_com: np.ndarray, th: float, goal_com: np.ndarray | None = None, goal_th: float = GOAL_TH) -> np.ndarray:
    gx, gy, gth = (256.0, 256.0, float(GOAL_TH)) if not exists(goal_com) else tuple(float(x) for x in goal_pose_from_com(goal_com, float(goal_th)))
    return np.array([
        float(agent[0]), float(agent[1]),
        float(block_com[0]), float(block_com[1] - 45.0),
        float(wrap_angle(th)), gx, gy, float(wrap_angle(gth))
    ], dtype = np.float32)

def min_dist_to_t_block(pt: np.ndarray, block_com: np.ndarray, block_th: float) -> float:
    bx = float(block_com[0] + 45.0 * np.sin(block_th))
    by = float(block_com[1] - 45.0 * np.cos(block_th))
    c, s = float(np.cos(block_th)), float(np.sin(block_th))
    dx, dy = float(pt[0] - bx), float(pt[1] - by)
    lx, ly = c * dx + s * dy, -s * dx + c * dy

    def _box_dist(x, y, x0, x1, y0, y1):
        ox, oy = max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1)
        return -min(x - x0, x1 - x, y - y0, y1 - y) if (ox == 0.0 and oy == 0.0) else float(np.hypot(ox, oy))

    return min(_box_dist(lx, ly, -60.0, 60.0, 0.0, 30.0), _box_dist(lx, ly, -15.0, 15.0, 30.0, 120.0))

def set_simulator_goal_from_vec_2d(env, goal_vec: np.ndarray):
    g = rearrange(np.asarray(goal_vec, dtype = np.float64), '... -> (...)')
    gx, gy = float(256.0 + 256.0 * g[0]), float(256.0 + 256.0 * g[1])
    gth = float(math.atan2(float(g[2]), float(g[3])) % (2.0 * math.pi))
    env.unwrapped.goal_pose = np.array([gx, gy, gth], dtype = np.float64)

def transform_pusht(obs: np.ndarray) -> np.ndarray:
    raw = np.asarray(obs, dtype = np.float32)
    pa, pb, th = raw[..., 0:2], raw[..., 2:4], raw[..., 4]
    return np.concatenate([
        (pb - 256.0) / 256.0,
        np.stack([np.sin(th), np.cos(th)], axis = -1),
        (pa - 256.0) / 256.0,
    ], axis = -1).astype(np.float32)

def extract_goal_pusht(obs: np.ndarray) -> np.ndarray:
    raw = np.asarray(obs, dtype = np.float32)
    pg, gth = (raw[..., 5:7], raw[..., 7]) if raw.shape[-1] >= 8 else (np.broadcast_to(GOAL_POSE[:2], raw[..., 2:4].shape), np.broadcast_to(np.float32(GOAL_TH), raw[..., 4].shape))
    return np.concatenate([
        (pg - 256.0) / 256.0,
        np.stack([np.sin(gth), np.cos(gth)], axis = -1),
    ], axis = -1).astype(np.float32)

def make_aug_state_2d(cur_s: np.ndarray, prev_s: np.ndarray, prev_a: np.ndarray) -> np.ndarray:
    return np.concatenate([cur_s, prev_s, prev_a], axis = -1).astype(np.float32)

# relative 2D delta action wrapper

class PushTRelativeActionWrapper(gym.Wrapper):
    def __init__(self, env, delta_scale: float = 55.0):
        super().__init__(env)
        self.delta_scale = float(delta_scale)
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        self._last_raw_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_raw_obs = np.asarray(obs, dtype = np.float32).copy()
        return self._last_raw_obs.copy(), info

    def step(self, action):
        act = np.clip(np.asarray(action, dtype = np.float32), -1.0, 1.0)
        env_act = np.clip(self._last_raw_obs[:2] + act * self.delta_scale, 15.0, 497.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(env_act)
        self._last_raw_obs = np.asarray(obs, dtype = np.float32).copy()
        return self._last_raw_obs.copy(), reward, terminated, truncated, info

def make_pusht(max_steps = 160, seed = 0, render_mode = None, delta_scale = 55.0):
    env = PushTRelativeActionWrapper(gym.make('gym_pusht/PushT-v0', max_episode_steps = max_steps, render_mode = render_mode), delta_scale = delta_scale)
    env.reset(seed = seed)
    return env

# 220-tier Success-Guided Sampling (SGS) curriculum bank

def native_config(rng, idx = 0, step = 0):
    ax, ay = rng.integers(50, 450, size = 2).astype(np.float32)
    bx, by = rng.integers(100, 400, size = 2).astype(np.float32)
    return np.array([ax, ay, bx, by, float(rng.uniform(-np.pi, np.pi)), 256.0, 256.0, GOAL_TH], dtype = np.float32)

def sample_valid_agent_pos(rng, block_com, block_th, push_dir_angle, angle_half_spread, r_min, r_max, min_clearance = 20.0):
    behind_angle = float(push_dir_angle + np.pi)
    for _ in range(80):
        alpha = behind_angle + float(rng.uniform(-angle_half_spread, angle_half_spread))
        r = float(rng.uniform(r_min, r_max))
        cand = np.clip(block_com + r * np.array([np.cos(alpha), np.sin(alpha)], dtype = np.float32), 45.0, 467.0)
        if min_dist_to_t_block(cand, block_com, block_th) >= min_clearance:
            return cand
    return np.clip(block_com + 92.0 * np.array([np.cos(behind_angle), np.sin(behind_angle)], dtype = np.float32), 45.0, 467.0)

def sample_random_agent_pos_outside_block(rng, block_com, block_th, min_clearance = 18.5):
    for _ in range(100):
        if rng.random() < 0.5:
            cand = rng.uniform(45.0, 467.0, size = 2).astype(np.float32)
        else:
            alpha, r = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(52.0, 150.0))
            cand = np.clip(block_com + r * np.array([np.cos(alpha), np.sin(alpha)], dtype = np.float32), 45.0, 467.0)
        if min_dist_to_t_block(cand, block_com, block_th) >= min_clearance:
            return cand
    return sample_valid_agent_pos(rng, block_com, block_th, 0.0, np.pi, 58.0, 110.0, min_clearance = min_clearance)

def sample_4mode_360_config(rng, mode, goal_com, goal_th, d, dth, clear, lat_jitter = 0.0, omni_blend = 0.0):
    m = int(mode) % 4
    lat = float(rng.uniform(-lat_jitter, lat_jitter)) if lat_jitter > 0.0 else 0.0
    if omni_blend < 0.20:
        max_phys_dth = np.deg2rad(1.4 + 0.34 * max(0.0, float(d) - 4.0))
        dth = float(np.clip(dth, -max_phys_dth, max_phys_dth))
        if (m == 2 and dth < 0.0) or (m == 3 and dth > 0.0):
            dth = -dth

    block_th = float(wrap_angle(goal_th + dth))
    c_b, s_b = float(np.cos(block_th)), float(np.sin(block_th))
    loc_to_world = lambda lx, ly: np.array([c_b * lx - s_b * ly, s_b * lx + c_b * ly], dtype = np.float32)

    if m == 0:
        b_dir_loc = np.array([0.25 * lat, -d], dtype = np.float32)
        a_rel_loc = np.array([float(np.clip(-420.0 * dth + lat, -46.0, 46.0)), -(45.0 + clear)], dtype = np.float32)
    elif m == 1:
        b_dir_loc = np.array([0.20 * lat, +d], dtype = np.float32)
        a_rel_loc = np.array([float(np.clip(+420.0 * dth + 0.35 * lat, -14.0, 14.0)), +(75.0 + clear)], dtype = np.float32)
    elif m == 2:
        b_dir_loc = np.array([-d, 0.20 * lat], dtype = np.float32)
        a_rel_loc = np.array([-(15.0 + clear), float(np.clip(0.8 + 420.0 * max(0.0, dth) + 0.25 * abs(lat), 0.8, 32.0))], dtype = np.float32)
    else:
        b_dir_loc = np.array([+d, 0.20 * lat], dtype = np.float32)
        a_rel_loc = np.array([+(15.0 + clear), float(np.clip(0.8 + 420.0 * max(0.0, -dth) + 0.25 * abs(lat), 0.8, 32.0))], dtype = np.float32)

    if omni_blend > 0.0 and rng.random() < omni_blend:
        psi = float(rng.uniform(-np.pi, np.pi))
        block_com = np.clip(goal_com - d * np.array([np.cos(psi), np.sin(psi)], dtype = np.float32), 95.0, 415.0)
        arc = min(np.pi, np.deg2rad(45.0 + 135.0 * omni_blend))
        agent_pos = sample_valid_agent_pos(rng, block_com, block_th, psi, arc, max(54.0, 42.0 + clear), max(78.0, 62.0 + 1.5 * clear), min_clearance = 17.5)
    else:
        block_com = np.clip(goal_com + loc_to_world(b_dir_loc[0], b_dir_loc[1]), 95.0, 415.0)
        agent_pos = np.clip(block_com + loc_to_world(a_rel_loc[0], a_rel_loc[1]), 42.0, 470.0)
        if min_dist_to_t_block(agent_pos, block_com, block_th) < 15.5:
            psi = float(np.arctan2(goal_com[1] - block_com[1], goal_com[0] - block_com[0]))
            agent_pos = sample_valid_agent_pos(rng, block_com, block_th, psi, np.deg2rad(45.0), 54.0, 82.0, min_clearance = 17.5)

    return state_from_com(agent_pos, block_com, block_th, goal_com = goal_com, goal_th = goal_th)

def build_anchor_configs(num_easy = NUM_EASY, num_frontier = NUM_FRONTIER):
    rng = np.random.default_rng(10_203)
    easy = [
        sample_4mode_360_config(rng, i % 4, GOAL_COM.copy(), float(GOAL_TH), 4.0 + 0.6 * i, (1.0 if i % 2 == 0 else -1.0) * np.deg2rad(0.2 * i), 15.8 + 0.10 * i)
        for i in range(num_easy)
    ]
    frontier = [
        sample_4mode_360_config(rng, j % 4, GOAL_COM.copy(), float(GOAL_TH), 24.0 + 6.0 * j, (1.0 if j % 2 == 0 else -1.0) * np.deg2rad(6.0 + 3.5 * j), 20.0 + 3.2 * j, lat_jitter = 4.0 + 2.0 * j, omni_blend = 0.20)
        for j in range(num_frontier)
    ]
    return [*easy, *frontier]

def _4mode_tier(d_range, max_dth, clear_range, lat_jitter, omni_blend, mode_offset = 0, native_prob = 0.):
    def fn(rng, idx = 0, step = 0):
        if native_prob > 0. and rng.random() < native_prob:
            return native_config(rng, idx, step)
        d = float(rng.uniform(*d_range))
        dth = float(rng.uniform(-max_dth, max_dth))
        clear = float(rng.uniform(*clear_range))
        return sample_4mode_360_config(rng, mode_offset + int(idx) + int(step), GOAL_COM.copy(), float(GOAL_TH), d, dth, clear, lat_jitter = lat_jitter, omni_blend = omni_blend)
    return fn

def anchor_foothold_config(anchor_idx: int, num_easy: int = NUM_EASY):
    a_idx = int(anchor_idx) % num_easy
    return _4mode_tier(
        (3.5 + 0.35 * a_idx, 5.8 + 0.38 * a_idx),
        np.deg2rad(0.4 + 0.14 * a_idx),
        (15.8, 16.4 + 0.05 * a_idx),
        1.5, 0.0, mode_offset = a_idx
    )

def _two_stage(w: float) -> tuple[float, float]:
    w = float(np.clip(w, 0., 1.))
    return min(1.0, w / 0.55), max(0.0, (w - 0.55) / 0.45)

def tier_dock_contact_engage_config(w):
    ws, wb = _two_stage(w)
    return _4mode_tier(
        (4.5 + 32.0 * (ws ** 1.15) + 45.0 * wb, 8.5 + 52.0 * (ws ** 1.15) + 75.0 * wb),
        np.deg2rad(0.8 + 10.0 * (ws ** 1.25) + 65.0 * (wb ** 1.35)),
        (15.8 + 2.2 * ws + 6.0 * wb, 16.8 + 4.5 * ws + 12.0 * wb),
        1.5 * ws + 14.0 * wb, 0.15 * (wb ** 1.2)
    )

def curriculum_config(alpha):
    ws, wb = _two_stage(alpha)
    return _4mode_tier(
        (4.5 + 38.0 * (ws ** 1.20) + 65.0 * (wb ** 1.20), 9.0 + 62.0 * (ws ** 1.15) + 115.0 * (wb ** 1.15)),
        np.deg2rad(1.0 + 14.0 * (ws ** 1.25)) + (np.pi - np.deg2rad(15.0)) * (wb ** 1.30),
        (15.8 + 10.0 * (ws ** 1.15) + 28.0 * wb, 17.2 + 24.0 * (ws ** 1.15) + 75.0 * wb),
        2.0 * ws + 35.0 * wb, 0.12 * (ws ** 1.5) + 0.88 * (wb ** 1.1),
        native_prob = 0.45 * (wb ** 1.4) if wb > 0.0 else 0.0
    )

def tier_runway_config(w):
    ws, wb = _two_stage(w)
    return _4mode_tier(
        (4.5 + 42.0 * (ws ** 1.20) + 60.0 * wb, 9.0 + 68.0 * (ws ** 1.15) + 110.0 * wb),
        np.deg2rad(0.8 + 12.0 * (ws ** 1.25)) + (np.pi - np.deg2rad(12.8)) * (wb ** 1.35),
        (15.8 + 8.0 * (ws ** 1.15) + 25.0 * wb, 17.0 + 18.0 * (ws ** 1.15) + 70.0 * wb),
        2.0 * ws + 38.0 * wb, 0.10 * (ws ** 1.5) + 0.90 * (wb ** 1.1)
    )

def tier_dock_circumnav_config(w):
    ws, wb = _two_stage(w)
    def fn(rng, idx = 0, step = 0):
        d = float(rng.uniform(4.5 + 32.0 * (ws ** 1.20) + 55.0 * wb, 9.0 + 54.0 * (ws ** 1.15) + 95.0 * wb))
        max_dth = np.deg2rad(0.8 + 12.0 * (ws ** 1.25)) + (np.pi - np.deg2rad(12.8)) * (wb ** 1.30)
        dth = float(rng.uniform(-max_dth, max_dth))
        if wb <= 0.0:
            clear = float(rng.uniform(15.8 + 10.0 * ws, 17.2 + 22.0 * ws))
            return sample_4mode_360_config(rng, int(idx) + int(step), GOAL_COM.copy(), float(GOAL_TH), d, dth, clear, lat_jitter = 2.5 * ws, omni_blend = 0.15 * (ws ** 1.4))
        psi = float(rng.uniform(-np.pi, np.pi))
        block_com = np.clip(GOAL_COM - d * np.array([np.cos(psi), np.sin(psi)], dtype = np.float32), 95.0, 415.0)
        block_th = float(wrap_angle(GOAL_TH + dth))
        center_ang = (psi + np.pi) if ((int(idx) + int(step)) % 2 == 1) else (psi + np.deg2rad(90.0))
        agent_pos = sample_valid_agent_pos(rng, block_com, block_th, center_ang, np.deg2rad(55.0 + 60.0 * wb), 56.0 + 18.0 * wb, 84.0 + 85.0 * wb, min_clearance = 18.5)
        return state_from_com(agent_pos, block_com, block_th, goal_com = GOAL_COM.copy(), goal_th = float(GOAL_TH))
    return fn

def tier_dock_rotation_config(w):
    ws, wb = _two_stage(w)
    def fn(rng, idx = 0, step = 0):
        d = float(rng.uniform(5.0 + 44.0 * (ws ** 1.15) + 55.0 * wb, 10.0 + 70.0 * (ws ** 1.15) + 95.0 * wb))
        side = 1.0 if ((int(idx) + int(step)) % 2 == 0) else -1.0
        dth = side * float(rng.uniform(np.deg2rad(0.5 + 10.0 * (ws ** 1.25) + 65.0 * (wb ** 1.25)), min(np.pi, np.deg2rad(2.0 + 22.0 * (ws ** 1.20) + 156.0 * (wb ** 1.15)))))
        clear = float(rng.uniform(15.8 + 8.0 * (ws ** 1.15) + 22.0 * wb, 17.2 + 18.0 * (ws ** 1.15) + 65.0 * wb))
        return sample_4mode_360_config(rng, int(idx) + int(step), GOAL_COM.copy(), float(GOAL_TH), d, dth, clear, lat_jitter = 2.0 * ws + 35.0 * wb, omni_blend = 0.10 * (ws ** 1.5) + 0.85 * (wb ** 1.1))
    return fn

def tier_dock_reach_config(w):
    ws, wb = _two_stage(w)
    return _4mode_tier(
        (4.5 + 34.0 * (ws ** 1.20) + 55.0 * wb, 9.0 + 56.0 * (ws ** 1.15) + 95.0 * wb),
        np.deg2rad(0.8 + 12.0 * (ws ** 1.25)) + (np.pi - np.deg2rad(12.8)) * (wb ** 1.35),
        (15.8 + 20.0 * (ws ** 1.20) + 38.0 * wb, 17.4 + 38.0 * (ws ** 1.20) + 90.0 * wb),
        2.0 * ws + 40.0 * wb, 0.10 * (ws ** 1.5) + 0.85 * (wb ** 1.1)
    )

def tier_circumnav_config(w):
    ws, wb = _two_stage(w)
    return _4mode_tier(
        (4.5 + 36.0 * (ws ** 1.20) + 60.0 * wb, 9.0 + 60.0 * (ws ** 1.15) + 105.0 * wb),
        np.deg2rad(0.8 + 13.0 * (ws ** 1.25)) + (np.pi - np.deg2rad(13.8)) * (wb ** 1.30),
        (15.8 + 12.0 * (ws ** 1.15) + 28.0 * wb, 17.2 + 24.0 * (ws ** 1.15) + 75.0 * wb),
        2.5 * ws + 42.0 * wb, 0.18 * (ws ** 1.35) + 0.82 * (wb ** 1.05)
    )

def native_difficulty_score(raw_state: np.ndarray) -> float:
    pa, pb, th = raw_state[0:2], raw_state[2:4], float(raw_state[4])
    gx, gy, gth = (float(raw_state[5]), float(raw_state[6]), float(raw_state[7])) if raw_state.shape[0] >= 8 else (256.0, 256.0, GOAL_TH)
    g_com = np.array([gx - 45.0 * np.sin(gth), gy + 45.0 * np.cos(gth)], dtype = np.float32) if raw_state.shape[0] >= 8 else GOAL_COM
    cb = np.array([pb[0], pb[1] + 45.], dtype = np.float32)
    d_cg, d_ab = g_com - cb, cb - pa
    r_cg, r_ab = float(np.linalg.norm(d_cg)), float(np.linalg.norm(d_ab))
    dth = abs(float(np.arctan2(np.sin(th - gth), np.cos(th - gth))))
    wrong_side = max(0.0, -float(np.dot(d_cg, d_ab) / (r_cg * r_ab + 1e-5)))
    return (r_cg / 160.0) + 1.35 * (dth / np.pi) + 0.65 * wrong_side + 0.45 * (r_ab / 240.0)

def tier_super_adversarial_native_config(w):
    w = float(np.clip(w, 0., 1.))
    num_candidates = int(2 + round(7 * w))
    b_low, b_high = int(round(100 - 20 * w)), int(round(400 + 20 * w))
    a_low, a_high = int(round(50 - 15 * w)), int(round(450 + 20 * w))
    def fn(rng, idx = 0, step = 0):
        gp = goal_pose_from_com(GOAL_COM.copy(), float(GOAL_TH))
        best_s, best_d = None, -1e9
        for _ in range(num_candidates):
            ax, ay = rng.integers(a_low, a_high, size = 2).astype(np.float32)
            bx, by = rng.integers(b_low, b_high, size = 2).astype(np.float32)
            th = float(rng.uniform(-np.pi, np.pi))
            cand = np.array([ax, ay, bx, by, th, gp[0], gp[1], gp[2]], dtype = np.float32)
            d_score = native_difficulty_score(cand)
            if d_score > best_d:
                best_d, best_s = d_score, cand
        return best_s
    return fn

def make_super_polar_tier(w: float, d_range: tuple, dth_min: tuple, r_range: tuple, spread_deg: float, occluded: bool = False, clip_lo: float = 90., clip_hi: float = 420., min_clear: float = 22.0):
    w = float(np.clip(w, 0., 1.))
    def fn(rng, idx = 0, step = 0):
        d = float(rng.uniform(d_range[0] + d_range[1] * w, d_range[2] + d_range[3] * w))
        psi = float(rng.uniform(-np.pi, np.pi))
        t_com = np.clip(GOAL_COM - d * np.array([np.cos(psi), np.sin(psi)], dtype = np.float32), clip_lo, clip_hi)
        side = 1. if (rng.random() < 0.5) else -1.
        lo_rad = min(np.pi - 0.08, np.deg2rad(dth_min[0] + dth_min[1] * w)) if not occluded else np.deg2rad(dth_min[0] + dth_min[1] * w)
        dth = side * float(rng.uniform(lo_rad, np.pi))
        th = float(wrap_angle(GOAL_TH + dth))
        agent = sample_valid_agent_pos(rng, t_com, th, (psi + np.pi) if occluded else psi, np.deg2rad(spread_deg), r_range[0] + r_range[1] * w, r_range[2] + r_range[3] * w, min_clearance = min_clear)
        return state_from_com(agent, t_com, th, goal_com = GOAL_COM.copy(), goal_th = float(GOAL_TH))
    return fn

def build_expanded_configs(num_easy = NUM_EASY, ladder_size = LADDER_SIZE, native_bank_size = NATIVE_BANK_SIZE, super_native_size = SUPER_NATIVE_SIZE):
    foothold_cfgs = [anchor_foothold_config(i, num_easy = num_easy) for i in range(num_easy)]
    sub_ladder_count = ladder_size - len(foothold_cfgs)

    families = (
        tier_dock_contact_engage_config,
        curriculum_config,
        tier_runway_config,
        tier_dock_reach_config,
        tier_dock_circumnav_config,
        tier_dock_rotation_config,
        tier_circumnav_config,
    )
    ladder_configs = [families[i % len(families)](float(((i + 1) / max(sub_ladder_count, 1)) ** 1.35)) for i in range(sub_ladder_count)]
    configs = [*foothold_cfgs, *ladder_configs, *([native_config] * native_bank_size)]

    super_fns = (
        tier_super_adversarial_native_config,
        lambda w: make_super_polar_tier(w, (105., 55., 180., 40.), (95., 45.), (70., 30., 145., 85.), 180.0, False, 90., 420., 22.0),
        lambda w: make_super_polar_tier(w, (95., 60., 175., 40.), (85., 50.), (66., 15., 120., 75.), 65.0, True, 90., 420., 22.0),
        lambda w: make_super_polar_tier(w, (120., 50., 195., 35.), (90., 45.), (120., 50., 195., 65.), 85.0, True, 85., 425., 24.0),
    )
    num_super_levels = super_native_size // len(super_fns)
    for lvl in range(num_super_levels):
        w = float(((lvl + 1) / max(num_super_levels, 1)) ** 1.20)
        for s_fn in super_fns:
            configs.append(s_fn(w))
    return configs

def materialize_config(c, rng, idx = 0, step = 0):
    return c(rng, idx, step) if callable(c) else c

def compute_frontier_k(p: Tensor, counts: Tensor, threshold = 0.25, min_count = 2, kernel = 3, fallback = 4) -> int:
    return ladder_frontier_index(p, counts, threshold = threshold, min_count = min_count, kernel = kernel, fallback = fallback)

def frontier_ladder_score_fn(p, counts, target = 0.40, kappa = 4.0, eps = 1e-4, unvisited_weight = 0.25):
    w = ladder_score_fn(
        p, counts,
        target = target, kappa = kappa, eps = eps, unvisited_weight = unvisited_weight,
        kernel = 3, threshold = 0.25,
        probe_offset = 4., probe_width = 6., envelope_span = 16., envelope_width = 6.
    )
    k_star = compute_frontier_k(p, counts)
    idx = arange(len(w), device = w.device, dtype = w.dtype)
    far_above = torch.clamp(idx - (float(k_star) + 22.0), min = 0.0)
    w = w * torch.exp(-far_above / 16.0)
    if k_star >= 80:
        far_below = torch.clamp((float(k_star) - 40.0) - idx, min = 0.0)
        w = w * torch.exp(-far_below / 28.0).clamp(min = 0.15)
        native_band = (idx >= LADDER_SIZE) & (idx < LADDER_SIZE + NATIVE_BANK_SIZE)
        w = torch.where(native_band, w * 1.65, w)
    return w

def make_sampler(configs, seed = 0, progression_path = None, target = 0.40, kappa = 4.0, eps = 1e-4, window = 10, unvisited_weight = 0.25, uniform_prob = 0.06):
    return SGSSampler(
        configs = configs,
        target = target,
        kappa = kappa,
        eps = eps,
        window = window,
        unvisited_weight = unvisited_weight,
        uniform_prob = uniform_prob,
        score_fn = frontier_ladder_score_fn,
        log_every = None,
        log_path = progression_path,
        seed = seed
    )

# Contrastive RL DistanceRewardPPO with HL-Gauss distributional value function

class PushTDistanceRewardPPO(DistanceRewardPPO):
    def __init__(
        self,
        *args,
        value_num_bins: int = 256,
        env_value_max_return: float = 1.0,
        self_goal_cos_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.self_goal_cos_weight = float(self_goal_cos_weight)
        self.last_self_cos_err = 0.0
        self.last_self_top1_err = 0.0
        self.env_value_hl_gauss = HLGaussLoss(
            min_value = 0.0,
            max_value = float(env_value_max_return),
            num_bins = int(value_num_bins),
            clamp_to_range = True
        ).to(self.device)

    def _sample_critic_pairs(self, states, dones, batch_size, next_states):
        dims = shape(states, 'n t ...').dims
        num_trajs, num_steps = dims['n'], dims['t']
        ends = self._episode_ends(dones, num_steps)
        if shape(ends, 'n ...').dims['n'] == 1 and num_trajs > 1:
            ends = repeat(ends, '1 t -> n t', n = num_trajs)

        times = repeat(arange(num_steps, device = self.device), 't -> n t', n = num_trajs)
        max_steps_all = (ends - times) + 1
        traj_indices, time_indices = torch.nonzero(max_steps_all >= 1, as_tuple = True)

        rand_indices = torch.randint(0, traj_indices.numel(), (batch_size,), device = self.device)
        traj_indices, time_indices = traj_indices[rand_indices], time_indices[rand_indices]

        max_steps = max_steps_all[traj_indices, time_indices]
        discount = rearrange(sample_discount(self.critic_discount, batch_size, device = self.device), 'b 1 -> b')
        delta_times = sample_truncated_geometric(max_steps, discount)
        future_times = time_indices + delta_times - 1

        return states[traj_indices, time_indices], next_states[traj_indices, future_times]

    def update_critic(self, states, dones, next_states):
        self.critic_wrapper.train()
        losses, self_cos_errs, self_top1_errs = [], [], []

        for _ in range(self.critic_steps):
            past_states, future_states = self._sample_critic_pairs(states, dones, self.critic_batch_size, next_states)
            future_goals = self.state_to_goal_fn(future_states)
            extra_negatives = maybe(self.state_to_goal_fn)(self._sample_negative_goals(self.critic_batch_size))

            crl_loss = self.critic_wrapper(past_states, future_goals, extra_negatives = extra_negatives)

            self_goals = self.state_to_goal_fn(past_states)
            z_s = F.normalize(self.critic_wrapper.encode(past_states), dim = -1)
            z_g = F.normalize(self.critic_wrapper.encode_future(self_goals), dim = -1)
            self_cos_err = (1.0 - reduce(z_s * z_g, 'b d -> b', 'sum')).mean()

            with torch.no_grad():
                top1_err = ((z_s @ z_g.T).argmax(dim = -1) != arange(shape(z_s, 'b ...').dims['b'], device = self.device)).float().mean().item()

            self_cos_errs.append(float(self_cos_err.item()))
            self_top1_errs.append(float(top1_err))

            loss = crl_loss + self.self_goal_cos_weight * self_cos_err if self.self_goal_cos_weight > 0.0 else crl_loss

            self.critic_opt.zero_grad()
            loss.backward()
            if exists(self.max_grad_norm):
                nn.utils.clip_grad_norm_(self.critic_wrapper.parameters(), self.max_grad_norm)
            self.critic_opt.step()
            losses.append(crl_loss.item())

        self.last_self_cos_err = sum(self_cos_errs) / len(self_cos_errs)
        self.last_self_top1_err = sum(self_top1_errs) / len(self_top1_errs)

        for state in rearrange(states.detach(), 'n t d -> (n t) d').cpu()[::2]:
            self.negative_buffer.append(state)
        for state in rearrange(next_states.detach(), 'n t d -> (n t) d').cpu()[::4]:
            self.negative_buffer.append(state)

        return sum(losses) / len(losses)

    def _hl_gauss_value_update(self, flat_states, flat_goals, flat_returns):
        batch_size = shape(flat_states, 'b ...').dims['b']
        minibatch_size = min(self.value_minibatch_size, batch_size)

        self.env_value.train()
        losses = []
        for _ in range(self.value_epochs):
            perm = torch.randperm(batch_size, device = self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                inputs = cat((flat_states[idx], flat_goals[idx]), dim = -1)
                loss = self.env_value_hl_gauss(self.env_value(inputs), flat_returns[idx])
                self.env_value_opt.zero_grad()
                loss.backward()
                if exists(self.max_grad_norm):
                    nn.utils.clip_grad_norm_(self.env_value.parameters(), self.max_grad_norm)
                self.env_value_opt.step()
                losses.append(loss.item())
        return sum(losses) / len(losses)

    def _ppo_update(self, states, goals, actions, old_log_probs, advantages):
        flat_states = rearrange(states, 'n t d -> (n t) d')
        flat_goals = rearrange(goals, 'n t g -> (n t) g')
        flat_actions = rearrange(actions, 'n t ... -> (n t) ...')
        flat_old_log_probs = rearrange(old_log_probs, 'n t -> (n t)')
        flat_advantages = rearrange(advantages, 'n t -> (n t)')

        batch_size = shape(flat_states, 'b ...').dims['b']
        minibatch_size = min(self.minibatch_size, batch_size)

        policy_losses, entropies, approx_kls, clip_fracs = [], [], [], []
        self.actor.train()

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(batch_size, device = self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                distr = self.actor(flat_states[idx], flat_goals[idx])
                new_log_probs = self.actor.log_prob(distr, flat_actions[idx])

                policy_loss = hpo_actor_loss(
                    new_log_probs,
                    flat_old_log_probs[idx],
                    flat_advantages[idx],
                    eps_clip = self.clip_eps,
                    normalize_advantages = self.normalize_advantages,
                    **self.ppo_kwargs
                ).mean()

                entropy = self.actor.entropy(distr)
                if exists(entropy) and self.entropy_coef > 0.:
                    policy_loss = policy_loss - entropy.mean() * self.entropy_coef

                self.actor_opt.zero_grad()
                policy_loss.backward()
                if exists(self.max_grad_norm):
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                policy_losses.append(policy_loss.item())
                if exists(entropy):
                    entropies.append(entropy.mean().item())
                log_ratio = new_log_probs - flat_old_log_probs[idx]
                ratio = log_ratio.exp()
                approx_kls.append(((ratio - 1.) - log_ratio).mean().item())
                clip_fracs.append(((ratio - 1.).abs() > self.clip_eps).float().mean().item())

        return dict(
            policy_loss = sum(policy_losses) / len(policy_losses),
            entropy = sum(entropies) / len(entropies) if entropies else 0.,
            approx_kl = sum(approx_kls) / len(approx_kls),
            clip_frac = sum(clip_fracs) / len(clip_fracs)
        )

    def update(self, states, goals, actions, old_log_probs, dones, next_states, terminated, base_rewards, env_reward_weight = None):
        states = self._to_tensor(states, torch.float32)
        next_states = self._to_tensor(next_states, torch.float32)
        actions = self._to_tensor(actions)
        old_log_probs = self._to_tensor(old_log_probs, torch.float32)
        dones = self._to_tensor(dones, torch.bool)
        terminated = self._to_tensor(terminated, torch.bool)
        base_rewards = self._to_tensor(base_rewards, torch.float32).clamp(0., 1.)

        dims = shape(states, 'n t ...').dims
        num_trajs, num_steps = dims['n'], dims['t']
        goals = self._broadcast_goals(goals, num_trajs, num_steps)

        critic_loss = self.update_critic(states, dones, next_states)

        flat_states = rearrange(states, 'n t d -> (n t) d')
        flat_next_states = rearrange(next_states, 'n t d -> (n t) d')
        flat_goals = rearrange(goals, 'n t g -> (n t) g')

        with torch.no_grad():
            next_distances = self.critic_wrapper.distance_pairs(flat_next_states, flat_goals, distance_type = self.distance_type)
            cur_distances = self.critic_wrapper.distance_pairs(flat_states, flat_goals, distance_type = self.distance_type)

            values = rearrange(-cur_distances, '(n t) -> n t', n = num_trajs)
            next_values = rearrange(-next_distances, '(n t) -> n t', n = num_trajs)
            crl_advantages = next_values - values
            distances = rearrange(next_distances, '(n t) -> n t', n = num_trajs)

            env_values = rearrange(self.env_value_hl_gauss(self.env_value(cat((flat_states, flat_goals), dim = -1))), '(n t) -> n t', n = num_trajs)
            env_next_values = rearrange(self.env_value_hl_gauss(self.env_value(cat((flat_next_states, flat_goals), dim = -1))), '(n t) -> n t', n = num_trajs)

        env_gae = calc_gae(
            base_rewards,
            env_values,
            masks = (~terminated).float(),
            done_masks = (~dones).float(),
            gamma = self.discount,
            lam = self.lam,
            next_values = env_next_values,
            return_advantages = True
        )
        env_returns, env_advantages = env_gae.returns, env_gae.advantages

        env_reward_weight = min(max(float(default(env_reward_weight, self.env_reward_weight)), 0.), 1.)
        advantages = combine_grouped_advantages(
            stack((env_advantages, crl_advantages), dim = -1),
            weights = torch.tensor((env_reward_weight, 1. - env_reward_weight), device = self.device),
            normalize = True,
            group_dim = -1,
            mask = None,
            reduce = 'sum'
        )

        env_value_loss = self._hl_gauss_value_update(flat_states, flat_goals, rearrange(env_returns, 'n t -> (n t)'))
        ppo_metrics = self._ppo_update(states, goals, actions, old_log_probs, advantages)
        crl_reward_mean = crl_advantages.mean().item()

        return dict(
            critic_loss = critic_loss,
            self_cos_err = float(self.last_self_cos_err),
            self_top1_err = float(self.last_self_top1_err),
            value_loss = 0.,
            env_value_loss = env_value_loss,
            distance_mean = distances.mean().item(),
            value_mean = values.mean().item(),
            reward_mean = crl_reward_mean,
            crl_reward_mean = crl_reward_mean,
            env_reward_mean = base_rewards.mean().item(),
            env_value_mean = env_values.mean().item(),
            env_reward_weight = env_reward_weight,
            advantage_mean = advantages.mean().item(),
            advantage_std = advantages.std().item(),
            env_advantage_mean = env_advantages.mean().item(),
            crl_advantage_mean = crl_reward_mean,
            **ppo_metrics
        )

# heldout evaluation & rollout grid renderer

@torch.no_grad()
def evaluate_heldout(trainer, bank_envs, eval_states, max_steps = 160, seed_base = 0, deterministic = True):
    B = len(eval_states)
    obs = np.stack([
        e.reset(seed = int(seed_base + i), options = dict(reset_to_state = c))[0]
        for i, (e, c) in enumerate(zip(bank_envs, eval_states))
    ]).astype(np.float32)
    cur_s = transform_pusht(obs)
    prev_s = cur_s.copy()
    prev_a = np.zeros((B, DIM_ACTION), dtype = np.float32)
    done = np.zeros(B, dtype = bool)
    succ = np.zeros(B, dtype = np.float32)
    max_cov = np.zeros(B, dtype = np.float32)
    goal_t = torch.from_numpy(extract_goal_pusht(obs)).float().to(trainer.device)

    with torch.random.fork_rng():
        torch.manual_seed(int(seed_base + 1))
        for _ in range(max_steps):
            if done.all():
                break
            state_t = torch.from_numpy(make_aug_state_2d(cur_s, prev_s, prev_a)).float().to(trainer.device)
            act_t, _ = trainer.act(state_t, goal_t, deterministic = deterministic)
            act = act_t.clamp(-1.0, 1.0).cpu().numpy()

            for i in np.where(~done)[0]:
                i = int(i)
                obs[i], _, term, trunc, info = bank_envs[i].step(act[i])
                prev_s[i], prev_a[i], cur_s[i] = cur_s[i], act[i], transform_pusht(obs[i])
                max_cov[i] = max(max_cov[i], float(info.get('coverage', 0.)))
                if term or trunc:
                    succ[i] = float(term)
                    done[i] = True

    return dict(success_rate = float(succ.mean()), max_cov = float(max_cov.mean()), per_state = [float(x) for x in succ])

@torch.no_grad()
def render_heldout_grid(trainer, eval_states, out_path, max_steps = 160, deterministic = True, grid_cols = 4, fps = 12, delta_scale = 55.0):
    n = min(len(eval_states), 16)
    sel = [eval_states[i] for i in np.linspace(0, len(eval_states) - 1, n, dtype = int)]
    render_envs = [make_pusht(max_steps = max_steps, seed = 80_000 + i, render_mode = 'rgb_array', delta_scale = delta_scale) for i in range(n)]
    obs = np.stack([
        e.reset(seed = 80_000 + i, options = dict(reset_to_state = c))[0]
        for i, (e, c) in enumerate(zip(render_envs, sel))
    ]).astype(np.float32)
    cur_s, prev_s = transform_pusht(obs), transform_pusht(obs)
    prev_a = np.zeros((n, DIM_ACTION), dtype = np.float32)
    done = np.zeros(n, dtype = bool)
    succ = np.zeros(n, dtype = bool)
    frozen_rgb = [None] * n
    goal_t = torch.from_numpy(extract_goal_pusht(obs)).float().to(trainer.device)
    frames = []

    def snap_panel(idx, green = False):
        raw = rearrange(torch.from_numpy(render_envs[idx].render()).float(), 'h w c -> 1 c h w')
        img = rearrange(F.interpolate(raw, size = (192, 192), mode = 'area'), '1 c h w -> h w c').byte().numpy().copy()
        if green:
            img[:5, :] = img[-5:, :] = img[:, :5] = img[:, -5:] = [40, 220, 80]
        return img

    with torch.random.fork_rng():
        torch.manual_seed(0)
        for step_i in range(max_steps):
            if step_i % 3 == 0 or done.all():
                rgb = np.stack([frozen_rgb[i] if exists(frozen_rgb[i]) else snap_panel(i, green = bool(succ[i])) for i in range(n)])
                frames.append(rearrange(rgb, '(r c) h w ch -> (r h) (c w) ch', c = grid_cols))
            if done.all():
                break
            state_t = torch.from_numpy(make_aug_state_2d(cur_s, prev_s, prev_a)).float().to(trainer.device)
            act_t, _ = trainer.act(state_t, goal_t, deterministic = deterministic)
            act = act_t.clamp(-1.0, 1.0).cpu().numpy()
            for i in np.where(~done)[0]:
                i = int(i)
                obs[i], _, term, trunc, _ = render_envs[i].step(act[i])
                prev_s[i], prev_a[i], cur_s[i] = cur_s[i], act[i], transform_pusht(obs[i])
                if term or trunc:
                    succ[i], done[i] = bool(term), True
                    frozen_rgb[i] = snap_panel(i, green = bool(term))

    for e in render_envs:
        e.close()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    with imageio.get_writer(str(out_path), fps = fps, codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1) as writer:
        for f in frames:
            writer.append_data(f[..., :3])
    imageio.mimsave(str(out_path.with_suffix('.gif')), [f[..., :3] for f in frames], fps = fps, loop = 0)
    return str(out_path), float(succ.mean())

# main training loop

def main(
    num_envs: int = 16,
    rollout_len: int = 128,
    num_updates: int = 10_000,
    max_steps: int = 160,
    delta_scale: float = 55.0,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    value_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 0.5,
    discount: float = 0.985,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 4e-4,
    ppo_epochs: int = 6,
    minibatch_size: int = 256,
    actor_hidden: int = 256,
    actor_depth: int = 12,
    actor_residual_every: int = 2,
    critic_hidden: int = 256,
    critic_depth: int = 16,
    critic_embed_dim: int = 64,
    critic_residual_every: int = 4,
    critic_steps: int = 16,
    critic_batch_size: int = 256,
    critic_discount: float = 0.96,
    critic_negative_buffer_size: int = 8192,
    critic_distance_scale: float = 10.0,
    value_hidden: int = 256,
    value_depth: int = 12,
    value_residual_every: int = 2,
    value_num_bins: int = 256,
    value_epochs: int = 6,
    value_minibatch_size: int = 256,
    crl_log_value: bool = True,
    env_reward_weight: float = 0.5,
    success_bonus: float = 1.0,
    self_goal_cos_weight: float = 1.0,
    num_easy: int = NUM_EASY,
    num_frontier: int = NUM_FRONTIER,
    ladder_size: int = LADDER_SIZE,
    native_bank_size: int = NATIVE_BANK_SIZE,
    super_native_size: int = SUPER_NATIVE_SIZE,
    num_heldout_tiers: int = NUM_HELDOUT_TIERS,
    num_heldout_native: int = NUM_HELDOUT_NATIVE,
    sgs_target: float = 0.40,
    sgs_kappa: float = 4.0,
    sgs_eps: float = 1e-4,
    sgs_window: int = 10,
    sgs_unvisited_weight: float = 0.25,
    sgs_uniform_prob: float = 0.06,
    exploration_goal_prob: float = 0.20,
    explore_random_agent_prob: float = 0.50,
    goal_buffer_size: int = 8192,
    goal_buffer_batch: int = 256,
    recent_succ_window: int = 200,
    eval_every: int = 10,
    log_every: int = 5,
    video_every: int = 20,
    video_fps: int = 12,
    video_grid_cols: int = 4,
    checkpoint_folder: str = './checkpoints-pusht2d-sgs-crl',
    video_folder: str = './videos-pusht2d-sgs-crl',
    artifact_folder: str | None = None,
    log_path: str = './logs-pusht2d-sgs-crl.jsonl',
    progression_path: str = './sgs-progression-pusht2d-crl.csv',
    start_update: int = 1,
    seed: int = 0,
    device: str = 'cpu'
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    for d in (checkpoint_folder, video_folder, Path(log_path).parent):
        Path(d).mkdir(parents = True, exist_ok = True)
    if exists(artifact_folder):
        Path(artifact_folder).mkdir(parents = True, exist_ok = True)
    print(f"[PushT-2D SGS+CRL] videos -> {Path(video_folder).resolve()} | checkpoints -> {Path(checkpoint_folder).resolve()} | logs -> {Path(log_path).resolve()}", flush = True)

    dim_state, dim_goal, dim_action = DIM_STATE, DIM_GOAL, DIM_ACTION

    anchor_configs = build_anchor_configs(num_easy = num_easy, num_frontier = num_frontier)
    configs = build_expanded_configs(num_easy = num_easy, ladder_size = ladder_size, native_bank_size = native_bank_size, super_native_size = super_native_size)
    sampler = make_sampler(configs, seed, progression_path, sgs_target, sgs_kappa, sgs_eps, sgs_window, sgs_unvisited_weight, sgs_uniform_prob)

    def make_train_env(idx):
        def _thunk():
            base = make_pusht(max_steps = max_steps, seed = seed + idx, delta_scale = delta_scale)
            return SuccessGuidedSamplingWrapper(
                base, sampler,
                success_fn = lambda _o, _r, term, _trunc, info: bool(term or info.get('success', False))
            )
        return _thunk

    envs = VecEnv([make_train_env(i) for i in range(num_envs)])

    eval_rng = np.random.default_rng(77_777)
    heldout_anchor = list(anchor_configs)
    heldout_tiers = [materialize_config(configs[int(idx)], eval_rng, i) for i, idx in enumerate(np.linspace(0, ladder_size - 1, num_heldout_tiers, dtype = int))]
    heldout_native = [native_config(np.random.default_rng(90_000 + i), i) for i in range(num_heldout_native)]

    anchor_envs = [make_pusht(max_steps = max_steps, seed = 10_000 + i, delta_scale = delta_scale) for i in range(len(heldout_anchor))]
    tier_envs = [make_pusht(max_steps = max_steps, seed = 30_000 + i, delta_scale = delta_scale) for i in range(len(heldout_tiers))]
    native_envs = [make_pusht(max_steps = max_steps, seed = 50_000 + i, delta_scale = delta_scale) for i in range(len(heldout_native))]

    obs_raw, _ = envs.reset(seed = seed)
    cur_single_obs = transform_pusht(obs_raw)
    prev_single_obs = cur_single_obs.copy()
    prev_act = np.zeros((num_envs, dim_action), dtype = np.float32)

    goal_rng = np.random.default_rng(seed + 135_791)
    goal_buffer = deque(maxlen = goal_buffer_size)
    for g_i in range(goal_buffer_batch):
        g_raw = materialize_config(configs[g_i % len(configs)], goal_rng, g_i)
        goal_buffer.append(transform_pusht(config_to_obs(g_raw))[:dim_goal])

    def randomize_env_agent_outside_block(env_i, raw_obs_i: np.ndarray) -> np.ndarray:
        th_cur = float(raw_obs_i[4])
        b_com_cur = np.array([float(raw_obs_i[2] - 45.0 * np.sin(th_cur)), float(raw_obs_i[3] + 45.0 * np.cos(th_cur))], dtype = np.float32)
        pa_start = sample_random_agent_pos_outside_block(goal_rng, b_com_cur, th_cur, min_clearance = 18.5)
        u_env = env_i.unwrapped
        u_env.agent.position = [float(pa_start[0]), float(pa_start[1])]
        u_env.agent.velocity = (0.0, 0.0)
        u_env.block.velocity = (0.0, 0.0)
        u_env.block.angular_velocity = 0.0
        u_env.space.step(u_env.dt)
        new_raw = u_env.get_obs().astype(np.float32)
        cur = env_i
        while hasattr(cur, 'env'):
            if isinstance(cur, PushTRelativeActionWrapper):
                cur._last_raw_obs = new_raw.copy()
                break
            cur = cur.env
        return new_raw

    def sample_goal(raw_obs_i: np.ndarray):
        if len(goal_buffer) > 0 and np.random.rand() < exploration_goal_prob:
            return goal_buffer[np.random.randint(len(goal_buffer))].copy(), True
        return extract_goal_pusht(raw_obs_i), False

    current_goals = np.zeros((num_envs, dim_goal), dtype = np.float32)
    is_explore_goal = np.zeros(num_envs, dtype = bool)

    def assign_env_goal(i: int, raw_obs_i: np.ndarray) -> np.ndarray:
        current_goals[i], is_explore_goal[i] = sample_goal(raw_obs_i)
        if is_explore_goal[i]:
            envs.envs[i].current_config_ids[0] = -1
            if np.random.rand() < explore_random_agent_prob:
                raw_obs_i = randomize_env_agent_outside_block(envs.envs[i], raw_obs_i)
                cur_single_obs[i] = prev_single_obs[i] = transform_pusht(raw_obs_i)
            set_simulator_goal_from_vec_2d(envs.envs[i], current_goals[i])
        return raw_obs_i

    for i in range(num_envs):
        obs_raw[i] = assign_env_goal(i, obs_raw[i])
    obs = make_aug_state_2d(cur_single_obs, prev_single_obs, prev_act)

    actor = ResidualBetaActor(dim_state, dim_goal, dim_action, actor_hidden, actor_depth, residual_every = actor_residual_every).to(device)
    state_encoder = ResidualEncoder(dim_state, critic_hidden, critic_depth, critic_embed_dim, residual_every = critic_residual_every).to(device)
    goal_encoder = ResidualEncoder(dim_goal, critic_hidden, critic_depth, critic_embed_dim, residual_every = critic_residual_every).to(device)
    value = ResidualEncoder(dim_state + dim_goal, value_hidden, value_depth, value_num_bins, residual_every = value_residual_every).to(device)
    env_value = ResidualEncoder(dim_state + dim_goal, value_hidden, value_depth, value_num_bins, residual_every = value_residual_every).to(device)

    contrastive_learn = ContrastiveLearning(use_euclidean = True, l2norm_embed = True, learned_temp = False, scale = critic_distance_scale, symmetric = True)
    critic_wrapper = ContrastiveWrapper(state_encoder, contrastive_learn, future_encoder = goal_encoder).to(device)

    trainer = PushTDistanceRewardPPO(
        actor, critic_wrapper, value = value,
        env_value = env_value,
        separate_env_rewards = True,
        env_reward_weight = env_reward_weight,
        action_type = 'continuous',
        actor_lr = actor_lr, critic_lr = critic_lr, value_lr = value_lr,
        value_epochs = value_epochs, value_minibatch_size = value_minibatch_size,
        value_num_bins = value_num_bins,
        env_value_max_return = float(max(success_bonus, 1.0)),
        weight_decay = weight_decay, max_grad_norm = max_grad_norm,
        discount = discount, lam = lam,
        clip_eps = clip_eps, entropy_coef = entropy_coef,
        ppo_epochs = ppo_epochs, minibatch_size = minibatch_size,
        critic_steps = critic_steps, critic_batch_size = critic_batch_size,
        critic_discount = critic_discount, critic_negative_buffer_size = critic_negative_buffer_size,
        self_goal_cos_weight = self_goal_cos_weight,
        state_to_goal_fn = lambda t: t[..., :dim_goal],
        device = device
    )

    def save_bundle(tag: str, upd: int, k_st: int, b_score: float):
        torch.save(actor.state_dict(), f'{checkpoint_folder}/actor-{tag}.pt')
        torch.save(critic_wrapper.state_dict(), f'{checkpoint_folder}/critic-{tag}.pt')
        torch.save(value.state_dict(), f'{checkpoint_folder}/value-{tag}.pt')
        torch.save(env_value.state_dict(), f'{checkpoint_folder}/env-value-{tag}.pt')
        save_json(f'{checkpoint_folder}/ckpt_meta-{tag}.json', dict(
            update = int(upd), env_steps = int(upd * num_envs * rollout_len),
            actor_depth = int(actor_depth), critic_depth = int(critic_depth), value_depth = int(value_depth),
            k_star = int(k_st), best_score = float(b_score)
        ))

    recent_succ = deque(maxlen = recent_succ_window)
    num_episodes, best_score, best_native_greedy = 0, float('-inf'), 0.0
    t0 = time.time()

    for update in range(start_update, num_updates + 1):
        states = np.zeros((num_envs, rollout_len, dim_state), dtype = np.float32)
        next_states = np.zeros_like(states)
        goals = np.zeros((num_envs, rollout_len, dim_goal), dtype = np.float32)
        actions = np.zeros((num_envs, rollout_len, dim_action), dtype = np.float32)
        log_probs = np.zeros((num_envs, rollout_len), dtype = np.float32)
        env_rewards = np.zeros((num_envs, rollout_len), dtype = np.float32)
        terminateds = np.zeros((num_envs, rollout_len), dtype = np.bool_)
        truncateds = np.zeros_like(terminateds)

        for step in range(rollout_len):
            with torch.no_grad():
                action, log_prob = trainer.act(
                    torch.from_numpy(obs).float().to(device),
                    torch.from_numpy(current_goals).float().to(device)
                )
            action_np = action.cpu().numpy()
            next_obs_raw, step_rewards, terminated, truncated, infos = envs.step(action_np)

            next_single_obs = transform_pusht(next_obs_raw)
            final_single_obs = np.stack([
                transform_pusht(info['final_obs']) if exists(info['final_obs']) else next_single_obs[i]
                for i, info in enumerate(infos)
            ])

            states[:, step] = obs
            next_states[:, step] = make_aug_state_2d(final_single_obs, cur_single_obs, action_np)
            goals[:, step] = current_goals
            actions[:, step] = action_np
            log_probs[:, step] = log_prob.cpu().numpy()
            env_rewards[:, step] = step_rewards
            terminateds[:, step], truncateds[:, step] = terminated, truncated

            for i in range(num_envs):
                if terminated[i] or truncated[i]:
                    num_episodes += 1
                    if not is_explore_goal[i]:
                        recent_succ.append(float(terminated[i]))
                    next_obs_raw[i] = assign_env_goal(i, next_obs_raw[i])
                    cur_single_obs[i] = prev_single_obs[i] = transform_pusht(next_obs_raw[i])
                    prev_act[i] = 0.0
                else:
                    prev_single_obs[i], prev_act[i], cur_single_obs[i] = cur_single_obs[i], action_np[i], next_single_obs[i]

            obs = make_aug_state_2d(cur_single_obs, prev_single_obs, prev_act)

        flat_next_goals = rearrange(next_states, 'n t d -> (n t) d')[:, :dim_goal]
        for g_vec in flat_next_goals[np.random.choice(len(flat_next_goals), size = min(goal_buffer_batch, len(flat_next_goals)), replace = False)]:
            goal_buffer.append(g_vec.copy())

        metrics = trainer.update(
            states = states, goals = goals, actions = actions, old_log_probs = log_probs,
            dones = terminateds | truncateds, next_states = next_states, terminated = terminateds,
            base_rewards = env_rewards, env_reward_weight = env_reward_weight
        )

        k_star = compute_frontier_k(sampler.success_rates, sampler.counts)
        train_succ = float(np.mean(recent_succ)) if recent_succ else 0.
        elapsed = time.time() - t0

        row = dict(
            update = update, env_steps = update * num_envs * rollout_len, episodes = num_episodes,
            train_succ = train_succ, k_star = int(k_star),
            actor_depth = actor_depth, critic_depth = critic_depth,
            elapsed_sec = elapsed, **metrics
        )

        if divisible_by(update, eval_every) or update == 1 or update == num_updates:
            anchor_g = evaluate_heldout(trainer, anchor_envs, heldout_anchor, max_steps, 10_000, True)
            anchor_s = evaluate_heldout(trainer, anchor_envs, heldout_anchor, max_steps, 20_000, False)
            tiers_g = evaluate_heldout(trainer, tier_envs, heldout_tiers, max_steps, 30_000, True)
            tiers_s = evaluate_heldout(trainer, tier_envs, heldout_tiers, max_steps, 40_000, False)
            native_g = evaluate_heldout(trainer, native_envs, heldout_native, max_steps, 50_000, True)
            native_s = evaluate_heldout(trainer, native_envs, heldout_native, max_steps, 60_000, False)

            anchor_f_g = float(np.mean(anchor_g['per_state'][num_easy:]))
            row.update(
                heldout_anchor_greedy = anchor_g['success_rate'],
                heldout_anchor_sampled = anchor_s['success_rate'],
                heldout_anchor_frontier_greedy = anchor_f_g,
                heldout_tiers_greedy = tiers_g['success_rate'],
                heldout_tiers_sampled = tiers_s['success_rate'],
                heldout_tiers_cov = max(tiers_g['max_cov'], tiers_s['max_cov']),
                heldout_native_greedy = native_g['success_rate'],
                heldout_native_sampled = native_s['success_rate'],
                heldout_native_cov = max(native_g['max_cov'], native_s['max_cov'])
            )

            print(
                f"upd {update:4d} | steps {row['env_steps']:7d} | k* {k_star:3d}/{len(configs)-1} | train {train_succ:4.1%} | id_err {metrics.get('self_cos_err', 0.0):.4f} | "
                f"anchor(g/s) {anchor_g['success_rate']:4.1%}/{anchor_s['success_rate']:4.1%} | "
                f"tiers(g/s) {tiers_g['success_rate']:4.1%}/{tiers_s['success_rate']:4.1%} (cov {row['heldout_tiers_cov']:.2f}) | "
                f"native(g/s) {native_g['success_rate']:4.1%}/{native_s['success_rate']:4.1%} (cov {row['heldout_native_cov']:.2f}) | "
                f"t {elapsed:5.1f}s",
                flush = True
            )

            cur_score = (
                2.0 * native_g['success_rate'] + 1.0 * native_s['success_rate'] + 0.5 * row['heldout_native_cov']
                + 1.0 * tiers_g['success_rate'] + 0.5 * tiers_s['success_rate']
                + 0.5 * anchor_f_g + 0.25 * anchor_g['success_rate'] + 0.15 * (float(k_star) / max(len(configs) - 1, 1))
            )
            if cur_score > best_score + 1e-4 or native_g['success_rate'] > best_native_greedy:
                best_score = max(best_score, cur_score)
                best_native_greedy = max(best_native_greedy, native_g['success_rate'])
                save_bundle('best', update, k_star, best_score)

            save_bundle('latest', update, k_star, best_score)

            if divisible_by(update, video_every) or update == 1 or update == num_updates:
                use_det = (tiers_g['success_rate'] > 0.) and (tiers_g['success_rate'] >= tiers_s['success_rate'] * 0.8)
                v_tiers, _ = render_heldout_grid(trainer, heldout_tiers[:16], Path(video_folder) / f'pusht2d_heldout_tiers_upd{update:04d}.mp4', max_steps, use_det, video_grid_cols, video_fps, delta_scale)
                v_native, _ = render_heldout_grid(trainer, heldout_native[:16], Path(video_folder) / f'pusht2d_heldout_native_upd{update:04d}.mp4', max_steps, use_det, video_grid_cols, video_fps, delta_scale)
                print(f"  -> saved heldout video trajectories (.mp4 & .gif): {v_tiers} , {v_native}", flush = True)
                if exists(artifact_folder) and Path(artifact_folder).exists():
                    for src, name in ((v_tiers, f'pusht2d_heldout_tiers_upd{update:04d}'), (v_native, f'pusht2d_heldout_native_upd{update:04d}')):
                        shutil.copy(Path(src), Path(artifact_folder) / f'{name}.mp4')
                        shutil.copy(Path(src).with_suffix('.gif'), Path(artifact_folder) / f'{name}.gif')

        elif divisible_by(update, log_every):
            print(
                f"upd {update:4d} | steps {row['env_steps']:7d} | critic {metrics['critic_loss']:6.3f} | "
                f"dist {metrics['distance_mean']:5.2f} | crl {metrics['crl_reward_mean']:+.4f} | "
                f"k* {k_star:3d}/{len(configs)-1} | train {train_succ:4.1%} | t {elapsed:5.1f}s",
                flush = True
            )

        append_jsonl(log_path, row)

    save_bundle('final', update, k_star, best_score)
    envs.close()
    for e in [*anchor_envs, *tier_envs, *native_envs]:
        e.close()

if __name__ == '__main__':
    Fire(main)
