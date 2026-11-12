# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "einops",
#   "env-ssl-wrapper>=0.6.8",
#   "fire",
#   "gym-pusht",
#   "gymnasium",
#   "hl-gauss-pytorch>=0.2.2",
#   "imageio",
#   "imageio-ffmpeg",
#   "mean-conc-beta>=0.2.1",
#   "numpy",
#   "pymunk",
#   "torch",
#   "torch-einops-utils>=0.2.0",
#   "x-mlps-pytorch>=0.6.1",
#   "x-ppo>=0.1.0",
# ]
#
# [tool.uv.sources]
# contrastive-rl-pytorch = { path = "." }
# ///

# pusht-2d with sparse binary task reward, contrastive rl distance rewards,
# multi-critic ppo with finite-time gae, and success-guided sampling (sgs)
# Octi Zhang et al. — https://arxiv.org/abs/2610.12465

from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import gymnasium as gym
from gymnasium import spaces
import gym_pusht  # noqa: F401
from gym_pusht.envs.pusht import PushTEnv
import imageio.v2 as imageio
import numpy as np
import pymunk

import torch
from torch import Tensor, arange
import torch.nn.functional as F

from einops import rearrange
from fire import Fire
from torch_einops_utils.shape import divisible_by, exists

from contrastive_rl_pytorch import ContrastiveLearning, ContrastiveWrapper
from train_ppo_with_crl_distance_rewards import (
    DistanceRewardPPO,
    ResidualBetaActor,
    ResidualEncoder,
    VecEnv,
    append_jsonl,
    save_json
)
from env_ssl_wrapper import (
    SGSSampler,
    SuccessGuidedSamplingWrapper,
    ladder_frontier_index,
    ladder_score_fn
)

# helpers

def wrap_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def polar(r, phi):
    return np.array([r * math.cos(phi), r * math.sin(phi)], dtype = np.float32)

# pusht-2d environment and world-pose representation

GOAL_POSE = np.array([256., 256., math.pi / 4.], dtype = np.float32)
GOAL_TH = math.pi / 4.
GOAL_COM = np.array([256. - 45. * math.sin(GOAL_TH), 256. + 45. * math.cos(GOAL_TH)], dtype = np.float32)

DIM_SINGLE_OBS = 6
DIM_ACTION = 2
DIM_STATE = DIM_SINGLE_OBS * 2 + DIM_ACTION  # [cur_obs(6), prev_obs(6), prev_action(2)]
DIM_GOAL = 4                                 # [(gx - 256) / 256, (gy - 256) / 256, sin(gth), cos(gth)]

T_LOCAL_QUADS = (
    np.array([[-60., 0.], [60., 0.], [60., 30.], [-60., 30.]], dtype = np.float64),
    np.array([[-15., 30.], [15., 30.], [15., 120.], [-15., 120.]], dtype = np.float64)
)
T_GOAL_AREA = 6300.

# analytical convex-polygon intersection and sparse binary step for PushTEnv

def _clip_poly_convex(subject, clip_quad):
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
                t = 0. if abs(denom) < 1e-12 else (edge[0] * (a[1] - prev_pt[1]) - edge[1] * (a[0] - prev_pt[0])) / denom
                pts.append(prev_pt + np.clip(t, 0., 1.) * d)

            if cur_in:
                pts.append(cur_pt)

        out = np.asarray(pts, dtype = np.float64) if pts else np.empty((0, 2), dtype = np.float64)

    return out

def _poly_area(p):
    if len(p) < 3:
        return 0.
    return 0.5 * float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))

def fast_t_coverage(block_pos, block_angle, goal_pos, goal_angle):
    cb, sb = math.cos(block_angle), math.sin(block_angle)
    cg, sg = math.cos(goal_angle), math.sin(goal_angle)
    rb = np.array([[cb, sb], [-sb, cb]], dtype = np.float64)
    rg = np.array([[cg, sg], [-sg, cg]], dtype = np.float64)
    tb = np.array([float(block_pos[0]), float(block_pos[1])], dtype = np.float64)
    tg = np.array([float(goal_pos[0]), float(goal_pos[1])], dtype = np.float64)

    b_quads = [q @ rb + tb for q in T_LOCAL_QUADS]
    g_quads = [q @ rg + tg for q in T_LOCAL_QUADS]

    inter_area = sum(
        _poly_area(poly)
        for bq in b_quads
        for gq in g_quads
        if len(poly := _clip_poly_convex(bq, gq)) >= 3
    )
    return float(min(max(inter_area / T_GOAL_AREA, 0.), 1.))

def patch_pusht_env():
    if getattr(PushTEnv, '_fast_sparse_patched', False):
        return

    def fast_reset(self, seed = None, options = None):
        super(PushTEnv, self).reset(seed = seed)
        if exists(seed):
            self._seed = seed

        self.rng = np.random.default_rng(seed = self._seed)
        self._setup()
        reset_state = (options or {}).get('reset_to_state')

        if exists(reset_state):
            state = np.asarray(reset_state, dtype = np.float32)
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
            (*self.agent.position, *self.block.position, self.block.angle % (2. * np.pi), *self.goal_pose),
            dtype = np.float64
        )

    def fast_step_sparse(self, action):
        self.n_contact_points = 0
        self.latest_action = action

        for _ in range(int(1. / (self.dt * self.control_hz))):
            accel = self.k_p * (action - self.agent.position) + self.k_v * (pymunk.Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity += accel * self.dt
            self.space.step(self.dt)

        coverage = fast_t_coverage(self.block.position, float(self.block.angle), self.goal_pose[:2], float(self.goal_pose[2]))
        terminated = bool(coverage > self.success_threshold)

        info = dict(
            pos_agent = np.array(self.agent.position),
            vel_agent = np.array(self.agent.velocity),
            block_pose = np.array([*self.block.position, self.block.angle]),
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

def goal_pose_from_com(goal_com, goal_th):
    return np.array([
        float(goal_com[0] + 45. * np.sin(goal_th)),
        float(goal_com[1] - 45. * np.cos(goal_th)),
        float(goal_th)
    ], dtype = np.float32)

def config_to_obs(cfg):
    c = np.asarray(cfg, dtype = np.float32)
    th = c[..., 4]
    bx = c[..., 2] + 45. * np.sin(th)
    by = c[..., 3] + 45. * (1. - np.cos(th))

    if c.shape[-1] >= 8:
        gx, gy, gth = c[..., 5], c[..., 6], c[..., 7]
    else:
        gx, gy, gth = np.full_like(th, 256.), np.full_like(th, 256.), np.full_like(th, GOAL_TH)

    return np.stack([c[..., 0], c[..., 1], bx, by, th % (2. * np.pi), gx, gy, gth], axis = -1).astype(np.float32)

def state_from_com(agent, block_com, th, goal_com = None, goal_th = GOAL_TH):
    gx, gy, gth = (256., 256., float(GOAL_TH)) if not exists(goal_com) else map(float, goal_pose_from_com(goal_com, float(goal_th)))
    return np.array([
        float(agent[0]), float(agent[1]),
        float(block_com[0]), float(block_com[1] - 45.),
        float(wrap_angle(th)), gx, gy, float(wrap_angle(gth))
    ], dtype = np.float32)

def min_dist_to_t_block(pt, block_com, block_th):
    bx = float(block_com[0] + 45. * np.sin(block_th))
    by = float(block_com[1] - 45. * np.cos(block_th))
    c, s = float(np.cos(block_th)), float(np.sin(block_th))
    dx, dy = float(pt[0] - bx), float(pt[1] - by)
    lx, ly = c * dx + s * dy, -s * dx + c * dy

    def box_dist(x, y, x0, x1, y0, y1):
        ox, oy = max(x0 - x, 0., x - x1), max(y0 - y, 0., y - y1)
        return -min(x - x0, x1 - x, y - y0, y1 - y) if (ox == 0. and oy == 0.) else float(np.hypot(ox, oy))

    return min(box_dist(lx, ly, -60., 60., 0., 30.), box_dist(lx, ly, -15., 15., 30., 120.))

def set_simulator_goal_from_vec_2d(env, goal_vec):
    g = rearrange(np.asarray(goal_vec, dtype = np.float64), '... -> (...)')
    gx, gy = float(256. + 256. * g[0]), float(256. + 256. * g[1])
    gth = float(math.atan2(float(g[2]), float(g[3])) % (2. * math.pi))
    env.unwrapped.goal_pose = np.array([gx, gy, gth], dtype = np.float64)

def transform_pusht(obs):
    raw = np.asarray(obs, dtype = np.float32)
    pa, pb, th = raw[..., 0:2], raw[..., 2:4], raw[..., 4]
    return np.concatenate([
        (pb - 256.) / 256.,
        np.stack([np.sin(th), np.cos(th)], axis = -1),
        (pa - 256.) / 256.
    ], axis = -1).astype(np.float32)

def extract_goal_pusht(obs):
    raw = np.asarray(obs, dtype = np.float32)
    if raw.shape[-1] >= 8:
        pg, gth = raw[..., 5:7], raw[..., 7]
    else:
        pg = np.broadcast_to(GOAL_POSE[:2], raw[..., 2:4].shape)
        gth = np.broadcast_to(np.float32(GOAL_TH), raw[..., 4].shape)

    return np.concatenate([
        (pg - 256.) / 256.,
        np.stack([np.sin(gth), np.cos(gth)], axis = -1)
    ], axis = -1).astype(np.float32)

def make_aug_state_2d(cur_s, prev_s, prev_a):
    return np.concatenate([cur_s, prev_s, prev_a], axis = -1).astype(np.float32)

class PushTRelativeActionWrapper(gym.Wrapper):
    def __init__(self, env, delta_scale = 55.):
        super().__init__(env)
        self.delta_scale = float(delta_scale)
        self.action_space = spaces.Box(low = -1., high = 1., shape = (2,), dtype = np.float32)
        self._last_raw_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_raw_obs = np.asarray(obs, dtype = np.float32).copy()
        return self._last_raw_obs.copy(), info

    def step(self, action):
        act = np.clip(np.asarray(action, dtype = np.float32), -1., 1.)
        env_act = np.clip(self._last_raw_obs[:2] + act * self.delta_scale, 15., 497.).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(env_act)
        self._last_raw_obs = np.asarray(obs, dtype = np.float32).copy()
        return self._last_raw_obs.copy(), reward, terminated, truncated, info

def make_pusht(max_steps = 160, seed = 0, render_mode = None, delta_scale = 55.):
    raw = gym.make('gym_pusht/PushT-v0', max_episode_steps = max_steps, render_mode = render_mode)
    env = PushTRelativeActionWrapper(raw, delta_scale = delta_scale)
    env.reset(seed = seed)
    return env

# se(2) curriculum and success-guided sampling
#
# each curriculum rung is an 8d reset config or callable `fn(rng, idx, step)`:
#   [agent_x, agent_y, block_x, block_y, block_th, goal_x, goal_y, goal_th]
#
# default 220-rung bank = LADDER_SIZE (120) + NATIVE_BANK_SIZE (36) + SUPER_NATIVE_SIZE (64):
#   stage 1 (0..9):     4-face contact footholds (`bottom_bar`, `stem_tip`, `left_flank`, `right_flank`)
#   stage 2 (10..119):  interleaved `SKILL_TRACKS` (`w` in (0, 1])
#   stage 3 (120..219): 36 `native_config` + 64 `SUPER_NATIVE_TRACKS` (adversarial & occluded polar spawns)
#
# to iterate on the curriculum, edit `SKILL_TRACKS`, `SUPER_NATIVE_TRACKS`, or `curriculum_score_fn`

NUM_EASY = 10
NUM_FRONTIER = 6
LADDER_SIZE = 120
NATIVE_BANK_SIZE = 36
SUPER_NATIVE_SIZE = 64
NUM_HELDOUT_TIERS = 24
NUM_HELDOUT_NATIVE = 16

def native_config(rng, idx = 0, step = 0):
    ax, ay = rng.integers(50, 450, size = 2).astype(np.float32)
    bx, by = rng.integers(100, 400, size = 2).astype(np.float32)
    return np.array([ax, ay, bx, by, float(rng.uniform(-np.pi, np.pi)), 256., 256., GOAL_TH], dtype = np.float32)

def sample_valid_agent_pos(rng, block_com, block_th, push_dir_angle, angle_half_spread, r_min, r_max, min_clearance = 20.):
    behind = float(push_dir_angle + np.pi)
    for _ in range(80):
        alpha = behind + float(rng.uniform(-angle_half_spread, angle_half_spread))
        cand = np.clip(block_com + polar(float(rng.uniform(r_min, r_max)), alpha), 45., 467.)
        if min_dist_to_t_block(cand, block_com, block_th) >= min_clearance:
            return cand
    return np.clip(block_com + polar(92., behind), 45., 467.)

def sample_random_agent_pos_outside_block(rng, block_com, block_th, min_clearance = 18.5):
    for _ in range(100):
        if rng.random() < 0.5:
            cand = rng.uniform(45., 467., size = 2).astype(np.float32)
        else:
            cand = np.clip(block_com + polar(float(rng.uniform(52., 150.)), float(rng.uniform(-np.pi, np.pi))), 45., 467.)
        if min_dist_to_t_block(cand, block_com, block_th) >= min_clearance:
            return cand
    return sample_valid_agent_pos(rng, block_com, block_th, 0., np.pi, 58., 110., min_clearance = min_clearance)

def sample_4mode_360_config(rng, mode, goal_com, goal_th, d, dth, clear, lat_jitter = 0., omni_blend = 0.):
    m = int(mode) % 4
    lat = float(rng.uniform(-lat_jitter, lat_jitter)) if lat_jitter > 0. else 0.

    if omni_blend < 0.20:
        max_phys_dth = np.deg2rad(1.4 + 0.34 * max(0., float(d) - 4.))
        dth = float(np.clip(dth, -max_phys_dth, max_phys_dth))
        if (m == 2 and dth < 0.) or (m == 3 and dth > 0.):
            dth = -dth

    block_th = float(wrap_angle(goal_th + dth))
    c_b, s_b = float(np.cos(block_th)), float(np.sin(block_th))
    loc_to_world = lambda lx, ly: np.array([c_b * lx - s_b * ly, s_b * lx + c_b * ly], dtype = np.float32)

    if m == 0:
        b_dir_loc = (0.25 * lat, -d)
        a_rel_loc = (float(np.clip(-420. * dth + lat, -46., 46.)), -(45. + clear))
    elif m == 1:
        b_dir_loc = (0.20 * lat, +d)
        a_rel_loc = (float(np.clip(+420. * dth + 0.35 * lat, -14., 14.)), +(75. + clear))
    elif m == 2:
        b_dir_loc = (-d, 0.20 * lat)
        a_rel_loc = (-(15. + clear), float(np.clip(0.8 + 420. * max(0., dth) + 0.25 * abs(lat), 0.8, 32.)))
    else:
        b_dir_loc = (+d, 0.20 * lat)
        a_rel_loc = (+(15. + clear), float(np.clip(0.8 + 420. * max(0., -dth) + 0.25 * abs(lat), 0.8, 32.)))

    if omni_blend > 0. and rng.random() < omni_blend:
        psi = float(rng.uniform(-np.pi, np.pi))
        block_com = np.clip(goal_com - polar(d, psi), 95., 415.)
        arc = min(np.pi, np.deg2rad(45. + 135. * omni_blend))
        agent_pos = sample_valid_agent_pos(rng, block_com, block_th, psi, arc, max(54., 42. + clear), max(78., 62. + 1.5 * clear), min_clearance = 17.5)
    else:
        block_com = np.clip(goal_com + loc_to_world(*b_dir_loc), 95., 415.)
        agent_pos = np.clip(block_com + loc_to_world(*a_rel_loc), 42., 470.)
        if min_dist_to_t_block(agent_pos, block_com, block_th) < 15.5:
            psi = float(np.arctan2(goal_com[1] - block_com[1], goal_com[0] - block_com[0]))
            agent_pos = sample_valid_agent_pos(rng, block_com, block_th, psi, np.deg2rad(45.), 54., 82., min_clearance = 17.5)

    return state_from_com(agent_pos, block_com, block_th, goal_com = goal_com, goal_th = goal_th)

def build_anchor_configs(num_easy = NUM_EASY, num_frontier = NUM_FRONTIER):
    rng = np.random.default_rng(10_203)
    easy = [
        sample_4mode_360_config(
            rng, i % 4, GOAL_COM.copy(), GOAL_TH,
            d = 4. + 0.6 * i,
            dth = (1. if divisible_by(i, 2) else -1.) * np.deg2rad(0.2 * i),
            clear = 15.8 + 0.10 * i
        )
        for i in range(num_easy)
    ]
    frontier = [
        sample_4mode_360_config(
            rng, j % 4, GOAL_COM.copy(), GOAL_TH,
            d = 24. + 6. * j,
            dth = (1. if divisible_by(j, 2) else -1.) * np.deg2rad(6. + 3.5 * j),
            clear = 20. + 3.2 * j,
            lat_jitter = 4. + 2. * j,
            omni_blend = 0.20
        )
        for j in range(num_frontier)
    ]
    return [*easy, *frontier]

def four_mode_tier(d_range, max_dth, clear_range, lat_jitter, omni_blend, mode_offset = 0, native_prob = 0.):
    def fn(rng, idx = 0, step = 0):
        if native_prob > 0. and rng.random() < native_prob:
            return native_config(rng, idx, step)

        return sample_4mode_360_config(
            rng,
            mode = mode_offset + int(idx) + int(step),
            goal_com = GOAL_COM.copy(),
            goal_th = GOAL_TH,
            d = float(rng.uniform(*d_range)),
            dth = float(rng.uniform(-max_dth, max_dth)),
            clear = float(rng.uniform(*clear_range)),
            lat_jitter = lat_jitter,
            omni_blend = omni_blend
        )

    return fn

def split_stage(w):
    w = float(np.clip(w, 0., 1.))
    return min(1., w / 0.55), max(0., (w - 0.55) / 0.45)

def anchor_foothold_config(anchor_idx, num_easy = NUM_EASY):
    a_idx = int(anchor_idx) % num_easy
    return four_mode_tier(
        d_range = (3.5 + 0.35 * a_idx, 5.8 + 0.38 * a_idx),
        max_dth = np.deg2rad(0.4 + 0.14 * a_idx),
        clear_range = (15.8, 16.4 + 0.05 * a_idx),
        lat_jitter = 1.5,
        omni_blend = 0.,
        mode_offset = a_idx
    )

# stage 2 skill tracks

def contact_engage_track(w):
    ws, wb = split_stage(w)
    return four_mode_tier(
        d_range = (4.5 + 32. * (ws ** 1.15) + 45. * wb, 8.5 + 52. * (ws ** 1.15) + 75. * wb),
        max_dth = np.deg2rad(0.8 + 10. * (ws ** 1.25) + 65. * (wb ** 1.35)),
        clear_range = (15.8 + 2.2 * ws + 6. * wb, 16.8 + 4.5 * ws + 12. * wb),
        lat_jitter = 1.5 * ws + 14. * wb,
        omni_blend = 0.15 * (wb ** 1.2)
    )

def native_blend_track(w):
    ws, wb = split_stage(w)
    return four_mode_tier(
        d_range = (4.5 + 38. * (ws ** 1.20) + 65. * (wb ** 1.20), 9. + 62. * (ws ** 1.15) + 115. * (wb ** 1.15)),
        max_dth = np.deg2rad(1. + 14. * (ws ** 1.25)) + (np.pi - np.deg2rad(15.)) * (wb ** 1.30),
        clear_range = (15.8 + 10. * (ws ** 1.15) + 28. * wb, 17.2 + 24. * (ws ** 1.15) + 75. * wb),
        lat_jitter = 2. * ws + 35. * wb,
        omni_blend = 0.12 * (ws ** 1.5) + 0.88 * (wb ** 1.1),
        native_prob = 0.45 * (wb ** 1.4) if wb > 0. else 0.
    )

def runway_push_track(w):
    ws, wb = split_stage(w)
    return four_mode_tier(
        d_range = (4.5 + 42. * (ws ** 1.20) + 60. * wb, 9. + 68. * (ws ** 1.15) + 110. * wb),
        max_dth = np.deg2rad(0.8 + 12. * (ws ** 1.25)) + (np.pi - np.deg2rad(12.8)) * (wb ** 1.35),
        clear_range = (15.8 + 8. * (ws ** 1.15) + 25. * wb, 17. + 18. * (ws ** 1.15) + 70. * wb),
        lat_jitter = 2. * ws + 38. * wb,
        omni_blend = 0.10 * (ws ** 1.5) + 0.90 * (wb ** 1.1)
    )

def wide_reach_track(w):
    ws, wb = split_stage(w)
    return four_mode_tier(
        d_range = (4.5 + 34. * (ws ** 1.20) + 55. * wb, 9. + 56. * (ws ** 1.15) + 95. * wb),
        max_dth = np.deg2rad(0.8 + 12. * (ws ** 1.25)) + (np.pi - np.deg2rad(12.8)) * (wb ** 1.35),
        clear_range = (15.8 + 20. * (ws ** 1.20) + 38. * wb, 17.4 + 38. * (ws ** 1.20) + 90. * wb),
        lat_jitter = 2. * ws + 40. * wb,
        omni_blend = 0.10 * (ws ** 1.5) + 0.85 * (wb ** 1.1)
    )

def occluded_circumnav_track(w):
    ws, wb = split_stage(w)

    def fn(rng, idx = 0, step = 0):
        d = float(rng.uniform(4.5 + 32. * (ws ** 1.20) + 55. * wb, 9. + 54. * (ws ** 1.15) + 95. * wb))
        max_dth = np.deg2rad(0.8 + 12. * (ws ** 1.25)) + (np.pi - np.deg2rad(12.8)) * (wb ** 1.30)
        dth = float(rng.uniform(-max_dth, max_dth))

        if wb <= 0.:
            clear = float(rng.uniform(15.8 + 10. * ws, 17.2 + 22. * ws))
            return sample_4mode_360_config(
                rng, int(idx) + int(step), GOAL_COM.copy(), GOAL_TH,
                d, dth, clear, lat_jitter = 2.5 * ws, omni_blend = 0.15 * (ws ** 1.4)
            )

        psi = float(rng.uniform(-np.pi, np.pi))
        block_com = np.clip(GOAL_COM - polar(d, psi), 95., 415.)
        block_th = float(wrap_angle(GOAL_TH + dth))
        center_ang = (psi + np.pi) if not divisible_by(int(idx) + int(step), 2) else (psi + 0.5 * np.pi)
        agent_pos = sample_valid_agent_pos(
            rng, block_com, block_th, center_ang,
            np.deg2rad(55. + 60. * wb), 56. + 18. * wb, 84. + 85. * wb, min_clearance = 18.5
        )
        return state_from_com(agent_pos, block_com, block_th, goal_com = GOAL_COM.copy(), goal_th = GOAL_TH)

    return fn

def large_rotation_track(w):
    ws, wb = split_stage(w)

    def fn(rng, idx = 0, step = 0):
        d = float(rng.uniform(5. + 44. * (ws ** 1.15) + 55. * wb, 10. + 70. * (ws ** 1.15) + 95. * wb))
        side = 1. if divisible_by(int(idx) + int(step), 2) else -1.
        dth_lo = np.deg2rad(0.5 + 10. * (ws ** 1.25) + 65. * (wb ** 1.25))
        dth_hi = min(np.pi, np.deg2rad(2. + 22. * (ws ** 1.20) + 156. * (wb ** 1.15)))
        dth = side * float(rng.uniform(dth_lo, dth_hi))
        clear = float(rng.uniform(15.8 + 8. * (ws ** 1.15) + 22. * wb, 17.2 + 18. * (ws ** 1.15) + 65. * wb))
        return sample_4mode_360_config(
            rng, int(idx) + int(step), GOAL_COM.copy(), GOAL_TH,
            d, dth, clear, lat_jitter = 2. * ws + 35. * wb, omni_blend = 0.10 * (ws ** 1.5) + 0.85 * (wb ** 1.1)
        )

    return fn

def omni_circumnav_track(w):
    ws, wb = split_stage(w)
    return four_mode_tier(
        d_range = (4.5 + 36. * (ws ** 1.20) + 60. * wb, 9. + 60. * (ws ** 1.15) + 105. * wb),
        max_dth = np.deg2rad(0.8 + 13. * (ws ** 1.25)) + (np.pi - np.deg2rad(13.8)) * (wb ** 1.30),
        clear_range = (15.8 + 12. * (ws ** 1.15) + 28. * wb, 17.2 + 24. * (ws ** 1.15) + 75. * wb),
        lat_jitter = 2.5 * ws + 42. * wb,
        omni_blend = 0.18 * (ws ** 1.35) + 0.82 * (wb ** 1.05)
    )

# stage 3 super-native tracks (adversarial best-of-k and occluded polar spawns)

def native_difficulty_score(raw_state):
    pa, pb, th = raw_state[0:2], raw_state[2:4], float(raw_state[4])
    gx, gy, gth = (float(raw_state[5]), float(raw_state[6]), float(raw_state[7])) if raw_state.shape[0] >= 8 else (256., 256., GOAL_TH)
    g_com = np.array([gx - 45. * np.sin(gth), gy + 45. * np.cos(gth)], dtype = np.float32) if raw_state.shape[0] >= 8 else GOAL_COM
    cb = np.array([pb[0], pb[1] + 45.], dtype = np.float32)

    d_cg, d_ab = g_com - cb, cb - pa
    r_cg, r_ab = float(np.linalg.norm(d_cg)), float(np.linalg.norm(d_ab))
    dth = abs(float(wrap_angle(th - gth)))
    wrong_side = max(0., -float(np.dot(d_cg, d_ab) / (r_cg * r_ab + 1e-5)))
    return (r_cg / 160.) + 1.35 * (dth / np.pi) + 0.65 * wrong_side + 0.45 * (r_ab / 240.)

def adversarial_native_track(w):
    w = float(np.clip(w, 0., 1.))
    num_candidates = int(2 + round(7 * w))
    b_low, b_high = int(round(100 - 20 * w)), int(round(400 + 20 * w))
    a_low, a_high = int(round(50 - 15 * w)), int(round(450 + 20 * w))

    def fn(rng, idx = 0, step = 0):
        gp = goal_pose_from_com(GOAL_COM.copy(), GOAL_TH)
        best_s, best_d = None, -1e9
        for _ in range(num_candidates):
            ax, ay = rng.integers(a_low, a_high, size = 2).astype(np.float32)
            bx, by = rng.integers(b_low, b_high, size = 2).astype(np.float32)
            cand = np.array([ax, ay, bx, by, float(rng.uniform(-np.pi, np.pi)), gp[0], gp[1], gp[2]], dtype = np.float32)
            score = native_difficulty_score(cand)
            if score > best_d:
                best_d, best_s = score, cand
        return best_s

    return fn

def super_polar_track(w, d_spec, dth_min, r_spec, spread_deg, occluded = False, clip_bounds = (90., 420.), min_clear = 22.):
    w = float(np.clip(w, 0., 1.))

    def fn(rng, idx = 0, step = 0):
        d = float(rng.uniform(d_spec[0] + d_spec[1] * w, d_spec[2] + d_spec[3] * w))
        psi = float(rng.uniform(-np.pi, np.pi))
        t_com = np.clip(GOAL_COM - polar(d, psi), *clip_bounds)
        side = 1. if rng.random() < 0.5 else -1.
        lo_rad = np.deg2rad(dth_min[0] + dth_min[1] * w)
        if not occluded:
            lo_rad = min(np.pi - 0.08, lo_rad)
        th = float(wrap_angle(GOAL_TH + side * float(rng.uniform(lo_rad, np.pi))))
        agent = sample_valid_agent_pos(
            rng, t_com, th, (psi + np.pi) if occluded else psi,
            np.deg2rad(spread_deg), r_spec[0] + r_spec[1] * w, r_spec[2] + r_spec[3] * w,
            min_clearance = min_clear
        )
        return state_from_com(agent, t_com, th, goal_com = GOAL_COM.copy(), goal_th = GOAL_TH)

    return fn

SKILL_TRACKS = (
    contact_engage_track,
    native_blend_track,
    runway_push_track,
    wide_reach_track,
    occluded_circumnav_track,
    large_rotation_track,
    omni_circumnav_track
)

SUPER_NATIVE_TRACKS = (
    adversarial_native_track,
    lambda w: super_polar_track(w, (105., 55., 180., 40.), (95., 45.), (70., 30., 145., 85.), 180., False),
    lambda w: super_polar_track(w, (95., 60., 175., 40.), (85., 50.), (66., 15., 120., 75.), 65., True),
    lambda w: super_polar_track(w, (120., 50., 195., 35.), (90., 45.), (120., 50., 195., 65.), 85., True, (85., 425.), 24.)
)

def build_curriculum_configs(
    num_easy = NUM_EASY,
    ladder_size = LADDER_SIZE,
    native_bank_size = NATIVE_BANK_SIZE,
    super_native_size = SUPER_NATIVE_SIZE,
    skill_tracks = SKILL_TRACKS,
    super_tracks = SUPER_NATIVE_TRACKS
):
    foothold_cfgs = [anchor_foothold_config(i, num_easy = num_easy) for i in range(num_easy)]
    sub_count = max(0, ladder_size - len(foothold_cfgs))
    ladder_cfgs = [
        skill_tracks[i % len(skill_tracks)](((i + 1) / max(sub_count, 1)) ** 1.35)
        for i in range(sub_count)
    ]

    num_super_levels = super_native_size // max(len(super_tracks), 1)
    super_cfgs = [
        s_fn(((lvl + 1) / max(num_super_levels, 1)) ** 1.20)
        for lvl in range(num_super_levels)
        for s_fn in super_tracks
    ]

    return [*foothold_cfgs, *ladder_cfgs, *([native_config] * native_bank_size), *super_cfgs]

build_expanded_configs = build_curriculum_configs

def materialize_config(c, rng, idx = 0, step = 0):
    return c(rng, idx, step) if callable(c) else c

def compute_frontier_k(p: Tensor, counts: Tensor, threshold = 0.25, min_count = 2, kernel = 3, fallback = 4):
    return ladder_frontier_index(p, counts, threshold = threshold, min_count = min_count, kernel = kernel, fallback = fallback)

def curriculum_score_fn(p, counts, target = 0.40, kappa = 4., eps = 1e-4, unvisited_weight = 0.25):
    w = ladder_score_fn(
        p, counts,
        target = target, kappa = kappa, eps = eps, unvisited_weight = unvisited_weight,
        kernel = 3, threshold = 0.25,
        probe_offset = 4., probe_width = 6., envelope_span = 16., envelope_width = 6.
    )
    k_star = compute_frontier_k(p, counts)
    idx = arange(len(w), device = w.device, dtype = w.dtype)
    w = w * torch.exp(-(idx - (float(k_star) + 22.)).clamp(min = 0.) / 16.)

    if k_star >= 80:
        w = w * torch.exp(-((float(k_star) - 40.) - idx).clamp(min = 0.) / 28.).clamp(min = 0.15)
        native_band = (idx >= LADDER_SIZE) & (idx < LADDER_SIZE + NATIVE_BANK_SIZE)
        w = torch.where(native_band, w * 1.65, w)

    return w

frontier_ladder_score_fn = curriculum_score_fn

def make_sampler(
    configs,
    seed = 0,
    progression_path = None,
    target = 0.40,
    kappa = 4.,
    eps = 1e-4,
    window = 10,
    unvisited_weight = 0.25,
    uniform_prob = 0.06,
    score_fn = curriculum_score_fn
):
    return SGSSampler(
        configs = configs,
        target = target,
        kappa = kappa,
        eps = eps,
        window = window,
        unvisited_weight = unvisited_weight,
        uniform_prob = uniform_prob,
        score_fn = score_fn,
        log_every = None,
        log_path = progression_path,
        seed = seed
    )

# held-out evaluation and video rendering

@torch.no_grad()
def evaluate_heldout(trainer, bank_envs, eval_states, max_steps = 160, seed_base = 0, deterministic = True):
    n = len(eval_states)
    obs = np.stack([
        e.reset(seed = int(seed_base + i), options = dict(reset_to_state = c))[0]
        for i, (e, c) in enumerate(zip(bank_envs, eval_states))
    ]).astype(np.float32)

    cur_s = transform_pusht(obs)
    prev_s = cur_s.copy()
    prev_a = np.zeros((n, DIM_ACTION), dtype = np.float32)
    done = np.zeros(n, dtype = bool)
    succ = np.zeros(n, dtype = np.float32)
    max_cov = np.zeros(n, dtype = np.float32)
    goal_t = torch.from_numpy(extract_goal_pusht(obs)).float().to(trainer.device)

    with torch.random.fork_rng():
        torch.manual_seed(int(seed_base + 1))

        for _ in range(max_steps):
            if done.all():
                break

            state_t = torch.from_numpy(make_aug_state_2d(cur_s, prev_s, prev_a)).float().to(trainer.device)
            act_t, _ = trainer.act(state_t, goal_t, deterministic = deterministic)
            act = act_t.clamp(-1., 1.).cpu().numpy()

            for i in np.where(~done)[0]:
                i = int(i)
                obs[i], _, term, trunc, info = bank_envs[i].step(act[i])
                prev_s[i], prev_a[i], cur_s[i] = cur_s[i], act[i], transform_pusht(obs[i])
                max_cov[i] = max(max_cov[i], float(info.get('coverage', 0.)))
                if term or trunc:
                    succ[i], done[i] = float(term), True

    return dict(success_rate = float(succ.mean()), max_cov = float(max_cov.mean()), per_state = [float(x) for x in succ])

@torch.no_grad()
def render_heldout_grid(trainer, eval_states, out_path, max_steps = 160, deterministic = True, grid_cols = 4, fps = 12, delta_scale = 55.):
    n = min(len(eval_states), 16)
    sel = [eval_states[i] for i in np.linspace(0, len(eval_states) - 1, n, dtype = int)]
    render_envs = [make_pusht(max_steps = max_steps, seed = 80_000 + i, render_mode = 'rgb_array', delta_scale = delta_scale) for i in range(n)]

    obs = np.stack([
        e.reset(seed = 80_000 + i, options = dict(reset_to_state = c))[0]
        for i, (e, c) in enumerate(zip(render_envs, sel))
    ]).astype(np.float32)

    cur_s = transform_pusht(obs)
    prev_s = cur_s.copy()
    prev_a = np.zeros((n, DIM_ACTION), dtype = np.float32)
    done = np.zeros(n, dtype = bool)
    succ = np.zeros(n, dtype = bool)
    frozen = [None] * n
    goal_t = torch.from_numpy(extract_goal_pusht(obs)).float().to(trainer.device)
    frames = []

    def snap_panel(idx, green = False):
        raw = rearrange(torch.from_numpy(render_envs[idx].render()).float(), 'h w c -> 1 c h w')
        img = rearrange(F.interpolate(raw, size = (192, 192), mode = 'area'), '1 c h w -> h w c').byte().numpy().copy()
        if green:
            img[:5, :] = img[-5:, :] = img[:, :5] = img[:, -5:] = [40, 220, 80]
        return img

    cols = next(c for c in range(min(grid_cols, n), 0, -1) if divisible_by(n, c))

    with torch.random.fork_rng():
        torch.manual_seed(0)

        for step_i in range(max_steps):
            if divisible_by(step_i, 3) or done.all():
                rgb = np.stack([frozen[i] if exists(frozen[i]) else snap_panel(i, green = bool(succ[i])) for i in range(n)])
                frames.append(rearrange(rgb, '(r c) h w ch -> (r h) (c w) ch', c = cols))

            if done.all():
                break

            state_t = torch.from_numpy(make_aug_state_2d(cur_s, prev_s, prev_a)).float().to(trainer.device)
            act_t, _ = trainer.act(state_t, goal_t, deterministic = deterministic)
            act = act_t.clamp(-1., 1.).cpu().numpy()

            for i in np.where(~done)[0]:
                i = int(i)
                obs[i], _, term, trunc, _ = render_envs[i].step(act[i])
                prev_s[i], prev_a[i], cur_s[i] = cur_s[i], act[i], transform_pusht(obs[i])
                if term or trunc:
                    succ[i], done[i] = bool(term), True
                    frozen[i] = snap_panel(i, green = bool(term))

    for e in render_envs:
        e.close()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    with imageio.get_writer(str(out_path), fps = fps, codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1) as writer:
        for f in frames:
            writer.append_data(f[..., :3])

    imageio.mimsave(str(out_path.with_suffix('.gif')), [f[..., :3] for f in frames], fps = fps, loop = 0)
    return str(out_path), float(succ.mean())

# main

def main(
    num_envs: int = 16,
    rollout_len: int = 128,
    num_updates: int = 15_000,
    max_steps: int = 160,
    delta_scale: float = 55.,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    value_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 0.5,
    discount: float = 0.985,
    lam: float = 0.95,
    finite_time_gae: bool = True,
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
    critic_distance_scale: float = 10.,
    value_hidden: int = 256,
    value_depth: int = 12,
    value_residual_every: int = 2,
    value_num_bins: int = 256,
    value_epochs: int = 6,
    value_minibatch_size: int = 256,
    env_reward_weight: float = 0.5,
    success_bonus: float = 1.,
    self_goal_cos_weight: float = 1.,
    num_easy: int = NUM_EASY,
    num_frontier: int = NUM_FRONTIER,
    ladder_size: int = LADDER_SIZE,
    native_bank_size: int = NATIVE_BANK_SIZE,
    super_native_size: int = SUPER_NATIVE_SIZE,
    num_heldout_tiers: int = NUM_HELDOUT_TIERS,
    num_heldout_native: int = NUM_HELDOUT_NATIVE,
    sgs_target: float = 0.40,
    sgs_kappa: float = 4.,
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
    resume_tag: str | None = None,
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

    print(
        f"[PushT-2D SGS+CRL] max_steps={max_steps} | finite_time_gae={finite_time_gae} | "
        f"videos -> {Path(video_folder).resolve()} | checkpoints -> {Path(checkpoint_folder).resolve()}",
        flush = True
    )

    anchor_configs = build_anchor_configs(num_easy = num_easy, num_frontier = num_frontier)
    configs = build_curriculum_configs(num_easy = num_easy, ladder_size = ladder_size, native_bank_size = native_bank_size, super_native_size = super_native_size)
    sampler = make_sampler(configs, seed, progression_path, sgs_target, sgs_kappa, sgs_eps, sgs_window, sgs_unvisited_weight, sgs_uniform_prob)

    success_fn = lambda _o, _r, term, _trunc, info: bool(term or info.get('success', False))
    envs = VecEnv([
        (lambda i = idx: SuccessGuidedSamplingWrapper(
            make_pusht(max_steps = max_steps, seed = seed + i, delta_scale = delta_scale),
            sampler,
            success_fn = success_fn
        ))
        for idx in range(num_envs)
    ])

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
    prev_act = np.zeros((num_envs, DIM_ACTION), dtype = np.float32)

    goal_rng = np.random.default_rng(seed + 135_791)
    goal_buffer = deque([
        transform_pusht(config_to_obs(materialize_config(configs[g_i % len(configs)], goal_rng, g_i)))[:DIM_GOAL]
        for g_i in range(goal_buffer_batch)
    ], maxlen = goal_buffer_size)

    def randomize_env_agent_outside_block(env_i, raw_obs_i):
        th_cur = float(raw_obs_i[4])
        b_com = np.array([raw_obs_i[2] - 45. * np.sin(th_cur), raw_obs_i[3] + 45. * np.cos(th_cur)], dtype = np.float32)
        pa_start = sample_random_agent_pos_outside_block(goal_rng, b_com, th_cur, min_clearance = 18.5)

        u_env = env_i.unwrapped
        u_env.agent.position = [float(pa_start[0]), float(pa_start[1])]
        u_env.agent.velocity = (0., 0.)
        u_env.block.velocity = (0., 0.)
        u_env.block.angular_velocity = 0.
        u_env.space.step(u_env.dt)
        new_raw = u_env.get_obs().astype(np.float32)

        cur = env_i
        while hasattr(cur, 'env'):
            if isinstance(cur, PushTRelativeActionWrapper):
                cur._last_raw_obs = new_raw.copy()
                break
            cur = cur.env

        return new_raw

    current_goals = np.zeros((num_envs, DIM_GOAL), dtype = np.float32)
    is_explore_goal = np.zeros(num_envs, dtype = bool)

    def assign_env_goal(i, raw_obs_i):
        if len(goal_buffer) > 0 and np.random.rand() < exploration_goal_prob:
            current_goals[i] = goal_buffer[np.random.randint(len(goal_buffer))].copy()
            is_explore_goal[i] = True
            envs.envs[i].current_config_ids[0] = -1

            if np.random.rand() < explore_random_agent_prob:
                raw_obs_i = randomize_env_agent_outside_block(envs.envs[i], raw_obs_i)
                cur_single_obs[i] = prev_single_obs[i] = transform_pusht(raw_obs_i)

            set_simulator_goal_from_vec_2d(envs.envs[i], current_goals[i])
        else:
            current_goals[i], is_explore_goal[i] = extract_goal_pusht(raw_obs_i), False

        return raw_obs_i

    for i in range(num_envs):
        obs_raw[i] = assign_env_goal(i, obs_raw[i])
    obs = make_aug_state_2d(cur_single_obs, prev_single_obs, prev_act)

    actor = ResidualBetaActor(DIM_STATE, DIM_GOAL, DIM_ACTION, actor_hidden, actor_depth, residual_every = actor_residual_every).to(device)
    state_encoder = ResidualEncoder(DIM_STATE, critic_hidden, critic_depth, critic_embed_dim, residual_every = critic_residual_every).to(device)
    goal_encoder = ResidualEncoder(DIM_GOAL, critic_hidden, critic_depth, critic_embed_dim, residual_every = critic_residual_every).to(device)
    value = ResidualEncoder(DIM_STATE + DIM_GOAL, value_hidden, value_depth, value_num_bins, residual_every = value_residual_every).to(device)
    env_value = ResidualEncoder(DIM_STATE + DIM_GOAL, value_hidden, value_depth, value_num_bins, residual_every = value_residual_every).to(device)

    contrastive_learn = ContrastiveLearning(use_euclidean = True, l2norm_embed = True, learned_temp = False, scale = critic_distance_scale, symmetric = True)
    critic_wrapper = ContrastiveWrapper(state_encoder, contrastive_learn, future_encoder = goal_encoder).to(device)

    trainer = DistanceRewardPPO(
        actor, critic_wrapper,
        value = value, env_value = env_value,
        value_num_bins = value_num_bins, env_value_max_return = float(max(success_bonus, 1.)),
        separate_env_rewards = True, env_reward_weight = env_reward_weight, mask_done_advantages = False,
        use_hpo = True, action_type = 'continuous',
        actor_lr = actor_lr, critic_lr = critic_lr, value_lr = value_lr,
        value_epochs = value_epochs, value_minibatch_size = value_minibatch_size,
        weight_decay = weight_decay, max_grad_norm = max_grad_norm,
        discount = discount, lam = lam, finite_time_gae = finite_time_gae,
        clip_eps = clip_eps, entropy_coef = entropy_coef,
        ppo_epochs = ppo_epochs, minibatch_size = minibatch_size,
        critic_steps = critic_steps, critic_batch_size = critic_batch_size,
        critic_discount = critic_discount, critic_negative_buffer_size = critic_negative_buffer_size,
        self_goal_cos_weight = self_goal_cos_weight,
        state_to_goal_fn = lambda t: t[..., :DIM_GOAL],
        device = device
    )

    ckpt_modules = (('actor', actor), ('critic', critic_wrapper), ('value', value), ('env-value', env_value))
    ckpt_opts = (('actor_opt', trainer.actor_opt), ('critic_opt', trainer.critic_opt), ('value_opt', trainer.value_opt), ('env_value_opt', trainer.env_value_opt))

    def save_bundle(tag, upd, k_st, b_score, b_native_g):
        for name, mod in ckpt_modules:
            torch.save(mod.state_dict(), f'{checkpoint_folder}/{name}-{tag}.pt')

        torch.save(
            dict(history = [list(h) for h in sampler.history], counts = sampler.counts.clone(), step_count = sampler.step_count),
            f'{checkpoint_folder}/sampler-{tag}.pt'
        )
        torch.save(
            {k: opt.state_dict() for k, opt in ckpt_opts if exists(opt)},
            f'{checkpoint_folder}/optim-{tag}.pt'
        )
        save_json(f'{checkpoint_folder}/ckpt_meta-{tag}.json', dict(
            update = int(upd), env_steps = int(upd * num_envs * rollout_len),
            actor_depth = int(actor_depth), critic_depth = int(critic_depth), value_depth = int(value_depth),
            k_star = int(k_st), best_score = float(b_score), best_native_greedy = float(b_native_g)
        ))

    num_episodes, best_score, best_native_greedy = 0, float('-inf'), 0.

    if exists(resume_tag):
        for name, mod in ckpt_modules:
            mod.load_state_dict(torch.load(f'{checkpoint_folder}/{name}-{resume_tag}.pt', map_location = device))

        if (sampler_path := Path(f'{checkpoint_folder}/sampler-{resume_tag}.pt')).exists():
            s_state = torch.load(sampler_path, map_location = 'cpu')
            n_common = min(len(sampler.history), len(s_state['history']))
            for h_deq, h_list in zip(sampler.history[:n_common], s_state['history'][:n_common]):
                h_deq.clear()
                h_deq.extend(h_list)
            sampler.counts[:n_common].copy_(s_state['counts'][:n_common])
            sampler.step_count = int(s_state['step_count'])

        if (optim_path := Path(f'{checkpoint_folder}/optim-{resume_tag}.pt')).exists():
            opt_state = torch.load(optim_path, map_location = 'cpu')
            for key, opt in ckpt_opts:
                if key in opt_state and exists(opt):
                    opt.load_state_dict(opt_state[key])

        if (best_meta := Path(f'{checkpoint_folder}/ckpt_meta-best.json')).exists():
            meta = json.loads(best_meta.read_text())
            best_score = float(meta.get('best_score', float('-inf')))
            best_native_greedy = float(meta.get('best_native_greedy', 0.))

    recent_succ = deque(maxlen = recent_succ_window)
    t0 = time.time()

    for update in range(start_update, num_updates + 1):
        states = np.zeros((num_envs, rollout_len, DIM_STATE), dtype = np.float32)
        next_states = np.zeros_like(states)
        goals = np.zeros((num_envs, rollout_len, DIM_GOAL), dtype = np.float32)
        actions = np.zeros((num_envs, rollout_len, DIM_ACTION), dtype = np.float32)
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
                    prev_act[i] = 0.
                else:
                    prev_single_obs[i], prev_act[i], cur_single_obs[i] = cur_single_obs[i], action_np[i], next_single_obs[i]

            obs = make_aug_state_2d(cur_single_obs, prev_single_obs, prev_act)

        flat_next_goals = rearrange(next_states, 'n t d -> (n t) d')[:, :DIM_GOAL]
        for g_vec in flat_next_goals[np.random.choice(len(flat_next_goals), size = min(goal_buffer_batch, len(flat_next_goals)), replace = False)]:
            goal_buffer.append(g_vec.copy())

        metrics = trainer.update(
            states = states, goals = goals, actions = actions, old_log_probs = log_probs,
            dones = terminateds | truncateds, next_states = next_states,
            terminated = terminateds, truncated = truncateds, success = terminateds,
            base_rewards = np.clip(env_rewards, 0., 1.), env_reward_weight = env_reward_weight
        )

        k_star = compute_frontier_k(sampler.success_rates, sampler.counts)
        train_succ = float(np.mean(recent_succ)) if recent_succ else 0.
        elapsed = time.time() - t0

        row = dict(
            update = update, env_steps = update * num_envs * rollout_len, episodes = num_episodes,
            train_succ = train_succ, k_star = int(k_star), finite_time_gae = bool(trainer.finite_time_gae),
            actor_depth = actor_depth, critic_depth = critic_depth, elapsed_sec = elapsed, **metrics
        )

        if divisible_by(update, eval_every) or update == start_update or update == num_updates:
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
                f"upd {update:4d} | steps {row['env_steps']:7d} | k* {k_star:3d}/{len(configs)-1} | "
                f"train {train_succ:4.1%} | id_err {metrics['self_cos_err']:.4f} | "
                f"anchor(g/s) {anchor_g['success_rate']:4.1%}/{anchor_s['success_rate']:4.1%} | "
                f"tiers(g/s) {tiers_g['success_rate']:4.1%}/{tiers_s['success_rate']:4.1%} (cov {row['heldout_tiers_cov']:.2f}) | "
                f"native(g/s) {native_g['success_rate']:4.1%}/{native_s['success_rate']:4.1%} (cov {row['heldout_native_cov']:.2f}) | "
                f"t {elapsed:5.1f}s",
                flush = True
            )

            cur_score = (
                2. * native_g['success_rate'] + 1. * native_s['success_rate'] + 0.5 * row['heldout_native_cov'] +
                1. * tiers_g['success_rate'] + 0.5 * tiers_s['success_rate'] +
                0.5 * anchor_f_g + 0.25 * anchor_g['success_rate'] + 0.15 * (float(k_star) / max(len(configs) - 1, 1))
            )

            if cur_score > best_score + 1e-4 or native_g['success_rate'] > best_native_greedy:
                best_score = max(best_score, cur_score)
                best_native_greedy = max(best_native_greedy, native_g['success_rate'])
                save_bundle('best', update, k_star, best_score, best_native_greedy)

            save_bundle('latest', update, k_star, best_score, best_native_greedy)

            if divisible_by(update, video_every) or update == start_update or update == num_updates:
                use_det = (tiers_g['success_rate'] > 0.) and (tiers_g['success_rate'] >= tiers_s['success_rate'] * 0.8)
                v_tiers_out = Path(video_folder) / f'pusht2d_heldout_tiers_upd{update:04d}.mp4'
                v_native_out = Path(video_folder) / f'pusht2d_heldout_native_upd{update:04d}.mp4'

                v_tiers, _ = render_heldout_grid(trainer, heldout_tiers[:16], v_tiers_out, max_steps, use_det, video_grid_cols, video_fps, delta_scale)
                v_native, _ = render_heldout_grid(trainer, heldout_native[:16], v_native_out, max_steps, use_det, video_grid_cols, video_fps, delta_scale)
                print(f"  -> saved heldout video trajectories (.mp4 & .gif): {v_tiers} , {v_native}", flush = True)

                if exists(artifact_folder) and Path(artifact_folder).exists():
                    for src, name in ((v_tiers, f'pusht2d_heldout_tiers_upd{update:04d}'), (v_native, f'pusht2d_heldout_native_upd{update:04d}')):
                        shutil.copy(Path(src), Path(artifact_folder) / f'{name}.mp4')
                        shutil.copy(Path(src).with_suffix('.gif'), Path(artifact_folder) / f'{name}.gif')

        elif divisible_by(update, log_every):
            print(
                f"upd {update:4d} | steps {row['env_steps']:7d} | critic {metrics['critic_loss']:6.3f} | "
                f"dist {metrics['distance_mean']:5.2f} | crl {metrics['crl_reward_mean']:+.4f} | "
                f"v_loss {metrics['value_loss']:5.3f}/{metrics['env_value_loss']:5.3f} | "
                f"k* {k_star:3d}/{len(configs)-1} | train {train_succ:4.1%} | t {elapsed:5.1f}s",
                flush = True
            )

        append_jsonl(log_path, row)

    save_bundle('final', update, k_star, best_score, best_native_greedy)
    envs.close()
    for e in [*anchor_envs, *tier_envs, *native_envs]:
        e.close()

if __name__ == '__main__':
    Fire(main)
