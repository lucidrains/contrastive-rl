# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "einops",
#   "env-ssl-wrapper>=0.6.8",
#   "fire",
#   "gymnasium",
#   "hl-gauss-pytorch>=0.2.2",
#   "imageio",
#   "imageio-ffmpeg",
#   "mani-skill==3.0.1",
#   "mean-conc-beta>=0.2.1",
#   "numpy",
#   "pin",
#   "sapien==3.0.3",
#   "torch",
#   "torch-einops-utils>=0.2.0",
#   "x-mlps-pytorch>=0.6.1",
#   "x-ppo>=0.1.0",
# ]
#
# [tool.uv.sources]
# contrastive-rl-pytorch = { path = "." }
# ///

# pusht-3d (maniskill panda stick) with sparse binary task reward,
# contrastive rl distance rewards, multi-critic ppo with finite-time gae,
# and success-guided sampling (sgs)

from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

if sys.platform == 'darwin' and Path('/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json').exists():
    os.environ.setdefault('VK_ICD_FILENAMES', '/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json')
    os.environ.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/lib')

import gymnasium as gym
from gymnasium import spaces
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
from mani_skill.utils.structs import Pose
import numpy as np

import torch
from torch import Tensor, arange, is_tensor
import torch.nn.functional as F

from einops import rearrange
from fire import Fire
from torch_einops_utils.shape import default, divisible_by, exists

from contrastive_rl_pytorch import ContrastiveLearning, ContrastiveWrapper
from train_ppo_with_crl_distance_rewards import (
    DistanceRewardPPO,
    ResidualBetaActor,
    ResidualEncoder,
    VecEnv,
    append_jsonl,
    save_json
)
from env_ssl_wrapper import SGSSampler, SuccessGuidedSamplingWrapper

# helpers

def wrap_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def lerp(a, b, t):
    return (1. - t) * a + t * b

def lerp_angle(a, b, t):
    return float(wrap_angle(a + t * wrap_angle(b - a)))

def polar(r, phi):
    return np.array([r * math.cos(phi), r * math.sin(phi)], dtype = np.float32)

# pusht-3d environment and world-pose representation

GOAL_POS = np.array([-0.156, -0.100], dtype = np.float32)
GOAL_YAW = 5. * math.pi / 3.
TEE_Z = 0.021
GOAL_Z = 1e-3

DEFAULT_QPOS = np.array([0.662, 0.212, 0.086, -2.685, -0.115, 2.898, 1.673], dtype = np.float32)
QPOS_NOISE = 0.02
QPOS_LOW = np.array([-2.880, -1.750, -2.880, -3.065, -2.880,  0.000, -2.880], dtype = np.float32)
QPOS_HIGH = np.array([ 2.880,  1.750,  2.880, -0.080,  2.880,  3.740,  2.880], dtype = np.float32)

DIM_SINGLE_OBS = 30
DIM_ACTION = 3
DIM_STATE = DIM_SINGLE_OBS * 2 + DIM_ACTION  # [cur_obs(30), prev_obs(30), prev_action(3)]
DIM_GOAL = 4                                 # [gx / 0.4, gy / 0.4, sin(gyaw), cos(gyaw)]

def quat_to_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))

def canonicalize_quat(q):
    return (q * np.where(q[..., :1] < 0., -1., 1.)).astype(np.float32)

def transform_pusht3d(obs):
    raw = np.asarray(obs, dtype = np.float32)
    tee_quat = canonicalize_quat(raw[..., 27:31])
    tee_yaw = quat_to_yaw(tee_quat)

    return np.concatenate([
        raw[..., 24:26] / 0.4,
        np.stack([np.sin(tee_yaw), np.cos(tee_yaw)], axis = -1),
        raw[..., 26:27] / 0.4,
        tee_quat,
        raw[..., 0:7] / np.pi,
        np.tanh(raw[..., 7:14] / 2.),
        raw[..., 14:17] / 0.4,
        canonicalize_quat(raw[..., 17:21])
    ], axis = -1).astype(np.float32)

def extract_goal_pusht3d(obs):
    raw = np.asarray(obs, dtype = np.float32)
    gxy = raw[..., 21:23] / 0.4
    gyaw = raw[..., 31] if raw.shape[-1] >= 32 else np.full_like(gxy[..., 0], GOAL_YAW)
    return np.concatenate([gxy, np.stack([np.sin(gyaw), np.cos(gyaw)], axis = -1)], axis = -1).astype(np.float32)

def config_to_goal_3d(cfg):
    tx, ty, tyaw = map(float, cfg[:3])
    return np.array([tx / 0.4, ty / 0.4, math.sin(tyaw), math.cos(tyaw)], dtype = np.float32)

def make_aug_state_3d(cur_s, prev_s, prev_a):
    return np.concatenate([cur_s, prev_s, prev_a], axis = -1).astype(np.float32)

def unwrap_pusht3d(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

def set_goal_pose_3d(raw_env, gx, gy, gyaw):
    device = getattr(raw_env, 'device', torch.device('cpu'))
    gc, gs = math.cos(gyaw / 2.), math.sin(gyaw / 2.)

    raw_env.goal_tee.set_pose(Pose.create_from_pq(
        p = torch.tensor([[gx, gy, GOAL_Z]], dtype = torch.float32, device = device),
        q = torch.tensor([[gc, 0., 0., gs]], dtype = torch.float32, device = device)
    ))
    raw_env.goal_offset = torch.tensor([gx, gy], dtype = torch.float32, device = device)
    raw_env.goal_z_rot = gyaw

    cg, sg = math.cos(gyaw), math.sin(gyaw)
    trans = torch.eye(3, device = device)
    trans[:2, :2] = torch.tensor([[cg, -sg], [sg, cg]], dtype = torch.float32, device = device)
    trans[0, 2], trans[1, 2] = gx, gy
    raw_env.world_to_goal_trans = torch.linalg.inv(trans)

def apply_reset_state_3d(raw_env, state):
    st = np.asarray(state, dtype = np.float32).reshape(-1)
    tx, ty, tyaw, gx, gy, gyaw = map(float, st[:6])
    device = getattr(raw_env, 'device', torch.device('cpu'))
    tc, ts = math.cos(tyaw / 2.), math.sin(tyaw / 2.)

    raw_env.tee.set_pose(Pose.create_from_pq(
        p = torch.tensor([[tx, ty, TEE_Z]], dtype = torch.float32, device = device),
        q = torch.tensor([[tc, 0., 0., ts]], dtype = torch.float32, device = device)
    ))
    set_goal_pose_3d(raw_env, gx, gy, gyaw)
    raw_env.agent.reset(torch.as_tensor(st[6:13], dtype = torch.float32, device = device)[None])

def install_reset_state_3d(raw_env):
    raw_env = getattr(raw_env, 'unwrapped', raw_env)
    if getattr(raw_env, '_sgs_reset_state_installed', False):
        return raw_env

    orig_init = raw_env._initialize_episode

    def patched(env_idx, options):
        orig_init(env_idx, options)
        state = (options or {}).get('reset_to_state')
        if exists(state):
            apply_reset_state_3d(raw_env, state[0] if isinstance(state, (list, tuple)) else state)

    raw_env._initialize_episode = patched
    raw_env._sgs_reset_state_installed = True
    return raw_env

def set_simulator_goal_from_vec_3d(env, goal_vec, raw_obs = None):
    g = rearrange(np.asarray(goal_vec, dtype = np.float32), '... -> (...)')
    gx, gy, gyaw = float(g[0] * 0.4), float(g[1] * 0.4), math.atan2(float(g[2]), float(g[3]))
    set_goal_pose_3d(unwrap_pusht3d(env).raw.unwrapped, gx, gy, gyaw)

    if not exists(raw_obs):
        return None

    out = np.asarray(raw_obs, dtype = np.float32).copy()
    out[21], out[22], out[23], out[31] = gx, gy, GOAL_Z, gyaw
    return out

# sparse binary pusht-v1 wrapper (reward = 1. iff terminated on >= 90% goal intersection)

class PushT3DEnv(gym.Env):
    def __init__(self, max_steps = 240, seed = 0, render_mode = None, action_scale = (0.25, 0.25, 0.05)):
        super().__init__()
        self.raw = gym.make(
            'PushT-v1',
            obs_mode = 'state',
            num_envs = 1,
            control_mode = 'pd_ee_delta_pos',
            sim_backend = 'physx_cpu',
            max_episode_steps = max_steps,
            render_mode = render_mode
        )

        u = install_reset_state_3d(self.raw.unwrapped)
        u.compute_dense_reward = lambda obs, action, info: torch.zeros(1, device = u.device)
        orig_pseudo = u.pseudo_render_intersection

        def fast_pseudo_render_intersection():
            d_xy = torch.norm(u.tee.pose.p[:, :2] - u.goal_tee.pose.p[:, :2], dim = -1)
            return torch.zeros_like(d_xy) if bool((d_xy > 0.06).all()) else orig_pseudo()

        u.pseudo_render_intersection = fast_pseudo_render_intersection

        self.seed_val = int(seed)
        self._step_idx = 0
        self.action_scale = np.asarray(action_scale, dtype = np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (32,), dtype = np.float32)
        self.action_space = spaces.Box(low = -1., high = 1., shape = (3,), dtype = np.float32)

    def _pack_obs(self, obs):
        if is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        flat = np.asarray(obs, dtype = np.float32).reshape(-1)
        return np.append(flat[:31], np.float32(getattr(self.raw.unwrapped, 'goal_z_rot', GOAL_YAW))).astype(np.float32)

    def _overlap_proxy(self, obs32):
        d_pos = float(np.linalg.norm(obs32[24:26] - obs32[21:23]))
        d_yaw = abs(float(wrap_angle(quat_to_yaw(obs32[27:31]) - obs32[31])))
        return float(np.exp(-4.5 * d_pos - 0.6 * d_yaw))

    def reset(self, seed = None, options = None):
        self._step_idx += 1
        obs, _ = self.raw.reset(seed = int(default(seed, self.seed_val + self._step_idx)), options = options)
        obs32 = self._pack_obs(obs)
        return obs32, dict(coverage = self._overlap_proxy(obs32), success = False, is_success = False)

    def step(self, action):
        scaled_act = np.clip(np.asarray(action, dtype = np.float32), -1., 1.) * self.action_scale
        obs, _, term, trunc, info = self.raw.step(scaled_act)
        obs32 = self._pack_obs(obs)

        succ = bool(np.asarray(info.get('success', False)).reshape(-1)[0])
        terminated = bool(np.asarray(term).reshape(-1)[0]) or succ
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        cov = 1. if succ else self._overlap_proxy(obs32)

        return obs32, float(terminated), terminated, truncated, dict(coverage = cov, success = terminated, is_success = terminated)

    def render(self):
        frame = self.raw.render()
        if is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        return np.asarray(frame[0] if frame.ndim == 4 else frame, dtype = np.uint8).copy()

    def close(self):
        self.raw.close()

def make_pusht3d(max_steps = 240, seed = 0, render_mode = None, action_scale = (0.25, 0.25, 0.05)):
    env = PushT3DEnv(max_steps = max_steps, seed = seed, render_mode = render_mode, action_scale = action_scale)
    env.reset(seed = seed)
    return env

# se(2) curriculum and success-guided sampling
#
# each curriculum rung is a 13d reset state or callable `fn(rng, idx, step)`:
#   [tee_x, tee_y, tee_yaw, goal_x, goal_y, goal_yaw, qpos(7)]
#   (`step == 0` is deterministic held-out evaluation; `step > 0` is training)
#
# default 220-rung bank = LADDER_SIZE (180) + NATIVE_BANK_SIZE (40):
#   stage 1 (0..19):    13 easy contact footholds + 7 precision docking rungs
#   stage 2 (20..179):  interleaved `SKILL_TRACKS` (`w` in (0, 1])
#   stage 3 (180..219): 18 `native_bridge_track` + 8 `native_config` + 14 `beyond_native_track`
#
# to iterate on the curriculum, edit `SKILL_TRACKS`, `build_curriculum_configs`, or `curriculum_score_fn`

EASY_RING_STATES = np.array([
    [-0.1231, -0.1587, -1.0472, -0.1329, -0.1413, -1.0472, -0.4307,  0.6026, -0.0261, -1.9500,  0.0497, 2.5721,  0.2278],
    [-0.0974, -0.1342, -1.0472, -0.1147, -0.1241, -1.0472, -0.3063,  0.6966, -0.0021, -1.7738,  0.0170, 2.4921,  0.4001],
    [-0.2263, -0.0480, -1.0472, -0.2102, -0.0599, -1.0472,  0.1426,  0.0516, -0.0129, -2.9708, -0.0044, 3.0459,  0.8556],
    [-0.2074, -0.0698, -1.0472, -0.1901, -0.0799, -1.0472,  0.1435, -0.0065, -0.1393, -3.0680,  0.0184, 3.0892,  0.7071],
    [-0.1963, -0.0890, -1.0472, -0.1770, -0.0942, -1.0472,  0.1188, -0.0343, -0.3485, -3.0680, -0.1261, 3.1060,  0.6177],
    [-0.1572, -0.0312, -1.0472, -0.1569, -0.0512, -1.0472,  0.1631,  0.3055,  0.0441, -2.5072, -0.0534, 2.8354,  0.9771],
    [-0.1197, -0.0381, -1.0472, -0.1298, -0.0554, -1.0472,  0.0860,  0.4802,  0.0411, -2.1811, -0.0451, 2.6841,  0.8839],
    [-0.1899, -0.0393, -1.0472, -0.1802, -0.0567, -1.0472,  0.1690,  0.1286,  0.0251, -2.8325, -0.0371, 2.9843,  0.9519],
    [-0.2119, -0.1009, -1.0472, -0.1919, -0.1006, -1.0472,  0.1515, -0.0239, -0.5431, -3.0680, -0.0340, 3.0950,  0.3637],
    [-0.2174, -0.1347, -1.0472, -0.2000, -0.1248, -1.0472, -0.1714,  0.1110, -0.3601, -2.8715,  0.3447, 2.9856, -0.1481],
    [-0.1926, -0.1607, -1.0472, -0.1823, -0.1436, -1.0472, -0.4925,  0.2769, -0.1335, -2.5606,  0.1797, 2.8487, -0.0709],
    [-0.1574, -0.1685, -1.0472, -0.1570, -0.1485, -1.0472, -0.5124,  0.4521, -0.0624, -2.2325,  0.0973, 2.7007,  0.0655],
    [-0.1215, -0.0990, -1.0472, -0.1415, -0.0996, -1.0472, -0.1666,  0.7047,  0.0160, -1.7591, -0.0079, 2.4866,  0.5731],
], dtype = np.float32)

EASY_RING_ANGLES = np.array([300., 330., 135., 150., 165., 90., 60., 120., 180., 210., 240., 270., 0.], dtype = np.float32)

# compass sectors k * 30 deg for k in 0..11 (k in {5, 6, 7} = 150..210 deg is the -x elbow-limit sector)

FRONTIER_RING_STATES = np.array([
    [-0.0129, -0.0979,  0.5236, -0.1560, -0.1000, -1.0472, -0.1500,  0.9356,  0.0194, -1.3264, -0.0138, 2.2848,  0.5916],  #   0 deg
    [ 0.0143, -0.0024,  0.5236, -0.1560, -0.1000, -1.0472,  0.0383,  0.8524,  0.0384, -1.4829, -0.0403, 2.3581,  0.8128],  #  30 deg
    [-0.0550,  0.0704,  0.5236, -0.1560, -0.1000, -1.0472,  0.2033,  0.6187,  0.0545, -1.9219, -0.0648, 2.5622,  1.0226],  #  60 deg
    [-0.1539,  0.0959, -2.6180, -0.1560, -0.1000, -1.0472,  0.3367,  0.3617,  0.0653, -2.4028, -0.0870, 2.7849,  1.2002],  #  90 deg
    [-0.2522,  0.0693, -2.6180, -0.1560, -0.1000, -1.0472,  0.4066,  0.0981,  0.0515, -2.8863, -0.1007, 3.0054,  1.2788],  # 120 deg
    [-0.2210, -0.0625, -2.6180, -0.1560, -0.1000, -1.0472,  0.0734, -0.0061, -0.0567, -3.0668,  0.0223, 3.0884,  0.7175],  # 150 deg
    [-0.2280, -0.1000,  0.5236, -0.1560, -0.1000, -1.0472,  0.0165,  0.0053, -0.3417, -3.0492,  0.1644, 3.0738,  0.2290],  # 180 deg
    [-0.2210, -0.1375, -2.6180, -0.1560, -0.1000, -1.0472, -0.1733,  0.1103, -0.3609, -2.8725,  0.3449, 2.9859, -0.1510],  # 210 deg
    [-0.2574, -0.2692, -2.6180, -0.1560, -0.1000, -1.0472, -0.4861,  0.3378, -0.3263, -2.4727,  0.3661, 2.7873, -0.4172],  # 240 deg
    [-0.1596, -0.2934, -2.6180, -0.1560, -0.1000, -1.0472, -0.6313,  0.5771, -0.0870, -1.9995,  0.1221, 2.5881, -0.0860],  # 270 deg
    [-0.0626, -0.2662,  0.5236, -0.1560, -0.1000, -1.0472, -0.5101,  0.8156, -0.0316, -1.5504,  0.0532, 2.3842,  0.1514],  # 300 deg
    [ 0.0097, -0.1965, -2.6180, -0.1560, -0.1000, -1.0472, -0.3360,  0.9953, -0.0015, -1.2143,  0.0138, 2.2310,  0.3763],  # 330 deg
], dtype = np.float32)

OUTER_SAFE_RING_INDICES = (0, 1, 2, 3, 4, 8, 9, 10, 11)
ANCHOR_FRONTIER_INDICES = (0, 1, 2, 3, 4, 7)
PHI_DEFAULT_RAY = math.atan2(0.284 - float(GOAL_POS[1]), -0.321 - float(GOAL_POS[0]))

NUM_EASY = len(EASY_RING_STATES)
NUM_FRONTIER = len(ANCHOR_FRONTIER_INDICES)
FOOTHOLD_SIZE = 20
LADDER_SIZE = 180
NATIVE_BANK_SIZE = 40
NUM_HELDOUT_TIERS = 24
NUM_HELDOUT_NATIVE = 16

# curriculum geometric helpers

def pick_index(rng, n, idx, is_eval):
    return (int(idx) % n) if is_eval else int(rng.integers(n))

def choose_sign(rng, idx = 0, is_eval = False):
    even = divisible_by(int(idx), 2) if is_eval else (int(rng.integers(2)) == 0)
    return 1. if even else -1.

def split_difficulty(w, split = 0.84):
    w = float(np.clip(w, 0., 1.))
    return min(1., w / split), max(0., (w - split) / max(1. - split, 1e-6))

def qpos_noise(rng, std = QPOS_NOISE):
    return rng.normal(0., float(std), size = 7).astype(np.float32)

def table_safe_qpos_noise(rng, std):
    noise = qpos_noise(rng, std)
    noise[1] = min(0., float(noise[1]))  # prevent downward shoulder pitch into table
    return noise

def make_state(tee, goal, tee_yaw, goal_yaw, qpos):
    q = np.clip(np.asarray(qpos, dtype = np.float32).reshape(7), QPOS_LOW, QPOS_HIGH)
    return np.concatenate([
        np.asarray(tee, dtype = np.float32).reshape(2),
        [float(wrap_angle(tee_yaw))],
        np.asarray(goal, dtype = np.float32).reshape(2),
        [float(wrap_angle(goal_yaw))],
        q
    ]).astype(np.float32)

def nearest_easy_idx(angle_deg):
    return int(np.abs((EASY_RING_ANGLES - angle_deg + 180.) % 360. - 180.).argmin())

def lerp_waypoints(pts, alpha):
    alpha = float(np.clip(alpha, 0., 1.))
    if alpha <= 0. or len(pts) <= 1:
        return np.asarray(pts[0], dtype = np.float32)
    if alpha >= 1.:
        return np.asarray(pts[-1], dtype = np.float32)

    scaled = alpha * (len(pts) - 1)
    seg = min(int(scaled), len(pts) - 2)
    return np.asarray(lerp(pts[seg], pts[seg + 1], scaled - seg), dtype = np.float32)

def retract_ring_qpos(q_ring, k, alpha):
    f = FRONTIER_RING_STATES[:, 6:13]
    if k in (5, 6):
        pts = [q_ring, f[4], DEFAULT_QPOS]
    elif k in (7, 8, 9):
        pts = [q_ring, f[10], f[0], DEFAULT_QPOS]
    elif k in (10, 11):
        pts = [q_ring, f[0], DEFAULT_QPOS]
    else:
        pts = [q_ring, DEFAULT_QPOS]
    return lerp_waypoints(pts, alpha)

def ring_sector_pose(k, angle_deg, goal_blend, rng, pos_noise = 0.002):
    e_base = EASY_RING_STATES[nearest_easy_idx(angle_deg)]
    f_base = FRONTIER_RING_STATES[k]
    b = float(np.clip(goal_blend, 0., 1.))
    goal = np.asarray(lerp(e_base[3:5], GOAL_POS, b), dtype = np.float32) + rng.normal(0., pos_noise, size = 2).astype(np.float32)
    return e_base, f_base, goal, lerp_angle(float(e_base[5]), GOAL_YAW, b)

def ring_sector_qpos(k, e_base, f_base, rad_blend, lat_blend = 0., retract = 0., adj_sign = 1):
    rb = float(np.clip(rad_blend, 0., 1.))
    q = lerp(e_base[6:13], f_base[6:13], rb)

    if lat_blend > 0.:
        k_adj = (k + adj_sign) % 12
        q_adj = lerp(EASY_RING_STATES[nearest_easy_idx(30. * k_adj), 6:13], FRONTIER_RING_STATES[k_adj, 6:13], rb)
        q = lerp(q, q_adj, lat_blend)

    return retract_ring_qpos(q, 4 if k in (5, 6) else k, retract) if retract > 0. else q

def sample_exploration_qpos(rng):
    u_draw = float(rng.uniform(0., 1.))
    if u_draw < 0.35:
        q_base = DEFAULT_QPOS
    elif u_draw < 0.65:
        q_base = FRONTIER_RING_STATES[int(rng.integers(12)), 6:13]
    else:
        q_base = EASY_RING_STATES[int(rng.integers(NUM_EASY)), 6:13]
    return np.clip(q_base + qpos_noise(rng), QPOS_LOW, QPOS_HIGH)

# curriculum track generators

def native_config(rng, idx = 0, step = 0):
    tee = GOAL_POS + np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.20)], dtype = np.float32)
    tee_yaw = rng.uniform(-math.pi, math.pi)
    return make_state(tee, GOAL_POS, tee_yaw, GOAL_YAW, DEFAULT_QPOS + qpos_noise(rng))

def ring_config(idx):
    base = EASY_RING_STATES[int(idx) % NUM_EASY]
    delta = base[3:5] - base[:2]
    base_push = float(np.linalg.norm(delta))
    direction = delta / (base_push + 1e-8)

    def fn(rng, i = 0, step = 0):
        tee = base[:2] + rng.normal(0., 0.004, size = 2)
        push_dist = max(base_push + float(rng.normal(0., 0.002)), 0.005)
        goal = tee + push_dist * direction + rng.normal(0., 0.001, size = 2)
        return make_state(tee, goal, float(base[2] + rng.normal(0., 0.03)), float(base[5] + rng.normal(0., 0.02)), base[6:13] + table_safe_qpos_noise(rng, 0.005))

    return fn

def precision_dock_track(w):
    w = float(np.clip(w, 0., 1.))

    def fn(rng, idx = 0, step = 0):
        is_eval = (int(step) == 0)
        ww = float(np.clip(w + rng.normal(0., 0.03), 0., 1.))
        k = OUTER_SAFE_RING_INDICES[pick_index(rng, len(OUTER_SAFE_RING_INDICES), idx, is_eval)]
        angle_deg = 30. * k + float(rng.uniform(-12., 12.))

        e_base, f_base, goal, goal_yaw = ring_sector_pose(k, angle_deg, 0.40 + 0.60 * ww, rng)
        r = float(rng.uniform(0.005, 0.016 + 0.012 * ww))
        dyaw = choose_sign(rng, idx, is_eval) * float(rng.uniform(0.04 + 0.12 * ww, 0.15 + 0.65 * ww))
        q_base = lerp(e_base[6:13], f_base[6:13], float(np.clip(0.15 + 0.63 * (abs(dyaw) / 0.85), 0.15, 0.78)))
        return make_state(goal + polar(r, math.radians(angle_deg)), goal, goal_yaw + dyaw, goal_yaw, q_base + table_safe_qpos_noise(rng, 0.005))

    return fn

# stage 2 track 1: 360-deg radial push + se(2) bridge out to and beyond FRONTIER_RING_STATES

def frontier_translation_track(w):
    w = float(np.clip(w, 0., 1.))

    def fn(rng, idx = 0, step = 0):
        is_eval = (int(step) == 0)
        k = pick_index(rng, 12, idx, is_eval)
        u, over = split_difficulty(float(np.clip(w + rng.normal(0., 0.015), 0., 1.)), split = 0.82)

        ang_spread = 10. * (1. - 0.85 * u) + 12. * over
        angle_deg = 30. * k + float(rng.uniform(-ang_spread, ang_spread))
        e_base, f_base, goal, goal_yaw = ring_sector_pose(k, angle_deg, 1.5 * u, rng, pos_noise = 0.002 * (1. - 0.7 * u))

        f_r = float(np.linalg.norm(f_base[:2] - f_base[3:5]))
        f_dyaw = float(wrap_angle(float(f_base[2]) - float(f_base[5])))
        canon_sign = 1. if f_dyaw >= 0. else -1.
        flip_sign = (over > 0.) and (((int(idx) % 5) == 4) if is_eval else (float(rng.uniform(0., 1.)) < 0.45 * over))
        sign = -canon_sign if flip_sign else canon_sign

        if over <= 0.:
            dy_c = abs(f_dyaw) * (u ** 1.25)
            dy_h = abs(f_dyaw) * 0.18 * (1. - 0.78 * u)
            dyaw = sign * float(rng.uniform(max(0., dy_c - dy_h), dy_c + dy_h)) if u > 0.02 else float(rng.normal(0., 0.03))
            lat_blend = 0.12 * math.sin(math.pi * u) * float(rng.uniform(0., 1.)) if (0.15 < u < 0.95 and not is_eval) else 0.
        else:
            dyaw = sign * float(rng.uniform(0.70 * abs(f_dyaw), abs(f_dyaw) + 0.72 * over))
            lat_blend = 0.16 * over * float(rng.uniform(0.3, 1.))

        dev_yaw = abs(float(wrap_angle(dyaw - f_dyaw)))
        margin = 0.030 * min(1., dev_yaw / 0.70) + 0.030 * lat_blend
        r = float(np.clip(0.018 + (f_r - 0.018) * (u ** 1.35) + rng.normal(0., 0.002), 0.015, max(0.020, f_r - margin)))
        rad_blend = max(u ** 1.15, (r + margin - 0.018) / max(f_r - 0.018, 0.04))

        q_base = ring_sector_qpos(k, e_base, f_base, rad_blend, lat_blend, retract = 0.12 * over, adj_sign = int(choose_sign(rng, idx, is_eval)))
        return make_state(goal + polar(r, math.radians(angle_deg)), goal, goal_yaw + dyaw, goal_yaw, q_base + table_safe_qpos_noise(rng, 0.004 + 0.008 * over))

    return fn

# stage 2 track 2: on-goal spin (outer ring and DEFAULT_QPOS) + coupled 180-deg lever rotation

def lever_rotation_track(w):
    w = float(np.clip(w, 0., 1.))

    def fn(rng, idx = 0, step = 0):
        is_eval = (int(step) == 0)
        mode = pick_index(rng, 3, idx, is_eval)
        u, over = split_difficulty(float(np.clip(w + rng.normal(0., 0.02), 0., 1.)), split = 0.84)
        max_dyaw = 0.16 + (math.pi - 0.16) * (u ** 1.35)
        dyaw = choose_sign(rng, int(idx) // 3, is_eval) * float(rng.uniform((0.24 + 0.43 * over) * max_dyaw, max_dyaw))

        if mode == 0:
            k = OUTER_SAFE_RING_INDICES[pick_index(rng, len(OUTER_SAFE_RING_INDICES), int(idx) // 3, is_eval)]
            angle_deg = 30. * k + float(rng.uniform(-15., 15.))
            e_base, f_base, goal, goal_yaw = ring_sector_pose(k, angle_deg, 0.55 + 1.2 * u, rng, pos_noise = 0.0015)
            r = float(rng.uniform(0.002, 0.015 + 0.015 * u + 0.006 * over))
            phi = math.radians(angle_deg) if u < 0.20 else float(rng.uniform(-math.pi, math.pi))
            q_base = lerp(e_base[6:13], f_base[6:13], float(np.clip(0.68 + 0.22 * (abs(dyaw) / math.pi) + 0.06 * over, 0.68, 0.94)))
            return make_state(goal + polar(r, phi), goal, goal_yaw + dyaw, goal_yaw, q_base + table_safe_qpos_noise(rng, 0.005 + 0.007 * over))

        if mode == 1:
            r = float(rng.uniform(0.003, 0.018 + 0.024 * u + 0.014 * over))
            phi = float(PHI_DEFAULT_RAY + rng.uniform(-0.8, 0.8)) if u < 0.25 else float(rng.uniform(-math.pi, math.pi))
            alpha_def = float(np.clip(0.45 + 1.30 * u, 0.45, 1.))
            q_base = retract_ring_qpos(FRONTIER_RING_STATES[4, 6:13], 4, alpha_def)
            noise_std = (QPOS_NOISE + 0.018 * over) if alpha_def >= 0.999 else lerp(0.008, QPOS_NOISE, alpha_def)
            return make_state(GOAL_POS + polar(r, phi), GOAL_POS, GOAL_YAW + dyaw, GOAL_YAW, q_base + qpos_noise(rng, noise_std))

        k = pick_index(rng, 12, idx, is_eval)
        angle_deg = 30. * k + float(rng.uniform(-14. - 4. * over, 14. + 4. * over))
        e_base, f_base, goal, goal_yaw = ring_sector_pose(k, angle_deg, 0.35 + 1.4 * u, rng)
        f_r = float(np.linalg.norm(f_base[:2] - f_base[3:5]))
        lat_blend = float(rng.uniform(0.08, 0.32 + 0.08 * over)) * min(1., abs(dyaw) / 0.8)
        margin = 0.032 * min(1., abs(dyaw) / 0.90) + 0.032 * lat_blend
        r_hi = max(0.024, min(f_r - margin, 0.026 + (0.76 + 0.10 * over) * (f_r - 0.022) * (u ** 1.10)))
        r = float(rng.uniform(max(0.016, (0.28 + 0.25 * over) * r_hi), r_hi))
        rad_blend = max(0.38 + 0.58 * u, (r + margin - 0.016) / max(f_r - 0.016, 0.04))
        retract = 0.12 * max(0., (abs(dyaw) - 1.1) / (math.pi - 1.1)) + 0.10 * lat_blend
        q_base = ring_sector_qpos(k, e_base, f_base, rad_blend, lat_blend, retract, adj_sign = int(choose_sign(rng, idx, is_eval)))
        return make_state(goal + polar(r, math.radians(angle_deg)), goal, goal_yaw + dyaw, goal_yaw, q_base + table_safe_qpos_noise(rng, 0.005 + 0.006 * over))

    return fn

# stage 2 track 3: 45cm reach from DEFAULT_QPOS + 360-deg wrong-side circumnavigation

def circumnav_reach_track(w):
    w = float(np.clip(w, 0., 1.))

    def fn(rng, idx = 0, step = 0):
        is_eval = (int(step) == 0)
        ww = float(np.clip(w + rng.normal(0., 0.02), 0., 1.))
        alpha_def = float(np.clip(0.35 + 2.70 * ww, 0.35, 1.))
        u = float(np.clip((ww - 0.24) / 0.58, 0., 1.))
        over = float(np.clip((ww - 0.82) / 0.18, 0., 1.))

        az_lo = math.radians(75. * u + 35. * over)
        az_hi = math.radians(min(180., 24. + 60. * min(1., ww / 0.24) + 96. * (u ** 0.85)))
        r_lo = 0.015 + 0.027 * u + 0.020 * over
        r_hi = 0.021 + 0.015 * min(1., ww / 0.24) + 0.098 * (u ** 1.15) + 0.036 * over
        max_dyaw = 0.09 + 0.19 * min(1., ww / 0.24) + 2.24 * (u ** 1.35) + (math.pi - 2.52) * over

        phi = PHI_DEFAULT_RAY + choose_sign(rng, idx, is_eval) * float(rng.uniform(az_lo, az_hi))
        tee = np.clip(
            GOAL_POS + polar(float(rng.uniform(r_lo, r_hi)), phi),
            GOAL_POS + np.array([-0.10 - 0.018 * over, -0.10 - 0.035 * over], dtype = np.float32),
            GOAL_POS + np.array([+0.10 + 0.030 * over, +0.195], dtype = np.float32)
        )
        dyaw = choose_sign(rng, int(idx) // 2, is_eval) * float(rng.uniform(0.85 * over, max_dyaw))

        if alpha_def >= 0.999:
            q_base, noise_std = DEFAULT_QPOS, QPOS_NOISE + 0.018 * over
        else:
            k = round((math.degrees(phi) % 360.) / 30.) % 12
            k_safe = 4 if k in (5, 6) else k
            q_base = retract_ring_qpos(FRONTIER_RING_STATES[k_safe, 6:13], k_safe, alpha_def)
            noise_std = lerp(0.006, QPOS_NOISE, alpha_def)

        return make_state(tee, GOAL_POS, GOAL_YAW + dyaw, GOAL_YAW, q_base + qpos_noise(rng, noise_std))

    return fn

# stage 3a: monotonic spawn-box and yaw expansion from aligned DEFAULT_QPOS sub-box to native_config

def native_bridge_track(w):
    w = float(np.clip(w, 0., 1.))
    f4 = FRONTIER_RING_STATES[4, 6:13]

    def fn(rng, idx = 0, step = 0):
        ww = float(np.clip(w + rng.normal(0., 0.02), 0., 1.))
        s = ww ** 1.20
        tee = GOAL_POS + np.array([
            rng.uniform(lerp(-0.065, -0.100, s), lerp(0.005, 0.100, s)),
            rng.uniform(lerp(-0.005, -0.100, s), lerp(0.105, 0.200, s))
        ], dtype = np.float32)
        dyaw = choose_sign(rng) * float(rng.uniform(0., 0.16 + (math.pi - 0.16) * (ww ** 1.35)))
        q_base = lerp(f4, DEFAULT_QPOS, 0.45 + 0.55 * (ww / 0.18)) if ww < 0.18 else DEFAULT_QPOS
        noise_std = lerp(0.010, QPOS_NOISE, ww / 0.18) if ww < 0.18 else QPOS_NOISE
        return make_state(tee, GOAL_POS, GOAL_YAW + dyaw, GOAL_YAW, q_base + qpos_noise(rng, noise_std))

    return fn

# stage 3c: beyond-native over-training (wider box, min radial/yaw floor, occluded-sector bias)

def beyond_native_track(w):
    w = float(np.clip(w, 0., 1.))

    def fn(rng, idx = 0, step = 0):
        wb = float(np.clip(w + rng.normal(0., 0.02), 0., 1.))
        dx_lo, dx_hi = -0.100 - 0.018 * wb, 0.100 + 0.035 * wb
        dy_lo, dy_hi = -0.100 - 0.035 * wb, 0.195

        cur_dx_lo, cur_dy_hi = dx_lo, dy_hi
        if rng.uniform(0., 1.) < 0.50 * wb:
            if int(rng.integers(2)) == 0:
                cur_dx_lo = -0.015
            else:
                cur_dy_hi = 0.015

        dx, dy = float(rng.uniform(cur_dx_lo, dx_hi)), float(rng.uniform(dy_lo, cur_dy_hi))
        offset = np.array([dx, dy], dtype = np.float32)

        cur_r, min_r = float(np.linalg.norm(offset)), 0.045 * wb
        if cur_r < min_r:
            phi = float(rng.uniform(-math.pi, math.pi)) if cur_r < 1e-5 else math.atan2(dy, dx)
            offset = np.clip(polar(min_r, phi), [dx_lo, dy_lo], [dx_hi, dy_hi])

        dyaw = choose_sign(rng) * float(rng.uniform(1.25 * wb, math.pi))
        return make_state(GOAL_POS + offset, GOAL_POS, GOAL_YAW + dyaw, GOAL_YAW, DEFAULT_QPOS + qpos_noise(rng, QPOS_NOISE + 0.022 * wb))

    return fn

# declarative curriculum assembly and sgs frontier scoring

SKILL_TRACKS = (
    frontier_translation_track,
    lever_rotation_track,
    circumnav_reach_track
)

def build_anchor_configs(num_easy = NUM_EASY, num_frontier = NUM_FRONTIER):
    rng = np.random.default_rng(10_203)
    easy = [ring_config(i)(rng, i) for i in range(num_easy)]
    frontier = [FRONTIER_RING_STATES[ANCHOR_FRONTIER_INDICES[j % len(ANCHOR_FRONTIER_INDICES)]].copy() for j in range(num_frontier)]
    return [*easy, *frontier]

def build_curriculum_configs(
    num_easy = NUM_EASY,
    ladder_size = LADDER_SIZE,
    native_bank_size = NATIVE_BANK_SIZE,
    skill_tracks = SKILL_TRACKS
):
    foothold_cfgs = [ring_config(i) for i in range(num_easy)]
    num_prec = max(0, min(ladder_size, FOOTHOLD_SIZE) - len(foothold_cfgs))
    prec_cfgs = [precision_dock_track((i + 1) / max(num_prec, 1)) for i in range(num_prec)]

    n_interleaved = max(0, ladder_size - len(foothold_cfgs) - len(prec_cfgs))
    n_tracks = len(skill_tracks)
    n_levels = max(1, math.ceil(n_interleaved / max(n_tracks, 1)))
    ladder_cfgs = [
        skill_tracks[j % n_tracks](((j // n_tracks + 1) / n_levels) ** 1.05)
        for j in range(n_interleaved)
    ]

    num_bridge, num_pure = round(native_bank_size * 0.45), round(native_bank_size * 0.20)
    num_beyond = max(0, native_bank_size - num_bridge - num_pure)
    native_cfgs = [
        *[native_bridge_track((j + 1) / max(num_bridge, 1)) for j in range(num_bridge)],
        *([native_config] * num_pure),
        *[beyond_native_track((j + 1) / max(num_beyond, 1)) for j in range(num_beyond)]
    ]

    return [*foothold_cfgs, *prec_cfgs, *ladder_cfgs, *native_cfgs]

build_expanded_configs = build_curriculum_configs

def materialize_config(c, rng, idx = 0, step = 0):
    return c(rng, idx, step) if callable(c) else np.asarray(c, dtype = np.float32).copy()

# highest rung before a sustained (> max_gap) wall below `threshold` under count-weighted smoothing

def contiguous_frontier_index(p_sub: Tensor, c_sub: Tensor, threshold = 0.22, min_count = 4, kernel = 5, max_gap = 3, fallback = 1):
    n = len(p_sub)
    if n == 0:
        return 0

    p, c = torch.nan_to_num(p_sub.float(), nan = 0.), torch.nan_to_num(c_sub.float(), nan = 0.)
    half = max(1, kernel // 2)
    win = 2 * half + 1

    w_c = c.clamp(0., 24.)
    tot_c = F.pad(w_c, (half, half)).unfold(0, win, 1).sum(dim = -1)
    sm_p = F.pad(w_c * p, (half, half)).unfold(0, win, 1).sum(dim = -1) / tot_c.clamp(min = 1e-8)
    sm_p = torch.where(tot_c > 0., sm_p, torch.zeros_like(sm_p))

    visited = (c >= min_count) | ((c >= 2.) & (tot_c >= 2.5 * min_count))
    best_k, gap = -1, 0
    for i, ok in enumerate((visited & (sm_p >= threshold)).tolist()):
        if ok:
            best_k, gap = i, 0
        else:
            gap += 1
            if gap > max_gap:
                break

    return best_k if best_k >= 0 else min(max(fallback, 0), n - 1)

def _track_weights(p_sub: Tensor, c_sub: Tensor, target = 0.40, kappa = 4., eps = 1e-4, unvisited_weight = 0.35, fallback = 1, retain_floor = 0.18):
    p_sub = torch.nan_to_num(p_sub, nan = 0.)
    n = len(p_sub)
    if n == 0:
        return p_sub, 0

    k_sub = contiguous_frontier_index(p_sub, c_sub, fallback = min(fallback, n - 1))
    idx = arange(n, device = p_sub.device, dtype = p_sub.dtype)
    kf = float(k_sub)

    bell = torch.where(c_sub >= 2., torch.exp(-kappa * (p_sub - target) ** 2) + eps, torch.full_like(p_sub, unvisited_weight))
    probe_decay = torch.rsqrt(1. + c_sub.clamp(min = 0.) / 5.)
    frontier_env = torch.exp(-0.5 * ((idx - kf) / 3.4) ** 2) + 0.55 * torch.exp(-0.5 * ((idx - (kf + 2.4)) / 2.1) ** 2) * probe_decay

    retain_decay = torch.exp(-(kf - idx).clamp(min = 0.) / 15.).clamp(min = retain_floor)
    retain = (idx <= kf).to(p_sub.dtype) * (0.14 + 0.45 * (0.88 - p_sub).clamp(min = 0.04)) * retain_decay
    w = (bell * frontier_env + retain) * torch.exp(-(idx - (kf + 5.)).clamp(min = 0.) / 2.)
    return w / w.sum().clamp(min = 1e-8), k_sub

def compute_frontier_k(p: Tensor, counts: Tensor, threshold = 0.22, min_count = 4, kernel = 5, fallback = 4, ladder_size = LADDER_SIZE, num_tracks = len(SKILL_TRACKS)):
    p = torch.nan_to_num(p, nan = 0.)
    n_total, n_main = len(p), min(len(p), ladder_size)
    n_foot = min(FOOTHOLD_SIZE, n_main)
    k_a = contiguous_frontier_index(p[:n_foot], counts[:n_foot], threshold, min_count, kernel = 3, max_gap = 2, fallback = fallback)

    if n_main <= n_foot or num_tracks <= 0:
        return k_a

    slices = [arange(n_foot + t, n_main, num_tracks, device = p.device) for t in range(num_tracks)]
    track_ks = [contiguous_frontier_index(p[idx], counts[idx], threshold, min_count, kernel, max_gap = 3, fallback = 0) for idx in slices]

    if not bool((counts[n_foot:n_main] >= min_count).any()) or max(track_ks) == 0:
        return k_a

    progs = [float(k_t) / max(len(idx) - 1, 1) for k_t, idx in zip(track_ks, slices)]
    prog_mean, prog_min = sum(progs) / len(progs), min(progs)

    if n_total > n_main and prog_min >= 0.68 and prog_mean >= 0.74:
        k_d = contiguous_frontier_index(p[n_main:], counts[n_main:], threshold, min_count, kernel, max_gap = 3, fallback = 0)
        if float(counts[n_main + k_d].item()) >= min_count:
            return n_main + k_d

    return max(n_foot, min(n_main - 1, n_foot + round(min(prog_mean, prog_min + 0.24) * (n_main - 1 - n_foot))))

def curriculum_score_fn(p: Tensor, counts: Tensor, target = 0.40, kappa = 4., eps = 1e-4, unvisited_weight = 0.35, ladder_size = LADDER_SIZE, num_tracks = len(SKILL_TRACKS)):
    p = torch.nan_to_num(p, nan = 0.)
    n_total, n_main = len(p), min(len(p), ladder_size)
    n_foot = min(FOOTHOLD_SIZE, n_main)
    out = torch.zeros_like(p)

    # stage 1 (~8%): foothold and precision docking retention

    p_a, c_a = p[:n_foot], counts[:n_foot]
    w_a = torch.where(
        c_a > 0,
        (0.92 - p_a).clamp(min = 0.08) + 0.35 * torch.exp(-kappa * (p_a - 0.55) ** 2),
        torch.full_like(p_a, unvisited_weight)
    )
    w_a = w_a / w_a.sum().clamp(min = 1e-8)

    if n_main <= n_foot or num_tracks <= 0:
        return w_a

    # stage 2: interleaved se(2) skill tracks with automatic lag-weighting

    slices = [arange(n_foot + t, n_main, num_tracks, device = p.device) for t in range(num_tracks)]
    track_wk = [_track_weights(p[idx], counts[idx], target, kappa, eps, unvisited_weight) for idx in slices]
    progs = [float(k_t) / max(len(idx) - 1, 1) for (_, k_t), idx in zip(track_wk, slices)]
    gains = [1. + 0.85 * (max(progs) - pr) for pr in progs]

    # stage 3 (24% -> 48%): native bridge -> pure native -> beyond native

    share_a = 0.08
    if n_total > n_main:
        p_d, c_d = p[n_main:], counts[n_main:]
        w_d, k_d = _track_weights(p_d, c_d, target, kappa, eps, unvisited_weight, retain_floor = 0.30)
        br_end = round(len(p_d) * 0.45)

        if len(p_d) > br_end:
            w_tail = torch.zeros_like(p_d)
            w_tail[br_end:] = torch.where(c_d[br_end:] >= 2., (0.88 - p_d[br_end:]).clamp(min = 0.12), torch.full_like(p_d[br_end:], unvisited_weight))
            w_d = 0.75 * w_d + 0.25 * (w_tail / w_tail.sum().clamp(min = 1e-8))

        share_d = float(np.clip(0.24 + 0.14 * (sum(progs) / len(progs)) + 0.10 * (k_d / max(len(p_d) - 1, 1)), 0.24, 0.48))
        out[n_main:] = share_d * (w_d / w_d.sum().clamp(min = 1e-8))
    else:
        share_d = 0.

    out[:n_foot] = share_a * w_a
    rem, gain_sum = 1. - share_a - share_d, sum(gains)
    for idx, (w_t, _), g in zip(slices, track_wk, gains):
        if len(idx) > 0:
            out[idx] = (rem * g / gain_sum) * w_t

    return out / out.sum().clamp(min = 1e-8)

frontier_ladder_score_fn = curriculum_score_fn

def make_sampler(configs, seed = 0, progression_path = None, target = 0.40, kappa = 4., eps = 1e-4, window = 30, unvisited_weight = 0.35, uniform_prob = 0.015, score_fn = curriculum_score_fn):
    return SGSSampler(
        configs = configs, target = target, kappa = kappa, eps = eps,
        window = window, unvisited_weight = unvisited_weight, uniform_prob = uniform_prob,
        score_fn = score_fn, log_every = None, log_path = progression_path, seed = seed
    )

# held-out evaluation and video rendering

@torch.no_grad()
def evaluate_heldout(trainer, bank_envs, eval_states, max_steps = 240, seed_base = 0, deterministic = True):
    n, m = len(eval_states), len(bank_envs)
    succ = np.zeros(n, dtype = np.float32)
    max_cov = np.zeros(n, dtype = np.float32)

    with torch.random.fork_rng():
        torch.manual_seed(int(seed_base + 1))

        for start in range(0, n, m):
            batch = eval_states[start:start + m]
            b = len(batch)
            obs = np.stack([
                bank_envs[j].reset(seed = int(seed_base + start + j), options = dict(reset_to_state = batch[j]))[0]
                for j in range(b)
            ]).astype(np.float32)

            cur_s = transform_pusht3d(obs)
            prev_s = cur_s.copy()
            prev_a = np.zeros((b, DIM_ACTION), dtype = np.float32)
            done = np.zeros(b, dtype = bool)
            goal_t = torch.from_numpy(extract_goal_pusht3d(obs)).float().to(trainer.device)

            for _ in range(max_steps):
                if done.all():
                    break

                state_t = torch.from_numpy(make_aug_state_3d(cur_s, prev_s, prev_a)).float().to(trainer.device)
                act_t, _ = trainer.act(state_t, goal_t, deterministic = deterministic)
                act = act_t.clamp(-1., 1.).cpu().numpy()

                for j in np.where(~done)[0]:
                    j = int(j)
                    obs[j], _, term, trunc, info = bank_envs[j].step(act[j])
                    prev_s[j], prev_a[j], cur_s[j] = cur_s[j], act[j], transform_pusht3d(obs[j])
                    max_cov[start + j] = max(max_cov[start + j], float(info.get('coverage', 0.)))
                    if term or trunc:
                        succ[start + j], done[j] = float(term), True

    return dict(success_rate = float(succ.mean()), max_cov = float(max_cov.mean()), per_state = [float(x) for x in succ])

@torch.no_grad()
def render_heldout_grid(trainer, render_env, eval_states, out_path, max_steps = 240, deterministic = True, grid_cols = 4, fps = 12):
    n = min(len(eval_states), 16)
    sel = [eval_states[i] for i in np.linspace(0, len(eval_states) - 1, n, dtype = int)]
    all_frames, succ = [], []

    def snap_panel(green = False):
        raw = rearrange(torch.from_numpy(render_env.render()).float(), 'h w c -> 1 c h w')
        img = rearrange(F.interpolate(raw, size = (192, 192), mode = 'area'), '1 c h w -> h w c').byte().numpy().copy()
        if green:
            img[:5, :] = img[-5:, :] = img[:, :5] = img[:, -5:] = [40, 220, 80]
        return img

    with torch.random.fork_rng():
        torch.manual_seed(0)

        for i, st in enumerate(sel):
            obs_i, _ = render_env.reset(seed = 80_000 + i, options = dict(reset_to_state = st))
            goal_t = torch.from_numpy(extract_goal_pusht3d(obs_i)[None]).float().to(trainer.device)
            cur_s = transform_pusht3d(obs_i)
            prev_s = cur_s.copy()
            prev_a = np.zeros((DIM_ACTION,), dtype = np.float32)
            ep_frames, solved, frozen = [], False, None

            for step_i in range(max_steps):
                if divisible_by(step_i, 4) or step_i == max_steps - 1:
                    ep_frames.append(frozen if exists(frozen) else snap_panel(green = solved))

                if not solved:
                    state_t = torch.from_numpy(make_aug_state_3d(cur_s, prev_s, prev_a)[None]).float().to(trainer.device)
                    act_t, _ = trainer.act(state_t, goal_t, deterministic = deterministic)
                    act = act_t[0].clamp(-1., 1.).cpu().numpy()
                    obs_i, _, term, trunc, _ = render_env.step(act)
                    prev_s, prev_a, cur_s = cur_s, act, transform_pusht3d(obs_i)
                    if term or trunc:
                        solved = bool(term)
                        frozen = snap_panel(green = solved)

            succ.append(float(solved))
            all_frames.append(ep_frames)

    cols = next(c for c in range(min(grid_cols, n), 0, -1) if divisible_by(n, c))
    frames = [
        rearrange(
            np.stack([all_frames[i][min(t, len(all_frames[i]) - 1)] for i in range(n)]),
            '(r c) h w ch -> (r h) (c w) ch',
            c = cols
        )
        for t in range(max(len(f) for f in all_frames))
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    with imageio.get_writer(str(out_path), fps = fps, codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1) as writer:
        for f in frames:
            writer.append_data(f[..., :3])

    imageio.mimsave(str(out_path.with_suffix('.gif')), [f[..., :3] for f in frames], fps = fps, loop = 0)
    return str(out_path), float(np.mean(succ))

# main

def main(
    num_envs: int = 16,
    rollout_len: int = 128,
    num_updates: int = 50_000,
    max_steps: int = 240,
    actor_lr: float = 3e-4,
    actor_lr_warmup: int = 0,
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
    critic_discount: float = 0.92,
    critic_negative_buffer_size: int = 8192,
    critic_distance_scale: float = 10.,
    critic_time_weight: float = 0.5,
    critic_sep_weight: float = 0.25,
    value_hidden: int = 256,
    value_depth: int = 12,
    value_residual_every: int = 2,
    value_num_bins: int = 256,
    value_epochs: int = 6,
    value_minibatch_size: int = 256,
    env_reward_weight: float = 0.5,
    success_bonus: float = 1.,
    self_goal_cos_weight: float = 0.15,
    num_easy: int = NUM_EASY,
    num_frontier: int = NUM_FRONTIER,
    ladder_size: int = LADDER_SIZE,
    native_bank_size: int = NATIVE_BANK_SIZE,
    num_heldout_tiers: int = NUM_HELDOUT_TIERS,
    num_heldout_native: int = NUM_HELDOUT_NATIVE,
    sgs_target: float = 0.40,
    sgs_kappa: float = 4.,
    sgs_eps: float = 1e-4,
    sgs_window: int = 30,
    sgs_unvisited_weight: float = 0.35,
    sgs_uniform_prob: float = 0.015,
    exploration_goal_prob: float = 0.10,
    explore_random_agent_prob: float = 0.50,
    goal_buffer_size: int = 8192,
    goal_buffer_batch: int = 256,
    recent_succ_window: int = 200,
    eval_every: int = 10,
    log_every: int = 5,
    video_every: int = 20,
    video_fps: int = 12,
    video_grid_cols: int = 4,
    checkpoint_folder: str = './checkpoints-pusht3d-sgs-crl',
    video_folder: str = './videos-pusht3d-sgs-crl',
    artifact_folder: str | None = None,
    log_path: str = './logs-pusht3d-sgs-crl.jsonl',
    progression_path: str = './sgs-progression-pusht3d-crl.csv',
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
        f"[PushT-3D SGS+CRL] max_steps={max_steps} | finite_time_gae={finite_time_gae} | "
        f"videos -> {Path(video_folder).resolve()} | checkpoints -> {Path(checkpoint_folder).resolve()}",
        flush = True
    )

    anchor_configs = build_anchor_configs(num_easy = num_easy, num_frontier = num_frontier)
    configs = build_curriculum_configs(num_easy = num_easy, ladder_size = ladder_size, native_bank_size = native_bank_size)
    sampler = make_sampler(configs, seed, progression_path, sgs_target, sgs_kappa, sgs_eps, sgs_window, sgs_unvisited_weight, sgs_uniform_prob)

    success_fn = lambda _o, _r, term, _trunc, info: bool(term or info.get('success', False))
    envs = VecEnv([
        (lambda i = idx: SuccessGuidedSamplingWrapper(
            make_pusht3d(max_steps = max_steps, seed = seed + i),
            sampler,
            success_fn = success_fn
        ))
        for idx in range(num_envs)
    ])

    eval_rng = np.random.default_rng(77_777)
    heldout_anchor = list(anchor_configs)
    heldout_tiers = [materialize_config(configs[int(idx)], eval_rng, i) for i, idx in enumerate(np.linspace(0, ladder_size - 1, num_heldout_tiers, dtype = int))]
    heldout_native = [native_config(np.random.default_rng(90_000 + i), i) for i in range(num_heldout_native)]

    eval_envs = [make_pusht3d(max_steps = max_steps, seed = 10_000 + i) for i in range(8)]
    render_env = make_pusht3d(max_steps = max_steps, seed = 80_000, render_mode = 'rgb_array')

    obs_raw, _ = envs.reset(seed = seed)
    cur_single_obs = transform_pusht3d(obs_raw)
    prev_single_obs = cur_single_obs.copy()
    prev_act = np.zeros((num_envs, DIM_ACTION), dtype = np.float32)

    goal_rng = np.random.default_rng(seed + 135_791)
    goal_buffer = deque([
        config_to_goal_3d(materialize_config(configs[g_i % len(configs)], goal_rng, g_i))
        for g_i in range(goal_buffer_batch)
    ], maxlen = goal_buffer_size)

    current_goals = np.zeros((num_envs, DIM_GOAL), dtype = np.float32)
    is_explore_goal = np.zeros(num_envs, dtype = bool)

    def assign_env_goal(i, raw_obs_i):
        if len(goal_buffer) > 0 and np.random.rand() < exploration_goal_prob:
            current_goals[i] = goal_buffer[np.random.randint(len(goal_buffer))].copy()
            is_explore_goal[i] = True
            envs.envs[i].current_config_ids[0] = -1

            if np.random.rand() < explore_random_agent_prob:
                base_env = unwrap_pusht3d(envs.envs[i])
                u = base_env.raw.unwrapped
                q_exp = torch.as_tensor(sample_exploration_qpos(goal_rng)[None], dtype = torch.float32, device = getattr(u, 'device', torch.device('cpu')))
                u.agent.reset(q_exp)
                raw_obs_i = base_env._pack_obs(u.get_obs())
                cur_single_obs[i] = prev_single_obs[i] = transform_pusht3d(raw_obs_i)

            raw_obs_i = set_simulator_goal_from_vec_3d(envs.envs[i], current_goals[i], raw_obs_i)
        else:
            current_goals[i], is_explore_goal[i] = extract_goal_pusht3d(raw_obs_i), False

        return raw_obs_i

    for i in range(num_envs):
        obs_raw[i] = assign_env_goal(i, obs_raw[i])
    obs = make_aug_state_3d(cur_single_obs, prev_single_obs, prev_act)

    actor = ResidualBetaActor(DIM_STATE, DIM_GOAL, DIM_ACTION, actor_hidden, actor_depth, residual_every = actor_residual_every).to(device)
    state_encoder = ResidualEncoder(DIM_STATE, critic_hidden, critic_depth, critic_embed_dim, residual_every = critic_residual_every).to(device)
    goal_encoder = ResidualEncoder(DIM_GOAL, critic_hidden, critic_depth, critic_embed_dim, residual_every = critic_residual_every).to(device)
    value = ResidualEncoder(DIM_STATE + DIM_GOAL, value_hidden, value_depth, value_num_bins, residual_every = value_residual_every).to(device)
    env_value = ResidualEncoder(DIM_STATE + DIM_GOAL, value_hidden, value_depth, value_num_bins, residual_every = value_residual_every).to(device)

    contrastive_learn = ContrastiveLearning(
        use_euclidean = True, l2norm_embed = True, learned_temp = False,
        scale = critic_distance_scale, symmetric = True,
        time_weight = critic_time_weight, sep_weight = critic_sep_weight, goal_eps = 0.05
    )
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
        frac = 1. if actor_lr_warmup <= 0 else min(1., (update - start_update + 1) / actor_lr_warmup)
        for g in trainer.actor_opt.param_groups:
            g['lr'] = actor_lr * frac

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

            next_single_obs = transform_pusht3d(next_obs_raw)
            final_single_obs = np.stack([
                transform_pusht3d(info['final_obs']) if exists(info['final_obs']) else next_single_obs[i]
                for i, info in enumerate(infos)
            ])

            states[:, step] = obs
            next_states[:, step] = make_aug_state_3d(final_single_obs, cur_single_obs, action_np)
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
                    cur_single_obs[i] = prev_single_obs[i] = transform_pusht3d(next_obs_raw[i])
                    prev_act[i] = 0.
                else:
                    prev_single_obs[i], prev_act[i], cur_single_obs[i] = cur_single_obs[i], action_np[i], next_single_obs[i]

            obs = make_aug_state_3d(cur_single_obs, prev_single_obs, prev_act)

        flat_next_goals = rearrange(next_states, 'n t d -> (n t) d')[:, :DIM_GOAL]
        for g_vec in flat_next_goals[np.random.choice(len(flat_next_goals), size = min(goal_buffer_batch, len(flat_next_goals)), replace = False)]:
            goal_buffer.append(g_vec.copy())

        metrics = trainer.update(
            states = states, goals = goals, actions = actions, old_log_probs = log_probs,
            dones = terminateds | truncateds, next_states = next_states,
            terminated = terminateds, truncated = truncateds, success = terminateds,
            base_rewards = np.clip(env_rewards, 0., 1.), env_reward_weight = env_reward_weight
        )

        k_star = compute_frontier_k(sampler.success_rates, sampler.counts, ladder_size = ladder_size)
        train_succ = float(np.mean(recent_succ)) if recent_succ else 0.
        elapsed = time.time() - t0

        row = dict(
            update = update, env_steps = update * num_envs * rollout_len, episodes = num_episodes,
            train_succ = train_succ, k_star = int(k_star), finite_time_gae = bool(trainer.finite_time_gae),
            actor_depth = actor_depth, critic_depth = critic_depth, elapsed_sec = elapsed, **metrics
        )

        if divisible_by(update, eval_every) or update == start_update or update == num_updates:
            anchor_g = evaluate_heldout(trainer, eval_envs, heldout_anchor, max_steps, 10_000, True)
            anchor_s = evaluate_heldout(trainer, eval_envs, heldout_anchor, max_steps, 20_000, False)
            tiers_g = evaluate_heldout(trainer, eval_envs, heldout_tiers, max_steps, 30_000, True)
            tiers_s = evaluate_heldout(trainer, eval_envs, heldout_tiers, max_steps, 40_000, False)
            native_g = evaluate_heldout(trainer, eval_envs, heldout_native, max_steps, 50_000, True)
            native_s = evaluate_heldout(trainer, eval_envs, heldout_native, max_steps, 60_000, False)

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
                v_tiers_out = Path(video_folder) / f'pusht3d_heldout_tiers_upd{update:04d}.mp4'
                v_native_out = Path(video_folder) / f'pusht3d_heldout_native_upd{update:04d}.mp4'

                v_tiers, _ = render_heldout_grid(trainer, render_env, heldout_tiers[:16], v_tiers_out, max_steps, use_det, video_grid_cols, video_fps)
                v_native, _ = render_heldout_grid(trainer, render_env, heldout_native[:16], v_native_out, max_steps, use_det, video_grid_cols, video_fps)
                print(f"  -> saved heldout video trajectories (.mp4 & .gif): {v_tiers} , {v_native}", flush = True)

                if exists(artifact_folder) and Path(artifact_folder).exists():
                    for src, name in ((v_tiers, f'pusht3d_heldout_tiers_upd{update:04d}'), (v_native, f'pusht3d_heldout_native_upd{update:04d}')):
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
    for e in eval_envs:
        e.close()
    render_env.close()

if __name__ == '__main__':
    Fire(main)
