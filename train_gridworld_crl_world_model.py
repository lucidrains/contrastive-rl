# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "fire",
#   "imageio",
#   "imageio-ffmpeg",
#   "numpy",
#   "pillow",
#   "torch",
#   "x-mlps-pytorch",
# ]
#
# [tool.uv.sources]
# contrastive-rl-pytorch = { path = "." }
# ///

# gridworld demo for the contrastive world model
#
# Q(s, a, g) = V(s', g) - Wang et al. https://arxiv.org/abs/2304.01203
# residual 3 layer mlp delta update - Teoh et al. https://arxiv.org/abs/2511.05963
# SPR - https://arxiv.org/abs/2007.05929
# LeWM - https://arxiv.org/abs/2603.19312
# C-Slerp - https://arxiv.org/abs/2610.20409

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

from fire import Fire

from contrastive_rl_pytorch import (
    ContrastiveLearning,
    ContrastiveWorldModel,
    WorldModelPlanner,
    sample_truncated_geometric
)

# config - the demo is deliberately small and fast

SIZE = 5
NUM_WALLS = 5
WALL_SEED = 0
MIN_DETOUR = 2

DIM_EMBED = 8
HIDDEN = 128
DEPTH = 2

HORIZON = 3
DISCOUNT = 0.85
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
TRAIN_STEPS = 2000
SEED = 2

MAX_EVAL_STEPS = 24
NUM_SAMPLES = 128
NUM_ELITES = 16
NUM_ITERS = 6
INTERMEDIATE_WEIGHT = 0.5
NUM_TRIALS = 10

# loss coefficients

WM_COEF = 100.
CRL_COEF = 0.1
VALUE_COEF = 5.

ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# gridworld

class GridWorld:
    def __init__(self, size = SIZE, num_walls = NUM_WALLS, wall_seed = WALL_SEED, min_detour = MIN_DETOUR):
        self.size = size
        self.start_cell = (0, 0)
        self.goal_cell = (size - 1, size - 1)

        all_cells = [(row, col) for row in range(size) for col in range(size)]
        candidates = [cell for cell in all_cells if cell not in (self.start_cell, self.goal_cell)]

        rng = np.random.default_rng(wall_seed)
        manhattan = 2 * (size - 1)

        # sample random walls that keep every free cell connected and force a detour

        for _ in range(10_000):
            chosen = rng.choice(len(candidates), size = min(num_walls, len(candidates)), replace = False)
            walls = {candidates[i] for i in chosen}

            reachable = self._reachable(self.start_cell, walls)
            if len(reachable) != len(all_cells) - len(walls):
                continue

            shortest = self._shortest_path(self.start_cell, self.goal_cell, walls)
            if num_walls == 0 or (shortest is not None and shortest >= manhattan + min_detour):
                break
        else:
            raise RuntimeError('could not sample a connected wall layout')

        self.walls = walls
        self.shortest_path_length = shortest
        self.grid = [cell for cell in all_cells if cell not in walls]
        self.index = {cell: i for i, cell in enumerate(self.grid)}
        self.start = self.index[self.start_cell]
        self.goal = self.index[self.goal_cell]

    @property
    def num_states(self):
        return len(self.grid)

    def _reachable(self, start_cell, walls):
        queue = deque([start_cell])
        seen = {start_cell}

        while queue:
            row, col = queue.popleft()

            for drow, dcol in ACTION_DELTAS:
                cell = (row + drow, col + dcol)
                if not (0 <= cell[0] < self.size and 0 <= cell[1] < self.size):
                    continue
                if cell in walls or cell in seen:
                    continue

                seen.add(cell)
                queue.append(cell)

        return seen

    def _shortest_path(self, start_cell, goal_cell, walls):
        queue = deque([(start_cell, 0)])
        seen = {start_cell}

        while queue:
            cell, distance = queue.popleft()

            if cell == goal_cell:
                return distance

            row, col = cell
            for drow, dcol in ACTION_DELTAS:
                next_cell = (row + drow, col + dcol)
                if not (0 <= next_cell[0] < self.size and 0 <= next_cell[1] < self.size):
                    continue
                if next_cell in walls or next_cell in seen:
                    continue

                seen.add(next_cell)
                queue.append((next_cell, distance + 1))

        return None

    def shortest_path_cells(self, start_cell, goal_cell):
        queue = deque([start_cell])
        parent = {start_cell: None}

        while queue:
            cell = queue.popleft()

            if cell == goal_cell:
                path = []
                while cell is not None:
                    path.append(cell)
                    cell = parent[cell]
                return path[::-1]

            row, col = cell
            for drow, dcol in ACTION_DELTAS:
                next_cell = (row + drow, col + dcol)
                if not (0 <= next_cell[0] < self.size and 0 <= next_cell[1] < self.size):
                    continue
                if next_cell in self.walls or next_cell in parent:
                    continue

                parent[next_cell] = cell
                queue.append(next_cell)

        return None

    def action_between(self, cell, next_cell):
        delta = (next_cell[0] - cell[0], next_cell[1] - cell[1])
        return ACTION_DELTAS.index(delta)

    def all_pairs_distances(self):
        distances = torch.zeros((self.num_states, self.num_states))

        for source_index, source_cell in enumerate(self.grid):
            queue = deque([(source_cell, 0)])
            seen = {source_cell}

            while queue:
                cell, distance = queue.popleft()
                distances[source_index, self.index[cell]] = distance

                row, col = cell
                for drow, dcol in ACTION_DELTAS:
                    next_cell = (row + drow, col + dcol)
                    if not (0 <= next_cell[0] < self.size and 0 <= next_cell[1] < self.size):
                        continue
                    if next_cell in self.walls or next_cell in seen:
                        continue

                    seen.add(next_cell)
                    queue.append((next_cell, distance + 1))

        return distances

    def reset(self):
        return self.start

    def step(self, state, action):
        row, col = self.grid[state]
        drow, dcol = ACTION_DELTAS[action]
        next_cell = (row + drow, col + dcol)

        if next_cell not in self.index:
            return state

        return self.index[next_cell]

    def one_hot(self, state):
        return F.one_hot(torch.as_tensor(state).long(), self.num_states).float()

    def all_one_hot(self):
        return torch.eye(self.num_states)

# data

def goal_directed_trajectories(env, num_episodes = 600, length = 64, seed = SEED):
    rng = np.random.default_rng(seed)

    states = np.empty((num_episodes, length + 1), dtype = np.int64)
    actions = np.zeros((num_episodes, length), dtype = np.int64)

    for i in range(num_episodes):
        current_cell = env.grid[int(rng.integers(0, len(env.grid)))]
        states[i, 0] = env.index[current_cell]
        time = 0

        while time < length:
            goal_cell = env.goal_cell if rng.random() < 0.5 else env.grid[int(rng.integers(0, len(env.grid)))]
            path = env.shortest_path_cells(current_cell, goal_cell)

            if path is None or len(path) < 2:
                current_cell = env.grid[int(rng.integers(0, len(env.grid)))]
                continue

            for cell in path[1:]:
                if time >= length:
                    break

                actions[i, time] = env.action_between(current_cell, cell)
                current_cell = cell
                states[i, time + 1] = env.index[current_cell]
                time += 1

    return torch.from_numpy(states), torch.from_numpy(actions)

def all_transitions(env):
    states, actions, next_states = [], [], []

    for state in range(env.num_states):
        for action in range(len(ACTION_DELTAS)):
            states.append(state)
            actions.append(action)
            next_states.append(env.step(state, action))

    return tuple(torch.tensor(x) for x in (states, actions, next_states))

# model

def make_model(env):
    state_encoder = nn.Linear(env.num_states, DIM_EMBED, bias = False)

    return ContrastiveWorldModel(
        state_encoder,
        action_dim = len(ACTION_DELTAS),
        dim_embed = DIM_EMBED,
        contrastive_learn = ContrastiveLearning(l2norm_embed = True),
        discrete_action = True,
        dim_hidden = HIDDEN,
        depth = DEPTH
    )

def make_planner(model, use_c_slerp = False):
    return WorldModelPlanner(
        model,
        horizon = HORIZON,
        num_samples = NUM_SAMPLES,
        num_elites = NUM_ELITES,
        num_iters = NUM_ITERS,
        discrete_action = True,
        num_actions = len(ACTION_DELTAS),
        intermediate_weight = INTERMEDIATE_WEIGHT,
        use_c_slerp = use_c_slerp,
        c_slerp_kwargs = dict(num_waypoints = 3, alpha = 0., threshold = 0.)
    )

# training

def train(
    model,
    env,
    train_steps = TRAIN_STEPS,
    batch_size = BATCH_SIZE,
    horizon = HORIZON,
    discount = DISCOUNT,
    learning_rate = LEARNING_RATE,
    seed = SEED,
    checkpoints = (),
    checkpoint_fn = None,
    log_every = 200
):
    torch.manual_seed(seed)

    trajectories, actions = goal_directed_trajectories(env, seed = seed)
    num_episodes, traj_len = trajectories.shape
    all_state, all_action, all_next_state = all_transitions(env)
    distance_matrix = env.all_pairs_distances()

    optimizer = AdamW(model.parameters(), lr = learning_rate)
    history = []

    if 0 in checkpoints and checkpoint_fn is not None:
        checkpoint_fn(0, model)
        model.train()

    for step in range(1, train_steps + 1):
        episode_indices = torch.randint(0, num_episodes, (batch_size,))
        times = torch.randint(0, traj_len - horizon - 1, (batch_size,))

        offsets = torch.arange(horizon + 1)
        window_states = trajectories[episode_indices[:, None], times[:, None] + offsets]
        window_actions = actions[episode_indices[:, None], times[:, None] + offsets[:-1]]

        state_indices = window_states[:, 0]
        state = F.one_hot(state_indices, env.num_states).float()

        goal_times = times + sample_truncated_geometric(traj_len - 1 - times, discount)
        goal_indices = trajectories[episode_indices, goal_times]
        goal = F.one_hot(goal_indices, env.num_states).float()

        window_states = F.one_hot(window_states, env.num_states).float()

        # next-latent prediction, plus full one-step coverage

        wm_loss = model.sequence_world_model_loss(window_states, window_actions)

        transition_indices = torch.randint(0, len(all_state), (batch_size,))
        wm_loss = wm_loss + model.world_model_loss(
            F.one_hot(all_state[transition_indices], env.num_states).float(),
            all_action[transition_indices],
            F.one_hot(all_next_state[transition_indices], env.num_states).float()
        )

        # CRL value

        crl_loss = model.contrastive_loss(state, goal)

        # metric auxiliary: anchor V(s, g) to true graph distance

        value_score = model.value(model.encode_state(state), model.encode_goal(goal))
        distance_target = 10. - 20. * distance_matrix[state_indices, goal_indices] / distance_matrix.max()
        value_loss = F.mse_loss(value_score, distance_target)

        loss = WM_COEF * wm_loss + CRL_COEF * crl_loss + VALUE_COEF * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            history.append(dict(
                step = step,
                loss = loss.item(),
                world_model = wm_loss.item(),
                contrastive = crl_loss.item(),
                value = value_loss.item()
            ))
            print(
                f'step {step:5d} | loss {loss.item():6.3f} | '
                f'wm {wm_loss.item():.4f} | crl {crl_loss.item():.3f} | value {value_loss.item():.3f}',
                flush = True
            )

        if step in checkpoints and checkpoint_fn is not None:
            checkpoint_fn(step, model)

    return history

# planning / evaluation

@torch.no_grad()
def decode_latents(model, env, latents):
    state_embeds = model.encode_state(env.all_one_hot())
    return (latents @ state_embeds.T).argmax(dim = -1).tolist()

@torch.no_grad()
def evaluate_episode(
    model,
    env,
    planner,
    *,
    use_c_slerp = False,
    start = None,
    max_steps = MAX_EVAL_STEPS,
    render_fn = None
):
    state = env.reset() if start is None else start
    goal = env.goal
    frames = []
    steps = 0

    for _ in range(max_steps):
        plan = planner.plan(
            env.one_hot(state),
            env.one_hot(goal),
            use_c_slerp = use_c_slerp
        )

        predicted_states = decode_latents(model, env, plan.latents)
        predicted_states = [state] + predicted_states[1:]
        planned_actions = plan.actions.cpu().tolist()
        waypoint_states = decode_latents(model, env, plan.waypoints) if plan.waypoints is not None else None
        subgoal_state = decode_latents(model, env, plan.subgoal[None])[0] if plan.subgoal is not None else None

        if render_fn is not None:
            frames.append(render_fn(
                model = model,
                env = env,
                state = state,
                goal = goal,
                predicted_states = predicted_states,
                planned_actions = planned_actions,
                waypoint_states = waypoint_states,
                subgoal_state = subgoal_state,
                step = steps,
                success = state == goal,
                planner_name = 'c-slerp' if use_c_slerp else 'cem'
            ))

        if state == goal:
            break

        state = env.step(state, int(plan.actions[0]))
        steps += 1

    if render_fn is not None:
        frames.append(render_fn(
            model = model,
            env = env,
            state = state,
            goal = goal,
            predicted_states = [state],
            step = steps,
            success = state == goal,
            planner_name = 'c-slerp' if use_c_slerp else 'cem'
        ))

    return dict(success = state == goal, steps = steps, frames = frames)

def evaluate_with_seed(seed, fn, *args, **kwargs):
    with torch.random.fork_rng(devices = []):
        torch.manual_seed(seed)
        return fn(*args, **kwargs)

def evaluate_trials(model, env, planner, use_c_slerp, trials = NUM_TRIALS):
    results = [
        evaluate_with_seed(
            10_000 + i,
            evaluate_episode,
            model,
            env,
            planner,
            use_c_slerp = use_c_slerp,
            max_steps = MAX_EVAL_STEPS
        )
        for i in range(trials)
    ]

    successes = [result['success'] for result in results]
    successful_steps = [result['steps'] for result in results if result['success']]

    return dict(
        trials = trials,
        successes = sum(successes),
        success_rate = sum(successes) / trials,
        mean_steps = float(np.mean(successful_steps)) if successful_steps else None,
        steps = [result['steps'] for result in results]
    )

# rendering

def get_font(size = 12):
    for path in (
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

    return ImageFont.load_default()

@torch.no_grad()
def render_frame(
    model,
    env,
    state,
    goal,
    predicted_states,
    planned_actions = None,
    waypoint_states = None,
    subgoal_state = None,
    step = 0,
    success = False,
    planner_name = 'cem',
    label = None,
    cell = 30,
    pad = 14,
    top = 50,
    bottom = 38
):
    width = pad * 2 + env.size * cell
    height = top + env.size * cell + bottom

    image = Image.new('RGB', (width, height), (246, 247, 251))
    draw = ImageDraw.Draw(image)

    font = get_font(12)
    font_bold = get_font(13)

    # value shading

    state_embeds = model.encode_state(env.all_one_hot())
    goal_embed = model.encode_goal(env.one_hot(goal)[None])
    values = model.value(state_embeds, goal_embed.expand_as(state_embeds)).detach().cpu().numpy()

    value_min, value_max = float(values.min()), float(values.max())
    value_span = max(value_max - value_min, 1e-6)

    for index, (row, col) in enumerate(env.grid):
        x0 = pad + col * cell
        y0 = top + row * cell

        normalized = (values[index] - value_min) / value_span
        blue = int(238 - 70 * normalized)
        green = int(244 - 28 * normalized)
        red = int(250 - 8 * normalized)
        draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill = (red, green, blue))

    for row, col in env.walls:
        x0 = pad + col * cell
        y0 = top + row * cell
        draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill = (78, 82, 96))

    for i in range(env.size + 1):
        x = pad + i * cell
        y = top + i * cell
        draw.line((x, top, x, top + env.size * cell), fill = (210, 214, 224), width = 1)
        draw.line((pad, y, pad + env.size * cell, y), fill = (210, 214, 224), width = 1)

    def cell_center(state_index):
        row, col = env.grid[state_index]
        return pad + col * cell + cell / 2, top + row * cell + cell / 2

    # planned action arrows

    if planned_actions is not None:
        cursor = state

        for action in planned_actions:
            next_state = env.step(cursor, int(action))

            if next_state == cursor:
                continue

            x0, y0 = cell_center(cursor)
            x1, y1 = cell_center(next_state)

            draw.line((x0, y0, x1, y1), fill = (70, 130, 210), width = 2)

            dx, dy = x1 - x0, y1 - y0
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            px, py = -dy / length * 5, dx / length * 5
            mx, my = x1 - dx * 0.25, y1 - dy * 0.25
            draw.polygon(((x1, y1), (mx + px, my + py), (mx - px, my - py)), fill = (70, 130, 210))

            cursor = next_state

    # decoded latent plan

    centers = [cell_center(index) for index in predicted_states]

    if len(centers) > 1:
        draw.line(centers, fill = (38, 105, 220), width = 3)

    for i, center in enumerate(centers[1:], start = 1):
        radius = 4 if i < len(centers) - 1 else 6
        x, y = center
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill = (38, 105, 220), outline = (255, 255, 255))

    # goal and agent

    x, y = cell_center(goal)
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill = (50, 190, 95), outline = (20, 120, 55), width = 2)

    x, y = cell_center(state)
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill = (225, 65, 70), outline = (140, 20, 25), width = 2)

    # c-slerp waypoints

    if waypoint_states is not None:
        for index in waypoint_states:
            x, y = cell_center(index)
            x, y = x + cell * 0.28, y - cell * 0.28
            draw.polygon(((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)), fill = (245, 155, 35), outline = (140, 80, 0))

    if subgoal_state is not None:
        x, y = cell_center(subgoal_state)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline = (235, 115, 20), width = 3)

    title = f'{label} | {planner_name} | step {step:02d}' if label else f'{planner_name} | step {step:02d}'
    draw.text((pad, 8), title, fill = (25, 28, 38), font = font_bold)

    status = 'success' if success else 'planning'
    color = (30, 150, 75) if success else (90, 95, 110)
    draw.text((pad, 28), f'{status}  dots + arrows', fill = color, font = font)

    if waypoint_states is not None:
        draw.text((pad, height - 28), 'orange: waypoints / subgoal', fill = (150, 90, 10), font = font)
    else:
        draw.text((pad, height - 28), f'value range [{value_min:+.2f}, {value_max:+.2f}]', fill = (90, 95, 110), font = font)

    return np.asarray(image)

# video helpers

def save_gif(frames, path, fps = 6):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(path), frames, fps = fps, loop = 0)
    return path

def save_mp4(frames, path, fps = 6):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    try:
        with imageio.get_writer(str(path), fps = fps, codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1) as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception as error:
        print(f'could not write mp4 ({error}); gif was still written')
        return None

    return path

def pad_frames(frames, length):
    return frames + [frames[-1]] * (length - len(frames)) if frames else frames

def stack_panels(all_frames, rows, cols, gap = 8, background = (232, 234, 240)):
    max_len = max(len(frames) for frames in all_frames)
    all_frames = [pad_frames(frames, max_len) for frames in all_frames]

    panel_height, panel_width, _ = all_frames[0][0].shape
    width = cols * panel_width + (cols + 1) * gap
    height = rows * panel_height + (rows + 1) * gap

    videos = []

    for time in range(max_len):
        canvas = Image.new('RGB', (width, height), background)

        for i, frames in enumerate(all_frames):
            row, col = divmod(i, cols)
            x = gap + col * (panel_width + gap)
            y = gap + row * (panel_height + gap)
            canvas.paste(Image.fromarray(frames[time]), (x, y))

        videos.append(np.asarray(canvas))

    return videos

# main

def main(
    train_steps: int = TRAIN_STEPS,
    seed: int = SEED,
    video_dir: str = 'videos-gridworld-crl-wm',
    cpu: bool = True
):
    device = torch.device('cpu' if cpu or not torch.cuda.is_available() else 'cuda')
    torch.manual_seed(seed)

    env = GridWorld()
    print(f'walls ({len(env.walls)}): {sorted(env.walls)}', flush = True)
    print(f'shortest path: {env.shortest_path_length}', flush = True)

    model = make_model(env).to(device)
    planner = make_planner(model, use_c_slerp = False)
    c_slerp_planner = make_planner(model, use_c_slerp = True)

    video_dir = Path(video_dir)
    video_dir.mkdir(parents = True, exist_ok = True)

    checkpoints = tuple(sorted({0, train_steps // 6, train_steps}))
    checkpoint_frames = {}

    def render_checkpoint(step, current_model):
        current_model.eval()
        render = lambda **kwargs: render_frame(**kwargs, label = f'train {step}')

        result = evaluate_with_seed(
            1000 + step,
            evaluate_episode,
            current_model,
            env,
            planner,
            use_c_slerp = False,
            max_steps = MAX_EVAL_STEPS,
            render_fn = render
        )

        checkpoint_frames[step] = result['frames']
        print(f'checkpoint {step:5d} | success {result["success"]} | steps {result["steps"]}', flush = True)
        current_model.train()

    print('training contrastive world model', flush = True)
    history = train(
        model,
        env,
        train_steps = train_steps,
        seed = seed,
        checkpoints = checkpoints,
        checkpoint_fn = render_checkpoint
    )

    torch.save(dict(model = model.state_dict(), history = history, walls = sorted(env.walls)), video_dir / 'world_model.pt')
    model.eval()

    # final canonical rollouts

    final_cem = evaluate_with_seed(
        90_001,
        evaluate_episode,
        model,
        env,
        planner,
        use_c_slerp = False,
        max_steps = MAX_EVAL_STEPS,
        render_fn = render_frame
    )

    final_slerp = evaluate_with_seed(
        90_002,
        evaluate_episode,
        model,
        env,
        c_slerp_planner,
        use_c_slerp = True,
        max_steps = MAX_EVAL_STEPS,
        render_fn = render_frame
    )

    # conclusive evaluation: repeated trials from the canonical start

    cem_summary = evaluate_trials(model, env, planner, use_c_slerp = False)
    slerp_summary = evaluate_trials(model, env, c_slerp_planner, use_c_slerp = True)

    summary = dict(
        walls = sorted(env.walls),
        shortest_path = env.shortest_path_length,
        canonical_rollout = dict(
            cem = dict(success = final_cem['success'], steps = final_cem['steps']),
            c_slerp = dict(success = final_slerp['success'], steps = final_slerp['steps'])
        ),
        repeated_trials = dict(cem = cem_summary, c_slerp = slerp_summary),
        training = history
    )

    with open(video_dir / 'summary.json', 'w') as file:
        json.dump(summary, file, indent = 2)

    print(
        f'CEM      | {cem_summary["successes"]}/{cem_summary["trials"]} trials | '
        f'mean steps {cem_summary["mean_steps"]}',
        flush = True
    )
    print(
        f'C-Slerp  | {slerp_summary["successes"]}/{slerp_summary["trials"]} trials | '
        f'mean steps {slerp_summary["mean_steps"]}',
        flush = True
    )

    # assets

    learning_frames = stack_panels(
        [checkpoint_frames[step] for step in checkpoints],
        rows = 1,
        cols = len(checkpoints)
    )

    save_gif(learning_frames, video_dir / 'learning_to_plan.gif')
    save_mp4(learning_frames, video_dir / 'learning_to_plan.mp4')

    save_gif(final_cem['frames'], video_dir / 'plan_cem.gif')
    save_mp4(final_cem['frames'], video_dir / 'plan_cem.mp4')

    save_gif(final_slerp['frames'], video_dir / 'plan_c_slerp.gif')
    save_mp4(final_slerp['frames'], video_dir / 'plan_c_slerp.mp4')

    side_by_side = stack_panels([final_cem['frames'], final_slerp['frames']], rows = 1, cols = 2)
    save_gif(side_by_side, video_dir / 'plan_cem_vs_c_slerp.gif')
    save_mp4(side_by_side, video_dir / 'plan_cem_vs_c_slerp.mp4')

    print(f'wrote assets to {video_dir.resolve()}', flush = True)

    return summary

if __name__ == '__main__':
    Fire(main)
