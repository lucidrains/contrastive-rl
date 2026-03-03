from __future__ import annotations

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich import box
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# custom progress bar with tqdm-compatible interface

class DashboardPBar:
    def __init__(self, dashboard, task_id, label):
        self.dashboard = dashboard
        self.task_id = task_id
        self.label = label
        self.iterable = None
        self.disable = False

    def __call__(self, iterable, disable = False, **kwargs):
        self.iterable = iterable
        self.disable = disable

        if not self.disable:
            total = len(iterable) if hasattr(iterable, '__len__') else 100
            self.dashboard.show_training_bar(self.task_id, total)

        return self

    def __iter__(self):
        for item in self.iterable:
            yield item
            if not self.disable:
                self.dashboard.advance_training_bar(self.task_id)
                self.dashboard.refresh()

        if not self.disable:
            self.dashboard.hide_training_bar(self.task_id)
            self.dashboard.refresh()

    def set_description(self, desc):
        if not self.disable:
            self.dashboard.train_progress.update(self.task_id, description = f'{self.label} ({desc})')
            self.dashboard.refresh()

# dashboard

class Dashboard:
    def __init__(
        self,
        num_episodes,
        title = 'Contrastive RL Training',
        env_name = 'Unknown',
        hyperparams = None
    ):
        self.title = title
        self.live = None
        self.is_training = False

        # episode progress

        self.progress = Progress(
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand = True
        )

        self.episode_task = self.progress.add_task('Episodes', total = num_episodes)

        # training progress

        self.train_progress = Progress(
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand = True
        )

        critic_task = self.train_progress.add_task('Critic', total = 100, visible = False)
        actor_task = self.train_progress.add_task('Actor', total = 100, visible = False)

        self.critic_pbar = DashboardPBar(self, critic_task, 'Critic')
        self.actor_pbar = DashboardPBar(self, actor_task, 'Actor')

        # data

        self.hyperparams = default(hyperparams, {})

        self.metrics = {
            'env_name': env_name,
            'avg_cum_reward_100': '0.00',
            'avg_steps_100': '0.0',
            'last_eps_reward': '0.00',
            'last_eps_steps': '0',
            'critic_loss': '0.0000',
            'actor_loss': '0.0000',
        }

    # training bar controls

    def show_training_bar(self, task_id, total):
        self.train_progress.update(task_id, total = total, completed = 0, visible = True)
        self.is_training = True

    def advance_training_bar(self, task_id):
        self.train_progress.update(task_id, advance = 1)

    def hide_training_bar(self, task_id):
        self.train_progress.update(task_id, visible = False)
        self.is_training = any(t.visible for t in self.train_progress.tasks)

    # episode progress

    def advance_progress(self):
        self.progress.update(self.episode_task, advance = 1)

    # metric updates

    def update_metrics(self, **kwargs):
        self.metrics.update({k: f'{v}' for k, v in kwargs.items()})

    # rendering

    def _make_table(self, data, columns, styles):
        table = Table(box = box.ROUNDED, expand = True)

        for col, style in zip(columns, styles):
            table.add_column(col, style = style)

        for k, v in data.items():
            table.add_row(k, f'{v}')

        return table

    def render(self):
        progress_panel = Panel(self.progress, title = 'Progress', border_style = 'green')
        metrics_panel = Panel(self._make_table(self.metrics, ('Metric', 'Value'), ('cyan', 'magenta')), title = f'[b]{self.title}[/b]', border_style = 'blue')
        config_panel = Panel(self._make_table(self.hyperparams, ('Hyperparameter', 'Value'), ('yellow', 'white')), title = '[b]Configuration[/b]', border_style = 'yellow')

        panels = [progress_panel, metrics_panel]

        if self.is_training:
            panels.append(Panel(self.train_progress, title = 'Training', border_style = 'red'))

        panels.append(config_panel)
        return Group(*panels)

    def refresh(self):
        if exists(self.live):
            self.live.update(self.render())

    def create_renderable(self):
        self.live = Live(self.render(), refresh_per_second = 4)
        return self.live
