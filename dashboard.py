from __future__ import annotations

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, Group
from rich import box
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)

class Dashboard:
    def __init__(self, num_episodes, title = "Contrastive RL Training", env_name = "Unknown", hyperparams = None):
        self.title = title
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand = True
        )
        self.pbar_task = self.progress.add_task("Episodes", total = num_episodes)

        self.hyperparams = hyperparams or {}

        self.episode_info = {
            "env_name": env_name,
            "avg_cum_reward_100": "0.00",
            "avg_steps_100": "0.0",
            "last_eps_reward": "0.00",
            "last_eps_steps": "0",
            "critic_loss": "0.0000",
            "actor_loss": "0.0000",
        }

    def update_episode_info(self, **kwargs):
        self.episode_info.update({k: str(v) for k, v in kwargs.items()})

    def update_diagnostics(self, **kwargs):
        self.episode_info.update({k: str(v) for k, v in kwargs.items()})

    def advance_progress(self):
        self.progress.update(self.pbar_task, advance = 1)

    def _make_table(self, data, columns, styles):
        table = Table(box = box.ROUNDED, expand = True)
        for col, style in zip(columns, styles):
            table.add_column(col, style = style, width = 30)
        for k, v in data.items():
            table.add_row(k, str(v))
        return table

    def render(self):
        progress_panel = Panel(self.progress, title = "Progress", border_style = "green")
        metrics_panel = Panel(self._make_table(self.episode_info, ("Metric", "Value"), ("cyan", "magenta")), title = f"[b]{self.title}[/b]", border_style = "blue")
        config_panel = Panel(self._make_table(self.hyperparams, ("Hyperparameter", "Value"), ("yellow", "white")), title = "[b]Configuration[/b]", border_style = "yellow")

        return Group(progress_panel, metrics_panel, config_panel)

    def create_renderable(self):
        return Live(self.render(), refresh_per_second = 4)
