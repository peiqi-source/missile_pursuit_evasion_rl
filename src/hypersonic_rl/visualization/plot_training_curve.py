"""训练曲线绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curve(metrics: str | Path | pd.DataFrame, save_dir: str | Path) -> list[Path]:
    """根据 metrics.csv 绘制训练曲线。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(metrics) if not isinstance(metrics, pd.DataFrame) else metrics
    saved_paths: list[Path] = []
    curve_specs = [
        ("episode_reward", "Episode Reward", "episode_reward.png"),
        ("min_distance", "Min Distance", "min_distance.png"),
        ("success_rate", "Success Rate", "success_rate.png"),
        ("actor_loss", "Actor Loss", "actor_loss.png"),
        ("critic_loss", "Critic Loss", "critic_loss.png"),
        ("alpha_loss", "Alpha Loss", "alpha_loss.png"),
    ]
    for column, title, filename in curve_specs:
        if column not in data.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data["episode"], data[column], label=title)
        ax.set(xlabel="Episode", ylabel=title, title=title)
        ax.grid(True)
        ax.legend()
        path = save_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths

