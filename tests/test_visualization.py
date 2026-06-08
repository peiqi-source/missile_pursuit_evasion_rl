"""可视化模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypersonic_rl.visualization.plot_episode import plot_episode
from hypersonic_rl.visualization.plot_training_curve import plot_training_curve
from hypersonic_rl.visualization.plot_trajectory import plot_trajectory


def _fake_trace() -> dict[str, np.ndarray]:
    """构造用于绘图的假 episode trace。"""
    time = np.linspace(0.0, 0.3, 4)
    red = np.column_stack([np.linspace(0.0, 3.0, 4), np.linspace(0.0, 1.0, 4)])
    blue = np.column_stack([np.linspace(3.0, 0.0, 4), np.linspace(1.0, 0.0, 4)])
    return {
        "time": time,
        "actions": np.zeros(4),
        "inertial_actions": np.zeros(4),
        "blue_commands": np.ones(4),
        "inertial_blue_commands": np.ones(4),
        "red_trajectory": red,
        "blue_trajectory": blue,
        "distances": np.array([3.0, 2.0, 1.5, 2.5]),
    }


def test_visualization_saves_files(tmp_path) -> None:
    """检查 episode 图、轨迹图和训练曲线能正常保存。"""
    trace = _fake_trace()
    episode_path = plot_episode(trace, tmp_path / "episode.png")
    trajectory_path = plot_trajectory(trace, tmp_path / "trajectory.png")
    metrics = pd.DataFrame(
        {
            "episode": [0, 1],
            "episode_reward": [1.0, 2.0],
            "min_distance": [3.0, 4.0],
            "success_rate": [0.0, 1.0],
            "actor_loss": [0.1, 0.2],
            "critic_loss": [0.3, 0.4],
            "alpha_loss": [0.01, 0.02],
        }
    )
    curve_paths = plot_training_curve(metrics, tmp_path / "curves")
    assert episode_path.exists()
    assert trajectory_path.exists()
    assert curve_paths
    assert all(path.exists() for path in curve_paths)

