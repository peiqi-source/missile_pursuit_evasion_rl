"""轨迹图绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trace: dict[str, np.ndarray], save_path: str | Path, plot_3d: bool = False) -> Path:
    """绘制 X-Z 平面轨迹，预留 3D 轨迹扩展参数。"""
    _ = plot_3d
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    red_trajectory = np.asarray(trace.get("red_trajectory", []), dtype=np.float64)
    blue_trajectory = np.asarray(trace.get("blue_trajectory", []), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    if red_trajectory.size:
        ax.plot(red_trajectory[:, 0], red_trajectory[:, 1], label="Red Trajectory", color="tab:red")
    if blue_trajectory.size:
        ax.plot(blue_trajectory[:, 0], blue_trajectory[:, 1], label="Blue Trajectory", color="tab:blue")
    ax.set(xlabel="X (m)", ylabel="Z (m)", title="Trajectory")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path

