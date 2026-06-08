"""单 episode 过程图绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_episode(trace: dict[str, np.ndarray], save_path: str | Path) -> Path:
    """绘制单个 episode 的控制、轨迹和距离图。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # time：episode 时间序列。
    time = np.asarray(trace.get("time", []), dtype=np.float64)
    actions = np.asarray(trace.get("actions", []), dtype=np.float64)
    inertial_actions = np.asarray(trace.get("inertial_actions", []), dtype=np.float64)
    blue_commands = np.asarray(trace.get("blue_commands", []), dtype=np.float64)
    inertial_blue = np.asarray(trace.get("inertial_blue_commands", []), dtype=np.float64)
    red_trajectory = np.asarray(trace.get("red_trajectory", []), dtype=np.float64)
    blue_trajectory = np.asarray(trace.get("blue_trajectory", []), dtype=np.float64)
    distances = np.asarray(trace.get("distances", []), dtype=np.float64)
    min_distance = float(np.min(distances)) if distances.size else 0.0
    min_idx = int(np.argmin(distances)) if distances.size else 0

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)
    axes[0].plot(time, actions, label="Control Input", color="tab:blue")
    axes[0].plot(time, inertial_actions, label="Inertial Output (Red)", color="tab:red", linestyle="--")
    axes[0].set(xlabel="Time (s)", ylabel="Control", title="Red Control")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, blue_commands, label="Guidance Input", color="tab:green")
    axes[1].plot(time, inertial_blue, label="Inertial Output (Blue)", color="tab:purple", linestyle="--")
    axes[1].set(xlabel="Time (s)", ylabel="Control", title="Blue Guidance Control")
    axes[1].legend()
    axes[1].grid(True)

    if red_trajectory.size and blue_trajectory.size:
        axes[2].plot(red_trajectory[:, 0], red_trajectory[:, 1], label="Red Trajectory", color="tab:red")
        axes[2].plot(blue_trajectory[:, 0], blue_trajectory[:, 1], label="Blue Trajectory", color="tab:blue")
    axes[2].set(xlabel="X (m)", ylabel="Z (m)", title="X-Z Plane Trajectory")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(time, distances, label="Relative Distance", color="tab:orange")
    if distances.size and time.size:
        axes[3].axvline(time[min_idx], color="black", linestyle="--", label="Min Distance")
        axes[3].scatter([time[min_idx]], [min_distance], color="black", s=20)
    axes[3].set(xlabel="Time (s)", ylabel="Distance (m)", title=f"Relative Distance, min={min_distance:.2f} m")
    axes[3].legend()
    axes[3].grid(True)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path

