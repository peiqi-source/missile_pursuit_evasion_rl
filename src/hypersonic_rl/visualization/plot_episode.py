"""
plot_episode.py

作用：
    绘制单个 episode 的综合诊断图。

图像内容：
    1. 红方与蓝方侧向过载控制量；
    2. 红蓝双方 X-Z 平面轨迹；
    3. 红蓝相对距离变化；
    4. 单步奖励变化。

工程说明：
    轨迹子图复用 plot_trajectory.py 中的公共函数，
    避免 X-Z 轨迹绘图逻辑在多个文件中重复出现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hypersonic_rl.visualization.plot_trajectory import (
    draw_trajectory_projection,
    extract_trajectories_from_env,
)


def _ensure_parent_dir(save_path: Path) -> None:
    """
    确保保存路径的父目录存在。

    参数：
        save_path：
            图像保存路径。
    """
    # parent_dir：图像文件所在目录。
    parent_dir = save_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)


def _to_array(trace: Any) -> np.ndarray:
    """
    将环境记录的列表转换为 numpy 数组。

    参数：
        trace：
            环境中记录的轨迹、控制量、距离或奖励列表。

    返回：
        array：
            转换后的 numpy 数组。
    """
    # array：转换后的数组。
    array = np.asarray(trace, dtype=np.float64)

    return array


def plot_episode_summary(
    env: Any,
    save_path: str | Path,
    title: Optional[str] = None,
    show: bool = False,
) -> Path:
    """
    绘制单回合综合诊断图。

    参数：
        env：
            已经完成 rollout 的环境对象。

        save_path：
            图像保存路径。

        title：
            图像标题。

        show：
            是否显示图像。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：图像保存路径。
    saved_path = Path(save_path)
    _ensure_parent_dir(saved_path)

    # time_trace：时间序列。
    time_trace = _to_array(getattr(env, "time_trace", []))

    # red_control：红方一阶惯性后的实际过载序列。
    red_control = _to_array(getattr(env, "red_inertial_control_trace", []))

    # blue_control：蓝方一阶惯性后的实际过载序列。
    blue_control = _to_array(getattr(env, "blue_inertial_control_trace", []))

    # distance_trace：红蓝距离序列。
    distance_trace = _to_array(getattr(env, "distance_trace", []))

    # reward_trace：单步奖励序列。
    reward_trace = _to_array(getattr(env, "reward_trace", []))

    if time_trace.size == 0:
        # step_axis：如果没有时间记录，则使用步数作为横坐标。
        step_axis = np.arange(len(distance_trace), dtype=np.float64)
        x_label = "Step"
    else:
        step_axis = time_trace
        x_label = "Time (s)"

    # figure/axes：2×2 综合图。
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    if title is not None:
        figure.suptitle(title)

    # ------------------------------------------------------------------
    # 1. 控制量曲线
    # ------------------------------------------------------------------
    axes[0, 0].plot(step_axis[: len(red_control)], red_control, label="Red actual overload")
    axes[0, 0].plot(step_axis[: len(blue_control)], blue_control, label="Blue actual overload")
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel("Lateral overload")
    axes[0, 0].set_title("Control trace")
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=8)

    # ------------------------------------------------------------------
    # 2. X-Z 轨迹子图
    # ------------------------------------------------------------------
    try:
        # red_trajectory/blue_trajectory：红蓝双方三维轨迹。
        red_trajectory, blue_trajectory = extract_trajectories_from_env(env)

        draw_trajectory_projection(
            ax=axes[0, 1],
            red_trajectory=red_trajectory,
            blue_trajectory=blue_trajectory,
            plane="xz",
            mark_closest=True,
            title="X-Z trajectory",
        )
    except ValueError:
        axes[0, 1].set_title("X-Z trajectory unavailable")
        axes[0, 1].grid(True)

    # ------------------------------------------------------------------
    # 3. 红蓝距离曲线
    # ------------------------------------------------------------------
    axes[1, 0].plot(step_axis[: len(distance_trace)], distance_trace)

    if distance_trace.size > 0:
        # min_distance_index：最小距离对应的索引。
        min_distance_index = int(np.argmin(distance_trace))

        axes[1, 0].scatter(
            step_axis[min_distance_index],
            distance_trace[min_distance_index],
            marker="o",
            label=f"min={distance_trace[min_distance_index]:.2f} m",
        )

        axes[1, 0].legend(fontsize=8)

    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Distance (m)")
    axes[1, 0].set_title("Relative distance")
    axes[1, 0].grid(True)

    # ------------------------------------------------------------------
    # 4. 奖励曲线
    # ------------------------------------------------------------------
    axes[1, 1].plot(step_axis[: len(reward_trace)], reward_trace)
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].set_title("Reward trace")
    axes[1, 1].grid(True)

    figure.tight_layout()
    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path