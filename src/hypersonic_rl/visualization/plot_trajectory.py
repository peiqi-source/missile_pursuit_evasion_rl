"""
plot_trajectory.py

作用：
    统一管理红方和蓝方轨迹类图像。

当前支持：
    1. 单视图轨迹图：X-Z、X-Y、Y-Z；
    2. 三视图组合轨迹图；
    3. 三维空间轨迹图；
    4. 三视图 + 三维轨迹综合图。

工程说明：
    本文件是轨迹绘制的统一入口。
    plot_episode.py 中的 X-Z 子图也会复用本文件中的公共绘图函数，
    避免不同文件重复维护同一套 X-Z 绘图逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# PlaneName：轨迹投影平面类型。
# "xz" 表示前向距离-侧向距离平面；
# "xy" 表示前向距离-高度平面；
# "yz" 表示高度-侧向距离平面。
PlaneName = Literal["xz", "xy", "yz"]


def extract_trajectories_from_env(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    从环境对象中提取红方和蓝方轨迹。

    参数：
        env：
            已经完成 rollout 的环境对象。
            该对象需要包含 red_trajectory 和 blue_trajectory。

    返回：
        red_trajectory：
            红方三维位置轨迹数组，形状为 [T, 3]。

        blue_trajectory：
            蓝方三维位置轨迹数组，形状为 [T, 3]。
    """
    # red_trajectory：红方轨迹，前三维分别为 x、y、z。
    red_trajectory = np.asarray(getattr(env, "red_trajectory", []), dtype=np.float64)

    # blue_trajectory：蓝方轨迹，前三维分别为 x、y、z。
    blue_trajectory = np.asarray(getattr(env, "blue_trajectory", []), dtype=np.float64)

    if red_trajectory.size == 0 or blue_trajectory.size == 0:
        raise ValueError("环境中没有轨迹记录，请先运行 env.step()。")

    if red_trajectory.ndim != 2 or red_trajectory.shape[1] < 3:
        raise ValueError(f"红方轨迹形状错误，应为 [T, 3]，实际为：{red_trajectory.shape}")

    if blue_trajectory.ndim != 2 or blue_trajectory.shape[1] < 3:
        raise ValueError(f"蓝方轨迹形状错误，应为 [T, 3]，实际为：{blue_trajectory.shape}")

    return red_trajectory[:, :3], blue_trajectory[:, :3]


def compute_closest_approach(
    red_trajectory: np.ndarray,
    blue_trajectory: np.ndarray,
) -> Tuple[int, float]:
    """
    计算红蓝双方最近接近点。

    参数：
        red_trajectory：
            红方轨迹数组，形状为 [T, 3]。

        blue_trajectory：
            蓝方轨迹数组，形状为 [T, 3]。

    返回：
        closest_index：
            最近距离对应的时间步索引。

        min_distance：
            最近距离，单位 m。
    """
    # min_length：两条轨迹共同拥有的最短长度。
    min_length = min(len(red_trajectory), len(blue_trajectory))

    if min_length <= 0:
        raise ValueError("轨迹长度为空，无法计算最近接近点。")

    # relative_position：红蓝双方逐时刻相对位置。
    relative_position = red_trajectory[:min_length] - blue_trajectory[:min_length]

    # distance_trace：红蓝双方逐时刻距离。
    distance_trace = np.linalg.norm(relative_position, axis=1)

    # closest_index：最近接近点索引。
    closest_index = int(np.argmin(distance_trace))

    # min_distance：最小距离。
    min_distance = float(distance_trace[closest_index])

    return closest_index, min_distance


def _get_projection_data(
    trajectory: np.ndarray,
    plane: PlaneName,
) -> Tuple[np.ndarray, np.ndarray, str, str, str]:
    """
    根据指定平面提取轨迹投影坐标。

    参数：
        trajectory：
            三维轨迹数组，形状为 [T, 3]。

        plane：
            投影平面名称。

    返回：
        x_values：
            绘图横坐标。

        y_values：
            绘图纵坐标。

        x_label：
            横坐标标签。

        y_label：
            纵坐标标签。

        title：
            子图标题。
    """
    if plane == "xz":
        # x_values：前向距离，转换为 km。
        x_values = trajectory[:, 0] / 1000.0

        # y_values：侧向距离，单位 m。
        y_values = trajectory[:, 2]

        x_label = "X position (km)"
        y_label = "Z lateral position (m)"
        title = "X-Z trajectory"

    elif plane == "xy":
        # x_values：前向距离，转换为 km。
        x_values = trajectory[:, 0] / 1000.0

        # y_values：高度，转换为 km。
        y_values = trajectory[:, 1] / 1000.0

        x_label = "X position (km)"
        y_label = "Altitude Y (km)"
        title = "X-Y trajectory"

    elif plane == "yz":
        # x_values：侧向距离，单位 m。
        x_values = trajectory[:, 2]

        # y_values：高度，转换为 km。
        y_values = trajectory[:, 1] / 1000.0

        x_label = "Z lateral position (m)"
        y_label = "Altitude Y (km)"
        title = "Y-Z trajectory"

    else:
        raise ValueError(f"未知 plane：{plane}，只能是 'xz'、'xy' 或 'yz'。")

    return x_values, y_values, x_label, y_label, title


def draw_trajectory_projection(
    ax: Any,
    red_trajectory: np.ndarray,
    blue_trajectory: np.ndarray,
    plane: PlaneName = "xz",
    mark_closest: bool = True,
    title: Optional[str] = None,
) -> Any:
    """
    在指定 matplotlib 坐标轴上绘制轨迹投影。

    参数：
        ax：
            matplotlib 坐标轴对象。

        red_trajectory：
            红方三维轨迹数组。

        blue_trajectory：
            蓝方三维轨迹数组。

        plane：
            投影平面名称。

        mark_closest：
            是否标记最近接近点。

        title：
            自定义标题。
            如果为 None，则使用默认平面标题。

    返回：
        ax：
            绘制后的坐标轴对象。
    """
    # red_x/red_y：红方在指定平面上的投影坐标。
    red_x, red_y, x_label, y_label, default_title = _get_projection_data(red_trajectory, plane)

    # blue_x/blue_y：蓝方在指定平面上的投影坐标。
    blue_x, blue_y, _, _, _ = _get_projection_data(blue_trajectory, plane)

    ax.plot(red_x, red_y, label="Red")
    ax.plot(blue_x, blue_y, label="Blue")

    # 标记起点。
    ax.scatter(red_x[0], red_y[0], marker="o", label="Red start")
    ax.scatter(blue_x[0], blue_y[0], marker="o", label="Blue start")

    # 标记终点。
    ax.scatter(red_x[-1], red_y[-1], marker="x", label="Red end")
    ax.scatter(blue_x[-1], blue_y[-1], marker="x", label="Blue end")

    if mark_closest:
        # closest_index：最近接近点索引。
        # min_distance：最近接近距离。
        closest_index, min_distance = compute_closest_approach(red_trajectory, blue_trajectory)

        ax.scatter(red_x[closest_index], red_y[closest_index], marker="s", label="Red closest")
        ax.scatter(blue_x[closest_index], blue_y[closest_index], marker="s", label="Blue closest")

        ax.plot(
            [red_x[closest_index], blue_x[closest_index]],
            [red_y[closest_index], blue_y[closest_index]],
            linestyle="--",
            linewidth=1.0,
            label=f"Min distance={min_distance:.2f} m",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(default_title if title is None else title)
    ax.grid(True)
    ax.legend(fontsize=8)

    return ax


def draw_3d_trajectory_on_axis(
    ax: Any,
    red_trajectory: np.ndarray,
    blue_trajectory: np.ndarray,
    mark_closest: bool = True,
) -> Any:
    """
    在指定 3D 坐标轴上绘制三维轨迹。

    参数：
        ax：
            matplotlib 3D 坐标轴对象。

        red_trajectory：
            红方三维轨迹数组。

        blue_trajectory：
            蓝方三维轨迹数组。

        mark_closest：
            是否标记最近接近点。

    返回：
        ax：
            绘制后的 3D 坐标轴对象。
    """
    # 红方轨迹坐标，统一转换为 km，便于三维图尺度更协调。
    red_x = red_trajectory[:, 0] / 1000.0
    red_y = red_trajectory[:, 1] / 1000.0
    red_z = red_trajectory[:, 2] / 1000.0

    # 蓝方轨迹坐标，统一转换为 km。
    blue_x = blue_trajectory[:, 0] / 1000.0
    blue_y = blue_trajectory[:, 1] / 1000.0
    blue_z = blue_trajectory[:, 2] / 1000.0

    ax.plot(red_x, red_y, red_z, label="Red")
    ax.plot(blue_x, blue_y, blue_z, label="Blue")

    # 标记起点。
    ax.scatter(red_x[0], red_y[0], red_z[0], marker="o", label="Red start")
    ax.scatter(blue_x[0], blue_y[0], blue_z[0], marker="o", label="Blue start")

    # 标记终点。
    ax.scatter(red_x[-1], red_y[-1], red_z[-1], marker="x", label="Red end")
    ax.scatter(blue_x[-1], blue_y[-1], blue_z[-1], marker="x", label="Blue end")

    if mark_closest:
        # closest_index：最近接近点索引。
        # min_distance：最近接近距离，单位 m。
        closest_index, min_distance = compute_closest_approach(red_trajectory, blue_trajectory)

        ax.scatter(
            red_x[closest_index],
            red_y[closest_index],
            red_z[closest_index],
            marker="s",
            label="Red closest",
        )

        ax.scatter(
            blue_x[closest_index],
            blue_y[closest_index],
            blue_z[closest_index],
            marker="s",
            label="Blue closest",
        )

        ax.plot(
            [red_x[closest_index], blue_x[closest_index]],
            [red_y[closest_index], blue_y[closest_index]],
            [red_z[closest_index], blue_z[closest_index]],
            linestyle="--",
            linewidth=1.0,
            label=f"Min distance={min_distance:.2f} m",
        )

    ax.set_xlabel("X position (km)")
    ax.set_ylabel("Altitude Y (km)")
    ax.set_zlabel("Z lateral position (km)")
    ax.set_title("3D trajectory")
    ax.legend(fontsize=8)

    return ax


def plot_trajectory(
    env: Any,
    save_path: str | Path,
    plane: PlaneName = "xz",
    show: bool = False,
) -> Path:
    """
    绘制单视图轨迹图。

    参数：
        env：
            已经完成 rollout 的环境对象。

        save_path：
            图像保存路径。

        plane：
            投影平面名称，可选 "xz"、"xy"、"yz"。

        show：
            是否显示图像。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：图像保存路径。
    saved_path = Path(save_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    # red_trajectory/blue_trajectory：从环境中提取的红蓝轨迹。
    red_trajectory, blue_trajectory = extract_trajectories_from_env(env)

    # figure/ax：图像对象和坐标轴对象。
    figure, ax = plt.subplots(figsize=(8, 5))

    draw_trajectory_projection(
        ax=ax,
        red_trajectory=red_trajectory,
        blue_trajectory=blue_trajectory,
        plane=plane,
        mark_closest=True,
    )

    figure.tight_layout()
    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path


def plot_three_view_trajectory(
    env: Any,
    save_path: str | Path,
    show: bool = False,
) -> Path:
    """
    绘制三视图轨迹图。

    三个子图分别为：
        1. X-Z 平面；
        2. X-Y 平面；
        3. Y-Z 平面。

    参数：
        env：
            已经完成 rollout 的环境对象。

        save_path：
            图像保存路径。

        show：
            是否显示图像。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：图像保存路径。
    saved_path = Path(save_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    # red_trajectory/blue_trajectory：红蓝轨迹。
    red_trajectory, blue_trajectory = extract_trajectories_from_env(env)

    # figure/axes：三视图图像对象和坐标轴数组。
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    draw_trajectory_projection(axes[0], red_trajectory, blue_trajectory, plane="xz", mark_closest=True)
    draw_trajectory_projection(axes[1], red_trajectory, blue_trajectory, plane="xy", mark_closest=True)
    draw_trajectory_projection(axes[2], red_trajectory, blue_trajectory, plane="yz", mark_closest=True)

    figure.suptitle("Three-view trajectory")
    figure.tight_layout()
    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path


def plot_3d_trajectory(
    env: Any,
    save_path: str | Path,
    elev: float = 25.0,
    azim: float = -60.0,
    show: bool = False,
) -> Path:
    """
    绘制三维空间轨迹图。

    参数：
        env：
            已经完成 rollout 的环境对象。

        save_path：
            图像保存路径。

        elev：
            三维视角的俯仰角。

        azim：
            三维视角的方位角。

        show：
            是否显示图像。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：图像保存路径。
    saved_path = Path(save_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    # red_trajectory/blue_trajectory：红蓝轨迹。
    red_trajectory, blue_trajectory = extract_trajectories_from_env(env)

    # figure：图像对象。
    figure = plt.figure(figsize=(9, 7))

    # ax：三维坐标轴对象。
    ax = figure.add_subplot(111, projection="3d")

    draw_3d_trajectory_on_axis(
        ax=ax,
        red_trajectory=red_trajectory,
        blue_trajectory=blue_trajectory,
        mark_closest=True,
    )

    ax.view_init(elev=elev, azim=azim)

    figure.tight_layout()
    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path


def plot_full_trajectory_summary(
    env: Any,
    save_path: str | Path,
    show: bool = False,
) -> Path:
    """
    绘制“三视图 + 三维轨迹”综合图。

    图像布局：
        左上：X-Z 轨迹；
        右上：X-Y 轨迹；
        左下：Y-Z 轨迹；
        右下：三维轨迹。

    参数：
        env：
            已经完成 rollout 的环境对象。

        save_path：
            图像保存路径。

        show：
            是否显示图像。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：图像保存路径。
    saved_path = Path(save_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    # red_trajectory/blue_trajectory：红蓝轨迹。
    red_trajectory, blue_trajectory = extract_trajectories_from_env(env)

    # figure：综合图对象。
    figure = plt.figure(figsize=(14, 10))

    # ax_xz：X-Z 子图。
    ax_xz = figure.add_subplot(2, 2, 1)

    # ax_xy：X-Y 子图。
    ax_xy = figure.add_subplot(2, 2, 2)

    # ax_yz：Y-Z 子图。
    ax_yz = figure.add_subplot(2, 2, 3)

    # ax_3d：三维轨迹子图。
    ax_3d = figure.add_subplot(2, 2, 4, projection="3d")

    draw_trajectory_projection(ax_xz, red_trajectory, blue_trajectory, plane="xz", mark_closest=True)
    draw_trajectory_projection(ax_xy, red_trajectory, blue_trajectory, plane="xy", mark_closest=True)
    draw_trajectory_projection(ax_yz, red_trajectory, blue_trajectory, plane="yz", mark_closest=True)
    draw_3d_trajectory_on_axis(ax_3d, red_trajectory, blue_trajectory, mark_closest=True)

    ax_3d.view_init(elev=25.0, azim=-60.0)

    figure.suptitle("Trajectory summary: three views and 3D")
    figure.tight_layout()
    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path