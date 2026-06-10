"""
plot_trajectory.py

作用：
    统一绘制红方与一枚或两枚拦截弹的二维/三维轨迹图。

当前约定：
    1. 环境轨迹字段为 red_trajectory 与 interceptor_trajectories；
    2. 单弹 helper 返回第一枚拦截弹轨迹，便于旧绘图调用方式继续使用函数名；
    3. 对外图例和文件字段统一使用 interceptor 命名。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# RED_COLOR：红方轨迹颜色。
RED_COLOR = "tab:red"

# INTERCEPTOR_COLORS：多枚拦截弹轨迹颜色，按编号循环使用。
INTERCEPTOR_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:cyan"]

# CLOSEST_LINE_COLOR：单弹最近点连线颜色。
CLOSEST_LINE_COLOR = "black"

# PlaneName：支持的二维投影平面。
PlaneName = Literal["xz", "xy", "yz"]


def extract_trajectories_from_env(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    从环境对象中提取红方和第一枚拦截弹轨迹。

    参数：
        env：
            已经完成 rollout 的环境对象。

    返回：
        red_trajectory：
            红方三维位置轨迹，形状为 [T, 3]。
        interceptor_trajectory：
            第一枚拦截弹三维位置轨迹，形状为 [T, 3]。
    """
    # red_trajectory：红方三维位置轨迹。
    red_trajectory, interceptor_trajectories = extract_multi_trajectories_from_env(env)

    # interceptor_trajectory：单弹 helper 使用第一枚拦截弹轨迹。
    interceptor_trajectory = interceptor_trajectories[0]

    return red_trajectory, interceptor_trajectory


def extract_multi_trajectories_from_env(env: Any) -> Tuple[np.ndarray, list[np.ndarray]]:
    """
    从环境对象中提取红方和全部拦截弹轨迹。

    参数：
        env：
            已经完成 rollout 的环境对象。

    返回：
        red_trajectory：
            红方三维位置轨迹，形状为 [T, 3]。
        interceptor_trajectories：
            每枚拦截弹三维位置轨迹列表。
    """
    # red_trajectory：红方轨迹记录。
    red_trajectory = np.asarray(getattr(env, "red_trajectory", []), dtype=np.float64)

    if red_trajectory.size == 0:
        raise ValueError("环境中没有红方轨迹记录，请先运行 env.step()。")

    if red_trajectory.ndim != 2 or red_trajectory.shape[1] < 3:
        raise ValueError(f"红方轨迹形状错误，应为 [T, 3]，实际为：{red_trajectory.shape}")

    # raw_trajectories：双弹环境中的拦截弹轨迹列表。
    raw_trajectories = getattr(env, "interceptor_trajectories", None)

    if not raw_trajectories:
        raise ValueError("环境中没有拦截弹轨迹记录，请先运行 env.step()。")

    # interceptor_trajectories：清洗后的三维轨迹列表。
    interceptor_trajectories: list[np.ndarray] = []

    for index, trajectory in enumerate(raw_trajectories, start=1):
        # trajectory_array：当前拦截弹轨迹数组。
        trajectory_array = np.asarray(trajectory, dtype=np.float64)

        if trajectory_array.size == 0:
            continue

        if trajectory_array.ndim != 2 or trajectory_array.shape[1] < 3:
            raise ValueError(f"第 {index} 枚拦截弹轨迹形状错误，应为 [T, 3]，实际为：{trajectory_array.shape}")

        interceptor_trajectories.append(trajectory_array[:, :3])

    if not interceptor_trajectories:
        raise ValueError("环境中没有有效的拦截弹轨迹记录，请先运行 env.step()。")

    return red_trajectory[:, :3], interceptor_trajectories


def compute_closest_approach(
    red_trajectory: np.ndarray,
    interceptor_trajectory: np.ndarray,
) -> Tuple[int, float]:
    """
    计算红方与单枚拦截弹的离散最近接近点。

    参数：
        red_trajectory：
            红方三维轨迹，形状为 [T, 3]。
        interceptor_trajectory：
            单枚拦截弹三维轨迹，形状为 [T, 3]。

    返回：
        closest_index：
            最近距离对应的时间步索引。
        min_distance：
            最近距离，单位 m。
    """
    # min_length：两条轨迹共同拥有的最短长度。
    min_length = min(len(red_trajectory), len(interceptor_trajectory))

    if min_length <= 0:
        raise ValueError("轨迹长度为空，无法计算最近接近点。")

    # relative_position：逐采样点相对位置。
    relative_position = red_trajectory[:min_length] - interceptor_trajectory[:min_length]

    # distance_trace：逐采样点距离。
    distance_trace = np.linalg.norm(relative_position, axis=1)

    # closest_index：离散最近点索引。
    closest_index = int(np.argmin(distance_trace))

    # min_distance：离散最近点距离。
    min_distance = float(distance_trace[closest_index])

    return closest_index, min_distance


def get_episode_min_distance(env: Any, sampled_min_distance: float) -> float:
    """
    优先返回环境记录的连续最小距离。

    参数：
        env：
            环境对象。
        sampled_min_distance：
            从离散轨迹采样点计算出的最小距离。

    返回：
        min_distance：
            用于图像标注的最小距离。
    """
    try:
        # episode_min_distance：环境连续最近点判据记录的最小距离。
        episode_min_distance = float(getattr(env, "min_distance", sampled_min_distance))
    except (TypeError, ValueError):
        return float(sampled_min_distance)

    if not np.isfinite(episode_min_distance):
        return float(sampled_min_distance)

    return min(float(sampled_min_distance), episode_min_distance)


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
            默认子图标题。
    """
    if plane == "xz":
        # x_values：前向距离，转换为 km。
        x_values = trajectory[:, 0] / 1000.0

        # y_values：侧向距离，单位 m。
        y_values = trajectory[:, 2]

        x_label = "X forward distance (km)"
        y_label = "Z lateral distance (m)"
        title = "X-Z trajectory"

    elif plane == "xy":
        # x_values：前向距离，转换为 km。
        x_values = trajectory[:, 0] / 1000.0

        # y_values：高度，转换为 km。
        y_values = trajectory[:, 1] / 1000.0

        x_label = "X forward distance (km)"
        y_label = "Y altitude (km)"
        title = "X-Y trajectory"

    elif plane == "yz":
        # x_values：侧向距离，单位 m。
        x_values = trajectory[:, 2]

        # y_values：高度，转换为 km。
        y_values = trajectory[:, 1] / 1000.0

        x_label = "Z lateral distance (m)"
        y_label = "Y altitude (km)"
        title = "Y-Z trajectory"

    else:
        raise ValueError(f"未知 plane：{plane}，只支持 'xz'、'xy' 或 'yz'。")

    return x_values, y_values, x_label, y_label, title


def draw_trajectory_projection(
    ax: Any,
    red_trajectory: np.ndarray,
    interceptor_trajectory: np.ndarray,
    plane: PlaneName = "xz",
    mark_closest: bool = True,
    title: Optional[str] = None,
    episode_min_distance: Optional[float] = None,
) -> Any:
    """
    在指定 matplotlib 坐标轴上绘制红方和单枚拦截弹二维投影。

    参数：
        ax：
            matplotlib 坐标轴对象。
        red_trajectory：
            红方三维轨迹。
        interceptor_trajectory：
            单枚拦截弹三维轨迹。
        plane：
            投影平面名称。
        mark_closest：
            是否标记最近接近点。
        title：
            可选自定义标题。
        episode_min_distance：
            可选连续最小距离。

    返回：
        ax：
            绘制后的坐标轴对象。
    """
    # red_x/red_y：红方在指定平面上的投影坐标。
    red_x, red_y, x_label, y_label, default_title = _get_projection_data(red_trajectory, plane)

    # interceptor_x/interceptor_y：拦截弹在指定平面上的投影坐标。
    interceptor_x, interceptor_y, _, _, _ = _get_projection_data(interceptor_trajectory, plane)

    ax.plot(red_x, red_y, color=RED_COLOR, linewidth=1.8, label="Red trajectory")
    ax.plot(
        interceptor_x,
        interceptor_y,
        color=INTERCEPTOR_COLORS[0],
        linewidth=1.8,
        label="Interceptor trajectory",
    )

    ax.scatter(red_x[0], red_y[0], color=RED_COLOR, marker="o", s=35, label="Red start")
    ax.scatter(interceptor_x[0], interceptor_y[0], color=INTERCEPTOR_COLORS[0], marker="o", s=35, label="Interceptor start")
    ax.scatter(red_x[-1], red_y[-1], color=RED_COLOR, marker="x", s=45, label="Red end")
    ax.scatter(interceptor_x[-1], interceptor_y[-1], color=INTERCEPTOR_COLORS[0], marker="x", s=45, label="Interceptor end")

    if mark_closest:
        # closest_index/min_distance：离散最近接近点。
        closest_index, min_distance = compute_closest_approach(red_trajectory, interceptor_trajectory)

        # display_min_distance：优先使用连续最近点距离。
        display_min_distance = min_distance if episode_min_distance is None else float(episode_min_distance)

        ax.scatter(red_x[closest_index], red_y[closest_index], color=RED_COLOR, marker="s", s=45, label="Red closest")
        ax.scatter(
            interceptor_x[closest_index],
            interceptor_y[closest_index],
            color=INTERCEPTOR_COLORS[0],
            marker="s",
            s=45,
            label="Interceptor closest",
        )
        ax.plot(
            [red_x[closest_index], interceptor_x[closest_index]],
            [red_y[closest_index], interceptor_y[closest_index]],
            color=CLOSEST_LINE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"Min distance={display_min_distance:.2f} m",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(default_title if title is None else title)
    ax.grid(True)
    ax.legend(fontsize=8)

    return ax


def draw_multi_trajectory_projection(
    ax: Any,
    red_trajectory: np.ndarray,
    interceptor_trajectories: list[np.ndarray],
    plane: PlaneName = "xz",
    mark_closest: bool = True,
    title: Optional[str] = None,
    episode_min_distance: Optional[float] = None,
) -> Any:
    """
    在二维投影平面上绘制红方和多枚拦截弹轨迹。

    参数：
        ax：
            matplotlib 坐标轴对象。
        red_trajectory：
            红方三维轨迹。
        interceptor_trajectories：
            拦截弹三维轨迹列表。
        plane：
            投影平面名称。
        mark_closest：
            是否标记每枚弹的最近接近点。
        title：
            可选自定义标题。
        episode_min_distance：
            环境连续最小距离。

    返回：
        ax：
            绘制后的坐标轴对象。
    """
    # red_x/red_y：红方在指定平面上的投影坐标。
    red_x, red_y, x_label, y_label, default_title = _get_projection_data(red_trajectory, plane)

    ax.plot(red_x, red_y, color=RED_COLOR, linewidth=1.8, label="Red trajectory")
    ax.scatter(red_x[0], red_y[0], color=RED_COLOR, marker="o", s=35, label="Red start")
    ax.scatter(red_x[-1], red_y[-1], color=RED_COLOR, marker="x", s=45, label="Red end")

    for index, trajectory in enumerate(interceptor_trajectories, start=1):
        # color：当前拦截弹颜色。
        color = INTERCEPTOR_COLORS[(index - 1) % len(INTERCEPTOR_COLORS)]

        # interceptor_x/interceptor_y：当前拦截弹投影坐标。
        interceptor_x, interceptor_y, _, _, _ = _get_projection_data(trajectory, plane)

        ax.plot(interceptor_x, interceptor_y, color=color, linewidth=1.6, label=f"Interceptor {index}")
        ax.scatter(interceptor_x[0], interceptor_y[0], color=color, marker="o", s=30, label=f"I{index} start")
        ax.scatter(interceptor_x[-1], interceptor_y[-1], color=color, marker="x", s=40, label=f"I{index} end")

        if mark_closest:
            # closest_index/min_distance：当前弹离散最近接近点。
            closest_index, min_distance = compute_closest_approach(red_trajectory, trajectory)

            # display_min_distance：第一枚弹可显示环境连续全局最小距离。
            display_min_distance = float(episode_min_distance) if episode_min_distance is not None and index == 1 else float(min_distance)

            ax.scatter(red_x[closest_index], red_y[closest_index], color=RED_COLOR, marker="s", s=35)
            ax.scatter(interceptor_x[closest_index], interceptor_y[closest_index], color=color, marker="s", s=35)
            ax.plot(
                [red_x[closest_index], interceptor_x[closest_index]],
                [red_y[closest_index], interceptor_y[closest_index]],
                color=color,
                linestyle="--",
                linewidth=0.9,
                label=f"I{index} min={display_min_distance:.2f} m",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(default_title if title is None else title)
    ax.grid(True)
    ax.legend(fontsize=7)

    return ax


def draw_3d_trajectory_on_axis(
    ax: Any,
    red_trajectory: np.ndarray,
    interceptor_trajectory: np.ndarray,
    mark_closest: bool = True,
    episode_min_distance: Optional[float] = None,
) -> Any:
    """
    在指定 3D 坐标轴上绘制红方和单枚拦截弹轨迹。

    参数：
        ax：
            matplotlib 3D 坐标轴对象。
        red_trajectory：
            红方三维轨迹。
        interceptor_trajectory：
            单枚拦截弹三维轨迹。
        mark_closest：
            是否标记最近接近点。
        episode_min_distance：
            可选连续最小距离。

    返回：
        ax：
            绘制后的 3D 坐标轴对象。
    """
    return draw_multi_3d_trajectory_on_axis(
        ax=ax,
        red_trajectory=red_trajectory,
        interceptor_trajectories=[interceptor_trajectory],
        mark_closest=mark_closest,
        episode_min_distance=episode_min_distance,
    )


def draw_multi_3d_trajectory_on_axis(
    ax: Any,
    red_trajectory: np.ndarray,
    interceptor_trajectories: list[np.ndarray],
    mark_closest: bool = True,
    episode_min_distance: Optional[float] = None,
) -> Any:
    """
    在指定 3D 坐标轴上绘制红方和多枚拦截弹轨迹。

    参数：
        ax：
            matplotlib 3D 坐标轴对象。
        red_trajectory：
            红方三维轨迹。
        interceptor_trajectories：
            拦截弹三维轨迹列表。
        mark_closest：
            是否标记最近接近点连线。
        episode_min_distance：
            环境连续最小距离。

    返回：
        ax：
            绘制后的 3D 坐标轴对象。
    """
    # red_*：红方轨迹坐标，统一转换为 km。
    red_forward_x = red_trajectory[:, 0] / 1000.0
    red_lateral_z = red_trajectory[:, 2] / 1000.0
    red_altitude_y = red_trajectory[:, 1] / 1000.0

    ax.plot(red_forward_x, red_lateral_z, red_altitude_y, color=RED_COLOR, linewidth=1.8, label="Red trajectory")
    ax.scatter(red_forward_x[0], red_lateral_z[0], red_altitude_y[0], color=RED_COLOR, marker="o", s=35, label="Red start")
    ax.scatter(red_forward_x[-1], red_lateral_z[-1], red_altitude_y[-1], color=RED_COLOR, marker="x", s=45, label="Red end")

    for index, trajectory in enumerate(interceptor_trajectories, start=1):
        # color：当前拦截弹颜色。
        color = INTERCEPTOR_COLORS[(index - 1) % len(INTERCEPTOR_COLORS)]

        # interceptor_*：当前拦截弹轨迹坐标，统一转换为 km。
        interceptor_forward_x = trajectory[:, 0] / 1000.0
        interceptor_lateral_z = trajectory[:, 2] / 1000.0
        interceptor_altitude_y = trajectory[:, 1] / 1000.0

        ax.plot(
            interceptor_forward_x,
            interceptor_lateral_z,
            interceptor_altitude_y,
            color=color,
            linewidth=1.6,
            label=f"Interceptor {index}",
        )
        ax.scatter(
            interceptor_forward_x[0],
            interceptor_lateral_z[0],
            interceptor_altitude_y[0],
            color=color,
            marker="o",
            s=30,
            label=f"I{index} start",
        )
        ax.scatter(
            interceptor_forward_x[-1],
            interceptor_lateral_z[-1],
            interceptor_altitude_y[-1],
            color=color,
            marker="x",
            s=40,
            label=f"I{index} end",
        )

        if mark_closest:
            # closest_index/min_distance：当前弹离散最近点。
            closest_index, min_distance = compute_closest_approach(red_trajectory, trajectory)

            # display_min_distance：第一枚弹可使用环境全局连续最小距离。
            display_min_distance = float(episode_min_distance) if episode_min_distance is not None and index == 1 else float(min_distance)

            ax.plot(
                [red_forward_x[closest_index], interceptor_forward_x[closest_index]],
                [red_lateral_z[closest_index], interceptor_lateral_z[closest_index]],
                [red_altitude_y[closest_index], interceptor_altitude_y[closest_index]],
                color=color,
                linestyle="--",
                linewidth=0.9,
                label=f"I{index} min={display_min_distance:.2f} m",
            )

    ax.set_xlabel("X forward distance (km)")
    ax.set_ylabel("Z lateral distance (km)")
    ax.set_zlabel("Y altitude (km)")
    ax.set_title("3D trajectory: red and interceptors")
    ax.grid(True)
    ax.legend(fontsize=7)

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
            投影平面名称。
        show：
            是否显示图像。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：图像保存路径。
    saved_path = Path(save_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    # red_trajectory/interceptor_trajectories：从环境中提取轨迹。
    red_trajectory, interceptor_trajectories = extract_multi_trajectories_from_env(env)

    # episode_min_distance：用于图例标注的连续最小距离。
    _, sampled_min_distance = compute_closest_approach(red_trajectory, interceptor_trajectories[0])
    episode_min_distance = get_episode_min_distance(env, sampled_min_distance)

    # figure/ax：图像和坐标轴对象。
    figure, ax = plt.subplots(figsize=(8, 5))

    draw_multi_trajectory_projection(
        ax=ax,
        red_trajectory=red_trajectory,
        interceptor_trajectories=interceptor_trajectories,
        plane=plane,
        mark_closest=True,
        episode_min_distance=episode_min_distance,
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
    绘制 X-Z、X-Y、Y-Z 三视图轨迹图。

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

    # red_trajectory/interceptor_trajectories：红方和全部拦截弹轨迹。
    red_trajectory, interceptor_trajectories = extract_multi_trajectories_from_env(env)

    # episode_min_distance：用于图例标注的连续最小距离。
    _, sampled_min_distance = compute_closest_approach(red_trajectory, interceptor_trajectories[0])
    episode_min_distance = get_episode_min_distance(env, sampled_min_distance)

    # figure/axes：三视图图像对象和坐标轴数组。
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, plane in zip(axes, ["xz", "xy", "yz"]):
        draw_multi_trajectory_projection(
            ax=ax,
            red_trajectory=red_trajectory,
            interceptor_trajectories=interceptor_trajectories,
            plane=plane,
            mark_closest=True,
            episode_min_distance=episode_min_distance,
        )

    figure.tight_layout()
    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path


def plot_3d_trajectory(
    env: Any,
    save_path: str | Path,
    show: bool = False,
) -> Path:
    """
    绘制三维轨迹图。

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

    # red_trajectory/interceptor_trajectories：红方和全部拦截弹轨迹。
    red_trajectory, interceptor_trajectories = extract_multi_trajectories_from_env(env)

    # episode_min_distance：用于图例标注的连续最小距离。
    _, sampled_min_distance = compute_closest_approach(red_trajectory, interceptor_trajectories[0])
    episode_min_distance = get_episode_min_distance(env, sampled_min_distance)

    # figure/ax：三维图像和坐标轴对象。
    figure = plt.figure(figsize=(9, 7))
    ax = figure.add_subplot(111, projection="3d")

    draw_multi_3d_trajectory_on_axis(
        ax=ax,
        red_trajectory=red_trajectory,
        interceptor_trajectories=interceptor_trajectories,
        mark_closest=True,
        episode_min_distance=episode_min_distance,
    )

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
    绘制三视图加三维综合轨迹图。

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

    # red_trajectory/interceptor_trajectories：红方和全部拦截弹轨迹。
    red_trajectory, interceptor_trajectories = extract_multi_trajectories_from_env(env)

    # episode_min_distance：用于图例标注的连续最小距离。
    _, sampled_min_distance = compute_closest_approach(red_trajectory, interceptor_trajectories[0])
    episode_min_distance = get_episode_min_distance(env, sampled_min_distance)

    # figure：综合轨迹图。
    figure = plt.figure(figsize=(16, 10))

    # axes：三个二维投影和一个三维坐标轴。
    axes = [
        figure.add_subplot(2, 2, 1),
        figure.add_subplot(2, 2, 2),
        figure.add_subplot(2, 2, 3),
    ]
    ax_3d = figure.add_subplot(2, 2, 4, projection="3d")

    for ax, plane in zip(axes, ["xz", "xy", "yz"]):
        draw_multi_trajectory_projection(
            ax=ax,
            red_trajectory=red_trajectory,
            interceptor_trajectories=interceptor_trajectories,
            plane=plane,
            mark_closest=True,
            episode_min_distance=episode_min_distance,
        )

    draw_multi_3d_trajectory_on_axis(
        ax=ax_3d,
        red_trajectory=red_trajectory,
        interceptor_trajectories=interceptor_trajectories,
        mark_closest=True,
        episode_min_distance=episode_min_distance,
    )

    figure.tight_layout()
    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path
