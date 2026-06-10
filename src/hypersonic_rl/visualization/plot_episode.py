"""
plot_episode.py

作用：
    绘制单个 episode 的综合诊断图。

图像内容：
    1. 红方横向过载与蓝方侧向 nz 过载；
    2. 蓝方纵向 ny 过载与平衡项 cos(theta)；
    3. X-Z 与 X-Y 两个轨迹投影视图；
    4. 红蓝相对距离和单步奖励。

工程说明：
    蓝方诊断字段已经统一为 ny / nz 双通道，本文件不再读取旧单通道字段。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hypersonic_rl.visualization.plot_trajectory import (
    INTERCEPTOR_COLORS,
    RED_COLOR,
    draw_multi_trajectory_projection,
    extract_multi_trajectories_from_env,
)


def _ensure_parent_dir(save_path: Path) -> None:
    """
    确保图片输出目录存在。

    参数：
        save_path：
            图片保存路径。

    返回：
        None。
    """
    # parent_dir：图片文件所在目录。
    parent_dir = save_path.parent

    parent_dir.mkdir(parents=True, exist_ok=True)


def _to_array(trace: Any) -> np.ndarray:
    """
    将环境记录序列转换为 numpy 数组。

    参数：
        trace：
            环境中的列表、元组或其他可转换序列。

    返回：
        array：
            浮点 numpy 数组。
    """
    # array：统一后的浮点数组。
    array = np.asarray(trace, dtype=np.float64)

    return array


def _axis_for(trace: np.ndarray, step_axis: np.ndarray) -> np.ndarray:
    """
    为某条曲线生成对齐的横坐标。

    参数：
        trace：
            待绘制的数据序列。
        step_axis：
            环境记录的时间轴或步数轴。

    返回：
        axis：
            与 trace 长度一致的横坐标数组。
    """
    if step_axis.size == 0:
        # axis：没有环境时间轴时使用步数作为横坐标。
        axis = np.arange(len(trace), dtype=np.float64)
    else:
        # axis：截取与当前 trace 长度一致的时间轴。
        axis = step_axis[: len(trace)]

    return axis


def _interceptor_theta_trace(env: Any, interceptor_index: int, length: int) -> np.ndarray:
    """
    从环境 trace 中提取指定拦截弹航迹倾角。

    参数：
        env：
            已经完成 rollout 的环境对象。
        interceptor_index：
            拦截弹索引，从 0 开始。
        length：
            期望的输出序列长度。

    返回：
        theta_trace：
            拦截弹航迹倾角序列，单位 rad。
    """
    # theta_traces：优先读取环境直接记录的每枚弹航迹倾角。
    theta_traces = getattr(env, "interceptor_theta_traces", [])

    if interceptor_index < len(theta_traces):
        theta_values = list(np.asarray(theta_traces[interceptor_index], dtype=np.float64)[:length])
    else:
        theta_values = []

    if len(theta_values) < length:
        # theta_values：缺失部分用 0 补齐，代表近似水平飞行。
        theta_values.extend([0.0] * (length - len(theta_values)))

    # theta_trace：转换为浮点数组并截断到目标长度。
    theta_trace = np.asarray(theta_values[:length], dtype=np.float64)

    return theta_trace


def plot_episode_summary(
    env: Any,
    save_path: str | Path,
    title: Optional[str] = None,
    show: bool = False,
) -> Path:
    """
    绘制并保存单回合综合诊断图。

    参数：
        env：
            已经完成 rollout 的环境对象。
        save_path：
            图片保存路径。
        title：
            可选图像标题。
        show：
            是否在保存后显示图像。

    返回：
        saved_path：
            实际保存的图片路径。
    """
    # saved_path：标准化后的输出路径。
    saved_path = Path(save_path)

    _ensure_parent_dir(saved_path)

    # time_trace：环境记录的仿真时间序列。
    time_trace = _to_array(getattr(env, "time_trace", []))

    # red_control：红方横向实际过载序列。
    red_control = _to_array(getattr(env, "red_inertial_control_trace", []))

    # interceptor_ny_actual_traces：每枚拦截弹纵向实际过载序列。
    interceptor_ny_actual_traces = [
        _to_array(trace)
        for trace in getattr(env, "interceptor_ny_actual_traces", [])
    ]

    # interceptor_nz_actual_traces：每枚拦截弹侧向实际过载序列。
    interceptor_nz_actual_traces = [
        _to_array(trace)
        for trace in getattr(env, "interceptor_nz_actual_traces", [])
    ]

    # distance_trace：红蓝相对距离采样序列。
    distance_trace = _to_array(getattr(env, "distance_trace", []))

    # reward_trace：单步奖励序列。
    reward_trace = _to_array(getattr(env, "reward_trace", []))

    if time_trace.size == 0:
        # axis_length：没有时间记录时，用最长曲线长度构造步数轴。
        axis_length = max(
            len(distance_trace),
            len(reward_trace),
            len(red_control),
            *[len(trace) for trace in interceptor_nz_actual_traces],
            *[len(trace) for trace in interceptor_ny_actual_traces],
        )

        # step_axis：步数横坐标。
        step_axis = np.arange(axis_length, dtype=np.float64)

        # x_label：横坐标标签。
        x_label = "Step"
    else:
        # step_axis：使用环境记录的真实仿真时间。
        step_axis = time_trace

        # x_label：横坐标标签。
        x_label = "Time (s)"

    # figure/axes：3×2 综合诊断图布局。
    figure, axes = plt.subplots(3, 2, figsize=(14, 11))

    if title is not None:
        figure.suptitle(title)

    # lateral_ax：红方横向过载和蓝方侧向过载对比。
    lateral_ax = axes[0, 0]
    lateral_ax.plot(
        _axis_for(red_control, step_axis),
        red_control,
        color=RED_COLOR,
        linewidth=1.6,
        label="Red nz actual",
    )

    for index, nz_actual in enumerate(interceptor_nz_actual_traces, start=1):
        # color：当前拦截弹曲线颜色。
        color = INTERCEPTOR_COLORS[(index - 1) % len(INTERCEPTOR_COLORS)]

        lateral_ax.plot(
            _axis_for(nz_actual, step_axis),
            nz_actual,
            color=color,
            linewidth=1.4,
            label=f"I{index} nz actual",
        )
    lateral_ax.set_xlabel(x_label)
    lateral_ax.set_ylabel("Lateral overload nz")
    lateral_ax.set_title("Lateral control")
    lateral_ax.grid(True)
    lateral_ax.legend(fontsize=8)

    # vertical_ax：蓝方纵向过载与平衡项对比。
    vertical_ax = axes[0, 1]

    for interceptor_index, ny_actual in enumerate(interceptor_ny_actual_traces):
        if ny_actual.size == 0:
            continue

        # color：当前拦截弹曲线颜色。
        color = INTERCEPTOR_COLORS[interceptor_index % len(INTERCEPTOR_COLORS)]

        # theta_trace：与该弹纵向控制序列对齐的航迹倾角。
        theta_trace = _interceptor_theta_trace(env, interceptor_index, len(ny_actual))

        # equilibrium_ny：纵向平衡项 cos(theta)。
        equilibrium_ny = np.cos(theta_trace)

        vertical_ax.plot(
            _axis_for(ny_actual, step_axis),
            ny_actual,
            color=color,
            linewidth=1.4,
            label=f"I{interceptor_index + 1} ny actual",
        )
        vertical_ax.plot(
            _axis_for(equilibrium_ny, step_axis),
            equilibrium_ny,
            color=color,
            linestyle="--",
            linewidth=0.9,
            label=f"I{interceptor_index + 1} cos(theta)",
        )

    vertical_ax.set_xlabel(x_label)
    vertical_ax.set_ylabel("Longitudinal overload ny")
    vertical_ax.set_title("Blue longitudinal control")
    vertical_ax.grid(True)
    vertical_ax.legend(fontsize=8)

    try:
        # red_trajectory/interceptor_trajectories：红方和全部拦截弹轨迹。
        red_trajectory, interceptor_trajectories = extract_multi_trajectories_from_env(env)

        draw_multi_trajectory_projection(
            ax=axes[1, 0],
            red_trajectory=red_trajectory,
            interceptor_trajectories=interceptor_trajectories,
            plane="xz",
            mark_closest=True,
            title="X-Z trajectory",
        )

        draw_multi_trajectory_projection(
            ax=axes[1, 1],
            red_trajectory=red_trajectory,
            interceptor_trajectories=interceptor_trajectories,
            plane="xy",
            mark_closest=True,
            title="X-Y trajectory",
        )
    except ValueError:
        # 轨迹为空时仍然保存其它诊断曲线。
        axes[1, 0].set_title("X-Z trajectory unavailable")
        axes[1, 0].grid(True)
        axes[1, 1].set_title("X-Y trajectory unavailable")
        axes[1, 1].grid(True)

    # distance_ax：红蓝相对距离曲线。
    distance_ax = axes[2, 0]
    distance_ax.plot(_axis_for(distance_trace, step_axis), distance_trace, label="Sampled distance")

    if distance_trace.size > 0:
        # sampled_index：采样点最小距离所在索引。
        sampled_index = int(np.argmin(distance_trace))

        # sampled_min：离散采样点最小距离。
        sampled_min = float(distance_trace[sampled_index])

        # env_min：环境记录的连续最近距离。
        env_min = float(getattr(env, "min_distance", sampled_min))

        distance_ax.scatter(
            _axis_for(distance_trace, step_axis)[sampled_index],
            sampled_min,
            marker="o",
            label=f"sampled min={sampled_min:.2f} m",
        )
        distance_ax.axhline(env_min, linestyle=":", linewidth=1.0, label=f"episode min={env_min:.2f} m")

    # kill_radius：杀伤半径，存在时画成参考线。
    kill_radius = getattr(getattr(env, "config", object()), "kill_radius", None)

    if kill_radius is not None:
        distance_ax.axhline(float(kill_radius), linestyle="--", linewidth=1.0, label="kill radius")

    distance_ax.set_xlabel(x_label)
    distance_ax.set_ylabel("Distance (m)")
    distance_ax.set_title("Relative distance")
    distance_ax.grid(True)
    distance_ax.legend(fontsize=8)

    # reward_ax：单步奖励曲线。
    reward_ax = axes[2, 1]
    reward_ax.plot(_axis_for(reward_trace, step_axis), reward_trace)
    reward_ax.set_xlabel(x_label)
    reward_ax.set_ylabel("Reward")
    reward_ax.set_title("Reward trace")
    reward_ax.grid(True)

    if title is None:
        figure.tight_layout()
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    figure.savefig(saved_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    return saved_path
