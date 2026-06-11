"""
metrics.py

作用：
    统一管理一对二突防任务的回合指标、汇总指标和 CSV 保存。

当前指标口径：
    1. 全局 min_distance 表示所有拦截弹中的连续最小距离；
    2. success=1 表示红方未被任一拦截弹命中；
    3. 拦截弹控制能耗使用 interceptor_* 命名；
    4. 每枚弹的脱靶量、能耗和终止状态都会独立写入 CSV。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def compute_control_energy(control_trace: Iterable[float], dt: float) -> float:
    """
    计算控制能耗近似值。

    参数：
        control_trace：
            控制量序列。
        dt：
            仿真步长，单位 s。

    返回：
        control_energy：
            sum(u^2) * dt 形式的离散能耗。
    """
    # control_array：把任意可迭代序列转成浮点数组。
    control_array = np.asarray(list(control_trace), dtype=np.float64)

    if control_array.size == 0:
        return 0.0

    # control_energy：平方积分近似。
    control_energy = float(np.sum(np.square(control_array)) * float(dt))

    return control_energy


def compute_absolute_control_effort(control_trace: Iterable[float], dt: float) -> float:
    """
    计算论文式控制消耗近似值。

    参数：
        control_trace：
            控制量序列。
        dt：
            仿真步长，单位 s。

    返回：
        control_effort：
            sum(abs(u)) * dt 形式的离散绝对过载积分。
    """
    # control_array：把任意可迭代序列转换成浮点数组。
    control_array = np.asarray(list(control_trace), dtype=np.float64)

    if control_array.size == 0:
        return 0.0

    # control_effort：绝对值积分，直接反映总机动使用量。
    control_effort = float(np.sum(np.abs(control_array)) * float(dt))

    return control_effort


def compute_success(min_distance: float, kill_radius: float) -> float:
    """
    根据全局最小距离判断红方是否突防成功。

    参数：
        min_distance：
            所有拦截弹中的连续最小距离。
        kill_radius：
            杀伤半径。

    返回：
        success：
            1.0 表示红方突防成功，0.0 表示被拦截。
    """
    # success：任一拦截弹进入杀伤半径即失败。
    success = 1.0 if float(min_distance) > float(kill_radius) else 0.0

    return success


def _finite_or_default(value: Any, default: float) -> float:
    """
    把输入值安全转换为有限浮点数。

    参数：
        value：
            待转换值。
        default：
            转换失败或非有限时使用的默认值。

    返回：
        result：
            有限浮点数。
    """
    try:
        # result：尝试转换为 float。
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(result):
        return float(default)

    return result


def _last_info(env: Any) -> Dict[str, Any]:
    """
    获取环境最后一步 info。

    参数：
        env：
            已经完成或正在运行的环境对象。

    返回：
        last_info：
            最后一步 info 字典；不存在时返回空字典。
    """
    # info_trace：环境逐步记录的诊断信息。
    info_trace = getattr(env, "info_trace", [])

    if not info_trace:
        return {}

    # last_info：复制最后一步，避免外部修改原记录。
    last_info = dict(info_trace[-1])

    return last_info


def _interceptor_count(env: Any, last_info: Dict[str, Any]) -> int:
    """
    推断当前回合拦截弹数量。

    参数：
        env：
            环境对象。
        last_info：
            最后一步诊断信息。

    返回：
        count：
            拦截弹数量。
    """
    # config_count：优先读取配置中的 interceptor_count。
    config_count = getattr(getattr(env, "config", object()), "interceptor_count", None)

    if config_count is not None:
        return int(config_count)

    # info_count：没有配置时读取 info。
    info_count = last_info.get("interceptor_count", 1)

    return int(info_count)


def _closest_distance_min(env: Any, sampled_min_distance: float) -> float:
    """
    从 info_trace 中提取连续最近距离最小值。

    参数：
        env：
            环境对象。
        sampled_min_distance：
            离散采样点上的兜底最小距离。

    返回：
        closest_distance：
            连续最近距离最小值。
    """
    # info_trace：环境逐步记录的诊断信息。
    info_trace = getattr(env, "info_trace", [])

    # closest_values：收集每一步全局连续最近距离。
    closest_values = [
        _finite_or_default(info.get("closest_distance_this_step"), np.inf)
        for info in info_trace
        if isinstance(info, dict)
    ]

    # finite_values：过滤 NaN 和 inf。
    finite_values = [value for value in closest_values if np.isfinite(value)]

    if finite_values:
        return float(min(finite_values))

    return float(sampled_min_distance)


def _info_trace_bool_ratio(env: Any, key: str) -> float:
    """
    统计 info_trace 中某个布尔字段为 True 的比例。

    参数：
        env：
            已完成 rollout 的环境对象。
        key：
            需要统计的 info 字段名。

    返回：
        ratio：
            字段存在且为 True 的步数比例；没有有效样本时返回 0。
    """
    # info_trace：环境逐步记录的诊断字段。
    info_trace = getattr(env, "info_trace", [])

    # values：只统计包含该字段的步，避免旧日志或非相关模式稀释比例。
    values = [
        bool(info[key])
        for info in info_trace
        if isinstance(info, dict) and key in info
    ]

    if not values:
        return 0.0

    return float(np.mean(values))


def _info_trace_first_finite(env: Any, key: str, default: float = np.nan) -> float:
    """
    从 info_trace 中读取某个字段第一次出现的有限数值。

    参数：
        env：
            已完成 rollout 的环境对象。
        key：
            需要读取的 info 字段名。
        default：
            没有有效值时的默认返回值。

    返回：
        value：
            第一个有限浮点值。
    """
    # info_trace：环境逐步记录的诊断字段。
    info_trace = getattr(env, "info_trace", [])

    for info in info_trace:
        if not isinstance(info, dict) or key not in info:
            continue
        value = _finite_or_default(info.get(key), np.nan)
        if np.isfinite(value):
            return float(value)

    return float(default)


def _compute_interceptor_energy(env: Any, dt: float) -> Dict[str, float]:
    """
    计算所有拦截弹的双通道控制能耗。

    参数：
        env：
            已完成 rollout 的环境。
        dt：
            仿真步长。

    返回：
        energy_info：
            总能耗、总 ny/nz 能耗和逐枚弹能耗字段。
    """
    # ny_traces/nz_traces/theta_traces：从环境读取每枚弹的 trace。
    ny_traces = getattr(env, "interceptor_ny_actual_traces", [])
    nz_traces = getattr(env, "interceptor_nz_actual_traces", [])
    theta_traces = getattr(env, "interceptor_theta_traces", [])

    # energy_info：逐步累加的能耗字段。
    energy_info: Dict[str, float] = {}

    # total_*：所有拦截弹聚合能耗。
    total_ny_energy = 0.0
    total_nz_energy = 0.0
    total_ny_effort_abs = 0.0
    total_nz_effort_abs = 0.0

    for index, (ny_trace, nz_trace) in enumerate(zip(ny_traces, nz_traces), start=1):
        # ny_array/nz_array：单枚弹双通道实际过载序列。
        ny_array = np.asarray(ny_trace, dtype=np.float64)
        nz_array = np.asarray(nz_trace, dtype=np.float64)

        if ny_array.size == 0 or nz_array.size == 0:
            ny_energy = 0.0
            nz_energy = 0.0
        else:
            # common_length：两通道共同有效长度。
            common_length = min(int(ny_array.size), int(nz_array.size))
            ny_array = ny_array[:common_length]
            nz_array = nz_array[:common_length]

            if index - 1 < len(theta_traces):
                # theta_array：该弹航迹倾角，用于扣除纵向平衡项 cos(theta)。
                theta_array = np.asarray(theta_traces[index - 1], dtype=np.float64)[:common_length]
            else:
                theta_array = np.zeros(common_length, dtype=np.float64)

            if theta_array.size < common_length:
                # theta_array：长度不足时补零，表示近似水平飞行。
                theta_array = np.pad(theta_array, (0, common_length - theta_array.size), constant_values=0.0)

            # ny_maneuver：纵向通道真正消耗机动能力的部分。
            ny_maneuver = ny_array - np.cos(theta_array)

            # ny_energy/nz_energy：单枚弹双通道能耗。
            ny_energy = compute_control_energy(ny_maneuver, dt)
            nz_energy = compute_control_energy(nz_array, dt)

            # ny_effort_abs/nz_effort_abs：单枚弹双通道论文式绝对过载积分。
            ny_effort_abs = compute_absolute_control_effort(ny_maneuver, dt)
            nz_effort_abs = compute_absolute_control_effort(nz_array, dt)

        if ny_array.size == 0 or nz_array.size == 0:
            # 空轨迹时绝对积分也置零，避免引用未定义变量。
            ny_effort_abs = 0.0
            nz_effort_abs = 0.0

        # energy_info：写入逐枚弹字段。
        energy_info[f"interceptor_{index}_ny_control_energy"] = float(ny_energy)
        energy_info[f"interceptor_{index}_nz_control_energy"] = float(nz_energy)
        energy_info[f"interceptor_{index}_control_energy"] = float(ny_energy + nz_energy)
        energy_info[f"interceptor_{index}_ny_control_effort_abs"] = float(ny_effort_abs)
        energy_info[f"interceptor_{index}_nz_control_effort_abs"] = float(nz_effort_abs)
        energy_info[f"interceptor_{index}_control_effort_abs"] = float(ny_effort_abs + nz_effort_abs)

        total_ny_energy += float(ny_energy)
        total_nz_energy += float(nz_energy)
        total_ny_effort_abs += float(ny_effort_abs)
        total_nz_effort_abs += float(nz_effort_abs)

    # 聚合字段：训练曲线默认读取这些字段。
    energy_info["interceptor_ny_control_energy"] = float(total_ny_energy)
    energy_info["interceptor_nz_control_energy"] = float(total_nz_energy)
    energy_info["interceptor_control_energy"] = float(total_ny_energy + total_nz_energy)
    energy_info["interceptor_ny_control_effort_abs"] = float(total_ny_effort_abs)
    energy_info["interceptor_nz_control_effort_abs"] = float(total_nz_effort_abs)
    energy_info["interceptor_control_effort_abs"] = float(total_ny_effort_abs + total_nz_effort_abs)

    return energy_info


def collect_episode_metrics(
    env: Any,
    episode_index: int,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从环境对象收集单回合指标。

    参数：
        env：
            已完成一个 rollout 的环境对象。
        episode_index：
            当前回合编号。
        extra_info：
            额外写入指标表的训练信息。

    返回：
        metrics：
            单回合指标字典。
    """
    # reward_trace：当前回合每一步奖励序列。
    reward_trace = np.asarray(getattr(env, "reward_trace", []), dtype=np.float64)

    # distance_trace：所有拦截弹中的当前最近距离序列。
    distance_trace = np.asarray(getattr(env, "distance_trace", []), dtype=np.float64)

    # red_control_trace：红方一阶实际横向过载序列。
    red_control_trace = getattr(env, "red_inertial_control_trace", [])

    # env_config：环境配置对象。
    env_config = getattr(env, "config", object())

    # dt/kill_radius/guidance_mode：基础评估参数。
    dt = float(getattr(env_config, "dt", 0.0))
    kill_radius = float(getattr(env_config, "kill_radius", 0.0))
    guidance_mode = str(getattr(env_config, "guidance_mode", "unknown"))

    # total_reward/mean_reward：回合奖励统计。
    total_reward = float(np.sum(reward_trace)) if reward_trace.size > 0 else 0.0
    mean_reward = float(np.mean(reward_trace)) if reward_trace.size > 0 else 0.0

    # sampled_min_distance：离散采样点最小距离。
    sampled_min_distance = float(np.min(distance_trace)) if distance_trace.size > 0 else np.inf

    # env_min_distance：环境内部连续最小距离。
    env_min_distance = _finite_or_default(getattr(env, "min_distance", np.inf), np.inf)

    # min_distance：评估口径优先使用连续最小距离。
    min_distance = min(env_min_distance, sampled_min_distance)
    if not np.isfinite(min_distance):
        min_distance = 0.0

    # final_distance：最后一步全局最近距离。
    final_distance = float(distance_trace[-1]) if distance_trace.size > 0 else min_distance

    # episode_steps/episode_time：回合长度。
    episode_steps = int(getattr(env, "current_step", len(reward_trace)))
    episode_time = float(getattr(env, "current_time", episode_steps * dt))

    # red_control_energy：红方控制能耗。
    red_control_energy = compute_control_energy(red_control_trace, dt)

    # red_control_effort_abs：论文式红方绝对过载积分。
    red_control_effort_abs = compute_absolute_control_effort(red_control_trace, dt)

    # target_offset：终端时刻红方到预设目标点的三维偏差。
    red_state = np.asarray(getattr(env, "red_state", np.zeros(3)), dtype=np.float64)
    target_position = np.asarray(getattr(env, "target_position", np.zeros(3)), dtype=np.float64)
    target_delta = red_state[:3] - target_position[:3]
    target_offset = float(np.linalg.norm(target_delta))

    # interceptor_energy：所有拦截弹控制能耗。
    interceptor_energy = _compute_interceptor_energy(env, dt)

    # last_info：最后一步诊断字段。
    last_info = _last_info(env)

    # success：优先采用环境自然终止给出的突防结果，避免时间截断被误判为成功。
    if "success" in last_info:
        success = float(bool(last_info["success"]))
    else:
        success = compute_success(min_distance=min_distance, kill_radius=kill_radius)

    # closest_distance_min：连续最近距离兜底统计。
    closest_distance_min = min(_closest_distance_min(env, sampled_min_distance), min_distance)

    # count：当前拦截弹数量。
    count = _interceptor_count(env, last_info)

    # metrics：单回合指标字典。
    metrics: Dict[str, Any] = {
        "episode": int(episode_index),
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "min_distance": float(min_distance),
        "sampled_min_distance": float(sampled_min_distance) if np.isfinite(sampled_min_distance) else float(min_distance),
        "closest_distance_min": float(closest_distance_min),
        "final_distance": final_distance,
        "success": success,
        "intercepted": float(bool(last_info.get("intercepted", min_distance <= kill_radius))),
        "passed": float(bool(last_info.get("passed", False))),
        "episode_steps": episode_steps,
        "episode_time": episode_time,
        "red_control_energy": red_control_energy,
        "red_control_effort_abs": red_control_effort_abs,
        "target_offset": target_offset,
        "target_offset_x": float(target_delta[0]),
        "target_offset_y": float(target_delta[1]),
        "target_offset_z": float(target_delta[2]),
        **interceptor_energy,
        "guidance_mode": str(last_info.get("guidance_mode", guidance_mode)),
        "termination_reason": str(last_info.get("termination_reason", "unknown")),
        "threat_interceptor_id": int(last_info.get("threat_interceptor_id", 0)),
        "interceptor_count": int(count),
    }

    for index in range(1, count + 1):
        # prefix：逐枚弹字段前缀。
        prefix = f"interceptor_{index}"

        metrics[f"{prefix}_min_distance"] = _finite_or_default(
            last_info.get(f"{prefix}_min_distance"),
            np.inf,
        )
        metrics[f"{prefix}_miss_distance"] = metrics[f"{prefix}_min_distance"]
        metrics[f"{prefix}_final_distance"] = _finite_or_default(
            last_info.get(f"{prefix}_distance"),
            np.inf,
        )
        metrics[f"{prefix}_intercepted"] = float(bool(last_info.get(f"{prefix}_intercepted", False)))
        metrics[f"{prefix}_passed"] = float(bool(last_info.get(f"{prefix}_passed", False)))
        metrics[f"{prefix}_phase_final"] = str(last_info.get(f"{prefix}_phase", "unknown"))
        metrics[f"{prefix}_pass_time"] = _finite_or_default(
            last_info.get(f"{prefix}_pass_time"),
            _info_trace_first_finite(env, f"{prefix}_pass_time"),
        )
        metrics[f"{prefix}_intercept_time"] = _finite_or_default(
            last_info.get(f"{prefix}_intercept_time"),
            _info_trace_first_finite(env, f"{prefix}_intercept_time"),
        )
        metrics[f"{prefix}_phase_switch_time"] = _finite_or_default(
            last_info.get(f"{prefix}_phase_switch_time"),
            _info_trace_first_finite(env, f"{prefix}_phase_switch_time"),
        )
        metrics[f"{prefix}_command_saturation_ratio"] = _info_trace_bool_ratio(
            env,
            f"{prefix}_command_saturated",
        )
        metrics[f"{prefix}_autopilot_rate_saturation_ratio"] = _info_trace_bool_ratio(
            env,
            f"{prefix}_autopilot_rate_saturated",
        )
        metrics[f"{prefix}_autopilot_output_saturation_ratio"] = _info_trace_bool_ratio(
            env,
            f"{prefix}_autopilot_output_saturated",
        )
        metrics[f"{prefix}_terminal_fallback_ratio"] = _info_trace_bool_ratio(
            env,
            f"{prefix}_terminal_fallback",
        )

    # 全局饱和比例：对所有拦截弹取平均，便于快速比较蓝方能力档位。
    if count > 0:
        metrics["interceptor_command_saturation_ratio"] = float(
            np.mean([metrics[f"interceptor_{index}_command_saturation_ratio"] for index in range(1, count + 1)])
        )
        metrics["interceptor_autopilot_rate_saturation_ratio"] = float(
            np.mean([metrics[f"interceptor_{index}_autopilot_rate_saturation_ratio"] for index in range(1, count + 1)])
        )
        metrics["interceptor_autopilot_output_saturation_ratio"] = float(
            np.mean([metrics[f"interceptor_{index}_autopilot_output_saturation_ratio"] for index in range(1, count + 1)])
        )
    else:
        metrics["interceptor_command_saturation_ratio"] = 0.0
        metrics["interceptor_autopilot_rate_saturation_ratio"] = 0.0
        metrics["interceptor_autopilot_output_saturation_ratio"] = 0.0

    if extra_info is not None:
        # metrics：合并训练器传入的额外指标。
        metrics.update(extra_info)

    return metrics


def summarize_metrics(metric_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    汇总多回合指标。

    参数：
        metric_records：
            多个回合的指标字典列表。

    返回：
        summary：
            多回合均值、标准差、成功率和能耗汇总。
    """
    if len(metric_records) == 0:
        return {
            "num_episodes": 0.0,
            "mean_total_reward": 0.0,
            "std_total_reward": 0.0,
            "mean_min_distance": 0.0,
            "std_min_distance": 0.0,
            "mean_final_distance": 0.0,
            "success_rate": 0.0,
            "intercept_rate": 0.0,
            "mean_episode_steps": 0.0,
            "mean_red_control_energy": 0.0,
            "mean_red_control_effort_abs": 0.0,
            "mean_target_offset": 0.0,
            "mean_interceptor_control_energy": 0.0,
            "mean_interceptor_ny_control_energy": 0.0,
            "mean_interceptor_nz_control_energy": 0.0,
            "mean_interceptor_control_effort_abs": 0.0,
            "mean_interceptor_command_saturation_ratio": 0.0,
            "mean_interceptor_autopilot_rate_saturation_ratio": 0.0,
            "mean_interceptor_autopilot_output_saturation_ratio": 0.0,
        }

    # dataframe：转换成表格便于统一统计。
    dataframe = pd.DataFrame(metric_records)

    def mean_or_zero(column: str) -> float:
        """
        安全计算某一列均值。

        参数：
            column：
                待统计列名。

        返回：
            mean_value：
                列存在时返回均值，否则返回 0。
        """
        if column not in dataframe.columns:
            return 0.0

        # mean_value：该列均值。
        mean_value = float(dataframe[column].mean())

        return mean_value

    # summary：多回合聚合指标。
    summary = {
        "num_episodes": float(len(dataframe)),
        "mean_total_reward": mean_or_zero("total_reward"),
        "std_total_reward": float(dataframe["total_reward"].std(ddof=0)) if "total_reward" in dataframe else 0.0,
        "mean_min_distance": mean_or_zero("min_distance"),
        "std_min_distance": float(dataframe["min_distance"].std(ddof=0)) if "min_distance" in dataframe else 0.0,
        "mean_final_distance": mean_or_zero("final_distance"),
        "success_rate": mean_or_zero("success"),
        "intercept_rate": mean_or_zero("intercepted"),
        "mean_episode_steps": mean_or_zero("episode_steps"),
        "mean_red_control_energy": mean_or_zero("red_control_energy"),
        "mean_red_control_effort_abs": mean_or_zero("red_control_effort_abs"),
        "mean_target_offset": mean_or_zero("target_offset"),
        "mean_interceptor_control_energy": mean_or_zero("interceptor_control_energy"),
        "mean_interceptor_ny_control_energy": mean_or_zero("interceptor_ny_control_energy"),
        "mean_interceptor_nz_control_energy": mean_or_zero("interceptor_nz_control_energy"),
        "mean_interceptor_control_effort_abs": mean_or_zero("interceptor_control_effort_abs"),
        "mean_interceptor_command_saturation_ratio": mean_or_zero("interceptor_command_saturation_ratio"),
        "mean_interceptor_autopilot_rate_saturation_ratio": mean_or_zero("interceptor_autopilot_rate_saturation_ratio"),
        "mean_interceptor_autopilot_output_saturation_ratio": mean_or_zero("interceptor_autopilot_output_saturation_ratio"),
    }

    # per-interceptor：如果存在逐枚弹字段，也写入汇总。
    for column in dataframe.columns:
        if column.startswith("interceptor_") and column.endswith("_min_distance"):
            summary[f"mean_{column}"] = mean_or_zero(column)
        if column.startswith("interceptor_") and column.endswith("_control_energy"):
            summary[f"mean_{column}"] = mean_or_zero(column)
        if column.startswith("interceptor_") and column.endswith("_control_effort_abs"):
            summary[f"mean_{column}"] = mean_or_zero(column)
        if column.startswith("interceptor_") and column.endswith("_saturation_ratio"):
            summary[f"mean_{column}"] = mean_or_zero(column)
        if column.startswith("interceptor_") and column.endswith("_pass_time"):
            summary[f"mean_{column}"] = mean_or_zero(column)
        if column.startswith("interceptor_") and column.endswith("_phase_switch_time"):
            summary[f"mean_{column}"] = mean_or_zero(column)

    return summary


def save_metrics_to_csv(metric_records: List[Dict[str, Any]], output_path: str | Path) -> Path:
    """
    将指标记录保存为 CSV 文件。

    参数：
        metric_records：
            指标字典列表。
        output_path：
            CSV 输出路径。

    返回：
        saved_path：
            实际保存的 CSV 路径。
    """
    # saved_path：标准化输出路径。
    saved_path = Path(output_path)

    # parent_dir：确保输出目录存在。
    parent_dir = saved_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # dataframe：把指标记录转换为表格。
    dataframe = pd.DataFrame(metric_records)

    # CSV：使用 utf-8-sig 便于中文表头在表格软件中打开。
    dataframe.to_csv(saved_path, index=False, encoding="utf-8-sig")

    return saved_path
