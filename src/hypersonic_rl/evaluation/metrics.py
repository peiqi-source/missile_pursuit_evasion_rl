"""
metrics.py

作用：
    统一管理强化学习突防任务中的评估指标。

当前支持：
    1. 单回合指标统计；
    2. 多回合指标汇总；
    3. 控制能耗计算；
    4. 指标保存为 CSV 文件。

工程说明：
    该文件不依赖 SAC 算法本身，只依赖环境中记录的轨迹、奖励和控制量。
    因此后续 SAC、LSTM-SAC、微分对策制导律版本都可以复用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def compute_control_energy(control_trace: Iterable[float], dt: float) -> float:
    """
    计算控制能耗近似指标。

    参数：
        control_trace：
            控制量序列，例如红方每一步实际侧向过载。

        dt：
            仿真步长，单位 s。

    返回：
        control_energy：
            控制能耗近似值。
            使用 sum(u^2) * dt 表示。
    """
    # control_array：转换后的控制量数组。
    control_array = np.asarray(list(control_trace), dtype=np.float64)

    if control_array.size == 0:
        return 0.0

    # control_energy：控制能耗近似值。
    control_energy = float(np.sum(np.square(control_array)) * float(dt))

    return control_energy


def compute_success(min_distance: float, kill_radius: float) -> float:
    """
    根据最小距离判断是否突防成功。

    参数：
        min_distance：
            当前回合红蓝双方最小距离，单位 m。

        kill_radius：
            杀伤半径，单位 m。

    返回：
        success：
            1.0 表示突防成功，0.0 表示突防失败。
    """
    # success：是否成功的浮点标志。
    success = 1.0 if float(min_distance) > float(kill_radius) else 0.0

    return success


def collect_episode_metrics(
    env: Any,
    episode_index: int,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从环境对象中收集单回合指标。

    参数：
        env：
            已经完成一个回合 rollout 的环境对象。
            该环境应包含 reward_trace、distance_trace、red_inertial_control_trace 等记录。

        episode_index：
            当前回合编号。

        extra_info：
            额外信息字典。
            例如训练时可以传入 global_step、actor_loss、critic_loss 等。

    返回：
        metrics：
            单回合指标字典。
    """
    # reward_trace：当前回合奖励序列。
    reward_trace = np.asarray(getattr(env, "reward_trace", []), dtype=np.float64)

    # distance_trace：当前回合距离序列。
    distance_trace = np.asarray(getattr(env, "distance_trace", []), dtype=np.float64)

    # red_control_trace：红方实际控制量序列。
    red_control_trace = getattr(env, "red_inertial_control_trace", [])

    # blue_control_trace：蓝方实际控制量序列。
    blue_control_trace = getattr(env, "blue_inertial_control_trace", [])

    # dt：环境仿真步长。
    dt = float(getattr(getattr(env, "config", object()), "dt", 0.0))

    # kill_radius：环境杀伤半径。
    kill_radius = float(getattr(getattr(env, "config", object()), "kill_radius", 0.0))

    # total_reward：当前回合累计奖励。
    total_reward = float(np.sum(reward_trace)) if reward_trace.size > 0 else 0.0

    # mean_reward：当前回合平均单步奖励。
    mean_reward = float(np.mean(reward_trace)) if reward_trace.size > 0 else 0.0

    # min_distance：当前回合最小距离。
    if distance_trace.size > 0:
        min_distance = float(np.min(distance_trace))
    else:
        min_distance = float(getattr(env, "min_distance", np.inf))

    # final_distance：当前回合最后一步距离。
    if distance_trace.size > 0:
        final_distance = float(distance_trace[-1])
    else:
        final_distance = min_distance

    # episode_steps：当前回合步数。
    episode_steps = int(getattr(env, "current_step", len(reward_trace)))

    # episode_time：当前回合仿真时间。
    episode_time = float(getattr(env, "current_time", episode_steps * dt))

    # red_control_energy：红方控制能耗近似值。
    red_control_energy = compute_control_energy(red_control_trace, dt)

    # blue_control_energy：蓝方控制能耗近似值。
    blue_control_energy = compute_control_energy(blue_control_trace, dt)

    # success：是否突防成功。
    success = compute_success(min_distance=min_distance, kill_radius=kill_radius)

    # metrics：单回合指标字典。
    metrics: Dict[str, Any] = {
        "episode": int(episode_index),
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "min_distance": min_distance,
        "final_distance": final_distance,
        "success": success,
        "episode_steps": episode_steps,
        "episode_time": episode_time,
        "red_control_energy": red_control_energy,
        "blue_control_energy": blue_control_energy,
    }

    if extra_info is not None:
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
            汇总后的统计结果。
    """
    if len(metric_records) == 0:
        return {
            "num_episodes": 0.0,
            "mean_total_reward": 0.0,
            "std_total_reward": 0.0,
            "mean_min_distance": 0.0,
            "std_min_distance": 0.0,
            "success_rate": 0.0,
            "mean_episode_steps": 0.0,
            "mean_red_control_energy": 0.0,
        }

    # dataframe：指标表格。
    dataframe = pd.DataFrame(metric_records)

    # summary：汇总指标字典。
    summary = {
        "num_episodes": float(len(dataframe)),
        "mean_total_reward": float(dataframe["total_reward"].mean()),
        "std_total_reward": float(dataframe["total_reward"].std(ddof=0)),
        "mean_min_distance": float(dataframe["min_distance"].mean()),
        "std_min_distance": float(dataframe["min_distance"].std(ddof=0)),
        "success_rate": float(dataframe["success"].mean()),
        "mean_episode_steps": float(dataframe["episode_steps"].mean()),
        "mean_red_control_energy": float(dataframe["red_control_energy"].mean()),
    }

    return summary


def save_metrics_to_csv(metric_records: List[Dict[str, Any]], output_path: str | Path) -> Path:
    """
    将指标记录保存为 CSV 文件。

    参数：
        metric_records：
            多回合指标字典列表。

        output_path：
            CSV 保存路径。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：输出文件路径。
    saved_path = Path(output_path)

    # parent_dir：输出文件父目录。
    parent_dir = saved_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # dataframe：待保存指标表。
    dataframe = pd.DataFrame(metric_records)

    dataframe.to_csv(saved_path, index=False, encoding="utf-8-sig")

    return saved_path