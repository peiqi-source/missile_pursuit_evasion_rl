"""
reward.py

作用：
    统一管理追逃突防环境的奖励函数。

当前版本：
    默认奖励对齐论文第 5 章端到端 SAC 任务：
        1. 过程奖励约束红方横向过载；
        2. 过程奖励鼓励红方继续接近预设打击目标；
        3. 终端奖励同时考虑两枚拦截弹的连续最小脱靶量；
        4. 任一拦截弹进入杀伤半径即判定突防失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class RewardConfig:
    """
    RewardConfig

    作用：
        管理端到端突防奖励函数的全部系数。

    属性：
        stage_action_weight：
            红方横向过载过程惩罚系数，对应论文中的 k_n。
        stage_target_weight：
            红方接近预设打击目标的过程奖励系数，对应论文中的 k_r。
        terminal_large_miss_slope / terminal_large_miss_bias：
            脱靶量大于 ideal_miss_distance 时的终端奖励线性系数。
        terminal_ideal_miss_slope / terminal_ideal_miss_bias：
            脱靶量位于 [kill_radius, ideal_miss_distance] 时的终端奖励线性系数。
        terminal_failure_penalty：
            脱靶量小于 kill_radius 时的单枚弹失败惩罚。
        ideal_miss_distance：
            期望脱靶量，过小不稳，过大代表偏离任务弹道过多。
        kill_radius：
            拦截弹杀伤半径。
    """

    stage_action_weight: float = 0.03
    stage_target_weight: float = 0.002
    terminal_large_miss_slope: float = 0.02
    terminal_large_miss_bias: float = 50.6
    terminal_ideal_miss_slope: float = 1.0
    terminal_ideal_miss_bias: float = 20.0
    terminal_failure_penalty: float = 30.0
    ideal_miss_distance: float = 50.0
    kill_radius: float = 5.0


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
            使用 sum(u^2) * dt 得到的离散控制能耗。
    """
    # control_array：把输入序列统一转成浮点数组。
    control_array = np.asarray(list(control_trace), dtype=np.float64)

    if control_array.size == 0:
        return 0.0

    # control_energy：平方积分近似。
    control_energy = float(np.sum(np.square(control_array)) * float(dt))

    return control_energy


def compute_success(min_distance: float, kill_radius: float) -> float:
    """
    根据全局最小距离判断红方是否突防成功。

    参数：
        min_distance：
            所有拦截弹中的全局连续最小距离。
        kill_radius：
            杀伤半径。

    返回：
        success：
            1.0 表示未被任何拦截弹命中，0.0 表示被拦截。
    """
    # success：任一弹进入杀伤半径即失败。
    success = 1.0 if float(min_distance) > float(kill_radius) else 0.0

    return success


def _terminal_reward_for_miss_distance(miss_distance: float, config: RewardConfig) -> float:
    """
    计算单枚拦截弹脱靶量对应的终端奖励。

    参数：
        miss_distance：
            当前拦截弹的连续最小脱靶量。
        config：
            奖励函数配置。

    返回：
        terminal_reward：
            单枚弹终端奖励。
    """
    # miss：统一转成非负浮点数，避免异常输入污染奖励。
    miss = max(float(miss_distance), 0.0)

    if miss < config.kill_radius:
        # 失败区间：进入杀伤半径，给固定负奖励。
        return -float(config.terminal_failure_penalty)

    if miss <= config.ideal_miss_distance:
        # 理想区间：脱靶量越大越安全，但仍保持在任务可接受范围内。
        return (
            float(config.terminal_ideal_miss_slope) * miss
            + float(config.terminal_ideal_miss_bias)
        )

    # 过大区间：脱靶量过大代表红方偏离任务弹道，因此奖励开始下降。
    terminal_reward = (
        -float(config.terminal_large_miss_slope) * miss
        + float(config.terminal_large_miss_bias)
    )

    return float(terminal_reward)


def calculate_end_to_end_reward(
    red_lateral_overload: float,
    previous_target_distance: float,
    current_target_distance: float,
    interceptor_min_distances: List[float],
    terminated: bool,
    config: RewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    计算论文第 5 章端到端突防任务的单步奖励。

    参数：
        red_lateral_overload：
            红方一阶自动驾驶仪之后的实际横向过载。
        previous_target_distance：
            上一步红方到预设目标的距离。
        current_target_distance：
            当前红方到预设目标的距离。
        interceptor_min_distances：
            每枚拦截弹到当前为止的连续最小距离。
        terminated：
            当前回合是否自然终止。
        config：
            奖励配置。

    返回：
        reward：
            当前步总奖励。
        reward_info：
            奖励分项诊断字段。
    """
    # action_penalty：约束红方横向过载能耗。
    action_penalty = -float(config.stage_action_weight) * abs(float(red_lateral_overload))

    # target_progress：大于 0 表示红方正在接近预设打击目标。
    target_progress = float(previous_target_distance) - float(current_target_distance)

    # target_progress_reward：鼓励红方完成突防后仍朝任务目标前进。
    target_progress_reward = float(config.stage_target_weight) * target_progress

    # terminal_reward：只有自然终止时才评估脱靶量。
    terminal_reward = 0.0

    # terminal_rewards：逐枚拦截弹终端奖励，便于 debug 判断是哪一枚弹主导。
    terminal_rewards: List[float] = []

    if terminated:
        for miss_distance in interceptor_min_distances:
            # per_reward：单枚弹脱靶量对应的终端奖励。
            per_reward = _terminal_reward_for_miss_distance(
                miss_distance=miss_distance,
                config=config,
            )
            terminal_rewards.append(per_reward)

        # terminal_reward：多弹任务中把每枚弹贡献相加。
        terminal_reward = float(np.sum(terminal_rewards)) if terminal_rewards else 0.0

    # reward：总奖励。
    reward = action_penalty + target_progress_reward + terminal_reward

    # global_min_distance：所有拦截弹中的全局最小脱靶量。
    global_min_distance = min(interceptor_min_distances) if interceptor_min_distances else np.inf

    # reward_info：用于训练日志和调试 CSV 的奖励分项。
    reward_info: Dict[str, float] = {
        "reward": float(reward),
        "action_penalty": float(action_penalty),
        "target_progress_reward": float(target_progress_reward),
        "target_progress": float(target_progress),
        "previous_target_distance": float(previous_target_distance),
        "target_distance": float(current_target_distance),
        "terminal_reward": float(terminal_reward),
        "min_distance": float(global_min_distance),
        "intercepted_reward_flag": float(global_min_distance <= config.kill_radius),
    }

    for index, per_reward in enumerate(terminal_rewards, start=1):
        # interceptor_i_terminal_reward：每枚弹终端奖励贡献。
        reward_info[f"interceptor_{index}_terminal_reward"] = float(per_reward)

    return float(reward), reward_info
