"""
reward.py

作用：
    统一管理追逃突防环境的端到端奖励函数。

当前版本核心修改：
    1. 奖励函数显式接收 intercepted / all_passed / truncated / termination_reason；
    2. 终端奖励先按任务成败分支，再计算脱靶量 shaping；
    3. 任一弹命中时，另一枚弹的大脱靶量不再抵消失败惩罚；
    4. time_limit 不再是无奖励漏洞，默认给超时惩罚；
    5. 突防成功但脱靶量过大时，距离 shaping 被裁剪，避免 success=True 但累计终端奖励极端为负。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class RewardConfig:
    """
    RewardConfig

    作用：
        管理端到端突防奖励函数的全部系数。

    设计原则：
        1. 第一优先级：任务成败，即 intercepted / passed / time_limit；
        2. 第二优先级：突防后保持合理脱靶量，避免贴边被杀伤，也避免大幅偏离任务弹道；
        3. 第三优先级：过程阶段继续接近目标并减少不必要机动。
    """

    # ============================================================
    # 1. 过程奖励
    # ============================================================

    # 过程项：红方横向过载惩罚。
    # 使用 abs(u) 而不是 u^2，避免大动作惩罚过度压制探索。
    stage_action_weight: float = 0.03

    # 过程项：接近预设打击目标奖励。
    # target_progress = previous_target_distance - current_target_distance。
    # 正值表示红方更接近目标。
    stage_target_weight: float = 0.002

    # ============================================================
    # 2. 单枚弹脱靶量 shaping
    # ============================================================

    # 脱靶量过大区间的线性斜率。
    # 原始形式：-terminal_large_miss_slope * miss + terminal_large_miss_bias。
    terminal_large_miss_slope: float = 0.02

    # 脱靶量过大区间的线性偏置。
    terminal_large_miss_bias: float = 50.6

    # 理想脱靶量区间内的奖励斜率。
    terminal_ideal_miss_slope: float = 1.0

    # 理想脱靶量区间内的奖励偏置。
    terminal_ideal_miss_bias: float = 20.0

    # 单枚弹进入杀伤半径时的局部惩罚。
    # 注意：全局失败惩罚由 terminal_intercept_penalty 控制，这里只作为局部诊断/补充。
    terminal_failure_penalty: float = 30.0

    # 理想脱靶量，单位 m。
    # 小于该值时，认为是“安全但仍贴近任务弹道”的突防。
    ideal_miss_distance: float = 50.0

    # 单枚弹距离 shaping 的下限。
    # 作用：避免突防成功但 miss 很大时，线性负奖励压过 success_bonus。
    terminal_distance_reward_clip_min: float = -50.0

    # 单枚弹距离 shaping 的上限。
    # 作用：避免脱靶量 shaping 过强，盖过任务成败奖励。
    terminal_distance_reward_clip_max: float = 120.0

    # 多枚拦截弹脱靶量 shaping 是否取平均。
    # True：双弹和单弹奖励尺度更一致；False：沿用逐枚相加。
    use_mean_terminal_distance_reward: bool = True

    # ============================================================
    # 3. 全局终端成败奖励
    # ============================================================

    # 任一拦截弹进入杀伤半径时的全局失败惩罚。
    terminal_intercept_penalty: float = 250.0

    # 所有拦截弹均错过并自然终止时的全局成功奖励。
    terminal_success_bonus: float = 200.0

    # 达到时间上限但既未明确突防也未被拦截时的惩罚。
    # 作用：防止智能体学习“拖到超时”而不是完成突防。
    terminal_time_limit_penalty: float = 80.0

    # 杀伤半径，单位 m。
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
    control_array = np.asarray(list(control_trace), dtype=np.float64)

    if control_array.size == 0:
        return 0.0

    control_energy = float(np.sum(np.square(control_array)) * float(dt))

    return control_energy


def compute_success(min_distance: float, kill_radius: float) -> float:
    """
    根据全局最小距离判断红方是否未进入杀伤半径。

    注意：
        该函数只判断“是否未被命中”，不等价于完整任务成功。
        完整任务成功应由环境中的 termination_reason == "passed" 判断。
    """
    success = 1.0 if float(min_distance) > float(kill_radius) else 0.0

    return success


def _global_min_distance(interceptor_min_distances: List[float]) -> float:
    """
    计算所有拦截弹中的全局连续最小脱靶量。
    """
    if not interceptor_min_distances:
        return float(np.inf)

    return float(min(float(distance) for distance in interceptor_min_distances))


def _terminal_reward_for_miss_distance(miss_distance: float, config: RewardConfig) -> float:
    """
    计算单枚拦截弹脱靶量 shaping 奖励。

    该函数只表达“脱靶量质量”，不直接决定任务成败。
    任务成败由 calculate_end_to_end_reward() 中的 termination_reason 分支决定。
    """
    miss = max(float(miss_distance), 0.0)

    if miss <= float(config.kill_radius):
        # 命中区间：局部负奖励，只作为全局失败惩罚的补充。
        raw_reward = -float(config.terminal_failure_penalty)

    elif miss <= float(config.ideal_miss_distance):
        # 理想区间：脱靶量越大，安全裕度越高。
        raw_reward = (
            float(config.terminal_ideal_miss_slope) * miss
            + float(config.terminal_ideal_miss_bias)
        )

    else:
        # 过大区间：说明红方可能偏离任务弹道，奖励下降。
        raw_reward = (
            -float(config.terminal_large_miss_slope) * miss
            + float(config.terminal_large_miss_bias)
        )

    # 裁剪距离 shaping，避免其压过任务成败奖励。
    clipped_reward = float(
        np.clip(
            raw_reward,
            float(config.terminal_distance_reward_clip_min),
            float(config.terminal_distance_reward_clip_max),
        )
    )

    return clipped_reward


def _aggregate_terminal_distance_reward(
    interceptor_min_distances: List[float],
    config: RewardConfig,
) -> Tuple[float, List[float]]:
    """
    聚合多枚拦截弹的脱靶量 shaping 奖励。

    返回：
        terminal_distance_reward：
            聚合后的距离 shaping。
        terminal_rewards：
            每枚弹的单独 shaping，便于 info 诊断。
    """
    terminal_rewards = [
        _terminal_reward_for_miss_distance(
            miss_distance=miss_distance,
            config=config,
        )
        for miss_distance in interceptor_min_distances
    ]

    if not terminal_rewards:
        return 0.0, []

    if bool(config.use_mean_terminal_distance_reward):
        terminal_distance_reward = float(np.mean(terminal_rewards))
    else:
        terminal_distance_reward = float(np.sum(terminal_rewards))

    return terminal_distance_reward, terminal_rewards


def _normalize_termination_reason(
    terminated: bool,
    truncated: bool,
    intercepted: bool,
    all_passed: bool,
    termination_reason: Optional[str],
) -> str:
    """
    统一终止原因字符串，避免环境未传入 termination_reason 时奖励逻辑失效。
    """
    if termination_reason is not None:
        reason = str(termination_reason)
        if reason:
            return reason

    if bool(intercepted):
        return "intercepted"

    if bool(all_passed):
        return "passed"

    if bool(truncated):
        return "time_limit"

    if bool(terminated):
        # 兼容旧接口：terminated=True 但没有显式原因时，根据 flags 兜底。
        return "terminated_unknown"

    return "running"


def calculate_end_to_end_reward(
    red_lateral_overload: float,
    previous_target_distance: float,
    current_target_distance: float,
    interceptor_min_distances: List[float],
    terminated: bool,
    config: RewardConfig,
    *,
    truncated: bool = False,
    intercepted: Optional[bool] = None,
    all_passed: Optional[bool] = None,
    termination_reason: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    计算端到端突防任务的单步奖励。

    参数：
        red_lateral_overload：
            红方当前实际横向过载，单位 g。
        previous_target_distance/current_target_distance：
            红方更新前后到预设目标的距离，单位 m。
        interceptor_min_distances：
            每枚拦截弹截至当前时刻的连续最小脱靶量，单位 m。
        terminated：
            环境自然终止标志。
        truncated：
            环境时间上限截断标志。
        intercepted：
            是否任一拦截弹命中红方。
        all_passed：
            是否所有拦截弹均已错过红方。
        termination_reason：
            环境给出的终止原因：running / intercepted / passed / time_limit。
        config：
            奖励函数配置。

    返回：
        reward：
            当前步总奖励。
        reward_info：
            奖励诊断字典。

    关键逻辑：
        1. intercepted：全局强负奖励，且不允许其他弹的大脱靶量抵消失败；
        2. passed：全局成功奖励 + 有界脱靶量 shaping；
        3. time_limit：给超时惩罚，避免智能体学习拖时间；
        4. running：只计算过程奖励。
    """
    # ------------------------------------------------------------
    # 1. 过程奖励：控制能耗约束
    # ------------------------------------------------------------
    action_penalty = (
        -float(config.stage_action_weight)
        * abs(float(red_lateral_overload))
    )

    # ------------------------------------------------------------
    # 2. 过程奖励：接近预设打击目标
    # ------------------------------------------------------------
    target_progress = (
        float(previous_target_distance)
        - float(current_target_distance)
    )

    target_progress_reward = (
        float(config.stage_target_weight)
        * target_progress
    )

    # ------------------------------------------------------------
    # 3. 终止状态标准化
    # ------------------------------------------------------------
    global_min_distance = _global_min_distance(interceptor_min_distances)

    inferred_intercepted = bool(global_min_distance <= float(config.kill_radius))
    intercepted_flag = inferred_intercepted if intercepted is None else bool(intercepted)

    if all_passed is None:
        all_passed_flag = bool(terminated and not intercepted_flag and not truncated)
    else:
        all_passed_flag = bool(all_passed)

    reason = _normalize_termination_reason(
        terminated=bool(terminated),
        truncated=bool(truncated),
        intercepted=intercepted_flag,
        all_passed=all_passed_flag,
        termination_reason=termination_reason,
    )

    success = bool(reason == "passed")

    # ------------------------------------------------------------
    # 4. 终端奖励：任务成败优先
    # ------------------------------------------------------------
    terminal_reward = 0.0
    terminal_global_reward = 0.0
    terminal_distance_reward = 0.0
    terminal_rewards: List[float] = []

    if reason == "intercepted":
        # 任务失败：只给全局失败惩罚和命中局部惩罚。
        # 不聚合非命中弹的正向脱靶量奖励，避免“另一枚弹大脱靶”抵消失败。
        hit_count = int(
            sum(
                1
                for miss_distance in interceptor_min_distances
                if float(miss_distance) <= float(config.kill_radius)
            )
        )
        hit_count = max(hit_count, 1)

        terminal_global_reward = -float(config.terminal_intercept_penalty)
        terminal_distance_reward = -float(config.terminal_failure_penalty) * float(hit_count)
        terminal_reward = terminal_global_reward + terminal_distance_reward

    elif reason == "passed":
        # 任务成功：全局成功奖励优先，脱靶量 shaping 作为有界辅助项。
        terminal_distance_reward, terminal_rewards = _aggregate_terminal_distance_reward(
            interceptor_min_distances=interceptor_min_distances,
            config=config,
        )
        terminal_global_reward = float(config.terminal_success_bonus)
        terminal_reward = terminal_global_reward + terminal_distance_reward

    elif reason == "time_limit":
        # 未明确突防且达到时间上限：给惩罚，防止智能体把拖时间当作策略。
        terminal_global_reward = -float(config.terminal_time_limit_penalty)
        terminal_distance_reward = 0.0
        terminal_reward = terminal_global_reward

    elif bool(terminated):
        # 兼容未知自然终止：保守处理为弱失败，避免无意中奖励漏洞。
        terminal_global_reward = -0.5 * float(config.terminal_time_limit_penalty)
        terminal_distance_reward = 0.0
        terminal_reward = terminal_global_reward

    # ------------------------------------------------------------
    # 5. 总奖励
    # ------------------------------------------------------------
    reward = (
        action_penalty
        + target_progress_reward
        + terminal_reward
    )

    # ------------------------------------------------------------
    # 6. 奖励诊断信息
    # ------------------------------------------------------------
    reward_info: Dict[str, Any] = {
        "reward": float(reward),

        "action_penalty": float(action_penalty),
        "target_progress_reward": float(target_progress_reward),
        "target_progress": float(target_progress),
        "previous_target_distance": float(previous_target_distance),
        "target_distance": float(current_target_distance),

        "terminal_reward": float(terminal_reward),
        "terminal_distance_reward": float(terminal_distance_reward),
        "terminal_global_reward": float(terminal_global_reward),

        "min_distance": float(global_min_distance),
        "intercepted_reward_flag": float(intercepted_flag),
        "all_passed_reward_flag": float(all_passed_flag),
        "success_reward_flag": float(success),
        "truncated_reward_flag": float(bool(truncated)),
        "reward_termination_reason": reason,
    }

    for index, per_reward in enumerate(terminal_rewards, start=1):
        reward_info[f"interceptor_{index}_terminal_reward"] = float(per_reward)

    return float(reward), reward_info
