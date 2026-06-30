"""
run_interceptor_monte_carlo_hit_test.py

作用：
    对蓝方拦截弹初始位置和初始方向进行 Monte Carlo 拉偏打靶实验。

实验目标：
    1. 固定红方机动策略，例如平飞、不机动；
    2. 每个 episode 对拦截弹初始位置、初始航向角、初始弹道倾角进行随机拉偏；
    3. 运行完整追逃仿真；
    4. 统计蓝方拦截成功率、红方突防率、超时率、最小脱靶量分布。

典型用途：
    检查弱化后的 PN 制导律在初始误差条件下的鲁棒性；
    评估拦截弹初始位置和方向误差对命中率的影响；
    为后续 SAC 训练设置合理的课程难度。

建议放置位置：
    scripts/run_interceptor_monte_carlo_hit_test.py

运行示例：
    python scripts/run_interceptor_monte_carlo_hit_test.py --episodes 1000

    python scripts/run_interceptor_monte_carlo_hit_test.py ^
      --episodes 1000 ^
      --guidance-mode source_pn ^
      --scenario-profile paper_30km_radar_engagement ^
      --interceptor-count 2 ^
      --position-randomization-m 1000 ^
      --heading-randomization-deg 3 ^
      --theta-randomization-deg 1 ^
      --red-maneuver level_flight
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# =============================================================================
# 1. 项目路径处理
# =============================================================================

# PROJECT_ROOT：
#     假设本脚本位于 scripts/ 目录下；
#     Path(__file__).resolve().parents[1] 即项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# 将 src 加入 Python 搜索路径，保证可以直接运行：
#     python scripts/run_interceptor_monte_carlo_hit_test.py
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# 2. 导入项目环境
# =============================================================================

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig


# =============================================================================
# 3. 基础工具函数
# =============================================================================

def resolve_project_path(path: str | Path) -> Path:
    """
    将相对路径解析到项目根目录下。

    参数：
        path：
            相对路径或绝对路径。

    返回：
        resolved_path：
            绝对路径。
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_dict(path: str | Path) -> Dict[str, Any]:
    """
    读取 YAML 配置文件。

    参数：
        path：
            YAML 文件路径。

    返回：
        config_dict：
            YAML 顶层字典。
    """
    resolved_path = resolve_project_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"环境配置文件不存在：{resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        raise ValueError(f"YAML 顶层必须是字典：{resolved_path}")

    return dict(config_dict)


def to_jsonable(value: Any) -> Any:
    """
    将 numpy 类型转换成 JSON 可保存的 Python 原生类型。

    作用：
        csv/json 保存时，np.float64、np.int64、np.bool_ 可能无法直接序列化；
        这里统一转换。
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    return value


def safe_float(value: Any, default: float = math.nan) -> float:
    """
    安全转换成 float。

    参数：
        value：
            待转换对象。

        default：
            转换失败时返回的默认值。

    返回：
        result：
            float 值。
    """
    try:
        return float(value)
    except Exception:
        return float(default)


def percentile(values: np.ndarray, q: float) -> float:
    """
    计算百分位数。

    参数：
        values：
            数值数组。

        q：
            百分位，例如 50 表示中位数。

    返回：
        value：
            百分位数。
    """
    if values.size == 0:
        return math.nan

    return float(np.percentile(values, q))


# =============================================================================
# 4. 环境配置构造
# =============================================================================

def build_env_config_from_yaml_and_overrides(
    env_config_path: str | Path,
    overrides: Dict[str, Any],
) -> PursueEscapeEnvConfig:
    """
    读取 YAML，并用命令行参数覆盖部分环境字段。

    注意：
        这里不会修改原始 YAML 文件；
        只是本次实验运行时临时覆盖配置。

    参数：
        env_config_path：
            环境配置 YAML 路径。

        overrides：
            需要覆盖的字段字典。value=None 的字段会被忽略。

    返回：
        env_config：
            PursueEscapeEnvConfig 配置对象。
    """
    config_dict = load_yaml_dict(env_config_path)

    valid_fields = {field.name for field in fields(PursueEscapeEnvConfig)}

    unknown_keys = sorted(set(config_dict.keys()) - valid_fields)
    if unknown_keys:
        raise ValueError(
            "环境 YAML 中存在 PursueEscapeEnvConfig 不支持的字段：\n"
            f"{unknown_keys}\n"
            "请检查字段名是否拼写错误，或者是否忘记同步更新 PursueEscapeEnvConfig。"
        )

    for key, value in overrides.items():
        if value is None:
            continue

        if key not in valid_fields:
            raise ValueError(f"命令行覆盖字段不属于 PursueEscapeEnvConfig：{key}")

        config_dict[key] = value

    return PursueEscapeEnvConfig(**config_dict)


# =============================================================================
# 5. 红方固定机动策略
# =============================================================================

def red_level_flight(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """
    红方平飞，不做侧向机动。

    这是打靶实验最常用设置：
        用来单独评估蓝方拦截弹在初始误差条件下的制导命中能力。
    """
    return 0.0


def red_left_full(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """
    红方持续左向打满。
    """
    return float(max_overload)


def red_right_full(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """
    红方持续右向打满。
    """
    return -float(max_overload)


def red_sine(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """
    红方正弦机动。

    说明：
        这里使用较低频率，让红方形成连续横向机动；
        不是高频抖动。
    """
    amplitude = 0.75 * float(max_overload)
    frequency_hz = 0.08
    return float(amplitude * np.sin(2.0 * np.pi * frequency_hz * t))


def red_bangbang(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """
    红方 bang-bang 机动。

    每隔 switch_period 秒在 +max_overload 和 -max_overload 之间切换。
    """
    switch_period = 4.0
    segment_index = int(t // switch_period)
    return float(max_overload) if segment_index % 2 == 0 else -float(max_overload)


def red_random_uniform(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """
    红方每一步随机机动。

    注意：
        该模式不是“打靶”常规模式；
        更适合测试蓝方在随机突防动作下的鲁棒性。
    """
    return float(rng.uniform(-float(max_overload), float(max_overload)))


RED_MANEUVER_POLICIES = {
    "level_flight": red_level_flight,
    "left_full": red_left_full,
    "right_full": red_right_full,
    "sine": red_sine,
    "bangbang": red_bangbang,
    "random_uniform": red_random_uniform,
}


# =============================================================================
# 6. 单回合 Monte Carlo 仿真
# =============================================================================

def run_one_episode(
    env: PursueEscapeEnv,
    episode_index: int,
    base_seed: int,
    red_maneuver: str,
    max_steps_per_episode: Optional[int],
) -> Dict[str, Any]:
    """
    运行一次 Monte Carlo 打靶 episode。

    参数：
        env：
            追逃环境对象。

        episode_index：
            当前回合编号，从 1 开始。

        base_seed：
            基础随机种子。

        red_maneuver：
            红方机动模式。

        max_steps_per_episode：
            外部最大步数。
            None 表示使用 env.max_steps。

    返回：
        result：
            当前 episode 的统计结果。
    """
    # episode_seed：
    #     每个 episode 使用不同 seed；
    #     这样既能保证随机拉偏，又能复现实验。
    episode_seed = int(base_seed + episode_index * 10007)

    # rng：
    #     用于红方 random_uniform 这类外部机动策略；
    #     环境内部随机化由 env.reset(seed=episode_seed) 控制。
    rng = np.random.default_rng(episode_seed)

    # reset：
    #     环境内部会根据 initial_randomization_enabled、
    #     interceptor_position_randomization_m、
    #     interceptor_heading_randomization_deg、
    #     interceptor_theta_randomization_deg
    #     对拦截弹初始位置和方向进行随机拉偏。
    observation, reset_info = env.reset(seed=episode_seed)

    if red_maneuver not in RED_MANEUVER_POLICIES:
        raise ValueError(
            f"未知红方机动模式：{red_maneuver}，"
            f"可选：{list(RED_MANEUVER_POLICIES.keys())}"
        )

    policy = RED_MANEUVER_POLICIES[red_maneuver]

    if max_steps_per_episode is None:
        max_steps = int(getattr(env, "max_steps", 0))
        if max_steps <= 0:
            raise ValueError("env.max_steps 无效，请检查环境配置中的 t 和 dt。")
    else:
        max_steps = int(max_steps_per_episode)

    # -------------------------------------------------------------------------
    # 记录初始条件
    # -------------------------------------------------------------------------
    # 注意：
    #     reset_info 里已经有 interceptor_i_initial_x/y/z/psi；
    #     theta 没在 reset_info 里显式返回，所以这里直接从 env.interceptor_states 读取。
    initial_record: Dict[str, Any] = {}

    for i, state in enumerate(env.interceptor_states, start=1):
        prefix = f"interceptor_{i}"
        initial_record[f"{prefix}_initial_x"] = float(state[0])
        initial_record[f"{prefix}_initial_y"] = float(state[1])
        initial_record[f"{prefix}_initial_z"] = float(state[2])
        initial_record[f"{prefix}_initial_v"] = float(state[3])
        initial_record[f"{prefix}_initial_theta_deg"] = float(np.degrees(state[4]))
        initial_record[f"{prefix}_initial_psi_deg"] = float(np.degrees(state[5]))

        # 初始斜距：
        #     方便事后检查随机拉偏后是否仍在合理初始接战范围内。
        initial_record[f"{prefix}_initial_range"] = float(
            np.linalg.norm(state[:3] - env.red_state[:3])
        )

    # -------------------------------------------------------------------------
    # rollout 主循环
    # -------------------------------------------------------------------------
    total_reward = 0.0
    terminated = False
    truncated = False
    external_cutoff = False
    last_info: Dict[str, Any] = dict(reset_info)
    steps = 0

    for step in range(1, max_steps + 1):
        # 当前仿真时间。
        t = float(env.current_time)

        # 红方动作：
        #     对打靶实验，默认 red_maneuver=level_flight，即 action=0；
        #     如果要测试机动目标，则可以选择 left_full/right_full/sine/bangbang。
        action_value = policy(t, float(env.config.nzc_h_max), rng)

        # 动作形状必须是 [1]。
        action = np.array([action_value], dtype=np.float32)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        last_info = dict(info)
        steps = step

        if bool(terminated) or bool(truncated):
            break
    else:
        # 如果外部 max_steps_per_episode 比 env.max_steps 小，可能出现外部截断。
        external_cutoff = True

    # -------------------------------------------------------------------------
    # 终止结果解析
    # -------------------------------------------------------------------------
    termination_reason = str(
        last_info.get(
            "termination_reason",
            "external_max_steps" if external_cutoff else "unknown",
        )
    )

    # 对本打靶实验：
    #     intercept_success 表示蓝方打靶成功；
    #     red_penetration_success 表示红方突防成功。
    intercept_success = bool(last_info.get("intercepted", False))
    red_penetration_success = bool(last_info.get("passed", False))
    time_limit = bool(last_info.get("truncated", truncated))

    min_distance = safe_float(last_info.get("min_distance", env.min_distance))
    kill_radius = safe_float(getattr(env.config, "kill_radius", math.nan))
    miss_margin = min_distance - kill_radius

    result: Dict[str, Any] = {
        "episode": int(episode_index),
        "seed": int(episode_seed),
        "steps": int(steps),
        "final_time": float(last_info.get("time", env.current_time)),
        "termination_reason": termination_reason,

        # 蓝方视角：
        #     命中即打靶成功。
        "intercept_success": bool(intercept_success),

        # 红方视角：
        #     全部拦截弹错过即突防成功。
        "red_penetration_success": bool(red_penetration_success),

        "time_limit": bool(time_limit),
        "external_cutoff": bool(external_cutoff),

        "total_reward": float(total_reward),
        "min_distance": float(min_distance),
        "kill_radius": float(kill_radius),
        "miss_margin": float(miss_margin),

        "guidance_mode": str(env.config.guidance_mode),
        "scenario_profile": str(env.config.scenario_profile),
        "interceptor_count": int(env.config.interceptor_count),
        "red_maneuver": str(red_maneuver),
    }

    # 记录每枚弹的最小距离。
    for i in range(1, int(env.config.interceptor_count) + 1):
        key = f"interceptor_{i}_min_distance"
        if key in last_info:
            result[key] = safe_float(last_info[key])

    # 追加初始条件记录。
    result.update(initial_record)

    return result


# =============================================================================
# 7. 多回合结果统计
# =============================================================================

def summarize_results(
    results: List[Dict[str, Any]],
    env_config: PursueEscapeEnvConfig,
) -> Dict[str, Any]:
    """
    汇总 1000 次 Monte Carlo 打靶结果。

    参数：
        results：
            每个 episode 的结果列表。

        env_config：
            当前实验使用的环境配置。

    返回：
        summary：
            汇总统计字典。
    """
    if not results:
        raise ValueError("results 为空，无法统计。")

    n = len(results)

    intercept_success_count = sum(bool(item["intercept_success"]) for item in results)
    red_penetration_success_count = sum(bool(item["red_penetration_success"]) for item in results)
    time_limit_count = sum(bool(item["time_limit"]) for item in results)
    external_cutoff_count = sum(bool(item["external_cutoff"]) for item in results)

    termination_counter = Counter(str(item["termination_reason"]) for item in results)

    min_distances = np.asarray([float(item["min_distance"]) for item in results], dtype=np.float64)
    miss_margins = np.asarray([float(item["miss_margin"]) for item in results], dtype=np.float64)
    steps = np.asarray([float(item["steps"]) for item in results], dtype=np.float64)
    rewards = np.asarray([float(item["total_reward"]) for item in results], dtype=np.float64)

    summary = {
        "episodes": int(n),

        # 蓝方视角：打靶成功率。
        "intercept_success_count": int(intercept_success_count),
        "intercept_success_rate": float(intercept_success_count / n),

        # 红方视角：突防成功率。
        "red_penetration_success_count": int(red_penetration_success_count),
        "red_penetration_success_rate": float(red_penetration_success_count / n),

        # 超时 / 外部截断。
        "time_limit_count": int(time_limit_count),
        "time_limit_rate": float(time_limit_count / n),
        "external_cutoff_count": int(external_cutoff_count),
        "external_cutoff_rate": float(external_cutoff_count / n),

        "termination_reason_counts": dict(termination_counter),

        # 最小脱靶量统计。
        "min_distance_mean": float(np.mean(min_distances)),
        "min_distance_std": float(np.std(min_distances)),
        "min_distance_min": float(np.min(min_distances)),
        "min_distance_p05": percentile(min_distances, 5),
        "min_distance_p10": percentile(min_distances, 10),
        "min_distance_p50": percentile(min_distances, 50),
        "min_distance_p90": percentile(min_distances, 90),
        "min_distance_p95": percentile(min_distances, 95),
        "min_distance_max": float(np.max(min_distances)),

        # miss_margin = min_distance - kill_radius。
        # miss_margin < 0 表示进入杀伤半径；
        # miss_margin > 0 表示脱靶量超过杀伤半径。
        "miss_margin_mean": float(np.mean(miss_margins)),
        "miss_margin_p50": percentile(miss_margins, 50),
        "miss_margin_p90": percentile(miss_margins, 90),

        # episode 步数和奖励。
        "steps_mean": float(np.mean(steps)),
        "steps_std": float(np.std(steps)),
        "steps_min": int(np.min(steps)),
        "steps_max": int(np.max(steps)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),

        # 当前环境快照，方便复现实验。
        "env_snapshot": {
            "guidance_mode": str(env_config.guidance_mode),
            "scenario_profile": str(env_config.scenario_profile),
            "interceptor_count": int(env_config.interceptor_count),
            "interceptor_ability_profile": str(env_config.interceptor_ability_profile),
            "source_pn_max_overload": float(env_config.source_pn_max_overload),
            "N": float(env_config.N),
            "source_pn_compensation_gain": float(env_config.source_pn_compensation_gain),
            "interceptor_max_overload": float(env_config.interceptor_max_overload),
            "kill_radius": float(env_config.kill_radius),
            "dt": float(env_config.dt),
            "t": float(env_config.t),

            "initial_randomization_enabled": bool(env_config.initial_randomization_enabled),
            "interceptor_position_randomization_m": float(env_config.interceptor_position_randomization_m),
            "randomize_interceptor_y": bool(env_config.randomize_interceptor_y),
            "interceptor_heading_randomization_deg": float(env_config.interceptor_heading_randomization_deg),
            "interceptor_theta_randomization_deg": float(env_config.interceptor_theta_randomization_deg),

            "radar_detection_distance": float(env_config.radar_detection_distance),
            "red_intelligent_activation_mode": str(env_config.red_intelligent_activation_mode),
        },
    }

    return summary


# =============================================================================
# 8. 保存 CSV / JSON
# =============================================================================

def save_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    """
    保存每个 episode 的详细结果到 CSV。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for item in results:
        for key in item.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow({key: to_jsonable(item.get(key, "")) for key in fieldnames})


def save_json(summary: Dict[str, Any], json_path: Path) -> None:
    """
    保存汇总结果到 JSON。
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(summary), file, ensure_ascii=False, indent=2)


# =============================================================================
# 9. 命令行参数
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="对拦截弹初始位置和方向进行 Monte Carlo 拉偏打靶实验。"
    )

    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/env/override_weak_pn.yaml",
        help="环境配置 YAML 路径。",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Monte Carlo 回合数，默认 1000。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="基础随机种子。",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/monte_carlo_hit_test",
        help="输出目录。",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="本次实验名称。默认使用时间戳自动生成。",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="每隔多少个 episode 打印一次进度；设为 0 表示只打印最终结果。",
    )

    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="外部最大步数。默认 None，使用 env.max_steps。一般不建议手动设小。",
    )

    # -------------------------------------------------------------------------
    # 常用环境覆盖参数
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--guidance-mode",
        type=str,
        default=None,
        choices=["source_pn", "mid_terminal_interceptor", "paper_mid_terminal"],
        help="覆盖蓝方制导模式。",
    )

    parser.add_argument(
        "--scenario-profile",
        type=str,
        default='paper_200km_end_to_end',
        choices=[
            "paper_200km_end_to_end",
            "paper_30km_radar_engagement",
            "manual_pair",
            "custom",
        ],
        help="覆盖初始态势 profile。",
    )

    parser.add_argument(
        "--interceptor-count",
        type=int,
        default=None,
        choices=[1, 2],
        help="覆盖拦截弹数量。",
    )

    parser.add_argument(
        "--red-maneuver",
        type=str,
        default="level_flight",
        choices=list(RED_MANEUVER_POLICIES.keys()),
        help="红方固定机动策略。打靶实验默认 level_flight。",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="覆盖仿真步长 dt。",
    )

    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help="覆盖单回合最大仿真时间 t。",
    )

    parser.add_argument(
        "--kill-radius",
        type=float,
        default=None,
        help="覆盖杀伤半径。",
    )

    # -------------------------------------------------------------------------
    # PN 弱化 / 能力参数
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--source-pn-max-overload",
        type=float,
        default=None,
        help="覆盖 source_pn 最大过载。",
    )

    parser.add_argument(
        "--navigation-gain",
        type=float,
        default=None,
        help="覆盖 source_pn 导航系数 N。",
    )

    parser.add_argument(
        "--source-pn-compensation-gain",
        type=float,
        default=None,
        help="覆盖 source_pn 目标机动前馈补偿系数。",
    )

    parser.add_argument(
        "--interceptor-ability-profile",
        type=str,
        default=None,
        choices=["custom", "weak", "paper", "strong"],
        help="覆盖蓝方能力档位。",
    )

    # -------------------------------------------------------------------------
    # Monte Carlo 初始拉偏参数
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--position-randomization-m",
        type=float,
        default=1000.0,
        help="拦截弹初始位置拉偏范围，单位 m。每个方向默认 Uniform[-1000, 1000]。",
    )

    parser.add_argument(
        "--randomize-interceptor-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否对拦截弹初始高度 y 也拉偏。默认 False，只拉偏 x-z 平面。",
    )

    parser.add_argument(
        "--heading-randomization-deg",
        type=float,
        default=3.0,
        help="拦截弹初始航向角拉偏范围，单位 degree。默认 ±3°。",
    )

    parser.add_argument(
        "--theta-randomization-deg",
        type=float,
        default=1.0,
        help="拦截弹初始弹道倾角拉偏范围，单位 degree。默认 ±1°。",
    )

    parser.add_argument(
        "--no-interceptor-randomization",
        action="store_true",
        help="关闭拦截弹初始拉偏，用于固定基准对照。",
    )

    return parser.parse_args()


# =============================================================================
# 10. 主函数
# =============================================================================

def main() -> None:
    """
    主入口。
    """
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes 必须为正整数。")

    # -------------------------------------------------------------------------
    # 构造覆盖配置
    # -------------------------------------------------------------------------
    # 注意：
    #     为了做“只拉偏拦截弹”的 Monte Carlo，
    #     这里会强制 red_initial_psi_delta_min/max = 0，
    #     避免红方初始航向也跟着随机。
    #
    #     环境内部的拦截弹初始位置/方向随机化依赖 initial_randomization_enabled=True。
    # -------------------------------------------------------------------------
    if args.no_interceptor_randomization:
        initial_randomization_enabled = False
        position_randomization_m = 0.0
        heading_randomization_deg = 0.0
        theta_randomization_deg = 0.0
    else:
        initial_randomization_enabled = True
        position_randomization_m = float(args.position_randomization_m)
        heading_randomization_deg = float(args.heading_randomization_deg)
        theta_randomization_deg = float(args.theta_randomization_deg)

    overrides: Dict[str, Any] = {
        # 常用环境参数。
        "guidance_mode": args.guidance_mode,
        "scenario_profile": args.scenario_profile,
        "interceptor_count": args.interceptor_count,
        "dt": args.dt,
        "t": args.t,
        "kill_radius": args.kill_radius,

        # PN 参数。
        "source_pn_max_overload": args.source_pn_max_overload,
        "N": args.navigation_gain,
        "source_pn_compensation_gain": args.source_pn_compensation_gain,
        "interceptor_ability_profile": args.interceptor_ability_profile,

        # Monte Carlo 拉偏参数。
        "initial_randomization_enabled": initial_randomization_enabled,
        "interceptor_position_randomization_m": position_randomization_m,
        "randomize_interceptor_y": bool(args.randomize_interceptor_y),
        "interceptor_heading_randomization_deg": heading_randomization_deg,
        "interceptor_theta_randomization_deg": theta_randomization_deg,

        # 保证红方自身初始航向不参与随机拉偏。
        "red_initial_psi_delta_min_deg": 0.0,
        "red_initial_psi_delta_max_deg": 0.0,
    }

    env_config = build_env_config_from_yaml_and_overrides(
        env_config_path=args.env_config,
        overrides=overrides,
    )

    env = PursueEscapeEnv(env_config)

    # -------------------------------------------------------------------------
    # 输出目录
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"mc_hit_{env_config.guidance_mode}_{timestamp}"
    output_dir = resolve_project_path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "episodes.csv"
    json_path = output_dir / "summary.json"

    # -------------------------------------------------------------------------
    # 打印实验设置
    # -------------------------------------------------------------------------
    print("=" * 100)
    print("Monte Carlo 拦截弹初始拉偏打靶实验")
    print("=" * 100)
    print(f"环境配置文件              : {resolve_project_path(args.env_config)}")
    print(f"输出目录                  : {output_dir}")
    print(f"episodes                  : {args.episodes}")
    print(f"seed                      : {args.seed}")
    print("-" * 100)
    print(f"guidance_mode             : {env_config.guidance_mode}")
    print(f"scenario_profile          : {env_config.scenario_profile}")
    print(f"interceptor_count         : {env_config.interceptor_count}")
    print(f"red_maneuver              : {args.red_maneuver}")
    print(f"dt, t                     : {env_config.dt}, {env_config.t}")
    print(f"env.max_steps             : {getattr(env, 'max_steps', None)}")
    print(f"kill_radius               : {env_config.kill_radius}")
    print("-" * 100)
    print(f"source_pn_max_overload    : {env_config.source_pn_max_overload}")
    print(f"N                         : {env_config.N}")
    print(f"source_pn_compensation    : {env_config.source_pn_compensation_gain}")
    print("-" * 100)
    print(f"initial_randomization     : {env_config.initial_randomization_enabled}")
    print(f"position_randomization_m  : {env_config.interceptor_position_randomization_m}")
    print(f"randomize_interceptor_y   : {env_config.randomize_interceptor_y}")
    print(f"heading_randomization_deg : {env_config.interceptor_heading_randomization_deg}")
    print(f"theta_randomization_deg   : {env_config.interceptor_theta_randomization_deg}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Monte Carlo 主循环
    # -------------------------------------------------------------------------
    results: List[Dict[str, Any]] = []

    for episode_index in range(1, int(args.episodes) + 1):
        result = run_one_episode(
            env=env,
            episode_index=episode_index,
            base_seed=int(args.seed),
            red_maneuver=str(args.red_maneuver),
            max_steps_per_episode=args.max_steps_per_episode,
        )
        results.append(result)

        if args.print_every > 0 and (
            episode_index == 1 or episode_index % int(args.print_every) == 0
        ):
            hit_count = sum(bool(item["intercept_success"]) for item in results)
            hit_rate = hit_count / len(results)
            print(
                f"[{episode_index:05d}/{args.episodes:05d}] "
                f"hit_rate={hit_rate:.2%}, "
                f"last_reason={result['termination_reason']}, "
                f"last_min_distance={result['min_distance']:.3f} m, "
                f"last_steps={result['steps']}"
            )

    summary = summarize_results(results, env_config=env_config)

    save_csv(results, csv_path)
    save_json(summary, json_path)

    # -------------------------------------------------------------------------
    # 打印最终结果
    # -------------------------------------------------------------------------
    print("=" * 100)
    print("Monte Carlo 实验完成")
    print("=" * 100)
    print(
        f"蓝方拦截成功率 intercept_success_rate : "
        f"{summary['intercept_success_rate']:.2%} "
        f"({summary['intercept_success_count']}/{summary['episodes']})"
    )
    print(
        f"红方突防成功率 red_penetration_success_rate : "
        f"{summary['red_penetration_success_rate']:.2%} "
        f"({summary['red_penetration_success_count']}/{summary['episodes']})"
    )
    print(f"超时率 time_limit_rate                  : {summary['time_limit_rate']:.2%}")
    print(f"终止原因统计 termination_reason_counts  : {summary['termination_reason_counts']}")
    print("-" * 100)
    print(
        "最小脱靶量 min_distance："
        f"mean={summary['min_distance_mean']:.3f} m, "
        f"std={summary['min_distance_std']:.3f} m, "
        f"p50={summary['min_distance_p50']:.3f} m, "
        f"p90={summary['min_distance_p90']:.3f} m, "
        f"min={summary['min_distance_min']:.3f} m, "
        f"max={summary['min_distance_max']:.3f} m"
    )
    print(
        "脱靶裕度 miss_margin = min_distance - kill_radius："
        f"mean={summary['miss_margin_mean']:.3f} m, "
        f"p50={summary['miss_margin_p50']:.3f} m, "
        f"p90={summary['miss_margin_p90']:.3f} m"
    )
    print("-" * 100)
    print(f"每回合明细 CSV：{csv_path}")
    print(f"汇总结果 JSON：{json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()