"""
record_radar_activation_snapshot.py

作用：
    记录“红方雷达刚开始探测到拦截弹 / 红方智能突防刚激活”这一刻的红蓝双方状态。

典型用途：
    1. 先让红方全程平飞，即 action = 0；
    2. 拦截弹按当前配置的制导律飞行，例如 source_pn / mid_terminal_interceptor / paper_mid_terminal；
    3. 从远程场景开始仿真，例如 paper_200km_end_to_end；
    4. 当任一拦截弹进入 radar_detection_distance，例如 30 km，记录该时刻红方和所有拦截弹的信息；
    5. 多次 Monte Carlo 后，得到“30 km 雷达接管截面”的状态分布，用于后续构造 30 km 训练环境。

建议放置位置：
    scripts/record_radar_activation_snapshot.py

推荐运行方式：
    python scripts/record_radar_activation_snapshot.py ^
      --episodes 1000 ^
      --env-config configs/env/pursue_escape_env_200km_radar_gate.yaml ^
      --guidance-mode source_pn ^
      --interceptor-count 2 ^
      --source-pn-max-overload 4 ^
      --navigation-gain 2.5 ^
      --source-pn-compensation-gain 0 ^
      --position-randomization-m 1000 ^
      --heading-randomization-deg 3 ^
      --theta-randomization-deg 1

输出：
    outputs/radar_activation_snapshots/<run_name>/snapshots.csv
        每个 episode 在雷达激活时刻的红蓝状态。

    outputs/radar_activation_snapshots/<run_name>/summary.json
        雷达激活成功率、激活时间、激活距离、相对速度等统计量。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# =============================================================================
# 1. 项目路径设置
# =============================================================================

# 本脚本假设放在 scripts/ 目录下。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.envs.dynamics import build_velocity_vector, compute_relative_geometry
from hypersonic_rl.utils import (
    build_dataclass_config,
    load_config_from_project,
    load_config_stack_from_project,
)


# =============================================================================
# 2. 通用工具函数
# =============================================================================

def resolve_project_path(path: str | Path) -> Path:
    """
    将项目相对路径转换为绝对路径。

    参数：
        path:
            相对路径或绝对路径。

    返回：
        Path:
            绝对路径。
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_jsonable(value: Any) -> Any:
    """
    将 numpy 类型转换成 JSON / CSV 更友好的 Python 原生类型。
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def safe_float(value: Any, default: float = math.nan) -> float:
    """
    安全转 float。
    """
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def rad_to_deg(value: float) -> float:
    """
    弧度转角度。
    """
    return float(np.degrees(float(value)))


def percentile(values: List[float], q: float) -> float:
    """
    计算百分位数。
    """
    finite_values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite_values.size == 0:
        return math.nan
    return float(np.percentile(finite_values, q))


def make_output_paths(output_dir: str | Path, run_name: Optional[str]) -> Dict[str, Path]:
    """
    构造本次实验输出路径。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_run_name = run_name or f"radar_activation_{timestamp}"
    run_dir = resolve_project_path(output_dir) / actual_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "csv_path": run_dir / "snapshots.csv",
        "summary_path": run_dir / "summary.json",
    }


# =============================================================================
# 3. 环境构造
# =============================================================================

def build_env_config(args: argparse.Namespace) -> PursueEscapeEnvConfig:
    """
    读取环境 YAML，并用命令行参数覆盖拦截弹和场景配置。

    重要说明：
        本脚本用于记录“雷达开始时刻”，因此默认将 red_intelligent_activation_mode
        覆盖为 radar_range_gate。这样红方在 30 km 之外始终不机动。
    """
    env_config_dict = load_config_stack_from_project(args.env_config)

    # overrides 中 value 为 None 的字段不会覆盖 YAML。
    overrides: Dict[str, Any] = {
        # ------------------------------------------------------------
        # 场景与门控
        # ------------------------------------------------------------
        "scenario_profile": args.scenario_profile,
        "guidance_mode": args.guidance_mode,
        "interceptor_count": args.interceptor_count,
        "radar_detection_distance": args.radar_detection_distance,

        # 强制红方只在雷达范围内激活；但本脚本红方 action 始终为 0，
        # 所以这个开关主要用于准确记录 red_intelligent_activation_time。
        "red_intelligent_activation_mode": "radar_range_gate",

        # ------------------------------------------------------------
        # 时间与自动驾驶仪
        # ------------------------------------------------------------
        "dt": args.dt,
        "t": args.t,
        "tau_i": args.tau_i,
        "interceptor_autopilot_rate_limit": args.interceptor_autopilot_rate_limit,

        # ------------------------------------------------------------
        # 红方：固定平飞，禁止红方初始航向随机拉偏
        # ------------------------------------------------------------
        "red_initial_psi_delta_min_deg": 0.0,
        "red_initial_psi_delta_max_deg": 0.0,
        "red_mach": args.red_mach,

        # ------------------------------------------------------------
        # 拦截弹速度、数量、能力
        # ------------------------------------------------------------
        "interceptor_mach": args.interceptor_mach,
        "interceptor_ability_profile": args.interceptor_ability_profile,
        "source_pn_max_overload": args.source_pn_max_overload,
        "N": args.navigation_gain,
        "source_pn_compensation_gain": args.source_pn_compensation_gain,

        # ------------------------------------------------------------
        # 拦截弹初始位置/方向拉偏
        # ------------------------------------------------------------
        "initial_randomization_enabled": not args.no_initial_randomization,
        "interceptor_position_randomization_m": args.position_randomization_m,
        "randomize_interceptor_y": args.randomize_interceptor_y,
        "interceptor_heading_randomization_deg": args.heading_randomization_deg,
        "interceptor_theta_randomization_deg": args.theta_randomization_deg,

        # paper profile 常用几何参数
        "paper_interceptor_x_distance": args.paper_interceptor_x_distance,
        "paper_interceptor_lateral_offset": args.paper_interceptor_lateral_offset,
        "paper_interceptor_y_offset_from_red": args.paper_interceptor_y_offset_from_red,
        "paper_interceptor_theta_deg": args.paper_interceptor_theta_deg,

        # 终止判据参数也可以覆盖，便于记录前检查
        "kill_radius": args.kill_radius,
    }

    # 删除 value 为 None 的覆盖项，避免误把 YAML 字段覆盖成 None。
    overrides = {key: value for key, value in overrides.items() if value is not None}

    env_config = build_dataclass_config(
        env_config_dict,
        PursueEscapeEnvConfig,
        overrides=overrides,
    )

    return env_config


# =============================================================================
# 4. 状态记录函数
# =============================================================================

def flatten_vehicle_state(prefix: str, state: np.ndarray) -> Dict[str, float]:
    """
    将单个飞行器状态向量展开成 CSV 字段。

    状态向量约定：
        state = [x, y, z, v, theta, psi, nx, ny, nz]
    """
    velocity = build_velocity_vector(state)

    return {
        f"{prefix}_x_m": float(state[0]),
        f"{prefix}_y_m": float(state[1]),
        f"{prefix}_z_m": float(state[2]),
        f"{prefix}_speed_mps": float(state[3]),
        f"{prefix}_theta_rad": float(state[4]),
        f"{prefix}_theta_deg": rad_to_deg(state[4]),
        f"{prefix}_psi_rad": float(state[5]),
        f"{prefix}_psi_deg": rad_to_deg(state[5]),
        f"{prefix}_nx_g": float(state[6]),
        f"{prefix}_ny_g": float(state[7]),
        f"{prefix}_nz_g": float(state[8]),
        f"{prefix}_vx_mps": float(velocity[0]),
        f"{prefix}_vy_mps": float(velocity[1]),
        f"{prefix}_vz_mps": float(velocity[2]),
    }


def add_interceptor_relative_fields(
    row: Dict[str, Any],
    env: PursueEscapeEnv,
    interceptor_index: int,
    interceptor_state: np.ndarray,
) -> None:
    """
    记录某一枚拦截弹相对红方的几何量。
    """
    prefix = f"interceptor_{interceptor_index}"
    rel = compute_relative_geometry(
        red_state=env.red_state,
        interceptor_state=interceptor_state,
    )

    row[f"{prefix}_relative_dx_m"] = float(rel["dx"])
    row[f"{prefix}_relative_dy_m"] = float(rel["dy"])
    row[f"{prefix}_relative_dz_m"] = float(rel["dz"])
    row[f"{prefix}_relative_dxdt_mps"] = float(rel["dxdt"])
    row[f"{prefix}_relative_dydt_mps"] = float(rel["dydt"])
    row[f"{prefix}_relative_dzdt_mps"] = float(rel["dzdt"])
    row[f"{prefix}_distance_m"] = float(rel["distance"])
    row[f"{prefix}_range_rate_mps"] = float(rel["range_rate"])
    row[f"{prefix}_closing_speed_mps"] = float(rel["closing_speed"])
    row[f"{prefix}_los_angle_rad"] = float(rel["los_angle"])
    row[f"{prefix}_los_angle_deg"] = rad_to_deg(rel["los_angle"])
    row[f"{prefix}_los_rate_radps"] = float(rel["los_rate_standard"])


def make_snapshot_row(
    env: PursueEscapeEnv,
    episode_index: int,
    seed: int,
    event: str,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    生成一行“雷达开始时刻”的记录。

    event 取值建议：
        radar_active_at_reset：reset 后已经在雷达探测范围内；
        radar_activated：仿真过程中首次进入雷达探测范围；
        no_radar_activation_before_end：直到终止/超时也没有进入雷达探测范围。
    """
    observation = env._get_observation()  # 只读当前观测，用于保存接管截面输入。
    row: Dict[str, Any] = {
        "episode": int(episode_index),
        "seed": int(seed),
        "event": str(event),
        "time_s": float(env.current_time),
        "step": int(env.current_step),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(info.get("termination_reason", "reset" if env.current_step == 0 else "unknown")),
        "intercepted": bool(info.get("intercepted", False)),
        "passed": bool(info.get("passed", False)),
        "min_distance_m": safe_float(info.get("min_distance", env.min_distance)),
        "radar_detection_distance_m": float(env.config.radar_detection_distance),
        "radar_detection_min_distance_m": float(env.radar_detection_min_distance),
        "red_intelligent_active": bool(env.red_intelligent_active),
        "red_intelligent_activation_time_s": safe_float(env.red_intelligent_activation_time),
        "scenario_profile": str(env.config.scenario_profile),
        "guidance_mode": str(env.config.guidance_mode),
        "interceptor_count": int(env.config.interceptor_count),
        "source_pn_max_overload_g": float(env.config.source_pn_max_overload),
        "source_pn_N": float(env.config.N),
        "source_pn_compensation_gain": float(env.config.source_pn_compensation_gain),
        "interceptor_max_overload_g": float(env.config.interceptor_max_overload),
        "interceptor_target_compensation_gain": float(env.config.interceptor_target_compensation_gain),
        "interceptor_mach": float(env.config.interceptor_mach),
        "red_mach": float(env.config.red_mach),
        "initial_randomization_enabled": bool(env.config.initial_randomization_enabled),
        "interceptor_position_randomization_m": float(env.config.interceptor_position_randomization_m),
        "randomize_interceptor_y": bool(env.config.randomize_interceptor_y),
        "interceptor_heading_randomization_deg": float(env.config.interceptor_heading_randomization_deg),
        "interceptor_theta_randomization_deg": float(env.config.interceptor_theta_randomization_deg),
    }

    # 保存当前 observation。后续可以直接分析“接管时 SAC 输入分布”。
    for i, value in enumerate(observation.tolist()):
        row[f"obs_{i}"] = float(value)

    # 红方状态。
    row.update(flatten_vehicle_state("red", env.red_state))

    # 红方到目标距离。
    row["target_x_m"] = float(env.target_position[0])
    row["target_y_m"] = float(env.target_position[1])
    row["target_z_m"] = float(env.target_position[2])
    row["red_to_target_distance_m"] = float(env._target_distance())

    # 每枚拦截弹状态与相对几何。
    for interceptor_index, interceptor_state in enumerate(env.interceptor_states, start=1):
        prefix = f"interceptor_{interceptor_index}"
        row.update(flatten_vehicle_state(prefix, interceptor_state))
        add_interceptor_relative_fields(row, env, interceptor_index, interceptor_state)

        # 尽量记录该弹当前累计最小脱靶量。不同制导模块可能导出字段不同，故安全读取。
        min_distance_key = f"interceptor_{interceptor_index}_min_distance"
        row[f"{prefix}_min_distance_m"] = safe_float(info.get(min_distance_key))

    return row


# =============================================================================
# 5. 单回合仿真
# =============================================================================

def run_one_episode(
    env: PursueEscapeEnv,
    episode_index: int,
    base_seed: int,
    max_steps_per_episode: Optional[int],
) -> Dict[str, Any]:
    """
    红方平飞，蓝方按配置制导，直到首次雷达激活或 episode 结束。
    """
    episode_seed = int(base_seed + episode_index * 10007)
    observation, reset_info = env.reset(seed=episode_seed)

    # 情况 1：如果 reset 后已经处于雷达探测范围内，直接记录 reset 截面。
    if bool(env.red_intelligent_active):
        return make_snapshot_row(
            env=env,
            episode_index=episode_index,
            seed=episode_seed,
            event="radar_active_at_reset",
            terminated=False,
            truncated=False,
            info=reset_info,
        )

    # 红方平飞：无论雷达是否激活，请求动作都固定为 0。
    # 在 radar_range_gate 模式下，30 km 前环境也会强制红方动作置 0。
    action = np.zeros(1, dtype=np.float32)

    if max_steps_per_episode is None:
        max_steps = int(getattr(env, "max_steps", 0))
    else:
        max_steps = int(max_steps_per_episode)

    if max_steps <= 0:
        raise ValueError("max_steps 无效，请检查环境配置中的 t 和 dt。")

    last_info: Dict[str, Any] = dict(reset_info)
    terminated = False
    truncated = False

    for _ in range(max_steps):
        was_active_before_step = bool(env.red_intelligent_active)

        observation, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)

        # 情况 2：本步后首次进入雷达探测范围。
        if (not was_active_before_step) and bool(env.red_intelligent_active):
            return make_snapshot_row(
                env=env,
                episode_index=episode_index,
                seed=episode_seed,
                event="radar_activated",
                terminated=terminated,
                truncated=truncated,
                info=last_info,
            )

        # 情况 3：在雷达激活前就已经终止或超时。
        if bool(terminated) or bool(truncated):
            break

    return make_snapshot_row(
        env=env,
        episode_index=episode_index,
        seed=episode_seed,
        event="no_radar_activation_before_end",
        terminated=terminated,
        truncated=truncated,
        info=last_info,
    )


# =============================================================================
# 6. 结果保存与统计
# =============================================================================

def save_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    """
    保存 snapshots.csv。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key, "")) for key in fieldnames})


def save_json(data: Dict[str, Any], json_path: Path) -> None:
    """
    保存 summary.json。
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, ensure_ascii=False, indent=2)


def summarize_rows(rows: List[Dict[str, Any]], env_config: PursueEscapeEnvConfig) -> Dict[str, Any]:
    """
    汇总雷达激活截面统计量。
    """
    n = len(rows)
    event_counter = Counter(str(row.get("event", "unknown")) for row in rows)

    activated_rows = [
        row for row in rows
        if str(row.get("event")) in {"radar_activated", "radar_active_at_reset"}
    ]

    activation_times = [safe_float(row.get("time_s")) for row in activated_rows]
    activation_distances = [safe_float(row.get("radar_detection_min_distance_m")) for row in activated_rows]
    min_distances = [safe_float(row.get("min_distance_m")) for row in rows]

    # 第一枚弹和第二枚弹的距离/闭合速度也单独统计，便于判断进入 30 km 时是否两枚弹状态接近。
    interceptor_distance_stats: Dict[str, Any] = {}
    for i in range(1, int(env_config.interceptor_count) + 1):
        d_values = [safe_float(row.get(f"interceptor_{i}_distance_m")) for row in activated_rows]
        closing_values = [safe_float(row.get(f"interceptor_{i}_closing_speed_mps")) for row in activated_rows]
        interceptor_distance_stats[f"interceptor_{i}"] = {
            "distance_mean_m": float(np.nanmean(d_values)) if d_values else math.nan,
            "distance_p50_m": percentile(d_values, 50),
            "closing_speed_mean_mps": float(np.nanmean(closing_values)) if closing_values else math.nan,
            "closing_speed_p50_mps": percentile(closing_values, 50),
        }

    summary: Dict[str, Any] = {
        "episodes": int(n),
        "event_counts": dict(event_counter),
        "radar_activation_count": int(len(activated_rows)),
        "radar_activation_rate": float(len(activated_rows) / max(n, 1)),
        "activation_time_mean_s": float(np.nanmean(activation_times)) if activation_times else math.nan,
        "activation_time_p05_s": percentile(activation_times, 5),
        "activation_time_p50_s": percentile(activation_times, 50),
        "activation_time_p95_s": percentile(activation_times, 95),
        "activation_distance_mean_m": float(np.nanmean(activation_distances)) if activation_distances else math.nan,
        "activation_distance_p50_m": percentile(activation_distances, 50),
        "min_distance_mean_m": float(np.nanmean(min_distances)) if min_distances else math.nan,
        "min_distance_p50_m": percentile(min_distances, 50),
        "interceptor_stats_at_activation": interceptor_distance_stats,
        "env_snapshot": {
            "scenario_profile": str(env_config.scenario_profile),
            "guidance_mode": str(env_config.guidance_mode),
            "interceptor_count": int(env_config.interceptor_count),
            "radar_detection_distance": float(env_config.radar_detection_distance),
            "red_intelligent_activation_mode": str(env_config.red_intelligent_activation_mode),
            "dt": float(env_config.dt),
            "t": float(env_config.t),
            "red_mach": float(env_config.red_mach),
            "interceptor_mach": float(env_config.interceptor_mach),
            "interceptor_ability_profile": str(env_config.interceptor_ability_profile),
            "source_pn_max_overload": float(env_config.source_pn_max_overload),
            "N": float(env_config.N),
            "source_pn_compensation_gain": float(env_config.source_pn_compensation_gain),
            "interceptor_max_overload": float(env_config.interceptor_max_overload),
            "interceptor_target_compensation_gain": float(env_config.interceptor_target_compensation_gain),
            "initial_randomization_enabled": bool(env_config.initial_randomization_enabled),
            "interceptor_position_randomization_m": float(env_config.interceptor_position_randomization_m),
            "randomize_interceptor_y": bool(env_config.randomize_interceptor_y),
            "interceptor_heading_randomization_deg": float(env_config.interceptor_heading_randomization_deg),
            "interceptor_theta_randomization_deg": float(env_config.interceptor_theta_randomization_deg),
            "paper_interceptor_x_distance": float(env_config.paper_interceptor_x_distance),
            "paper_interceptor_lateral_offset": float(env_config.paper_interceptor_lateral_offset),
            "paper_interceptor_y_offset_from_red": float(env_config.paper_interceptor_y_offset_from_red),
        },
    }

    return summary


# =============================================================================
# 7. 命令行参数
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="记录雷达探测开始时刻的红蓝双方状态。红方全程平飞，拦截弹参数可命令行调节。"
    )

    parser.add_argument(
        "--env-config",
        type=str,
        nargs="+",
        default=["configs/env/base_200km.yaml"],
        help=(
            "环境 YAML 路径，可以传入一个或多个。"
            "例如：--env-config configs/env/base_200km.yaml configs/env/override_record_radar_activation.yaml"
        ),
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Monte Carlo 回合数。")
    parser.add_argument("--seed", type=int, default=0, help="基础随机种子。")
    parser.add_argument("--output-dir", type=str, default="outputs/radar_activation_snapshots", help="输出目录。")
    parser.add_argument("--run-name", type=str, default=None, help="本次实验名称。默认使用时间戳。")
    parser.add_argument("--print-every", type=int, default=50, help="进度打印间隔。设为 0 表示不打印中间进度。")
    parser.add_argument("--max-steps-per-episode", type=int, default=None, help="外部最大步数；默认使用 env.max_steps。")

    # 场景和制导模式。
    parser.add_argument("--scenario-profile", type=str, default='paper_200km_end_to_end', choices=["paper_200km_end_to_end", "paper_30km_radar_engagement", "manual_pair", "custom"], help="覆盖初始态势 profile。记录接管截面通常使用 paper_200km_end_to_end。")
    parser.add_argument("--guidance-mode", type=str, default='source_pn', choices=["source_pn", "mid_terminal_interceptor", "paper_mid_terminal"], help="覆盖蓝方制导模式。")
    parser.add_argument("--interceptor-count", type=int, default=2, choices=[1, 2], help="覆盖拦截弹数量。")
    parser.add_argument("--radar-detection-distance", type=float, default=None, help="覆盖雷达探测距离，单位 m。")

    # 时间和速度。
    parser.add_argument("--dt", type=float, default=None, help="覆盖仿真步长。")
    parser.add_argument("--t", type=float, default=None, help="覆盖最大仿真时间。")
    parser.add_argument("--red-mach", type=float, default=None, help="覆盖红方 Mach 数。")
    parser.add_argument("--interceptor-mach", type=float, default=None, help="覆盖拦截弹 Mach 数。")
    parser.add_argument("--kill-radius", type=float, default=None, help="覆盖杀伤半径。")

    # 拦截弹制导/能力调节。
    parser.add_argument("--interceptor-ability-profile", type=str, default='custom', choices=["custom", "weak", "paper", "strong"], help="覆盖蓝方能力档位。")
    parser.add_argument("--source-pn-max-overload", type=float, default=None, help="覆盖 source_pn 最大过载，单位 g。")
    parser.add_argument("--navigation-gain", type=float, default=None, help="覆盖 source_pn 导航系数 N。")
    parser.add_argument("--source-pn-compensation-gain", type=float, default=None, help="覆盖 source_pn 目标横向机动补偿系数。")
    parser.add_argument("--tau-i", type=float, default=None, help="覆盖拦截弹自动驾驶仪时间常数。")
    parser.add_argument("--interceptor-autopilot-rate-limit", type=float, default=None, help="覆盖拦截弹自动驾驶仪过载变化率限制，单位 g/s。")

    # 初始几何。
    parser.add_argument("--paper-interceptor-x-distance", type=float, default=None, help="覆盖 paper profile 拦截弹前向距离，单位 m。")
    parser.add_argument("--paper-interceptor-lateral-offset", type=float, default=None, help="覆盖 paper profile 双弹侧向夹击偏置，单位 m。")
    parser.add_argument("--paper-interceptor-y-offset-from-red", type=float, default=None, help="覆盖 paper profile 拦截弹相对红方高度偏置，单位 m。")
    parser.add_argument("--paper-interceptor-theta-deg", type=float, default=None, help="覆盖 paper profile 拦截弹基础弹道倾角，单位 deg。")

    # 初始随机化。
    parser.add_argument("--no-initial-randomization", action="store_true", help="关闭拦截弹初始位置/方向随机拉偏。")
    parser.add_argument("--position-randomization-m", type=float, default=None, help="覆盖拦截弹初始位置拉偏范围，单位 m。")
    parser.add_argument("--randomize-interceptor-y", action=argparse.BooleanOptionalAction, default=None, help="是否允许拦截弹高度方向也随机拉偏。")
    parser.add_argument("--heading-randomization-deg", type=float, default=None, help="覆盖拦截弹初始航向角拉偏范围，单位 deg。")
    parser.add_argument("--theta-randomization-deg", type=float, default=None, help="覆盖拦截弹初始弹道倾角拉偏范围，单位 deg。")

    return parser.parse_args()


# =============================================================================
# 8. 主入口
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes 必须为正整数。")

    env_config = build_env_config(args)
    env = PursueEscapeEnv(env_config)
    paths = make_output_paths(args.output_dir, args.run_name)

    print("=" * 100)
    print("记录雷达开始时刻红蓝状态")
    print("=" * 100)
    print("环境配置        :")
    for config_path in args.env_config:
        print(f"  - {resolve_project_path(config_path)}")
    print(f"输出目录        : {paths['run_dir']}")
    print(f"episodes        : {args.episodes}")
    print(f"seed            : {args.seed}")
    print("-" * 100)
    print(f"scenario_profile: {env_config.scenario_profile}")
    print(f"guidance_mode   : {env_config.guidance_mode}")
    print(f"interceptor_num : {env_config.interceptor_count}")
    print(f"radar_distance  : {env_config.radar_detection_distance:.3f} m")
    print(f"red action      : fixed level flight, action = 0")
    print("-" * 100)
    print(f"source_pn_max_overload      : {env_config.source_pn_max_overload}")
    print(f"N                            : {env_config.N}")
    print(f"source_pn_compensation_gain  : {env_config.source_pn_compensation_gain}")
    print(f"interceptor_max_overload     : {env_config.interceptor_max_overload}")
    print(f"interceptor_target_comp_gain : {env_config.interceptor_target_compensation_gain}")
    print("-" * 100)
    print(f"initial_randomization        : {env_config.initial_randomization_enabled}")
    print(f"position_randomization_m     : {env_config.interceptor_position_randomization_m}")
    print(f"randomize_interceptor_y      : {env_config.randomize_interceptor_y}")
    print(f"heading_randomization_deg    : {env_config.interceptor_heading_randomization_deg}")
    print(f"theta_randomization_deg      : {env_config.interceptor_theta_randomization_deg}")
    print("=" * 100)

    rows: List[Dict[str, Any]] = []

    for episode in range(1, int(args.episodes) + 1):
        row = run_one_episode(
            env=env,
            episode_index=episode,
            base_seed=int(args.seed),
            max_steps_per_episode=args.max_steps_per_episode,
        )
        rows.append(row)

        if args.print_every > 0 and (episode == 1 or episode % int(args.print_every) == 0):
            activated_count = sum(
                str(r.get("event")) in {"radar_activated", "radar_active_at_reset"}
                for r in rows
            )
            print(
                f"[{episode:05d}/{args.episodes:05d}] "
                f"activation_rate={activated_count / len(rows):.2%}, "
                f"last_event={row['event']}, "
                f"last_time={row['time_s']:.3f}s, "
                f"last_radar_min_distance={row['radar_detection_min_distance_m']:.3f}m"
            )

    summary = summarize_rows(rows, env_config)
    save_csv(rows, paths["csv_path"])
    save_json(summary, paths["summary_path"])

    print("=" * 100)
    print("记录完成")
    print("=" * 100)
    print(f"雷达激活数量       : {summary['radar_activation_count']}/{summary['episodes']}")
    print(f"雷达激活率         : {summary['radar_activation_rate']:.2%}")
    print(f"事件统计           : {summary['event_counts']}")
    print(f"激活时间 mean/p50  : {summary['activation_time_mean_s']:.3f}s / {summary['activation_time_p50_s']:.3f}s")
    print(f"激活距离 mean/p50  : {summary['activation_distance_mean_m']:.3f}m / {summary['activation_distance_p50_m']:.3f}m")
    print("-" * 100)
    print(f"CSV 明细           : {paths['csv_path']}")
    print(f"JSON 汇总          : {paths['summary_path']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
