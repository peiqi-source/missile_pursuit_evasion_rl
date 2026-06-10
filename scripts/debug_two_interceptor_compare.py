"""
debug_two_interceptor_compare.py

作用：
    固定初始条件、随机种子和红方动作序列，公平比较 source_pn 与完整中末制导双拦截弹。

运行方式：
    在项目根目录执行：
        $env:PYTHONPATH="$PWD\\src"
        python scripts/debug_two_interceptor_compare.py

常用参数：
    python scripts/debug_two_interceptor_compare.py --profile paper_200km_end_to_end --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# src_path：加入源码目录，保证直接运行脚本时可以导入项目包。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.evaluation.metrics import collect_episode_metrics
from hypersonic_rl.utils import load_config_from_project
from hypersonic_rl.visualization import plot_episode_summary, plot_full_trajectory_summary


ActionPolicy = Callable[[PursueEscapeEnv, int, np.random.Generator], float]


def filter_dataclass_config(config_dict: Dict[str, Any], dataclass_type: Any) -> Dict[str, Any]:
    """
    从配置字典中过滤出 dataclass 支持的字段。

    参数：
        config_dict：
            原始配置字典。
        dataclass_type：
            目标 dataclass 类型。

    返回：
        filtered_config：
            可以安全传入 dataclass 构造函数的配置字典。
    """
    # valid_field_names：目标 dataclass 支持的字段名集合。
    valid_field_names = {field.name for field in fields(dataclass_type)}

    # filtered_config：剔除 YAML 中与当前环境无关的字段。
    filtered_config = {key: value for key, value in config_dict.items() if key in valid_field_names}

    return filtered_config


def build_debug_config(profile: str, guidance_mode: str, max_time: float | None) -> PursueEscapeEnvConfig:
    """
    构造双拦截弹公平对比用环境配置。

    参数：
        profile：
            初始态势 profile，例如 debug_50km_head_on 或 paper_200km_end_to_end。
        guidance_mode：
            蓝方制导模式。
        max_time：
            可选单回合最大仿真时间；为 None 时沿用 YAML 默认值。

    返回：
        config：
            双拦截弹环境配置对象。
    """
    # env_config_dict：读取默认环境配置，保证能力参数与训练入口一致。
    env_config_dict = load_config_from_project("configs/env/pursue_escape_env.yaml")

    # filtered_config：只保留 PursueEscapeEnvConfig 当前支持的字段。
    filtered_config = filter_dataclass_config(env_config_dict, PursueEscapeEnvConfig)

    # overrides：公平对比强制固定双弹、固定初始条件、固定制导模式。
    overrides = {
        "guidance_mode": guidance_mode,
        "interceptor_count": 2,
        "scenario_profile": profile,
        "observation_mode": "thesis_end_to_end_10d",
        "initial_randomization_enabled": False,
    }
    if max_time is not None:
        overrides["t"] = float(max_time)

    filtered_config.update(overrides)

    return PursueEscapeEnvConfig(**filtered_config)


def build_action_cases() -> Dict[str, ActionPolicy]:
    """
    构造红方动作序列集合。

    返回：
        action_cases：
            case 名称到动作生成函数的映射。
    """

    def zero_action(env: PursueEscapeEnv, step_index: int, rng: np.random.Generator) -> float:
        """
        红方不机动。
        """
        return 0.0

    def positive_max_action(env: PursueEscapeEnv, step_index: int, rng: np.random.Generator) -> float:
        """
        红方持续正最大横向过载。
        """
        return float(env.config.nzc_h_max)

    def negative_max_action(env: PursueEscapeEnv, step_index: int, rng: np.random.Generator) -> float:
        """
        红方持续负最大横向过载。
        """
        return -float(env.config.nzc_h_max)

    def sine_action(env: PursueEscapeEnv, step_index: int, rng: np.random.Generator) -> float:
        """
        红方执行平滑正弦横向机动。
        """
        # period：正弦动作周期，50 km debug 下 6 s 足够覆盖明显转向。
        period = 6.0

        # phase：根据当前步数生成连续相位。
        phase = 2.0 * np.pi * step_index * env.config.dt / period

        return float(env.config.nzc_h_max * np.sin(phase))

    def random_action(env: PursueEscapeEnv, step_index: int, rng: np.random.Generator) -> float:
        """
        红方使用固定 seed 的随机动作，模拟未训练 SAC 的探索输入。
        """
        return float(rng.uniform(-env.config.nzc_h_max, env.config.nzc_h_max))

    return {
        "zero_action": zero_action,
        "positive_max_action": positive_max_action,
        "negative_max_action": negative_max_action,
        "sine_action": sine_action,
        "random_action": random_action,
    }


def compute_saturation_ratios(env: PursueEscapeEnv) -> Dict[str, float]:
    """
    统计每枚拦截弹指令和实际过载的饱和比例。

    参数：
        env：
            已完成一个回合的环境对象。

    返回：
        ratios：
            包含全局和 per-interceptor 饱和比例的字典。
    """
    # max_overload：根据制导模式选择当前拦截弹能力上限。
    if env.config.guidance_mode == "source_pn":
        max_overload = float(env.config.source_pn_max_overload)
    else:
        max_overload = float(env.config.interceptor_max_overload)

    # ratios：最终输出的饱和统计字段。
    ratios: Dict[str, float] = {}

    # command_flags/actual_flags：全局饱和布尔序列，用于求所有弹平均比例。
    command_flags: List[bool] = []
    actual_flags: List[bool] = []

    for interceptor_index in range(1, env.config.interceptor_count + 1):
        # prefix：当前拦截弹字段前缀。
        prefix = f"interceptor_{interceptor_index}"

        # per_command_flags/per_actual_flags：单枚弹每一步是否接近过载上限。
        per_command_flags: List[bool] = []
        per_actual_flags: List[bool] = []

        for info in env.info_trace:
            # theta：当前航迹倾角，用于扣除纵向平衡过载。
            theta = float(info.get(f"{prefix}_theta", 0.0))

            # equilibrium_ny：保持当前航迹倾角的纵向平衡项。
            equilibrium_ny = float(np.cos(theta))

            # command_norm：指令过载相对平衡项的二维合成幅值。
            command_norm = float(
                np.hypot(
                    float(info.get(f"{prefix}_ny_command", equilibrium_ny)) - equilibrium_ny,
                    float(info.get(f"{prefix}_nz_command", 0.0)),
                )
            )

            # actual_norm：实际过载相对平衡项的二维合成幅值。
            actual_norm = float(
                np.hypot(
                    float(info.get(f"{prefix}_ny_actual", equilibrium_ny)) - equilibrium_ny,
                    float(info.get(f"{prefix}_nz_actual", 0.0)),
                )
            )

            per_command_flags.append(command_norm >= max_overload - 1e-6)
            per_actual_flags.append(actual_norm >= max_overload - 1e-6)

        # 单枚弹饱和比例。
        ratios[f"{prefix}_command_saturation_ratio"] = float(np.mean(per_command_flags)) if per_command_flags else 0.0
        ratios[f"{prefix}_actual_saturation_ratio"] = float(np.mean(per_actual_flags)) if per_actual_flags else 0.0

        command_flags.extend(per_command_flags)
        actual_flags.extend(per_actual_flags)

    # 全局饱和比例：两枚弹所有步的平均。
    ratios["interceptor_command_saturation_ratio"] = float(np.mean(command_flags)) if command_flags else 0.0
    ratios["interceptor_actual_saturation_ratio"] = float(np.mean(actual_flags)) if actual_flags else 0.0

    return ratios


def run_episode(
    guidance_mode: str,
    case_name: str,
    action_policy: ActionPolicy,
    seed: int,
    profile: str,
    max_time: float | None,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    运行一个固定制导模式、动作 case 和 seed 的双弹回合。

    参数：
        guidance_mode：
            source_pn 或 mid_terminal_interceptor。
        case_name：
            当前红方动作序列名称。
        action_policy：
            动作生成函数。
        seed：
            环境随机种子。
        profile：
            初始态势 profile。
        max_time：
            可选最大仿真时间。
        output_dir：
            图像和 CSV 输出目录。

    返回：
        record：
            当前回合汇总指标。
    """
    # env：当前对比回合环境。
    env = PursueEscapeEnv(build_debug_config(profile=profile, guidance_mode=guidance_mode, max_time=max_time))

    # rng：动作序列随机数，保证同一 seed 下两个制导模式使用同一随机动作。
    rng = np.random.default_rng(seed)

    env.reset(seed=seed)

    # done：统一终止标志。
    done = False

    while not done:
        # action_value：红方 1 维横向过载动作。
        action_value = action_policy(env, env.current_step, rng)

        # action：Gym 接口要求 shape=(1,)。
        action = np.array([action_value], dtype=np.float32)

        _, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    # metrics：通用单回合评估指标。
    metrics = collect_episode_metrics(env, episode_index=seed)

    # final_info：最后一步诊断信息。
    final_info = env.info_trace[-1] if env.info_trace else {}

    # record：汇总当前回合核心对比字段。
    record: Dict[str, Any] = {
        "guidance_mode": guidance_mode,
        "case": case_name,
        "seed": seed,
        "profile": profile,
        "episode_steps": int(metrics["episode_steps"]),
        "episode_time": float(metrics["episode_time"]),
        "total_reward": float(metrics["total_reward"]),
        "min_distance": float(metrics["min_distance"]),
        "final_distance": float(metrics["final_distance"]),
        "closest_distance_min": float(metrics["closest_distance_min"]),
        "intercepted": float(metrics["intercepted"]),
        "success": float(metrics["success"]),
        "termination_reason": final_info.get("termination_reason", "unknown"),
        "threat_interceptor_id": int(final_info.get("threat_interceptor_id", 0)),
        "interceptor_control_energy": float(metrics["interceptor_control_energy"]),
        "interceptor_ny_control_energy": float(metrics["interceptor_ny_control_energy"]),
        "interceptor_nz_control_energy": float(metrics["interceptor_nz_control_energy"]),
    }

    # per-interceptor 字段：每枚弹的脱靶量、阶段、能耗和最终控制量。
    for interceptor_index in range(1, env.config.interceptor_count + 1):
        prefix = f"interceptor_{interceptor_index}"
        record.update(
            {
                f"{prefix}_min_distance": float(metrics.get(f"{prefix}_min_distance", np.nan)),
                f"{prefix}_final_distance": float(metrics.get(f"{prefix}_final_distance", np.nan)),
                f"{prefix}_intercepted": float(metrics.get(f"{prefix}_intercepted", 0.0)),
                f"{prefix}_passed": float(metrics.get(f"{prefix}_passed", 0.0)),
                f"{prefix}_phase_final": str(metrics.get(f"{prefix}_phase_final", "unknown")),
                f"{prefix}_ny_control_energy": float(metrics.get(f"{prefix}_ny_control_energy", 0.0)),
                f"{prefix}_nz_control_energy": float(metrics.get(f"{prefix}_nz_control_energy", 0.0)),
                f"{prefix}_ny_command_final": float(final_info.get(f"{prefix}_ny_command", np.nan)),
                f"{prefix}_nz_command_final": float(final_info.get(f"{prefix}_nz_command", np.nan)),
                f"{prefix}_ny_actual_final": float(final_info.get(f"{prefix}_ny_actual", np.nan)),
                f"{prefix}_nz_actual_final": float(final_info.get(f"{prefix}_nz_actual", np.nan)),
            }
        )

    record.update(compute_saturation_ratios(env))

    # figure_stem：当前回合图像文件名前缀。
    figure_stem = f"{guidance_mode}_{case_name}_seed{seed}"

    # 保存综合诊断图和轨迹图。
    plot_episode_summary(env, output_dir / f"{figure_stem}_episode.png")
    plot_full_trajectory_summary(env, output_dir / f"{figure_stem}_trajectory.png")

    return record


def save_records(records: Iterable[Dict[str, Any]], csv_path: Path) -> None:
    """
    将调试结果保存为 CSV。

    参数：
        records：
            回合汇总记录。
        csv_path：
            CSV 保存路径。

    返回：
        None。
    """
    # records_list：物化迭代器，便于提取字段名。
    records_list = list(records)

    if not records_list:
        return

    # fieldnames：合并所有记录字段，避免某些 case 字段缺失导致写入失败。
    fieldnames = sorted({key for record in records_list for key in record.keys()})

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        # writer：标准 CSV 写入器。
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records_list)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回：
        args：
            命令行参数命名空间。
    """
    # parser：双弹 debug 命令行参数。
    parser = argparse.ArgumentParser(description="双拦截弹公平制导对比调试。")
    parser.add_argument(
        "--profile",
        default="debug_50km_head_on",
        choices=["debug_50km_head_on", "paper_200km_end_to_end"],
        help="初始态势 profile。",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="需要运行的随机种子列表。",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="覆盖单回合最大仿真时间，默认沿用环境 YAML。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "outputs" / "debug_two_interceptor_compare"),
        help="CSV 和图像输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    """
    双拦截弹公平对比调试主函数。

    返回：
        None。
    """
    # args：命令行参数。
    args = parse_args()

    # output_dir：统一输出目录。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # action_cases：待覆盖的红方动作序列。
    action_cases = build_action_cases()

    # records：所有回合汇总结果。
    records: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for case_name, action_policy in action_cases.items():
            for guidance_mode in ["source_pn", "mid_terminal_interceptor"]:
                # record：单个制导模式、动作 case、seed 的汇总结果。
                record = run_episode(
                    guidance_mode=guidance_mode,
                    case_name=case_name,
                    action_policy=action_policy,
                    seed=seed,
                    profile=args.profile,
                    max_time=args.max_time,
                    output_dir=output_dir,
                )
                records.append(record)
                print(
                    f"{guidance_mode:24s} {case_name:20s} seed={seed:2d} "
                    f"min_distance={record['min_distance']:.3f} m "
                    f"termination={record['termination_reason']}"
                )

    # csv_path：调试汇总 CSV 路径。
    csv_path = output_dir / "summary.csv"
    save_records(records, csv_path)

    print("=" * 80)
    print(f"双弹公平对比完成：{csv_path}")
    print(f"图像输出目录：{output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
