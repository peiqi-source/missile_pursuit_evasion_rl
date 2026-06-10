"""
calibrate_interceptor_guidance.py

作用：
    扫描完整中末制导双拦截弹的能力参数，寻找稳定命中且能耗较低的工程基线。

运行方式：
    在项目根目录执行：
        $env:PYTHONPATH="$PWD\\src"
        python scripts/calibrate_interceptor_guidance.py

说明：
    默认只扫描候选网格前若干组，便于快速 smoke。
    需要完整扫描时使用：
        python scripts/calibrate_interceptor_guidance.py --full-grid
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# src_path：加入源码目录，保证脚本可直接运行。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.evaluation.metrics import collect_episode_metrics
from hypersonic_rl.utils import load_config_from_project


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
            可安全传入 dataclass 构造函数的配置字典。
    """
    # valid_field_names：目标 dataclass 支持的字段集合。
    valid_field_names = {field.name for field in fields(dataclass_type)}

    return {key: value for key, value in config_dict.items() if key in valid_field_names}


def build_action_value(case_name: str, env: PursueEscapeEnv, rng: np.random.Generator) -> float:
    """
    根据 case 名称生成当前步红方横向过载动作。

    参数：
        case_name：
            动作序列名称。
        env：
            当前环境对象。
        rng：
            固定 seed 的随机数生成器。

    返回：
        action_value：
            红方横向过载动作值。
    """
    # max_action：红方动作上限。
    max_action = float(env.config.nzc_h_max)

    if case_name == "zero_action":
        return 0.0

    if case_name == "positive_max_action":
        return max_action

    if case_name == "negative_max_action":
        return -max_action

    if case_name == "sine_action":
        # phase：根据当前仿真时间生成正弦机动相位。
        phase = 2.0 * np.pi * env.current_time / 6.0
        return float(max_action * np.sin(phase))

    if case_name == "random_action":
        return float(rng.uniform(-max_action, max_action))

    raise ValueError(f"未知动作 case：{case_name}")


def candidate_grid(full_grid: bool, limit_combos: int) -> List[Dict[str, float]]:
    """
    生成拦截弹能力参数候选网格。

    参数：
        full_grid：
            是否使用完整候选网格。
        limit_combos：
            非完整网格时保留的候选组合数量。

    返回：
        candidates：
            候选参数字典列表。
    """
    # candidate_values：候选范围来自当前工程修正计划。
    candidate_values = {
        "interceptor_max_overload": [8.0, 10.0, 12.0, 14.0, 16.0],
        "interceptor_midcourse_navigation_gain": [4.0, 6.0, 8.0],
        "interceptor_terminal_navigation_gain": [6.0, 8.0, 10.0],
        "interceptor_terminal_distance_threshold": [20000.0, 60000.0, 120000.0],
        "interceptor_target_compensation_gain": [0.0, 1.0, 2.0, 3.5, 4.0, 4.5],
    }

    # keys：保持字段顺序稳定，便于复现实验。
    keys = list(candidate_values.keys())

    # candidates：笛卡尔积候选参数。
    candidates = [
        dict(zip(keys, values))
        for values in itertools.product(*(candidate_values[key] for key in keys))
    ]

    if full_grid:
        return candidates

    return candidates[: max(1, int(limit_combos))]


def build_env_config(base_config: Dict[str, Any], candidate: Dict[str, float], profile: str) -> PursueEscapeEnvConfig:
    """
    根据基础配置和候选参数构造双弹标定环境。

    参数：
        base_config：
            YAML 中读取的基础环境配置。
        candidate：
            当前候选能力参数。
        profile：
            初始态势 profile。

    返回：
        config：
            PursueEscapeEnvConfig 配置对象。
    """
    # filtered_config：只保留环境支持字段。
    filtered_config = filter_dataclass_config(base_config, PursueEscapeEnvConfig)

    # overrides：标定固定为双弹、完整中末制导、无随机初始扰动。
    overrides: Dict[str, Any] = {
        "guidance_mode": "mid_terminal_interceptor",
        "interceptor_count": 2,
        "scenario_profile": profile,
        "observation_mode": "thesis_end_to_end_10d",
        "initial_randomization_enabled": False,
        "interceptor_ability_profile": "custom",
    }
    overrides.update(candidate)
    filtered_config.update(overrides)

    return PursueEscapeEnvConfig(**filtered_config)


def run_case(config: PursueEscapeEnvConfig, case_name: str, seed: int) -> Dict[str, Any]:
    """
    运行单个候选参数、动作 case 和 seed。

    参数：
        config：
            环境配置。
        case_name：
            红方动作 case 名称。
        seed：
            随机种子。

    返回：
        record：
            单个回合指标。
    """
    # env：当前标定环境。
    env = PursueEscapeEnv(config)

    # rng：动作随机数生成器。
    rng = np.random.default_rng(seed)

    env.reset(seed=seed)

    # done：统一终止标志。
    done = False

    while not done:
        # action_value：当前红方 1 维动作。
        action_value = build_action_value(case_name=case_name, env=env, rng=rng)

        _, _, terminated, truncated, _ = env.step(np.array([action_value], dtype=np.float32))

        done = terminated or truncated

    # metrics：单回合评估指标。
    metrics = collect_episode_metrics(env, episode_index=seed)

    # final_info：最后一步诊断信息。
    final_info = env.info_trace[-1] if env.info_trace else {}

    return {
        "case": case_name,
        "seed": seed,
        "min_distance": float(metrics["min_distance"]),
        "intercepted": float(metrics["intercepted"]),
        "success": float(metrics["success"]),
        "interceptor_control_energy": float(metrics["interceptor_control_energy"]),
        "termination_reason": final_info.get("termination_reason", "unknown"),
        "threat_interceptor_id": int(final_info.get("threat_interceptor_id", 0)),
    }


def summarize_candidate(candidate_index: int, candidate: Dict[str, float], case_records: List[Dict[str, Any]], kill_radius: float) -> Dict[str, Any]:
    """
    汇总单组候选参数在所有基准 case 上的表现。

    参数：
        candidate_index：
            候选序号。
        candidate：
            当前候选参数。
        case_records：
            所有动作 case 的回合记录。
        kill_radius：
            杀伤半径。

    返回：
        summary：
            候选参数汇总指标。
    """
    # miss_distances：所有 case 的连续最小脱靶量。
    miss_distances = [float(record["min_distance"]) for record in case_records]

    # energies：所有 case 的拦截弹能耗。
    energies = [float(record["interceptor_control_energy"]) for record in case_records]

    # failed_count：未进入杀伤半径的 case 数量。
    failed_count = int(sum(distance > kill_radius for distance in miss_distances))

    # summary：用于排序和 CSV 输出的候选汇总。
    summary: Dict[str, Any] = {
        "candidate_index": candidate_index,
        "failed_count": failed_count,
        "all_hit": int(failed_count == 0),
        "max_min_distance": float(np.max(miss_distances)),
        "mean_min_distance": float(np.mean(miss_distances)),
        "mean_interceptor_control_energy": float(np.mean(energies)),
    }
    summary.update(candidate)

    return summary


def save_csv(records: Iterable[Dict[str, Any]], csv_path: Path) -> None:
    """
    保存 CSV 文件。

    参数：
        records：
            字典记录序列。
        csv_path：
            输出 CSV 路径。

    返回：
        None。
    """
    # rows：物化记录，便于统一字段名。
    rows = list(records)

    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # fieldnames：合并所有字段，保证不同记录可同时写入。
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回：
        args：
            命令行参数命名空间。
    """
    # parser：标定脚本命令行参数。
    parser = argparse.ArgumentParser(description="双拦截弹中末制导能力参数标定。")
    parser.add_argument(
        "--profile",
        default="debug_50km_head_on",
        choices=["debug_50km_head_on", "paper_200km_end_to_end"],
        help="标定使用的初始态势 profile。",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="标定 seed 列表。",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["zero_action", "positive_max_action", "negative_max_action", "sine_action", "random_action"],
        help="红方动作 case 列表。",
    )
    parser.add_argument(
        "--limit-combos",
        type=int,
        default=30,
        help="非 full-grid 模式下扫描的候选组合数量。",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="扫描完整候选网格。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "outputs" / "calibrate_interceptor_guidance"),
        help="CSV 输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    """
    双拦截弹能力参数标定主函数。

    返回：
        None。
    """
    # args：命令行参数。
    args = parse_args()

    # base_config：环境基础 YAML 配置。
    base_config = load_config_from_project("configs/env/pursue_escape_env.yaml")

    # candidates：待扫描候选参数。
    candidates = candidate_grid(full_grid=bool(args.full_grid), limit_combos=int(args.limit_combos))

    # output_dir：输出目录。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # summaries/details：候选汇总和逐 case 明细。
    summaries: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates):
        # config：当前候选对应环境配置。
        config = build_env_config(base_config=base_config, candidate=candidate, profile=args.profile)

        # case_records：当前候选所有 seed/case 的结果。
        case_records: List[Dict[str, Any]] = []

        for seed in args.seeds:
            for case_name in args.cases:
                # record：当前候选、case、seed 的结果。
                record = run_case(config=config, case_name=case_name, seed=seed)
                record.update({"candidate_index": candidate_index, **candidate})
                case_records.append(record)
                details.append(record)

        # summary：当前候选汇总指标。
        summary = summarize_candidate(
            candidate_index=candidate_index,
            candidate=candidate,
            case_records=case_records,
            kill_radius=float(config.kill_radius),
        )
        summaries.append(summary)

        print(
            f"candidate={candidate_index:03d} failed={summary['failed_count']:2d} "
            f"max_miss={summary['max_min_distance']:.3f} "
            f"energy={summary['mean_interceptor_control_energy']:.3f}"
        )

    # best_summary：排序规则先看失败数，再看最大脱靶量和平均能耗。
    best_summary = sorted(
        summaries,
        key=lambda row: (
            int(row["failed_count"]),
            float(row["max_min_distance"]),
            float(row["mean_interceptor_control_energy"]),
        ),
    )[0]

    save_csv(summaries, output_dir / "candidate_summary.csv")
    save_csv(details, output_dir / "case_details.csv")
    save_csv([best_summary], output_dir / "best_candidate.csv")

    print("=" * 80)
    print("最佳候选：")
    for key, value in best_summary.items():
        print(f"{key}: {value}")
    print(f"输出目录：{output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
