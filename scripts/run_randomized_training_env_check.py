import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from hypersonic_rl.envs.pursue_escape_env import PursueEscapeEnv, PursueEscapeEnvConfig


POLICY_NAMES = [
    "level_flight",
    "sine",
    "square_wave",
    "max_left",
    "max_right",
    "random_uniform",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run randomized-initial-condition environment checks for SAC training setup.",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--guidance-mode", type=str, default="paper_mid_terminal")
    parser.add_argument("--interceptor-count", type=int, choices=[2], default=2)
    parser.add_argument("--scenario-profile", type=str, default="paper_200km_end_to_end")
    parser.add_argument("--ability-profile", type=str, default="paper")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--max-time", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--position-randomization-m", type=float, default=3000.0)
    parser.add_argument("--heading-randomization-deg", type=float, default=3.0)
    parser.add_argument("--theta-randomization-deg", type=float, default=3.0)
    parser.add_argument("--randomize-interceptor-y", action="store_true", default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "randomized_training_env_check",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 2) - 1)),
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def red_command(policy_name: str, t: float, max_overload: float, rng: np.random.Generator) -> float:
    if policy_name == "level_flight":
        return 0.0
    if policy_name == "sine":
        return float(0.75 * max_overload * math.sin(2.0 * math.pi * 0.08 * t))
    if policy_name == "square_wave":
        return float(max_overload if int(t // 4.0) % 2 == 0 else -max_overload)
    if policy_name == "max_left":
        return float(max_overload)
    if policy_name == "max_right":
        return -float(max_overload)
    if policy_name == "random_uniform":
        return float(rng.uniform(-max_overload, max_overload))
    raise ValueError(f"Unknown policy: {policy_name}")


def build_config(args_dict: Dict[str, Any]) -> PursueEscapeEnvConfig:
    return PursueEscapeEnvConfig(
        guidance_mode=str(args_dict["guidance_mode"]),
        interceptor_count=int(args_dict["interceptor_count"]),
        scenario_profile=str(args_dict["scenario_profile"]),
        interceptor_ability_profile=str(args_dict["ability_profile"]),
        initial_randomization_enabled=True,
        interceptor_position_randomization_m=float(args_dict["position_randomization_m"]),
        randomize_interceptor_y=bool(args_dict["randomize_interceptor_y"]),
        interceptor_heading_randomization_deg=float(args_dict["heading_randomization_deg"]),
        interceptor_theta_randomization_deg=float(args_dict["theta_randomization_deg"]),
        t=float(args_dict["max_time"]),
        dt=float(args_dict["dt"]),
    )


def initial_state_fields(prefix: str, state: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_initial_x": float(state[0]),
        f"{prefix}_initial_y": float(state[1]),
        f"{prefix}_initial_z": float(state[2]),
        f"{prefix}_initial_speed": float(state[3]),
        f"{prefix}_initial_theta_rad": float(state[4]),
        f"{prefix}_initial_theta_deg": float(np.rad2deg(state[4])),
        f"{prefix}_initial_psi_rad": float(state[5]),
        f"{prefix}_initial_psi_deg": float(np.rad2deg(state[5])),
    }


def rollout(task: Tuple[str, int, Dict[str, Any]]) -> Dict[str, Any]:
    policy_name, episode_index, args_dict = task
    env_seed = int(args_dict["seed"]) + int(episode_index)
    action_seed = int(args_dict["seed"]) * 1000003 + int(episode_index)

    env = PursueEscapeEnv(config=build_config(args_dict))
    env.reset(seed=env_seed)

    initial_red_state = env.red_state.copy()
    initial_interceptor_states = [state.copy() for state in env.interceptor_states]

    rng = np.random.default_rng(action_seed)
    terminated = False
    truncated = False
    last_info: Dict[str, Any] = {}
    total_reward = 0.0

    while not (terminated or truncated):
        command = red_command(
            policy_name=policy_name,
            t=float(env.current_time),
            max_overload=float(env.config.nzc_h_max),
            rng=rng,
        )
        _, reward, terminated, truncated, last_info = env.step(
            np.asarray([command], dtype=np.float32)
        )
        total_reward += float(reward)

    sampled_min_distance = (
        float(np.nanmin(np.asarray(env.distance_trace, dtype=np.float64)))
        if env.distance_trace
        else float("nan")
    )
    final_distance = (
        float(env.distance_trace[-1])
        if env.distance_trace
        else float(last_info.get("distance", float("nan")))
    )

    record: Dict[str, Any] = {
        "policy": policy_name,
        "episode_index": int(episode_index),
        "env_seed": int(env_seed),
        "action_seed": int(action_seed),
        "guidance_mode": str(args_dict["guidance_mode"]),
        "interceptor_count": int(args_dict["interceptor_count"]),
        "scenario_profile": str(args_dict["scenario_profile"]),
        "ability_profile": str(args_dict["ability_profile"]),
        "position_randomization_m": float(args_dict["position_randomization_m"]),
        "heading_randomization_deg": float(args_dict["heading_randomization_deg"]),
        "theta_randomization_deg": float(args_dict["theta_randomization_deg"]),
        "randomize_interceptor_y": bool(args_dict["randomize_interceptor_y"]),
        "success": bool(last_info.get("success", False)),
        "intercepted": bool(last_info.get("intercepted", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(last_info.get("termination_reason", "unknown")),
        "min_distance": float(last_info.get("min_distance", env.min_distance)),
        "continuous_min_distance": float(last_info.get("min_distance", env.min_distance)),
        "sampled_min_distance": sampled_min_distance,
        "final_distance": final_distance,
        "final_time": float(env.current_time),
        "episode_steps": int(env.current_step),
        "total_reward": float(total_reward),
        "threat_interceptor_id": int(last_info.get("threat_interceptor_id", 0)),
        "red_initial_x": float(initial_red_state[0]),
        "red_initial_y": float(initial_red_state[1]),
        "red_initial_z": float(initial_red_state[2]),
        "red_initial_speed": float(initial_red_state[3]),
        "red_initial_theta_rad": float(initial_red_state[4]),
        "red_initial_theta_deg": float(np.rad2deg(initial_red_state[4])),
        "red_initial_psi_rad": float(initial_red_state[5]),
        "red_initial_psi_deg": float(np.rad2deg(initial_red_state[5])),
    }

    for index, state in enumerate(initial_interceptor_states, start=1):
        record.update(initial_state_fields(f"interceptor_{index}", state))

    for index in range(1, int(args_dict["interceptor_count"]) + 1):
        prefix = f"interceptor_{index}"
        for suffix in [
            "min_distance",
            "current_distance",
            "closest_distance_this_step",
            "intercepted",
            "passed",
            "phase",
            "pass_time",
            "intercept_time",
            "phase_switch_time",
        ]:
            key = f"{prefix}_{suffix}"
            if key in last_info:
                record[key] = last_info[key]

    return record


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def finite_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        if key not in record:
            continue
        value = float(record[key])
        if np.isfinite(value):
            values.append(value)
    return values


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    policy = str(records[0]["policy"])
    min_distances = finite_values(records, "min_distance")
    sampled_min_distances = finite_values(records, "sampled_min_distance")
    final_distances = finite_values(records, "final_distance")
    final_times = finite_values(records, "final_time")

    termination_counts: Dict[str, int] = {}
    for record in records:
        reason = str(record.get("termination_reason", "unknown"))
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    summary: Dict[str, Any] = {
        "policy": policy,
        "episodes": len(records),
        "success_rate": float(np.mean([bool(r["success"]) for r in records])),
        "intercept_rate": float(np.mean([bool(r["intercepted"]) for r in records])),
        "mean_min_distance": float(np.mean(min_distances)),
        "std_min_distance": float(np.std(min_distances, ddof=0)),
        "min_min_distance": float(np.min(min_distances)),
        "max_min_distance": float(np.max(min_distances)),
        "p05_min_distance": float(np.quantile(min_distances, 0.05)),
        "p50_min_distance": float(np.quantile(min_distances, 0.50)),
        "p95_min_distance": float(np.quantile(min_distances, 0.95)),
        "mean_sampled_min_distance": float(np.mean(sampled_min_distances)),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_final_time": float(np.mean(final_times)),
        "termination_counts": ";".join(
            f"{key}:{value}" for key, value in sorted(termination_counts.items())
        ),
    }

    for index in [1, 2]:
        key = f"interceptor_{index}_min_distance"
        values = finite_values(records, key)
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"p50_{key}"] = float(np.quantile(values, 0.50))

    return summary


def write_summary_csv(summaries: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "policy",
        "episodes",
        "success_rate",
        "intercept_rate",
        "mean_min_distance",
        "std_min_distance",
        "min_min_distance",
        "max_min_distance",
        "p05_min_distance",
        "p50_min_distance",
        "p95_min_distance",
        "mean_sampled_min_distance",
        "mean_final_distance",
        "mean_final_time",
        "termination_counts",
    ]
    extra = sorted({key for summary in summaries for key in summary if key not in preferred})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=preferred + extra)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = {
        "episodes": int(args.episodes),
        "guidance_mode": args.guidance_mode,
        "interceptor_count": int(args.interceptor_count),
        "scenario_profile": args.scenario_profile,
        "ability_profile": args.ability_profile,
        "seed": int(args.seed),
        "max_time": float(args.max_time),
        "dt": float(args.dt),
        "position_randomization_m": float(args.position_randomization_m),
        "heading_randomization_deg": float(args.heading_randomization_deg),
        "theta_randomization_deg": float(args.theta_randomization_deg),
        "randomize_interceptor_y": bool(args.randomize_interceptor_y),
    }

    print("=" * 80)
    print(
        "Randomized training environment check: "
        f"episodes_per_policy={args.episodes}, guidance={args.guidance_mode}, "
        f"workers={args.workers}, output={output_dir}"
    )
    print(
        "Blue randomization: "
        f"position=+-{args.position_randomization_m} m, "
        f"heading=+-{args.heading_randomization_deg} deg, "
        f"theta=+-{args.theta_randomization_deg} deg, "
        f"randomize_y={args.randomize_interceptor_y}"
    )

    all_records: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for policy_name in POLICY_NAMES:
        print("=" * 80)
        print(f"Running policy={policy_name}")
        tasks = [(policy_name, episode_index, args_dict) for episode_index in range(int(args.episodes))]
        if int(args.workers) <= 1:
            records = [rollout(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                records = list(executor.map(rollout, tasks, chunksize=8))

        records.sort(key=lambda item: int(item["episode_index"]))
        policy_summary = summarize(records)
        summaries.append(policy_summary)
        all_records.extend(records)

        print(
            f"policy={policy_name}, success_rate={policy_summary['success_rate']:.3f}, "
            f"intercept_rate={policy_summary['intercept_rate']:.3f}, "
            f"mean_min_distance={policy_summary['mean_min_distance']:.3f} m, "
            f"p50={policy_summary['p50_min_distance']:.3f} m, "
            f"termination={policy_summary['termination_counts']}"
        )

    write_csv(all_records, output_dir / "episode_records.csv")
    write_summary_csv(summaries, output_dir / "summary.csv")

    print("=" * 80)
    print(f"Saved records: {output_dir / 'episode_records.csv'}")
    print(f"Saved summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
