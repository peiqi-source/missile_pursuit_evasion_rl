import argparse
import csv
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


# ---------------------------------------------------------------------
# 路径处理：保证直接运行 scripts/xxx.py 时可以找到 src/hypersonic_rl
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from hypersonic_rl.envs.pursue_escape_env import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.visualization.plot_episode import plot_episode_summary
from hypersonic_rl.visualization.plot_trajectory import (
    INTERCEPTOR_COLORS,
    RED_COLOR,
    extract_multi_trajectories_from_env,
    plot_full_trajectory_summary,
)


# ---------------------------------------------------------------------
# 红方固定机动策略
# ---------------------------------------------------------------------
ManeuverPolicy = Callable[[float, float, np.random.Generator], float]


def red_level_flight(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """平飞：红方不主动侧向机动。"""
    return 0.0


def red_sine_maneuver(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """正弦机动：红方侧向过载按正弦变化。"""
    amplitude = 0.75 * float(max_overload)
    frequency_hz = 0.08
    return float(amplitude * np.sin(2.0 * np.pi * frequency_hz * t))


def red_bangbang_maneuver(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """bang-bang 机动：红方每隔固定时间在正负最大过载之间切换。"""
    switch_period = 4.0
    segment_index = int(t // switch_period)
    return float(max_overload) if segment_index % 2 == 0 else -float(max_overload)


def red_left_full(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """左打满：红方持续给正向最大侧向过载。"""
    return float(max_overload)


def red_right_full(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """右打满：红方持续给负向最大侧向过载。"""
    return -float(max_overload)


def red_random_uniform(t: float, max_overload: float, rng: np.random.Generator) -> float:
    """随机机动：每一步在 [-max_overload, max_overload] 内均匀采样。"""
    return float(rng.uniform(-float(max_overload), float(max_overload)))


MANEUVER_POLICIES: Dict[str, ManeuverPolicy] = {
    "level_flight": red_level_flight,
    "sine": red_sine_maneuver,
    "bangbang": red_bangbang_maneuver,
    "left_full": red_left_full,
    "right_full": red_right_full,
    "random_uniform": red_random_uniform,
}


GUIDANCE_MODES: List[str] = [
    "source_pn",
    "mid_terminal_interceptor",
    "paper_mid_terminal",
]


# ---------------------------------------------------------------------
# 配置读取与覆盖
# ---------------------------------------------------------------------
def resolve_project_path(path: Path) -> Path:
    """将相对路径解析到项目根目录下。"""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_env_config_dict(env_config_path: Optional[Path]) -> Dict[str, Any]:
    """读取环境 YAML 配置；env_config_path=None 时返回空字典。"""
    if env_config_path is None:
        return {}

    resolved_path = resolve_project_path(env_config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"环境配置文件不存在：{resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        raise ValueError(f"环境配置文件必须是 YAML 字典：{resolved_path}")

    return dict(config_dict)


def build_env_config_from_yaml_and_overrides(
    env_config_path: Optional[Path],
    overrides: Dict[str, Any],
) -> PursueEscapeEnvConfig:
    """
    先读取 YAML，再用当前测试 case 的命令行参数覆盖。

    这样可以保证：
        1. 默认环境参数来自 configs/env/pursue_escape_env.yaml；
        2. guidance_mode / maneuver / interceptor_count 等测试变量仍可由命令行显式控制；
        3. YAML 拼错字段时能尽早报错。
    """
    config_dict = load_env_config_dict(env_config_path)

    valid_fields = {field.name for field in fields(PursueEscapeEnvConfig)}
    unknown_keys = sorted(set(config_dict.keys()) - valid_fields)
    if unknown_keys:
        raise ValueError(f"YAML 中存在 PursueEscapeEnvConfig 不支持的字段：{unknown_keys}")

    config_dict.update(overrides)
    return PursueEscapeEnvConfig(**config_dict)


# ---------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="可视化固定红方机动下的拦截弹制导轨迹，并检查终止原因与奖励逻辑。",
    )

    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/env/pursue_escape_env.yaml"),
        help="环境 YAML 配置路径；默认读取 configs/env/pursue_escape_env.yaml。",
    )
    parser.add_argument(
        "--no-env-config",
        action="store_true",
        help="不读取 YAML，只使用 PursueEscapeEnvConfig 默认值和命令行覆盖。",
    )

    parser.add_argument(
        "--interceptor-count",
        type=int,
        choices=[1, 2],
        default=2,
        help="拦截弹数量，支持 1 或 2；默认 2，更接近一对二突防任务。",
    )

    parser.add_argument(
        "--guidance-modes",
        nargs="+",
        choices=GUIDANCE_MODES,
        default=list(GUIDANCE_MODES),
        help="需要对比的蓝方制导模式。",
    )

    parser.add_argument(
        "--maneuvers",
        nargs="+",
        choices=list(MANEUVER_POLICIES.keys()),
        default=list(MANEUVER_POLICIES.keys()),
        help="需要运行的红方固定机动策略。",
    )

    parser.add_argument(
        "--scenario-profile",
        choices=["paper_200km_end_to_end", "manual_pair", "custom"],
        default="paper_200km_end_to_end",
        help="初始态势 profile。",
    )

    parser.add_argument(
        "--interceptor-ability-profile",
        choices=["weak", "paper", "strong", "custom"],
        default="paper",
        help="蓝方能力档位。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="环境随机种子。",
    )

    parser.add_argument(
        "--interceptor_initial_theta_deg",
        type=int,
        default=80,
        help="初始弹道倾角。",
    )

    parser.add_argument(
        "--enable-initial-randomization",
        action="store_true",
        help="启用环境初始随机化；默认关闭，保证固定机动测试可复现。",
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=80.0,
        help="单回合最大仿真时长，单位 s。",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="仿真步长，单位 s；默认 0.01，避免 5 m 杀伤半径下漏判。",
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="只保存 summary CSV，不生成轨迹图；适合快速批量检查奖励和终止原因。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；相对路径会解析到项目根目录下。",
    )

    return parser.parse_args()


def resolve_output_dir(output_dir: Path | None, interceptor_count: int) -> Path:
    """根据命令行参数生成最终输出目录。"""
    if output_dir is None:
        return PROJECT_ROOT / "outputs" / f"guidance_maneuver_visualization_{interceptor_count}missile"
    return output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir


# ---------------------------------------------------------------------
# 单次仿真
# ---------------------------------------------------------------------
def run_single_case(
    guidance_mode: str,
    maneuver_name: str,
    maneuver_policy: ManeuverPolicy,
    seed: int = 0,
    interceptor_count: int = 2,
    scenario_profile: str = "paper_200km_end_to_end",
    interceptor_ability_profile: str = "paper",
    interceptor_initial_theta_deg: int = 80,
    max_time: float = 80.0,
    dt: float = 0.01,
    env_config_path: Optional[Path] = Path("configs/env/pursue_escape_env.yaml"),
    initial_randomization_enabled: bool = False,
) -> Tuple[PursueEscapeEnv, Dict[str, object]]:
    """运行单个红方固定机动 + 蓝方制导模式组合。"""
    overrides: Dict[str, Any] = {
        "guidance_mode": guidance_mode,
        "initial_randomization_enabled": bool(initial_randomization_enabled),
        "interceptor_count": int(interceptor_count),
        "scenario_profile": scenario_profile,
        "interceptor_ability_profile": interceptor_ability_profile,
        "interceptor_initial_theta_deg": interceptor_initial_theta_deg,
        "t": float(max_time),
        "dt": float(dt),
    }

    config = build_env_config_from_yaml_and_overrides(
        env_config_path=env_config_path,
        overrides=overrides,
    )

    env = PursueEscapeEnv(config=config)
    _, info = env.reset(seed=seed)

    maneuver_rng = np.random.default_rng(seed + 1009)

    terminated = False
    truncated = False
    reward = 0.0
    last_info = dict(info)

    for _ in range(env.max_steps):
        current_time = float(env.current_time)
        red_command = maneuver_policy(
            current_time,
            float(env.config.nzc_h_max),
            maneuver_rng,
        )
        action = np.array([red_command], dtype=np.float32)

        _, reward, terminated, truncated, last_info = env.step(action)

        if terminated or truncated:
            break

    if len(env.distance_trace) > 0:
        sampled_min_distance = float(np.nanmin(np.asarray(env.distance_trace, dtype=np.float64)))
        final_distance = float(env.distance_trace[-1])
    else:
        sampled_min_distance = np.nan
        final_distance = np.nan

    continuous_min_distance = float(last_info.get("min_distance", env.min_distance))
    total_reward = float(np.sum(env.reward_trace)) if env.reward_trace else float(reward)

    summary: Dict[str, object] = {
        "maneuver_name": maneuver_name,
        "guidance_mode": guidance_mode,
        "interceptor_count": int(env.config.interceptor_count),
        "scenario_profile": str(env.config.scenario_profile),
        "ability_profile": str(env.config.interceptor_ability_profile),
        "dt": float(env.config.dt),
        "max_time": float(env.config.t),
        "min_distance": continuous_min_distance,
        "continuous_min_distance": continuous_min_distance,
        "sampled_min_distance": sampled_min_distance,
        "final_distance": final_distance,
        "target_distance": float(last_info.get("target_distance", np.nan)),
        "final_time": float(env.current_time),
        "step_reward": float(reward),
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(last_info.get("success", False)),
        "intercepted": bool(last_info.get("intercepted", False)),
        "passed": bool(last_info.get("passed", False)),
        "termination_reason": str(last_info.get("termination_reason", "unknown")),
        "threat_interceptor_id": int(last_info.get("threat_interceptor_id", 0)),
    }

    # 保留每枚弹自己的命中、错过、最小距离、阶段字段，以及 reward.py 新增的诊断字段。
    for key, value in last_info.items():
        if (
            key.endswith("intercepted")
            or key.endswith("passed")
            or key.endswith("min_distance")
            or key.endswith("phase")
            or key.startswith("reward")
            or key.endswith("reward")
        ):
            summary[key] = value

    return env, summary


# ---------------------------------------------------------------------
# 同一红方机动下，对比多种蓝方制导模式的 X-Z 轨迹
# ---------------------------------------------------------------------
def plot_guidance_comparison_xz(
    maneuver_name: str,
    env_by_guidance: Dict[str, PursueEscapeEnv],
    save_path: Path,
) -> Path:
    """将同一红方机动下的多种制导模式画在同一张 X-Z 俯视图中。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(10, 7))
    red_plotted = False

    for guidance_index, (guidance_mode, env) in enumerate(env_by_guidance.items()):
        red_trajectory, interceptor_trajectories = extract_multi_trajectories_from_env(env)

        if (not red_plotted) and red_trajectory.size > 0:
            ax.plot(
                red_trajectory[:, 0] / 1000.0,
                red_trajectory[:, 2],
                color=RED_COLOR,
                linewidth=2.3,
                label="Red trajectory",
            )
            ax.scatter(
                red_trajectory[0, 0] / 1000.0,
                red_trajectory[0, 2],
                color=RED_COLOR,
                marker="o",
                s=35,
                label="Red start",
            )
            ax.scatter(
                red_trajectory[-1, 0] / 1000.0,
                red_trajectory[-1, 2],
                color=RED_COLOR,
                marker="x",
                s=45,
                label="Red end",
            )
            red_plotted = True

        for interceptor_index, trajectory in enumerate(interceptor_trajectories, start=1):
            if trajectory.size == 0:
                continue

            color = INTERCEPTOR_COLORS[
                (guidance_index * 2 + interceptor_index - 1) % len(INTERCEPTOR_COLORS)
            ]

            ax.plot(
                trajectory[:, 0] / 1000.0,
                trajectory[:, 2],
                color=color,
                linewidth=1.4,
                label=f"{guidance_mode} - I{interceptor_index}",
            )

    ax.set_title(f"Guidance comparison in X-Z plane | {maneuver_name}")
    ax.set_xlabel("X forward distance (km)")
    ax.set_ylabel("Z lateral distance (m)")
    ax.grid(True)
    ax.legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(save_path, dpi=220)
    plt.close(figure)

    return save_path


def save_summary_csv(summaries: List[Dict[str, object]], save_path: Path) -> Path:
    """保存所有 case 的 summary。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    base_keys = [
        "maneuver_name",
        "guidance_mode",
        "interceptor_count",
        "scenario_profile",
        "ability_profile",
        "dt",
        "max_time",
        "min_distance",
        "continuous_min_distance",
        "sampled_min_distance",
        "final_distance",
        "target_distance",
        "final_time",
        "step_reward",
        "total_reward",
        "terminated",
        "truncated",
        "success",
        "intercepted",
        "passed",
        "termination_reason",
        "threat_interceptor_id",
    ]

    extra_keys = sorted({key for item in summaries for key in item.keys() if key not in base_keys})
    keys = base_keys + extra_keys

    with save_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item.get(key, "") for key in keys})

    return save_path


def main() -> None:
    """脚本主入口：批量运行固定红方机动和蓝方制导模式组合。"""
    args = parse_args()

    env_config_path: Optional[Path]
    env_config_path = None if args.no_env_config else args.env_config

    root_output_dir = resolve_output_dir(args.output_dir, args.interceptor_count)
    comparison_output_dir = root_output_dir / "_comparisons_by_maneuver"
    root_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    selected_maneuvers = list(dict.fromkeys(args.maneuvers))
    selected_guidance_modes = list(dict.fromkeys(args.guidance_modes))

    print("=" * 80)
    print(
        "Visualization config: "
        f"env_config={env_config_path}, "
        f"interceptor_count={args.interceptor_count}, "
        f"scenario_profile={args.scenario_profile}, "
        f"ability_profile={args.interceptor_ability_profile}, "
        f"initial_randomization={args.enable_initial_randomization}, "
        f"seed={args.seed}, "
        f"max_time={args.max_time}, "
        f"dt={args.dt}"
    )
    print(f"Guidance modes: {selected_guidance_modes}")
    print(f"Maneuvers: {selected_maneuvers}")

    all_summaries: List[Dict[str, object]] = []
    summaries_by_guidance: Dict[str, List[Dict[str, object]]] = {
        guidance_mode: [] for guidance_mode in selected_guidance_modes
    }

    for maneuver_name in selected_maneuvers:
        maneuver_policy = MANEUVER_POLICIES[maneuver_name]
        env_by_guidance: Dict[str, PursueEscapeEnv] = {}

        for guidance_mode in selected_guidance_modes:
            print("=" * 80)
            print(
                f"Running maneuver={maneuver_name}, "
                f"guidance={guidance_mode}, "
                f"interceptor_count={args.interceptor_count}"
            )

            env, summary = run_single_case(
                guidance_mode=guidance_mode,
                maneuver_name=maneuver_name,
                maneuver_policy=maneuver_policy,
                seed=args.seed,
                interceptor_count=args.interceptor_count,
                scenario_profile=args.scenario_profile,
                interceptor_ability_profile=args.interceptor_ability_profile,
                interceptor_initial_theta_deg=args.interceptor_initial_theta_deg,
                max_time=args.max_time,
                dt=args.dt,
                env_config_path=env_config_path,
                initial_randomization_enabled=args.enable_initial_randomization,
            )

            env_by_guidance[guidance_mode] = env
            all_summaries.append(summary)
            summaries_by_guidance[guidance_mode].append(summary)

            guidance_output_dir = root_output_dir / guidance_mode
            guidance_output_dir.mkdir(parents=True, exist_ok=True)
            case_prefix = f"{maneuver_name}"

            if not args.skip_plots:
                plot_episode_summary(
                    env=env,
                    save_path=guidance_output_dir / f"{case_prefix}__episode_summary.png",
                    title=f"{maneuver_name} | {guidance_mode}",
                    show=False,
                )

                plot_full_trajectory_summary(
                    env=env,
                    save_path=guidance_output_dir / f"{case_prefix}__full_trajectory.png",
                    show=False,
                )

            print(
                f"min_distance={float(summary['min_distance']):.3f} m, "
                f"sampled_min_distance={float(summary['sampled_min_distance']):.3f} m, "
                f"final_time={float(summary['final_time']):.3f} s, "
                f"total_reward={float(summary['total_reward']):.3f}, "
                f"step_reward={float(summary['step_reward']):.3f}, "
                f"termination_reason={summary['termination_reason']}, "
                f"success={summary['success']}, "
                f"intercepted={summary['intercepted']}, "
                f"truncated={summary['truncated']}"
            )

        if not args.skip_plots:
            plot_guidance_comparison_xz(
                maneuver_name=maneuver_name,
                env_by_guidance=env_by_guidance,
                save_path=comparison_output_dir / f"{maneuver_name}__guidance_comparison_xz.png",
            )

    summary_path = save_summary_csv(
        summaries=all_summaries,
        save_path=root_output_dir / "summary_all.csv",
    )

    for guidance_mode, guidance_summaries in summaries_by_guidance.items():
        save_summary_csv(
            summaries=guidance_summaries,
            save_path=root_output_dir / guidance_mode / "summary.csv",
        )

    print("=" * 80)
    print(f"All outputs saved to: {root_output_dir}")
    if not args.skip_plots:
        print(f"Comparison figures saved to: {comparison_output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
