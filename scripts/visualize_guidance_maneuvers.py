import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
from hypersonic_rl.utils import (
    build_dataclass_config,
    load_config_stack_from_project,
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


def build_env_config_from_yaml_and_overrides(
    env_config_path: Optional[str | Path | List[str | Path]],
    overrides: Dict[str, Any],
) -> PursueEscapeEnvConfig:
    """
    读取一个或多个环境 YAML，并用当前测试 case 的命令行参数覆盖。

    加载顺序：
        1. base YAML
        2. override YAML
        3. 当前 case 的 overrides

    后面的配置覆盖前面的配置。
    """
    if env_config_path is None:
        config_dict: Dict[str, Any] = {}
    else:
        config_dict = load_config_stack_from_project(env_config_path)

    # 只让非 None 的命令行参数覆盖 YAML。
    # 这样没有在命令行显式指定的字段，会继续沿用 base/override YAML。
    clean_overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }

    return build_dataclass_config(
        config_dict,
        PursueEscapeEnvConfig,
        overrides=clean_overrides,
    )


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
        type=str,
        nargs="+",
        default=["configs/env/base_200km.yaml"],
        help=(
            "环境 YAML 路径，可以传入一个或多个。"
            "例如：--env-config configs/env/base_200km.yaml configs/env/override_record_radar_activation.yaml"
        ),
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
        default=None,
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
        choices=["paper_200km_end_to_end", "paper_30km_radar_engagement", "manual_pair", "custom"],
        default=None,
        help="初始态势 profile。",
    )

    parser.add_argument(
        "--interceptor-ability-profile",
        choices=["weak", "paper", "strong", "custom"],
        default=None,
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

    pitch_over_group = parser.add_mutually_exclusive_group()
    pitch_over_group.add_argument(
        "--enable-launch-pitch-over",
        dest="enable_launch_pitch_over",
        action="store_true",
        help="显式开启发射后 pitch-over；未指定时沿用 YAML 配置。",
    )
    pitch_over_group.add_argument(
        "--disable-launch-pitch-over",
        dest="enable_launch_pitch_over",
        action="store_false",
        help="显式关闭发射后 pitch-over；用于和开启状态做对照。",
    )
    parser.set_defaults(enable_launch_pitch_over=None)

    initial_randomization_group = parser.add_mutually_exclusive_group()
    initial_randomization_group.add_argument(
        "--enable-initial-randomization",
        dest="initial_randomization_enabled",
        action="store_true",
        help="显式启用环境初始随机化。",
    )
    initial_randomization_group.add_argument(
        "--disable-initial-randomization",
        dest="initial_randomization_enabled",
        action="store_false",
        help="显式关闭环境初始随机化。",
    )

    parser.set_defaults(initial_randomization_enabled=None)

    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="单回合最大仿真时长，单位 s。",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
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


def resolve_output_dir(output_dir: Path | None, interceptor_count: Optional[int]) -> Path:
    """根据命令行参数生成最终输出目录。"""
    if output_dir is None:
        suffix = f"{interceptor_count}missile" if interceptor_count is not None else "config"
        return PROJECT_ROOT / "outputs" / f"guidance_maneuver_visualization_{suffix}"

    return output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir



def _info_trace_float_array(env: PursueEscapeEnv, key: str) -> np.ndarray:
    """
    从 env.info_trace 中提取指定字段的浮点序列。

    说明：
        pitch-over 诊断字段只在该阶段执行时出现；未出现的步用 NaN 补齐，
        这样绘图和 summary 统计都能自然跳过无效点。
    """
    values: List[float] = []
    for info in getattr(env, "info_trace", []):
        try:
            values.append(float(info.get(key, np.nan)))
        except (TypeError, ValueError):
            values.append(np.nan)

    return np.asarray(values, dtype=np.float64)


def _info_trace_bool_array(env: PursueEscapeEnv, key: str) -> np.ndarray:
    """从 env.info_trace 中提取布尔诊断序列。"""
    values = [bool(info.get(key, False)) for info in getattr(env, "info_trace", [])]
    return np.asarray(values, dtype=bool)


def _info_time_axis(env: PursueEscapeEnv) -> np.ndarray:
    """返回与 info_trace 对齐的时间轴；缺少 time_trace 时退化为 step * dt。"""
    info_length = len(getattr(env, "info_trace", []))
    time_trace = np.asarray(getattr(env, "time_trace", []), dtype=np.float64)

    if time_trace.size == info_length:
        return time_trace

    dt = float(getattr(getattr(env, "config", object()), "dt", 1.0))
    return np.arange(info_length, dtype=np.float64) * dt


def add_launch_pitch_over_summary(env: PursueEscapeEnv, summary: Dict[str, object]) -> None:
    """
    将 pitch-over 阶段统计写入 summary。

    工程意图：
        last_info 往往已经处于中制导或末制导，里面可能不再携带 pitch-over 字段；
        因此这里从完整 info_trace 回看，统计是否进入、持续多久以及因何退出。
    """
    config = getattr(env, "config", object())
    interceptor_count = int(getattr(config, "interceptor_count", 1))
    dt = float(getattr(config, "dt", np.nan))

    summary["enable_launch_pitch_over"] = bool(
        getattr(config, "enable_launch_pitch_over", False)
    )
    summary["launch_pitch_over_activation_theta_deg"] = float(
        getattr(config, "launch_pitch_over_activation_theta_deg", np.nan)
    )
    summary["launch_pitch_over_fixed_theta_deg"] = float(
        getattr(config, "launch_pitch_over_fixed_theta_deg", np.nan)
    )

    for index in range(1, interceptor_count + 1):
        prefix = f"interceptor_{index}"
        active = _info_trace_bool_array(env, f"{prefix}_launch_pitch_over_active_this_step")
        exit_ready = _info_trace_bool_array(env, f"{prefix}_launch_pitch_over_exit_ready")
        finished = _info_trace_bool_array(env, f"{prefix}_launch_pitch_over_finished")

        active_steps = int(np.count_nonzero(active))
        summary[f"{prefix}_launch_pitch_over_entered"] = bool(active_steps > 0)
        summary[f"{prefix}_launch_pitch_over_steps"] = active_steps
        summary[f"{prefix}_launch_pitch_over_duration"] = (
            float(active_steps) * dt if np.isfinite(dt) else np.nan
        )
        summary[f"{prefix}_launch_pitch_over_finished"] = bool(np.any(finished))

        exit_reason = ""
        for info in getattr(env, "info_trace", []):
            if bool(info.get(f"{prefix}_launch_pitch_over_exit_ready", False)):
                exit_reason = str(
                    info.get(f"{prefix}_launch_pitch_over_exit_reason", "")
                )
                break
        summary[f"{prefix}_launch_pitch_over_exit_reason"] = exit_reason

        reference_theta = _info_trace_float_array(
            env, f"{prefix}_launch_pitch_over_reference_theta_deg"
        )
        blend_weight = _info_trace_float_array(
            env, f"{prefix}_launch_pitch_over_blend_weight"
        )

        finite_reference = reference_theta[np.isfinite(reference_theta)]
        finite_blend = blend_weight[np.isfinite(blend_weight)]

        summary[f"{prefix}_launch_pitch_over_exit_ready"] = bool(np.any(exit_ready))
        summary[f"{prefix}_launch_pitch_over_reference_theta_initial_deg"] = (
            float(finite_reference[0]) if finite_reference.size > 0 else np.nan
        )
        summary[f"{prefix}_launch_pitch_over_reference_theta_final_deg"] = (
            float(finite_reference[-1]) if finite_reference.size > 0 else np.nan
        )
        summary[f"{prefix}_launch_pitch_over_max_blend_weight"] = (
            float(np.max(finite_blend)) if finite_blend.size > 0 else np.nan
        )


# ---------------------------------------------------------------------
# 单次仿真
# ---------------------------------------------------------------------
def run_single_case(
    guidance_mode: str,
    maneuver_name: str,
    maneuver_policy: ManeuverPolicy,
    seed: int = 0,
    interceptor_count: Optional[int] = None,
    scenario_profile: Optional[str] = None,
    interceptor_ability_profile: Optional[str] = None,
    interceptor_initial_theta_deg: Optional[int] = None,
    enable_launch_pitch_over: Optional[bool] = None,
    max_time: Optional[float] = None,
    dt: Optional[float] = None,
    env_config_path: Optional[str | Path | List[str | Path]] = None,
    initial_randomization_enabled: Optional[bool] = None,
) -> Tuple[PursueEscapeEnv, Dict[str, object]]:
    """运行单个红方固定机动 + 蓝方制导模式组合。"""
    overrides: Dict[str, Any] = {
        # guidance_mode 是本脚本要对比的变量，所以始终覆盖 YAML。
        "guidance_mode": guidance_mode,

        # 下面这些只有命令行显式传入时才覆盖 YAML。
        "initial_randomization_enabled": initial_randomization_enabled,
        "interceptor_count": interceptor_count,
        "scenario_profile": scenario_profile,
        "interceptor_ability_profile": interceptor_ability_profile,
        "interceptor_initial_theta_deg": interceptor_initial_theta_deg,
        "t": max_time,
        "dt": dt,
    }

    if enable_launch_pitch_over is not None:
        # enable_launch_pitch_over：
        #     None 表示完全沿用 YAML；True/False 表示命令行显式覆盖。
        #     这样直接改配置文件可以响应，也能在批量对比时临时开关 pitch-over。
        overrides["enable_launch_pitch_over"] = bool(enable_launch_pitch_over)

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
        "radar_detection_distance": float(env.config.radar_detection_distance),
        "red_intelligent_activation_mode": str(env.config.red_intelligent_activation_mode),
        "red_intelligent_active": bool(last_info.get("red_intelligent_active", False)),
        "red_intelligent_activation_time": last_info.get("red_intelligent_activation_time", ""),
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

    add_launch_pitch_over_summary(env=env, summary=summary)

    # 保留每枚弹自己的命中、错过、最小距离、阶段字段，以及 reward.py 新增的诊断字段。
    for key, value in last_info.items():
        if (
            key.endswith("intercepted")
            or key.endswith("passed")
            or key.endswith("min_distance")
            or key.endswith("phase")
            or "launch_pitch_over" in key
            or key.startswith("radar_detection")
            or key.startswith("red_intelligent")
            or key == "red_requested_overload"
            or key == "red_command_gated_by_radar"
            or key.endswith("target_compensation_skipped")
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


def plot_launch_pitch_over_diagnostics(
    env: PursueEscapeEnv,
    save_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    绘制发射后 pitch-over 阶段诊断图。

    图中重点检查三件事：
        1. 大倾角初始段参考角是否固定为 20 deg；
        2. theta 进入 30 deg -> 20 deg 后是否平滑融合到 LOS pitch angle；
        3. pitch-over 阶段纵向指令是否推动弹道倾角下降。
    """
    info_trace = getattr(env, "info_trace", [])
    if not info_trace:
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)

    time_axis = _info_time_axis(env)
    interceptor_count = int(getattr(env.config, "interceptor_count", 1))
    theta_traces = getattr(env, "interceptor_theta_traces", [])
    ny_command_traces = getattr(env, "interceptor_ny_command_traces", [])

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    if title is not None:
        figure.suptitle(title)

    active_any = False

    def plot_finite(
        ax: Any,
        values: np.ndarray,
        *,
        label: str,
        color: str,
        linestyle: str = "-",
        linewidth: float = 1.3,
    ) -> bool:
        common_length = min(time_axis.size, values.size)
        if common_length <= 0:
            return False

        x_values = time_axis[:common_length]
        y_values = values[:common_length]
        finite_mask = np.isfinite(y_values)
        if not np.any(finite_mask):
            return False

        ax.plot(
            x_values[finite_mask],
            y_values[finite_mask],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        return True

    for index in range(1, interceptor_count + 1):
        prefix = f"interceptor_{index}"
        color = INTERCEPTOR_COLORS[(index - 1) % len(INTERCEPTOR_COLORS)]

        if index - 1 < len(theta_traces):
            theta_deg = np.degrees(np.asarray(theta_traces[index - 1], dtype=np.float64))
            plot_finite(
                axes[0],
                theta_deg,
                label=f"I{index} theta",
                color=color,
                linewidth=1.5,
            )

        reference_theta_deg = _info_trace_float_array(
            env, f"{prefix}_launch_pitch_over_reference_theta_deg"
        )
        los_theta_deg = _info_trace_float_array(
            env, f"{prefix}_launch_pitch_over_los_theta_deg"
        )
        blend_weight = _info_trace_float_array(
            env, f"{prefix}_launch_pitch_over_blend_weight"
        )
        vertical_maneuver = _info_trace_float_array(
            env, f"{prefix}_launch_pitch_over_vertical_maneuver"
        )
        active = _info_trace_bool_array(
            env, f"{prefix}_launch_pitch_over_active_this_step"
        )

        active_any = active_any or bool(np.any(active))

        plot_finite(
            axes[0],
            reference_theta_deg,
            label=f"I{index} ref theta",
            color=color,
            linestyle="--",
            linewidth=1.2,
        )
        plot_finite(
            axes[0],
            los_theta_deg,
            label=f"I{index} LOS theta",
            color=color,
            linestyle=":",
            linewidth=1.1,
        )
        plot_finite(
            axes[1],
            blend_weight,
            label=f"I{index} blend",
            color=color,
            linewidth=1.4,
        )
        plot_finite(
            axes[2],
            vertical_maneuver,
            label=f"I{index} vertical maneuver",
            color=color,
            linestyle="--",
            linewidth=1.1,
        )

        if index - 1 < len(ny_command_traces):
            ny_command = np.asarray(ny_command_traces[index - 1], dtype=np.float64)
            plot_finite(
                axes[2],
                ny_command,
                label=f"I{index} ny command",
                color=color,
                linewidth=1.5,
            )

        common_length = min(time_axis.size, active.size)
        if common_length > 0 and np.any(active[:common_length]):
            axes[1].fill_between(
                time_axis[:common_length],
                0.0,
                1.0,
                where=active[:common_length],
                color=color,
                alpha=0.10,
            )

    if not active_any:
        axes[1].text(
            0.5,
            0.5,
            "No active launch pitch-over steps",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    axes[0].set_ylabel("Pitch angle (deg)")
    axes[0].set_title("Pitch-over reference tracking")
    axes[0].grid(True)
    axes[0].legend(fontsize=7)

    axes[1].set_ylabel("Blend weight")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Fixed 20 deg to LOS blending")
    axes[1].grid(True)
    axes[1].legend(fontsize=7)

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Overload command (g)")
    axes[2].set_title("Longitudinal command during pitch-over")
    axes[2].grid(True)
    axes[2].legend(fontsize=7)

    if title is None:
        figure.tight_layout()
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

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
        "radar_detection_distance",
        "red_intelligent_activation_mode",
        "red_intelligent_active",
        "red_intelligent_activation_time",
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

    env_config_path: Optional[List[str]]
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
        f"initial_randomization={args.initial_randomization_enabled}, "
        f"launch_pitch_over_override={args.enable_launch_pitch_over}, "
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
                enable_launch_pitch_over=args.enable_launch_pitch_over,
                max_time=args.max_time,
                dt=args.dt,
                env_config_path=env_config_path,
                initial_randomization_enabled=args.initial_randomization_enabled,
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

                pitch_over_enabled = bool(
                    getattr(env.config, "enable_launch_pitch_over", False)
                )
                pitch_over_entered = any(
                    bool(summary.get(f"interceptor_{index}_launch_pitch_over_entered", False))
                    for index in range(1, int(env.config.interceptor_count) + 1)
                )

                if pitch_over_enabled or pitch_over_entered:
                    plot_launch_pitch_over_diagnostics(
                        env=env,
                        save_path=guidance_output_dir / f"{case_prefix}__launch_pitch_over.png",
                        title=f"{maneuver_name} | {guidance_mode} | launch pitch-over",
                    )

            print(
                f"min_distance={float(summary['min_distance']):.3f} m, "
                f"sampled_min_distance={float(summary['sampled_min_distance']):.3f} m, "
                f"final_time={float(summary['final_time']):.3f} s, "
                f"total_reward={float(summary['total_reward']):.3f}, "
                f"step_reward={float(summary['step_reward']):.3f}, "
                f"pitch_over={summary.get('interceptor_1_launch_pitch_over_entered', False)}, "
                f"pitch_over_duration={float(summary.get('interceptor_1_launch_pitch_over_duration', 0.0)):.3f} s, "
                f"pitch_over_exit={summary.get('interceptor_1_launch_pitch_over_exit_reason', '')}, "
                f"red_intelligent_active={summary.get('red_intelligent_active', False)}, "
                f"red_intelligent_activation_time={summary.get('red_intelligent_activation_time', '')}, "
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
