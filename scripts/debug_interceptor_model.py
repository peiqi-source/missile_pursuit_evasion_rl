"""
debug_interceptor_model.py

作用：
    测试完整中末制导拦截弹是否形成有效拦截闭环。

测试 case：
    1. red_action = 0
    2. red_action = +max
    3. red_action = -max
    4. source_rule_action

对比模式：
    1. source_pn
    2. mid_terminal_interceptor

输出：
    outputs/debug_interceptor_model/
        interceptor_debug_summary.csv
        各 case 图像
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.utils import load_config_from_project
from hypersonic_rl.visualization import (
    plot_3d_trajectory,
    plot_episode_summary,
    plot_full_trajectory_summary,
    plot_three_view_trajectory,
)


def filter_dataclass_config(config_dict: Dict[str, Any], dataclass_type: Any) -> Dict[str, Any]:
    """
    过滤配置字典，只保留 dataclass 支持的字段。
    """
    valid_field_names = {field.name for field in fields(dataclass_type)}

    filtered_config = {
        key: value
        for key, value in config_dict.items()
        if key in valid_field_names
    }

    return filtered_config


def build_env(guidance_mode: str, seed_fixed_head_on: bool = False) -> PursueEscapeEnv:
    """
    创建指定制导模式下的环境。

    参数：
        guidance_mode：
            蓝方制导模式。

        seed_fixed_head_on：
            是否使用严格迎头初始条件。
            True 时，蓝方 x=30000m，航向角=180deg。

    返回：
        env：
            PursueEscapeEnv 环境对象。
    """
    env_config_dict = load_config_from_project("configs/env/pursue_escape_env.yaml")
    filtered_config = filter_dataclass_config(env_config_dict, PursueEscapeEnvConfig)

    filtered_config["guidance_mode"] = guidance_mode
    filtered_config["dt"] = 0.01
    filtered_config["t"] = 20.0

    if seed_fixed_head_on:
        filtered_config["blue_initial_x_min"] = 30000.0
        filtered_config["blue_initial_x_max"] = 30000.0
        filtered_config["blue_initial_heading_min_deg"] = 180.0
        filtered_config["blue_initial_heading_max_deg"] = 180.0

    env_config = PursueEscapeEnvConfig(**filtered_config)
    env = PursueEscapeEnv(env_config)

    return env


def run_case(
    case_name: str,
    guidance_mode: str,
    action_fn: Callable[[PursueEscapeEnv], np.ndarray],
    output_dir: Path,
    seed: int = 0,
    fixed_head_on: bool = False,
) -> Dict[str, Any]:
    """
    运行一个 debug case。
    """
    case_dir = output_dir / guidance_mode / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(
        guidance_mode=guidance_mode,
        seed_fixed_head_on=fixed_head_on,
    )

    observation, info = env.reset(seed=seed)

    initial_distance = float(info["min_distance"])

    done = False

    while not done:
        action = action_fn(env)

        observation, reward, terminated, truncated, info = env.step(action)

        done = bool(terminated or truncated)

    episode_metrics = env.get_episode_metrics()
    final_info = env.info_trace[-1] if env.info_trace else {}

    plot_episode_summary(
        env=env,
        save_path=case_dir / f"{case_name}_summary.png",
        title=f"{case_name} | {guidance_mode}",
        show=False,
    )

    plot_three_view_trajectory(
        env=env,
        save_path=case_dir / f"{case_name}_three_views.png",
        show=False,
    )

    plot_3d_trajectory(
        env=env,
        save_path=case_dir / f"{case_name}_trajectory_3d.png",
        show=False,
    )

    plot_full_trajectory_summary(
        env=env,
        save_path=case_dir / f"{case_name}_full_trajectory_summary.png",
        show=False,
    )

    record = {
        "case_name": case_name,
        "guidance_mode": guidance_mode,
        "fixed_head_on": bool(fixed_head_on),
        "initial_distance": initial_distance,
        "min_distance": float(episode_metrics["min_distance"]),
        "final_distance": float(episode_metrics["final_distance"]),
        "success": float(episode_metrics["success"]),
        "episode_steps": int(episode_metrics["episode_steps"]),
        "episode_time": float(episode_metrics["episode_time"]),
        "termination_reason": final_info.get("termination_reason", "unknown"),
        "intercepted": bool(final_info.get("intercepted", False)),
        "passed": bool(final_info.get("passed", False)),
        "interceptor_phase_final": final_info.get("interceptor_phase", "unknown"),
        "blue_ny_actual_final": float(final_info.get("blue_ny_actual", np.nan)),
        "blue_nz_actual_final": float(final_info.get("blue_nz_actual", np.nan)),
        "blue_ny_command_final": float(final_info.get("blue_ny_command", np.nan)),
        "blue_nz_command_final": float(final_info.get("blue_nz_command", np.nan)),
        "figure_dir": str(case_dir),
    }

    return record


def main() -> None:
    """
    主函数。
    """
    output_dir = project_root / "outputs" / "debug_interceptor_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_env = build_env(guidance_mode="mid_terminal_interceptor")
    action_max = float(temp_env.config.nzc_h_max)

    cases: Dict[str, Callable[[PursueEscapeEnv], np.ndarray]] = {
        "zero_action": lambda env: np.array([0.0], dtype=np.float32),

        "positive_max_action": lambda env: np.array([action_max], dtype=np.float32),

        "negative_max_action": lambda env: np.array([-action_max], dtype=np.float32),

        "source_rule_action": lambda env: np.array(
            [
                action_max
                if len(env.distance_trace) > 0 and min(env.distance_trace) < 7000.0
                else -action_max
            ],
            dtype=np.float32,
        ),
    }

    guidance_modes = [
        "source_pn",
        "mid_terminal_interceptor",
    ]

    all_records = []

    print("=" * 100)
    print("开始测试完整中末制导拦截弹闭环")
    print("=" * 100)

    for fixed_head_on in [True, False]:
        for guidance_mode in guidance_modes:
            for case_name, action_fn in cases.items():
                record = run_case(
                    case_name=case_name,
                    guidance_mode=guidance_mode,
                    action_fn=action_fn,
                    output_dir=output_dir,
                    seed=0,
                    fixed_head_on=fixed_head_on,
                )

                all_records.append(record)

                print(
                    f"[{guidance_mode} | fixed={fixed_head_on} | {case_name}] "
                    f"initial={record['initial_distance']:.2f} m, "
                    f"min={record['min_distance']:.2f} m, "
                    f"final={record['final_distance']:.2f} m, "
                    f"reason={record['termination_reason']}, "
                    f"intercepted={record['intercepted']}, "
                    f"phase={record['interceptor_phase_final']}"
                )

    summary_path = output_dir / "interceptor_debug_summary.csv"
    pd.DataFrame(all_records).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("完整拦截弹 debug 完成")
    print(f"结果表：{summary_path}")
    print(f"图像目录：{output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()