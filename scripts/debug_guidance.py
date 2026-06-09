"""
作用：
    检查当前环境坐标轴、运动学符号和蓝方制导律是否能形成有效追逃。

测试内容：
    1. 红方不机动：red_action = 0
    2. 红方最大正向机动：red_action = +nzc_h_max
    3. 红方最大负向机动：red_action = -nzc_h_max
    4. 源程序注释中的简单规则：
        如果最小距离小于 7000 m，则 action = +3；
        否则 action = -3。

输出内容：
    outputs/debug_guidance_compare/
        guidance_compare_summary.csv
        zero_action/
        positive_max_action/
        negative_max_action/
        source_rule_action/
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# src_path：源码目录。
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

    return {
        key: value
        for key, value in config_dict.items()
        if key in valid_field_names
    }


def build_debug_env() -> PursueEscapeEnv:
    """
    创建 debug 环境。

    参数：
        guidance_mode：
            蓝方制导模式。

    返回：
        env：
            PursueEscapeEnv 环境对象。
    """
    # env_config_dict：从 yaml 读取环境配置。
    env_config_dict = load_config_from_project("configs/env/pursue_escape_env.yaml")

    # filtered_config：过滤掉 dataclass 不支持的字段。
    filtered_config = filter_dataclass_config(env_config_dict, PursueEscapeEnvConfig)

    # debug 时建议使用源程序训练脚本中的 dt=0.01。
    filtered_config["dt"] = 0.01
    filtered_config["t"] = 20.0

    # env_config：环境配置对象。
    env_config = PursueEscapeEnvConfig(**filtered_config)

    # env：环境对象。
    env = PursueEscapeEnv(env_config)

    return env


def run_case(
    case_name: str,
    action_fn: Callable[[PursueEscapeEnv], np.ndarray],
    output_dir: Path,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    运行一个固定动作测试案例。

    参数：
        case_name：
            案例名称。

        action_fn：
            根据当前环境返回红方动作的函数。

        output_dir：
            输出目录。

        guidance_mode：
            蓝方制导模式。

        seed：
            随机种子。

    返回：
        metrics：
            当前案例的统计结果。
    """
    # case_dir：当前案例输出目录。
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # env：debug 环境。
    env = build_debug_env()

    # observation/info：环境初始化。
    observation, info = env.reset(seed=seed)

    # initial_distance：初始距离。
    initial_distance = float(info["min_distance"])

    # done：统一终止标志。
    done = False

    while not done:
        # action：当前红方固定或规则动作。
        action = action_fn(env)

        # 环境推进一步。
        observation, reward, terminated, truncated, info = env.step(action)

        done = bool(terminated or truncated)

    # episode_metrics：环境自带回合指标。
    episode_metrics = env.get_episode_metrics()

    # final_info：最后一步 info。
    final_info = env.info_trace[-1] if env.info_trace else {}

    # 保存图像。
    plot_episode_summary(
        env=env,
        save_path=case_dir / f"{case_name}_summary.png",
        title=f"{case_name}",
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

    # metrics：当前案例统计结果。
    metrics: Dict[str, Any] = {
        "case_name": case_name,
        "initial_distance": initial_distance,
        "min_distance": float(episode_metrics["min_distance"]),
        "final_distance": float(episode_metrics["final_distance"]),
        "success": float(episode_metrics["success"]),
        "episode_steps": int(episode_metrics["episode_steps"]),
        "episode_time": float(episode_metrics["episode_time"]),
        "termination_reason": final_info.get("termination_reason", "unknown"),
        "intercepted": bool(final_info.get("intercepted", False)),
        "passed": bool(final_info.get("passed", False)),
        "final_red_x": float(env.red_state[0]),
        "final_red_z": float(env.red_state[2]),
        "final_blue_x": float(env.blue_state[0]),
        "final_blue_z": float(env.blue_state[2]),
        "final_blue_command": float(final_info.get("blue_command_overload", np.nan)),
        "final_blue_inertial": float(final_info.get("blue_inertial_overload", np.nan)),
        "figure_dir": str(case_dir),
    }

    return metrics


def main() -> None:
    """
    debug 主函数。
    """
    # output_dir：debug 输出目录。
    output_dir = project_root / "outputs" / "debug_guidance"
    output_dir.mkdir(parents=True, exist_ok=True)

    # env：debug 环境。
    env = build_debug_env()
    action_max = float(env.config.nzc_h_max)

    # cases：固定动作与规则动作案例。
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

    # all_records：全部测试记录。
    all_records = []

    print("=" * 100)
    print("开始检查蓝方制导律是否形成有效追逃")
    print("=" * 100)

    # 先测试默认源程序 PN。
    for case_name, action_fn in cases.items():
        record = run_case(
            case_name=case_name,
            action_fn=action_fn,
            output_dir=output_dir / "guidance_pn",
            seed=0,
        )

        all_records.append(record)

        print(
            f"[source_pn | {case_name}] "
            f"initial={record['initial_distance']:.2f} m, "
            f"min={record['min_distance']:.2f} m, "
            f"final={record['final_distance']:.2f} m, "
            f"reason={record['termination_reason']}, "
            f"intercepted={record['intercepted']}"
        )


    # 保存 summary。
    summary_path = output_dir / "guidance_pn.csv"
    pd.DataFrame(all_records).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("制导律 debug 完成")
    print(f"结果表：{summary_path}")
    print(f"图像目录：{output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()