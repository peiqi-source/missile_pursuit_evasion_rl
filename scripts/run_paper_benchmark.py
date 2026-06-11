"""
run_paper_benchmark.py

作用：
    运行论文式一对二端到端突防 benchmark。

覆盖对象：
    1. 固定动作 baseline：zero / sine / random 等；
    2. 普通 SAC checkpoint；
    3. LSTM-SAC checkpoint；
    4. source_pn 与 mid_terminal_interceptor 两种蓝方制导模式。

说明：
    本脚本用于正式实验入口和小规模工程验证。配置中 checkpoint 不存在且 required=false 时会记录 skipped，
    不会阻塞固定动作 baseline 的快速验证。
"""

from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# src_path：保证直接运行脚本时可以导入 hypersonic_rl。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from eval_policy import build_agent, build_env, filter_dataclass_config, resolve_project_path
from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.evaluation.metrics import collect_episode_metrics, save_metrics_to_csv, summarize_metrics
from hypersonic_rl.utils import (
    build_run_manifest,
    create_logger,
    get_device,
    load_checkpoint,
    load_config_from_project,
    save_run_manifest,
)
from hypersonic_rl.visualization import plot_full_trajectory_summary


ActionPolicy = Callable[[Any, int, np.random.Generator], np.ndarray]


def build_fixed_action_policy(policy_config: Dict[str, Any]) -> ActionPolicy:
    """
    构造固定动作 baseline。

    参数：
        policy_config：
            单个 policy 配置，包含 name/type 等字段。

    返回：
        policy：
            接收 env、step_index、rng 并返回动作数组的函数。
    """
    # policy_name：固定动作策略名称。
    policy_name = str(policy_config.get("name", "zero_action"))

    def policy(env: Any, step_index: int, rng: np.random.Generator) -> np.ndarray:
        """
        根据固定规则生成红方一维横向过载动作。

        参数：
            env：
                当前评估环境。
            step_index：
                当前回合步数。
            rng：
                当前回合随机数生成器。

        返回：
            action：
                形状为 [1] 的动作数组。
        """
        # max_action：红方横向过载动作上限。
        max_action = float(env.action_space.high[0])

        if policy_name in {"zero", "zero_action"}:
            value = 0.0
        elif policy_name in {"positive_max", "positive_max_action"}:
            value = max_action
        elif policy_name in {"negative_max", "negative_max_action"}:
            value = -max_action
        elif policy_name in {"sine", "sine_action"}:
            # period_steps：正弦动作周期，默认 5s。
            period_seconds = float(policy_config.get("period_seconds", 5.0))
            period_steps = max(int(round(period_seconds / max(float(env.unwrapped.config.dt), 1e-8))), 1)

            # amplitude：正弦幅值比例。
            amplitude = float(policy_config.get("amplitude", 1.0))
            value = amplitude * max_action * math.sin(2.0 * math.pi * float(step_index) / float(period_steps))
        elif policy_name in {"random", "random_action"}:
            value = float(rng.uniform(-max_action, max_action))
        else:
            raise ValueError(f"未知固定动作 baseline：{policy_name}")

        return np.array([value], dtype=np.float32)

    return policy


def run_fixed_policy_episode(
    env: PursueEscapeEnv,
    policy: ActionPolicy,
    episode_index: int,
    seed: int,
    policy_name: str,
    guidance_mode: str,
) -> Dict[str, Any]:
    """
    运行一个固定动作 baseline 回合。

    参数：
        env：
            评估环境。
        policy：
            固定动作策略函数。
        episode_index：
            当前 benchmark 内部回合编号。
        seed：
            环境随机种子。
        policy_name：
            baseline 名称。
        guidance_mode：
            蓝方制导模式。

    返回：
        metrics：
            单回合指标。
    """
    # rng：固定动作策略内部随机源，保证 random baseline 可复现。
    rng = np.random.default_rng(seed)

    # observation：重置环境并丢弃初始 info。
    reset_result = env.reset(seed=seed)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    # done：统一终止标记。
    done = False
    step_index = 0

    while not done:
        # action：由固定策略生成的一维横向过载。
        action = policy(env, step_index, rng)

        # step_result：兼容 Gymnasium 五返回值。
        step_result = env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            observation, reward, done, info = step_result

        step_index += 1

    # metrics：收集标准单回合指标，并补充 benchmark 维度。
    metrics = collect_episode_metrics(
        env=env,
        episode_index=episode_index,
        extra_info={
            "seed": int(seed),
            "policy_name": policy_name,
            "policy_type": "fixed_action",
            "guidance_mode": guidance_mode,
        },
    )

    return metrics


def run_agent_episode(
    env: Any,
    agent: Any,
    episode_index: int,
    seed: int,
    policy_name: str,
    policy_type: str,
    guidance_mode: str,
    deterministic: bool,
) -> Dict[str, Any]:
    """
    运行一个 SAC/LSTM-SAC checkpoint 回合。

    参数：
        env：
            普通环境或 StateSequenceWrapper 包装环境。
        agent：
            已加载 checkpoint 的智能体。
        episode_index：
            当前 benchmark 内部回合编号。
        seed：
            环境随机种子。
        policy_name：
            策略名称。
        policy_type：
            sac 或 lstm_sac。
        guidance_mode：
            蓝方制导模式。
        deterministic：
            是否使用确定性动作。

    返回：
        metrics：
            单回合指标。
    """
    # reset_result：重置环境。
    reset_result = env.reset(seed=seed)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    done = False
    while not done:
        # action：由策略网络输出。
        action = agent.select_action(observation, deterministic=deterministic)

        # step_result：推进环境。
        step_result = env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            observation, reward, done, info = step_result

    # raw_env：collect_episode_metrics 需要读取底层环境 trace。
    raw_env = getattr(env, "env", env)

    return collect_episode_metrics(
        env=raw_env,
        episode_index=episode_index,
        extra_info={
            "seed": int(seed),
            "policy_name": policy_name,
            "policy_type": policy_type,
            "guidance_mode": guidance_mode,
        },
    )


def build_agent_policy(
    policy_config: Dict[str, Any],
    env_config_dict: Dict[str, Any],
    guidance_mode: str,
    device: str,
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """
    构造 checkpoint 智能体 policy。

    参数：
        policy_config：
            策略配置。
        env_config_dict：
            环境配置字典。
        guidance_mode：
            蓝方制导模式。
        device：
            PyTorch 设备字符串。

    返回：
        env：
            评估环境；checkpoint 缺失时为 None。
        agent：
            已加载 checkpoint 的智能体；checkpoint 缺失时为 None。
        checkpoint：
            checkpoint 字典；checkpoint 缺失时为 None。
        skipped_reason：
            跳过原因；未跳过时为 None。
    """
    # checkpoint_path：解析 checkpoint 路径。
    checkpoint_path = resolve_project_path(str(policy_config.get("checkpoint_path", "")))
    required = bool(policy_config.get("required", False))

    if not checkpoint_path.exists():
        reason = f"checkpoint 不存在：{checkpoint_path}"
        if required:
            raise FileNotFoundError(reason)
        return None, None, None, reason

    # checkpoint：读取模型状态。
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    if "agent_state" not in checkpoint:
        raise KeyError(f"checkpoint 中没有 agent_state 字段：{checkpoint_path}")

    # agent_type：sac 或 lstm_sac。
    agent_type = str(policy_config.get("type", "lstm_sac"))

    # checkpoint_agent_config：优先用 checkpoint 内部网络结构。
    checkpoint_agent_config = checkpoint["agent_state"].get("config", {})

    # sequence_length：LSTM-SAC 序列长度。
    sequence_length = int(checkpoint_agent_config.get("sequence_length", policy_config.get("sequence_length", 3)))

    # env_config_for_policy：为当前蓝方制导模式构造环境。
    env_config_for_policy = deepcopy(env_config_dict)
    env_config_for_policy["guidance_mode"] = guidance_mode

    # env：普通 SAC 使用一维观测，LSTM-SAC 自动包序列。
    env = build_env(env_config_for_policy, agent_type=agent_type, sequence_length=sequence_length)

    # agent_config_dict：读取对应 agent YAML，作为缺省配置。
    default_config_path = "configs/agent/lstm_sac.yaml" if agent_type == "lstm_sac" else "configs/agent/sac.yaml"
    agent_config_dict = load_config_from_project(policy_config.get("agent_config_path", default_config_path))

    # agent：构造并加载参数。
    agent = build_agent(
        agent_type=agent_type,
        agent_config_dict=agent_config_dict,
        env=env,
        device=device,
        checkpoint_agent_config=checkpoint_agent_config,
    )
    agent.load_state_dict(checkpoint["agent_state"], load_optimizers=False)
    agent.eval_mode()

    return env, agent, checkpoint, None


def summarize_by_group(metric_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 policy_name 和 guidance_mode 汇总 benchmark 指标。

    参数：
        metric_records：
            逐回合指标列表。

    返回：
        summaries：
            分组汇总指标列表。
    """
    if not metric_records:
        return []

    # dataframe：便于按策略和制导模式分组。
    dataframe = pd.DataFrame(metric_records)
    summaries: List[Dict[str, Any]] = []

    for (policy_name, guidance_mode), group in dataframe.groupby(["policy_name", "guidance_mode"], dropna=False):
        # records：当前分组的回合指标。
        records = group.to_dict(orient="records")

        # summary：复用通用 metrics 汇总。
        summary = summarize_metrics(records)
        summary["policy_name"] = str(policy_name)
        summary["guidance_mode"] = str(guidance_mode)

        # success_ci95_half_width：二项成功率 95% 置信区间半宽。
        n = max(int(len(records)), 1)
        p = float(summary.get("success_rate", 0.0))
        summary["success_ci95_half_width"] = float(1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n))

        # min_distance_ci95_half_width：最小脱靶量均值 95% 置信区间半宽。
        min_distance_values = group["min_distance"].astype(float).to_numpy()
        if min_distance_values.size > 1:
            summary["min_distance_ci95_half_width"] = float(
                1.96 * np.std(min_distance_values, ddof=1) / math.sqrt(float(min_distance_values.size))
            )
        else:
            summary["min_distance_ci95_half_width"] = 0.0

        summaries.append(summary)

    return summaries


def write_markdown_report(
    output_path: Path,
    summary_records: List[Dict[str, Any]],
    skipped_records: List[Dict[str, Any]],
) -> Path:
    """
    保存 benchmark Markdown 报告。

    参数：
        output_path：
            报告路径。
        summary_records：
            分组汇总记录。
        skipped_records：
            被跳过的 checkpoint baseline。

    返回：
        saved_path：
            实际保存路径。
    """
    # lines：Markdown 文本行。
    lines = [
        "# Paper Benchmark Report",
        "",
        "本报告由 `scripts/run_paper_benchmark.py` 自动生成，用于工程验证和论文式对比实验归档。",
        "",
        "## Summary",
        "",
        "| Policy | Guidance | Episodes | Success | Mean Miss (m) | Mean Target Offset (m) | Red Effort | Interceptor Saturation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for record in summary_records:
        lines.append(
            "| {policy} | {guidance} | {episodes:.0f} | {success:.3f} ± {ci:.3f} | "
            "{miss:.3f} | {target:.3f} | {red:.3f} | {sat:.3f} |".format(
                policy=record.get("policy_name", "unknown"),
                guidance=record.get("guidance_mode", "unknown"),
                episodes=float(record.get("num_episodes", 0.0)),
                success=float(record.get("success_rate", 0.0)),
                ci=float(record.get("success_ci95_half_width", 0.0)),
                miss=float(record.get("mean_min_distance", 0.0)),
                target=float(record.get("mean_target_offset", 0.0)),
                red=float(record.get("mean_red_control_effort_abs", 0.0)),
                sat=float(record.get("mean_interceptor_command_saturation_ratio", 0.0)),
            )
        )

    if skipped_records:
        lines.extend(["", "## Skipped", ""])
        for record in skipped_records:
            lines.append(f"- `{record['policy_name']}` / `{record['guidance_mode']}`: {record['reason']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main(config_path: str = "configs/eval/benchmark_paper.yaml") -> None:
    """
    benchmark 主入口。

    参数：
        config_path：
            benchmark YAML 配置路径。
    """
    # benchmark_config：读取 benchmark 配置。
    benchmark_config = load_config_from_project(config_path)

    # output_dir：benchmark 输出目录。
    output_dir = resolve_project_path(benchmark_config.get("output_dir", "outputs/benchmark/paper"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # logger：benchmark 日志。
    logger = create_logger(
        logger_name="paper_benchmark",
        log_dir=output_dir / "logs",
        log_filename="benchmark.log",
    )

    # env_config_dict：基础环境配置，可被 benchmark env_overrides 覆盖。
    env_config_dict = load_config_from_project(benchmark_config.get("env_config_path", "configs/env/pursue_escape_env.yaml"))
    env_config_dict.update(benchmark_config.get("env_overrides", {}))

    # seeds/guidance_modes/policies：benchmark 维度。
    seeds = [int(seed) for seed in benchmark_config.get("seeds", [0, 1])]
    guidance_modes = [str(mode) for mode in benchmark_config.get("guidance_modes", ["source_pn", "mid_terminal_interceptor"])]
    policies = list(benchmark_config.get("policies", []))

    # device：checkpoint policy 使用的设备。
    device = str(get_device(prefer_gpu=bool(benchmark_config.get("prefer_gpu", True))))

    # metric_records/skipped_records：逐回合结果和跳过记录。
    metric_records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []

    # episode_index：全局回合编号。
    episode_index = 0

    for guidance_mode in guidance_modes:
        for policy_config in policies:
            # policy_name/policy_type：当前策略描述。
            policy_name = str(policy_config.get("name", "unknown_policy"))
            policy_type = str(policy_config.get("type", "fixed_action"))

            if policy_type == "fixed_action":
                # env_config_for_policy：固定动作 baseline 使用原始环境。
                env_config_for_policy = deepcopy(env_config_dict)
                env_config_for_policy["guidance_mode"] = guidance_mode
                env = PursueEscapeEnv(
                    PursueEscapeEnvConfig(**filter_dataclass_config(env_config_for_policy, PursueEscapeEnvConfig))
                )
                fixed_policy = build_fixed_action_policy(policy_config)

                for seed in seeds:
                    logger.info("运行 fixed policy=%s guidance=%s seed=%d", policy_name, guidance_mode, seed)
                    metrics = run_fixed_policy_episode(
                        env=env,
                        policy=fixed_policy,
                        episode_index=episode_index,
                        seed=seed,
                        policy_name=policy_name,
                        guidance_mode=guidance_mode,
                    )
                    metric_records.append(metrics)
                    episode_index += 1

                    if bool(benchmark_config.get("save_first_episode_plot", True)) and seed == seeds[0]:
                        plot_path = output_dir / "figures" / f"{policy_name}_{guidance_mode}_seed{seed}_trajectory.png"
                        plot_full_trajectory_summary(env=env, save_path=plot_path, show=False)

            elif policy_type in {"sac", "lstm_sac"}:
                env, agent, checkpoint, skipped_reason = build_agent_policy(
                    policy_config=policy_config,
                    env_config_dict=env_config_dict,
                    guidance_mode=guidance_mode,
                    device=device,
                )
                if skipped_reason is not None:
                    logger.warning("跳过 policy=%s guidance=%s：%s", policy_name, guidance_mode, skipped_reason)
                    skipped_records.append(
                        {
                            "policy_name": policy_name,
                            "policy_type": policy_type,
                            "guidance_mode": guidance_mode,
                            "reason": skipped_reason,
                        }
                    )
                    continue

                for seed in seeds:
                    logger.info("运行 agent policy=%s guidance=%s seed=%d", policy_name, guidance_mode, seed)
                    metrics = run_agent_episode(
                        env=env,
                        agent=agent,
                        episode_index=episode_index,
                        seed=seed,
                        policy_name=policy_name,
                        policy_type=policy_type,
                        guidance_mode=guidance_mode,
                        deterministic=bool(policy_config.get("deterministic", True)),
                    )
                    metric_records.append(metrics)
                    episode_index += 1

                    if bool(benchmark_config.get("save_first_episode_plot", True)) and seed == seeds[0]:
                        raw_env = getattr(env, "env", env)
                        plot_path = output_dir / "figures" / f"{policy_name}_{guidance_mode}_seed{seed}_trajectory.png"
                        plot_full_trajectory_summary(env=raw_env, save_path=plot_path, show=False)
            else:
                raise ValueError(f"未知 benchmark policy type：{policy_type}")

    # 保存逐回合和汇总结果。
    metrics_path = save_metrics_to_csv(metric_records, output_dir / "metrics" / "benchmark_episodes.csv")
    summary_records = summarize_by_group(metric_records)
    summary_path = save_metrics_to_csv(summary_records, output_dir / "metrics" / "benchmark_summary.csv")

    if skipped_records:
        save_metrics_to_csv(skipped_records, output_dir / "metrics" / "benchmark_skipped.csv")

    report_path = write_markdown_report(
        output_path=output_dir / "benchmark_report.md",
        summary_records=summary_records,
        skipped_records=skipped_records,
    )

    # manifest：保存 benchmark 复现清单。
    manifest = build_run_manifest(
        run_name="paper_benchmark",
        project_root=project_root,
        env_config=env_config_dict,
        trainer_config=benchmark_config,
        seed=int(seeds[0]) if seeds else None,
        extra={
            "config_path": config_path,
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "skipped_records": skipped_records,
        },
    )
    save_run_manifest(manifest, output_dir / "run_manifest.json")

    logger.info("benchmark 完成：%s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行论文式一对二端到端 benchmark。")
    parser.add_argument("--config", default="configs/eval/benchmark_paper.yaml", help="benchmark YAML 配置路径。")
    args = parser.parse_args()
    main(config_path=args.config)
