"""
eval_policy.py

作用：
    统一评估普通 SAC 与第 5 章 LSTM-SAC checkpoint。

运行示例：
    $env:PYTHONPATH="$PWD\\src"
    python scripts/eval_policy.py --config configs/eval/eval_policy.yaml
    python scripts/eval_policy.py --agent-type lstm_sac --checkpoint experiments/lstm_sac_chapter5/checkpoints/latest.pth
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional


# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# src_path：保证直接运行脚本时可以导入 hypersonic_rl。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.agents import LSTMSACAgent, LSTMSACAgentConfig, SACAgent, SACAgentConfig
from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig, StateSequenceWrapper
from hypersonic_rl.evaluation import Evaluator, EvaluatorConfig
from hypersonic_rl.utils import (
    build_run_manifest,
    create_logger,
    get_device,
    load_checkpoint,
    load_config_from_project,
    save_run_manifest,
)


def filter_dataclass_config(config_dict: Dict[str, Any], dataclass_type: Any) -> Dict[str, Any]:
    """
    过滤出 dataclass 支持的配置字段。

    参数：
        config_dict：
            原始配置字典。
        dataclass_type：
            目标 dataclass 类型。

    返回：
        filtered_config：
            只包含目标 dataclass 字段的配置字典。
    """
    # valid_field_names：目标 dataclass 支持的字段名。
    valid_field_names = {field.name for field in fields(dataclass_type)}

    return {key: value for key, value in config_dict.items() if key in valid_field_names}


def resolve_project_path(path_value: str | Path) -> Path:
    """
    把配置中的路径解析为绝对路径。

    参数：
        path_value：
            绝对路径或相对项目根目录的路径。

    返回：
        resolved_path：
            绝对路径。
    """
    # path：标准化传入路径。
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def load_optional_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    读取可选 YAML 配置文件。

    参数：
        config_path：
            配置文件路径；None 时返回空字典。

    返回：
        config：
            配置字典。
    """
    if not config_path:
        return {}

    return load_config_from_project(config_path)


def build_env(env_config_dict: Dict[str, Any], agent_type: str, sequence_length: int) -> Any:
    """
    构造评估环境。

    参数：
        env_config_dict：
            PursueEscapeEnv 配置字典。
        agent_type：
            sac 或 lstm_sac。
        sequence_length：
            LSTM-SAC 状态序列长度。

    返回：
        env：
            普通环境或 StateSequenceWrapper 包装后的环境。
    """
    # env_config：过滤并构造环境配置。
    env_config = PursueEscapeEnvConfig(**filter_dataclass_config(env_config_dict, PursueEscapeEnvConfig))

    # raw_env：论文式一对二端到端环境。
    raw_env = PursueEscapeEnv(env_config)

    if agent_type == "lstm_sac":
        # LSTM-SAC：评估时自动包状态序列窗口。
        return StateSequenceWrapper(raw_env, sequence_length=sequence_length)

    return raw_env


def build_agent(
    agent_type: str,
    agent_config_dict: Dict[str, Any],
    env: Any,
    device: str,
    checkpoint_agent_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    根据 agent_type 和环境维度构造智能体。

    参数：
        agent_type：
            sac 或 lstm_sac。
        agent_config_dict：
            YAML 中的智能体配置。
        env：
            已构造好的评估环境。
        device：
            PyTorch 设备字符串。
        checkpoint_agent_config：
            checkpoint 中保存的智能体配置，可用于补齐网络结构参数。

    返回：
        agent：
            SACAgent 或 LSTMSACAgent。
    """
    # action_dim/action_low/action_high：动作空间由环境决定，保证 checkpoint 和当前环境一致。
    action_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    if agent_type == "lstm_sac":
        # observation_shape：LSTM 评估环境应为 [sequence_length, state_dim]。
        observation_shape = env.observation_space.shape
        if len(observation_shape) != 2:
            raise ValueError(f"LSTM-SAC 评估环境观测应为二维序列，实际 shape={observation_shape}")

        # config_base：优先使用 checkpoint 网络结构，再用 YAML 覆盖非结构参数。
        config_base: Dict[str, Any] = dict(checkpoint_agent_config or {})
        config_base.update(filter_dataclass_config(agent_config_dict, LSTMSACAgentConfig))
        config_base.update(
            {
                "state_dim": int(observation_shape[-1]),
                "action_dim": action_dim,
                "action_low": action_low,
                "action_high": action_high,
                "sequence_length": int(observation_shape[0]),
                "device": str(device),
            }
        )
        return LSTMSACAgent(LSTMSACAgentConfig(**filter_dataclass_config(config_base, LSTMSACAgentConfig)))

    # SAC：观测是一维论文式 10 维状态。
    config_base = dict(checkpoint_agent_config or {})
    config_base.update(filter_dataclass_config(agent_config_dict, SACAgentConfig))
    config_base.update(
        {
            "state_dim": int(env.observation_space.shape[0]),
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "device": str(device),
        }
    )
    return SACAgent(SACAgentConfig(**filter_dataclass_config(config_base, SACAgentConfig)))


def main(
    config_path: str = "configs/eval/eval_policy.yaml",
    agent_type_override: Optional[str] = None,
    checkpoint_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    base_seed_override: Optional[int] = None,
) -> None:
    """
    统一评估入口。

    参数：
        config_path：
            评估配置 YAML 路径。
        agent_type_override：
            命令行覆盖的 agent 类型。
        checkpoint_override：
            命令行覆盖的 checkpoint 路径。
        output_dir_override：
            命令行覆盖的输出目录。
        base_seed_override：
            命令行覆盖的基础 seed。
    """
    # eval_config：读取统一评估配置。
    eval_config = load_config_from_project(config_path)

    # agent_type：评估对象类型，默认 LSTM-SAC。
    agent_type = str(agent_type_override or eval_config.get("agent_type", "lstm_sac"))
    if agent_type not in {"sac", "lstm_sac"}:
        raise ValueError(f"agent_type 只支持 sac 或 lstm_sac，当前为：{agent_type}")

    # config paths：读取环境与智能体配置。
    env_config_dict = load_config_from_project(eval_config.get("env_config_path", "configs/env/pursue_escape_env.yaml"))
    default_agent_config_path = "configs/agent/lstm_sac.yaml" if agent_type == "lstm_sac" else "configs/agent/sac.yaml"
    agent_config_dict = load_config_from_project(eval_config.get("agent_config_path", default_agent_config_path))

    # checkpoint_path：模型文件路径，支持命令行覆盖。
    checkpoint_path = resolve_project_path(
        checkpoint_override
        or eval_config.get(
            "checkpoint_path",
            "experiments/lstm_sac_chapter5/checkpoints/latest.pth"
            if agent_type == "lstm_sac"
            else "experiments/sac_baseline/checkpoints/latest.pth",
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint：{checkpoint_path}")

    # output_dir：评估输出根目录。
    output_dir = resolve_project_path(
        output_dir_override
        or eval_config.get("output_dir", f"outputs/evaluation/{agent_type}")
    )

    # base_seed：多回合评估的基础随机种子。
    base_seed = int(base_seed_override if base_seed_override is not None else eval_config.get("base_seed", 2026))

    # device：评估设备。
    device = str(get_device(prefer_gpu=bool(eval_config.get("prefer_gpu", True))))

    # checkpoint：先读取 checkpoint，拿到内部 agent config 以保证网络结构匹配。
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    if "agent_state" not in checkpoint:
        raise KeyError(f"checkpoint 中没有 agent_state 字段：{checkpoint_path}")

    checkpoint_agent_config = checkpoint["agent_state"].get("config", {})

    # sequence_length：LSTM-SAC 优先使用 checkpoint 中的序列长度，再回退到 YAML。
    sequence_length = int(
        checkpoint_agent_config.get(
            "sequence_length",
            agent_config_dict.get("sequence_length", eval_config.get("sequence_length", 3)),
        )
    )

    # env/agent：构造评估环境与智能体。
    env = build_env(env_config_dict=env_config_dict, agent_type=agent_type, sequence_length=sequence_length)
    agent = build_agent(
        agent_type=agent_type,
        agent_config_dict=agent_config_dict,
        env=env,
        device=device,
        checkpoint_agent_config=checkpoint_agent_config,
    )
    agent.load_state_dict(checkpoint["agent_state"], load_optimizers=False)
    agent.eval_mode()

    # logger：评估日志。
    logger = create_logger(
        logger_name=f"eval_{agent_type}",
        log_dir=output_dir / "logs",
        log_filename="eval.log",
    )
    logger.info("已加载 checkpoint：%s", checkpoint_path)

    # evaluator_config：统一评估配置。
    evaluator_config = EvaluatorConfig(
        eval_episodes=int(eval_config.get("eval_episodes", 10)),
        deterministic=bool(eval_config.get("deterministic", True)),
        render=bool(eval_config.get("render", False)),
        save_episode_plots=bool(eval_config.get("save_episode_plots", True)),
        save_single_view_trajectory=bool(eval_config.get("save_single_view_trajectory", False)),
        save_three_view_trajectory=bool(eval_config.get("save_three_view_trajectory", True)),
        save_3d_trajectory=bool(eval_config.get("save_3d_trajectory", True)),
        save_full_trajectory_summary=bool(eval_config.get("save_full_trajectory_summary", True)),
        output_dir=str(output_dir),
    )

    # evaluator：执行评估并保存逐回合 CSV、summary CSV 和轨迹图。
    evaluator = Evaluator(env=env, agent=agent, config=evaluator_config, logger=logger)
    metric_records, summary = evaluator.evaluate(base_seed=base_seed)

    # manifest：保存评估复现清单。
    manifest = build_run_manifest(
        run_name=f"eval_{agent_type}",
        project_root=project_root,
        env_config=env_config_dict,
        agent_config=agent_config_dict,
        trainer_config=evaluator_config,
        checkpoint_path=checkpoint_path,
        seed=base_seed,
        extra={
            "agent_type": agent_type,
            "config_path": config_path,
            "sequence_length": sequence_length,
            "summary": summary,
            "num_records": len(metric_records),
        },
    )
    save_run_manifest(manifest, output_dir / "run_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一评估 SAC / LSTM-SAC checkpoint。")
    parser.add_argument("--config", default="configs/eval/eval_policy.yaml", help="评估配置 YAML 路径。")
    parser.add_argument("--agent-type", choices=["sac", "lstm_sac"], default=None, help="覆盖 agent 类型。")
    parser.add_argument("--checkpoint", default=None, help="覆盖 checkpoint 路径。")
    parser.add_argument("--output-dir", default=None, help="覆盖评估输出目录。")
    parser.add_argument("--base-seed", type=int, default=None, help="覆盖评估基础 seed。")
    args = parser.parse_args()

    main(
        config_path=args.config,
        agent_type_override=args.agent_type,
        checkpoint_override=args.checkpoint,
        output_dir_override=args.output_dir,
        base_seed_override=args.base_seed,
    )
