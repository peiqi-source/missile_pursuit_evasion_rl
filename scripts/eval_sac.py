"""
eval_sac.py

作用：
    加载训练好的 SAC 模型并进行评估。

运行方式：
    在项目根目录执行：

        $env:PYTHONPATH="$PWD\\src"
        python scripts/eval_sac.py

默认加载：
    experiments/sac_baseline/checkpoints/latest.pth
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# 将 src 加入 Python 搜索路径。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.agents import SACAgent, SACAgentConfig
from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.evaluation import Evaluator, EvaluatorConfig
from hypersonic_rl.utils import create_logger, get_device, load_checkpoint, load_config_from_project


def filter_dataclass_config(config_dict: Dict[str, Any], dataclass_type: Any) -> Dict[str, Any]:
    """
    从配置字典中过滤出 dataclass 支持的字段。
    """
    valid_field_names = {field.name for field in fields(dataclass_type)}

    return {
        key: value
        for key, value in config_dict.items()
        if key in valid_field_names
    }


def build_env(env_config_dict: Dict[str, Any]) -> PursueEscapeEnv:
    """
    根据配置字典创建环境。
    """
    filtered_env_config = filter_dataclass_config(env_config_dict, PursueEscapeEnvConfig)
    env_config = PursueEscapeEnvConfig(**filtered_env_config)
    env = PursueEscapeEnv(env_config)
    return env


def main() -> None:
    """
    SAC 评估主函数。
    """
    env_config_dict = load_config_from_project("configs/env/pursue_escape_env.yaml")
    agent_config_dict = load_config_from_project("configs/agent/sac.yaml")
    eval_config_dict = load_config_from_project("configs/eval/eval_sac.yaml")
    train_config_dict = load_config_from_project("configs/train/train_sac.yaml")

    # experiment_name：实验名称。
    experiment_name = train_config_dict.get("experiment_name", "sac_baseline")

    # checkpoint_path：待加载 checkpoint 路径。
    checkpoint_path = project_root / eval_config_dict.get(
        "checkpoint_path",
        f"experiments/{experiment_name}/checkpoints/latest.pth",
    )

    # device：评估设备。
    device = get_device(prefer_gpu=True)

    # logger：评估日志。
    logger = create_logger(
        logger_name="eval_sac",
        log_dir=project_root / "experiments" / experiment_name / "logs",
        log_filename="eval.log",
    )

    # env：评估环境。
    env = build_env(env_config_dict)

    # 状态和动作维度。
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    # agent_config_base：SAC 配置。
    agent_config_base = filter_dataclass_config(agent_config_dict, SACAgentConfig)
    agent_config_base.update(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "device": str(device),
        }
    )

    agent_config = SACAgentConfig(**agent_config_base)

    # agent：SAC 智能体。
    agent = SACAgent(agent_config)

    # checkpoint：加载模型检查点。
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    if "agent_state" not in checkpoint:
        raise KeyError(f"checkpoint 中没有 agent_state 字段：{checkpoint_path}")

    agent.load_state_dict(checkpoint["agent_state"], load_optimizers=False)
    agent.eval_mode()

    logger.info("已加载 checkpoint：%s", checkpoint_path)

    # evaluator_config：评估器配置。
    evaluator_config = EvaluatorConfig(
        eval_episodes=int(eval_config_dict.get("eval_episodes", 10)),
        deterministic=True,
        render=bool(eval_config_dict.get("render", False)),
        save_episode_plots=True,
        save_single_view_trajectory=False,
        save_three_view_trajectory=True,
        save_3d_trajectory=True,
        save_full_trajectory_summary=True,
        output_dir=str(project_root / "experiments" / experiment_name / "final_evaluation"),
    )

    evaluator = Evaluator(
        env=env,
        agent=agent,
        config=evaluator_config,
        logger=logger,
    )

    evaluator.evaluate(base_seed=2026)


if __name__ == "__main__":
    main()