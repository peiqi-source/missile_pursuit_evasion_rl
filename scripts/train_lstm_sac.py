"""
train_lstm_sac.py

作用：
    论文第 5 章 LSTM-SAC 训练入口。

运行方式：
    在项目根目录执行：

        $env:PYTHONPATH="$PWD\\src"
        python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac_smoke.yaml

说明：
    普通 SAC 入口 scripts/train_sac.py 保留为基线；
    本脚本只负责第 5 章状态序列版 LSTM-SAC。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict


# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# src_path：将 src 加入 Python 搜索路径，保证直接运行脚本时可以导入 hypersonic_rl。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.agents import LSTMSACAgent, LSTMSACAgentConfig
from hypersonic_rl.buffers import SequenceReplayBuffer, SequenceReplayBufferConfig
from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig, StateSequenceWrapper
from hypersonic_rl.trainers import LSTMSACTrainer, LSTMSACTrainerConfig
from hypersonic_rl.utils import (
    create_logger,
    describe_device,
    get_device,
    load_config_from_project,
    set_global_seed,
)


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
            只保留 dataclass 字段后的配置字典。
    """
    # valid_field_names：目标 dataclass 支持的字段名。
    valid_field_names = {field.name for field in fields(dataclass_type)}

    # filtered_config：过滤掉 yaml 中仅用于脚本的额外字段。
    filtered_config = {
        key: value
        for key, value in config_dict.items()
        if key in valid_field_names
    }

    return filtered_config


def build_env(env_config_dict: Dict[str, Any], sequence_length: int) -> StateSequenceWrapper:
    """
    创建 LSTM-SAC 使用的状态序列环境。

    参数：
        env_config_dict：
            原始 PursueEscapeEnv 配置字典。
        sequence_length：
            LSTM 输入状态序列长度。

    返回：
        env：
            已包装为 [sequence_length, state_dim] 观测的环境。
    """
    # filtered_env_config：过滤出环境 dataclass 支持的字段。
    filtered_env_config = filter_dataclass_config(env_config_dict, PursueEscapeEnvConfig)

    # env_config：环境配置对象。
    env_config = PursueEscapeEnvConfig(**filtered_env_config)

    # raw_env：原始双拦截弹端到端环境。
    raw_env = PursueEscapeEnv(env_config)

    # env：状态序列包装后的 LSTM-SAC 环境。
    env = StateSequenceWrapper(raw_env, sequence_length=sequence_length)

    return env


def main(train_config_path: str = "configs/train/train_lstm_sac.yaml") -> None:
    """
    LSTM-SAC 训练主函数。

    参数：
        train_config_path：
            训练配置 YAML 路径。
    """
    # env_config_dict：双拦截弹端到端环境配置。
    env_config_dict = load_config_from_project("configs/env/pursue_escape_env.yaml")

    # agent_config_dict：LSTM-SAC 智能体配置。
    agent_config_dict = load_config_from_project("configs/agent/lstm_sac.yaml")

    # train_config_dict：训练器配置。
    train_config_dict = load_config_from_project(train_config_path)

    # train_config：过滤并构造训练器配置。
    filtered_train_config = filter_dataclass_config(train_config_dict, LSTMSACTrainerConfig)
    train_config = LSTMSACTrainerConfig(**filtered_train_config)

    # seed：设置全局随机种子。
    set_global_seed(train_config.seed)

    # device：训练设备。
    device = get_device(prefer_gpu=True)

    # sequence_length：论文第 5 章状态序列长度，默认 n=3。
    sequence_length = int(agent_config_dict.get("sequence_length", 3))

    # experiment_dir：当前实验目录。
    experiment_dir = project_root / "experiments" / train_config.experiment_name

    # logger：训练日志。
    logger = create_logger(
        logger_name="train_lstm_sac",
        log_dir=experiment_dir / "logs",
        log_filename="train.log",
    )

    logger.info("设备信息：%s", describe_device(device))
    logger.info("LSTM 状态序列长度 sequence_length=%d", sequence_length)

    # env/eval_env：训练与评估环境分离，避免评估轨迹覆盖训练轨迹。
    env = build_env(env_config_dict=env_config_dict, sequence_length=sequence_length)
    eval_env = build_env(env_config_dict=env_config_dict, sequence_length=sequence_length)

    # observation_shape：[sequence_length, state_dim]。
    observation_shape = env.observation_space.shape

    if len(observation_shape) != 2:
        raise ValueError(f"LSTM-SAC 观测应为二维序列，实际 shape={observation_shape}")

    # state_dim：单步状态维度。
    state_dim = int(observation_shape[-1])

    # action_dim/action_low/action_high：从环境动作空间读取。
    action_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    logger.info("序列观测维度 observation_shape=%s", observation_shape)
    logger.info("单步状态维度 state_dim=%d", state_dim)
    logger.info("动作维度 action_dim=%d", action_dim)
    logger.info("动作范围 action ∈ [%.3f, %.3f]", action_low, action_high)

    # agent_config_base：过滤 LSTMSACAgentConfig 支持的字段。
    agent_config_base = filter_dataclass_config(agent_config_dict, LSTMSACAgentConfig)

    # agent_config_base：补充环境决定的维度和设备。
    agent_config_base.update(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "sequence_length": sequence_length,
            "device": str(device),
        }
    )

    # agent_config：LSTM-SAC 智能体配置对象。
    agent_config = LSTMSACAgentConfig(**agent_config_base)

    # agent：LSTM-SAC 智能体。
    agent = LSTMSACAgent(agent_config)

    # replay_buffer_size：序列经验池容量，论文第 5 章默认 30000。
    replay_buffer_size = int(agent_config_dict.get("replay_buffer_size", 30000))

    # replay_buffer_config：序列经验池配置。
    replay_buffer_config = SequenceReplayBufferConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=replay_buffer_size,
        sequence_length=sequence_length,
        device=str(device),
    )

    # replay_buffer：固定窗口序列经验池。
    replay_buffer = SequenceReplayBuffer(replay_buffer_config)

    # trainer：LSTM-SAC 训练器。
    trainer = LSTMSACTrainer(
        env=env,
        eval_env=eval_env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=train_config,
        experiment_dir=experiment_dir,
        logger=logger,
    )

    # 开始训练。
    trainer.train()


if __name__ == "__main__":
    # parser：允许命令行切换 full/smoke 训练配置。
    parser = argparse.ArgumentParser(description="运行论文第 5 章 LSTM-SAC 训练入口。")
    parser.add_argument(
        "--train-config",
        default="configs/train/train_lstm_sac.yaml",
        help="训练配置 YAML 路径。",
    )
    args = parser.parse_args()
    main(train_config_path=args.train_config)
