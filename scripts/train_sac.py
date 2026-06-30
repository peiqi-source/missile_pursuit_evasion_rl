"""
train_sac.py

作用：
    SAC 训练入口脚本。

运行方式：
    在项目根目录执行：

        $env:PYTHONPATH="$PWD\\src"
        python scripts/train_sac.py

输出内容：
    experiments/sac_baseline/checkpoints/
        latest.pth
        final.pth
        episode_xxxx.pth

    experiments/sac_baseline/metrics/
        training_metrics.csv
        loss_metrics.csv

    experiments/sac_baseline/figures/
        reward_curve.png
        min_distance_curve.png
        success_rate_curve.png
        critic_loss_curve.png
        actor_loss_curve.png
        alpha_curve.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# 将 src 加入 Python 搜索路径，保证直接运行脚本时可以导入 hypersonic_rl。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.agents import SACAgent, SACAgentConfig
from hypersonic_rl.buffers import ReplayBuffer, ReplayBufferConfig
from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.trainers import SACTrainer, SACTrainerConfig
from hypersonic_rl.utils import (
    TrainingStartConfig,
    apply_training_start_mode,
    build_dataclass_config,
    create_logger,
    describe_device,
    get_device,
    load_config_from_project,
    load_config_stack_from_project,
    set_global_seed,
)


AGENT_CONFIG_EXTRA_KEYS = {"replay_buffer_size"}
TRAINING_START_CONFIG_KEYS = set(TrainingStartConfig.__dataclass_fields__.keys())

TRAIN_CONFIG_EXTRA_KEYS = {
    "env_config_path",
    "eval_env_config_path",
    "agent_config_path",
} | TRAINING_START_CONFIG_KEYS

def build_env(env_config_dict: Dict[str, Any]) -> PursueEscapeEnv:
    """
    根据配置字典创建环境。

    参数：
        env_config_dict：
            环境配置字典。

    返回：
        env：
            PursueEscapeEnv 环境对象。
    """
    # env_config：环境配置对象。
    env_config = build_dataclass_config(env_config_dict, PursueEscapeEnvConfig)

    # env：环境对象。
    env = PursueEscapeEnv(env_config)

    return env


def main(train_config_path: str = "configs/train/train_sac.yaml") -> None:
    """
    SAC 训练主函数。
    """

    train_config_dict = load_config_from_project(train_config_path)

    # env_config_path：
    #     训练环境配置，可以是单个 YAML，也可以是多个 YAML 的列表。
    env_config_path = train_config_dict.get("env_config_path")

    # eval_env_config_path：
    #     评估环境配置。
    #     如果没有单独指定，则默认使用训练环境配置。
    eval_env_config_path = train_config_dict.get("eval_env_config_path")

    # agent_config_path：
    #     SAC agent 配置，也保留支持单文件或配置栈。
    agent_config_path = train_config_dict.get("agent_config_path")

    # 读取并合并配置文件。
    env_config_dict = load_config_stack_from_project(env_config_path)
    eval_env_config_dict = load_config_stack_from_project(eval_env_config_path)
    agent_config_dict = load_config_stack_from_project(agent_config_path)

    # train_config：训练器配置。
    train_config = build_dataclass_config(
        train_config_dict,
        SACTrainerConfig,
        allow_extra_keys=TRAIN_CONFIG_EXTRA_KEYS,
    )

    # 设置随机种子。
    set_global_seed(train_config.seed)

    # device：训练设备。
    device = get_device(prefer_gpu=True)

    # experiment_dir：当前实验目录。
    experiment_dir = project_root / "experiments" / train_config.experiment_name

    # logger：训练日志。
    logger = create_logger(
        logger_name="train_sac",
        log_dir=experiment_dir / "logs",
        log_filename="train.log",
    )

    logger.info("设备信息：%s", describe_device(device))

    # env：训练环境。
    env = build_env(env_config_dict)

    # eval_env：评估环境。
    # 注意：这里使用 eval_env_config_dict，而不是 env_config_dict。
    eval_env = build_env(eval_env_config_dict)

    # 从环境读取状态和动作维度。
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    logger.info("状态维度 state_dim=%d", state_dim)
    logger.info("动作维度 action_dim=%d", action_dim)
    logger.info("动作范围 action ∈ [%.3f, %.3f]", action_low, action_high)

    logger.info("训练环境配置：")
    logger.info("  scenario_profile=%s", env.config.scenario_profile)
    logger.info("  radar_snapshot_csv_path=%s", getattr(env.config, "radar_snapshot_csv_path", ""))
    logger.info("  radar_snapshot_sampling_mode=%s", getattr(env.config, "radar_snapshot_sampling_mode", ""))
    logger.info("  red_intelligent_activation_mode=%s", env.config.red_intelligent_activation_mode)
    logger.info("  t=%.3f, dt=%.3f", env.config.t, env.config.dt)

    logger.info("评估环境配置：")
    logger.info("  scenario_profile=%s", eval_env.config.scenario_profile)
    logger.info("  radar_snapshot_csv_path=%s", getattr(eval_env.config, "radar_snapshot_csv_path", ""))
    logger.info("  radar_snapshot_sampling_mode=%s", getattr(eval_env.config, "radar_snapshot_sampling_mode", ""))
    logger.info("  red_intelligent_activation_mode=%s", eval_env.config.red_intelligent_activation_mode)
    logger.info("  t=%.3f, dt=%.3f", eval_env.config.t, eval_env.config.dt)

    # agent_config：SACAgent 配置对象。
    agent_config = build_dataclass_config(
        agent_config_dict,
        SACAgentConfig,
        allow_extra_keys=AGENT_CONFIG_EXTRA_KEYS,
        overrides={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "device": str(device),
        },
    )

    # agent：SAC 智能体。
    agent = SACAgent(agent_config)

    # training_start_config：
    #     控制从零训练 / actor fine-tune / actor+critic fine-tune / network resume。
    training_start_config = build_dataclass_config(
        train_config_dict,
        TrainingStartConfig,
        allow_extra_keys=TRAIN_CONFIG_EXTRA_KEYS,
    )

    apply_training_start_mode(
        agent=agent,
        config=training_start_config,
        project_root=project_root,
        device=device,
        logger=logger,
    )

    # replay_buffer_size：经验池容量。
    replay_buffer_size = int(agent_config_dict.get("replay_buffer_size", 100000))

    # replay_buffer_config：经验回放池配置。
    replay_buffer_config = ReplayBufferConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=replay_buffer_size,
        device=str(device),
    )

    # replay_buffer：经验回放池。
    replay_buffer = ReplayBuffer(replay_buffer_config)

    # trainer：SAC 训练器。
    trainer = SACTrainer(
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
    # parser：允许命令行指定训练配置，便于复用同一套训练入口。
    parser = argparse.ArgumentParser(description="运行 SAC 训练入口。")
    parser.add_argument(
        "--train-config",
        default="configs/train/train_sac.yaml",
        help="训练配置 YAML 路径。",
    )
    args = parser.parse_args()
    main(train_config_path=args.train_config)
