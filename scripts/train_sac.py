"""启动普通 SAC 训练。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hypersonic_rl.trainers.sac_trainer import SACTrainer
from hypersonic_rl.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train SAC for pursuit-evasion environment.")
    parser.add_argument("--env-config", default="configs/env/pursue_escape_env.yaml")
    parser.add_argument("--agent-config", default="configs/agent/sac.yaml")
    parser.add_argument("--train-config", default="configs/train/train_sac.yaml")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--total-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """读取配置并启动训练器。"""
    args = parse_args()
    env_config = load_yaml(PROJECT_ROOT / args.env_config)
    agent_config = load_yaml(PROJECT_ROOT / args.agent_config)
    train_config = load_yaml(PROJECT_ROOT / args.train_config)
    if args.experiment_dir is not None:
        train_config["experiment_dir"] = args.experiment_dir
    if args.total_episodes is not None:
        train_config["total_episodes"] = args.total_episodes
    if args.max_steps_per_episode is not None:
        train_config["max_steps_per_episode"] = args.max_steps_per_episode
    trainer = SACTrainer(env_config, agent_config, train_config)
    trainer.train()


if __name__ == "__main__":
    main()
