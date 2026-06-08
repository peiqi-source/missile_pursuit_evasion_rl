"""随机动作环境 rollout 调试脚本。"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hypersonic_rl.envs.pursue_escape_env import PursueEscapeEnv
from hypersonic_rl.utils.config import load_yaml
from hypersonic_rl.visualization.plot_episode import plot_episode


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run one random rollout for environment debug.")
    parser.add_argument("--env-config", default="configs/env/pursue_escape_env.yaml")
    parser.add_argument("--train-config", default="configs/train/train_sac.yaml")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """随机动作运行一轮环境并保存 episode 图。"""
    args = parse_args()
    env_config = load_yaml(PROJECT_ROOT / args.env_config)
    train_config = load_yaml(PROJECT_ROOT / args.train_config)
    env = PursueEscapeEnv(env_config)
    state, _ = env.reset(seed=args.seed)
    _ = state
    for _ in range(args.max_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / train_config.get("experiment_dir", "experiments/sac_baseline")
    figure_path = output_dir / f"debug_{timestamp}" / "figures" / "debug_env_rollout.png"
    plot_episode(env.get_episode_trace(), figure_path)
    print(f"debug figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
