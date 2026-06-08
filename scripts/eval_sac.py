"""评估已训练 SAC checkpoint。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hypersonic_rl.evaluation.evaluator import Evaluator
from hypersonic_rl.utils.config import load_yaml
from hypersonic_rl.utils.device import get_device


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate SAC checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", default="configs/env/pursue_escape_env.yaml")
    parser.add_argument("--agent-config", default="configs/agent/sac.yaml")
    parser.add_argument("--eval-config", default="configs/eval/eval_sac.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """加载 checkpoint 并运行测试 episode。"""
    args = parse_args()
    env_config = load_yaml(PROJECT_ROOT / args.env_config)
    agent_config = load_yaml(PROJECT_ROOT / args.agent_config)
    eval_config = load_yaml(PROJECT_ROOT / args.eval_config)
    device = str(get_device())
    checkpoint_path = Path(args.checkpoint)
    evaluator = Evaluator.from_checkpoint(str(checkpoint_path), env_config, agent_config, device)
    aggregate, summaries = evaluator.evaluate(
        num_episodes=int(eval_config.get("num_episodes", 5)),
        max_steps_per_episode=int(eval_config.get("max_steps_per_episode", 5000)),
        deterministic=bool(eval_config.get("deterministic", True)),
        seed=int(eval_config.get("seed", 0)),
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.resolve().parents[1] / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(output_dir / "eval_episodes.csv", index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
