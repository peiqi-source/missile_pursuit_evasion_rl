"""从 metrics.csv 重新绘制训练曲线。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hypersonic_rl.visualization.plot_training_curve import plot_training_curve


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """读取指标文件并保存训练曲线。"""
    args = parse_args()
    metrics_path = Path(args.metrics)
    save_dir = Path(args.save_dir) if args.save_dir else metrics_path.parent.parent / "figures"
    saved_paths = plot_training_curve(metrics_path, save_dir)
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
