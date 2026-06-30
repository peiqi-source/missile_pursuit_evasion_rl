from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 snapshots.csv 路径")
    parser.add_argument("--output-dir", required=True, help="输出 split 目录")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # 随机打乱，保证 train/val/test 分布接近
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    train_df.to_csv(output_dir / "train_snapshots.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(output_dir / "val_snapshots.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "test_snapshots.csv", index=False, encoding="utf-8-sig")

    print(f"总数: {n}")
    print(f"train: {len(train_df)} -> {output_dir / 'train_snapshots.csv'}")
    print(f"val:   {len(val_df)} -> {output_dir / 'val_snapshots.csv'}")
    print(f"test:  {len(test_df)} -> {output_dir / 'test_snapshots.csv'}")


if __name__ == "__main__":
    main()