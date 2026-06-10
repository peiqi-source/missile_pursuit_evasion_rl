"""
train_two_interceptor_sac.py

作用：
    使用双拦截弹端到端 SAC smoke 配置启动训练链路。

运行方式：
    在项目根目录执行：
        $env:PYTHONPATH="$PWD\\src"
        python scripts/train_two_interceptor_sac.py
"""

from __future__ import annotations

from train_sac import main as run_sac_training


def main() -> None:
    """
    启动双拦截弹端到端 SAC smoke 训练。

    返回：
        None。
    """
    # train_config_path：双弹 smoke 配置，默认只跑短回合验证链路。
    train_config_path = "configs/train/train_two_interceptor_sac.yaml"

    run_sac_training(train_config_path=train_config_path)


if __name__ == "__main__":
    main()
