"""神经网络权重初始化工具。"""

from __future__ import annotations

import torch.nn as nn


def init_layer(layer: nn.Module, gain: float = 1.0) -> None:
    """初始化单个线性层。

    参数:
        layer: 待初始化的 PyTorch 层。
        gain: Xavier 初始化增益。
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)


def init_mlp(module: nn.Module) -> None:
    """对 MLP 网络中的线性层执行 Xavier 初始化。"""
    for layer in module.modules():
        init_layer(layer)

