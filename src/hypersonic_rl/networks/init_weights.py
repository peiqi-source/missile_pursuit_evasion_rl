"""
init_weights.py

作用：
    提供神经网络权重初始化函数。

说明：
    合理的权重初始化有助于提升 SAC 训练稳定性。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def initialize_linear_layer(
    layer: nn.Linear,
    weight_std: float = 0.01,
    bias_value: float = 0.0,
) -> None:
    """
    初始化线性层参数。

    参数：
        layer：
            需要初始化的 nn.Linear 层。

        weight_std：
            权重正态分布标准差。

        bias_value：
            偏置初始化常数。
    """
    # weight：线性层权重参数。
    nn.init.normal_(layer.weight, mean=0.0, std=weight_std)

    # bias：线性层偏置参数。
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_value)


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """
    对线性层进行正交初始化。

    参数：
        layer：
            待初始化的网络层。

        gain：
            正交初始化增益系数。

    说明：
        如果传入的不是 nn.Linear，则不做处理。
    """
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain=gain)

        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0.0)


def fanin_init(layer: nn.Linear) -> None:
    """
    使用 fan-in 方式初始化线性层。

    参数：
        layer：
            需要初始化的 nn.Linear 层。
    """
    # fan_in：输入维度数量。
    fan_in = layer.weight.data.size(0)

    # bound：均匀分布边界。
    bound = 1.0 / fan_in ** 0.5

    layer.weight.data.uniform_(-bound, bound)

    if layer.bias is not None:
        layer.bias.data.uniform_(-bound, bound)