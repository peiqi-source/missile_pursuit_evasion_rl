"""
mlp_critic.py

作用：
    实现 SAC 使用的双 Q Critic 网络。

SAC 为什么使用双 Q 网络：
    单个 Q 网络容易高估动作价值。
    双 Q 网络会同时估计 Q1(s,a) 和 Q2(s,a)，训练 Actor 时取较小值，
    可以缓解价值过估计问题，提高训练稳定性。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    单个 Q 网络。

    输入：
        state 和 action 拼接后的向量。

    输出：
        Q(s, a)，表示在状态 s 下执行动作 a 的价值。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        """
        初始化单 Q 网络。

        参数：
            state_dim：
                状态向量维度。

            action_dim：
                动作向量维度。

            hidden_dim：
                隐藏层神经元数量。
        """
        super().__init__()

        # input_dim：Q 网络输入维度，由状态维度和动作维度相加得到。
        input_dim = int(state_dim) + int(action_dim)

        # hidden_dim：隐藏层宽度。
        hidden_dim = int(hidden_dim)

        if input_dim <= 0:
            raise ValueError(f"input_dim 必须大于 0，当前为：{input_dim}")

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须大于 0，当前为：{hidden_dim}")

        # layer_1：第一层特征提取网络。
        self.layer_1 = nn.Linear(input_dim, hidden_dim)

        # layer_2：第二层特征提取网络。
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)

        # q_layer：输出 Q 值的线性层。
        self.q_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Q 值。

        参数：
            state：
                状态张量，形状为 [batch_size, state_dim]。

            action：
                动作张量，形状为 [batch_size, action_dim]。

        返回：
            q_value：
                Q 值张量，形状为 [batch_size, 1]。
        """
        # state_action：状态和动作在最后一个维度拼接后的输入。
        state_action = torch.cat([state, action], dim=-1)

        # hidden：第一层隐藏特征。
        hidden = F.relu(self.layer_1(state_action))

        # hidden：第二层隐藏特征。
        hidden = F.relu(self.layer_2(hidden))

        # q_value：当前 Q 网络输出的动作价值。
        q_value = self.q_layer(hidden)

        return q_value


class MLPTwinCritic(nn.Module):
    """
    SAC 双 Q Critic 网络。

    包含：
        q1_network：
            第一个 Q 网络。

        q2_network：
            第二个 Q 网络。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        """
        初始化双 Q 网络。

        参数：
            state_dim：
                状态向量维度。

            action_dim：
                动作向量维度。

            hidden_dim：
                隐藏层神经元数量。
        """
        super().__init__()

        # q1_network：第一个 Q 网络。
        self.q1_network = QNetwork(state_dim, action_dim, hidden_dim)

        # q2_network：第二个 Q 网络。
        self.q2_network = QNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时计算 Q1 和 Q2。

        参数：
            state：
                状态张量。

            action：
                动作张量。

        返回：
            q1_value：
                第一个 Q 网络输出。

            q2_value：
                第二个 Q 网络输出。
        """
        # q1_value：第一个 Q 网络估计值。
        q1_value = self.q1_network(state, action)

        # q2_value：第二个 Q 网络估计值。
        q2_value = self.q2_network(state, action)

        return q1_value, q2_value

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        只计算 Q1。

        用途：
            一些算法在策略评估或日志记录时只需要一个 Q 值。
        """
        return self.q1_network(state, action)