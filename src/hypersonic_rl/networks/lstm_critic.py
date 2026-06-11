"""
lstm_critic.py

作用：
    实现论文第 5 章 LSTM-SAC 使用的双 Q Critic 网络。

网络输入：
    state_sequence，形状为 [batch_size, sequence_length, state_dim]；
    action，形状为 [batch_size, action_dim]。

网络输出：
    Q1(state_sequence, action)、Q2(state_sequence, action)。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMQNetwork(nn.Module):
    """
    单个 LSTM Q 网络。

    说明：
        先用单向 LSTM 编码连续状态序列，再将最后时刻的时序特征与当前动作拼接，
        经过多层全连接网络输出一个动作价值。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_layer_sizes: Sequence[int] = (256, 256, 128),
        critic_hidden_dims: Sequence[int] = (128, 128, 64),
    ) -> None:
        """
        初始化单个 LSTM Q 网络。

        参数：
            state_dim：
                单个状态向量维度。
            action_dim：
                动作向量维度。
            lstm_layer_sizes：
                状态序列编码器的 LSTM 层宽度。
            critic_hidden_dims：
                拼接动作后的 Q 网络全连接层宽度。
        """
        super().__init__()

        # state_dim/action_dim：输入维度。
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        # lstm_layer_sizes：论文式 LSTM 编码器宽度。
        self.lstm_layer_sizes = tuple(int(size) for size in lstm_layer_sizes)

        # critic_hidden_dims：Q 值头部全连接层宽度。
        self.critic_hidden_dims = tuple(int(size) for size in critic_hidden_dims)

        if self.state_dim <= 0:
            raise ValueError(f"state_dim 必须大于 0，当前为：{self.state_dim}")

        if self.action_dim <= 0:
            raise ValueError(f"action_dim 必须大于 0，当前为：{self.action_dim}")

        if not self.lstm_layer_sizes or any(size <= 0 for size in self.lstm_layer_sizes):
            raise ValueError(f"lstm_layer_sizes 必须为正整数序列，当前为：{self.lstm_layer_sizes}")

        if not self.critic_hidden_dims or any(size <= 0 for size in self.critic_hidden_dims):
            raise ValueError(f"critic_hidden_dims 必须为正整数序列，当前为：{self.critic_hidden_dims}")

        # recurrent_layers：逐层保存单向 LSTM，支持 256 -> 256 -> 128 这种变宽结构。
        self.recurrent_layers = nn.ModuleList()

        # input_dim：当前 LSTM 层输入维度。
        input_dim = self.state_dim

        for hidden_dim in self.lstm_layer_sizes:
            # lstm_layer：单层单向 LSTM，输入为 [B, L, input_dim]。
            lstm_layer = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.recurrent_layers.append(lstm_layer)
            input_dim = hidden_dim

        # q_layers：动作拼接后的多层感知机。
        self.q_layers = nn.ModuleList()

        # q_input_dim：LSTM 最后时刻特征与当前动作拼接后的维度。
        q_input_dim = self.lstm_layer_sizes[-1] + self.action_dim

        for hidden_dim in self.critic_hidden_dims:
            # q_layer：Q 网络的一层全连接特征提取层。
            self.q_layers.append(nn.Linear(q_input_dim, hidden_dim))
            q_input_dim = hidden_dim

        # q_output_layer：输出标量 Q 值。
        self.q_output_layer = nn.Linear(q_input_dim, 1)

    def _encode_sequence(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        将状态序列编码为最后时刻的时序特征。

        参数：
            state_sequence：
                状态序列张量，形状为 [batch_size, sequence_length, state_dim]。

        返回：
            encoded_feature：
                最后时刻的 LSTM 特征。
        """
        if state_sequence.ndim != 3:
            raise ValueError(
                "state_sequence 必须为三维张量 "
                f"[batch_size, sequence_length, state_dim]，实际为：{tuple(state_sequence.shape)}"
            )

        if state_sequence.shape[-1] != self.state_dim:
            raise ValueError(
                f"state_sequence 最后一维应为 {self.state_dim}，实际为 {state_sequence.shape[-1]}"
            )

        # hidden_sequence：逐层传递的 LSTM 序列特征。
        hidden_sequence = state_sequence

        for recurrent_layer in self.recurrent_layers:
            # hidden_sequence：当前 LSTM 层输出的完整时间序列。
            hidden_sequence, _ = recurrent_layer(hidden_sequence)

            # ReLU：对齐论文表 5-1 隐藏层激活函数。
            hidden_sequence = F.relu(hidden_sequence)

        # encoded_feature：取当前决策时刻对应的最后一个序列特征。
        encoded_feature = hidden_sequence[:, -1, :]

        return encoded_feature

    def forward(self, state_sequence: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算单个 Q 网络的动作价值。

        参数：
            state_sequence：
                状态序列张量，形状为 [batch_size, sequence_length, state_dim]。
            action：
                动作张量，形状为 [batch_size, action_dim]。

        返回：
            q_value：
                Q 值张量，形状为 [batch_size, 1]。
        """
        if action.ndim != 2:
            raise ValueError(f"action 必须为二维张量 [batch_size, action_dim]，实际为：{tuple(action.shape)}")

        if action.shape[-1] != self.action_dim:
            raise ValueError(f"action 最后一维应为 {self.action_dim}，实际为 {action.shape[-1]}")

        # sequence_feature：LSTM 编码出的历史态势特征。
        sequence_feature = self._encode_sequence(state_sequence)

        # q_feature：状态序列特征与当前动作拼接。
        q_feature = torch.cat([sequence_feature, action], dim=-1)

        for q_layer in self.q_layers:
            # q_feature：逐层提取状态-动作价值特征。
            q_feature = F.relu(q_layer(q_feature))

        # q_value：标量动作价值。
        q_value = self.q_output_layer(q_feature)

        return q_value


class LSTMTwinCritic(nn.Module):
    """
    LSTM-SAC 双 Q Critic 网络。

    作用：
        同时估计 Q1 和 Q2，训练时取较小值降低过估计风险。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_layer_sizes: Sequence[int] = (256, 256, 128),
        critic_hidden_dims: Sequence[int] = (128, 128, 64),
    ) -> None:
        """
        初始化双 Q 网络。

        参数：
            state_dim：
                单个状态向量维度。
            action_dim：
                动作向量维度。
            lstm_layer_sizes：
                LSTM 编码器层宽。
            critic_hidden_dims：
                Q 网络全连接层宽。
        """
        super().__init__()

        # q1_network/q2_network：两个互相独立的 Q 网络。
        self.q1_network = LSTMQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_layer_sizes=lstm_layer_sizes,
            critic_hidden_dims=critic_hidden_dims,
        )
        self.q2_network = LSTMQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_layer_sizes=lstm_layer_sizes,
            critic_hidden_dims=critic_hidden_dims,
        )

    def forward(self, state_sequence: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时计算 Q1 和 Q2。

        参数：
            state_sequence：
                状态序列张量。
            action：
                动作张量。

        返回：
            q1_value/q2_value：
                两个 Q 网络的估计值。
        """
        # q1_value：第一个 Q 网络估计。
        q1_value = self.q1_network(state_sequence, action)

        # q2_value：第二个 Q 网络估计。
        q2_value = self.q2_network(state_sequence, action)

        return q1_value, q2_value

    def q1(self, state_sequence: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        只计算 Q1。

        参数：
            state_sequence：
                状态序列张量。
            action：
                动作张量。

        返回：
            q1_value：
                第一个 Q 网络估计值。
        """
        return self.q1_network(state_sequence, action)


class LSTMCritic(LSTMTwinCritic):
    """
    LSTM Critic 兼容命名。

    说明：
        项目早期类名为 LSTMCritic；当前实际实现为双 Q Critic。
        因此该类直接继承 LSTMTwinCritic，供已有导入路径继续使用。
    """
