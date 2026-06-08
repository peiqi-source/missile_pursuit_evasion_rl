"""第 5 章 LSTM-SAC Actor 预留结构。"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMActor(nn.Module):
    """LSTM Actor 预留类。

    后续用于接收形状为 [batch, sequence_length, state_dim] 的状态序列，并输出
    tanh-squashed Gaussian 动作分布。当前阶段只保留接口，不参与普通 SAC 训练。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.mean_head = nn.Linear(lstm_hidden_dim, action_dim)
        self.log_std_head = nn.Linear(lstm_hidden_dim, action_dim)

    def forward(self, state_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回动作均值和 log_std，完整采样逻辑将在 LSTM-SAC 阶段补齐。"""
        output, _ = self.lstm(state_sequence)
        last_feature = output[:, -1, :]
        return self.mean_head(last_feature), self.log_std_head(last_feature)

