"""第 5 章 LSTM-SAC Critic 预留结构。"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMCritic(nn.Module):
    """LSTM Twin Critic 预留类。

    后续用于根据状态序列和当前动作估计 Q1/Q2。当前阶段只提供可实例化结构。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        q_input_dim = lstm_hidden_dim + action_dim
        self.q1 = nn.Sequential(nn.Linear(q_input_dim, lstm_hidden_dim), nn.ReLU(), nn.Linear(lstm_hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(q_input_dim, lstm_hidden_dim), nn.ReLU(), nn.Linear(lstm_hidden_dim, 1))

    def forward(self, state_sequence: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据状态序列和动作输出 Q1/Q2。"""
        output, _ = self.lstm(state_sequence)
        last_feature = output[:, -1, :]
        critic_input = torch.cat([last_feature, action], dim=-1)
        return self.q1(critic_input), self.q2(critic_input)

