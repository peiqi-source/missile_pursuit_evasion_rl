"""SAC 使用的双 Q Critic 网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypersonic_rl.networks.init_weights import init_mlp


class TwinQNetwork(nn.Module):
    """双 Q 网络，用两个独立分支缓解 Q 值过估计。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        self.q2_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        init_mlp(self)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据状态和动作输出 Q1、Q2。"""
        # state_action：Critic 输入，由状态和动作拼接而成。
        state_action = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_fc1(state_action))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(state_action))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2

