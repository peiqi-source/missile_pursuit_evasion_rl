"""SAC 网络结构测试。"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from hypersonic_rl.networks.mlp_actor import MLPActor
from hypersonic_rl.networks.mlp_critic import TwinQNetwork


def test_actor_output_shape() -> None:
    """检查 Actor 输出动作和 log_prob 维度。"""
    actor = MLPActor(12, 1, hidden_dim=32, action_scale=3.0, action_bias=0.0)
    state = torch.zeros(4, 12)
    action, log_prob, mean = actor.sample(state)
    assert action.shape == (4, 1)
    assert log_prob is not None
    assert log_prob.shape == (4, 1)
    assert mean.shape == (4, 1)


def test_critic_output_shape() -> None:
    """检查 Twin Critic 输出 Q1/Q2 维度。"""
    critic = TwinQNetwork(12, 1, hidden_dim=32)
    state = torch.zeros(4, 12)
    action = torch.zeros(4, 1)
    q1, q2 = critic(state, action)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)

