"""SACAgent 单次更新测试。"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from hypersonic_rl.agents.sac_agent import SACAgent
from hypersonic_rl.buffers.replay_buffer import ReplayBuffer


def test_sac_update_one_step() -> None:
    """构造少量 transition 并检查 update_parameters 可完成一次更新。"""
    config = {
        "hidden_dim": 32,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "batch_size": 4,
        "automatic_entropy_tuning": True,
        "target_entropy": "auto",
    }
    agent = SACAgent(
        state_dim=12,
        action_dim=1,
        action_low=np.array([-3.0], dtype=np.float32),
        action_high=np.array([3.0], dtype=np.float32),
        config=config,
        device="cpu",
    )
    buffer = ReplayBuffer(state_dim=12, action_dim=1, capacity=20)
    rng = np.random.default_rng(0)
    for _ in range(8):
        state = rng.normal(size=12).astype(np.float32)
        action = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
        reward = float(rng.normal())
        next_state = rng.normal(size=12).astype(np.float32)
        buffer.add(state, action, reward, next_state, False)
    losses = agent.update_parameters(buffer, batch_size=4)
    assert {"actor_loss", "critic_loss", "alpha_loss", "alpha"} <= set(losses)

