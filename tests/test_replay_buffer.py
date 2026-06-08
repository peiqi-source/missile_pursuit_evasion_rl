"""ReplayBuffer 测试。"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from hypersonic_rl.buffers.replay_buffer import ReplayBuffer


def test_replay_buffer_add_sample() -> None:
    """检查 add 和 sample 能正常工作。"""
    buffer = ReplayBuffer(state_dim=12, action_dim=1, capacity=10)
    for index in range(6):
        state = np.full(12, index, dtype=np.float32)
        action = np.array([0.1], dtype=np.float32)
        buffer.add(state, action, 1.0, state + 1.0, False)
    batch = buffer.sample(4)
    assert len(buffer) == 6
    assert batch[0].shape == (4, 12)
    assert batch[1].shape == (4, 1)
    assert batch[2].shape == (4, 1)

