"""
test_replay_buffer.py

作用：
    测试普通 ReplayBuffer 是否能够正常存储和采样。
"""

import numpy as np

from hypersonic_rl.buffers import ReplayBuffer, ReplayBufferConfig


def test_replay_buffer_add_and_sample():
    """
    测试 ReplayBuffer 的 add 和 sample 功能。
    """
    # state_dim：测试状态维度。
    state_dim = 12

    # action_dim：测试动作维度。
    action_dim = 1

    # capacity：测试经验池容量。
    capacity = 100

    # config：经验池配置对象。
    config = ReplayBufferConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=capacity,
        device="cpu",
    )

    # buffer：经验回放池对象。
    buffer = ReplayBuffer(config)

    for index in range(20):
        # state：当前状态测试数组。
        state = np.ones(state_dim, dtype=np.float32) * index

        # action：当前动作测试数组。
        action = np.array([0.1], dtype=np.float32)

        # reward：当前奖励测试值。
        reward = 1.0

        # next_state：下一状态测试数组。
        next_state = state + 1.0

        # done：终止标志。
        done = False

        buffer.add(state, action, reward, next_state, done)

    assert len(buffer) == 20

    # batch：采样得到的小批量数据。
    batch = buffer.sample(batch_size=8)

    assert batch["state"].shape == (8, state_dim)
    assert batch["action"].shape == (8, action_dim)
    assert batch["reward"].shape == (8, 1)
    assert batch["next_state"].shape == (8, state_dim)
    assert batch["done"].shape == (8, 1)