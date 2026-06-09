"""
test_training_one_step.py

作用：
    测试 SACAgent 是否能够完成一次参数更新。

测试目标：
    1. ReplayBuffer 可以提供 batch；
    2. SACAgent.update() 可以正常执行；
    3. update() 返回 critic_loss、actor_loss、alpha 等关键日志信息。
"""

import numpy as np

from hypersonic_rl.agents import SACAgent, SACAgentConfig
from hypersonic_rl.buffers import ReplayBuffer, ReplayBufferConfig


def test_sac_agent_update_one_step():
    """
    测试 SAC 智能体的一次训练更新。
    """
    # state_dim：测试状态维度。
    state_dim = 12

    # action_dim：测试动作维度。
    action_dim = 1

    # buffer_capacity：经验回放池容量。
    buffer_capacity = 200

    # batch_size：采样训练 batch 大小。
    batch_size = 32

    # agent_config：SAC 智能体配置。
    agent_config = SACAgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=-3.0,
        action_high=3.0,
        hidden_dim=64,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        automatic_entropy_tuning=True,
        device="cpu",
    )

    # agent：SAC 智能体。
    agent = SACAgent(agent_config)

    # buffer_config：经验回放池配置。
    buffer_config = ReplayBufferConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=buffer_capacity,
        device="cpu",
    )

    # replay_buffer：经验回放池。
    replay_buffer = ReplayBuffer(buffer_config)

    for index in range(100):
        # state：随机当前状态。
        state = np.random.randn(state_dim).astype(np.float32)

        # action：随机动作，范围限制在 [-3, 3]。
        action = np.random.uniform(low=-3.0, high=3.0, size=(action_dim,)).astype(np.float32)

        # reward：随机奖励。
        reward = float(np.random.randn())

        # next_state：随机下一状态。
        next_state = np.random.randn(state_dim).astype(np.float32)

        # done：随机终止标志。
        done = bool(index % 20 == 0)

        replay_buffer.add(state, action, reward, next_state, done)

    # batch：从经验池采样一个 batch。
    batch = replay_buffer.sample(batch_size=batch_size)

    # loss_info：执行一次 SAC 更新。
    loss_info = agent.update(batch)

    assert "critic_loss" in loss_info
    assert "actor_loss" in loss_info
    assert "alpha_loss" in loss_info
    assert "alpha" in loss_info

    assert isinstance(loss_info["critic_loss"], float)
    assert isinstance(loss_info["actor_loss"], float)
    assert isinstance(loss_info["alpha"], float)

    assert loss_info["alpha"] > 0.0


def test_sac_agent_select_action_shape():
    """
    测试 SAC 智能体输出动作形状是否正确。
    """
    # state_dim：测试状态维度。
    state_dim = 12

    # action_dim：测试动作维度。
    action_dim = 1

    # agent_config：SAC 智能体配置。
    agent_config = SACAgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=-3.0,
        action_high=3.0,
        hidden_dim=64,
        device="cpu",
    )

    # agent：SAC 智能体。
    agent = SACAgent(agent_config)

    # state：单个随机状态。
    state = np.random.randn(state_dim).astype(np.float32)

    # action：智能体输出动作。
    action = agent.select_action(state, deterministic=False)

    assert action.shape == (action_dim,)
    assert action[0] >= -3.0
    assert action[0] <= 3.0