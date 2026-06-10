"""
test_sac_networks.py

作用：
    测试 SAC Actor 和 Critic 网络输入输出维度是否正确。
"""

import torch

from hypersonic_rl.networks import MLPActor, MLPTwinCritic


def test_mlp_actor_output_shape():
    """
    测试 Actor 输出动作和 log_prob 的形状。
    """
    # state_dim：测试状态维度。
    state_dim = 10

    # action_dim：测试动作维度。
    action_dim = 1

    # hidden_dim：测试隐藏层维度。
    hidden_dim = 256

    # batch_size：测试 batch 样本数。
    batch_size = 16

    # actor：MLP Actor 网络。
    actor = MLPActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        action_low=-3.0,
        action_high=3.0,
    )

    # state：随机测试状态。
    state = torch.randn(batch_size, state_dim)

    # action：采样动作。
    action, log_prob, mean_action = actor.sample(state)

    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size, 1)
    assert mean_action.shape == (batch_size, action_dim)


def test_mlp_critic_output_shape():
    """
    测试双 Q Critic 输出形状。
    """
    # state_dim：测试状态维度。
    state_dim = 10

    # action_dim：测试动作维度。
    action_dim = 1

    # hidden_dim：测试隐藏层维度。
    hidden_dim = 256

    # batch_size：测试 batch 样本数。
    batch_size = 16

    # critic：MLP 双 Q Critic 网络。
    critic = MLPTwinCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    # state：随机测试状态。
    state = torch.randn(batch_size, state_dim)

    # action：随机测试动作。
    action = torch.randn(batch_size, action_dim)

    # q1_value/q2_value：两个 Q 网络输出。
    q1_value, q2_value = critic(state, action)

    assert q1_value.shape == (batch_size, 1)
    assert q2_value.shape == (batch_size, 1)
