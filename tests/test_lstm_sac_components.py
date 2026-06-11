"""
test_lstm_sac_components.py

作用：
    测试论文第 5 章 LSTM-SAC 的核心组件。
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from hypersonic_rl.agents import LSTMSACAgent, LSTMSACAgentConfig
from hypersonic_rl.buffers import SequenceReplayBuffer, SequenceReplayBufferConfig
from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig, StateSequenceWrapper
from hypersonic_rl.networks import LSTMActor, LSTMTwinCritic
from hypersonic_rl.trainers import LSTMSACTrainer, LSTMSACTrainerConfig


class DummySequenceEnv:
    """
    用于测试 StateSequenceWrapper 的极简环境。
    """

    def __init__(self) -> None:
        """
        初始化测试环境。
        """
        # observation_space：单步观测为 3 维。
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)

        # action_space：动作维度为 1。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # step_count：测试用计数器。
        self.step_count = 0

    def reset(self, seed: int | None = None):
        """
        返回固定初始观测。
        """
        # step_count：每个回合重新计数。
        self.step_count = 0

        # observation：固定初始观测。
        observation = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        return observation, {}

    def step(self, action: np.ndarray):
        """
        返回随步数递增的观测。
        """
        # step_count：推进测试步数。
        self.step_count += 1

        # observation：构造便于断言的下一观测。
        observation = np.array(
            [float(self.step_count), float(self.step_count + 1), float(self.step_count + 2)],
            dtype=np.float32,
        )

        return observation, 0.0, False, False, {}


def _small_lstm_config() -> LSTMSACAgentConfig:
    """
    构造测试用的小型 LSTM-SAC 配置。

    返回：
        config：
            网络规模较小、适合单元测试快速运行的配置。
    """
    return LSTMSACAgentConfig(
        state_dim=5,
        action_dim=1,
        action_low=-2.0,
        action_high=2.0,
        sequence_length=3,
        lstm_layer_sizes=(16, 12),
        actor_fc_hidden_dim=12,
        critic_hidden_dims=(12, 8),
        gamma=0.98,
        tau=0.01,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        device="cpu",
    )


def test_state_sequence_wrapper_shape_and_rolling():
    """
    测试状态序列包装器的 reset 填充、step 滚动和 observation_space。
    """
    # wrapper：固定窗口长度为 3 的状态序列包装器。
    wrapper = StateSequenceWrapper(DummySequenceEnv(), sequence_length=3)

    assert wrapper.observation_space.shape == (3, 3)

    # sequence：reset 后三行都应为初始观测。
    sequence, _ = wrapper.reset(seed=0)

    assert sequence.shape == (3, 3)
    assert np.allclose(sequence[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(sequence[1], sequence[0])
    assert np.allclose(sequence[2], sequence[0])

    # next_sequence：step 后最后一行变成最新观测。
    next_sequence, _, _, _, _ = wrapper.step(np.array([0.0], dtype=np.float32))

    assert next_sequence.shape == (3, 3)
    assert np.allclose(next_sequence[-1], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_sequence_replay_buffer_add_sample_clear():
    """
    测试固定窗口序列经验池的写入、采样和清空。
    """
    # config：小容量序列经验池配置。
    config = SequenceReplayBufferConfig(
        state_dim=5,
        action_dim=1,
        capacity=10,
        sequence_length=3,
        device="cpu",
    )

    # buffer：序列经验池。
    buffer = SequenceReplayBuffer(config)

    for index in range(6):
        # state_sequence：当前状态序列。
        state_sequence = np.full((3, 5), fill_value=float(index), dtype=np.float32)

        # next_state_sequence：下一状态序列。
        next_state_sequence = state_sequence + 1.0

        # action：测试动作。
        action = np.array([0.1 * index], dtype=np.float32)

        buffer.add(
            state_sequence=state_sequence,
            action=action,
            reward=float(index),
            next_state_sequence=next_state_sequence,
            done=False,
        )

    assert len(buffer) == 6

    # batch：随机采样得到的训练 batch。
    batch = buffer.sample(batch_size=4)

    assert batch["state_sequence"].shape == (4, 3, 5)
    assert batch["next_state_sequence"].shape == (4, 3, 5)
    assert batch["action"].shape == (4, 1)
    assert batch["reward"].shape == (4, 1)
    assert batch["done"].shape == (4, 1)

    buffer.clear()

    assert len(buffer) == 0


def test_lstm_actor_and_critic_output_shapes():
    """
    测试 LSTM Actor/Critic 的输出形状和动作边界。
    """
    # actor：小型 LSTM 策略网络。
    actor = LSTMActor(
        state_dim=5,
        action_dim=1,
        action_low=-2.0,
        action_high=2.0,
        lstm_layer_sizes=(16, 12),
        fc_hidden_dim=12,
    )

    # critic：小型 LSTM 双 Q 网络。
    critic = LSTMTwinCritic(
        state_dim=5,
        action_dim=1,
        lstm_layer_sizes=(16, 12),
        critic_hidden_dims=(12, 8),
    )

    # state_sequence：batch 状态序列。
    state_sequence = torch.randn(4, 3, 5)

    # action/log_prob/mean_action：Actor 采样结果。
    action, log_prob, mean_action = actor.sample(state_sequence)

    assert action.shape == (4, 1)
    assert log_prob is not None
    assert log_prob.shape == (4, 1)
    assert mean_action.shape == (4, 1)
    assert torch.all(action <= 2.0 + 1e-6)
    assert torch.all(action >= -2.0 - 1e-6)

    # q1/q2：Critic 双 Q 输出。
    q1, q2 = critic(state_sequence, action)

    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)


def test_lstm_sac_agent_update_and_checkpoint_roundtrip():
    """
    测试 LSTM-SAC 智能体一次更新和状态加载。
    """
    # config：小型 agent 配置。
    config = _small_lstm_config()

    # agent：待更新的 LSTM-SAC 智能体。
    agent = LSTMSACAgent(config)

    # buffer：测试用序列经验池。
    buffer = SequenceReplayBuffer(
        SequenceReplayBufferConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            capacity=40,
            sequence_length=config.sequence_length,
            device="cpu",
        )
    )

    for _ in range(24):
        # state_sequence/next_state_sequence：随机序列样本。
        state_sequence = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)
        next_state_sequence = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)

        # action：动作范围内随机采样。
        action = np.random.uniform(-2.0, 2.0, size=(config.action_dim,)).astype(np.float32)

        buffer.add(
            state_sequence=state_sequence,
            action=action,
            reward=float(np.random.randn()),
            next_state_sequence=next_state_sequence,
            done=False,
        )

    # batch：采样一个小 batch。
    batch = buffer.sample(batch_size=8)

    # loss_info：执行一次 LSTM-SAC 更新。
    loss_info = agent.update(batch)

    assert "critic_loss" in loss_info
    assert "actor_loss" in loss_info
    assert "alpha_loss" in loss_info
    assert "alpha" in loss_info
    assert loss_info["alpha"] > 0.0

    # state_dict：导出 checkpoint 状态。
    state_dict = agent.state_dict()

    # loaded_agent：重新创建并加载参数。
    loaded_agent = LSTMSACAgent(config)
    loaded_agent.load_state_dict(state_dict, load_optimizers=False)

    # test_sequence：用于 deterministic 动作一致性检查。
    test_sequence = np.random.randn(config.sequence_length, config.state_dim).astype(np.float32)

    original_action = agent.select_action(test_sequence, deterministic=True)
    loaded_action = loaded_agent.select_action(test_sequence, deterministic=True)

    assert np.allclose(original_action, loaded_action, atol=1e-5)


def test_lstm_sac_trainer_smoke(tmp_path):
    """
    测试 LSTM-SAC 训练器极短 smoke，确认输出文件链路可用。
    """
    # env_config：关闭随机化，缩短环境最大时长。
    env_config = PursueEscapeEnvConfig(
        initial_randomization_enabled=False,
        t=1.0,
    )

    # env/eval_env：状态序列包装后的训练和评估环境。
    env = StateSequenceWrapper(PursueEscapeEnv(env_config), sequence_length=3)
    eval_env = StateSequenceWrapper(PursueEscapeEnv(env_config), sequence_length=3)

    # config：小型 LSTM-SAC agent 配置。
    config = LSTMSACAgentConfig(
        state_dim=10,
        action_dim=1,
        action_low=float(env.action_space.low[0]),
        action_high=float(env.action_space.high[0]),
        sequence_length=3,
        lstm_layer_sizes=(8,),
        actor_fc_hidden_dim=8,
        critic_hidden_dims=(8,),
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        device="cpu",
    )

    # agent：测试用 LSTM-SAC 智能体。
    agent = LSTMSACAgent(config)

    # buffer：测试用序列经验池。
    buffer = SequenceReplayBuffer(
        SequenceReplayBufferConfig(
            state_dim=10,
            action_dim=1,
            capacity=50,
            sequence_length=3,
            device="cpu",
        )
    )

    # trainer_config：极短训练配置，关闭评估避免测试耗时过长。
    trainer_config = LSTMSACTrainerConfig(
        total_episodes=1,
        max_steps_per_episode=5,
        start_steps=1,
        update_after=1,
        update_every=1,
        updates_per_step=1,
        batch_size=2,
        eval_interval=0,
        save_interval=1,
        log_interval=1,
        plot_interval=1,
        seed=0,
        experiment_name="pytest_lstm_sac_smoke",
    )

    # trainer：LSTM-SAC 训练器。
    trainer = LSTMSACTrainer(
        env=env,
        eval_env=eval_env,
        agent=agent,
        replay_buffer=buffer,
        config=trainer_config,
        experiment_dir=tmp_path,
        logger=None,
    )

    # records：执行极短训练。
    records = trainer.train()

    assert len(records) == 1
    assert (tmp_path / "checkpoints" / "final.pth").exists()
    assert (tmp_path / "metrics" / "training_metrics.csv").exists()
