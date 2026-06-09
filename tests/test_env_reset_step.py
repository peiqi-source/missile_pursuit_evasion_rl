"""
test_env_reset_step.py

作用：
    测试 PursueEscapeEnv 的 reset 和 step 是否正常工作。
"""

import numpy as np

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig


def test_env_reset_output_shape():
    """
    测试 reset() 返回的观测维度是否正确。
    """
    # config：环境配置。
    config = PursueEscapeEnvConfig(dt=0.01, t=1.0)

    # env：追逃突防环境。
    env = PursueEscapeEnv(config)

    # observation：初始观测。
    # info：初始信息字典。
    observation, info = env.reset(seed=0)

    assert observation.shape == (12,)
    assert observation.dtype == np.float32
    assert "min_distance" in info


def test_env_step_output_shape_and_type():
    """
    测试 step() 返回值类型和维度是否正确。
    """
    # config：环境配置。
    config = PursueEscapeEnvConfig(dt=0.01, t=1.0)

    # env：追逃突防环境。
    env = PursueEscapeEnv(config)

    # observation：初始观测。
    observation, info = env.reset(seed=0)

    # action：随机动作。
    action = env.action_space.sample()

    # next_observation/reward/terminated/truncated/info：执行一步环境交互。
    next_observation, reward, terminated, truncated, info = env.step(action)

    assert next_observation.shape == (12,)
    assert next_observation.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info
    assert "red_inertial_overload" in info
    assert "blue_inertial_overload" in info


def test_env_rollout_until_done_or_truncated():
    """
    测试环境能否连续运行多个 step。
    """
    # config：为了测试速度，将单回合时间设置得短一点。
    config = PursueEscapeEnvConfig(dt=0.01, t=0.2)

    # env：追逃突防环境。
    env = PursueEscapeEnv(config)

    # observation：初始观测。
    observation, info = env.reset(seed=0)

    # done：统一终止标志。
    done = False

    while not done:
        # action：随机动作。
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

    # metrics：回合统计指标。
    metrics = env.get_episode_metrics()

    assert "total_reward" in metrics
    assert "min_distance" in metrics
    assert env.current_step > 0