"""环境 reset/step 最小测试。"""

from __future__ import annotations

import numpy as np

from hypersonic_rl.envs.pursue_escape_env import PursueEscapeEnv


def test_env_reset_step() -> None:
    """检查环境 reset 和 step 输出符合 Gymnasium 风格。"""
    env = PursueEscapeEnv({"dt": 0.01, "max_time": 1.0})
    observation, info = env.reset(seed=0)
    assert observation.shape == env.observation_space.shape
    assert "distance" in info

    action = np.array([0.0], dtype=np.float32)
    next_observation, reward, terminated, truncated, step_info = env.step(action)
    assert next_observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_info" in step_info

