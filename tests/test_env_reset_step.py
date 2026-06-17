"""
test_env_reset_step.py

作用：
    验证双拦截弹端到端环境的 reset、step 和短回合滚动是否保持稳定。
"""

import numpy as np

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig


def _short_env_config(**kwargs) -> PursueEscapeEnvConfig:
    """
    构造测试用的短回合双弹环境配置。

    参数：
        **kwargs：
            需要覆盖的环境配置字段。

    返回：
        config：
            PursueEscapeEnvConfig 配置对象。
    """
    # config_kwargs：测试默认使用论文 200 km 双弹态势，并关闭随机化方便断言。
    config_kwargs = {
        "dt": 0.01,
        "t": 1.0,
        "interceptor_count": 2,
        "scenario_profile": "paper_200km_end_to_end",
        "observation_mode": "thesis_end_to_end_10d",
        "initial_randomization_enabled": False,
    }
    config_kwargs.update(kwargs)
    return PursueEscapeEnvConfig(**config_kwargs)


def test_env_reset_output_shape():
    """
    验证 reset() 返回论文第 5 章端到端 10 维观测。
    """
    # env：双拦截弹追逃突防环境。
    env = PursueEscapeEnv(_short_env_config())

    # observation/info：初始观测与初始诊断信息。
    observation, info = env.reset(seed=0)

    assert observation.shape == (10,)
    assert observation.dtype == np.float32
    assert env.action_space.shape == (1,)
    assert info["interceptor_count"] == 2
    assert "interceptor_1_min_distance" in info
    assert "interceptor_2_min_distance" in info
    assert "min_distance" in info


def test_env_step_output_shape_and_type():
    """
    验证 step() 返回值类型、维度和 per-interceptor 诊断字段。
    """
    # env：双拦截弹追逃突防环境。
    env = PursueEscapeEnv(_short_env_config())
    env.reset(seed=0)

    # action：红方 1 维横向过载动作。
    action = env.action_space.sample()

    # next_observation/reward/terminated/truncated/info：执行一步后的环境交互结果。
    next_observation, reward, terminated, truncated, info = env.step(action)

    assert next_observation.shape == (10,)
    assert next_observation.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info
    assert "red_inertial_overload" in info
    assert "threat_interceptor_id" in info
    assert "interceptor_1_ny_actual" in info
    assert "interceptor_1_nz_actual" in info
    assert "interceptor_2_ny_actual" in info
    assert "interceptor_2_nz_actual" in info


def test_paper_mid_terminal_defaults_match_thesis_table():
    """
    验证 paper_mid_terminal 默认中制导参数与论文表 2-5 保持一致。
    """
    config = PursueEscapeEnvConfig()

    assert np.isclose(config.paper_midcourse_navigation_gain, 4.0)
    assert np.isclose(config.paper_midcourse_theta_bias_gain, 0.07)
    assert np.isclose(config.paper_midcourse_psi_bias_gain, 0.09)
    assert np.isclose(config.paper_midcourse_shaping_angle_deg, 20.0)
    assert np.isclose(config.paper_terminal_navigation_gain, 6.0)


def test_paper_profile_initial_positions_and_heading_are_deterministic():
    """
    验证固定 seed 且关闭随机化时，论文 200 km 双弹初始态势稳定。
    """
    # env：论文 200 km 双弹 profile，关闭表 5-2 随机扰动。
    env = PursueEscapeEnv(_short_env_config())

    # info：reset 返回的初始诊断字段。
    _, info = env.reset(seed=0)

    # red_state：红方位于 (0, 25km, 0)，航向角为 0。
    assert np.allclose(env.red_state[:3], np.array([0.0, 25000.0, 0.0]))
    assert np.isclose(env.red_state[5], 0.0)

    # interceptor_y：paper profile 使用配置中的高度偏置，而不是写死同高度。
    interceptor_y = env.config.red_initial_y + env.config.paper_interceptor_y_offset_from_red

    # interceptor_states：两枚拦截弹位于红方前方 200 km，侧向分别为 -10 km / +10 km。
    assert np.allclose(env.interceptor_states[0][:3], np.array([200000.0, interceptor_y, -10000.0]))
    assert np.allclose(env.interceptor_states[1][:3], np.array([200000.0, interceptor_y, 10000.0]))

    # 航向角：两枚拦截弹均沿 180 度迎头飞向红方。
    assert np.isclose(env.interceptor_states[0][5], np.pi)
    assert np.isclose(env.interceptor_states[1][5], np.pi)

    # info：初始信息同步导出每枚拦截弹编号字段。
    assert info["interceptor_count"] == 2
    assert info["interceptor_1_initial_z"] == -10000.0
    assert info["interceptor_2_initial_z"] == 10000.0


def test_single_interceptor_profile_is_explicit_and_observation_second_slot_is_zero():
    """
    验证单拦截弹 debug 模式可以显式构造，并且 10 维观测中的第二弹槽位安全补零。
    """
    # env：论文 200 km profile 下只启用 1 枚拦截弹。
    env = PursueEscapeEnv(_short_env_config(interceptor_count=1))

    # observation/info：单弹 reset 的初始观测和诊断字段。
    observation, info = env.reset(seed=0)

    # 单弹位置：paper profile 中单枚弹放在红方正前方，不使用双弹 ±10km 夹击偏置。
    assert info["interceptor_count"] == 1
    interceptor_y = env.config.red_initial_y + env.config.paper_interceptor_y_offset_from_red

    assert np.allclose(env.interceptor_states[0][:3], np.array([200000.0, interceptor_y, 0.0]))
    assert info["interceptor_1_initial_z"] == 0.0
    assert "interceptor_2_initial_z" not in info

    # 观测仍保持论文式 10 维接口，第二枚弹的 [r, dr, q, dq] 槽位用 0 填充。
    assert observation.shape == (10,)
    assert np.allclose(observation[4:8], np.zeros(4, dtype=np.float32))

    # step_info：执行一步后只出现 interceptor_1 字段，不伪造 interceptor_2 诊断字段。
    _, _, _, _, step_info = env.step(np.array([0.0], dtype=np.float32))
    assert step_info["interceptor_count"] == 1
    assert "interceptor_1_ny_actual" in step_info
    assert "interceptor_2_ny_actual" not in step_info


def test_env_rollout_until_done_or_truncated():
    """
    验证环境可以连续滚动到自然终止或时间截断。
    """
    # env：短时长环境，用于快速覆盖多步滚动。
    env = PursueEscapeEnv(_short_env_config(t=0.2))
    env.reset(seed=0)

    # done：统一终止标志。
    done = False

    while not done:
        # action：随机合法动作。
        action = env.action_space.sample()

        _, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    # metrics：回合结束后的聚合指标。
    metrics = env.get_episode_metrics()

    assert "total_reward" in metrics
    assert "min_distance" in metrics
    assert "interceptor_count" in metrics
    assert env.current_step > 0
