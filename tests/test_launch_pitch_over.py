"""
test_launch_pitch_over.py

作用：
    验证发射后 pitch-over 阶段的独立指令计算和环境调用链路。
"""

import numpy as np

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.envs.interceptor import InterceptorConfig
from hypersonic_rl.envs.launch_pitch_over import compute_launch_pitch_over_command


def _relative_info(desired_theta_deg: float = 10.0) -> dict:
    """
    构造 pitch-over 单元测试需要的最小相对几何信息。
    """
    return {
        "desired_theta": float(np.radians(desired_theta_deg)),
        "closing_speed": 1200.0,
    }


def _pitch_over_config(**kwargs) -> InterceptorConfig:
    """
    构造启用 pitch-over 的拦截弹配置。
    """
    config_kwargs = {
        "enable_launch_pitch_over": True,
        "launch_pitch_over_min_duration": 0.0,
        "launch_pitch_over_max_duration": 100.0,
        "launch_pitch_over_min_altitude": 0.0,
    }
    config_kwargs.update(kwargs)
    return InterceptorConfig(**config_kwargs)


def test_launch_pitch_over_uses_fixed_20_deg_before_blend_region():
    """
    当当前弹道倾角仍大于融合开始角时，参考角应固定为 20 度。
    """
    state = np.zeros(9, dtype=np.float64)
    state[1] = 25000.0
    state[4] = np.radians(80.0)

    command = compute_launch_pitch_over_command(
        interceptor_state=state,
        relative_info=_relative_info(desired_theta_deg=5.0),
        config=_pitch_over_config(),
        elapsed_time=0.0,
    )

    assert command.phase == "launch_pitch_over"
    assert np.isclose(command.info["launch_pitch_over_reference_theta_deg"], 20.0)
    assert np.isclose(command.info["launch_pitch_over_blend_weight"], 0.0)
    assert command.ny_command < np.cos(state[4])


def test_launch_pitch_over_blends_fixed_theta_to_los_theta():
    """
    当当前弹道倾角进入 30 度到 20 度融合区间时，参考角应位于 20 度和 LOS 角之间。
    """
    state = np.zeros(9, dtype=np.float64)
    state[1] = 25000.0
    state[4] = np.radians(25.0)

    command = compute_launch_pitch_over_command(
        interceptor_state=state,
        relative_info=_relative_info(desired_theta_deg=10.0),
        config=_pitch_over_config(),
        elapsed_time=0.0,
    )

    reference_theta_deg = float(command.info["launch_pitch_over_reference_theta_deg"])
    blend_weight = float(command.info["launch_pitch_over_blend_weight"])

    assert 0.0 < blend_weight < 1.0
    assert 10.0 < reference_theta_deg < 20.0


def test_env_uses_launch_pitch_over_for_80_deg_initial_theta():
    """
    启用 pitch-over 且初始弹道倾角为 80 度时，环境第一步应先进入 pitch-over。
    """
    env = PursueEscapeEnv(
        PursueEscapeEnvConfig(
            guidance_mode="mid_terminal_interceptor",
            interceptor_count=1,
            scenario_profile="custom",
            initial_randomization_enabled=False,
            enable_launch_pitch_over=True,
            interceptor_initial_theta_deg=80.0,
            interceptor_initial_x_min=30000.0,
            interceptor_initial_x_max=30000.0,
            interceptor_initial_y_min=25000.0,
            interceptor_initial_y_max=25000.0,
            interceptor_initial_z_min=0.0,
            interceptor_initial_z_max=0.0,
            interceptor_initial_heading_min_deg=180.0,
            interceptor_initial_heading_max_deg=180.0,
            t=2.0,
            dt=0.01,
        )
    )
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["interceptor_1_phase"] == "launch_pitch_over"
    assert info["interceptor_1_launch_pitch_over_active_this_step"]
    assert np.isclose(info["interceptor_1_launch_pitch_over_reference_theta_deg"], 20.0)
    assert np.isclose(info["interceptor_1_launch_pitch_over_blend_weight"], 0.0)
    assert info["interceptor_1_target_compensation_skipped"]
    assert info["interceptor_1_phase_switch_step"] == -1


def test_launch_pitch_over_exit_does_not_record_terminal_phase_switch():
    """
    pitch-over 退出到中制导不应写入末制导 phase_switch_time。
    """
    env = PursueEscapeEnv(
        PursueEscapeEnvConfig(
            guidance_mode="mid_terminal_interceptor",
            interceptor_count=1,
            scenario_profile="custom",
            initial_randomization_enabled=False,
            enable_launch_pitch_over=True,
            interceptor_initial_theta_deg=80.0,
            interceptor_initial_x_min=30000.0,
            interceptor_initial_x_max=30000.0,
            interceptor_initial_y_min=25000.0,
            interceptor_initial_y_max=25000.0,
            interceptor_initial_z_min=0.0,
            interceptor_initial_z_max=0.0,
            interceptor_initial_heading_min_deg=180.0,
            interceptor_initial_heading_max_deg=180.0,
            launch_pitch_over_min_duration=0.0,
            launch_pitch_over_max_duration=0.0,
            t=2.0,
            dt=0.01,
        )
    )
    env.reset(seed=0)

    _, _, _, _, first_info = env.step(np.array([0.0], dtype=np.float32))
    _, _, _, _, second_info = env.step(np.array([0.0], dtype=np.float32))

    assert first_info["interceptor_1_phase"] == "launch_pitch_over"
    assert first_info["interceptor_1_launch_pitch_over_exit_ready"]
    assert second_info["interceptor_1_phase"] in {"midcourse", "terminal"}
    assert second_info["interceptor_1_phase_switch_step"] == -1
