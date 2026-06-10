"""
test_guidance_closure.py

作用：
    验证动力学符号、单弹制导闭环和环境 per-interceptor 诊断字段。
"""

import numpy as np

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.envs.dynamics import build_velocity_vector, compute_relative_geometry
from hypersonic_rl.envs.interceptor import Interceptor, InterceptorConfig


def _state(x: float, speed: float, psi: float) -> np.ndarray:
    """
    构造用于几何和速度测试的简化状态向量。

    参数：
        x：
            初始 x 坐标。
        speed：
            飞行速度。
        psi：
            航向角，单位 rad。

    返回：
        state：
            长度为 9 的状态向量。
    """
    # state：按照环境约定构造 [x, y, z, v, theta, psi, nx, ny, nz]。
    state = np.zeros(9, dtype=np.float64)
    state[0] = x
    state[1] = 25000.0
    state[3] = speed
    state[5] = psi
    state[7] = 1.0
    return state


def _fixed_head_on_config(**kwargs) -> PursueEscapeEnvConfig:
    """
    构造固定迎头单弹测试场景配置。

    参数：
        **kwargs：
            覆盖默认测试配置的关键字参数。

    返回：
        config：
            PursueEscapeEnvConfig 配置对象。
    """
    # config_kwargs：固定 30 km 迎头场景，便于快速触发连续最近点命中。
    config_kwargs = {
        "dt": 0.01,
        "t": 20.0,
        "kill_radius": 5.0,
        "interceptor_count": 1,
        "scenario_profile": "custom",
        "observation_mode": "thesis_end_to_end_10d",
        "initial_randomization_enabled": False,
        "red_initial_x": 0.0,
        "red_initial_y": 25000.0,
        "red_initial_z": 0.0,
        "red_initial_psi_deg": 0.0,
        "interceptor_initial_x_min": 30000.0,
        "interceptor_initial_x_max": 30000.0,
        "interceptor_initial_y_min": 25000.0,
        "interceptor_initial_y_max": 25000.0,
        "interceptor_initial_z_min": 0.0,
        "interceptor_initial_z_max": 0.0,
        "interceptor_initial_heading_min_deg": 180.0,
        "interceptor_initial_heading_max_deg": 180.0,
    }
    config_kwargs.update(kwargs)
    return PursueEscapeEnvConfig(**config_kwargs)


def test_velocity_vector_uses_original_lateral_sign():
    """
    验证速度向量的 z 轴符号与原始模型约定一致。
    """
    # head_left_state：航向角为 pi 时应沿 x 负方向飞行。
    head_left_state = _state(x=0.0, speed=100.0, psi=np.pi)

    # left_velocity：对应的三维速度向量。
    left_velocity = build_velocity_vector(head_left_state)

    assert left_velocity[0] < 0.0
    assert np.isclose(left_velocity[2], 0.0, atol=1e-10)

    # lateral_state：航向角为 pi/2 时主要体现 z 轴速度符号。
    lateral_state = _state(x=0.0, speed=100.0, psi=np.pi / 2.0)

    # lateral_velocity：侧向速度向量。
    lateral_velocity = build_velocity_vector(lateral_state)

    assert np.isclose(lateral_velocity[0], 0.0, atol=1e-10)
    assert lateral_velocity[2] < 0.0


def test_relative_geometry_points_from_interceptor_to_red_for_head_on_case():
    """
    验证相对几何采用从拦截弹指向红方的 red-interceptor 定义。
    """
    # red_state：红方从原点沿 x 正方向飞行。
    red_state = _state(x=0.0, speed=2040.0, psi=0.0)

    # interceptor_state：拦截弹位于前方并沿 x 负方向迎头飞行。
    interceptor_state = _state(x=30000.0, speed=1360.0, psi=np.pi)

    # relative_info：红方和拦截弹的相对几何诊断信息。
    relative_info = compute_relative_geometry(red_state, interceptor_state)

    assert relative_info["dx"] < 0.0
    assert relative_info["closing_speed"] > 0.0
    assert relative_info["range_rate"] < 0.0


def test_mid_terminal_interceptor_has_no_initial_lateral_command_when_head_on():
    """
    验证完整拦截弹在严格迎头时不产生额外侧向修正。
    """
    # env：固定迎头、完整中末制导模式环境。
    env = PursueEscapeEnv(_fixed_head_on_config(guidance_mode="mid_terminal_interceptor"))
    env.reset(seed=0)

    # info：执行一步后的制导诊断信息。
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert np.isclose(abs(info["interceptor_1_desired_psi"]), np.pi)
    assert np.isclose(info["interceptor_1_psi_error"], 0.0, atol=1e-10)
    assert np.isclose(info["interceptor_1_nz_command"], 0.0, atol=1e-10)


def test_mid_terminal_target_compensation_matches_source_pn_sign():
    """
    验证完整拦截弹目标机动前馈与 source_pn 使用同一侧向符号约定。
    """
    # positive_env：红方给正向侧向机动时，拦截弹应加入负向补偿。
    positive_env = PursueEscapeEnv(
        _fixed_head_on_config(
            guidance_mode="mid_terminal_interceptor",
            interceptor_target_compensation_gain=4.0,
        )
    )
    positive_env.reset(seed=0)
    _, _, _, _, positive_info = positive_env.step(np.array([2.0], dtype=np.float32))

    assert positive_info["red_inertial_overload"] > 0.0
    assert positive_info["interceptor_1_target_compensation"] < 0.0
    assert np.isclose(
        positive_info["interceptor_1_target_compensation"],
        -4.0 * positive_info["red_inertial_overload"],
    )

    # negative_env：红方给负向侧向机动时，拦截弹应加入正向补偿。
    negative_env = PursueEscapeEnv(
        _fixed_head_on_config(
            guidance_mode="mid_terminal_interceptor",
            interceptor_target_compensation_gain=4.0,
        )
    )
    negative_env.reset(seed=0)
    _, _, _, _, negative_info = negative_env.step(np.array([-2.0], dtype=np.float32))

    assert negative_info["red_inertial_overload"] < 0.0
    assert negative_info["interceptor_1_target_compensation"] > 0.0
    assert np.isclose(
        negative_info["interceptor_1_target_compensation"],
        -4.0 * negative_info["red_inertial_overload"],
    )


def test_midcourse_lateral_pn_remains_active_without_psi_shaping():
    """
    验证侧向弹道成型增益为 0 时，中制导仍然依靠 LOS rate PN 输出侧向指令。
    """
    # interceptor：关闭侧向成型项，只保留中制导 PN 主项。
    interceptor = Interceptor(
        InterceptorConfig(
            midcourse_psi_shaping_gain=0.0,
            target_compensation_gain=0.0,
        )
    )

    # interceptor_state：拦截弹位于前方且带有侧向偏置。
    interceptor_state = _state(x=30000.0, speed=1360.0, psi=np.deg2rad(175.0))
    interceptor_state[2] = 1000.0
    interceptor.reset(interceptor_state)

    # red_state：红方从原点沿 x 正方向飞行。
    red_state = _state(x=0.0, speed=2040.0, psi=0.0)

    # relative_info：当前几何下的中制导输入。
    relative_info = interceptor.compute_interceptor_relative_info(red_state)

    # nz_command：中制导 PN 侧向输出。
    _, nz_command = interceptor.compute_midcourse_guidance(relative_info)

    assert not np.isclose(nz_command, 0.0)


def test_mid_terminal_interceptor_corrects_175_degree_heading_toward_target():
    """
    验证完整拦截弹会把 175 度初始航向修正到红方视线方向。
    """
    # initial_heading：拦截弹初始航向角。
    initial_heading = np.deg2rad(175.0)

    # env：设置拦截弹固定 175 度初始航向。
    env = PursueEscapeEnv(
        _fixed_head_on_config(
            guidance_mode="mid_terminal_interceptor",
            interceptor_initial_heading_min_deg=175.0,
            interceptor_initial_heading_max_deg=175.0,
        )
    )
    env.reset(seed=0)

    # info：执行一步后的制导诊断信息。
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["interceptor_1_psi_error"] > 0.0
    assert info["interceptor_1_nz_command"] < 0.0
    assert env.interceptor_states[0][5] > initial_heading


def test_guidance_mode_source_pn_does_not_use_interceptor_phase():
    """
    验证 source_pn 分支不进入完整拦截弹阶段机。
    """
    # env：source_pn 模式环境。
    env = PursueEscapeEnv(_fixed_head_on_config(guidance_mode="source_pn"))
    env.reset(seed=0)

    # info：执行一步后的环境诊断信息。
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["guidance_mode"] == "source_pn"
    assert info["interceptor_1_phase"] == "unknown"
    assert "interceptor_1_ny_command" in info
    assert "interceptor_1_nz_command" in info


def test_source_pn_uses_vertical_channel_for_altitude_error():
    """
    验证 source_pn 在存在高度误差时会启用拦截弹纵向 ny 通道。
    """
    # env：source_pn 模式环境。
    env = PursueEscapeEnv(_fixed_head_on_config(guidance_mode="source_pn"))
    env.reset(seed=0)

    # initial_equilibrium_ny：拦截弹初始航迹倾角对应的纵向平衡项。
    initial_equilibrium_ny = float(np.cos(env.interceptor_states[0][4]))

    # red_state[1]：人为制造红方高于拦截弹的高度误差。
    env.red_state[1] += 1000.0

    # info：执行一步后的拦截弹双通道制导诊断信息。
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["interceptor_1_ny_command"] > initial_equilibrium_ny
    assert info["interceptor_1_ny_actual"] > initial_equilibrium_ny
    assert env.interceptor_states[0][7] == info["interceptor_1_ny_actual"]


def test_guidance_mode_mid_terminal_reports_interceptor_phase():
    """
    验证完整中末制导分支会输出阶段和双通道字段。
    """
    # env：完整中末制导模式环境。
    env = PursueEscapeEnv(_fixed_head_on_config(guidance_mode="mid_terminal_interceptor"))
    env.reset(seed=0)

    # info：执行一步后的环境诊断信息。
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["guidance_mode"] == "mid_terminal_interceptor"
    assert info["interceptor_1_phase"] in {"midcourse", "terminal"}
    assert "interceptor_1_ny_command" in info
    assert "interceptor_1_nz_command" in info


def test_continuous_closest_distance_detects_head_on_intercept():
    """
    验证连续最近点命中判据能捕获一步跨过杀伤半径的迎头拦截。
    """
    # env：固定迎头、完整中末制导模式环境。
    env = PursueEscapeEnv(_fixed_head_on_config(guidance_mode="mid_terminal_interceptor"))
    env.reset(seed=0)

    # terminated/truncated：自然终止和时间截断标志。
    terminated = False
    truncated = False

    # info：保存最后一步诊断信息。
    info = {}

    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))

    assert terminated
    assert not truncated
    assert info["intercepted"]
    assert info["termination_reason"] == "intercepted"
    assert env.min_distance <= env.config.kill_radius
    assert info["closest_distance_this_step"] <= env.config.kill_radius
    assert info["interceptor_1_closest_distance_this_step"] <= env.config.kill_radius
