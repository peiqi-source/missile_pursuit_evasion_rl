import numpy as np

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.envs.dynamics import build_velocity_vector, compute_relative_geometry


def _state(x: float, speed: float, psi: float) -> np.ndarray:
    state = np.zeros(9, dtype=np.float64)
    state[0] = x
    state[1] = 27000.0
    state[3] = speed
    state[5] = psi
    state[7] = 1.0
    return state


def _fixed_head_on_config(**kwargs) -> PursueEscapeEnvConfig:
    config_kwargs = {
        "dt": 0.01,
        "t": 20.0,
        "kill_radius": 5.0,
        "blue_initial_x_min": 30000.0,
        "blue_initial_x_max": 30000.0,
        "blue_initial_heading_min_deg": 180.0,
        "blue_initial_heading_max_deg": 180.0,
    }
    config_kwargs.update(kwargs)
    return PursueEscapeEnvConfig(**config_kwargs)


def test_velocity_vector_uses_original_lateral_sign():
    head_left_state = _state(x=0.0, speed=100.0, psi=np.pi)
    left_velocity = build_velocity_vector(head_left_state)

    assert left_velocity[0] < 0.0
    assert np.isclose(left_velocity[2], 0.0, atol=1e-10)

    lateral_state = _state(x=0.0, speed=100.0, psi=np.pi / 2.0)
    lateral_velocity = build_velocity_vector(lateral_state)

    assert np.isclose(lateral_velocity[0], 0.0, atol=1e-10)
    assert lateral_velocity[2] < 0.0


def test_relative_geometry_points_from_blue_to_red_for_head_on_case():
    red_state = _state(x=0.0, speed=1870.0, psi=0.0)
    blue_state = _state(x=30000.0, speed=1360.0, psi=np.pi)

    relative_info = compute_relative_geometry(red_state=red_state, blue_state=blue_state)

    assert relative_info["dx"] < 0.0
    assert relative_info["closing_speed"] > 0.0
    assert relative_info["range_rate"] < 0.0


def test_mid_terminal_interceptor_has_no_initial_lateral_command_when_head_on():
    env = PursueEscapeEnv(
        _fixed_head_on_config(guidance_mode="mid_terminal_interceptor")
    )
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert np.isclose(abs(info["desired_psi"]), np.pi)
    assert np.isclose(info["psi_error"], 0.0, atol=1e-10)
    assert np.isclose(info["blue_nz_command"], 0.0, atol=1e-10)


def test_mid_terminal_interceptor_corrects_175_degree_heading_toward_target():
    initial_heading = np.deg2rad(175.0)
    env = PursueEscapeEnv(
        _fixed_head_on_config(
            guidance_mode="mid_terminal_interceptor",
            blue_initial_heading_min_deg=175.0,
            blue_initial_heading_max_deg=175.0,
        )
    )
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["psi_error"] > 0.0
    assert info["blue_nz_command"] < 0.0
    assert env.blue_state[5] > initial_heading


def test_guidance_mode_source_pn_does_not_use_interceptor_phase():
    env = PursueEscapeEnv(_fixed_head_on_config(guidance_mode="source_pn"))
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["guidance_mode"] == "source_pn"
    assert info["interceptor_phase"] == "unknown"


def test_guidance_mode_mid_terminal_reports_interceptor_phase():
    env = PursueEscapeEnv(
        _fixed_head_on_config(guidance_mode="mid_terminal_interceptor")
    )
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["guidance_mode"] == "mid_terminal_interceptor"
    assert info["interceptor_phase"] in {"midcourse", "terminal"}
    assert "blue_ny_command" in info
    assert "blue_nz_command" in info


def test_continuous_closest_distance_detects_head_on_intercept():
    env = PursueEscapeEnv(
        _fixed_head_on_config(guidance_mode="mid_terminal_interceptor")
    )
    env.reset(seed=0)

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))

    assert terminated
    assert not truncated
    assert info["intercepted"]
    assert info["termination_reason"] == "intercepted"
    assert env.min_distance <= env.config.kill_radius
    assert info["closest_distance_this_step"] <= env.config.kill_radius
