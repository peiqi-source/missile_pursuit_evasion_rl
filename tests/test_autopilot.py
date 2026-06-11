"""
test_autopilot.py

作用：
    测试统一自动驾驶仪和二维过载限幅工具。
"""

import numpy as np

from hypersonic_rl.envs.autopilot import FirstOrderAutopilot, limit_planar_overload


def test_first_order_autopilot_scalar_response_matches_formula():
    """
    验证一阶自动驾驶仪的标量响应符合离散一阶惯性公式。
    """
    # autopilot：使用初始输出为 0 的一阶自动驾驶仪。
    autopilot = FirstOrderAutopilot(tau=0.5, initial_output=0.0)

    # output：对 2g 指令推进一个 0.1s 步长后的输出。
    output = autopilot.step(command=2.0, dt=0.1)

    assert np.isclose(float(output), 0.4)


def test_first_order_autopilot_vector_reset_and_response():
    """
    验证一阶自动驾驶仪支持 ny/nz 双通道向量响应和重置。
    """
    # autopilot：初始输出设置为平飞平衡项和零侧向过载。
    autopilot = FirstOrderAutopilot(tau=0.5, initial_output=np.array([1.0, 0.0]))

    # output：同时对纵向和侧向指令进行一阶响应。
    output = autopilot.step(command=np.array([3.0, -2.0]), dt=0.25)

    assert np.allclose(output, np.array([2.0, -1.0]))

    # 重置后再次推进，确认内部状态已被覆盖。
    autopilot.reset(initial_output=np.array([0.5, 0.5]))
    output_after_reset = autopilot.step(command=np.array([0.5, 1.5]), dt=0.5)

    assert np.allclose(output_after_reset, np.array([0.5, 1.5]))


def test_first_order_autopilot_rate_limit_reports_saturation():
    """
    验证一阶自动驾驶仪启用速率限制后会裁剪输出变化量并返回诊断字段。
    """
    # autopilot：设置每秒最多变化 2g，0.1s 内最多变化 0.2g。
    autopilot = FirstOrderAutopilot(tau=0.1, initial_output=0.0, rate_limit=2.0)

    # output/info：纯一阶响应希望到 10g，但速率限制会把本步输出裁剪为 0.2g。
    output, info = autopilot.step_with_info(command=10.0, dt=0.1)

    assert np.isclose(float(output), 0.2)
    assert info["rate_saturated"] is True
    assert info["output_saturated"] is False


def test_limit_planar_overload_preserves_equilibrium_ny_when_lateral_only():
    """
    验证二维过载限幅只约束机动分量，不破坏纵向平衡项。
    """
    # theta：当前航迹倾角为 60 度，因此平衡项为 cos(theta)=0.5。
    theta = np.pi / 3.0

    # limited_ny/limited_nz：给出超限侧向指令后应按合成过载缩放。
    limited_ny, limited_nz = limit_planar_overload(
        ny_command=np.cos(theta),
        nz_command=10.0,
        max_overload=4.0,
        theta=theta,
    )

    assert np.isclose(limited_ny, np.cos(theta))
    assert np.isclose(abs(limited_nz), 4.0)


def test_limit_planar_overload_scales_two_channel_maneuver_norm():
    """
    验证二维过载限幅会按比例缩放纵向和侧向机动分量。
    """
    # theta：平飞状态下纵向平衡项为 1g。
    theta = 0.0

    # limited_ny/limited_nz：机动分量 [3, 4] 超过 2.5g 后应缩放到合成范数 2.5。
    limited_ny, limited_nz = limit_planar_overload(
        ny_command=4.0,
        nz_command=4.0,
        max_overload=2.5,
        theta=theta,
    )

    # maneuver_norm：限幅后相对平衡项的二维机动过载范数。
    maneuver_norm = np.hypot(limited_ny - np.cos(theta), limited_nz)

    assert np.isclose(maneuver_norm, 2.5)
    assert limited_ny > np.cos(theta)
    assert limited_nz > 0.0
