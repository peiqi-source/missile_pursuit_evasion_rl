"""
test_evaluation_metrics.py

作用：
    验证评估指标已经升级为 per-interceptor 字段。
"""

import numpy as np

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.evaluation.metrics import (
    collect_episode_metrics,
    compute_absolute_control_effort,
    summarize_metrics,
)


def test_compute_absolute_control_effort_matches_paper_style_integral():
    """
    验证论文式绝对过载积分使用 sum(abs(u))*dt 口径。
    """
    effort = compute_absolute_control_effort([1.0, -2.0, 0.5], dt=0.1)

    assert np.isclose(effort, 0.35)


def test_collect_episode_metrics_uses_continuous_min_distance_and_interceptor_channels():
    """
    验证单回合评估优先使用连续最小距离，并统计每枚拦截弹的双通道能耗。
    """
    # env：构造一个可直接写入 trace 的短回合双弹环境。
    env = PursueEscapeEnv(
        PursueEscapeEnvConfig(
            dt=0.01,
            kill_radius=5.0,
            interceptor_count=2,
            initial_randomization_enabled=False,
        )
    )
    env.reset(seed=0)

    # reward_trace/distance_trace：模拟已完成回合的奖励和采样距离。
    env.reward_trace = [1.0, 2.0]
    env.distance_trace = [100.0, 10.0]

    # min_distance：模拟连续最近点判据得到的全局最小距离。
    env.min_distance = 3.0

    # current_step/current_time：模拟回合长度。
    env.current_step = 2
    env.current_time = 0.02

    # red_inertial_control_trace：红方一维实际横向过载记录。
    env.red_inertial_control_trace = [1.0, -1.0]

    # interceptor_*_traces：两枚拦截弹的双通道实际过载和航迹倾角记录。
    env.interceptor_ny_actual_traces = [[1.0, 1.1], [1.0, 0.9]]
    env.interceptor_nz_actual_traces = [[0.0, 2.0], [0.0, -1.0]]
    env.interceptor_theta_traces = [[0.0, 0.0], [0.0, 0.0]]

    # info_trace：模拟每一步诊断信息和终止原因。
    env.info_trace = [
        {
            "closest_distance_this_step": 50.0,
            "interceptor_1_closest_distance_this_step": 50.0,
            "interceptor_2_closest_distance_this_step": 80.0,
        },
        {
            "closest_distance_this_step": 3.0,
            "interceptor_1_closest_distance_this_step": 3.0,
            "interceptor_2_closest_distance_this_step": 20.0,
            "interceptor_1_min_distance": 3.0,
            "interceptor_2_min_distance": 20.0,
            "interceptor_1_intercepted": True,
            "interceptor_2_intercepted": False,
            "intercepted": True,
            "termination_reason": "intercepted",
            "guidance_mode": "source_pn",
            "interceptor_1_phase": "unknown",
            "interceptor_2_phase": "unknown",
            "interceptor_1_command_saturated": True,
            "interceptor_2_command_saturated": False,
            "interceptor_1_pass_time": 0.02,
            "interceptor_1_phase_switch_time": 0.01,
            "interceptor_count": 2,
        },
    ]

    # metrics：单回合评估指标。
    metrics = collect_episode_metrics(env, episode_index=7)

    assert metrics["episode"] == 7
    assert metrics["min_distance"] == 3.0
    assert metrics["sampled_min_distance"] == 10.0
    assert metrics["closest_distance_min"] == 3.0
    assert metrics["success"] == 0.0
    assert metrics["intercepted"] == 1.0
    assert metrics["interceptor_count"] == 2
    assert np.isclose(metrics["red_control_energy"], 0.02)
    assert np.isclose(metrics["red_control_effort_abs"], 0.02)
    assert "target_offset" in metrics
    assert np.isclose(metrics["interceptor_1_ny_control_energy"], 0.0001)
    assert np.isclose(metrics["interceptor_1_nz_control_energy"], 0.04)
    assert np.isclose(metrics["interceptor_1_nz_control_effort_abs"], 0.02)
    assert np.isclose(metrics["interceptor_2_ny_control_energy"], 0.0001)
    assert np.isclose(metrics["interceptor_2_nz_control_energy"], 0.01)
    assert np.isclose(metrics["interceptor_control_energy"], 0.0502)
    assert np.isclose(metrics["interceptor_command_saturation_ratio"], 0.5)
    assert np.isclose(metrics["interceptor_1_pass_time"], 0.02)


def test_summarize_metrics_includes_interceptor_energy_fields():
    """
    验证多回合汇总保留双拦截弹能耗统计字段。
    """
    # summary：对单条模拟指标记录做汇总。
    summary = summarize_metrics(
        [
            {
                "total_reward": 1.0,
                "min_distance": 2.0,
                "final_distance": 3.0,
                "success": 0.0,
                "intercepted": 1.0,
                "episode_steps": 4,
                "red_control_energy": 5.0,
                "red_control_effort_abs": 2.0,
                "target_offset": 10.0,
                "interceptor_control_energy": 6.0,
                "interceptor_ny_control_energy": 2.0,
                "interceptor_nz_control_energy": 4.0,
                "interceptor_control_effort_abs": 3.0,
                "interceptor_command_saturation_ratio": 0.25,
            }
        ]
    )

    assert summary["mean_interceptor_control_energy"] == 6.0
    assert summary["mean_interceptor_ny_control_energy"] == 2.0
    assert summary["mean_interceptor_nz_control_energy"] == 4.0
    assert summary["mean_interceptor_control_effort_abs"] == 3.0
    assert summary["mean_interceptor_command_saturation_ratio"] == 0.25
    assert summary["mean_target_offset"] == 10.0
    assert summary["intercept_rate"] == 1.0
