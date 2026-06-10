"""
test_visualization.py

作用：
    验证双拦截弹轨迹、单回合诊断图和训练曲线图能够正常保存。
"""

from pathlib import Path

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.visualization import (
    plot_3d_trajectory,
    plot_episode_summary,
    plot_full_trajectory_summary,
    plot_three_view_trajectory,
    plot_training_curves,
    plot_trajectory,
)


def _run_short_random_episode() -> PursueEscapeEnv:
    """
    运行一个很短的随机动作双弹回合。

    返回：
        env：
            已经记录轨迹、控制量和诊断信息的环境对象。
    """
    # config：短回合环境配置，用于快速测试绘图链路。
    config = PursueEscapeEnvConfig(
        dt=0.01,
        t=0.1,
        interceptor_count=2,
        initial_randomization_enabled=False,
    )

    # env：追逃突防环境。
    env = PursueEscapeEnv(config)
    env.reset(seed=0)

    # done：统一终止标志。
    done = False

    while not done:
        # action：随机合法红方横向过载动作。
        action = env.action_space.sample()

        _, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    return env


def test_plot_episode_summary(tmp_path: Path):
    """
    测试单回合综合诊断图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # saved_path：实际保存路径。
    saved_path = plot_episode_summary(env, tmp_path / "episode_summary.png")

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_trajectory_single_view(tmp_path: Path):
    """
    测试单视图轨迹图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    for plane in ["xz", "xy", "yz"]:
        # saved_path：当前平面轨迹图保存路径。
        saved_path = plot_trajectory(env, tmp_path / f"trajectory_{plane}.png", plane=plane)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0


def test_plot_three_view_trajectory(tmp_path: Path):
    """
    测试三视图轨迹图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # saved_path：三视图图像保存路径。
    saved_path = plot_three_view_trajectory(env, tmp_path / "trajectory_three_views.png")

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_3d_trajectory(tmp_path: Path):
    """
    测试三维轨迹图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # saved_path：三维轨迹图保存路径。
    saved_path = plot_3d_trajectory(env, tmp_path / "trajectory_3d.png")

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_full_trajectory_summary(tmp_path: Path):
    """
    测试三视图加三维综合轨迹图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # saved_path：综合轨迹图保存路径。
    saved_path = plot_full_trajectory_summary(env, tmp_path / "trajectory_full_summary.png")

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_training_curves(tmp_path: Path):
    """
    测试训练曲线是否可以保存，并验证拦截弹能耗字段使用新命名。
    """
    # metrics：模拟训练指标。
    metrics = [
        {
            "episode": 0,
            "total_reward": 10.0,
            "min_distance": 5.0,
            "success": 0.0,
            "interceptor_control_energy": 3.0,
            "interceptor_ny_control_energy": 1.0,
            "interceptor_nz_control_energy": 2.0,
        },
        {
            "episode": 1,
            "total_reward": 20.0,
            "min_distance": 8.0,
            "success": 1.0,
            "interceptor_control_energy": 4.0,
            "interceptor_ny_control_energy": 1.5,
            "interceptor_nz_control_energy": 2.5,
        },
        {
            "episode": 2,
            "total_reward": 15.0,
            "min_distance": 7.5,
            "success": 1.0,
            "interceptor_control_energy": 5.0,
            "interceptor_ny_control_energy": 2.0,
            "interceptor_nz_control_energy": 3.0,
        },
    ]

    # saved_paths：保存的图像路径字典。
    saved_paths = plot_training_curves(metrics, save_dir=tmp_path, rolling_window=2)

    assert saved_paths["reward_curve"].exists()
    assert saved_paths["min_distance_curve"].exists()
    assert saved_paths["success_rate_curve"].exists()
    assert saved_paths["interceptor_control_energy_curve"].exists()
    assert saved_paths["interceptor_ny_control_energy_curve"].exists()
    assert saved_paths["interceptor_nz_control_energy_curve"].exists()
