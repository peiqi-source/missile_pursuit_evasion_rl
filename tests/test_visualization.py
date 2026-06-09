"""
test_visualization.py

作用：
    测试可视化模块能否正常保存图像。

测试内容：
    1. 单回合综合诊断图；
    2. 单视图轨迹图；
    3. 三视图轨迹图；
    4. 三维轨迹图；
    5. 三视图 + 三维综合图；
    6. 训练曲线图。
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


def _run_short_random_episode():
    """
    运行一个很短的随机动作回合。

    返回：
        env：
            已经有轨迹记录的环境对象。
    """
    # config：短回合环境配置，用于快速测试。
    config = PursueEscapeEnvConfig(dt=0.01, t=0.1)

    # env：追逃突防环境。
    env = PursueEscapeEnv(config)

    # observation：初始观测。
    observation, info = env.reset(seed=0)

    # done：统一终止标志。
    done = False

    while not done:
        # action：随机合法动作。
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

    return env


def test_plot_episode_summary(tmp_path: Path):
    """
    测试单回合综合图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # save_path：测试图像保存路径。
    save_path = tmp_path / "episode_summary.png"

    # saved_path：实际保存路径。
    saved_path = plot_episode_summary(env, save_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_trajectory_single_view(tmp_path: Path):
    """
    测试单视图轨迹图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    for plane in ["xz", "xy", "yz"]:
        # save_path：当前平面轨迹图保存路径。
        save_path = tmp_path / f"trajectory_{plane}.png"

        # saved_path：实际保存路径。
        saved_path = plot_trajectory(env, save_path, plane=plane)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0


def test_plot_three_view_trajectory(tmp_path: Path):
    """
    测试三视图轨迹图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # save_path：三视图图像保存路径。
    save_path = tmp_path / "trajectory_three_views.png"

    # saved_path：实际保存路径。
    saved_path = plot_three_view_trajectory(env, save_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_3d_trajectory(tmp_path: Path):
    """
    测试三维轨迹图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # save_path：三维轨迹图保存路径。
    save_path = tmp_path / "trajectory_3d.png"

    # saved_path：实际保存路径。
    saved_path = plot_3d_trajectory(env, save_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_full_trajectory_summary(tmp_path: Path):
    """
    测试三视图 + 三维综合图是否可以保存。
    """
    # env：已运行过的环境。
    env = _run_short_random_episode()

    # save_path：综合轨迹图保存路径。
    save_path = tmp_path / "trajectory_full_summary.png"

    # saved_path：实际保存路径。
    saved_path = plot_full_trajectory_summary(env, save_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_training_curves(tmp_path: Path):
    """
    测试训练曲线是否可以保存。
    """
    # metrics：模拟训练指标。
    metrics = [
        {"episode": 0, "total_reward": 10.0, "min_distance": 5.0, "success": 0.0},
        {"episode": 1, "total_reward": 20.0, "min_distance": 8.0, "success": 1.0},
        {"episode": 2, "total_reward": 15.0, "min_distance": 7.5, "success": 1.0},
    ]

    # saved_paths：保存的图像路径字典。
    saved_paths = plot_training_curves(metrics, save_dir=tmp_path, rolling_window=2)

    assert saved_paths["reward_curve"].exists()
    assert saved_paths["min_distance_curve"].exists()
    assert saved_paths["success_rate_curve"].exists()