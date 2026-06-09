"""
debug_env_rollout.py

作用：
    使用随机动作快速测试环境是否能完整运行。

运行方式：
    在项目根目录执行：

        $env:PYTHONPATH="$PWD\\src"
        python scripts/debug_env_rollout.py

说明：
    该脚本不训练 SAC，只用于检查环境 step/reset、奖励、终止条件和指标统计是否正常。
"""

from __future__ import annotations

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig


def main() -> None:
    """
    随机动作环境调试主函数。
    """
    # config：环境配置。
    config = PursueEscapeEnvConfig(dt=0.01, t=5.0)

    # env：追逃突防环境。
    env = PursueEscapeEnv(config)

    # observation：初始观测。
    # info：初始信息。
    observation, info = env.reset(seed=0)

    print("环境初始化完成")
    print(f"初始观测维度：{observation.shape}")
    print(f"初始最小距离：{info['min_distance']:.2f} m")

    # done：统一终止标志。
    done = False

    while not done:
        # action：从动作空间随机采样的动作。
        # action_space 是一个 spaces.Box 对象 ,sample() 是 Gym 的 spaces.Box 自带的方法
        action = env.action_space.sample()

        # 执行一步环境交互。
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        if env.current_step % 100 == 0:
            print(
                f"step={env.current_step}, "
                f"time={info['time']:.2f}, "
                f"distance={info['distance']:.2f}, "
                f"min_distance={info['min_distance']:.2f}, "
                f"reward={reward:.3f}"
            )

    # metrics：回合统计指标。
    metrics = env.get_episode_metrics()

    print("=" * 80)
    print("随机动作环境测试结束")
    print(f"累计奖励：{metrics['total_reward']:.3f}")
    print(f"最小距离：{metrics['min_distance']:.3f} m")
    print(f"是否成功：{bool(metrics['success'])}")
    print(f"回合步数：{metrics['episode_steps']:.0f}")
    print(f"回合时间：{metrics['episode_time']:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()