"""SAC 模型评估器。"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypersonic_rl.agents.sac_agent import SACAgent
from hypersonic_rl.envs.pursue_escape_env import PursueEscapeEnv
from hypersonic_rl.evaluation.metrics import compute_episode_summary


class Evaluator:
    """运行若干测试 episode 并输出聚合指标。"""

    def __init__(
        self,
        env: PursueEscapeEnv,
        agent: SACAgent,
        kill_radius: float = 7.0,
    ) -> None:
        self.env = env
        self.agent = agent
        self.kill_radius = float(kill_radius)

    def evaluate(
        self,
        num_episodes: int = 5,
        max_steps_per_episode: int = 5000,
        deterministic: bool = True,
        seed: int | None = None,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """执行评估并返回平均指标和逐 episode 指标。"""
        self.agent.eval_mode()
        summaries: list[dict[str, float]] = []
        for episode in range(num_episodes):
            episode_seed = None if seed is None else seed + episode
            state, _ = self.env.reset(seed=episode_seed)
            reward_total = 0.0
            for _ in range(max_steps_per_episode):
                action = self.agent.select_action(state, deterministic=deterministic)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                reward_total += float(reward)
                state = next_state
                if terminated or truncated:
                    break
            trace = self.env.get_episode_trace()
            summaries.append(compute_episode_summary(trace, reward_total, self.kill_radius))
        self.agent.train_mode()
        aggregate = self._aggregate(summaries)
        return aggregate, summaries

    @staticmethod
    def _aggregate(summaries: list[dict[str, float]]) -> dict[str, float]:
        """聚合评估指标。"""
        if not summaries:
            return {
                "average_reward": 0.0,
                "average_min_distance": 0.0,
                "success_rate": 0.0,
                "average_control_energy": 0.0,
            }
        return {
            "average_reward": float(np.mean([s["episode_reward"] for s in summaries])),
            "average_min_distance": float(np.mean([s["min_distance"] for s in summaries])),
            "success_rate": float(np.mean([s["success"] for s in summaries])),
            "average_control_energy": float(np.mean([s["control_energy"] for s in summaries])),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        env_config: dict[str, Any],
        agent_config: dict[str, Any],
        device: str,
    ) -> "Evaluator":
        """从 checkpoint 构造评估器。"""
        env = PursueEscapeEnv(env_config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            config=agent_config,
            device=device,
        )
        agent.load(checkpoint_path, load_optimizers=False)
        return cls(env=env, agent=agent, kill_radius=env.kill_radius)

