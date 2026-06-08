"""普通 SAC 训练器。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hypersonic_rl.agents.sac_agent import SACAgent
from hypersonic_rl.buffers.replay_buffer import ReplayBuffer
from hypersonic_rl.envs.pursue_escape_env import PursueEscapeEnv
from hypersonic_rl.evaluation.evaluator import Evaluator
from hypersonic_rl.evaluation.metrics import compute_episode_summary
from hypersonic_rl.utils.device import get_device
from hypersonic_rl.utils.logger import setup_logger
from hypersonic_rl.utils.seed import set_seed
from hypersonic_rl.visualization.plot_episode import plot_episode
from hypersonic_rl.visualization.plot_training_curve import plot_training_curve


class SACTrainer:
    """封装环境交互、SAC 更新、评估、保存和绘图。"""

    def __init__(
        self,
        env_config: dict[str, Any],
        agent_config: dict[str, Any],
        train_config: dict[str, Any],
    ) -> None:
        self.env_config = env_config
        self.agent_config = agent_config
        self.train_config = train_config
        self.seed = int(train_config.get("seed", 0))
        set_seed(self.seed)
        self.device = get_device()
        self.env = PursueEscapeEnv(env_config)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=self.env.action_space.low,
            action_high=self.env.action_space.high,
            config=agent_config,
            device=self.device,
        )
        capacity = int(agent_config.get("replay_buffer_size", 1000000))
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity, self.device)
        self.run_dir = self._create_run_dir(train_config)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.figure_dir = self.run_dir / "figures"
        self.metric_dir = self.run_dir / "metrics"
        self.log_dir = self.run_dir / "logs"
        for directory in [self.checkpoint_dir, self.figure_dir, self.metric_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("SACTrainer", self.log_dir / "train.log")
        self.metrics_path = self.metric_dir / "metrics.csv"
        self.metrics: list[dict[str, float]] = []
        self.total_env_steps = 0

    def train(self) -> Path:
        """运行 SAC 训练循环并返回本次实验目录。"""
        total_episodes = int(self.train_config.get("total_episodes", 200))
        for episode in range(total_episodes):
            row = self._run_episode(episode)
            self.metrics.append(row)
            self._write_metrics()
            if self._should_plot(episode):
                self._plot_current_episode(episode)
                plot_training_curve(self.metrics_path, self.figure_dir)
            if self._should_evaluate(episode):
                self._evaluate_and_log(episode)
            if self._should_save(episode):
                self._save_checkpoint(episode)
            self.logger.info(
                "episode=%s reward=%.3f min_distance=%.3f success=%.0f",
                episode,
                row["episode_reward"],
                row["min_distance"],
                row["success"],
            )
        self._save_checkpoint(total_episodes - 1)
        plot_training_curve(self.metrics_path, self.figure_dir)
        return self.run_dir

    def _run_episode(self, episode: int) -> dict[str, float]:
        """运行一个训练 episode。"""
        state, _ = self.env.reset(seed=self.seed + episode)
        reward_total = 0.0
        updates: list[dict[str, float]] = []
        max_steps = int(self.train_config.get("max_steps_per_episode", 5000))
        for step in range(max_steps):
            action = self._select_training_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            self.replay_buffer.add(state, action, reward, next_state, done)
            self.total_env_steps += 1
            updates.extend(self._maybe_update())
            reward_total += float(reward)
            state = next_state
            if done:
                break
        trace = self.env.get_episode_trace()
        row = compute_episode_summary(trace, reward_total, self.env.kill_radius)
        row.update(self._mean_update_metrics(updates))
        row["episode"] = float(episode)
        row["steps"] = float(step + 1)
        row["env_steps"] = float(self.total_env_steps)
        row["success_rate"] = self._rolling_success(row["success"])
        return row

    def _select_training_action(self, state: np.ndarray) -> np.ndarray:
        """根据 warmup 阶段或当前策略选择动作。"""
        start_steps = int(self.train_config.get("start_steps", 1000))
        if self.total_env_steps < start_steps:
            return self.env.action_space.sample()
        return self.agent.select_action(state, deterministic=False)

    def _maybe_update(self) -> list[dict[str, float]]:
        """按 update_after/update_every 控制 SAC 更新频率。"""
        update_after = int(self.train_config.get("update_after", 1000))
        update_every = int(self.train_config.get("update_every", 1))
        num_updates = int(self.train_config.get("num_updates_per_step", 1))
        batch_size = int(self.agent_config.get("batch_size", 256))
        if self.total_env_steps < update_after or self.total_env_steps % update_every != 0:
            return []
        if len(self.replay_buffer) < batch_size:
            return []
        return [self.agent.update_parameters(self.replay_buffer, batch_size) for _ in range(num_updates)]

    @staticmethod
    def _mean_update_metrics(updates: list[dict[str, float]]) -> dict[str, float]:
        """计算本 episode 的平均损失。"""
        keys = ["actor_loss", "critic_loss", "alpha_loss", "alpha"]
        if not updates:
            return {key: float("nan") for key in keys}
        return {key: float(np.mean([update[key] for update in updates])) for key in keys}

    def _rolling_success(self, current_success: float, window: int = 20) -> float:
        """计算滚动成功率。"""
        values = [row["success"] for row in self.metrics[-window:]] + [current_success]
        return float(np.mean(values))

    def _write_metrics(self) -> None:
        """保存 metrics.csv。"""
        if not self.metrics:
            return
        fieldnames = list(self.metrics[-1].keys())
        with self.metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)

    def _plot_current_episode(self, episode: int) -> None:
        """保存当前 episode 图。"""
        trace = self.env.get_episode_trace()
        plot_episode(trace, self.figure_dir / f"episode_{episode:04d}.png")

    def _evaluate_and_log(self, episode: int) -> None:
        """定期评估并写日志。"""
        eval_env = PursueEscapeEnv(self.env_config)
        evaluator = Evaluator(eval_env, self.agent, self.env.kill_radius)
        aggregate, _ = evaluator.evaluate(
            num_episodes=3,
            max_steps_per_episode=int(self.train_config.get("max_steps_per_episode", 5000)),
            deterministic=True,
            seed=self.seed + 10000 + episode,
        )
        self.logger.info("eval episode=%s metrics=%s", episode, aggregate)

    def _save_checkpoint(self, episode: int) -> None:
        """保存 latest 和按 episode 命名的 checkpoint。"""
        extra = {
            "episode": episode,
            "env_config": self.env_config,
            "agent_config": self.agent_config,
            "train_config": self.train_config,
        }
        self.agent.save(self.checkpoint_dir / "latest.pt", extra=extra)
        self.agent.save(self.checkpoint_dir / f"episode_{episode:04d}.pt", extra=extra)

    def _should_plot(self, episode: int) -> bool:
        """判断是否需要绘图。"""
        interval = int(self.train_config.get("plot_interval", 10))
        return episode == 0 or (episode + 1) % interval == 0

    def _should_evaluate(self, episode: int) -> bool:
        """判断是否需要评估。"""
        interval = int(self.train_config.get("eval_interval", 10))
        return (episode + 1) % interval == 0

    def _should_save(self, episode: int) -> bool:
        """判断是否需要保存 checkpoint。"""
        interval = int(self.train_config.get("save_interval", 20))
        return (episode + 1) % interval == 0

    @staticmethod
    def _create_run_dir(train_config: dict[str, Any]) -> Path:
        """创建时间戳实验目录。"""
        if train_config.get("run_dir"):
            return Path(train_config["run_dir"]).resolve()
        experiment_dir = Path(train_config.get("experiment_dir", "experiments/sac_baseline"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (experiment_dir / timestamp).resolve()

