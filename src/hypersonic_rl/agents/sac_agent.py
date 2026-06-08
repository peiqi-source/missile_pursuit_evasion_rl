"""完整 SAC 智能体实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hypersonic_rl.buffers.replay_buffer import ReplayBuffer
from hypersonic_rl.networks.mlp_actor import MLPActor
from hypersonic_rl.networks.mlp_critic import TwinQNetwork


class SACAgent:
    """Soft Actor-Critic 智能体。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 256))
        hidden_dim = int(config.get("hidden_dim", 256))
        actor_lr = float(config.get("actor_lr", 3e-4))
        critic_lr = float(config.get("critic_lr", 3e-4))
        alpha_lr = float(config.get("alpha_lr", 3e-4))
        self.automatic_entropy_tuning = bool(config.get("automatic_entropy_tuning", True))

        action_low = np.asarray(action_low, dtype=np.float32).reshape(1, action_dim)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(1, action_dim)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        self.actor = MLPActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_scale=action_scale,
            action_bias=action_bias,
        ).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.hard_update(self.critic_target, self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        target_entropy = config.get("target_entropy", "auto")
        if target_entropy == "auto":
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # log_alpha：熵系数 alpha 的对数参数，优化时更稳定。
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = float(self.log_alpha.exp().detach().cpu().item())

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """根据当前策略选择动作。"""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, _ = self.actor.sample(state_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update_parameters(
        self, replay_buffer: ReplayBuffer, batch_size: int | None = None
    ) -> dict[str, float]:
        """执行一次完整 SAC 参数更新。"""
        batch_size = int(batch_size or self.batch_size)
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_min_q = torch.min(target_q1, target_q2)
            assert next_log_probs is not None
            target_q = rewards + (1.0 - dones) * self.gamma * (
                target_min_q - self.log_alpha.exp() * next_log_probs
            )

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs, _ = self.actor.sample(states)
        assert log_probs is not None
        q1_pi, q2_pi = self.critic(states, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp().detach() * log_probs - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha.exp() * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros(1, device=self.device)

        self.alpha = float(self.log_alpha.exp().detach().cpu().item())
        self.soft_update(self.critic_target, self.critic, self.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "alpha": self.alpha,
        }

    @staticmethod
    def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
        """软更新目标网络参数。"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    @staticmethod
    def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
        """硬拷贝网络参数。"""
        target.load_state_dict(source.state_dict())

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """保存智能体 checkpoint。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load(self, path: str | Path, load_optimizers: bool = True) -> dict[str, Any]:
        """加载智能体 checkpoint。"""
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if load_optimizers:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            self.alpha = float(self.log_alpha.exp().detach().cpu().item())
        return checkpoint

    def train_mode(self) -> None:
        """切换到训练模式。"""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """切换到评估模式。"""
        self.actor.eval()
        self.critic.eval()

