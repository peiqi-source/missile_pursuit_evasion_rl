"""
sac_agent.py

作用：
    实现完整 Soft Actor-Critic（SAC）智能体。

SAC 的核心组成：
    1. Actor 策略网络：
        输入状态，输出连续动作分布。

    2. Critic 双 Q 网络：
        输入状态和动作，输出 Q1(s,a)、Q2(s,a)。

    3. Target Critic 目标双 Q 网络：
        用于计算稳定的 TD 目标值。

    4. Alpha 熵系数：
        控制探索强度。alpha 越大，策略越鼓励探索。

    5. Soft Update：
        使用 tau 将 Critic 网络参数缓慢同步到 Target Critic。

工程说明：
    本文件只实现 SAC 算法本身，不直接负责环境交互。
    环境交互和训练循环后续放在 trainers/sac_trainer.py 中。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hypersonic_rl.networks import MLPActor, MLPTwinCritic


@dataclass
class SACAgentConfig:
    """
    SACAgentConfig

    作用：
        管理 SAC 智能体所需的全部参数。

    属性：
        state_dim：
            状态向量维度。
            当前环境中通常为 12，即红方前 6 个状态和蓝方前 6 个状态拼接。

        action_dim：
            动作向量维度。
            当前环境中通常为 1，即红方横向机动过载控制量。

        action_low：
            动作下界。
            当前环境中通常为 -nzc_h_max。

        action_high：
            动作上界。
            当前环境中通常为 nzc_h_max。

        hidden_dim：
            Actor 和 Critic MLP 隐藏层宽度。

        gamma：
            折扣因子，用于计算未来累计奖励。

        tau：
            目标 Q 网络软更新系数。

        actor_lr：
            Actor 网络学习率。

        critic_lr：
            Critic 网络学习率。

        alpha_lr：
            熵系数 alpha 的学习率。

        automatic_entropy_tuning：
            是否自动调节熵系数 alpha。

        initial_alpha：
            初始熵系数。
            如果 automatic_entropy_tuning=False，则固定使用该值。

        target_entropy：
            目标策略熵。
            如果为 None，则默认设置为 -action_dim。

        device：
            训练设备，例如 "cpu" 或 "cuda:0"。
    """

    state_dim: int
    action_dim: int
    action_low: float
    action_high: float
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    automatic_entropy_tuning: bool = True
    initial_alpha: float = 0.2
    target_entropy: Optional[float] = None
    device: str = "cpu"


class SACAgent:
    """
    Soft Actor-Critic 智能体。

    对外主要接口：
        select_action():
            根据状态选择动作。

        update():
            根据 replay buffer 采样 batch 更新网络。

        state_dict():
            导出模型参数。

        load_state_dict():
            加载模型参数。
    """

    def __init__(self, config: SACAgentConfig) -> None:
        """
        初始化 SAC 智能体。

        参数：
            config：
                SACAgentConfig 配置对象。
        """
        # config：保存智能体配置，便于后续保存模型和复现实验。
        self.config = config

        # device：PyTorch 张量和网络所在设备。
        self.device = torch.device(config.device)

        # state_dim：状态维度。
        self.state_dim = int(config.state_dim)

        # action_dim：动作维度。
        self.action_dim = int(config.action_dim)

        # gamma：折扣因子。
        self.gamma = float(config.gamma)

        # tau：目标网络软更新系数。
        self.tau = float(config.tau)

        # automatic_entropy_tuning：是否自动学习 alpha 熵系数。
        self.automatic_entropy_tuning = bool(config.automatic_entropy_tuning)

        # actor：SAC 策略网络。
        self.actor = MLPActor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            action_low=config.action_low,
            action_high=config.action_high,
        ).to(self.device)

        # critic：当前双 Q 网络。
        self.critic = MLPTwinCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # target_critic：目标双 Q 网络，用于计算 TD target。
        self.target_critic = MLPTwinCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # 初始化时，目标网络参数完全复制 critic 参数。
        self.target_critic.load_state_dict(self.critic.state_dict())

        # actor_optimizer：Actor 网络优化器。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)

        # critic_optimizer：Critic 网络优化器。
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if self.automatic_entropy_tuning:
            # target_entropy：目标熵，默认取 -action_dim。
            self.target_entropy = (
                float(config.target_entropy)
                if config.target_entropy is not None
                else -float(config.action_dim)
            )

            # log_alpha：alpha 的对数形式。
            # 使用 log_alpha 而不是直接优化 alpha，是为了保证 alpha 始终为正数。
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

            # alpha_optimizer：熵系数优化器。
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            # target_entropy：固定 alpha 模式下不需要目标熵。
            self.target_entropy = None

            # fixed_alpha：固定熵系数。
            fixed_alpha = max(float(config.initial_alpha), 1e-8)

            # log_alpha：固定 alpha 的对数，不参与梯度更新。
            self.log_alpha = torch.tensor(np.log(fixed_alpha), dtype=torch.float32, device=self.device)

            # alpha_optimizer：固定 alpha 模式下没有优化器。
            self.alpha_optimizer = None

    @property
    def alpha(self) -> torch.Tensor:
        """
        当前熵系数 alpha。

        返回：
            alpha：
                正数张量，用于控制策略熵奖励强度。
        """
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        根据单个状态选择动作。

        参数：
            state：
                单个状态数组，形状为 [state_dim]。

            deterministic：
                是否使用确定性动作。
                训练时通常为 False，评估时通常为 True。

        返回：
            action：
                NumPy 格式动作数组，形状为 [action_dim]。
        """
        # state_array：将输入状态转换为 float32，并拉平成一维向量。
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)

        if state_array.shape[0] != self.state_dim:
            raise ValueError(f"state 维度错误，应为 {self.state_dim}，实际为 {state_array.shape[0]}")

        # state_tensor：增加 batch 维度后的状态张量，形状为 [1, state_dim]。
        state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            # action_tensor：Actor 输出的动作张量，形状为 [1, action_dim]。
            action_tensor = self.actor.act(state_tensor, deterministic=deterministic)
        self.actor.train()

        # action：去掉 batch 维度并转回 numpy。
        action = action_tensor.squeeze(0).cpu().numpy()

        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        使用一个 batch 更新 SAC 网络。

        参数：
            batch：
                ReplayBuffer.sample() 返回的字典，包含：
                state, action, reward, next_state, done

        返回：
            loss_info：
                当前更新步骤的损失和 alpha 信息，方便 trainer 记录日志。
        """
        # state：当前状态张量。
        state = batch["state"].to(self.device)

        # action：当前动作张量。
        action = batch["action"].to(self.device)

        # reward：奖励张量。
        reward = batch["reward"].to(self.device)

        # next_state：下一状态张量。
        next_state = batch["next_state"].to(self.device)

        # done：终止标志张量。done=1 表示回合结束。
        done = batch["done"].to(self.device)

        # ============================================================
        # 1. 更新 Critic 双 Q 网络
        # ============================================================

        with torch.no_grad():
            # next_action：下一状态下由当前策略采样的动作。
            next_action, next_log_prob, _ = self.actor.sample(next_state, deterministic=False, with_log_prob=True)

            # target_q1/target_q2：目标 Critic 对下一状态动作的 Q 值估计。
            target_q1, target_q2 = self.target_critic(next_state, next_action)

            # target_min_q：双 Q 中较小的 Q 值，降低过估计风险。
            target_min_q = torch.min(target_q1, target_q2)

            # target_v：SAC 的 soft value，包含熵奖励项。
            target_v = target_min_q - self.alpha.detach() * next_log_prob

            # target_q：Bellman 目标值。
            # 如果 done=1，则不再加未来价值。
            target_q = reward + (1.0 - done) * self.gamma * target_v

        # current_q1/current_q2：当前 Critic 对真实 batch 动作的 Q 值估计。
        current_q1, current_q2 = self.critic(state, action)

        # critic_loss：两个 Q 网络的均方误差损失。
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ============================================================
        # 2. 更新 Actor 策略网络
        # ============================================================

        # sampled_action：当前状态下策略网络重新采样动作。
        sampled_action, log_prob, _ = self.actor.sample(state, deterministic=False, with_log_prob=True)

        # q1_pi/q2_pi：Critic 对策略动作的 Q 值估计。
        q1_pi, q2_pi = self.critic(state, sampled_action)

        # min_q_pi：双 Q 中较小的值。
        min_q_pi = torch.min(q1_pi, q2_pi)

        # actor_loss：SAC 策略损失。
        # 目标是最大化 Q 值，同时保留足够熵。
        # PyTorch 优化器默认最小化 loss，因此写成 alpha * log_prob - Q。
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ============================================================
        # 3. 更新 alpha 熵系数
        # ============================================================

        if self.automatic_entropy_tuning:
            # alpha_loss：自动熵调节损失。
            # log_prob + target_entropy 用 detach，避免 alpha 更新影响 Actor 梯度。
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            assert self.alpha_optimizer is not None
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            # alpha_loss：固定 alpha 时记为 0，方便日志统一。
            alpha_loss = torch.tensor(0.0, device=self.device)

        # ============================================================
        # 4. 软更新 Target Critic 网络
        # ============================================================

        self.soft_update()

        # loss_info：返回给 Trainer 的日志字典。
        loss_info = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "q1_mean": float(current_q1.detach().mean().cpu().item()),
            "q2_mean": float(current_q2.detach().mean().cpu().item()),
            "target_q_mean": float(target_q.detach().mean().cpu().item()),
            "log_prob_mean": float(log_prob.detach().mean().cpu().item()),
        }

        return loss_info

    def soft_update(self) -> None:
        """
        软更新目标 Critic 网络。

        更新公式：
            target_param = tau * source_param + (1 - tau) * target_param

        作用：
            避免目标 Q 值剧烈变化，提高训练稳定性。
        """
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def train_mode(self) -> None:
        """
        将 Actor、Critic 设置为训练模式。
        """
        self.actor.train()
        self.critic.train()
        self.target_critic.train()

    def eval_mode(self) -> None:
        """
        将 Actor、Critic 设置为评估模式。
        """
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def state_dict(self) -> Dict[str, Any]:
        """
        导出 SAC 智能体完整状态。

        返回：
            state_dict：
                包含网络参数、优化器参数和 alpha 参数的字典。
        """
        # result：最终导出的状态字典。
        result: Dict[str, Any] = {
            "config": self.config.__dict__,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
        }

        if self.alpha_optimizer is not None:
            result["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        return result

    def load_state_dict(self, state_dict: Dict[str, Any], load_optimizers: bool = True) -> None:
        """
        加载 SAC 智能体完整状态。

        参数：
            state_dict：
                由 self.state_dict() 导出的字典。

            load_optimizers：
                是否加载优化器状态。
                继续训练时为 True，只做测试时可以为 False。
        """
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])

        # loaded_log_alpha：加载到当前设备上的 log_alpha。
        loaded_log_alpha = state_dict.get("log_alpha")

        if loaded_log_alpha is not None:
            if self.automatic_entropy_tuning:
                self.log_alpha.data.copy_(loaded_log_alpha.to(self.device))
            else:
                self.log_alpha = loaded_log_alpha.to(self.device)

        if load_optimizers:
            if "actor_optimizer" in state_dict:
                self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])

            if "critic_optimizer" in state_dict:
                self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

            if self.alpha_optimizer is not None and "alpha_optimizer" in state_dict:
                self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SACAgent":
        """
        从普通字典创建 SACAgent。

        参数：
            config_dict：
                配置字典。通常来自 yaml 文件和环境参数组合。

        返回：
            agent：
                SACAgent 实例。
        """
        # config：转换后的 SACAgentConfig 对象。
        config = SACAgentConfig(**config_dict)

        return cls(config)