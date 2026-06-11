"""
lstm_sac_agent.py

作用：
    实现论文第 5 章 LSTM-SAC 智能体。

核心变化：
    普通 SAC 输入单步状态 s_t；
    LSTM-SAC 输入连续状态序列 [s_{t-L+1}, ..., s_t]。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hypersonic_rl.networks import LSTMActor, LSTMTwinCritic


@dataclass
class LSTMSACAgentConfig:
    """
    LSTMSACAgentConfig

    作用：
        管理论文第 5 章 LSTM-SAC 智能体参数。

    属性：
        state_dim：
            单个状态向量维度，当前双弹端到端环境为 10。
        action_dim：
            动作向量维度，当前红方横向过载为 1。
        sequence_length：
            LSTM 输入状态序列长度，论文最佳结果对应 n=3。
        lstm_layer_sizes：
            Actor/Critic 共享的 LSTM 层宽度，默认对齐论文表 5-1。
        actor_fc_hidden_dim：
            Actor LSTM 后全连接层宽度。
        critic_hidden_dims：
            Critic LSTM 后全连接层宽度。
    """

    state_dim: int
    action_dim: int
    action_low: float
    action_high: float
    sequence_length: int = 3
    lstm_layer_sizes: Sequence[int] = (256, 256, 128)
    actor_fc_hidden_dim: int = 128
    critic_hidden_dims: Sequence[int] = (128, 128, 64)
    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 3e-3
    critic_lr: float = 5e-3
    alpha_lr: float = 1e-3
    automatic_entropy_tuning: bool = True
    initial_alpha: float = 0.2
    target_entropy: Optional[float] = None
    device: str = "cpu"


class LSTMSACAgent:
    """
    LSTM-SAC 智能体。

    对外接口：
        select_action():
            根据状态序列输出动作。
        update():
            使用序列 replay buffer batch 更新 Actor、Critic 和 alpha。
        state_dict()/load_state_dict():
            保存和恢复完整训练状态。
    """

    def __init__(self, config: LSTMSACAgentConfig) -> None:
        """
        初始化 LSTM-SAC 智能体。

        参数：
            config：
                LSTMSACAgentConfig 配置对象。
        """
        # config：保存配置，便于 checkpoint 复现实验。
        self.config = config

        # device：PyTorch 运行设备。
        self.device = torch.device(config.device)

        # state_dim/action_dim/sequence_length：核心输入输出维度。
        self.state_dim = int(config.state_dim)
        self.action_dim = int(config.action_dim)
        self.sequence_length = int(config.sequence_length)

        # gamma/tau：SAC 时序差分目标和软更新参数。
        self.gamma = float(config.gamma)
        self.tau = float(config.tau)

        # automatic_entropy_tuning：是否自动学习熵系数 alpha。
        self.automatic_entropy_tuning = bool(config.automatic_entropy_tuning)

        # actor：LSTM 高斯策略网络。
        self.actor = LSTMActor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            action_low=config.action_low,
            action_high=config.action_high,
            lstm_layer_sizes=config.lstm_layer_sizes,
            fc_hidden_dim=config.actor_fc_hidden_dim,
        ).to(self.device)

        # critic：当前双 Q 网络。
        self.critic = LSTMTwinCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            lstm_layer_sizes=config.lstm_layer_sizes,
            critic_hidden_dims=config.critic_hidden_dims,
        ).to(self.device)

        # target_critic：目标双 Q 网络。
        self.target_critic = LSTMTwinCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            lstm_layer_sizes=config.lstm_layer_sizes,
            critic_hidden_dims=config.critic_hidden_dims,
        ).to(self.device)

        # 初始化时目标 Critic 与当前 Critic 完全一致。
        self.target_critic.load_state_dict(self.critic.state_dict())

        # actor_optimizer/critic_optimizer：论文第 5 章对应的 Adam 优化器。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if self.automatic_entropy_tuning:
            # target_entropy：默认使用 SAC 常用设定 -action_dim。
            self.target_entropy = (
                float(config.target_entropy)
                if config.target_entropy is not None
                else -float(config.action_dim)
            )

            # log_alpha：优化 alpha 的对数，保证 alpha 始终为正。
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

            # alpha_optimizer：熵系数优化器。
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            # target_entropy：固定 alpha 时不参与计算。
            self.target_entropy = None

            # fixed_alpha：固定熵系数，最小值保护避免 log(0)。
            fixed_alpha = max(float(config.initial_alpha), 1e-8)

            # log_alpha：固定 alpha 的对数张量。
            self.log_alpha = torch.tensor(np.log(fixed_alpha), dtype=torch.float32, device=self.device)

            # alpha_optimizer：固定 alpha 模式下没有优化器。
            self.alpha_optimizer = None

    @property
    def alpha(self) -> torch.Tensor:
        """
        当前熵系数 alpha。

        返回：
            alpha：
                正数张量。
        """
        return self.log_alpha.exp()

    def _prepare_single_sequence(self, state_sequence: np.ndarray) -> torch.Tensor:
        """
        将单个 numpy 状态序列转换为带 batch 维度的 torch 张量。

        参数：
            state_sequence：
                单个状态序列，形状为 [sequence_length, state_dim]。

        返回：
            state_sequence_tensor：
                形状为 [1, sequence_length, state_dim] 的张量。
        """
        # sequence_array：转换后的 float32 序列。
        sequence_array = np.asarray(state_sequence, dtype=np.float32)

        if sequence_array.ndim == 1 and self.sequence_length == 1:
            # sequence_length=1 时允许调用方传入单步状态。
            sequence_array = sequence_array.reshape(1, -1)

        expected_shape = (self.sequence_length, self.state_dim)

        if sequence_array.shape != expected_shape:
            raise ValueError(f"state_sequence 形状应为 {expected_shape}，实际为 {sequence_array.shape}")

        # state_sequence_tensor：增加 batch 维度后送入网络。
        state_sequence_tensor = torch.as_tensor(
            sequence_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        return state_sequence_tensor

    def select_action(self, state_sequence: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        根据单个状态序列选择动作。

        参数：
            state_sequence：
                状态序列数组，形状为 [sequence_length, state_dim]。
            deterministic：
                是否使用确定性均值动作。

        返回：
            action：
                NumPy 动作数组，形状为 [action_dim]。
        """
        # state_sequence_tensor：Actor 输入张量，形状为 [1, L, state_dim]。
        state_sequence_tensor = self._prepare_single_sequence(state_sequence)

        self.actor.eval()
        with torch.no_grad():
            # action_tensor：Actor 输出动作，形状为 [1, action_dim]。
            action_tensor = self.actor.act(
                state_sequence=state_sequence_tensor,
                deterministic=deterministic,
            )
        self.actor.train()

        # action：去掉 batch 维度并转回 numpy。
        action = action_tensor.squeeze(0).cpu().numpy()

        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        使用一个 batch 更新 LSTM-SAC 网络。

        参数：
            batch：
                SequenceReplayBuffer.sample() 返回的字典，包含：
                state_sequence, action, reward, next_state_sequence, done。

        返回：
            loss_info：
                当前更新步骤的损失和诊断信息。
        """
        # state_sequence：当前状态序列，形状为 [B, L, state_dim]。
        state_sequence = batch["state_sequence"].to(self.device)

        # action：真实执行的动作，形状为 [B, action_dim]。
        action = batch["action"].to(self.device)

        # reward：奖励，形状为 [B, 1]。
        reward = batch["reward"].to(self.device)

        # next_state_sequence：下一状态序列，形状为 [B, L, state_dim]。
        next_state_sequence = batch["next_state_sequence"].to(self.device)

        # done：终止标志，形状为 [B, 1]。
        done = batch["done"].to(self.device)

        # ============================================================
        # 1. 更新 Critic 双 Q 网络
        # ============================================================

        with torch.no_grad():
            # next_action：下一状态序列下由当前 Actor 采样的动作。
            next_action, next_log_prob, _ = self.actor.sample(
                state_sequence=next_state_sequence,
                deterministic=False,
                with_log_prob=True,
            )

            # target_q1/target_q2：目标 Critic 对下一序列动作的 Q 值。
            target_q1, target_q2 = self.target_critic(next_state_sequence, next_action)

            # target_min_q：双 Q 中较小者，降低过估计。
            target_min_q = torch.min(target_q1, target_q2)

            # target_v：SAC soft value，包含熵奖励修正。
            target_v = target_min_q - self.alpha.detach() * next_log_prob

            # target_q：Bellman 目标，done=1 时截断未来价值。
            target_q = reward + (1.0 - done) * self.gamma * target_v

        # current_q1/current_q2：当前 Critic 对真实动作的估计。
        current_q1, current_q2 = self.critic(state_sequence, action)

        # critic_loss：两个 Q 网络的均方误差。
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ============================================================
        # 2. 更新 Actor 策略网络
        # ============================================================

        # sampled_action/log_prob：当前状态序列下重新采样动作。
        sampled_action, log_prob, _ = self.actor.sample(
            state_sequence=state_sequence,
            deterministic=False,
            with_log_prob=True,
        )

        # q1_pi/q2_pi：Critic 对策略动作的 Q 值估计。
        q1_pi, q2_pi = self.critic(state_sequence, sampled_action)

        # min_q_pi：双 Q 中较小者。
        min_q_pi = torch.min(q1_pi, q2_pi)

        # actor_loss：最大化 Q 同时保留策略熵；优化器最小化所以写成 alpha*log_prob-Q。
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ============================================================
        # 3. 更新 alpha 熵系数
        # ============================================================

        if self.automatic_entropy_tuning:
            # alpha_loss：自动熵调节损失。
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            assert self.alpha_optimizer is not None
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            # alpha_loss：固定 alpha 时统一记为 0。
            alpha_loss = torch.tensor(0.0, device=self.device)

        # ============================================================
        # 4. 软更新 Target Critic
        # ============================================================

        self.soft_update()

        # loss_info：训练器记录用诊断字典。
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

    def update_parameters(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        兼容早期命名的参数更新接口。

        参数：
            batch：
                序列经验 batch。

        返回：
            loss_info：
                update() 返回的训练诊断信息。
        """
        return self.update(batch)

    def soft_update(self) -> None:
        """
        软更新目标 Critic 网络。
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
        导出 LSTM-SAC 智能体完整状态。

        返回：
            state_dict：
                包含网络、优化器、alpha 和配置的字典。
        """
        # result：checkpoint 中保存的智能体状态。
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
        加载 LSTM-SAC 智能体完整状态。

        参数：
            state_dict：
                由 self.state_dict() 导出的字典。
            load_optimizers：
                是否加载优化器状态。
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LSTMSACAgent":
        """
        从普通字典创建 LSTMSACAgent。

        参数：
            config_dict：
                配置字典。

        返回：
            agent：
                LSTMSACAgent 实例。
        """
        # config：转换后的 LSTMSACAgentConfig 对象。
        config = LSTMSACAgentConfig(**config_dict)

        return cls(config)
