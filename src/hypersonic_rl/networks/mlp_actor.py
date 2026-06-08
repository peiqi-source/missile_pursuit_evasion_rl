"""SAC 使用的 tanh-squashed Gaussian Actor 网络。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from hypersonic_rl.networks.init_weights import init_mlp

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    """高斯策略网络，输出经 tanh 压缩并映射到动作空间的动作。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_scale: np.ndarray | float = 1.0,
        action_bias: np.ndarray | float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # mean_linear：动作分布均值输出层。
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        # log_std_linear：动作分布标准差对数输出层。
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        init_mlp(self)

        # action_scale：动作缩放系数，将 tanh 输出映射到环境动作范围。
        scale = torch.as_tensor(action_scale, dtype=torch.float32).reshape(1, action_dim)
        # action_bias：动作平移系数，将 tanh 输出映射到环境动作中心。
        bias = torch.as_tensor(action_bias, dtype=torch.float32).reshape(1, action_dim)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def _distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Normal]:
        """根据状态构造高斯动作分布。"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # mean：动作分布均值。
        mean = self.mean_linear(x)
        # log_std：动作分布标准差的对数。
        log_std = torch.clamp(self.log_std_linear(x), LOG_STD_MIN, LOG_STD_MAX)
        # std：动作分布标准差。
        std = log_std.exp()
        return mean, log_std, Normal(mean, std)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """输出动作、动作对数概率和均值动作。

        参数:
            state: 状态批量张量，形状为 [batch, state_dim]。
            deterministic: 是否使用均值动作，评估时通常为 True。

        返回:
            action: 映射到环境动作范围后的动作。
            log_prob: 当前动作的对数概率，用于 SAC 熵正则项。
            mean_action: 均值动作映射到环境动作范围后的结果。
        """
        mean, _, normal = self._distribution(state)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = normal.rsample()
        tanh_action = torch.tanh(pre_tanh)
        action = tanh_action * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        if deterministic:
            return action, None, mean_action

        normal_log_prob = normal.log_prob(pre_tanh)
        # log_prob：加入 tanh 变换修正项后的动作对数概率。
        log_prob = normal_log_prob - torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean_action

    def sample(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """兼容常见 SAC 实现的采样接口。"""
        return self.forward(state, deterministic=deterministic)


class MLPActor(GaussianPolicy):
    """普通 MLP Actor 别名，便于配置中区分后续 LSTM Actor。"""

