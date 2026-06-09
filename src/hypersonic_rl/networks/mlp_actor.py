"""
mlp_actor.py

作用：
    实现 SAC 的 MLP 高斯策略 Actor 网络。

SAC Actor 的核心逻辑：
    1. 输入状态 state；
    2. 输出动作分布的均值 mean 和标准差 std；
    3. 使用重参数化技巧采样动作；
    4. 使用 tanh 将动作限制在 [-1, 1]；
    5. 再缩放到环境真实动作范围 [action_low, action_high]；
    6. 计算 tanh 修正后的 log_prob，用于 SAC 策略损失。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


class MLPActor(nn.Module):
    """
    SAC 高斯策略网络。

    输入：
        state，形状为 [batch_size, state_dim]

    输出：
        action，形状为 [batch_size, action_dim]
        log_prob，形状为 [batch_size, 1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_low: float,
        action_high: float,
    ) -> None:
        """
        初始化 MLP Actor。

        参数：
            state_dim：
                状态维度。

            action_dim：
                动作维度。

            hidden_dim：
                隐藏层神经元数量。

            action_low：
                环境动作下界。当前环境中通常为 -nzc_h_max。

            action_high：
                环境动作上界。当前环境中通常为 nzc_h_max。
        """
        super().__init__()

        # state_dim：状态向量维度。
        self.state_dim = int(state_dim)

        # action_dim：动作向量维度。
        self.action_dim = int(action_dim)

        # hidden_dim：隐藏层宽度。
        self.hidden_dim = int(hidden_dim)

        if self.state_dim <= 0:
            raise ValueError(f"state_dim 必须大于 0，当前为：{self.state_dim}")

        if self.action_dim <= 0:
            raise ValueError(f"action_dim 必须大于 0，当前为：{self.action_dim}")

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须大于 0，当前为：{self.hidden_dim}")

        # feature_layer_1：第一层状态特征提取网络。
        self.feature_layer_1 = nn.Linear(self.state_dim, self.hidden_dim)

        # feature_layer_2：第二层状态特征提取网络。
        self.feature_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # mean_layer：输出高斯分布均值。
        self.mean_layer = nn.Linear(self.hidden_dim, self.action_dim)

        # log_std_layer：输出高斯分布标准差的对数。
        self.log_std_layer = nn.Linear(self.hidden_dim, self.action_dim)

        # action_scale：tanh 动作从 [-1, 1] 映射到真实动作范围所需的缩放系数。
        action_scale = (float(action_high) - float(action_low)) / 2.0

        # action_bias：tanh 动作从 [-1, 1] 映射到真实动作范围所需的平移系数。
        action_bias = (float(action_high) + float(action_low)) / 2.0

        # register_buffer：将动作缩放参数注册为模型 buffer，随模型移动到 CPU/GPU，但不参与训练。
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据状态输出高斯分布参数。

        参数：
            state：
                当前状态张量，形状为 [batch_size, state_dim]。

        返回：
            mean：
                动作分布均值，形状为 [batch_size, action_dim]。

            log_std：
                动作分布标准差对数，形状为 [batch_size, action_dim]。
        """
        # hidden：状态经过第一层后的隐藏特征。
        hidden = F.relu(self.feature_layer_1(state))

        # hidden：状态经过第二层后的隐藏特征。
        hidden = F.relu(self.feature_layer_2(hidden))

        # mean：高斯策略均值。
        mean = self.mean_layer(hidden)

        # log_std：高斯策略标准差对数。
        log_std = self.log_std_layer(hidden)

        # log_std：裁剪后避免标准差过大或过小。
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        根据状态采样动作。

        参数：
            state：
                当前状态张量，形状为 [batch_size, state_dim]。

            deterministic：
                是否使用确定性动作。
                训练时通常为 False，测试时通常为 True。

            with_log_prob：
                是否计算 log_prob。
                训练 Actor 时需要 True，测试时可为 False。

        返回：
            action：
                缩放到环境真实动作范围后的动作。

            log_prob：
                动作的对数概率。
                如果 with_log_prob=False，则返回 None。

            mean_action：
                确定性均值动作，已经经过 tanh 和动作缩放。
        """
        # mean：高斯分布均值。
        mean, log_std = self.forward(state)

        # std：高斯分布标准差。
        std = log_std.exp()

        # normal_dist：根据 mean 和 std 构造的正态分布。
        normal_dist = torch.distributions.Normal(mean, std)

        if deterministic:
            # raw_action：确定性模式下直接使用均值。
            raw_action = mean
        else:
            # raw_action：重参数化采样结果，允许梯度从动作回传到策略网络。
            raw_action = normal_dist.rsample()

        # tanh_action：压缩到 [-1, 1] 的动作。
        tanh_action = torch.tanh(raw_action)

        # action：映射到环境真实动作范围后的动作。
        action = tanh_action * self.action_scale + self.action_bias

        # mean_action：均值动作经过 tanh 和真实范围缩放后的结果。
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        if not with_log_prob:
            return action, None, mean_action

        # log_prob：原始高斯分布下 raw_action 的对数概率。
        log_prob = normal_dist.log_prob(raw_action)

        # log_prob：tanh 变换的概率密度修正项。
        log_prob = log_prob - torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + EPS)

        # log_prob：多维动作时对动作维度求和，得到 [batch_size, 1]。
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean_action

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        给外部调用的动作选择接口。

        参数：
            state：
                单个状态或 batch 状态张量。

            deterministic：
                是否使用确定性动作。

        返回：
            action：
                环境真实动作范围内的动作张量。
        """
        # action：采样得到的动作。
        action, _, _ = self.sample(state, deterministic=deterministic, with_log_prob=False)

        return action