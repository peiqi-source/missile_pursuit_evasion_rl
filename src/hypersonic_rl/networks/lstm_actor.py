"""
lstm_actor.py

作用：
    实现论文第 5 章 LSTM-SAC 使用的 Actor 策略网络。

网络输入：
    state_sequence，形状为 [batch_size, sequence_length, state_dim]。

网络输出：
    当前时刻动作分布，并通过 tanh 将动作限制到环境真实动作范围。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


class LSTMActor(nn.Module):
    """
    LSTM-SAC 高斯策略网络。

    说明：
        论文第 5 章 Actor 使用 3 层单向 LSTM 和 1 层全连接层。
        本实现允许通过 lstm_layer_sizes 配置不同层宽，默认对应 256 -> 256 -> 128。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        lstm_layer_sizes: Sequence[int] = (256, 256, 128),
        fc_hidden_dim: int = 128,
    ) -> None:
        """
        初始化 LSTM Actor。

        参数：
            state_dim：
                单个时刻状态维度，论文第 5 章默认为 10。
            action_dim：
                动作维度，当前端到端突防任务为 1。
            action_low/action_high：
                环境动作上下界。
            lstm_layer_sizes：
                每一层单向 LSTM 的隐藏状态维度。
            fc_hidden_dim：
                LSTM 时序特征后的全连接层宽度。
        """
        super().__init__()

        # state_dim：单个状态向量维度。
        self.state_dim = int(state_dim)

        # action_dim：动作向量维度。
        self.action_dim = int(action_dim)

        # lstm_layer_sizes：论文式多层 LSTM 隐藏宽度。
        self.lstm_layer_sizes = tuple(int(size) for size in lstm_layer_sizes)

        # fc_hidden_dim：策略头全连接层宽度。
        self.fc_hidden_dim = int(fc_hidden_dim)

        if self.state_dim <= 0:
            raise ValueError(f"state_dim 必须大于 0，当前为：{self.state_dim}")

        if self.action_dim <= 0:
            raise ValueError(f"action_dim 必须大于 0，当前为：{self.action_dim}")

        if not self.lstm_layer_sizes or any(size <= 0 for size in self.lstm_layer_sizes):
            raise ValueError(f"lstm_layer_sizes 必须为正整数序列，当前为：{self.lstm_layer_sizes}")

        if self.fc_hidden_dim <= 0:
            raise ValueError(f"fc_hidden_dim 必须大于 0，当前为：{self.fc_hidden_dim}")

        # recurrent_layers：逐层保存单向 LSTM；使用 ModuleList 支持每层不同 hidden_size。
        self.recurrent_layers = nn.ModuleList()

        # input_dim：当前 LSTM 层输入维度，第一层接收原始状态。
        input_dim = self.state_dim

        for hidden_dim in self.lstm_layer_sizes:
            # lstm_layer：单层单向 LSTM，batch_first=True 对应 [B, L, S] 输入。
            lstm_layer = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.recurrent_layers.append(lstm_layer)
            input_dim = hidden_dim

        # feature_layer：将最后一个时刻的 LSTM 特征映射到策略头隐藏空间。
        self.feature_layer = nn.Linear(self.lstm_layer_sizes[-1], self.fc_hidden_dim)

        # mean_layer：输出高斯策略均值。
        self.mean_layer = nn.Linear(self.fc_hidden_dim, self.action_dim)

        # log_std_layer：输出高斯策略标准差对数。
        self.log_std_layer = nn.Linear(self.fc_hidden_dim, self.action_dim)

        # action_scale/action_bias：tanh 动作到真实动作范围的仿射映射参数。
        action_scale = (float(action_high) - float(action_low)) / 2.0
        action_bias = (float(action_high) + float(action_low)) / 2.0

        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def _encode_sequence(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        将状态序列编码为最后时刻的时序特征。

        参数：
            state_sequence：
                状态序列张量，形状为 [batch_size, sequence_length, state_dim]。

        返回：
            encoded_feature：
                最后一个时间步的 LSTM 特征，形状为 [batch_size, last_lstm_hidden_dim]。
        """
        if state_sequence.ndim != 3:
            raise ValueError(
                "state_sequence 必须为三维张量 "
                f"[batch_size, sequence_length, state_dim]，实际为：{tuple(state_sequence.shape)}"
            )

        if state_sequence.shape[-1] != self.state_dim:
            raise ValueError(
                f"state_sequence 最后一维应为 {self.state_dim}，实际为 {state_sequence.shape[-1]}"
            )

        # hidden_sequence：逐层传递的序列特征。
        hidden_sequence = state_sequence

        for recurrent_layer in self.recurrent_layers:
            # hidden_sequence：当前 LSTM 层输出的完整时间序列特征。
            hidden_sequence, _ = recurrent_layer(hidden_sequence)

            # ReLU：与论文表 5-1 的隐藏层激活函数保持一致。
            hidden_sequence = F.relu(hidden_sequence)

        # encoded_feature：取最后一个状态对应的时序特征用于当前动作决策。
        encoded_feature = hidden_sequence[:, -1, :]

        return encoded_feature

    def forward(self, state_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据状态序列输出高斯动作分布参数。

        参数：
            state_sequence：
                状态序列张量，形状为 [batch_size, sequence_length, state_dim]。

        返回：
            mean：
                动作分布均值，形状为 [batch_size, action_dim]。
            log_std：
                动作分布标准差对数，形状为 [batch_size, action_dim]。
        """
        # sequence_feature：LSTM 提取的历史态势特征。
        sequence_feature = self._encode_sequence(state_sequence)

        # policy_feature：全连接层后的策略特征。
        policy_feature = F.relu(self.feature_layer(sequence_feature))

        # mean/log_std：高斯策略分布参数。
        mean = self.mean_layer(policy_feature)
        log_std = self.log_std_layer(policy_feature)

        # log_std：裁剪避免方差过大或过小造成数值不稳定。
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        return mean, log_std

    def sample(
        self,
        state_sequence: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        从策略分布中采样动作。

        参数：
            state_sequence：
                状态序列张量，形状为 [batch_size, sequence_length, state_dim]。
            deterministic：
                是否使用确定性均值动作。
            with_log_prob：
                是否计算 tanh 修正后的动作 log_prob。

        返回：
            action：
                映射到环境真实范围后的动作。
            log_prob：
                动作对数概率；如果 with_log_prob=False，则为 None。
            mean_action：
                确定性均值动作。
        """
        # mean/log_std：当前状态序列下的高斯分布参数。
        mean, log_std = self.forward(state_sequence)

        # std：高斯分布标准差。
        std = log_std.exp()

        # normal_dist：重参数采样使用的正态分布。
        normal_dist = torch.distributions.Normal(mean, std)

        if deterministic:
            # raw_action：评估时直接使用均值动作。
            raw_action = mean
        else:
            # raw_action：训练时使用可回传梯度的 rsample。
            raw_action = normal_dist.rsample()

        # tanh_action：压缩到 [-1, 1] 的动作。
        tanh_action = torch.tanh(raw_action)

        # action：映射到环境真实动作范围。
        action = tanh_action * self.action_scale + self.action_bias

        # mean_action：用于 deterministic select_action 的动作。
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        if not with_log_prob:
            return action, None, mean_action

        # log_prob：原始高斯分布下 raw_action 的对数概率。
        log_prob = normal_dist.log_prob(raw_action)

        # log_prob：加入 tanh 和动作缩放的概率密度修正。
        log_prob = log_prob - torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + EPS)

        # log_prob：多维动作对动作维求和，得到 [batch_size, 1]。
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean_action

    def act(self, state_sequence: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        对外动作选择接口。

        参数：
            state_sequence：
                单个或 batch 状态序列张量。
            deterministic：
                是否使用确定性动作。

        返回：
            action：
                环境真实动作范围内的动作张量。
        """
        # action：采样或均值得到的动作。
        action, _, _ = self.sample(
            state_sequence=state_sequence,
            deterministic=deterministic,
            with_log_prob=False,
        )

        return action
