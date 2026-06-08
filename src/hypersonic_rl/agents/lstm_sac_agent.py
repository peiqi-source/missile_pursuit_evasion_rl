"""第 5 章 LSTM-SAC 智能体预留。"""

from __future__ import annotations

from typing import Any


class LSTMSACAgent:
    """LSTM-SAC 智能体骨架。

    后续将在普通 SACAgent 的基础上替换 Actor/Critic 为 LSTM 结构，并使用序列回放缓存。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def select_action(self, *args: Any, **kwargs: Any) -> Any:
        """预留 LSTM 状态序列选动作接口。"""
        raise NotImplementedError("LSTMSACAgent 将在论文第 5 章复现阶段实现。")

    def update_parameters(self, *args: Any, **kwargs: Any) -> Any:
        """预留 LSTM-SAC 参数更新接口。"""
        raise NotImplementedError("LSTMSACAgent 将在论文第 5 章复现阶段实现。")

