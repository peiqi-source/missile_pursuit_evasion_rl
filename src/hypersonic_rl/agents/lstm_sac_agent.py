"""第 5 章 LSTM-SAC 智能体预留。"""

from __future__ import annotations

from typing import Any


class LSTMSACAgent:
    """LSTM-SAC 智能体骨架。

    后续将在普通 SACAgent 的基础上替换 Actor/Critic 为 LSTM 结构，并使用序列回放缓存。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化 LSTM-SAC 智能体占位对象。

        参数：
            *args：
                预留位置参数。
            **kwargs：
                预留关键字参数。

        返回：
            None。
        """
        # args：暂存调用方传入的位置参数，便于后续实现时保持接口兼容。
        self.args = args

        # kwargs：暂存调用方传入的关键字参数。
        self.kwargs = kwargs

    def select_action(self, *args: Any, **kwargs: Any) -> Any:
        """预留 LSTM 状态序列选动作接口。"""
        raise NotImplementedError("LSTMSACAgent 将在论文第 5 章复现阶段实现。")

    def update_parameters(self, *args: Any, **kwargs: Any) -> Any:
        """预留 LSTM-SAC 参数更新接口。"""
        raise NotImplementedError("LSTMSACAgent 将在论文第 5 章复现阶段实现。")
