"""第 5 章 LSTM-SAC 训练器预留。"""

from __future__ import annotations

from typing import Any


class LSTMSACTrainer:
    """LSTM-SAC 训练器骨架。

    后续将组合 StateSequenceWrapper、SequenceReplayBuffer、LSTMSACAgent 完成端到端训练。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def train(self) -> None:
        """预留训练入口。"""
        raise NotImplementedError("LSTMSACTrainer 将在论文第 5 章复现阶段实现。")

