"""
lstm_actor.py

作用：
    为论文第 5 章 LSTM-SAC 预留 LSTM Actor 网络。

后续目标：
    输入连续状态序列：
        [s_{t-L+1}, ..., s_t]

    输出当前时刻横向过载动作：
        a_t
"""


class LSTMActor:
    """
    LSTM Actor 预留类。

    当前阶段先不实现，避免普通 SAC 重构阶段引入过多复杂度。
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化 LSTM Actor 占位对象。

        参数：
            *args：
                预留位置参数。
            **kwargs：
                预留关键字参数。

        返回：
            None。当前阶段直接抛出 NotImplementedError。
        """
        # error_message：明确提示该网络尚未进入实现阶段。
        error_message = "LSTMActor 将在第 5 章 LSTM-SAC 阶段实现。"

        raise NotImplementedError(error_message)
