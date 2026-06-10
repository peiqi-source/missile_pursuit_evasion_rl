"""
lstm_critic.py

作用：
    为论文第 5 章 LSTM-SAC 预留 LSTM Critic 网络。

后续目标：
    输入连续状态序列和当前动作：
        [s_{t-L+1}, ..., s_t], a_t

    输出动作价值：
        Q(s_sequence, a_t)
"""


class LSTMCritic:
    """
    LSTM Critic 预留类。

    当前阶段先不实现，避免普通 SAC 重构阶段引入过多复杂度。
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化 LSTM Critic 占位对象。

        参数：
            *args：
                预留位置参数。
            **kwargs：
                预留关键字参数。

        返回：
            None。当前阶段直接抛出 NotImplementedError。
        """
        # error_message：明确提示该网络尚未进入实现阶段。
        error_message = "LSTMCritic 将在第 5 章 LSTM-SAC 阶段实现。"

        raise NotImplementedError(error_message)
