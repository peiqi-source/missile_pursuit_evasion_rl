"""
lstm_sac_trainer.py

作用：
    实现论文第 5 章 LSTM-SAC 训练器。

工程策略：
    训练循环复用 SACTrainer 的成熟实现；
    只覆盖经验写入逻辑，使其写入固定窗口状态序列。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hypersonic_rl.agents import LSTMSACAgent
from hypersonic_rl.buffers import SequenceReplayBuffer
from hypersonic_rl.trainers.sac_trainer import SACTrainer, SACTrainerConfig


@dataclass
class LSTMSACTrainerConfig(SACTrainerConfig):
    """
    LSTMSACTrainerConfig

    作用：
        管理 LSTM-SAC 训练工程参数。

    说明：
        当前字段与 SACTrainerConfig 保持一致，区别在于环境观测为状态序列，
        replay buffer 为 SequenceReplayBuffer，agent 为 LSTMSACAgent。
    """

    experiment_name: str = "lstm_sac_chapter5"


class LSTMSACTrainer(SACTrainer):
    """
    LSTM-SAC 训练器。

    说明：
        StateSequenceWrapper 已经把环境观测转换为 [L, state_dim]。
        因此训练器只需要把 observation/next_observation 作为序列写入 buffer。
    """

    def __init__(
        self,
        env: Any,
        eval_env: Any,
        agent: LSTMSACAgent,
        replay_buffer: SequenceReplayBuffer,
        config: LSTMSACTrainerConfig,
        experiment_dir: str | Path,
        logger: Optional[Any] = None,
    ) -> None:
        """
        初始化 LSTM-SAC 训练器。

        参数：
            env：
                已经经过 StateSequenceWrapper 包装的训练环境。
            eval_env：
                已经经过 StateSequenceWrapper 包装的评估环境。
            agent：
                LSTM-SAC 智能体。
            replay_buffer：
                固定窗口序列经验回放池。
            config：
                训练器配置。
            experiment_dir：
                实验输出目录。
            logger：
                日志对象，可为 None。
        """
        # algorithm_name：先于父类初始化设置，保证 run_manifest 中算法名称准确。
        self.algorithm_name = "LSTM-SAC"

        super().__init__(
            env=env,
            eval_env=eval_env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=config,
            experiment_dir=experiment_dir,
            logger=logger,
        )

        # agent/replay_buffer/config：收窄类型，便于阅读 LSTM-SAC 专用逻辑。
        self.agent: LSTMSACAgent = agent
        self.replay_buffer: SequenceReplayBuffer = replay_buffer
        self.config: LSTMSACTrainerConfig = config

        # run_manifest：父类已按 algorithm_name 写入 LSTM-SAC 运行清单。

    def _store_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        向 SequenceReplayBuffer 存储一条序列 transition。

        参数：
            observation：
                当前状态序列，形状为 [sequence_length, state_dim]。
            action：
                当前动作。
            reward：
                当前奖励。
            next_observation：
                下一状态序列。
            terminated/truncated：
                Gymnasium 终止和截断标志。
        """
        # done_for_buffer：训练时将自然终止和时间截断都视作序列 bootstrap 截断。
        done_for_buffer = bool(terminated or truncated)

        self.replay_buffer.add(
            state_sequence=observation,
            action=action,
            reward=reward,
            next_state_sequence=next_observation,
            done=done_for_buffer,
        )

    def train(self) -> List[Dict[str, Any]]:
        """
        启动 LSTM-SAC 训练。

        返回：
            training_metric_records：
                逐 episode 训练指标列表。
        """
        self._log("启动第 5 章 LSTM-SAC 训练流程。")

        return super().train()
