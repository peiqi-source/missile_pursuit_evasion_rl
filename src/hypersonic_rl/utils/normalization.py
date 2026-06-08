"""归一化工具。"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """在线均值方差估计器，预留给观测归一化使用。"""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        """使用新 batch 更新均值和方差。"""
        values = np.asarray(values, dtype=np.float64)
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """按当前均值和标准差归一化。"""
        return (values - self.mean) / np.sqrt(self.var + epsilon)

