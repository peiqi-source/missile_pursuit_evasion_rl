"""
normalization.py

作用：
    提供状态归一化、动作缩放等工具函数。

说明：
    强化学习训练中，不同状态变量的量纲差异会影响网络训练稳定性。
    例如位置单位是米，角度单位是弧度，速度单位是 m/s。
    因此后续可以在环境或 wrapper 中调用这些工具。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clip_array(array: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """
    对数组进行裁剪。

    参数：
        array：
            输入数组。

        min_value：
            最小允许值。

        max_value：
            最大允许值。

    返回：
        clipped_array：
            裁剪后的数组。
    """
    # clipped_array：裁剪后的数组。
    clipped_array = np.clip(array, min_value, max_value)
    return clipped_array


def normalize_to_range(
    value: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    target_min: float = -1.0,
    target_max: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    将数值从原始范围线性映射到指定范围。

    参数：
        value：
            原始数值数组。

        lower_bound：
            原始范围下界。

        upper_bound：
            原始范围上界。

        target_min：
            目标范围下界，默认 -1。

        target_max：
            目标范围上界，默认 1。

        eps：
            防止除零的小常数。

    返回：
        normalized_value：
            归一化后的数值数组。
    """
    # value_array：转换为 numpy 数组后的输入值。
    value_array = np.asarray(value, dtype=np.float32)

    # lower_array：原始范围下界数组。
    lower_array = np.asarray(lower_bound, dtype=np.float32)

    # upper_array：原始范围上界数组。
    upper_array = np.asarray(upper_bound, dtype=np.float32)

    # scale：原始范围跨度。
    scale = upper_array - lower_array

    # normalized_01：映射到 [0, 1] 的中间结果。
    normalized_01 = (value_array - lower_array) / (scale + eps)

    # normalized_value：映射到 [target_min, target_max] 的最终结果。
    normalized_value = normalized_01 * (target_max - target_min) + target_min

    return normalized_value.astype(np.float32)


def denormalize_from_range(
    normalized_value: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    target_min: float = -1.0,
    target_max: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    将归一化数值还原到原始范围。

    参数：
        normalized_value：
            归一化后的数值数组。

        lower_bound：
            原始范围下界。

        upper_bound：
            原始范围上界。

        target_min：
            归一化范围下界。

        target_max：
            归一化范围上界。

        eps：
            防止除零的小常数。

    返回：
        original_value：
            还原后的原始数值数组。
    """
    # normalized_array：转换为 numpy 数组后的归一化输入。
    normalized_array = np.asarray(normalized_value, dtype=np.float32)

    # lower_array：原始范围下界数组。
    lower_array = np.asarray(lower_bound, dtype=np.float32)

    # upper_array：原始范围上界数组。
    upper_array = np.asarray(upper_bound, dtype=np.float32)

    # normalized_01：从 [target_min, target_max] 映射回 [0, 1]。
    normalized_01 = (normalized_array - target_min) / (target_max - target_min + eps)

    # original_value：从 [0, 1] 映射回原始范围。
    original_value = normalized_01 * (upper_array - lower_array) + lower_array

    return original_value.astype(np.float32)


@dataclass
class RunningMeanStd:
    """
    运行均值方差统计器。

    用途：
        后续可用于在线状态标准化。
        例如将状态转换为：
        normalized_state = (state - mean) / sqrt(var)

    属性：
        shape：
            需要统计的变量形状。

        eps：
            初始计数，避免方差除零。
    """

    shape: tuple
    eps: float = 1e-4

    def __post_init__(self) -> None:
        """
        初始化均值、方差和样本计数。
        """
        # mean：运行均值。
        self.mean = np.zeros(self.shape, dtype=np.float64)

        # var：运行方差。
        self.var = np.ones(self.shape, dtype=np.float64)

        # count：已经统计的样本数量。
        self.count = self.eps

    def update(self, batch: np.ndarray) -> None:
        """
        使用一个 batch 更新运行均值和方差。

        参数：
            batch：
                输入样本数组，第一维通常是 batch 维度。
        """
        # batch_array：转换后的 batch 数组。
        batch_array = np.asarray(batch, dtype=np.float64)

        # batch_mean：当前 batch 均值。
        batch_mean = np.mean(batch_array, axis=0)

        # batch_var：当前 batch 方差。
        batch_var = np.var(batch_array, axis=0)

        # batch_count：当前 batch 样本数量。
        batch_count = batch_array.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, value: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """
        根据运行均值方差对输入进行标准化。

        参数：
            value：
                待标准化数组。

            clip：
                标准化结果裁剪范围。

        返回：
            normalized_value：
                标准化后的数组。
        """
        # value_array：输入数组。
        value_array = np.asarray(value, dtype=np.float32)

        # normalized_value：标准化结果。
        normalized_value = (value_array - self.mean) / np.sqrt(self.var + 1e-8)

        return np.clip(normalized_value, -clip, clip).astype(np.float32)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """
        根据 batch 的均值、方差和数量更新全局统计量。

        参数：
            batch_mean：
                当前 batch 均值。

            batch_var：
                当前 batch 方差。

            batch_count：
                当前 batch 样本数量。
        """
        # delta：batch 均值与历史均值的差。
        delta = batch_mean - self.mean

        # total_count：更新后的总样本数。
        total_count = self.count + batch_count

        # new_mean：更新后的均值。
        new_mean = self.mean + delta * batch_count / total_count

        # m_a：历史方差对应的平方和。
        m_a = self.var * self.count

        # m_b：当前 batch 方差对应的平方和。
        m_b = batch_var * batch_count

        # m_2：合并后的平方和。
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        # new_var：更新后的方差。
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count