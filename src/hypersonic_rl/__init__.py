"""高超声速追逃强化学习工程包。"""

from __future__ import annotations

import os

# KMP_DUPLICATE_LIB_OK：兼容 Windows/Anaconda 下 torch 与 numpy/matplotlib 的 OpenMP 重复加载。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
