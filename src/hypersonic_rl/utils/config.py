"""配置文件读取工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return dict(data)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典。"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result

