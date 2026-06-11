"""
manifest.py

作用：
    为训练、评估和 benchmark 输出可复现运行清单。

设计原则：
    1. 不改变训练和评估主流程；
    2. 只采集轻量元信息，例如配置快照、seed、checkpoint、git 摘要；
    3. 统一保存为 JSON，便于论文复现实验归档和后续脚本读取。
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def to_jsonable(value: Any) -> Any:
    """
    把常见 Python/NumPy/dataclass 对象转换为 JSON 可序列化对象。

    参数：
        value：
            任意待转换对象。

    返回：
        jsonable：
            JSON 可序列化对象。
    """
    # dataclass：优先转换为字典，保留字段名。
    if is_dataclass(value):
        return to_jsonable(asdict(value))

    # Path：保存为字符串路径。
    if isinstance(value, Path):
        return str(value)

    # numpy 标量：转换为 Python 标量。
    if isinstance(value, np.generic):
        return value.item()

    # numpy 数组：转换为列表。
    if isinstance(value, np.ndarray):
        return value.tolist()

    # dict：递归转换键值。
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    # list/tuple：递归转换元素。
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    # 基础类型：直接返回。
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # 其他对象：优先使用 __dict__，否则退化为字符串。
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))

    return str(value)


def _run_git_command(project_root: Path, args: list[str]) -> str:
    """
    执行只读 git 命令并返回文本输出。

    参数：
        project_root：
            项目根目录。
        args：
            git 子命令参数。

    返回：
        output：
            命令标准输出；失败时返回 "unknown"。
    """
    try:
        # result：运行只读 git 命令，避免影响工作区。
        result = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    return result.stdout.strip()


def collect_git_summary(project_root: str | Path) -> Dict[str, Any]:
    """
    采集当前工作区的 git 摘要。

    参数：
        project_root：
            项目根目录。

    返回：
        git_summary：
            包含 commit、branch、dirty 状态和短状态文本的字典。
    """
    # root：标准化项目根目录。
    root = Path(project_root)

    # status_short：短状态文本，用于判断实验是否来自脏工作区。
    status_short = _run_git_command(root, ["status", "--short"])

    return {
        "commit": _run_git_command(root, ["rev-parse", "HEAD"]),
        "branch": _run_git_command(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": status_short,
        "is_dirty": bool(status_short and status_short != "unknown"),
    }


def build_run_manifest(
    run_name: str,
    project_root: str | Path,
    env_config: Optional[Any] = None,
    agent_config: Optional[Any] = None,
    trainer_config: Optional[Any] = None,
    checkpoint_path: Optional[str | Path] = None,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构造统一运行清单。

    参数：
        run_name：
            当前运行名称。
        project_root：
            项目根目录。
        env_config：
            环境配置快照。
        agent_config：
            智能体配置快照。
        trainer_config：
            训练器或评估器配置快照。
        checkpoint_path：
            关联 checkpoint 路径。
        seed：
            基础随机种子。
        extra：
            额外上下文字段。

    返回：
        manifest：
            JSON 可序列化运行清单。
    """
    # root：标准化项目根目录。
    root = Path(project_root)

    # manifest：统一运行清单主体。
    manifest: Dict[str, Any] = {
        "run_name": str(run_name),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "seed": None if seed is None else int(seed),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "git": collect_git_summary(root),
        "env_config": to_jsonable(env_config),
        "agent_config": to_jsonable(agent_config),
        "trainer_config": to_jsonable(trainer_config),
        "extra": to_jsonable(extra or {}),
    }

    return manifest


def save_run_manifest(manifest: Dict[str, Any], output_path: str | Path) -> Path:
    """
    保存运行清单为 JSON 文件。

    参数：
        manifest：
            运行清单字典。
        output_path：
            输出 JSON 路径。

    返回：
        saved_path：
            实际保存路径。
    """
    # saved_path：标准化输出路径并创建父目录。
    saved_path = Path(output_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入 UTF-8 JSON，ensure_ascii=False 便于中文配置和注释可读。
    saved_path.write_text(
        json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return saved_path
