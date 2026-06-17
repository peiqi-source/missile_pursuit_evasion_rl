"""
config.py

作用：
    负责读取 YAML 配置文件，并提供项目根目录查找、配置合并等工具函数。

工程说明：
    后续环境参数、SAC 超参数、训练参数都不直接写死在代码中，
    而是统一从 configs/ 目录下的 yaml 文件读取。
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Type, TypeVar

import yaml

T = TypeVar("T")


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    查找项目根目录。

    参数：
        start_path：
            起始查找路径。
            如果为 None，则默认从当前文件所在位置开始向上查找。

    返回：
        project_root：
            项目根目录路径。
            判定依据是目录中存在 pyproject.toml 或 README.md。

    说明：
        这样做可以避免在不同脚本里手动写绝对路径。
    """
    # current_path：当前用于向上查找的路径。
    current_path = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve()

    # 如果传入的是文件路径，则从它的父目录开始查找。
    if current_path.is_file():
        current_path = current_path.parent

    # parent_path：当前路径及其所有父路径。
    for parent_path in [current_path, *current_path.parents]:
        has_pyproject = (parent_path / "pyproject.toml").exists()
        has_readme = (parent_path / "README.md").exists()

        if has_pyproject or has_readme:
            return parent_path

    raise FileNotFoundError("未找到项目根目录，请确认项目中存在 pyproject.toml 或 README.md。")


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    读取 YAML 配置文件。

    参数：
        config_path：
            YAML 配置文件路径，可以是字符串或 Path 对象。

    返回：
        config：
            Python 字典格式的配置内容。

    异常：
        FileNotFoundError：
            当配置文件不存在时抛出。
        ValueError：
            当 YAML 文件内容为空或不是字典时抛出。
    """
    # path：配置文件路径对象。
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        # config：从 YAML 文件中读取到的配置字典。
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"配置文件为空：{path}")

    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是字典结构：{path}")

    return config


def resolve_project_path(relative_path: str | Path) -> Path:
    """
    Resolve a project-relative path while preserving absolute paths.
    """
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return find_project_root() / path


def load_config_from_project(relative_path: str | Path) -> Dict[str, Any]:
    """
    从项目根目录读取配置文件。

    参数：
        relative_path：
            相对于项目根目录的配置文件路径。
            例如：configs/agent/sac.yaml

    返回：
        config：
            YAML 配置字典。
    """
    # config_path：配置文件完整路径。
    config_path = resolve_project_path(relative_path)

    return load_yaml_config(config_path)


def dataclass_field_names(dataclass_type: Type[Any]) -> set[str]:
    """
    Return field names declared by a dataclass type.
    """
    try:
        return {field.name for field in fields(dataclass_type)}
    except TypeError as error:
        raise TypeError(f"{dataclass_type!r} must be a dataclass type.") from error


def filter_dataclass_config(
    config_dict: Mapping[str, Any],
    dataclass_type: Type[Any],
    *,
    allow_extra_keys: Iterable[str] = (),
) -> Dict[str, Any]:
    """
    Keep only fields supported by a dataclass and fail fast on unknown keys.

    Unknown YAML keys used to be silently discarded, which made misspelled or
    misplaced parameters look like they had no effect at runtime.
    """
    valid_field_names = dataclass_field_names(dataclass_type)
    allowed_extra_keys = set(allow_extra_keys)
    unknown_keys = sorted(set(config_dict) - valid_field_names - allowed_extra_keys)

    if unknown_keys:
        allowed_text = ", ".join(sorted(valid_field_names | allowed_extra_keys))
        unknown_text = ", ".join(unknown_keys)
        raise ValueError(
            f"{dataclass_type.__name__} 不支持配置字段：{unknown_text}。"
            f"请检查参数名，或先在对应 dataclass 中添加字段。"
            f"当前可用字段：{allowed_text}"
        )

    return {
        key: value
        for key, value in config_dict.items()
        if key in valid_field_names
    }


def build_dataclass_config(
    config_dict: Mapping[str, Any],
    dataclass_type: Type[T],
    *,
    allow_extra_keys: Iterable[str] = (),
    overrides: Optional[Mapping[str, Any]] = None,
) -> T:
    """
    Build a dataclass config object from YAML data plus runtime overrides.
    """
    merged_config = dict(config_dict)

    if overrides:
        merged_config.update(overrides)

    filtered_config = filter_dataclass_config(
        merged_config,
        dataclass_type,
        allow_extra_keys=allow_extra_keys,
    )

    return dataclass_type(**filtered_config)


def deep_update(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个配置字典。

    参数：
        base_config：
            基础配置字典。

        override_config：
            覆盖配置字典。
            如果与 base_config 存在相同键，则 override_config 的值优先。

    返回：
        merged_config：
            合并后的新配置字典。

    示例：
        base = {"agent": {"lr": 0.001, "gamma": 0.99}}
        override = {"agent": {"lr": 0.0003}}
        结果为 {"agent": {"lr": 0.0003, "gamma": 0.99}}
    """
    # merged_config：最终合并结果，先复制基础配置，避免原地修改输入字典。
    merged_config = dict(base_config)

    for key, override_value in override_config.items():
        base_value = merged_config.get(key)

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged_config[key] = deep_update(base_value, override_value)
        else:
            merged_config[key] = override_value

    return merged_config


def ensure_dir(directory: str | Path) -> Path:
    """
    确保目录存在。

    参数：
        directory：
            需要创建或确认存在的目录路径。

    返回：
        directory_path：
            Path 格式的目录路径。
    """
    # directory_path：目录路径对象。
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path
