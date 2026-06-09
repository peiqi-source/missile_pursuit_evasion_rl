"""
logger.py

作用：
    统一创建日志记录器。
    日志同时输出到控制台和文件，便于后续复现实验过程。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from hypersonic_rl.utils.config import ensure_dir


def create_logger(
    logger_name: str,
    log_dir: str | Path,
    log_filename: str = "run.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    创建日志记录器。

    参数：
        logger_name：
            日志器名称。
            建议不同训练任务使用不同名称。

        log_dir：
            日志文件保存目录。

        log_filename：
            日志文件名。

        level：
            日志等级。
            常用 logging.INFO 或 logging.DEBUG。

    返回：
        logger：
            Python logging.Logger 对象。
    """
    # logger：日志记录器对象。
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # 如果 logger 已经有 handler，说明已经创建过，直接返回，避免重复打印。
    if logger.handlers:
        return logger

    # log_directory：日志保存目录。
    log_directory = ensure_dir(log_dir)

    # log_path：日志文件完整路径。
    log_path = log_directory / log_filename

    # formatter：日志格式。
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console_handler：控制台日志输出。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # file_handler：文件日志输出。
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("日志系统初始化完成。")
    logger.info("日志文件路径：%s", log_path)

    return logger


def get_logger(logger_name: Optional[str] = None) -> logging.Logger:
    """
    获取已有 logger。

    参数：
        logger_name：
            logger 名称。
            如果为 None，则返回 root logger。

    返回：
        logger：
            logging.Logger 对象。
    """
    return logging.getLogger(logger_name)