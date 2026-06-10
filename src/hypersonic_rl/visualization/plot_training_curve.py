"""
plot_training_curve.py

作用：
    根据训练或评估指标绘制曲线。

当前支持：
    1. 累计奖励曲线；
    2. 最小距离曲线；
    3. 成功率滑动平均曲线；
    4. SAC Critic loss 曲线；
    5. SAC Actor loss 曲线；
    6. Alpha 熵系数曲线；
    7. Q 值曲线；
    8. Log probability 曲线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MetricInput = Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]


def _load_dataframe(data: MetricInput) -> pd.DataFrame:
    """
    将不同格式的指标输入转换为 DataFrame。

    参数：
        data：
            可以是 CSV 路径、DataFrame 或 list[dict]。

    返回：
        dataframe：
            指标表格。
    """
    if isinstance(data, pd.DataFrame):
        dataframe = data.copy()
    elif isinstance(data, (str, Path)):
        dataframe = pd.read_csv(Path(data))
    elif isinstance(data, list):
        dataframe = pd.DataFrame(data)
    else:
        raise TypeError(f"不支持的数据类型：{type(data)}")

    return dataframe


def _require_columns(dataframe: pd.DataFrame, required_columns: set) -> None:
    """
    检查 DataFrame 是否包含必需字段。

    参数：
        dataframe：
            待检查表格。

        required_columns：
            必需字段集合。
    """
    # missing_columns：缺失字段。
    missing_columns = required_columns.difference(set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"指标数据缺少字段：{missing_columns}")


def _plot_single_curve(
    x_values,
    y_values,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> Path:
    """
    绘制单条曲线。

    参数：
        x_values：
            横坐标序列。

        y_values：
            纵坐标序列。

        xlabel：
            横坐标名称。

        ylabel：
            纵坐标名称。

        title：
            图标题。

        save_path：
            图像保存路径。

    返回：
        save_path：
            实际保存路径。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # figure：图像对象。
    figure = plt.figure(figsize=(8, 5))

    plt.plot(x_values, y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(figure)

    return save_path


def plot_training_curves(
    metrics: MetricInput,
    save_dir: str | Path,
    rolling_window: int = 10,
) -> Dict[str, Path]:
    """
    绘制训练 episode 层面的指标曲线。

    参数：
        metrics：
            训练指标输入。

        save_dir：
            图像保存目录。

        rolling_window：
            成功率滑动平均窗口长度。

    返回：
        saved_paths：
            保存的图像路径字典。
    """
    # dataframe：训练指标表。
    dataframe = _load_dataframe(metrics)

    _require_columns(
        dataframe=dataframe,
        required_columns={"episode", "total_reward", "min_distance", "success"},
    )

    # save_directory：图像保存目录。
    save_directory = Path(save_dir)
    save_directory.mkdir(parents=True, exist_ok=True)

    # episode：训练回合编号。
    episode = dataframe["episode"]

    # success_rate_rolling：成功率滑动平均。
    success_rate_rolling = dataframe["success"].rolling(
        window=rolling_window,
        min_periods=1,
    ).mean()

    # saved_paths：保存路径字典。
    saved_paths = {
        "reward_curve": _plot_single_curve(
            x_values=episode,
            y_values=dataframe["total_reward"],
            xlabel="Episode",
            ylabel="Total reward",
            title="Training reward curve",
            save_path=save_directory / "reward_curve.png",
        ),
        "min_distance_curve": _plot_single_curve(
            x_values=episode,
            y_values=dataframe["min_distance"],
            xlabel="Episode",
            ylabel="Minimum distance (m)",
            title="Minimum distance curve",
            save_path=save_directory / "min_distance_curve.png",
        ),
        "success_rate_curve": _plot_single_curve(
            x_values=episode,
            y_values=success_rate_rolling,
            xlabel="Episode",
            ylabel="Rolling success rate",
            title=f"Success rate curve (window={rolling_window})",
            save_path=save_directory / "success_rate_curve.png",
        ),
    }

    if "red_control_energy" in dataframe.columns:
        saved_paths["red_control_energy_curve"] = _plot_single_curve(
            x_values=episode,
            y_values=dataframe["red_control_energy"],
            xlabel="Episode",
            ylabel="Red control energy",
            title="Red control energy curve",
            save_path=save_directory / "red_control_energy_curve.png",
        )

    if "interceptor_control_energy" in dataframe.columns:
        saved_paths["interceptor_control_energy_curve"] = _plot_single_curve(
            x_values=episode,
            y_values=dataframe["interceptor_control_energy"],
            xlabel="Episode",
            ylabel="Interceptor control energy",
            title="Interceptor control energy curve",
            save_path=save_directory / "interceptor_control_energy_curve.png",
        )

    if "interceptor_ny_control_energy" in dataframe.columns:
        saved_paths["interceptor_ny_control_energy_curve"] = _plot_single_curve(
            x_values=episode,
            y_values=dataframe["interceptor_ny_control_energy"],
            xlabel="Episode",
            ylabel="Interceptor ny control energy",
            title="Interceptor longitudinal control energy curve",
            save_path=save_directory / "interceptor_ny_control_energy_curve.png",
        )

    if "interceptor_nz_control_energy" in dataframe.columns:
        saved_paths["interceptor_nz_control_energy_curve"] = _plot_single_curve(
            x_values=episode,
            y_values=dataframe["interceptor_nz_control_energy"],
            xlabel="Episode",
            ylabel="Interceptor nz control energy",
            title="Interceptor lateral control energy curve",
            save_path=save_directory / "interceptor_nz_control_energy_curve.png",
        )

    return saved_paths


def _plot_optional_curve(
    dataframe: pd.DataFrame,
    x_key: str,
    y_key: str,
    save_directory: Path,
    filename: str,
    ylabel: str,
    title: str,
    rolling_window: int,
) -> Path | None:
    """
    绘制可选曲线。

    参数：
        dataframe：
            指标表。

        x_key：
            横坐标字段。

        y_key：
            纵坐标字段。

        save_directory：
            保存目录。

        filename：
            保存文件名。

        ylabel：
            纵坐标名称。

        title：
            图标题。

        rolling_window：
            滑动平均窗口。

    返回：
        saved_path：
            如果字段存在，则返回图像路径；否则返回 None。
    """
    if y_key not in dataframe.columns:
        return None

    # y_values：原始曲线。
    y_values = dataframe[y_key]

    # smoothed_values：滑动平均曲线。
    smoothed_values = y_values.rolling(window=rolling_window, min_periods=1).mean()

    # save_path：图像保存路径。
    save_path = save_directory / filename

    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(8, 5))
    plt.plot(dataframe[x_key], y_values, label="Raw")
    plt.plot(dataframe[x_key], smoothed_values, label=f"Rolling mean ({rolling_window})")
    plt.xlabel("Global step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(figure)

    return save_path


def plot_sac_loss_curves(
    loss_metrics: MetricInput,
    save_dir: str | Path,
    rolling_window: int = 20,
) -> Dict[str, Path]:
    """
    绘制 SAC 更新层面的 loss 和 alpha 曲线。

    参数：
        loss_metrics：
            SAC 更新日志。
            通常来自 loss_metrics.csv 或 trainer.loss_records。

        save_dir：
            图像保存目录。

        rolling_window：
            滑动平均窗口长度。

    返回：
        saved_paths：
            实际保存的图像路径字典。
    """
    # dataframe：loss 指标表。
    dataframe = _load_dataframe(loss_metrics)

    _require_columns(
        dataframe=dataframe,
        required_columns={"global_step", "critic_loss", "actor_loss", "alpha"},
    )

    # save_directory：保存目录。
    save_directory = Path(save_dir)
    save_directory.mkdir(parents=True, exist_ok=True)

    # saved_paths：保存路径字典。
    saved_paths: Dict[str, Path] = {}

    # curve_specs：需要绘制的曲线配置。
    curve_specs = [
        ("critic_loss", "critic_loss_curve.png", "Critic loss", "SAC critic loss curve"),
        ("actor_loss", "actor_loss_curve.png", "Actor loss", "SAC actor loss curve"),
        ("alpha_loss", "alpha_loss_curve.png", "Alpha loss", "SAC alpha loss curve"),
        ("alpha", "alpha_curve.png", "Alpha", "Entropy coefficient alpha curve"),
        ("q1_mean", "q1_mean_curve.png", "Q1 mean", "Q1 mean curve"),
        ("q2_mean", "q2_mean_curve.png", "Q2 mean", "Q2 mean curve"),
        ("target_q_mean", "target_q_mean_curve.png", "Target Q mean", "Target Q mean curve"),
        ("log_prob_mean", "log_prob_mean_curve.png", "Log probability mean", "Log probability curve"),
    ]

    for y_key, filename, ylabel, title in curve_specs:
        # saved_path：当前曲线保存路径。
        saved_path = _plot_optional_curve(
            dataframe=dataframe,
            x_key="global_step",
            y_key=y_key,
            save_directory=save_directory,
            filename=filename,
            ylabel=ylabel,
            title=title,
            rolling_window=rolling_window,
        )

        if saved_path is not None:
            saved_paths[y_key] = saved_path

    return saved_paths
