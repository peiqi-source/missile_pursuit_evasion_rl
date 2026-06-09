"""
evaluator.py

作用：
    实现模型评估器。

评估器职责：
    1. 使用训练好的 agent 与环境交互；
    2. 统计每个测试回合的突防指标；
    3. 可选保存 episode 综合图、三视图轨迹图和三维轨迹图；
    4. 输出多回合平均性能。

工程说明：
    Evaluator 不关心 agent 内部是 SAC 还是 LSTM-SAC。
    只要求 agent 提供 select_action(state, deterministic=True/False) 接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hypersonic_rl.evaluation.metrics import (
    collect_episode_metrics,
    save_metrics_to_csv,
    summarize_metrics,
)
from hypersonic_rl.visualization import (
    plot_3d_trajectory,
    plot_episode_summary,
    plot_full_trajectory_summary,
    plot_three_view_trajectory,
    plot_trajectory,
)


@dataclass
class EvaluatorConfig:
    """
    EvaluatorConfig

    作用：
        管理模型评估参数。

    属性：
        eval_episodes：
            测试回合数。

        deterministic：
            是否使用确定性动作。
            测试时通常为 True，表示使用 Actor 均值动作。

        render：
            是否每一步打印环境状态。

        save_episode_plots：
            是否保存单回合综合诊断图。

        save_single_view_trajectory：
            是否保存单独的 X-Z、X-Y、Y-Z 三张投影图。

        save_three_view_trajectory：
            是否保存三视图组合图。

        save_3d_trajectory：
            是否保存三维空间轨迹图。

        save_full_trajectory_summary：
            是否保存“三视图 + 三维轨迹”综合图。

        output_dir：
            评估结果保存目录。
    """

    eval_episodes: int = 10
    deterministic: bool = True
    render: bool = False
    save_episode_plots: bool = True
    save_single_view_trajectory: bool = False
    save_three_view_trajectory: bool = True
    save_3d_trajectory: bool = True
    save_full_trajectory_summary: bool = False
    output_dir: str = "outputs/evaluation"


class Evaluator:
    """
    强化学习策略评估器。
    """

    def __init__(
        self,
        env: Any,
        agent: Any,
        config: Optional[EvaluatorConfig] = None,
        logger: Optional[Any] = None,
    ) -> None:
        """
        初始化评估器。

        参数：
            env：
                强化学习环境对象。

            agent：
                智能体对象，需要提供 select_action() 方法。

            config：
                EvaluatorConfig 配置对象。

            logger：
                日志对象，可为 None。
        """
        # env：环境对象。
        self.env = env

        # agent：智能体对象。
        self.agent = agent

        # config：评估配置。
        self.config = config if config is not None else EvaluatorConfig()

        # logger：日志对象。
        self.logger = logger

        # output_dir：评估输出目录。
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # figure_dir：图像保存目录。
        self.figure_dir = self.output_dir / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # metrics_dir：指标保存目录。
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        """
        输出日志信息。

        参数：
            message：
                待输出文本。
        """
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _save_episode_figures(self, episode_index: int, metrics: Dict[str, Any]) -> None:
        """
        保存当前测试回合的图像。

        参数：
            episode_index：
                当前测试回合编号。

            metrics：
                当前回合指标字典。
                图像路径会被写入该字典中。
        """
        # episode_prefix：当前回合图像文件名前缀。
        episode_prefix = f"eval_episode_{episode_index:04d}"

        if self.config.save_episode_plots:
            # episode_summary_path：单回合综合诊断图路径。
            episode_summary_path = self.figure_dir / f"{episode_prefix}_summary.png"

            plot_episode_summary(
                env=self.env,
                save_path=episode_summary_path,
                title=f"Evaluation Episode {episode_index}",
                show=False,
            )

            metrics["episode_summary_path"] = str(episode_summary_path)

        if self.config.save_single_view_trajectory:
            # xz_path/xy_path/yz_path：三张单独投影图路径。
            xz_path = self.figure_dir / f"{episode_prefix}_trajectory_xz.png"
            xy_path = self.figure_dir / f"{episode_prefix}_trajectory_xy.png"
            yz_path = self.figure_dir / f"{episode_prefix}_trajectory_yz.png"

            plot_trajectory(self.env, xz_path, plane="xz", show=False)
            plot_trajectory(self.env, xy_path, plane="xy", show=False)
            plot_trajectory(self.env, yz_path, plane="yz", show=False)

            metrics["trajectory_xz_path"] = str(xz_path)
            metrics["trajectory_xy_path"] = str(xy_path)
            metrics["trajectory_yz_path"] = str(yz_path)

        if self.config.save_three_view_trajectory:
            # three_view_path：三视图组合图路径。
            three_view_path = self.figure_dir / f"{episode_prefix}_three_views.png"

            plot_three_view_trajectory(
                env=self.env,
                save_path=three_view_path,
                show=False,
            )

            metrics["three_view_trajectory_path"] = str(three_view_path)

        if self.config.save_3d_trajectory:
            # trajectory_3d_path：三维轨迹图路径。
            trajectory_3d_path = self.figure_dir / f"{episode_prefix}_trajectory_3d.png"

            plot_3d_trajectory(
                env=self.env,
                save_path=trajectory_3d_path,
                show=False,
            )

            metrics["trajectory_3d_path"] = str(trajectory_3d_path)

        if self.config.save_full_trajectory_summary:
            # full_summary_path：三视图 + 三维综合图路径。
            full_summary_path = self.figure_dir / f"{episode_prefix}_full_trajectory_summary.png"

            plot_full_trajectory_summary(
                env=self.env,
                save_path=full_summary_path,
                show=False,
            )

            metrics["full_trajectory_summary_path"] = str(full_summary_path)

    def run_one_episode(self, episode_index: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        运行一个测试回合。

        参数：
            episode_index：
                测试回合编号。

            seed：
                环境随机种子。

        返回：
            metrics：
                当前测试回合指标。
        """
        # reset_result：环境 reset 返回值，兼容新旧 Gym。
        reset_result = self.env.reset(seed=seed)

        if isinstance(reset_result, tuple):
            # observation：当前观测。
            observation = reset_result[0]
        else:
            observation = reset_result

        # done：统一终止标志。
        done = False

        while not done:
            # action：智能体根据当前观测输出动作。
            action = self.agent.select_action(
                observation,
                deterministic=self.config.deterministic,
            )

            # step_result：环境 step 返回值，兼容 Gymnasium 五返回值和旧 Gym 四返回值。
            step_result = self.env.step(action)

            if len(step_result) == 5:
                next_observation, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                next_observation, reward, done, info = step_result

            # observation：更新当前观测。
            observation = next_observation

            if self.config.render and hasattr(self.env, "render"):
                self.env.render()

        # metrics：当前回合统计指标。
        metrics = collect_episode_metrics(
            env=self.env,
            episode_index=episode_index,
        )

        # 保存当前回合图像。
        self._save_episode_figures(
            episode_index=episode_index,
            metrics=metrics,
        )

        return metrics

    def evaluate(self, base_seed: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        执行多回合评估。

        参数：
            base_seed：
                基础随机种子。
                如果不为 None，则第 k 个回合使用 base_seed + k。

        返回：
            metric_records：
                每个回合的指标列表。

            summary：
                多回合汇总指标。
        """
        # metric_records：每个测试回合的指标记录。
        metric_records: List[Dict[str, Any]] = []

        self._log("=" * 80)
        self._log("开始模型评估")
        self._log(f"评估回合数：{self.config.eval_episodes}")
        self._log(f"确定性动作：{self.config.deterministic}")
        self._log("=" * 80)

        for episode_index in range(self.config.eval_episodes):
            # episode_seed：当前回合随机种子。
            episode_seed = None if base_seed is None else int(base_seed) + episode_index

            # metrics：当前回合指标。
            metrics = self.run_one_episode(
                episode_index=episode_index,
                seed=episode_seed,
            )

            metric_records.append(metrics)

            self._log(
                f"[Eval {episode_index:03d}] "
                f"reward={metrics['total_reward']:.3f}, "
                f"min_distance={metrics['min_distance']:.3f}, "
                f"success={bool(metrics['success'])}, "
                f"steps={metrics['episode_steps']}"
            )

        # summary：多回合汇总结果。
        summary = summarize_metrics(metric_records)

        # metrics_path：逐回合指标保存路径。
        metrics_path = self.metrics_dir / "evaluation_metrics.csv"
        save_metrics_to_csv(metric_records, metrics_path)

        # summary_path：汇总指标保存路径。
        summary_path = self.metrics_dir / "evaluation_summary.csv"
        save_metrics_to_csv([summary], summary_path)

        self._log("=" * 80)
        self._log("模型评估完成")
        self._log(f"成功率：{summary['success_rate']:.3f}")
        self._log(f"平均累计奖励：{summary['mean_total_reward']:.3f}")
        self._log(f"平均最小距离：{summary['mean_min_distance']:.3f}")
        self._log(f"指标保存目录：{self.metrics_dir}")
        self._log("=" * 80)

        return metric_records, summary