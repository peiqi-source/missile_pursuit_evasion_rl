"""
sac_trainer.py

作用：
    实现 SAC 智能体训练器。

训练器职责：
    1. 管理环境交互；
    2. 管理经验回放池；
    3. 调用 SACAgent.update() 更新网络；
    4. 记录训练指标和损失；
    5. 保存 checkpoint；
    6. 绘制训练曲线；
    7. 周期性评估当前策略。

工程说明：
    SACAgent 只负责算法更新；
    SACTrainer 负责完整训练流程。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hypersonic_rl.agents import SACAgent
from hypersonic_rl.buffers import ReplayBuffer
from hypersonic_rl.evaluation import Evaluator, EvaluatorConfig
from hypersonic_rl.evaluation.metrics import collect_episode_metrics, save_metrics_to_csv
from hypersonic_rl.utils import (
    build_run_manifest,
    ensure_dir,
    find_project_root,
    load_checkpoint,
    save_checkpoint,
    save_run_manifest,
    set_global_seed,
)
from hypersonic_rl.visualization import plot_sac_loss_curves, plot_training_curves


@dataclass
class SACTrainerConfig:
    """
    SACTrainerConfig

    作用：
        管理 SAC 训练过程中的工程参数。

    属性：
        total_episodes：
            训练总回合数。

        max_steps_per_episode：
            单回合最大交互步数。
            即使环境没有 terminated，也会在达到该步数时停止当前回合。

        start_steps：
            训练初期使用随机动作探索的环境步数。
            在 global_step < start_steps 时，不使用 Actor 输出动作，而是使用 env.action_space.sample()。

        update_after：
            经验池中至少积累多少环境步后才开始更新网络。

        update_every：
            每隔多少个环境交互步执行一次更新。

        updates_per_step：
            每次触发更新时，执行多少次 SACAgent.update()。

        batch_size：
            每次网络更新从 replay buffer 中采样的样本数量。

        eval_interval：
            每隔多少个 episode 评估一次当前策略。
            如果 <= 0，则不做周期性评估。

        eval_episodes：
            每次评估运行多少个测试回合。

        save_interval：
            每隔多少个 episode 保存一次 checkpoint。

        log_interval：
            每隔多少个 episode 打印一次训练日志。

        plot_interval：
            每隔多少个 episode 绘制一次训练曲线。
            如果 <= 0，则只在训练结束后绘图。

        seed：
            随机种子。

        experiment_name：
            实验名称。
            对应 experiments/{experiment_name}/ 目录。
    """

    total_episodes: int = 200
    max_steps_per_episode: int = 5000
    start_steps: int = 1000
    update_after: int = 1000
    update_every: int = 1
    updates_per_step: int = 1
    batch_size: int = 128
    eval_interval: int = 10
    eval_episodes: int = 3
    save_interval: int = 20
    log_interval: int = 1
    plot_interval: int = 10
    seed: int = 0
    experiment_name: str = "sac_baseline"
    resume_from_checkpoint: Optional[str] = None
    resume_load_optimizers: bool = True


class SACTrainer:
    """
    SAC 训练器。

    对外主要接口：
        train()
            启动完整训练流程。
    """

    def __init__(
        self,
        env: Any,
        eval_env: Any,
        agent: SACAgent,
        replay_buffer: ReplayBuffer,
        config: SACTrainerConfig,
        experiment_dir: str | Path,
        logger: Optional[Any] = None,
    ) -> None:
        """
        初始化 SAC 训练器。

        参数：
            env：
                训练环境。

            eval_env：
                评估环境。
                建议与训练环境分开，避免评估轨迹覆盖训练轨迹。

            agent：
                SAC 智能体。

            replay_buffer：
                经验回放池。

            config：
                训练器配置。

            experiment_dir：
                当前实验目录。

            logger：
                日志对象，可为 None。
        """
        # env：训练环境。
        self.env = env

        # eval_env：评估环境。
        self.eval_env = eval_env

        # agent：SAC 智能体。
        self.agent = agent

        # replay_buffer：经验回放池。
        self.replay_buffer = replay_buffer

        # config：训练配置。
        self.config = config

        # logger：日志对象。
        self.logger = logger

        # experiment_dir：当前实验根目录。
        self.experiment_dir = Path(experiment_dir)

        # checkpoint_dir：模型保存目录。
        self.checkpoint_dir = ensure_dir(self.experiment_dir / "checkpoints")

        # metrics_dir：指标保存目录。
        self.metrics_dir = ensure_dir(self.experiment_dir / "metrics")

        # figure_dir：训练曲线保存目录。
        self.figure_dir = ensure_dir(self.experiment_dir / "figures")

        # eval_dir：评估输出目录。
        self.eval_dir = ensure_dir(self.experiment_dir / "evaluation")

        # global_step：全局环境交互步数。
        self.global_step = 0

        # update_step：网络更新次数。
        self.update_step = 0

        # start_episode：恢复训练时从 checkpoint episode+1 开始。
        self.start_episode = 1

        # training_metric_records：逐 episode 训练指标。
        self.training_metric_records: List[Dict[str, Any]] = []

        # loss_records：逐 update 训练损失。
        self.loss_records: List[Dict[str, Any]] = []

        # last_loss_info：最近一次 SAC 更新返回的损失信息。
        self.last_loss_info: Dict[str, float] = {}

        # project_root：用于记录 git 状态和相对配置快照；pytest 临时目录不在项目内时回退到包路径查找。
        try:
            self.project_root = find_project_root(self.experiment_dir)
        except FileNotFoundError:
            self.project_root = find_project_root()

        # run_manifest：训练运行清单，记录配置、seed、环境 profile 和 git 状态。
        self.run_manifest = build_run_manifest(
            run_name=self.config.experiment_name,
            project_root=self.project_root,
            env_config=getattr(self.env, "config", None),
            agent_config=getattr(self.agent, "config", None),
            trainer_config=self.config,
            seed=self.config.seed,
            extra={
                "algorithm": str(getattr(self, "algorithm_name", "SAC")),
                "experiment_dir": str(self.experiment_dir),
            },
        )
        save_run_manifest(self.run_manifest, self.experiment_dir / "run_manifest.json")

        # resume：如配置指定 checkpoint，则恢复智能体和训练计数器。
        self._resume_if_configured()

    def _log(self, message: str) -> None:
        """
        输出日志。

        参数：
            message：
                待输出文本。
        """
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _resume_if_configured(self) -> None:
        """
        根据配置从 checkpoint 恢复训练状态。

        说明：
            当前 checkpoint 恢复 agent、优化器、global_step、update_step 和历史指标；
            replay buffer 暂不序列化，恢复后会重新积累经验。
        """
        # resume_path_value：为空时完全跳过恢复逻辑。
        resume_path_value = getattr(self.config, "resume_from_checkpoint", None)
        if not resume_path_value:
            return

        # resume_path：支持绝对路径或相对项目根目录路径。
        resume_path = Path(str(resume_path_value))
        if not resume_path.is_absolute():
            resume_path = self.project_root / resume_path

        if not resume_path.exists():
            raise FileNotFoundError(f"恢复训练 checkpoint 不存在：{resume_path}")

        # checkpoint：读取旧训练状态。
        checkpoint = load_checkpoint(resume_path, device=getattr(self.agent, "device", None))

        if "agent_state" not in checkpoint:
            raise KeyError(f"checkpoint 中缺少 agent_state 字段：{resume_path}")

        # agent：恢复网络和优化器状态。
        self.agent.load_state_dict(
            checkpoint["agent_state"],
            load_optimizers=bool(self.config.resume_load_optimizers),
        )

        # 训练计数器和历史曲线：用于继续编号和绘图。
        self.global_step = int(checkpoint.get("global_step", self.global_step))
        self.update_step = int(checkpoint.get("update_step", self.update_step))
        self.start_episode = int(checkpoint.get("episode", 0)) + 1
        self.training_metric_records = list(checkpoint.get("training_metric_records", self.training_metric_records))
        self.loss_records = list(checkpoint.get("loss_records", self.loss_records))

        # manifest：记录恢复来源和 replay buffer 处理方式。
        self.run_manifest["extra"]["resume_from_checkpoint"] = str(resume_path)
        self.run_manifest["extra"]["resume_load_optimizers"] = bool(self.config.resume_load_optimizers)
        self.run_manifest["extra"]["resume_replay_buffer"] = "not_restored"
        save_run_manifest(self.run_manifest, self.experiment_dir / "run_manifest.json")

        self._log(f"已从 checkpoint 恢复训练：{resume_path}")
        self._log("注意：当前实现不恢复 replay buffer，恢复后将重新积累经验。")

    def _reset_env(self, episode: int):
        """
        重置训练环境。

        参数：
            episode：
                当前训练回合编号。

        返回：
            observation：
                初始观测。
        """
        # episode_seed：当前 episode 的随机种子。
        episode_seed = int(self.config.seed) + int(episode)

        # reset_result：环境 reset 返回值，兼容新旧 Gym。
        reset_result = self.env.reset(seed=episode_seed)

        if isinstance(reset_result, tuple):
            observation = reset_result[0]
        else:
            observation = reset_result

        return observation

    def _select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        根据当前训练阶段选择动作。

        参数：
            observation：
                当前环境观测。

        返回：
            action：
                将要传入环境的动作。
        """
        if self.global_step < self.config.start_steps:
            # 训练初期使用随机动作，增加经验池多样性。
            action = self.env.action_space.sample()
        else:
            # start_steps 之后使用 SAC Actor 输出动作。
            action = self.agent.select_action(observation, deterministic=False)

        return action

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
        向 ReplayBuffer 存储一条经验。

        参数：
            observation：
                当前状态。

            action：
                当前动作。

            reward：
                当前奖励。

            next_observation：
                下一状态。

            terminated：
                是否自然终止。

            truncated：
                是否时间截断。
        """
        # done_for_buffer：写入经验池的终止标志。
        # 当前工程中将 terminated 和 truncated 都视为回合结束。
        # 如果后续严格区分时间截断，可以改为 done_for_buffer = terminated。
        done_for_buffer = bool(terminated or truncated)

        self.replay_buffer.add(
            state=observation,
            action=action,
            reward=reward,
            next_state=next_observation,
            done=done_for_buffer,
        )

    def _should_update(self) -> bool:
        """
        判断当前是否应该更新 SAC 网络。

        返回：
            should_update：
                True 表示执行更新。
        """
        if self.global_step < self.config.update_after:
            return False

        if len(self.replay_buffer) < self.config.batch_size:
            return False

        if self.global_step % self.config.update_every != 0:
            return False

        return True

    def _update_agent(self, episode: int) -> None:
        """
        更新 SACAgent。

        参数：
            episode：
                当前训练回合编号。
        """
        if not self._should_update():
            return

        for _ in range(self.config.updates_per_step):
            # batch：从经验池采样训练数据。
            batch = self.replay_buffer.sample(batch_size=self.config.batch_size)

            # loss_info：一次 SAC 更新的损失信息。
            loss_info = self.agent.update(batch)

            self.update_step += 1
            self.last_loss_info = loss_info

            # loss_record：当前更新步的日志记录。
            loss_record = {
                "episode": int(episode),
                "global_step": int(self.global_step),
                "update_step": int(self.update_step),
            }
            loss_record.update(loss_info)

            self.loss_records.append(loss_record)

    def _save_training_outputs(self) -> None:
        """
        保存训练指标、损失指标和曲线图。
        """
        # training_metrics_path：逐回合训练指标 CSV。
        training_metrics_path = self.metrics_dir / "training_metrics.csv"

        # loss_metrics_path：逐更新损失指标 CSV。
        loss_metrics_path = self.metrics_dir / "loss_metrics.csv"

        if self.training_metric_records:
            save_metrics_to_csv(self.training_metric_records, training_metrics_path)

            plot_training_curves(
                metrics=self.training_metric_records,
                save_dir=self.figure_dir,
                rolling_window=10,
            )

        if self.loss_records:
            save_metrics_to_csv(self.loss_records, loss_metrics_path)

            plot_sac_loss_curves(
                loss_metrics=self.loss_records,
                save_dir=self.figure_dir,
                rolling_window=20,
            )

    def _save_checkpoint(self, episode: int, filename: Optional[str] = None) -> Path:
        """
        保存当前 SACAgent 检查点。

        参数：
            episode：
                当前训练回合编号。

            filename：
                检查点文件名。
                如果为 None，则自动生成 episode_xxxx.pth。

        返回：
            checkpoint_path：
                保存后的文件路径。
        """
        if filename is None:
            filename = f"episode_{episode:04d}.pth"

        # checkpoint：标准 checkpoint 字典。
        checkpoint = {
            "episode": int(episode),
            "global_step": int(self.global_step),
            "update_step": int(self.update_step),
            "agent_state": self.agent.state_dict(),
            "trainer_config": asdict(self.config),
            "training_metric_records": self.training_metric_records,
            "loss_records": self.loss_records,
            "run_manifest": self.run_manifest,
        }

        checkpoint_path = save_checkpoint(
            checkpoint=checkpoint,
            checkpoint_dir=self.checkpoint_dir,
            filename=filename,
            also_save_latest=True,
        )

        # run_manifest：保存 checkpoint 后回写路径，方便从 manifest 快速定位模型文件。
        self.run_manifest["checkpoint_path"] = str(checkpoint_path)
        save_run_manifest(self.run_manifest, self.experiment_dir / "run_manifest.json")

        return checkpoint_path

    def _evaluate_if_needed(self, episode: int) -> None:
        """
        按周期评估当前策略。

        参数：
            episode：
                当前训练回合编号。
        """
        if self.config.eval_interval <= 0:
            return

        if episode % self.config.eval_interval != 0:
            return

        if episode == 0:
            return

        # eval_output_dir：当前 episode 评估输出目录。
        eval_output_dir = self.eval_dir / f"episode_{episode:04d}"

        # evaluator_config：评估器配置。
        evaluator_config = EvaluatorConfig(
            eval_episodes=self.config.eval_episodes,
            deterministic=True,
            render=False,
            save_episode_plots=True,
            save_single_view_trajectory=False,
            save_three_view_trajectory=True,
            save_3d_trajectory=True,
            save_full_trajectory_summary=False,
            output_dir=str(eval_output_dir),
        )

        # evaluator：评估器。
        evaluator = Evaluator(
            env=self.eval_env,
            agent=self.agent,
            config=evaluator_config,
            logger=self.logger,
        )

        evaluator.evaluate(base_seed=self.config.seed + 10000 + episode)

    def train(self) -> List[Dict[str, Any]]:
        """
        启动 SAC 训练。

        返回：
            training_metric_records：
                逐 episode 训练指标列表。
        """
        set_global_seed(self.config.seed)

        # algorithm_name：训练日志中的算法名称；子类可覆盖为 LSTM-SAC 等名称。
        algorithm_name = str(getattr(self, "algorithm_name", "SAC"))

        self._log("=" * 80)
        self._log(f"开始 {algorithm_name} 训练")
        self._log(f"实验名称：{self.config.experiment_name}")
        self._log(f"训练回合数：{self.config.total_episodes}")
        self._log(f"单回合最大步数：{self.config.max_steps_per_episode}")
        self._log(f"随机探索步数 start_steps：{self.config.start_steps}")
        self._log(f"网络更新起始步 update_after：{self.config.update_after}")
        self._log(f"Batch size：{self.config.batch_size}")
        self._log("=" * 80)

        for episode in range(int(self.start_episode), self.config.total_episodes + 1):
            # observation：当前 episode 初始观测。
            observation = self._reset_env(episode)

            # done：统一回合结束标志。
            done = False

            # episode_step：当前 episode 内部步数。
            episode_step = 0

            while not done and episode_step < self.config.max_steps_per_episode:
                # action：当前步动作。
                action = self._select_action(observation)

                # step_result：环境交互结果。
                step_result = self.env.step(action)

                if len(step_result) == 5:
                    next_observation, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    next_observation, reward, done, info = step_result
                    terminated = bool(done)
                    truncated = False

                # 存储经验。
                self._store_transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )

                # 推进状态和计数器。
                observation = next_observation
                episode_step += 1
                self.global_step += 1

                # 更新 SACAgent。
                self._update_agent(episode=episode)

            # extra_info：写入当前 episode 指标中的额外训练信息。
            extra_info: Dict[str, Any] = {
                "global_step": int(self.global_step),
                "update_step": int(self.update_step),
                "replay_buffer_size": int(len(self.replay_buffer)),
            }
            extra_info.update(self.last_loss_info)

            # episode_metrics：当前 episode 训练指标。
            episode_metrics = collect_episode_metrics(
                env=self.env,
                episode_index=episode,
                extra_info=extra_info,
            )

            self.training_metric_records.append(episode_metrics)

            if episode % self.config.log_interval == 0:
                self._log(
                    f"[Train {episode:04d}] "
                    f"steps={episode_metrics['episode_steps']}, "
                    f"global_step={self.global_step}, "
                    f"reward={episode_metrics['total_reward']:.3f}, "
                    f"min_distance={episode_metrics['min_distance']:.3f}, "
                    f"success={bool(episode_metrics['success'])}, "
                    f"buffer={len(self.replay_buffer)}, "
                    f"alpha={self.last_loss_info.get('alpha', np.nan):.4f}"
                )

            if episode % self.config.save_interval == 0:
                checkpoint_path = self._save_checkpoint(episode=episode)
                self._log(f"已保存 checkpoint：{checkpoint_path}")

            if self.config.plot_interval > 0 and episode % self.config.plot_interval == 0:
                self._save_training_outputs()

            self._evaluate_if_needed(episode=episode)

        # 训练结束后保存全部输出。
        self._save_checkpoint(episode=self.config.total_episodes, filename="final.pth")
        self._save_training_outputs()

        self._log("=" * 80)
        self._log(f"{algorithm_name} 训练完成")
        self._log(f"训练指标目录：{self.metrics_dir}")
        self._log(f"训练图像目录：{self.figure_dir}")
        self._log(f"模型保存目录：{self.checkpoint_dir}")
        self._log("=" * 80)

        return self.training_metric_records
