"""
train_curriculum_sac.py

作用：
    按由易到难的课程顺序训练 SAC 智能体。

课程策略：
    1. 每个课程阶段使用独立环境配置；
    2. 阶段之间默认继承 SAC 网络权重；
    3. 阶段之间默认重置 replay buffer，避免旧动力学样本污染新课程；
    4. 每个阶段输出独立 checkpoint、metrics 和 figures。

运行示例：
    python scripts/train_curriculum_sac.py
    python scripts/train_curriculum_sac.py --episodes-per-course 1 --max-steps-per-episode 5 --disable-eval
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# project_root：项目根目录。
project_root = Path(__file__).resolve().parents[1]

# src_path：源码目录，保证直接运行脚本时可以导入 hypersonic_rl。
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hypersonic_rl.agents import SACAgent, SACAgentConfig
from hypersonic_rl.buffers import ReplayBuffer, ReplayBufferConfig
from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.trainers import SACTrainer, SACTrainerConfig
from hypersonic_rl.utils import (
    create_logger,
    describe_device,
    get_device,
    load_config_from_project,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    """
    解析课程训练命令行参数。

    参数：
        无。

    返回：
        args：
            命令行参数命名空间。
    """
    # parser：课程训练命令行解析器。
    parser = argparse.ArgumentParser(description="运行 SAC 课程训练。")

    # config：课程训练 YAML 路径。
    parser.add_argument(
        "--config",
        default="configs/train/curriculum_sac.yaml",
        help="课程训练 YAML 配置路径。",
    )

    # episodes_per_course：用于 smoke test 的每课程回合数覆盖。
    parser.add_argument(
        "--episodes-per-course",
        type=int,
        default=None,
        help="覆盖每个课程阶段的 total_episodes，默认使用配置文件。",
    )

    # max_steps_per_episode：用于 smoke test 的单回合最大步数覆盖。
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="覆盖每回合最大步数，默认使用配置文件。",
    )

    # disable_eval：smoke test 时可关闭周期评估。
    parser.add_argument(
        "--disable-eval",
        action="store_true",
        help="关闭课程训练过程中的周期评估。",
    )

    # experiment_name：可选覆盖课程训练总实验名。
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="覆盖课程训练总实验目录名称。",
    )

    # args：解析后的命令行参数。
    args = parser.parse_args()

    return args


def filter_dataclass_config(config_dict: Dict[str, Any], dataclass_type: Any) -> Dict[str, Any]:
    """
    从配置字典中过滤 dataclass 支持的字段。

    参数：
        config_dict：
            原始配置字典。
        dataclass_type：
            目标 dataclass 类型。

    返回：
        filtered_config：
            只包含目标 dataclass 字段的配置字典。
    """
    # valid_field_names：目标 dataclass 的字段名集合。
    valid_field_names = {field.name for field in fields(dataclass_type)}

    # filtered_config：剔除 dataclass 不支持的字段。
    filtered_config = {
        key: value
        for key, value in config_dict.items()
        if key in valid_field_names
    }

    return filtered_config


def merge_config(base_config: Dict[str, Any], override_config: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    合并基础配置和课程覆盖配置。

    参数：
        base_config：
            基础配置字典。
        override_config：
            当前课程的覆盖配置字典。

    返回：
        merged_config：
            合并后的新配置字典。
    """
    # merged_config：复制基础配置，避免原字典被修改。
    merged_config = dict(base_config)

    if override_config:
        # override_config：课程配置优先级高于基础配置。
        merged_config.update(override_config)

    return merged_config


def build_env(env_config_dict: Dict[str, Any]) -> PursueEscapeEnv:
    """
    根据环境配置字典创建 PursueEscapeEnv。

    参数：
        env_config_dict：
            环境配置字典。

    返回：
        env：
            追逃突防环境对象。
    """
    # filtered_config：过滤掉环境配置类不识别的字段。
    filtered_config = filter_dataclass_config(env_config_dict, PursueEscapeEnvConfig)

    # env_config：环境配置对象。
    env_config = PursueEscapeEnvConfig(**filtered_config)

    # env：环境实例。
    env = PursueEscapeEnv(env_config)

    return env


def build_agent(
    env: PursueEscapeEnv,
    agent_config_dict: Dict[str, Any],
    device: Any,
) -> SACAgent:
    """
    根据环境空间和智能体配置创建 SACAgent。

    参数：
        env：
            当前课程环境。
        agent_config_dict：
            SAC 智能体基础配置字典。
        device：
            PyTorch 训练设备。

    返回：
        agent：
            SAC 智能体对象。
    """
    # state_dim：环境观测维度。
    state_dim = int(env.observation_space.shape[0])

    # action_dim：红方动作维度。
    action_dim = int(env.action_space.shape[0])

    # action_low/action_high：动作空间上下界。
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    # agent_config_base：过滤 SACAgentConfig 支持的字段。
    agent_config_base = filter_dataclass_config(agent_config_dict, SACAgentConfig)

    # agent_config_base：补充由环境决定的维度和动作范围。
    agent_config_base.update(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "device": str(device),
        }
    )

    # agent_config：SAC 智能体配置对象。
    agent_config = SACAgentConfig(**agent_config_base)

    # agent：SAC 智能体实例。
    agent = SACAgent(agent_config)

    return agent


def build_replay_buffer(
    env: PursueEscapeEnv,
    agent_config_dict: Dict[str, Any],
    device: Any,
) -> ReplayBuffer:
    """
    创建与当前环境匹配的 replay buffer。

    参数：
        env：
            当前课程环境。
        agent_config_dict：
            SAC 智能体配置字典。
        device：
            PyTorch 训练设备。

    返回：
        replay_buffer：
            新建的经验回放池。
    """
    # replay_buffer_size：经验回放池容量。
    replay_buffer_size = int(agent_config_dict.get("replay_buffer_size", 100000))

    # replay_buffer_config：经验回放池配置对象。
    replay_buffer_config = ReplayBufferConfig(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        capacity=replay_buffer_size,
        device=str(device),
    )

    # replay_buffer：经验回放池实例。
    replay_buffer = ReplayBuffer(replay_buffer_config)

    return replay_buffer


def apply_smoke_overrides(train_config_dict: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    应用命令行 smoke test 覆盖项。

    参数：
        train_config_dict：
            当前课程训练配置字典。
        args：
            命令行参数。

    返回：
        train_config_dict：
            应用覆盖后的训练配置字典。
    """
    # updated_config：复制训练配置，避免修改调用方字典。
    updated_config = dict(train_config_dict)

    if args.episodes_per_course is not None:
        # total_episodes：覆盖每个课程阶段训练回合数。
        updated_config["total_episodes"] = int(args.episodes_per_course)

    if args.max_steps_per_episode is not None:
        # max_steps_per_episode：覆盖单回合最大步数。
        updated_config["max_steps_per_episode"] = int(args.max_steps_per_episode)

    if args.disable_eval:
        # eval_interval：设为 0 表示关闭周期评估。
        updated_config["eval_interval"] = 0

    return updated_config


def run_course(
    course_index: int,
    course_config: Dict[str, Any],
    base_env_config: Dict[str, Any],
    base_train_config: Dict[str, Any],
    agent_config_dict: Dict[str, Any],
    agent: SACAgent | None,
    replay_buffer: ReplayBuffer | None,
    device: Any,
    args: argparse.Namespace,
) -> Tuple[SACAgent, ReplayBuffer, Dict[str, Any]]:
    """
    运行一个课程阶段。

    参数：
        course_index：
            当前课程序号。
        course_config：
            当前课程配置字典。
        base_env_config：
            基础环境配置字典。
        base_train_config：
            基础训练配置字典。
        agent_config_dict：
            SAC 智能体配置字典。
        agent：
            上一课程阶段训练后的智能体；第一阶段为 None。
        replay_buffer：
            上一阶段经验池；按课程配置决定是否复用。
        device：
            PyTorch 训练设备。
        args：
            命令行参数。

    返回：
        agent：
            当前课程训练后的智能体。
        replay_buffer：
            当前课程使用的经验池。
        course_record：
            当前课程摘要记录。
    """
    # course_name：当前课程阶段名称。
    course_name = str(course_config["name"])

    # env_overrides：当前课程环境覆盖项。
    env_overrides = course_config.get("env_overrides", {})

    # train_overrides：当前课程训练覆盖项。
    train_overrides = course_config.get("train_overrides", {})

    # env_config_dict：合并后的环境配置。
    env_config_dict = merge_config(base_env_config, env_overrides)

    # train_config_dict：合并后的训练配置。
    train_config_dict = merge_config(base_train_config, train_overrides)

    # train_config_dict：应用 smoke test 覆盖项。
    train_config_dict = apply_smoke_overrides(train_config_dict, args)

    if args.experiment_name is not None:
        # experiment_name：命令行覆盖课程训练总目录时，每个课程写入其子目录。
        train_config_dict["experiment_name"] = f"{args.experiment_name}/{course_name}"

    # train_config：训练器配置对象。
    train_config = SACTrainerConfig(**filter_dataclass_config(train_config_dict, SACTrainerConfig))

    # env/eval_env：训练环境和评估环境分开构造。
    env = build_env(env_config_dict)
    eval_env = build_env(env_config_dict)

    # should_reinitialize_agent：第一阶段或显式不继承时，新建智能体。
    should_reinitialize_agent = agent is None or not bool(course_config.get("load_previous_checkpoint", course_index > 0))

    if should_reinitialize_agent:
        # agent：新建 SAC 智能体。
        agent = build_agent(env=env, agent_config_dict=agent_config_dict, device=device)

    # should_reset_replay_buffer：默认每个课程都重置经验池。
    should_reset_replay_buffer = replay_buffer is None or bool(course_config.get("reset_replay_buffer", True))

    if should_reset_replay_buffer:
        # replay_buffer：为当前课程新建经验池。
        replay_buffer = build_replay_buffer(env=env, agent_config_dict=agent_config_dict, device=device)

    # experiment_dir：当前课程输出目录。
    experiment_dir = project_root / "experiments" / train_config.experiment_name

    # logger：当前课程独立日志对象。
    logger = create_logger(
        logger_name=f"train_curriculum_{course_name}",
        log_dir=experiment_dir / "logs",
        log_filename="train.log",
    )

    logger.info("=" * 80)
    logger.info("开始课程阶段 %d：%s", course_index + 1, course_name)
    logger.info("设备信息：%s", describe_device(device))
    logger.info("环境制导模式：%s", env.config.guidance_mode)
    logger.info("重置 replay buffer：%s", should_reset_replay_buffer)
    logger.info("继承上一阶段智能体：%s", not should_reinitialize_agent)
    logger.info("=" * 80)

    # trainer：当前课程阶段 SAC 训练器。
    trainer = SACTrainer(
        env=env,
        eval_env=eval_env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=train_config,
        experiment_dir=experiment_dir,
        logger=logger,
    )

    # training_records：运行当前课程训练并收集指标。
    training_records = trainer.train()

    # last_record：当前课程最后一个 episode 的指标。
    last_record = training_records[-1] if training_records else {}

    # course_record：课程级摘要记录。
    course_record: Dict[str, Any] = {
        "course_index": int(course_index),
        "course_name": course_name,
        "experiment_dir": str(experiment_dir),
        "guidance_mode": env.config.guidance_mode,
        "total_episodes": int(train_config.total_episodes),
        "final_total_reward": float(last_record.get("total_reward", 0.0)),
        "final_min_distance": float(last_record.get("min_distance", 0.0)),
        "final_success": float(last_record.get("success", 0.0)),
        "final_interceptor_control_energy": float(last_record.get("interceptor_control_energy", 0.0)),
    }

    return agent, replay_buffer, course_record


def main() -> None:
    """
    课程训练主函数。

    参数：
        无。

    返回：
        None。
    """
    # args：命令行参数。
    args = parse_args()

    # curriculum_config：课程训练总配置。
    curriculum_config = load_config_from_project(args.config)

    # base_env_config：基础环境配置。
    base_env_config = load_config_from_project(curriculum_config["base_env_config"])

    # agent_config_dict：基础 SAC 智能体配置。
    agent_config_dict = load_config_from_project(curriculum_config["base_agent_config"])

    # base_train_config：基础训练器配置。
    base_train_config = load_config_from_project(curriculum_config["base_train_config"])

    # experiment_name：课程训练总实验名称。
    experiment_name = args.experiment_name or str(curriculum_config.get("experiment_name", "sac_curriculum"))

    # courses：课程阶段列表。
    courses: List[Dict[str, Any]] = list(curriculum_config.get("courses", []))

    if not courses:
        raise ValueError("课程训练配置中 courses 不能为空。")

    # seed：使用基础训练配置中的随机种子。
    seed = int(base_train_config.get("seed", 0))
    set_global_seed(seed)

    # device：训练设备。
    device = get_device(prefer_gpu=True)

    # agent：跨课程继承的 SAC 智能体。
    agent: SACAgent | None = None

    # replay_buffer：跨课程可选复用的经验池。
    replay_buffer: ReplayBuffer | None = None

    # course_records：课程级摘要记录。
    course_records: List[Dict[str, Any]] = []

    for course_index, course_config in enumerate(courses):
        # course_config：复制课程配置，避免写回原始 YAML 数据。
        course_config = dict(course_config)

        # train_overrides：保证每个课程写入课程总目录下的独立子目录。
        train_overrides = dict(course_config.get("train_overrides", {}))
        train_overrides.setdefault("experiment_name", f"{experiment_name}/{course_config['name']}")
        course_config["train_overrides"] = train_overrides

        # agent/replay_buffer/course_record：运行一个课程阶段。
        agent, replay_buffer, course_record = run_course(
            course_index=course_index,
            course_config=course_config,
            base_env_config=base_env_config,
            base_train_config=base_train_config,
            agent_config_dict=agent_config_dict,
            agent=agent,
            replay_buffer=replay_buffer,
            device=device,
            args=args,
        )

        course_records.append(course_record)

    # summary_dir：课程训练总指标目录。
    summary_dir = project_root / "experiments" / experiment_name / "metrics"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # summary_path：课程级摘要 CSV 路径。
    summary_path = summary_dir / "curriculum_summary.csv"

    # summary_csv：保存课程级摘要。
    pd.DataFrame(course_records).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("SAC 课程训练完成")
    print(f"课程摘要：{summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
