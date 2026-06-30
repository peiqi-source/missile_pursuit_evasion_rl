from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


@dataclass
class TrainingStartConfig:
    """
    控制 SAC 训练从哪里开始。

    training_start_mode:
        scratch:
            从零训练。
        finetune_actor:
            只加载 Actor，适合换 reward 后 fine-tuning。
        finetune_actor_critic:
            加载 Actor + Critic，ReplayBuffer 仍然清空。
        network_resume:
            加载 Actor + Critic + alpha，但不恢复 ReplayBuffer / trainer 计数。
    """
    training_start_mode: str = "scratch"
    checkpoint_path: str = ""

    # 是否严格匹配网络参数名。
    strict_load: bool = True

    # 是否在加载 checkpoint 后重置 SAC alpha。
    reset_alpha: bool = False
    alpha_value: float = 0.01


VALID_TRAINING_START_MODES = {
    "scratch",
    "finetune_actor",
    "finetune_actor_critic",
    "network_resume",
}


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    """
    将项目相对路径转换为绝对路径。
    """
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def find_agent_module(agent: Any, names: Iterable[str]) -> Optional[Any]:
    """
    按候选名称从 agent 中查找网络模块。

    用于兼容不同命名：
        actor / policy / actor_net
        critic1 / q1 / critic_1
    """
    for name in names:
        if hasattr(agent, name):
            return getattr(agent, name)

    return None


def load_first_available_state_dict(
    module: Any,
    checkpoint: Dict[str, Any],
    checkpoint_keys: Iterable[str],
    module_label: str,
    strict: bool,
    logger: Any,
) -> bool:
    """
    从 checkpoint 中按候选 key 加载 state_dict。
    """
    if module is None:
        logger.warning("未找到 agent 中的 %s 模块，跳过。", module_label)
        return False

    for key in checkpoint_keys:
        if key not in checkpoint:
            continue

        state_dict = checkpoint[key]

        if not isinstance(state_dict, dict):
            logger.warning(
                "checkpoint['%s'] 不是 state_dict，实际类型为 %s，跳过。",
                key,
                type(state_dict),
            )
            continue

        module.load_state_dict(state_dict, strict=strict)
        logger.info("已加载 %s <- checkpoint['%s']", module_label, key)
        return True

    logger.warning(
        "checkpoint 中没有找到 %s 的权重。候选 key=%s",
        module_label,
        list(checkpoint_keys),
    )
    return False


def set_agent_alpha(agent: Any, alpha_value: float, logger: Any) -> None:
    """
    重置 SAC 熵系数 alpha。
    """
    alpha_value = float(alpha_value)

    if hasattr(agent, "log_alpha"):
        with torch.no_grad():
            agent.log_alpha.data.fill_(math.log(alpha_value))
        logger.info("已重置 log_alpha，使 alpha≈%.6f", alpha_value)
        return

    if hasattr(agent, "alpha"):
        try:
            agent.alpha = alpha_value
            logger.info("已重置 alpha=%.6f", alpha_value)
            return
        except Exception as exc:
            logger.warning("设置 agent.alpha 失败：%s", exc)

    logger.warning("未找到 agent.log_alpha 或 agent.alpha，跳过 alpha 重置。")


def load_alpha_from_checkpoint(
    agent: Any,
    checkpoint: Dict[str, Any],
    logger: Any,
) -> bool:
    """
    尝试从 checkpoint 恢复 alpha。
    """
    if hasattr(agent, "log_alpha") and "log_alpha" in checkpoint:
        value = checkpoint["log_alpha"]

        with torch.no_grad():
            if torch.is_tensor(value):
                agent.log_alpha.data.copy_(value.to(agent.log_alpha.device))
            else:
                agent.log_alpha.data.fill_(float(value))

        logger.info("已从 checkpoint['log_alpha'] 恢复 alpha。")
        return True

    if hasattr(agent, "alpha") and "alpha" in checkpoint:
        try:
            agent.alpha = float(checkpoint["alpha"])
            logger.info("已从 checkpoint['alpha'] 恢复 alpha=%.6f", float(agent.alpha))
            return True
        except Exception as exc:
            logger.warning("从 checkpoint 恢复 alpha 失败：%s", exc)

    logger.info("checkpoint 中没有可恢复的 alpha 字段。")
    return False


def apply_training_start_mode(
    agent: Any,
    config: TrainingStartConfig,
    project_root: Path,
    device: Any,
    logger: Any,
) -> None:
    """
    根据 training_start_mode 初始化 agent。

    这个函数只负责 agent 网络和 alpha；
    ReplayBuffer 是否新建、Trainer 如何启动，仍由 train_sac.py 控制。
    """
    mode = str(config.training_start_mode).strip()

    if mode not in VALID_TRAINING_START_MODES:
        raise ValueError(
            f"未知 training_start_mode={mode}，"
            f"可选值为 {sorted(VALID_TRAINING_START_MODES)}"
        )

    if mode == "scratch":
        logger.info("training_start_mode=scratch：从零训练，不加载 checkpoint。")
        return

    if not str(config.checkpoint_path).strip():
        raise ValueError(
            f"training_start_mode={mode} 需要指定 checkpoint_path。"
        )

    checkpoint_path = resolve_project_path(project_root, config.checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"checkpoint 应该是 dict，但实际类型为 {type(checkpoint)}：{checkpoint_path}"
        )

    logger.info("加载训练启动 checkpoint：%s", checkpoint_path)
    logger.info("training_start_mode=%s", mode)
    logger.info("checkpoint keys=%s", list(checkpoint.keys()))

    strict = bool(config.strict_load)

    # ------------------------------------------------------------
    # 1. Actor：finetune / resume 都要加载
    # ------------------------------------------------------------
    actor = find_agent_module(
        agent,
        ["actor", "policy", "actor_net", "policy_net"],
    )

    actor_loaded = load_first_available_state_dict(
        module=actor,
        checkpoint=checkpoint,
        checkpoint_keys=[
            "actor_state_dict",
            "policy_state_dict",
            "actor",
            "policy",
            "actor_net_state_dict",
            "policy_net_state_dict",
        ],
        module_label="actor",
        strict=strict,
        logger=logger,
    )

    if not actor_loaded:
        raise RuntimeError(
            "Actor 加载失败。请查看日志中的 checkpoint keys，"
            "确认 checkpoint 里 Actor 权重字段叫什么。"
        )

    # ------------------------------------------------------------
    # 2. Critic：只有部分模式加载
    # ------------------------------------------------------------
    if mode in {"finetune_actor_critic", "network_resume"}:
        critic = find_agent_module(
            agent,
            ["critic", "q_network", "critic_net"],
        )
        critic1 = find_agent_module(
            agent,
            ["critic1", "q1", "critic_1", "q_net1"],
        )
        critic2 = find_agent_module(
            agent,
            ["critic2", "q2", "critic_2", "q_net2"],
        )
        target_critic = find_agent_module(
            agent,
            ["target_critic", "critic_target", "target_q_network"],
        )
        target_critic1 = find_agent_module(
            agent,
            ["target_critic1", "target_q1", "target_critic_1"],
        )
        target_critic2 = find_agent_module(
            agent,
            ["target_critic2", "target_q2", "target_critic_2"],
        )

        load_first_available_state_dict(
            critic,
            checkpoint,
            ["critic_state_dict", "q_network_state_dict", "critic", "q_network"],
            "critic",
            strict=False,
            logger=logger,
        )
        load_first_available_state_dict(
            critic1,
            checkpoint,
            ["critic1_state_dict", "q1_state_dict", "critic_1_state_dict", "critic1", "q1"],
            "critic1",
            strict=False,
            logger=logger,
        )
        load_first_available_state_dict(
            critic2,
            checkpoint,
            ["critic2_state_dict", "q2_state_dict", "critic_2_state_dict", "critic2", "q2"],
            "critic2",
            strict=False,
            logger=logger,
        )
        load_first_available_state_dict(
            target_critic,
            checkpoint,
            ["target_critic_state_dict", "critic_target_state_dict", "target_critic", "critic_target"],
            "target_critic",
            strict=False,
            logger=logger,
        )
        load_first_available_state_dict(
            target_critic1,
            checkpoint,
            ["target_critic1_state_dict", "target_q1_state_dict", "target_critic1", "target_q1"],
            "target_critic1",
            strict=False,
            logger=logger,
        )
        load_first_available_state_dict(
            target_critic2,
            checkpoint,
            ["target_critic2_state_dict", "target_q2_state_dict", "target_critic2", "target_q2"],
            "target_critic2",
            strict=False,
            logger=logger,
        )

    # ------------------------------------------------------------
    # 3. Alpha
    # ------------------------------------------------------------
    if bool(config.reset_alpha):
        set_agent_alpha(agent, config.alpha_value, logger)
    elif mode == "network_resume":
        load_alpha_from_checkpoint(agent, checkpoint, logger)

    logger.info("训练启动模式初始化完成：%s", mode)