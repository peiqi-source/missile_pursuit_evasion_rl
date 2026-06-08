# Hypersonic Pursuit-Evasion RL

本项目将原始 `PursueEscapeEnv.py`、`SAC.py` 和 `plot.py` 重构为规范的强化学习工程项目。第一阶段保留当前端到端 SAC 逻辑：智能体直接输出红方横向机动过载，蓝方采用比例导引；同时在 `guidance.py`、LSTM 网络、序列缓存和训练器中预留论文第 4 章和第 5 章扩展接口。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

如在 Windows/Anaconda 环境遇到 OpenMP 重复加载提示，项目入口会自动设置 `KMP_DUPLICATE_LIB_OK=TRUE`，与原始 `SAC.py` 的兼容处理保持一致。

## 常用命令

```bash
python scripts/debug_env_rollout.py
python scripts/train_sac.py
python scripts/eval_sac.py --checkpoint experiments/sac_baseline/<timestamp>/checkpoints/latest.pt
python scripts/plot_results.py --metrics experiments/sac_baseline/<timestamp>/metrics/metrics.csv
pytest
```

训练会在 `experiments/sac_baseline/<timestamp>/` 下生成：

- `checkpoints/latest.pt` 和按 episode 命名的 checkpoint。
- `logs/train.log`。
- `metrics/metrics.csv`。
- `figures/episode_*.png` 和训练曲线图。

## 目录说明

- `src/hypersonic_rl/envs/`：追逃环境、动力学、制导、奖励和包装器。
- `src/hypersonic_rl/agents/`：SAC 智能体与 LSTM-SAC 预留类。
- `src/hypersonic_rl/networks/`：Actor、Twin Critic、LSTM 网络预留和初始化函数。
- `src/hypersonic_rl/buffers/`：经验回放缓存与序列缓存预留。
- `src/hypersonic_rl/trainers/`：训练循环与后续 LSTM-SAC 训练器接口。
- `src/hypersonic_rl/visualization/`：episode、轨迹和训练曲线绘图。
- `src/hypersonic_rl/legacy/`：保存重构前的原始代码，便于对照、复核和回退；该目录中的文件不直接改动。
- `experiments/`：训练输出目录，每次训练会生成独立时间戳子目录。
- `docs/`：工程设计、SAC 框架、环境模型，以及论文第 4、5 章映射说明。

## 当前阶段边界

当前版本完整实现普通 SAC 工程框架和端到端过载输出环境；暂不完整复现论文第 5 章 LSTM-SAC，也不把 SAC 输出改成微分对策制导律参数。后续可在 `differential_game_parameterized_guidance()` 中接入论文第 3、4 章制导律，并让 Actor 输出导航增益/制导参数。
