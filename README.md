# Hypersonic Pursuit-Evasion RL

本项目是在学位论文复现基础上整理出的高超声速一对二突防强化学习工程。当前主线对齐论文第 5 章端到端任务：红方智能体输出 1 维横向过载，两枚蓝方拦截弹独立闭环制导，默认观测为 10 维论文式状态。

## 当前状态

- 环境：`PursueEscapeEnv` 默认 `interceptor_count=2`、`scenario_profile=paper_200km_end_to_end`、`observation_mode=thesis_end_to_end_10d`。
- 制导：`source_pn` 与 `mid_terminal_interceptor` 均支持每枚弹独立 `ny/nz` 双通道、一阶自动驾驶仪、二维过载限幅和连续最近点命中判据。
- 训练：普通 SAC 保留为基线；第 5 章 LSTM-SAC 已实现 `StateSequenceWrapper`、`SequenceReplayBuffer`、LSTM Actor/Critic、Agent 和 Trainer。
- 评估：新增统一评估与论文式 benchmark，输出逐回合 CSV、summary CSV、轨迹图、运行 manifest 和报告。
- 保真：`legacy/` 作为复现基准保留，不直接修改；当前工程默认仍使用三自由度质点模型，不引入六自由度、发动机或气动高保真模型。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## 快速验证

```bash
python -m pytest
python scripts/debug_two_interceptor_compare.py --seeds 0 --max-time 20
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac_smoke.yaml
python scripts/run_paper_benchmark.py --config configs/eval/benchmark_smoke.yaml
```

Smoke 配置只用于检查代码链路、文件输出和图像生成，不作为论文实验结果。

## 正式入口

普通 SAC paper baseline：

```bash
python scripts/train_sac.py --train-config configs/train/train_sac_paper.yaml
```

第 5 章 LSTM-SAC：

```bash
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac.yaml
```

统一评估：

```bash
python scripts/eval_policy.py --config configs/eval/eval_policy.yaml
```

论文式 benchmark：

```bash
python scripts/run_paper_benchmark.py --config configs/eval/benchmark_paper.yaml
```

## 默认接口

- 红方动作：`action=[nzc_h]`，维度固定为 1。
- 原始观测：10 维 `[r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]`。
- LSTM-SAC 观测：`[sequence_length, 10]`，默认 `sequence_length=3`。
- 多弹诊断：统一使用 `interceptor_{i}_*`，例如 `interceptor_1_min_distance`、`interceptor_2_phase`、`interceptor_1_nz_actual`。

## 输出与复现

训练输出位于 `experiments/<experiment_name>/`：

- `checkpoints/latest.pth`、`checkpoints/final.pth` 和 episode checkpoint。
- `metrics/training_metrics.csv`、`metrics/loss_metrics.csv`。
- `figures/` 下的训练曲线和轨迹图。
- `run_manifest.json`，记录 config snapshot、seed、git 状态、checkpoint 路径和环境 profile。

评估与 benchmark 输出位于 `outputs/`，同样包含 CSV、summary、图像和 `run_manifest.json`。训练配置支持 `resume_from_checkpoint`，可从 `latest.pth` 恢复 agent/optimizer、计数器和历史指标；当前 replay buffer 不随 checkpoint 恢复，恢复后会重新积累经验。

## 文档

- [模型保真矩阵](docs/model_fidelity_matrix.md)
- [环境模型](docs/env_model.md)
- [第 5 章 LSTM-SAC](docs/chapter5_lstm_sac_plan.md)
- [实验流程](docs/experiment_workflow.md)
- [项目进度](docs/project_progress.md)
