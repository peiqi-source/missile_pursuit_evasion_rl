# 第 5 章 LSTM-SAC 实现说明

本文件记录论文第 5 章“基于记忆增强深度强化学习算法的突防策略设计”在当前工程中的实现方式。LSTM-SAC 已进入可训练、可评估、可 benchmark 的工程状态；smoke 配置仅用于快速验证。

## 论文接口

- 状态空间：10 维端到端观测 `[r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]`。
- 动作空间：红方 1 维横向过载指令 `action=[nzc_h]`。
- 状态序列：LSTM-SAC 输入连续状态序列 `[s_{t-n+1}, ..., s_t]`，默认 `n=3`。
- 网络结构：Actor/Critic 使用单向 LSTM，默认层宽 `256 -> 256 -> 128`；Actor 接 FC128 后输出 tanh-squashed Gaussian 动作；Critic 拼接动作后接 `128 -> 128 -> 64 -> Q`。
- 训练参数：默认 `gamma=0.98`、Actor lr `3e-3`、Critic lr `5e-3`、alpha lr `1e-3`、batch size `128`、序列 replay buffer 容量 `30000`。

## 工程实现

- `StateSequenceWrapper`：将原始 10 维观测包装为 `[sequence_length, 10]`；reset 时用初始观测填满窗口，step 时滚动追加最新观测。
- `SequenceReplayBuffer`：直接保存固定窗口 transition，不做 episode 动态重采样。
- `LSTMActor`：实现 LSTM 时序编码、高斯策略、tanh 动作限幅和 log-prob 修正。
- `LSTMTwinCritic`：实现双 Q LSTM Critic 和 target 双 Q。
- `LSTMSACAgent`：训练公式与普通 SAC 对齐，输入改为 `[B, L, 10]`。
- `LSTMSACTrainer`：复用普通 SAC 训练主循环，覆盖 transition 写入逻辑以匹配序列 replay buffer。
- `scripts/train_lstm_sac.py`：第 5 章训练入口。
- `scripts/eval_policy.py`：统一 SAC/LSTM-SAC 评估入口，会为 LSTM-SAC 自动包装状态序列。

## 命令

快速链路验证：

```bash
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac_smoke.yaml
```

正式 paper profile 训练：

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

序列长度消融时，修改 `configs/agent/lstm_sac.yaml` 中的 `sequence_length` 为 `1`、`3` 或 `5`。

## 输出

训练输出位于 `experiments/lstm_sac_chapter5/`：

- `checkpoints/latest.pth`、`checkpoints/final.pth`。
- `metrics/training_metrics.csv`、`metrics/loss_metrics.csv`。
- `figures/` 训练曲线。
- `run_manifest.json`，记录配置、seed、git 状态、checkpoint 路径和恢复信息。

评估和 benchmark 输出位于 `outputs/`，包含逐回合 CSV、summary CSV、轨迹图、Markdown 报告和 manifest。

## 验收标准

- `python -m pytest` 全量通过。
- LSTM-SAC smoke 能生成 checkpoint、metrics、loss metrics 和训练曲线。
- `eval_policy.py` 能加载 SAC 与 LSTM-SAC checkpoint。
- `run_paper_benchmark.py` 能输出固定动作 baseline、可选 checkpoint baseline、summary 和 report。
- 文档和 active 代码均将 LSTM-SAC 描述为可运行、可训练、可评估的正式工程入口。

## 当前边界

- 第一版采用固定窗口序列缓存，不实现 episode 动态序列采样。
- 当前 LSTM 隐状态不跨 replay 样本持久化；历史信息由输入序列承担。
- 本轮不提供 1200 episode 长训练结果和 5000 次 Monte Carlo 统计结果，只提供可复现实验入口和小规模工程验证。
