# SAC 与 LSTM-SAC 框架说明

当前项目同时保留普通 SAC baseline 和论文第 5 章 LSTM-SAC 主线。两者都面向连续动作控制，区别在于输入状态形态、网络结构和 replay buffer。

## 普通 SAC

实现位置：

- Agent：`src/hypersonic_rl/agents/sac_agent.py`
- Actor/Critic：`src/hypersonic_rl/networks/mlp_actor.py`、`src/hypersonic_rl/networks/mlp_critic.py`
- Buffer：`src/hypersonic_rl/buffers/replay_buffer.py`
- Trainer：`src/hypersonic_rl/trainers/sac_trainer.py`
- 入口：`scripts/train_sac.py`

特征：

- 输入单步 10 维论文式观测。
- Actor 为 tanh-squashed Gaussian MLP policy。
- Critic 为双 Q MLP。
- ReplayBuffer 保存 `state/action/reward/next_state/done`。

正式 paper baseline：

```bash
python scripts/train_sac.py --train-config configs/train/train_sac_paper.yaml
```

## LSTM-SAC

实现位置：

- Agent：`src/hypersonic_rl/agents/lstm_sac_agent.py`
- Actor/Critic：`src/hypersonic_rl/networks/lstm_actor.py`、`src/hypersonic_rl/networks/lstm_critic.py`
- Buffer：`src/hypersonic_rl/buffers/sequence_replay_buffer.py`
- Wrapper：`src/hypersonic_rl/envs/wrappers.py`
- Trainer：`src/hypersonic_rl/trainers/lstm_sac_trainer.py`
- 入口：`scripts/train_lstm_sac.py`

特征：

- 输入固定窗口状态序列 `[sequence_length, 10]`。
- 默认 `sequence_length=3`，可配置为 `1/5` 做消融。
- Actor/Critic 使用单向 LSTM 时序编码。
- SequenceReplayBuffer 保存 `state_sequence/action/reward/next_state_sequence/done`。

正式第 5 章训练：

```bash
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac.yaml
```

## SAC 损失

两种 agent 使用相同 SAC 目标：

```text
target_q = reward + (1 - done) * gamma * (min(Q1, Q2) - alpha * log_prob)
actor_loss = E[alpha * log_prob - min(Q1, Q2)]
```

`log_alpha` 可学习；若 `automatic_entropy_tuning=true`，训练会自动调节熵系数，使策略熵接近 `target_entropy`。

## 训练输出

训练器统一输出：

- checkpoint：`checkpoints/latest.pth`、`checkpoints/final.pth`、episode checkpoint。
- metrics：`metrics/training_metrics.csv`、`metrics/loss_metrics.csv`。
- figures：奖励、脱靶量、成功率和 loss 曲线。
- manifest：`run_manifest.json`，记录配置、seed、git 状态和 checkpoint 路径。

## 恢复训练

训练配置支持：

```yaml
resume_from_checkpoint: experiments/lstm_sac_chapter5/checkpoints/latest.pth
resume_load_optimizers: true
```

恢复内容包括 agent、优化器、`global_step`、`update_step` 和历史 metrics。当前 replay buffer 不随 checkpoint 恢复，恢复后会重新积累经验。

## 评估与 benchmark

统一评估：

```bash
python scripts/eval_policy.py --config configs/eval/eval_policy.yaml
```

论文式 benchmark：

```bash
python scripts/run_paper_benchmark.py --config configs/eval/benchmark_paper.yaml
```

评估和 benchmark 统一输出成功率、连续最小脱靶量、红方能耗、目标偏移、每枚弹遭遇时间、饱和比例、CSV、图像和 manifest。

## 第 5 章默认参数

`configs/agent/lstm_sac.yaml` 对齐论文第 5 章默认训练设置：

- `sequence_length=3`
- `replay_buffer_size=30000`
- `batch_size=128`
- `gamma=0.98`
- `actor_lr=3e-3`
- `critic_lr=5e-3`
- `alpha_lr=1e-3`

长训练结果需要通过正式配置实际运行获得；仓库默认不附带 1200 episode 训练产物。
