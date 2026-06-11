# 实验流程

本文件给出当前项目的推荐实验流程。原则是先验证模型闭环和输出链路，再启动长训练，最后用统一评估与 benchmark 做对比。

## 1. 基础回归

```bash
python -m pytest
```

覆盖内容包括：

- 动力学符号和积分模式。
- 自动驾驶仪一阶响应、速率限制和二维过载限幅。
- 双拦截弹命中/错过判据。
- LSTM-SAC 网络、agent、sequence replay buffer 和 trainer smoke。
- 评估指标与可视化输出。

## 2. 蓝方制导 debug

```bash
python scripts/debug_two_interceptor_compare.py --seeds 0 --max-time 20
```

用于检查 `source_pn` 与 `mid_terminal_interceptor` 在同一初始条件和红方动作序列下的差异。重点查看：

- 每枚弹连续最小脱靶量。
- `ny/nz` 指令和实际输出。
- 阶段切换时间。
- 指令饱和和自动驾驶仪速率饱和。
- 目标机动前馈是否按预期生效。

## 3. Smoke 验证

```bash
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac_smoke.yaml
python scripts/run_paper_benchmark.py --config configs/eval/benchmark_smoke.yaml
```

Smoke 只用于确认本机环境和输出链路正常，不作为论文结果。

## 4. 正式训练

普通 SAC baseline：

```bash
python scripts/train_sac.py --train-config configs/train/train_sac_paper.yaml
```

第 5 章 LSTM-SAC：

```bash
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac.yaml
```

如需从 `latest.pth` 恢复训练，在训练 YAML 中设置：

```yaml
resume_from_checkpoint: experiments/lstm_sac_chapter5/checkpoints/latest.pth
resume_load_optimizers: true
```

当前恢复逻辑会加载 agent、优化器、训练计数器和历史 metrics；replay buffer 不随 checkpoint 恢复，恢复后会重新积累经验。

## 5. 统一评估

```bash
python scripts/eval_policy.py --config configs/eval/eval_policy.yaml
```

可通过命令行覆盖：

```bash
python scripts/eval_policy.py --agent-type sac --checkpoint experiments/sac_paper_baseline/checkpoints/latest.pth --output-dir outputs/evaluation/sac_paper
python scripts/eval_policy.py --agent-type lstm_sac --checkpoint experiments/lstm_sac_chapter5/checkpoints/latest.pth --output-dir outputs/evaluation/lstm_sac_paper
```

输出：

- `metrics/evaluation_metrics.csv`
- `metrics/evaluation_summary.csv`
- `figures/` 轨迹图
- `run_manifest.json`

## 6. 论文式 benchmark

```bash
python scripts/run_paper_benchmark.py --config configs/eval/benchmark_paper.yaml
```

默认覆盖：

- 固定动作 baseline：`zero_action`、`sine_action`、`random_action`。
- 可选普通 SAC checkpoint。
- 可选 LSTM-SAC checkpoint。
- 蓝方模式：`source_pn`、`mid_terminal_interceptor`。
- seeds：`0..4`。

输出指标：

- 成功率及 95% 置信区间。
- 平均连续最小脱靶量。
- 红方平方能耗和 `sum(|nHz|)dt` 论文式能耗。
- 目标偏移 `target_offset`。
- 每枚弹命中/错过/阶段切换时间。
- 指令饱和和自动驾驶仪饱和比例。

## 7. 结果归档

每次训练、评估和 benchmark 均会输出 `run_manifest.json`，其中包括：

- 配置快照。
- seed。
- checkpoint 路径。
- 环境 profile。
- git commit、branch、dirty 状态。
- 运行脚本的额外上下文。

论文结果建议同时归档 `run_manifest.json`、CSV、图像和报告，避免后续无法追溯实验设置。
