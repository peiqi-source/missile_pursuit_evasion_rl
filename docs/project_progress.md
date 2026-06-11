# 项目进度总结

## 已完成能力

- 双拦截弹端到端环境：默认两枚拦截弹、论文式 10 维观测、红方 1 维横向过载动作。
- 蓝方制导闭环：`source_pn` 与 `mid_terminal_interceptor` 均支持每枚弹独立 `ny/nz` 双通道、一阶自动驾驶仪、二维限幅、目标机动前馈和连续最近点命中判据。
- 模型保真基础：统一 `g=9.81`、显式积分模式、相对几何符号、速度分量符号和环境 profile。
- 普通 SAC baseline：MLP Actor/Critic、ReplayBuffer、Trainer、周期评估、checkpoint 和曲线输出。
- 第 5 章 LSTM-SAC：状态序列包装、固定窗口序列 replay buffer、LSTM Actor/Critic、LSTMSACAgent、LSTMSACTrainer 和训练入口。
- 指标体系：连续最小脱靶量、成功率、每枚弹 miss/pass/intercept/phase 时间、平方能耗、论文式绝对过载积分、目标偏移和饱和比例。
- 工程复现：训练、评估、benchmark 输出 `run_manifest.json`，包含配置、seed、checkpoint、git 状态和环境 profile。
- 实验入口：`eval_policy.py` 支持 SAC/LSTM-SAC；`run_paper_benchmark.py` 支持固定动作 baseline、checkpoint baseline 和两种蓝方制导模式。

## 当前默认接口

- 环境：`PursueEscapeEnv`。
- 默认 profile：`paper_200km_end_to_end`。
- 默认蓝方数量：`interceptor_count=2`。
- 默认观测：`[r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]`。
- 默认 LSTM 序列：`[sequence_length, 10]`，`sequence_length=3`。
- 默认动作：`action=[nzc_h]`。
- 默认诊断：全局字段 + `interceptor_{i}_*` per-interceptor 字段。

## 推荐验证顺序

```bash
python -m pytest
python scripts/debug_two_interceptor_compare.py --seeds 0 --max-time 20
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac_smoke.yaml
python scripts/run_paper_benchmark.py --config configs/eval/benchmark_smoke.yaml
```

## 正式实验入口

```bash
python scripts/train_sac.py --train-config configs/train/train_sac_paper.yaml
python scripts/train_lstm_sac.py --train-config configs/train/train_lstm_sac.yaml
python scripts/eval_policy.py --config configs/eval/eval_policy.yaml
python scripts/run_paper_benchmark.py --config configs/eval/benchmark_paper.yaml
```

## 当前边界与风险

- 当前动力学仍为论文/legacy 可对照的三自由度质点模型，不是六自由度高保真飞行器模型。
- 当前没有发动机、气动数据库、热防护或结构约束。
- LSTM-SAC 第一版使用固定窗口 replay buffer，不实现 episode 动态序列采样。
- 恢复训练不恢复 replay buffer，恢复后需要重新积累经验。
- 本轮不提供 1200 episode 长训练结果和 5000 次 Monte Carlo 统计结果，只提供正式工程入口和小规模验证。
- 第 4 章微分对策解析制导不是当前 active baseline，只保留为后续可扩展方向。

## 后续建议

- 在固定 `paper_200km_end_to_end` profile 下跑普通 SAC 与 LSTM-SAC 长训练，并用 benchmark 汇总成功率、脱靶量、红方能耗和目标偏移。
- 做 `sequence_length=1/3/5` 消融，验证论文中 `n=3` 的效率优势是否在当前工程参数下成立。
- 对蓝方能力 profile 做 `weak/paper/strong` 扫描，确定训练难度曲线。
- 若需要更高模型保真，可在保持当前接口不变的前提下扩展自动驾驶仪二阶响应、执行机构带宽和更完整的飞行力学模型。
