# 项目设计

本项目采用 `src/` 布局，将环境、智能体、网络、缓存、训练器、评估器和可视化拆成独立模块。原始代码完整保存在 `src/hypersonic_rl/legacy/`，新工程代码不直接修改 legacy 文件。

## 当前主线

当前主线是论文第 5 章一对二端到端突防任务：

```text
双拦截弹环境 -> 10 维态势观测 -> SAC/LSTM-SAC 输出红方横向过载 -> 环境推进 -> 奖励和诊断
```

普通 SAC 保留为基线；LSTM-SAC 是第 5 章主训练入口。

## 数据流

1. `PursueEscapeEnv.reset()` 生成红方和两枚拦截弹初始状态。
2. 原始环境输出 10 维论文式观测 `[r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]`。
3. 普通 SAC 直接使用 10 维观测；LSTM-SAC 通过 `StateSequenceWrapper` 使用 `[sequence_length, 10]` 状态序列。
4. `SACAgent` 或 `LSTMSACAgent` 输出 1 维红方横向过载动作。
5. `PursueEscapeEnv.step()` 推进红方动力学，并通过 `InterceptorFleet` 独立推进每枚拦截弹。
6. `reward.py` 根据过程能耗、目标接近和终端脱靶量计算端到端奖励。
7. `ReplayBuffer` 或 `SequenceReplayBuffer` 保存 transition，训练器调用 agent 更新网络。
8. 训练器保存 checkpoint、metrics、loss 曲线，并可周期性调用 evaluator 输出轨迹图。

## 模块职责

- `envs/`：环境主类、动力学、拦截弹制导、编队管理、奖励、自动驾驶仪和状态序列包装。
- `networks/`：普通 MLP Actor/Critic 与第 5 章 LSTM Actor/Critic。
- `agents/`：普通 `SACAgent` 与 `LSTMSACAgent`，二者共享 SAC 更新思想但输入形态不同。
- `buffers/`：普通单步经验池与固定窗口序列经验池。
- `trainers/`：普通 SAC 训练循环与 LSTM-SAC 序列训练循环。
- `evaluation/`：统一输出累计奖励、最小距离、成功率、每枚拦截弹脱靶量和控制能耗。
- `visualization/`：保存训练曲线、episode 诊断图、三视图轨迹和三维轨迹图。
- `scripts/`：debug、标定、普通 SAC、双弹 SAC、课程训练和 LSTM-SAC 入口。

## 设计约束

- 红方动作维度保持 1，不在本轮引入纵向动作。
- 默认环境观测为 10 维，LSTM-SAC 只在 wrapper 层扩展为状态序列。
- 多拦截弹诊断统一使用 `interceptor_{i}_*` 字段，不再使用含糊的单个蓝方诊断字段。
- 普通 SAC 与 LSTM-SAC 并存，便于对比第 5 章记忆增强方法的训练效率。
