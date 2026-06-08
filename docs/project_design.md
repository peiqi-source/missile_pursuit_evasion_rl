# 项目设计

本项目采用 `src/` 布局，将环境、智能体、网络、缓存、训练器、评估器和可视化拆成独立模块。原始三文件完整保存在 `src/hypersonic_rl/legacy/`，新代码不直接修改 legacy 文件。

## 数据流

1. `PursueEscapeEnv.reset()` 生成红方和蓝方初始状态。
2. `SACAgent.select_action()` 根据 12 维观测输出 1 维红方横向过载。
3. `PursueEscapeEnv.step()` 调用 `dynamics.py` 更新红方状态，调用 `guidance.py` 计算蓝方比例导引，再更新蓝方状态。
4. `reward.py` 返回总奖励和奖励项明细。
5. `ReplayBuffer` 保存 transition，`SACAgent.update_parameters()` 完成 Critic、Actor 和 entropy alpha 更新。
6. `SACTrainer` 负责训练循环、warmup、定期评估、checkpoint、metrics.csv 和图片保存。

## 模块职责

- `envs/`：环境主类只组织交互流程，动力学、制导和奖励分别在独立文件中实现。
- `networks/`：普通 SAC 使用 MLP Actor 和 Twin Critic；LSTM 网络只保留第 5 章接口。
- `agents/`：`SACAgent` 完整实现普通 SAC；`LSTMSACAgent` 为后续扩展骨架。
- `trainers/`：`SACTrainer` 负责工程训练流程；`LSTMSACTrainer` 为后续预留。
- `evaluation/`：统一输出平均奖励、最小距离、成功率和控制能量。
- `visualization/`：只保存图片，不在训练中强制 `plt.show()`。

