# 第 5 章 LSTM-SAC 预留计划

第 5 章 LSTM-SAC 需要把单步状态输入扩展为历史状态序列输入。当前阶段不完整实现 LSTM-SAC，只保留结构和接口。

## 已预留模块

- `LSTMActor`：接收 `[batch, sequence_length, state_dim]` 状态序列，输出动作分布参数。
- `LSTMCritic`：接收状态序列和动作，输出 Q1/Q2。
- `SequenceReplayBuffer`：按 episode 保存连续 transition，后续采样固定长度序列。
- `StateSequenceWrapper`：把普通环境观测包装为固定长度状态序列。
- `LSTMSACAgent`：后续复用 SAC 更新逻辑并替换网络输入形式。
- `LSTMSACTrainer`：后续组合序列环境、序列缓存和 LSTM-SAC 智能体。

## 后续实现步骤

1. 明确论文第 5 章的状态序列构造方式和序列长度。
2. 完成 `SequenceReplayBuffer.sample()`，保证序列不跨 episode 边界。
3. 在 `LSTMActor` 中补 tanh-squashed Gaussian 采样与 log_prob 修正。
4. 在 `LSTMSACAgent` 中实现 target Q、Critic、Actor 和 alpha 更新。
5. 使用 `StateSequenceWrapper` 运行端到端训练，并与普通 SAC baseline 对比。

