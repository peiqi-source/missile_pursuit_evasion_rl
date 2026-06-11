# 论文第 4 章映射说明

论文第 4 章研究的是“基于 SAC 的制导律参数智能设计”。核心链路是：

```text
状态观测 -> SAC Actor 输出制导律参数/导航增益 -> 解析制导律计算最终过载指令
```

当前工程主线已经转向第 5 章端到端突防任务：

```text
状态观测或状态序列 -> SAC/LSTM-SAC Actor 直接输出红方横向过载 -> 环境动力学更新
```

因此当前 active baseline 是第 5 章的一对二端到端 SAC/LSTM-SAC，不是第 4 章的参数化解析制导。

## 当前代码状态

`src/hypersonic_rl/envs/guidance.py` 中保留 `differential_game_guidance_placeholder()`，该函数只返回结构化非激活状态说明：

```python
{
    "active": False,
    "baseline_name": "chapter4_differential_game_guidance",
    "reason": "...",
}
```

它不会参与训练、评估或 benchmark，也不会作为蓝方/红方 active 制导律。

## 与第 5 章的关系

第 5 章 LSTM-SAC 不再输出制导律参数，而是直接输出红方横向过载。当前项目优先保证：

- 双拦截弹端到端环境可复现。
- 普通 SAC 与 LSTM-SAC 可训练、可评估。
- `source_pn` 与 `mid_terminal_interceptor` 两条蓝方 baseline 可公平对比。
- 论文式 benchmark 可输出成功率、脱靶量、能耗、目标偏移和置信区间。

若后续需要完整复现第 4 章，可新增独立解析制导模块和独立 benchmark 分组，不应混入当前第 5 章训练主链路。
