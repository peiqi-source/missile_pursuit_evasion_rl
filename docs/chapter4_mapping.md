# 论文第 4 章映射

论文第 4 章研究的是基于 SAC 的制导律参数智能设计。其核心链路是：

```text
状态观测 -> SAC Actor 输出微分对策制导律参数/导航增益 -> 制导律计算最终过载指令
```

当前原始代码和本工程第一阶段保留的链路是：

```text
状态观测 -> SAC Actor 直接输出红方横向机动过载 -> 环境动力学更新
```

因此，当前实现更接近端到端 SAC 控制，而不是论文第 4 章的“参数优化式 SAC”。

## 预留扩展位置

`src/hypersonic_rl/envs/guidance.py` 中预留了：

- `differential_game_guidance()`
- `differential_game_parameterized_guidance()`

后续扩展时，可让 Actor 输出导航系数、增益或其他制导律参数，再由 `differential_game_parameterized_guidance()` 根据论文第 3、4 章公式生成红方或蓝方过载指令。这样可以保持环境和训练器接口稳定，只替换动作解释和制导律计算。

