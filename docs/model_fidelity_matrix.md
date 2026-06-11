# 模型保真矩阵

本文件用于记录论文、`legacy/` 原始代码和当前工程实现之间的对应关系。目标不是把模型升级到六自由度，而是在三自由度质点模型范围内把符号、单位、初始态势、制导闭环和指标口径讲清楚，便于复现实验和后续论文对比。

## 总览

| 项目 | 论文/legacy 基准 | 当前工程实现 | 验证位置 |
|---|---|---|---|
| 坐标轴 | `x` 为纵向距离，`y` 为高度，`z` 为横向距离 | 保持 `x/y/z` 三维坐标，速度由 `build_velocity_vector()` 统一生成 | `tests/test_guidance_closure.py` |
| 航向符号 | `z_dot = -V cos(theta) sin(psi)` | 已保持该符号；`psi=pi` 沿 `x` 负方向飞行 | `build_velocity_vector()` |
| 相对几何 | 拦截弹指向目标 | `compute_relative_geometry()` 使用 `red - interceptor`，`range_rate<0` 表示接近 | `compute_relative_geometry()` |
| 重力常数 | 标准重力加速度 | `GRAVITY = 9.81` | `test_gravity_and_integration_mode_are_explicit` |
| 动力学模型 | 三自由度质点 | 默认三自由度质点，不含发动机/气动/六自由度 | `update_point_mass_state()` |
| 积分模式 | legacy 等价离散更新 | 默认 `semi_implicit_euler`，可切换 `explicit_euler` 做对照 | `dynamics_integration_mode` |
| 自动驾驶仪 | 一阶惯性 | `FirstOrderAutopilot` 统一一阶响应，可选速率限制和输出限幅 | `tests/test_autopilot.py` |
| 红方动作 | 1 维横向过载 | `action=[nzc_h]`，动作空间不变 | `PursueEscapeEnv.action_space` |
| 蓝方制导 | PN/中末制导 | `source_pn` 与 `mid_terminal_interceptor` 两条 baseline，均为双通道 `ny/nz` | `InterceptorFleet` |
| 命中判据 | 杀伤半径 | 连续线段最近点判据，避免离散步长漏判 | `compute_segment_closest_distance()` |
| 终止判据 | 命中或错过 | 任一弹命中则红方失败；全部弹 passed 且未命中则红方成功 | `InterceptorFleet.step()` |
| 观测状态 | 第 5 章 10 维状态 | `[r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]` | `PursueEscapeEnv._build_observation()` |
| LSTM 输入 | 连续状态序列 | `StateSequenceWrapper` 输出 `[sequence_length, 10]`，默认 3 | `tests/test_lstm_sac_components.py` |
| 奖励 | 第 5 章端到端奖励 | 过程约束红方机动并鼓励接近目标；终端按连续最小脱靶量计算 | `reward.py` |
| 能耗指标 | 过载积分 | 同时输出平方能耗 `sum(u^2)dt` 和论文式 `sum(|u|)dt` | `metrics.py` |
| 实验复现 | 记录 seed/config | 训练、评估、benchmark 均输出 `run_manifest.json` | `utils/manifest.py` |

## 坐标与符号

当前工程约定每个飞行器状态为：

```text
[x, y, z, speed, theta, psi, nx, ny, nz]
```

速度分量为：

```text
vx = V cos(theta) cos(psi)
vy = V sin(theta)
vz = -V cos(theta) sin(psi)
```

因此侧向制导符号必须和 `psi_dot = -g*nz/(V cos(theta))` 配套使用。所有 PN、MPN/ZEM 和目标机动前馈都按这个符号约定实现。

## 初始态势

默认训练 profile 为 `paper_200km_end_to_end`：

- 红方：`(0, 25km, 0)`，`psi=0deg`，Mach 6。
- 拦截弹 1：`(200km, 25km, -10km)`，`psi=180deg`，Mach 4。
- 拦截弹 2：`(200km, 25km, +10km)`，`psi=180deg`，Mach 4。
- 训练随机化：拦截弹位置默认 `±3km`，初始航向默认 `±3deg`。

近距制导调试 profile 为 `debug_50km_head_on`，只用于快速验证闭环，不作为论文结果。

## 当前非范围

- 不实现六自由度刚体动力学。
- 不实现发动机、气动数据库、热防护或结构约束。
- 不把第 4 章微分对策解析制导作为 active baseline；`differential_game_guidance_placeholder()` 只返回非激活状态说明。
- 不在本轮运行 1200 episode 长训练或 5000 次 Monte Carlo，只提供正式入口和小规模验证链路。
