# 环境模型

当前环境位于 `src/hypersonic_rl/envs/pursue_escape_env.py`，默认实现论文第 5 章一对二端到端突防任务。

## 参与方

- 红方：高超声速飞行器，由 SAC/LSTM-SAC 输出 1 维横向过载动作。
- 蓝方：一枚或两枚拦截弹，默认两枚；每枚弹独立运行 `source_pn` 或 `mid_terminal_interceptor` 制导闭环。

每个飞行器状态为 9 维：

```text
[x, y, z, speed, theta, psi, nx, ny, nz]
```

其中 `x/y/z` 为位置，`speed` 为速度大小，`theta` 为航迹倾角，`psi` 为水平航向角，`nx/ny/nz` 为过载状态。

## 动力学

当前模型为三自由度质点模型：

```text
v_dot     = g * (nx - sin(theta))
theta_dot = g / v * (ny - cos(theta))
psi_dot   = -g * nz / (v cos(theta))
```

工程实现使用 `g=9.81`。位置积分模式由 `dynamics_integration_mode` 控制：

- `semi_implicit_euler`：默认模式，先更新速度和角度，再推进位置，保持当前工程闭环行为。
- `explicit_euler`：使用当前速度和角度推进位置，主要用于模型保真对照测试。

速度分量统一由 `build_velocity_vector()` 生成：

```text
vx = V cos(theta) cos(psi)
vy = V sin(theta)
vz = -V cos(theta) sin(psi)
```

## 初始态势

默认 profile 为 `paper_200km_end_to_end`：

- 红方初始位置 `(0, 25km, 0)`，初始航向 `psi=0deg`。
- 两枚拦截弹初始位置 `(200km, 25km, -10km)` 与 `(200km, 25km, +10km)`，初始航向约 `180deg`。
- 默认训练随机化：拦截弹发射位置 `±3km`，初始航向 `±3deg`。

近距调试 profile `debug_50km_head_on` 仅用于快速检查制导闭环。

## 观测空间

原始环境观测为论文第 5 章 10 维端到端状态：

```text
[r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]
```

- `ri`：红方与第 `i` 枚拦截弹的相对距离归一化值。
- `dri`：signed range-rate 归一化值，小于 0 表示双方接近。
- `qi`：水平 `X-Z` 平面视线角归一化值。
- `dqi`：水平视线角速率。
- `rHT`：红方到预设打击目标距离归一化值。
- `nHz`：红方实际横向过载归一化值。

LSTM-SAC 训练时，`StateSequenceWrapper` 将观测扩展为：

```text
[sequence_length, 10]
```

默认 `sequence_length=3`。

## 动作空间

红方动作固定为 1 维横向过载指令：

```text
action = [nzc_h]
```

动作范围为 `[-nzc_h_max, nzc_h_max]`，论文式默认红方能力为 `2g`。红方动作经过 `FirstOrderAutopilot` 一阶惯性响应后成为实际横向过载。

## 蓝方制导

蓝方支持两条 baseline：

- `source_pn`：论文/legacy 风格修正比例导引，按每枚弹独立计算 `ny/nz`。
- `mid_terminal_interceptor`：中制导 + 末制导框架，中制导以 LOS-rate PN 为主，末制导使用 MPN/ZEM 结构，并支持红方横向机动前馈。

两种模式均使用：

- 双通道 `ny/nz` 指令。
- 二维机动过载限幅 `[ny-cos(theta), nz]`。
- 一阶自动驾驶仪 `FirstOrderAutopilot`。
- 可选自动驾驶仪速率限制 `interceptor_autopilot_rate_limit`。

## 终止条件

- `intercepted=True`：任一拦截弹连续最近点进入 `kill_radius`，回合终止，红方失败。
- `success=True`：全部拦截弹 passed 且未命中，红方突防成功。
- `truncated=True`：达到最大仿真步数。

连续最近点命中判据检查一步更新前后红蓝相对位置线段到原点的最小距离，可避免高速交会时离散步长漏判。

## 诊断字段

全局字段包括：

- `min_distance`
- `closest_distance_this_step`
- `threat_interceptor_id`
- `intercepted`
- `success`
- `termination_reason`

每枚弹字段采用 `interceptor_{i}_*`：

- `interceptor_1_min_distance`
- `interceptor_1_phase`
- `interceptor_1_ny_command`
- `interceptor_1_nz_command`
- `interceptor_1_ny_actual`
- `interceptor_1_nz_actual`
- `interceptor_1_planar_saturation_ratio`
- `interceptor_1_command_saturated`
- `interceptor_1_autopilot_rate_saturated`
- `interceptor_1_phase_switch_time`
- `interceptor_1_pass_time`

训练、评估、benchmark CSV 均使用该 per-interceptor 命名。
