# 环境模型

当前环境为端到端 SAC 追逃仿真，代码位于 `src/hypersonic_rl/envs/pursue_escape_env.py`。

## 状态

红方状态 `red_state` 和蓝方状态 `blue_state` 均为 9 维：

```text
[x, y, z, speed, theta, psi, nx, ny, nz]
```

其中 `x,y,z` 是位置，`speed` 是速度大小，`theta` 是弹道倾角，`psi` 是弹道偏角，`nx,ny,nz` 是三轴过载。

## 观测空间

观测为红蓝双方前 6 个状态拼接：

```text
[red_x, red_y, red_z, red_speed, red_theta, red_psi,
 blue_x, blue_y, blue_z, blue_speed, blue_theta, blue_psi]
```

因此观测维度为 12。

## 动作空间

动作是 1 维红方横向机动过载，范围为 `[-nzc_h_max, nzc_h_max]`。当前默认值为 `[-3, 3]`。

## 奖励函数

奖励由四部分组成：

- 动作能量惩罚：抑制过大机动。
- 最小距离奖励：脱靶后按最小距离给奖励。
- 视线角速度奖励：鼓励红方制造横向几何变化。
- 横向脱靶奖励：鼓励红蓝双方在 Z 方向拉开距离。

奖励函数返回 `reward_info`，用于训练后分析各项贡献。

## 终止条件

当蓝方在 X 方向越过红方超过 50 m 时，认为本轮追逃结束。达到 `max_time / dt` 步数时返回 `truncated=True`。

## 与原始代码对应关系

原始 `PursueEscapeEnv.py` 中的状态更新逻辑拆入 `dynamics.py`，蓝方 `calculate_blue_control()` 拆入 `guidance.py`，奖励 `calculate_reward()` 拆入 `reward.py`，轨迹记录统一由 `get_episode_trace()` 返回。

