# SAC 框架说明

当前普通 SAC 实现在 `src/hypersonic_rl/agents/sac_agent.py`。

## Actor

`MLPActor` 使用 tanh-squashed Gaussian policy。网络输出 `mean` 和 `log_std`，采样时使用重参数化技巧得到动作，并通过 `action_scale` 和 `action_bias` 映射到环境动作范围。`log_prob` 包含 tanh 修正项，用于 SAC 熵正则。

## Twin Critic

`TwinQNetwork` 包含 Q1 和 Q2 两个独立分支。训练 Actor 和计算 target Q 时取二者较小值，以缓解 Q 值过估计。

## Replay Buffer

`ReplayBuffer` 使用 numpy 数组保存 `state`、`action`、`reward`、`next_state` 和 `done`，采样时转为 torch tensor 并移动到指定设备。

## Entropy Alpha

`log_alpha` 是可学习参数。若 `automatic_entropy_tuning=true`，训练会自动调节熵系数，使策略熵接近 `target_entropy`。配置为 `auto` 时，目标熵为 `-action_dim`。

## Soft Update

Critic target 网络通过

```text
target = (1 - tau) * target + tau * source
```

软更新，避免目标值剧烈震荡。

