"""
autopilot.py

作用：
    统一管理红方飞行器和蓝方拦截弹的一阶自动驾驶仪响应模型，以及二维机动过载限幅。

工程约定：
    1. 制导律或智能体输出的是“期望过载”；
    2. 自动驾驶仪把期望过载变成带动态滞后的“实际过载”；
    3. 二维过载限幅作用在 [ny - cos(theta), nz] 这两个真正消耗机动能力的分量上；
    4. 默认行为仍是论文/legacy 中常用的一阶惯性环节，速率限制和硬限幅只在配置显式打开时生效。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from hypersonic_rl.envs.dynamics import EPS


class FirstOrderAutopilot:
    """
    一阶自动驾驶仪。

    作用：
        根据一阶惯性环节模拟控制指令到实际过载输出的动态滞后。

    参数：
        tau：
            一阶惯性时间常数，单位 s。数值越大，响应越慢。
        initial_output：
            初始实际输出，可以是标量，也可以是 numpy 数组。
        rate_limit：
            可选输出变化率限制，单位 g/s；None 表示不启用速率限制。
        output_min/output_max：
            可选输出硬限幅；None 表示对应方向不限制。

    公式：
        output_raw = output_old + (command - output_old) * dt / tau
    """

    def __init__(
        self,
        tau: float,
        initial_output: Any = 0.0,
        rate_limit: Optional[float] = None,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
    ) -> None:
        """
        初始化一阶自动驾驶仪。

        参数：
            tau：
                一阶惯性时间常数，单位 s。
            initial_output：
                初始实际输出，支持 float 或 numpy 数组。
            rate_limit：
                可选输出变化率限制，单位 g/s。
            output_min/output_max：
                可选输出硬限幅。
        """
        # tau：保存自动驾驶仪响应时间常数，并避免后续除零。
        self.tau = max(float(tau), EPS)

        # output：保存当前自动驾驶仪实际输出。
        self.output = np.asarray(initial_output, dtype=np.float64).copy()

        # rate_limit：保存输出变化率限制；None 表示沿用纯一阶惯性模型。
        self.rate_limit = None if rate_limit is None else abs(float(rate_limit))

        # output_min/output_max：保存可选硬限幅，供带执行机构约束的工程模型使用。
        self.output_min = None if output_min is None else float(output_min)
        self.output_max = None if output_max is None else float(output_max)

        # last_info：缓存最近一步诊断字段，便于环境 info 和评估指标统一读取。
        self.last_info: Dict[str, Any] = {}

    @staticmethod
    def compute_response_with_info(
        command: Any,
        previous_output: Any,
        dt: float,
        tau: float,
        rate_limit: Optional[float] = None,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算带可选速率限制和输出限幅的一阶自动驾驶仪单步响应。

        参数：
            command：
                当前制导律或智能体给出的期望控制量。
            previous_output：
                上一仿真步的实际输出。
            dt：
                仿真步长，单位 s。
            tau：
                一阶惯性时间常数，单位 s。
            rate_limit：
                可选输出变化率限制，单位 g/s。
            output_min/output_max：
                可选输出硬限幅。

        返回：
            current_output：
                当前仿真步的实际输出。
            info：
                自动驾驶仪限速、限幅和中间量诊断。
        """
        # command_array：把输入指令统一转换为 numpy 数组，便于标量和向量共用。
        command_array = np.asarray(command, dtype=np.float64)

        # previous_array：把上一时刻输出统一转换为 numpy 数组。
        previous_array = np.asarray(previous_output, dtype=np.float64)

        # safe_tau：防止 tau 过小导致除零或响应发散。
        safe_tau = max(float(tau), EPS)

        # raw_output：纯一阶惯性输出，尚未经过速率限制或硬限幅。
        raw_output = previous_array + (command_array - previous_array) * float(dt) / safe_tau

        # rate_limited_output：经过速率限制后的输出；默认与 raw_output 相同。
        rate_limited_output = np.asarray(raw_output, dtype=np.float64).copy()

        # rate_saturated：标记本步是否触发输出变化率限制。
        rate_saturated = False
        if rate_limit is not None:
            # max_delta：单个仿真步允许的最大输出变化量。
            max_delta = abs(float(rate_limit)) * float(dt)

            # raw_delta：一阶惯性响应希望产生的输出变化量。
            raw_delta = raw_output - previous_array

            # clipped_delta：按速率限制裁剪后的输出变化量。
            clipped_delta = np.clip(raw_delta, -max_delta, max_delta)

            # rate_saturated：只要任一通道被裁剪，就记为本步速率饱和。
            rate_saturated = bool(np.any(np.abs(raw_delta - clipped_delta) > 1e-10))
            rate_limited_output = previous_array + clipped_delta

        # current_output：经过硬限幅后的最终输出。
        current_output = np.asarray(rate_limited_output, dtype=np.float64).copy()

        # output_saturated：标记本步是否触发输出硬限幅。
        output_saturated = False
        if output_min is not None or output_max is not None:
            # lower/upper：None 时使用无穷边界，保持 numpy.clip 的接口简单。
            lower = -np.inf if output_min is None else float(output_min)
            upper = np.inf if output_max is None else float(output_max)

            # clipped_output：按硬限幅裁剪后的输出。
            clipped_output = np.clip(current_output, lower, upper)
            output_saturated = bool(np.any(np.abs(current_output - clipped_output) > 1e-10))
            current_output = clipped_output

        # info：统一打包诊断字段，后续环境/评估脚本不再各自猜测饱和状态。
        info = {
            "raw_output": np.asarray(raw_output, dtype=np.float64).copy(),
            "rate_limited_output": np.asarray(rate_limited_output, dtype=np.float64).copy(),
            "output": np.asarray(current_output, dtype=np.float64).copy(),
            "rate_saturated": rate_saturated,
            "output_saturated": output_saturated,
        }

        return np.asarray(current_output, dtype=np.float64), info

    @staticmethod
    def compute_response(
        command: Any,
        previous_output: Any,
        dt: float,
        tau: float,
        rate_limit: Optional[float] = None,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        计算一阶惯性环节单步响应。

        参数：
            command：
                当前制导律或智能体给出的期望控制量。
            previous_output：
                上一仿真步的实际输出。
            dt：
                仿真步长，单位 s。
            tau：
                一阶惯性时间常数，单位 s。
            rate_limit：
                可选输出变化率限制，单位 g/s。
            output_min/output_max：
                可选输出硬限幅。

        返回：
            current_output：
                当前仿真步的实际输出。
        """
        # current_output：复用带诊断版本，旧接口只返回输出本身。
        current_output, _ = FirstOrderAutopilot.compute_response_with_info(
            command=command,
            previous_output=previous_output,
            dt=dt,
            tau=tau,
            rate_limit=rate_limit,
            output_min=output_min,
            output_max=output_max,
        )

        return current_output

    def reset(self, initial_output: Any = 0.0) -> None:
        """
        重置自动驾驶仪内部输出。

        参数：
            initial_output：
                重置后的实际输出，支持 float 或 numpy 数组。
        """
        # output：覆盖内部状态，使下一步响应从指定输出开始。
        self.output = np.asarray(initial_output, dtype=np.float64).copy()

        # last_info：重置后清空上一轮仿真的诊断缓存。
        self.last_info = {}

    def step_with_info(self, command: Any, dt: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        推进一步自动驾驶仪响应，并返回诊断字段。

        参数：
            command：
                当前仿真步期望控制量。
            dt：
                仿真步长，单位 s。

        返回：
            output：
                当前仿真步的实际输出。
            info：
                速率限制、硬限幅和中间输出诊断。
        """
        # output：用当前指令和上一输出计算新的响应，并写回内部状态。
        self.output, self.last_info = self.compute_response_with_info(
            command=command,
            previous_output=self.output,
            dt=dt,
            tau=self.tau,
            rate_limit=self.rate_limit,
            output_min=self.output_min,
            output_max=self.output_max,
        )

        return self.output.copy(), dict(self.last_info)

    def step(self, command: Any, dt: float) -> np.ndarray:
        """
        推进一步自动驾驶仪响应。

        参数：
            command：
                当前仿真步期望控制量。
            dt：
                仿真步长，单位 s。

        返回：
            output：
                当前仿真步的一阶惯性实际输出。
        """
        # output：保留旧接口，仅返回实际输出；诊断可通过 step_with_info() 或 last_info 读取。
        output, _ = self.step_with_info(command=command, dt=dt)

        return output


def limit_planar_overload(
    ny_command: float,
    nz_command: float,
    max_overload: float,
    theta: float,
) -> Tuple[float, float]:
    """
    对纵向和侧向机动过载做二维合成限幅。

    参数：
        ny_command：
            纵向总过载指令。
        nz_command：
            侧向过载指令。
        max_overload：
            允许的最大机动过载幅值，单位 g。
        theta：
            当前航迹倾角，单位 rad。

    返回：
        limited_ny：
            限幅后的纵向总过载。
        limited_nz：
            限幅后的侧向过载。

    说明：
        纵向通道包含维持当前航迹倾角的平衡项 cos(theta)。
        真正消耗机动能力的是 ny - cos(theta)，因此二维限幅对象是 [ny-cos(theta), nz]。
    """
    # equilibrium_ny：当前航迹倾角下维持平衡所需的近似纵向过载。
    equilibrium_ny = float(np.cos(float(theta)))

    # delta_ny：纵向机动过载分量，不包含平衡项。
    delta_ny = float(ny_command) - equilibrium_ny

    # delta_nz：侧向机动过载分量。
    delta_nz = float(nz_command)

    # maneuver_norm：二维合成机动过载幅值。
    maneuver_norm = float(np.sqrt(delta_ny * delta_ny + delta_nz * delta_nz))

    # max_overload_value：最大允许机动过载。
    max_overload_value = float(max_overload)

    if maneuver_norm <= max_overload_value:
        return float(ny_command), float(nz_command)

    # scale：超过约束时按比例缩放两个机动分量，保持方向不变。
    scale = max_overload_value / max(maneuver_norm, EPS)

    # limited_delta_ny：限幅后的纵向机动分量。
    limited_delta_ny = delta_ny * scale

    # limited_delta_nz：限幅后的侧向机动分量。
    limited_delta_nz = delta_nz * scale

    # limited_ny：把纵向平衡项加回总过载。
    limited_ny = equilibrium_ny + limited_delta_ny

    # limited_nz：侧向通道本身就是机动分量。
    limited_nz = limited_delta_nz

    return float(limited_ny), float(limited_nz)
