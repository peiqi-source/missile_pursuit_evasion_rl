"""
interceptor_fleet.py

作用：
    管理一枚或两枚蓝方拦截弹的闭环制导、状态推进和命中/错过判据。

设计说明：
    环境只负责红方智能体、奖励和 Gym 接口；本文件负责蓝方拦截弹编队。
    每枚拦截弹拥有独立状态、独立自动驾驶仪输出、独立最小脱靶量和独立阶段状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from hypersonic_rl.envs.dynamics import compute_relative_geometry, update_point_mass_state
from hypersonic_rl.envs.guidance import (
    ProportionalNavigationConfig,
    compute_interceptor_proportional_navigation,
)
from hypersonic_rl.envs.interceptor import Interceptor, InterceptorConfig


def compute_segment_closest_distance(start_relative_position: np.ndarray, end_relative_position: np.ndarray) -> float:
    """
    计算一个仿真步长内相对位置线段到原点的最近距离。

    参数：
        start_relative_position：
            步长开始时的相对位置，可以使用 blue-red 或 red-blue，距离结果不受符号影响。
        end_relative_position：
            步长结束时的相对位置。

    返回：
        closest_distance：
            线段最近点到原点的距离，单位 m。
    """
    # start_position/end_position：统一转成浮点向量，避免调用方传入列表导致广播异常。
    start_position = np.asarray(start_relative_position, dtype=np.float64)
    end_position = np.asarray(end_relative_position, dtype=np.float64)

    # segment_delta：相对位置在当前步长内的变化量。
    segment_delta = end_position - start_position

    # segment_norm_squared：线段长度平方，用于判断是否退化为一个点。
    segment_norm_squared = float(np.dot(segment_delta, segment_delta))

    if segment_norm_squared <= 1e-12:
        return float(np.linalg.norm(end_position))

    # interpolation：原点在线段方向上的投影参数，截断到 [0, 1] 后得到线段最近点。
    interpolation = -float(np.dot(start_position, segment_delta)) / segment_norm_squared
    interpolation = float(np.clip(interpolation, 0.0, 1.0))

    # closest_position：当前步长线段上距离原点最近的相对位置。
    closest_position = start_position + interpolation * segment_delta

    return float(np.linalg.norm(closest_position))


@dataclass
class InterceptorTrack:
    """
    保存单枚拦截弹的运行状态。

    属性：
        index：
            拦截弹编号，从 1 开始，直接用于 info/CSV 字段命名。
        state：
            当前拦截弹状态向量 [x, y, z, V, theta, psi, nx, ny, nz]。
        interceptor：
            完整中末制导拦截弹对象；source_pn 模式下也保留对象以便切换模式。
        ny_actual/nz_actual：
            当前一阶自动驾驶仪之后的纵向/侧向实际过载。
        min_distance：
            当前回合内该拦截弹与红方的连续最小距离。
        passed/intercepted：
            该拦截弹是否已经错过或命中红方。
    """

    index: int
    state: np.ndarray
    interceptor: Interceptor
    ny_actual: float = 1.0
    nz_actual: float = 0.0
    ny_command: float = 1.0
    nz_command: float = 0.0
    phase: str = "unknown"
    min_distance: float = np.inf
    current_distance: float = np.inf
    closest_distance_this_step: float = np.inf
    passed: bool = False
    intercepted: bool = False


class InterceptorFleet:
    """
    蓝方拦截弹编队管理器。

    外部调用流程：
        reset(red_state, initial_states)
        step(red_state, previous_red_position, red_lateral_overload, dt, current_step)

    说明：
        该类不直接计算红方奖励，也不管理 Gym 终止标志；它只返回每枚弹的状态、
        最近距离、阶段和控制诊断，由环境统一组装 info。
    """

    def __init__(
        self,
        guidance_mode: str,
        navigation_config: ProportionalNavigationConfig,
        interceptor_config: InterceptorConfig,
        kill_radius: float,
        passed_distance_margin: float,
        use_range_rate_passed_termination: bool,
        termination_x_margin: float,
    ) -> None:
        """
        初始化拦截弹编队。

        参数：
            guidance_mode：
                蓝方制导模式，支持 source_pn 与 mid_terminal_interceptor。
            navigation_config：
                source_pn 模式的比例导引配置。
            interceptor_config：
                完整中末制导拦截弹配置。
            kill_radius：
                杀伤半径，单位 m。
            passed_distance_margin：
                通过 range-rate 判断错过时的距离回升余量。
            use_range_rate_passed_termination：
                是否优先使用 range-rate 判断错过。
            termination_x_margin：
                兼容 x 坐标错过判据时使用的阈值。
        """
        # guidance_mode：保存当前蓝方制导模式，step 时按模式分支推进。
        self.guidance_mode = str(guidance_mode)

        # navigation_config/interceptor_config：保存两类蓝方制导配置。
        self.navigation_config = navigation_config
        self.interceptor_config = interceptor_config

        # kill_radius/passed_distance_margin：保存命中和错过判据参数。
        self.kill_radius = float(kill_radius)
        self.passed_distance_margin = float(passed_distance_margin)

        # use_range_rate_passed_termination：远距离迎头场景默认使用距离变化率判断是否错过。
        self.use_range_rate_passed_termination = bool(use_range_rate_passed_termination)

        # termination_x_margin：只在关闭 range-rate 判据时作为退化判据。
        self.termination_x_margin = float(termination_x_margin)

        # tracks：每枚拦截弹的独立状态记录。
        self.tracks: List[InterceptorTrack] = []

    @property
    def states(self) -> List[np.ndarray]:
        """
        返回当前所有拦截弹状态的副本列表。

        返回：
            states：
                每枚拦截弹的状态向量副本。
        """
        # states：复制后返回，避免环境外部误改编队内部状态。
        states = [track.state.copy() for track in self.tracks]

        return states

    def reset(self, red_state: np.ndarray, initial_states: List[np.ndarray]) -> None:
        """
        重置编队内所有拦截弹。

        参数：
            red_state：
                红方初始状态，用于初始化每枚弹的初始距离。
            initial_states：
                每枚拦截弹的初始状态列表。

        返回：
            None。
        """
        # tracks：清空上一回合编队。
        self.tracks = []

        for index, initial_state in enumerate(initial_states, start=1):
            # state：复制初始状态，保证后续推进只修改编队内部副本。
            state = np.asarray(initial_state, dtype=np.float64).copy()

            if state.shape[0] != 9:
                raise ValueError(f"第 {index} 枚拦截弹状态维度应为 9，实际为 {state.shape[0]}")

            # interceptor：为该弹创建独立完整拦截弹对象和自动驾驶仪状态。
            interceptor = Interceptor(self.interceptor_config)
            interceptor.reset(state)

            # distance：初始红蓝三维距离。
            distance = float(np.linalg.norm(state[:3] - np.asarray(red_state[:3], dtype=np.float64)))

            # track：该弹的完整跟踪记录。
            track = InterceptorTrack(
                index=index,
                state=state,
                interceptor=interceptor,
                ny_actual=float(state[7]),
                nz_actual=float(state[8]),
                ny_command=float(state[7]),
                nz_command=float(state[8]),
                phase="unknown",
                min_distance=distance,
                current_distance=distance,
                closest_distance_this_step=distance,
            )

            self.tracks.append(track)

    def _step_source_pn(
        self,
        track: InterceptorTrack,
        red_state: np.ndarray,
        red_lateral_overload: float,
        dt: float,
    ) -> Dict[str, float]:
        """
        推进单枚拦截弹的 source_pn 分支。

        参数：
            track：
                单枚拦截弹跟踪记录。
            red_state：
                红方当前状态。
            red_lateral_overload：
                红方一阶自动驾驶仪之后的实际横向过载。
            dt：
                仿真步长。

        返回：
            guidance_info：
                原始比例导引诊断信息。
        """
        # command_pair/actual_pair：source_pn 给出的双通道指令和一阶响应输出。
        command_pair, actual_pair, guidance_info = compute_interceptor_proportional_navigation(
            red_state=red_state,
            interceptor_state=track.state,
            red_lateral_overload=red_lateral_overload,
            previous_interceptor_ny_actual=track.ny_actual,
            previous_interceptor_nz_actual=track.nz_actual,
            dt=dt,
            config=self.navigation_config,
        )

        # track：保存当前指令和实际过载。
        track.ny_command = float(command_pair[0])
        track.nz_command = float(command_pair[1])
        track.ny_actual = float(actual_pair[0])
        track.nz_actual = float(actual_pair[1])
        track.phase = "unknown"

        # state：使用双通道实际过载推进拦截弹质点状态。
        track.state = update_point_mass_state(
            state=track.state,
            nx=0.0,
            ny=track.ny_actual,
            nz=track.nz_actual,
            dt=dt,
        )
        track.state[6] = 0.0
        track.state[7] = track.ny_actual
        track.state[8] = track.nz_actual

        return guidance_info

    def _step_mid_terminal(
        self,
        track: InterceptorTrack,
        red_state: np.ndarray,
        red_lateral_overload: float,
        dt: float,
    ) -> Dict[str, float]:
        """
        推进单枚拦截弹的完整中末制导分支。

        参数：
            track：
                单枚拦截弹跟踪记录。
            red_state：
                红方当前状态。
            red_lateral_overload：
                红方一阶自动驾驶仪之后的实际横向过载。
            dt：
                仿真步长。

        返回：
            guidance_info：
                完整拦截弹对象返回的制导诊断信息。
        """
        # next_state/control/guidance_info：完整拦截弹闭环推进结果。
        next_state, control, guidance_info = track.interceptor.step(
            target_state=red_state,
            dt=dt,
            target_lateral_overload=red_lateral_overload,
        )

        # track：同步该弹状态、指令、实际过载和制导阶段。
        track.state = next_state
        track.ny_command = float(control.ny_command)
        track.nz_command = float(control.nz_command)
        track.ny_actual = float(control.ny_actual)
        track.nz_actual = float(control.nz_actual)
        track.phase = str(control.phase)

        return guidance_info

    def _mark_passed(
        self,
        track: InterceptorTrack,
        red_state: np.ndarray,
        current_relative_info: Dict[str, float],
        current_step: int,
    ) -> None:
        """
        根据当前几何关系标记单枚拦截弹是否已经错过红方。

        参数：
            track：
                单枚拦截弹跟踪记录。
            red_state：
                红方当前状态。
            current_relative_info：
                当前步长结束后的相对几何信息。
            current_step：
                当前环境步数，用于避免初始瞬态误判。

        返回：
            None。
        """
        if track.intercepted:
            return

        if self.use_range_rate_passed_termination:
            # range_rate：大于 0 表示双方距离开始增大，即已经越过最近交会点。
            range_rate = float(current_relative_info.get("range_rate", -1.0))

            # passed：距离已经从历史最小值回升超过余量后才认为该弹错过。
            track.passed = bool(
                (range_rate > 0.0)
                and (track.current_distance > track.min_distance + self.passed_distance_margin)
                and (current_step > 10)
            )
        else:
            # 退化 x 判据：只用于近距或旧 debug 场景。
            track.passed = bool((track.state[0] - red_state[0]) < self.termination_x_margin)

    def _pack_track_info(
        self,
        track: InterceptorTrack,
        guidance_info: Dict[str, Any],
        current_relative_info: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        把单枚拦截弹诊断信息打包为 interceptor_{i}_* 字段。

        参数：
            track：
                单枚拦截弹跟踪记录。
            guidance_info：
                制导律原始诊断信息。
            current_relative_info：
                当前步长结束后的相对几何信息。

        返回：
            info：
                带编号前缀的诊断字段。
        """
        # prefix：该弹在 info/CSV 中的字段前缀。
        prefix = f"interceptor_{track.index}"

        # merged_info：当前几何优先覆盖制导前几何，保证距离和 range-rate 是步长结束值。
        merged_info: Dict[str, Any] = dict(guidance_info)
        merged_info.update(current_relative_info)

        # info：对外暴露的 per-interceptor 诊断字段。
        info: Dict[str, Any] = {
            f"{prefix}_distance": float(track.current_distance),
            f"{prefix}_min_distance": float(track.min_distance),
            f"{prefix}_closest_distance_this_step": float(track.closest_distance_this_step),
            f"{prefix}_ny_command": float(track.ny_command),
            f"{prefix}_nz_command": float(track.nz_command),
            f"{prefix}_ny_actual": float(track.ny_actual),
            f"{prefix}_nz_actual": float(track.nz_actual),
            f"{prefix}_phase": str(track.phase),
            f"{prefix}_speed": float(track.state[3]),
            f"{prefix}_theta": float(track.state[4]),
            f"{prefix}_psi": float(track.state[5]),
            f"{prefix}_passed": bool(track.passed),
            f"{prefix}_intercepted": bool(track.intercepted),
        }

        # scalar_keys：常用几何和制导诊断字段，存在时按编号导出。
        scalar_keys = [
            "closing_speed",
            "range_rate",
            "dx",
            "dy",
            "dz",
            "dxdt",
            "dydt",
            "dzdt",
            "los_angle",
            "los_rate_standard",
            "horizontal_distance",
            "tgo",
            "desired_theta",
            "desired_psi",
            "theta_error",
            "psi_error",
            "lambda_theta_dot",
            "lambda_psi_dot",
            "source_tgo",
            "source_dqy",
            "source_dqz",
            "raw_ny_command",
            "raw_nz_command",
            "interceptor_target_lateral_overload",
            "interceptor_target_compensation",
            "interceptor_target_compensation_gain",
        ]

        for key in scalar_keys:
            if key in merged_info:
                # exported_key：移除内部 interceptor_ 前缀，避免重复编号。
                exported_key = key
                if exported_key.startswith("interceptor_"):
                    exported_key = exported_key.replace("interceptor_", "", 1)
                info[f"{prefix}_{exported_key}"] = float(merged_info[key])

        return info

    def step(
        self,
        red_state: np.ndarray,
        previous_red_position: np.ndarray,
        red_lateral_overload: float,
        dt: float,
        current_step: int,
    ) -> Dict[str, Any]:
        """
        推进编队中所有尚未错过的拦截弹一步。

        参数：
            red_state：
                红方已经更新后的当前状态。
            previous_red_position：
                红方步长开始时的位置，用于连续最近点判据。
            red_lateral_overload：
                红方一阶自动驾驶仪之后的实际横向过载。
            dt：
                仿真步长。
            current_step：
                当前环境步数。

        返回：
            fleet_info：
                编队级和每枚弹的诊断字段。
        """
        if self.guidance_mode not in {"source_pn", "mid_terminal_interceptor"}:
            raise ValueError(f"未知蓝方制导模式：{self.guidance_mode}")

        # fleet_info：最终返回给环境的诊断字典。
        fleet_info: Dict[str, Any] = {}

        # closest_values：记录当前步长每枚弹的连续最近距离，用于全局命中判据。
        closest_values: List[float] = []

        for track in self.tracks:
            # previous_interceptor_position：该弹步长开始时的位置。
            previous_interceptor_position = track.state[:3].copy()

            # previous_relative_position：连续最近点判据使用 blue-red 相对位置。
            previous_relative_position = previous_interceptor_position - previous_red_position

            if not track.passed and not track.intercepted:
                if self.guidance_mode == "source_pn":
                    guidance_info = self._step_source_pn(
                        track=track,
                        red_state=red_state,
                        red_lateral_overload=red_lateral_overload,
                        dt=dt,
                    )
                else:
                    guidance_info = self._step_mid_terminal(
                        track=track,
                        red_state=red_state,
                        red_lateral_overload=red_lateral_overload,
                        dt=dt,
                    )
            else:
                # guidance_info：已错过的拦截弹冻结状态，但仍输出当前几何诊断。
                guidance_info = compute_relative_geometry(red_state=red_state, interceptor_state=track.state)

            # current_relative_position：该弹步长结束时的 blue-red 相对位置。
            current_relative_position = track.state[:3] - red_state[:3]

            # current_distance：离散采样点上的当前距离。
            track.current_distance = float(np.linalg.norm(current_relative_position))

            # closest_distance_this_step：当前步长连续线段最近距离。
            track.closest_distance_this_step = compute_segment_closest_distance(
                start_relative_position=previous_relative_position,
                end_relative_position=current_relative_position,
            )

            # min_distance：该弹回合内连续最小距离。
            track.min_distance = min(
                float(track.min_distance),
                float(track.current_distance),
                float(track.closest_distance_this_step),
            )

            # intercepted：该弹是否进入杀伤半径。
            track.intercepted = bool(track.min_distance <= self.kill_radius)

            # current_relative_info：步长结束后的最新相对几何。
            current_relative_info = compute_relative_geometry(red_state=red_state, interceptor_state=track.state)

            # passed：没有命中时才继续判断是否已经错过。
            self._mark_passed(
                track=track,
                red_state=red_state,
                current_relative_info=current_relative_info,
                current_step=current_step,
            )

            # track_info：导出当前拦截弹编号字段。
            track_info = self._pack_track_info(
                track=track,
                guidance_info=guidance_info,
                current_relative_info=current_relative_info,
            )
            fleet_info.update(track_info)
            closest_values.append(float(track.closest_distance_this_step))

        # current_distances：每枚弹当前离散距离。
        current_distances = [float(track.current_distance) for track in self.tracks]

        # min_distances：每枚弹历史连续最小距离。
        min_distances = [float(track.min_distance) for track in self.tracks]

        # threat_index：当前距离最近的拦截弹编号。
        threat_index = int(np.argmin(current_distances)) + 1 if current_distances else 0

        # any_intercepted/all_passed：编队级终止判据。
        any_intercepted = any(track.intercepted for track in self.tracks)
        all_passed = all(track.passed for track in self.tracks)

        # fleet_info：补充全局聚合字段。
        fleet_info.update(
            {
                "distance": float(min(current_distances)) if current_distances else np.inf,
                "min_distance": float(min(min_distances)) if min_distances else np.inf,
                "closest_distance_this_step": float(min(closest_values)) if closest_values else np.inf,
                "threat_interceptor_id": threat_index,
                "intercepted": bool(any_intercepted),
                "all_interceptors_passed": bool(all_passed),
                "interceptor_count": int(len(self.tracks)),
            }
        )

        if threat_index > 0:
            # threat_prefix：把最危险拦截弹的几何量同步到全局便捷字段。
            threat_prefix = f"interceptor_{threat_index}"
            for key in ["range_rate", "closing_speed", "los_angle", "los_rate_standard"]:
                full_key = f"{threat_prefix}_{key}"
                if full_key in fleet_info:
                    fleet_info[key] = fleet_info[full_key]

        return fleet_info
