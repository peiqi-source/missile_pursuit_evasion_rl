import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd
import os

from numpy import ndarray
# 设置随机种子
np.random.seed(0)  # 你可以选择任何整数值作为种子


class PursueEscapeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nzc_h_max=3, nzc_i_max=6, dt=0.0001, t=20, tau_h=0.5, tau_i=0.5, N=4):
        super(PursueEscapeEnv, self).__init__()

        self.nzc_h_max = nzc_h_max
        self.nzc_i_max = nzc_i_max
        self.dt = dt
        self.t = t
        self.tau_h = tau_h
        self.tau_i = tau_i
        self.N = N
        self.g = 9.81  # 重力加速度

        # 观测
        self.observation = np.zeros((12, ), dtype=float)
        # 红方状态
        self.red_state = np.zeros((9, ), dtype=float)
        # 蓝方状态
        self.blue_state = np.zeros((9, ), dtype=float)

        # 红方控制量
        self.action = spaces.Box(low=-nzc_h_max, high=nzc_h_max, shape=(1,), dtype=np.float32)
        # 经过一阶惯性环节后的控制量
        self.inertial_action = np.zeros((1,), dtype=float)
        # 上一次的控制量
        self.last_action = np.zeros((1,), dtype=float)

        # 蓝方制导控制量
        self.nzc_i = np.zeros((1,), dtype=float)
        # 经过一阶惯性环节后的蓝方制导控制量
        self.inertial_nzc_i = np.zeros((1,), dtype=float)
        # 上一次的蓝方制导控制量
        self.last_nzc_i = np.zeros((1,), dtype=float)

        # 存储
        # 红蓝双方的轨迹
        self.red_trajectory = []
        self.blue_trajectory = []

        # 红方控制量
        self.control_trace = []
        self.inertial_control_trace = []
        # 蓝方控制量
        self.nzc_i_trace = []
        self.inertial_nzc_i_trace = []

        # 相对距离变化
        self.distance_trace = []

        # 存储过程中的全部数据
        self.save_red_states = []
        self.save_blue_states = []
        self.save_actions = []
        self.save_ncz_i = []
        self.save_distance = []

        # 初始化
        # 红方
        x_h0 = 0
        y_h0 = 27 * 1000
        z_h0 = 0

        psiv_h0 = 0 / 180 * np.pi
        theta_h0 = 0
        speed_h0 = 5.5 * 340

        nx_h0 = 0
        ny_h0 = 1
        nz_h0 = 0

        self.red_state = np.array([x_h0, y_h0, z_h0, speed_h0, theta_h0, psiv_h0, nx_h0, ny_h0, nz_h0])

        # 蓝方
        x_i0 = 30 * 1000
        y_i0 = 27 * 1000
        z_i0 = 0

        psiv_i0 = 180 / 180 * np.pi
        theta_i0 = 0
        speed_i0 = 3.5 * 340

        nx_i0 = 0
        ny_i0 = 1
        nz_i0 = 0

        self.blue_state = np.array([x_i0, y_i0, z_i0, speed_i0, theta_i0, psiv_i0, nx_i0, ny_i0, nz_i0])

    def step(self, action):

        # 更新状态
        self.update_state(action)

        # 观测
        observation = self.env_observation()

        # 计算奖励
        reward = self.calculate_reward(action)

        # 检查是否结束
        done = self.is_done()

        truncated = False

        # 可选的信息字典
        info = {}

        # 记录轨迹
        self.red_trajectory.append((self.red_state[0], self.red_state[2]))
        self.blue_trajectory.append((self.blue_state[0], self.blue_state[2]))
        # 红方控制量
        self.control_trace.append(action)
        self.inertial_control_trace.append(self.inertial_action[0])
        # 蓝方控制量
        self.nzc_i_trace.append(self.nzc_i)
        self.inertial_nzc_i_trace.append(self.inertial_nzc_i[0])
        # 相对距离
        distance = np.sqrt((self.red_state[0] - self.blue_state[0])**2 +
                           (self.red_state[1] - self.blue_state[1])**2 +
                           (self.red_state[2] - self.blue_state[2])**2)
        self.distance_trace.append(distance)

        # 存储过程中的全部数据
        self.save_red_states.append(self.red_state.copy())
        self.save_blue_states.append(self.blue_state.copy())
        self.save_actions.append(self.action.copy())
        self.save_ncz_i.append(self.nzc_i.copy())
        self.save_distance.append(distance.copy())

        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        # 红方
        x_h0 = 0
        y_h0 = 27*1000
        z_h0 = 0

        psiv_h0 = 0/180*np.pi
        theta_h0 = 0
        speed_h0 = 5.5*340

        nx_h0 = 0
        ny_h0 = 1
        nz_h0 = 0

        self.red_state = np.array([x_h0, y_h0, z_h0, speed_h0, theta_h0, psiv_h0, nx_h0, ny_h0, nz_h0])

        # 蓝方
        x_i0 = (28+4*np.random.rand())*1000
        y_i0 = 27 * 1000
        z_i0 = 0

        psiv_i0 = (175+5*np.random.rand()) / 180 * np.pi
        theta_i0 = 0
        speed_i0 = 3.5 * 340

        nx_i0 = 0
        ny_i0 = 1
        nz_i0 = 0

        self.blue_state = np.array([x_i0, y_i0, z_i0, speed_i0, theta_i0, psiv_i0, nx_i0, ny_i0, nz_i0])

        # 可选的信息字典
        info = {}

        # 清空轨迹
        self.red_trajectory = []
        self.blue_trajectory = []

        # 过载
        self.control_trace = []
        self.inertial_control_trace = []
        self.nzc_i_trace = []
        self.inertial_nzc_i_trace = []

        self.distance_trace = []

        self.save_red_states = []
        self.save_blue_states = []
        self.save_actions = []
        self.save_ncz_i = []
        self.save_distance = []

        self.observation = np.concatenate((self.red_state[:6], self.blue_state[:6]))

        return self.observation, info

    def update_state(self, action):
        # 更新红方和蓝方的状态

        # 红方
        # nx_h(6) ny_h(7) nz_h(8)
        self.red_state[6], self.red_state[7], self.red_state[8] = self.calculate_red_control(action)  # 根据策略计算红方控制输入
        # speed_h(3)
        self.red_state[3] = self.red_state[3] + self.g * (self.red_state[6] - np.sin(self.red_state[4])) * self.dt
        # theta_h(4)
        self.red_state[4] = self.red_state[4] + self.g / self.red_state[3] * (self.red_state[7] - np.cos(self.red_state[4])) * self.dt
        # psiv_h(5)
        self.red_state[5] = self.red_state[5] - self.g / (self.red_state[3] * np.cos(self.red_state[4])) * self.red_state[8] * self.dt

        # 红方运动学方程
        # x_h（0）
        self.red_state[0] = self.red_state[0] + self.red_state[3] * np.cos(self.red_state[4]) * np.cos(self.red_state[5]) * self.dt
        # y_h（1）
        self.red_state[1] = self.red_state[1] + self.red_state[3] * np.sin(self.red_state[4]) * self.dt
        # z_h（2）
        self.red_state[2] = self.red_state[2] - self.red_state[3] * np.cos(self.red_state[4]) * np.sin(self.red_state[5]) * self.dt

        # 蓝方
        # nx_i(6) ny_i(7) nz_i(8)
        self.blue_state[6], self.blue_state[7], self.blue_state[8], _ = self.calculate_blue_control()  # 根据策略计算蓝方控制输入
        # speed_i(3)
        self.blue_state[3] = self.blue_state[3] + self.g * (self.blue_state[6] - np.sin(self.blue_state[4])) * self.dt
        # theta_i(4)
        self.blue_state[4] = self.blue_state[4] + self.g / self.blue_state[3] * (self.blue_state[7] - np.cos(self.blue_state[4])) * self.dt
        # psiv_i(5)
        self.blue_state[5] = self.blue_state[5] - self.g / (self.blue_state[3] * np.cos(self.blue_state[4])) * self.blue_state[8] * self.dt

        # 红方运动学方程
        # x_i（0）
        self.blue_state[0] = self.blue_state[0] + self.blue_state[3] * np.cos(self.blue_state[4]) * np.cos(self.blue_state[5]) * self.dt
        # y_i（1）
        self.blue_state[1] = self.blue_state[1] + self.blue_state[3] * np.sin(self.blue_state[4]) * self.dt
        # z_i（2）
        self.blue_state[2] = self.blue_state[2] - self.blue_state[3] * np.cos(self.blue_state[4]) * np.sin(self.blue_state[5]) * self.dt

    def env_observation(self):
        self.observation = np.concatenate((self.red_state[:6], self.blue_state[:6]))
        return self.observation

    def calculate_reward(self, action):
        # 根据红蓝双方的距离计算奖励
        r1 = action ** 2 * (-0.015)

        if len(self.distance_trace) > 0 and np.min(self.distance_trace) >= 7 and (self.blue_state[0]-self.red_state[0]) <= 0.05:
            r2 = 160 + 20 * np.min(self.distance_trace)
        elif len(self.distance_trace) > 0 and np.min(self.distance_trace) < 7 and (self.blue_state[0]-self.red_state[0]) <= 0.05:
            r2 = 20 * np.min(self.distance_trace)
        else:
            r2 = 0
        _, _, _, dqz = self.calculate_blue_control()
        r3 = 0.15*0.7*np.log(np.abs(dqz))

        if (np.abs(self.red_state[2] - self.blue_state[2]) - 7) > 0:
            r4 = (np.abs(self.red_state[2] - self.blue_state[2]) - 7) * 0.005
        else:
            r4 = -0.15

        reward = r1 + r2 + r3 + r4

        return reward

    def is_done(self):
        # 如果红蓝双方距离小于某个阈值，则认为红方被捕捉
        xx = (self.blue_state[0] - self.red_state[0])
        if xx < -50:
            return 1
        else:
            return 0

    def render(self, mode='human', close=False):
        # 渲染环境状态
        # print(f"Red: {self.red_state}, Blue: {self.blue_state}")
        print(f"min_distance: {np.min(self.distance_trace) }")

    def calculate_red_control(self, action):
        self.inertial_action = self.last_action + self.dt * (action - self.last_action) / self.tau_h
        self.action = self.inertial_action.copy()  # 将经过惯性环节的处理后的控制量赋值给self.action
        self.last_action = self.inertial_action.copy()  # 更新上一次的控制量
        return 0, 1, self.inertial_action

    def calculate_blue_control(self):
        """
           计算蓝方的控制输入，使用比例导引法（Proportional Navigation）
        """
        # 红方和蓝方的位置差
        dx = self.red_state[0] - self.blue_state[0]  # 红方x - 蓝方x
        dy = self.red_state[1] - self.blue_state[1]  # 红方y - 蓝方y
        dz = self.red_state[2] - self.blue_state[2]  # 红方z - 蓝方z

        # 计算相对距离
        dis_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # 红方和蓝方的速度差（即相对速度）
        dxdt = (self.red_state[3] * np.cos(self.red_state[4]) * np.cos(self.red_state[5])) - \
               (self.blue_state[3] * np.cos(self.blue_state[4]) * np.cos(self.blue_state[5]))
        dydt = (self.red_state[3] * np.sin(self.red_state[4])) - (self.blue_state[3] * np.sin(self.blue_state[4]))
        dzdt = (-self.red_state[3] * np.cos(self.red_state[4]) * np.sin(self.red_state[5])) - \
               (-self.blue_state[3] * np.cos(self.blue_state[4]) * np.sin(self.blue_state[5]))

        # 相对速度 v_r
        v_r = (dx * dxdt + dy * dydt + dz * dzdt) / dis_r

        # 计算时间到达（TGO，Time to Go）
        tgo = -dis_r / v_r if v_r != 0 else 0  # 避免除零错误

        # 计算视线角速度
        dqy = -dy / (v_r * tgo ** 2) - dydt / (v_r * tgo)  # 高低视线角速度
        dqz = dz / (v_r * tgo ** 2) + dzdt / (v_r * tgo)  # 方位视线角速度

        # 计算纵向控制输入 nzc_i
        self.nzc_i = -self.N * v_r * dqz / self.g - 0.5*self.inertial_action

        # 经过一阶惯性环节
        # 蓝方制导控制量，限制在-nzc_i_max和nzc_i_max之间
        self.nzc_i = np.clip(self.nzc_i, -self.nzc_i_max, self.nzc_i_max)
        # 蓝方制导控制量经过一阶惯性环节
        self.inertial_nzc_i = self.last_nzc_i + self.dt * (self.nzc_i - self.last_nzc_i) / self.tau_i
        self.last_nzc_i = self.inertial_nzc_i.copy()  # 更新上一次的蓝方制导控制量
        return 0, 1, self.inertial_nzc_i, dqz

# # 确保save文件夹存在
# save_dir = 'save'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# # 使用环境
# env = PursueEscapeEnv()
# done = env.is_done()
# i = 0
# while not done:
#     i += 1
#     if len(env.distance_trace) > 0 and np.min(env.distance_trace) < 7000:
#         action = 3
#     else:
#         action = -3
#     observation, reward, done, info = env.step(action)
#     env.render()
#
# # 计算时间序列
# time_steps = np.arange(len(env.control_trace)) * env.dt
#
# # 准备画布，设置足够大的尺寸以容纳所有子图
# plt.figure(figsize=(14, 12))
#
# # 绘制控制输入和惯性控制输出图
# plt.subplot(4, 1, 1)
# plt.plot(time_steps, env.control_trace, label='Control Input', color='tab:blue', linestyle='-')
# plt.plot(time_steps, env.inertial_control_trace, label='Inertial Control Output (Red)', color='tab:red', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Control Value')
# plt.title('Control Inputs')
# plt.legend(loc='upper left')
# plt.grid(True)
#
# # 绘制蓝方制导控制量和惯性控制输出图
# plt.subplot(4, 1, 2)
# plt.plot(time_steps, env.nzc_i_trace, label='Control Input', color='tab:green', linestyle='-')
# plt.plot(time_steps, env.inertial_nzc_i_trace, label='Inertial Control Output (Blue)', color='tab:purple', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Control Value')
# plt.title('Guidance Control Inputs')
# plt.legend(loc='upper left')
# plt.grid(True)
#
# # 绘制红蓝双方在X-Z平面上的轨迹图
# plt.subplot(4, 1, 3)
# plt.plot([x for x, _ in env.red_trajectory], [z for _, z in env.red_trajectory], 'r-', label='Red Trajectory', linewidth=2)
# plt.plot([x for x, _ in env.blue_trajectory], [z for _, z in env.blue_trajectory], 'b-', label='Blue Trajectory', linewidth=2)
# plt.xlabel('X Coordinate')
# plt.ylabel('Z Coordinate')
# plt.title('Red and Blue Trajectories on X-Z Plane')
# plt.legend(loc='upper left')
# plt.grid(True)
#
# # 绘制相对距离变化图，并标出最小距离
# plt.subplot(4, 1, 4)
# plt.plot(time_steps, env.distance_trace, label='Relative Distance', color='tab:orange', linestyle='-')
# plt.xlabel('Time Step')
# plt.ylabel('Distance (m)')
# plt.title('Relative Distance Over Time')
# plt.legend(loc='upper left')
# plt.grid(True)
#
# min_distance_idx = np.argmin(env.distance_trace)
# min_distance = env.distance_trace[min_distance_idx]
# plt.axvline(x=min_distance_idx*env.dt, color='black', linestyle='--', label=f'Min Distance at Step {min_distance_idx}')
# plt.text(min_distance_idx*env.dt/2, min_distance, f'Min Distance: {min_distance:.2f}m', color='black', fontsize=14, verticalalignment='bottom', horizontalalignment='center')
#
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
#
# # 保存图片
# plt.savefig(os.path.join(save_dir, 'all_plots.png'), bbox_inches='tight')
#
# # 显示图表
# plt.show()
# plt.close()
#
# # 保存过程中的全部数据到Excel
# red_states_df = pd.DataFrame(env.save_red_states, columns=['x_h', 'y_h', 'z_h', 'speed_h', 'theta_h', 'psiv_h',
#                                                            'nx_h', 'ny_h', 'nz_h'])
# blue_states_df = pd.DataFrame(env.save_blue_states, columns=['x_i', 'y_i', 'z_i', 'speed_i', 'theta_i', 'psiv_i',
#                                                              'nx_i', 'ny_i', 'nz_i'])
# actions_df = pd.DataFrame(env.save_actions, columns=['action'])
# nzc_i_df = pd.DataFrame(env.save_ncz_i, columns=['nzc_i'])
# distance_df = pd.DataFrame(env.save_distance, columns=['distance'])
#
# red_states_df.to_excel(os.path.join(save_dir, 'save_red_states.xlsx'), index=False)
# blue_states_df.to_excel(os.path.join(save_dir, 'save_blue_states.xlsx'), index=False)
# actions_df.to_excel(os.path.join(save_dir, 'save_actions.xlsx'), index=False)
# nzc_i_df.to_excel(os.path.join(save_dir, 'save_nzc_i.xlsx'), index=False)
# distance_df.to_excel(os.path.join(save_dir, 'save_distance.xlsx'), index=False)
