import matplotlib.pyplot as plt
import os
import numpy as np


def save_plots(episode,save_dir, env, step):

    timestep = np.linspace(0, step, len(env.control_trace))
    min_distance_idx = np.argmin(env.distance_trace)
    # 确保save文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 准备画布，设置足够大的尺寸以容纳所有子图
    plt.figure(figsize=(14, 12))

    # 绘制控制输入和惯性控制输出图
    plt.subplot(4, 1, 1)
    plt.plot(timestep[:min_distance_idx], env.control_trace[:min_distance_idx], label='Control Input', color='tab:blue', linestyle='-')
    plt.plot(timestep[:min_distance_idx], env.inertial_control_trace[:min_distance_idx], label='Inertial Control Output (Red)', color='tab:red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Control Value')
    plt.title('Control Inputs')
    plt.legend(loc='upper left')
    plt.grid(True)

    # 绘制蓝方制导控制量和惯性控制输出图
    plt.subplot(4, 1, 2)
    plt.plot(timestep[:min_distance_idx], env.nzc_i_trace[:min_distance_idx], label='Control Input', color='tab:green', linestyle='-')
    plt.plot(timestep[:min_distance_idx], env.inertial_nzc_i_trace[:min_distance_idx], label='Inertial Control Output (Blue)', color='tab:purple', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Control Value')
    plt.title('Guidance Control Inputs')
    plt.legend(loc='upper left')
    plt.grid(True)

    # 绘制红蓝双方在X-Z平面上的轨迹图
    plt.subplot(4, 1, 3)
    plt.plot([x for x, _ in env.red_trajectory[:min_distance_idx]], [z for _, z in env.red_trajectory[:min_distance_idx]], 'r-', label='Red Trajectory', linewidth=2)
    plt.plot([x for x, _ in env.blue_trajectory[:min_distance_idx]], [z for _, z in env.blue_trajectory[:min_distance_idx]], 'b-', label='Blue Trajectory', linewidth=2)
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Red and Blue Trajectories on X-Z Plane')
    plt.legend(loc='upper left')
    plt.grid(True)

    # 绘制相对距离变化图，并标出最小距离
    plt.subplot(4, 1, 4)
    plt.plot(timestep[:min_distance_idx], env.distance_trace[:min_distance_idx], label='Relative Distance', color='tab:orange', linestyle='-')
    plt.xlabel('Time Step')
    plt.ylabel('Distance (m)')
    plt.title('Relative Distance Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)

    min_distance_idx = np.argmin(env.distance_trace)
    min_distance = env.distance_trace[min_distance_idx]
    plt.axvline(x=min_distance_idx*env.dt, color='black', linestyle='--', label=f'Min Distance at Step {min_distance_idx}')
    plt.text(min_distance_idx*env.dt/2, min_distance, f'Min Distance: {min_distance:.2f}m', color='black', fontsize=14, verticalalignment='bottom', horizontalalignment='center')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)

    episode_image_name = f"Ep_{episode}_{'Yes' if min_distance > 7 else 'No'}.png"
    plt.savefig(os.path.join(save_dir, episode_image_name), bbox_inches='tight')
    plt.close()