#准备测试brax,物理可微模型
'''

基于brax，生成一个独立的二个点的模型
1.点是一维的，只具有位置和速度属性
2.点就是一个方块，没有质量等属性。彼此不能重叠，一旦重叠就发生碰撞报错
3.点只能在一维线上运动，不能离开这条线。只能向一个方向运行或者停止
4.如果点的运动方向前面没有点，就按照idm跟车模型运动，预期速度是最大速度
5.如果点的运动方向前面有点，就按照idm跟车模型运动
6.设置红灯位置，点一旦接近预定的红灯位置位置，就停止运动，直到停止到预定位置。
7.后面的点不能超过前面的点
8.参考pyGameInterface3.py的写法，完成brax模型的编写。注意brax的环境编写方式和pyGameInterface3.py不一样
9.完成后，编写一个简单的测试脚本，测试二个点的运动情况，观察它们是否按照预期运动
10.测试脚本中，设置点的初始位置和预定位置，观察它们的运动轨迹和最终位置
11.把过程进行可视化，方便观察点的运动情况，同时保存为gif文件。设定开关控制是否保存gif
12.每个点的idm参数可以自行设定，测试脚本中可以设置不同的参数进行测试
13.模型中print和可视化参数设定开关控制是否保存，方便调试和观察
14.注意留下接口，方便后续扩展更多点的模型
15.注意留下接口，用于神经网络或者全局搜索方法（类似模拟退火）的调用，用于训练和获得最优idm参数。

检查达成目标逻辑，应该是所有车辆都通过路口，并且没有发生碰撞。
检查idm跟车模型逻辑，是否正确
代码增加注释以及简洁一点
设置目标位置为统一目标点1000，
'''
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import numpy as np

# ==========================================
# 1. 定义数据结构 (JAX PyTree)
# ==========================================

@struct.dataclass
class IDMParams:
    """IDM模型参数 (支持JAX变换)"""
    v0: jnp.ndarray      # 期望速度
    T: jnp.ndarray       # 安全车头时距
    s0: jnp.ndarray      # 静止安全距离
    a: jnp.ndarray       # 最大加速度
    b: jnp.ndarray       # 舒适减速度
    delta: jnp.ndarray   # 加速度指数
    length: jnp.ndarray  # 车长（已加入，建议设为5.0）
    rtime: jnp.ndarray   # 反应时间 (用于平滑加速度)

@struct.dataclass
class EnvState:
    """环境状态"""
    position: jnp.ndarray      # 车辆位置 (N,)
    velocity: jnp.ndarray      # 车辆速度 (N,)
    acceleration: jnp.ndarray  # 当前加速度 (N,)
    target_pos: jnp.ndarray    # 目标停止位置 (N,)
    params: IDMParams          # 车辆参数
    step_count: int            # 步数
    crashed: bool              # 是否发生碰撞

# ==========================================
# 2. 核心物理逻辑 (JAX Functional Style)
# ==========================================

def compute_idm_acc(
    v: jnp.ndarray,
    v_front: jnp.ndarray,
    dist_gap: jnp.ndarray,
    params: IDMParams
) -> jnp.ndarray:
    """
    计算IDM加速度 (支持 vmap)
    dist_gap 已经包含了车长的影响（见step函数），无需再减车长
    """
    # 自由流项
    free_acc = params.a * (1.0 - (v / params.v0) ** params.delta)
    
    # 交互项
    # 期望间距 s*
    # s* = s0 + v*T + (v * dv) / (2 * sqrt(a*b))
    delta_v = v - v_front
    s_star = params.s0 + jnp.maximum(0.0, v * params.T + (v * delta_v) / (2.0 * jnp.sqrt(params.a * params.b)))
    
    # 避免除以零
    safe_gap = jnp.maximum(dist_gap, 0.1)
    
    interaction_acc = -params.a * (s_star / safe_gap) ** 2
    
    # 如果前方没有车 (dist_gap 非常大)，interaction_acc 趋近于 0
    # 我们通过 mask 在外部控制，或者在这里假设 dist_gap 很大
    
    return free_acc + interaction_acc

def compute_stopping_acc(
    v: jnp.ndarray,
    dist_to_target: jnp.ndarray,
    params: IDMParams
) -> jnp.ndarray:
    """
    计算为了在目标位置停止所需的加速度
    """
    # 距离目标的净距离
    net_dist = dist_to_target - params.s0
    
    # 如果已经到达或超过停止线 (net_dist <= 0)，施加最大制动
    # 使用 sigmoid 或类似平滑函数保持可微性，这里简化处理
    
    # 物理公式: v^2 = 2 * a * d  => a = -v^2 / (2d)
    req_acc = -(v ** 2) / (2.0 * jnp.maximum(net_dist, 0.1))
    
    # 限制最大减速度，防止数值爆炸，但允许紧急制动
    return jnp.maximum(req_acc, -9.0)

# 距离目标位置越近的点为头车，为第一辆车
# 可用于仿真前对车辆排序，确保IDM跟车关系正确

def sort_by_target_distance(position: jnp.ndarray, target_pos: jnp.ndarray):
    """
    按照距离目标点的距离升序排序，返回排序后的索引。
    距离目标最近的为头车（索引0），依次类推。
    """
    dist = target_pos - position
    sorted_idx = jnp.argsort(dist)
    return sorted_idx

# ==========================================
# 3. Brax 风格环境定义
# ==========================================

class BraxIDMEnv:
    def __init__(self, num_vehicles: int = 2, dt: float = 0.1, red_light_pos: float = 100.0):
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.red_light_pos = red_light_pos  # 红灯位置

    def reset(self, rng: jnp.ndarray, init_pos: jnp.ndarray, params: IDMParams) -> EnvState:
        """初始化环境，所有车辆目标点为1000"""
        target_pos = jnp.ones(self.num_vehicles) * 1000.0
        return EnvState(
            position=init_pos,
            velocity=jnp.zeros(self.num_vehicles),
            acceleration=jnp.zeros(self.num_vehicles),
            target_pos=target_pos,
            params=params,
            step_count=0,
            crashed=False
        )

    def step(self, state: EnvState, action: Optional[jnp.ndarray] = None) -> EnvState:
        """
        单步仿真，返回新状态。
        距离目标最近的为头车，头车前方无车，跟车车辆根据IDM模型跟随头车。
        红灯逻辑：若车辆接近红灯位置，则目标点临时设为红灯位置。
        """
        N = self.num_vehicles
        p = state.params

        # 按距离目标点排序，头车排在第0位
        idx = sort_by_target_distance(state.position, state.target_pos)
        pos = state.position[idx]
        vel = state.velocity[idx]
        acc = state.acceleration[idx]
        tgt = state.target_pos[idx]
        p_sorted = IDMParams(
            v0=p.v0[idx], T=p.T[idx], s0=p.s0[idx], a=p.a[idx], b=p.b[idx],
            delta=p.delta[idx], length=p.length[idx], rtime=p.rtime[idx]
        )

        # 红灯逻辑：如果车辆距离红灯小于一定阈值，则目标点临时设为红灯位置
        # 例如阈值为10m
        red_light_arr = jnp.ones_like(pos) * self.red_light_pos
        near_red = (red_light_arr - pos) < 10.0
        # 只对未通过红灯的车辆生效
        not_passed_red = pos < self.red_light_pos
        # 需要停车的车辆
        stop_mask = near_red & not_passed_red
        # 临时目标点
        tgt = jnp.where(stop_mask, self.red_light_pos, tgt)

        # 头车前方无车，gap设为极大，前车速度设为0
        pos_front = jnp.concatenate([pos[1:], jnp.array([1e9])])
        vel_front = jnp.concatenate([vel[1:], jnp.array([0.0])])
        # 跟车净距 = 前车头位置 - 本车头位置 - 本车车长
        gaps = pos_front - pos - p_sorted.length

        # IDM加速度
        acc_idm = compute_idm_acc(vel, vel_front, gaps, p_sorted)

        # 到目标点的距离
        dist_to_target = tgt - pos - p_sorted.length
        acc_stop = compute_stopping_acc(vel, dist_to_target, p_sorted)

        # 取更强制动（停车优先）
        target_mask = tgt < 1e8
        final_acc = jnp.where(target_mask, jnp.minimum(acc_idm, acc_stop), acc_idm)

        # 平滑加速度
        alpha = 1.0 - jnp.exp(-p_sorted.rtime / self.dt)
        smoothed_acc = acc + alpha * (final_acc - acc)
        smoothed_acc = jnp.clip(smoothed_acc, -9.0, p_sorted.a)

        # 积分更新
        new_vel = jnp.maximum(vel + smoothed_acc * self.dt, 0.0)
        new_pos = pos + new_vel * self.dt

        # 碰撞检测：后车不能超过前车
        pos_diff = new_pos[1:] - new_pos[:-1]
        is_crashed = jnp.any(pos_diff <= 0.0)

        # 恢复原始顺序
        inv_idx = jnp.argsort(idx)
        new_state = EnvState(
            position=new_pos[inv_idx],
            velocity=new_vel[inv_idx],
            acceleration=smoothed_acc[inv_idx],
            target_pos=tgt[inv_idx],
            params=IDMParams(
                v0=p_sorted.v0[inv_idx], T=p_sorted.T[inv_idx], s0=p_sorted.s0[inv_idx],
                a=p_sorted.a[inv_idx], b=p_sorted.b[inv_idx], delta=p_sorted.delta[inv_idx],
                length=p_sorted.length[inv_idx], rtime=p_sorted.rtime[inv_idx]
            ),
            step_count=state.step_count + 1,
            crashed=bool(is_crashed)
        )
        return new_state

    def rollout(self, state: EnvState, max_steps: int = 200):
        """
        仿真直到所有车辆通过目标点且无碰撞，或达到最大步数。
        达成目标：所有车辆都已通过目标点（位置大于目标点），且未发生碰撞。
        """
        traj = []
        for _ in range(max_steps):
            traj.append(state)
            if state.crashed:
                break
            # 达成目标：所有车都已通过目标点
            arrived = jnp.all(state.position - state.target_pos > 0.0)
            if arrived:
                break
            state = self.step(state)
        return traj

# ==========================================
# 4. 测试脚本与可视化
# ==========================================

def plot_traj(traj, save_gif=False, gif_path="idm_traj.gif"):
    """可视化轨迹并可选保存为gif（先保存为jpg，再合成gif）"""
    output_dir = "brax_sim_frames"
    frames = []
    if save_gif:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for idx, state in enumerate(traj):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.set_xlim(0, float(jnp.max(state.target_pos)) + 10)
        ax.set_ylim(-1, 1)
        for i, (x, v, t) in enumerate(zip(state.position, state.velocity, state.target_pos)):
            ax.plot([x], [0], 'o', label=f"car{i} pos={float(x):.2f} v={float(v):.2f}")
            ax.axvline(t, color='r', linestyle='--', alpha=0.5)
            ax.text(x, 0.2, f"v={float(v):.2f}", ha='center')
        ax.legend()
        ax.set_title(f"step={state.step_count} crashed={state.crashed}")
        
        #plt.tight_layout()
        if save_gif:
            jpg_path = os.path.join(output_dir, f"frame_{state.step_count:05d}.jpg")
            fig.savefig(jpg_path, dpi=300)
        plt.close(fig)
    if save_gif:
        # 合成gif
        frame_files = sorted(
            [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')],
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )
        if frame_files:
            images = [imageio.imread(f) for f in frame_files]
            imageio.mimsave(gif_path, images, duration=0.1)
            print(f"已保存gif到 {gif_path}")
        else:
            print(f"未找到jpg帧，无法生成gif。")

if __name__ == "__main__":
    # 参数可调
    params = IDMParams(
        v0=jnp.array([12.0, 10.0]),  # 两辆车的期望速度
        T=jnp.array([1.2, 1.5]),
        s0=jnp.array([2.0, 2.0]),
        a=jnp.array([1.5, 1.2]),
        b=jnp.array([2.0, 2.0]),
        delta=jnp.array([4.0, 4.0]),
        length=jnp.array([5.0, 5.0]),
        rtime=jnp.array([0.2, 0.2])
    )
    env = BraxIDMEnv(num_vehicles=2, dt=0.1, red_light_pos=100.0)
    # 初始位置和目标
    init_pos = jnp.array([30.0, 20.0])  # 车1在后，车0在前
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, init_pos, params)
    traj = env.rollout(state, max_steps=300)
    plot_traj(traj, save_gif=True, gif_path="idm_2cars_redlight.gif")

'''

# 例：可微损失函数（用于神经网络或优化器）
def loss_fn(idm_params_flat):
    # idm_params_flat: shape=(2*8,)  # 2辆车，每辆8个参数
    params = IDMParams(
        v0=idm_params_flat[0:2],
        T=idm_params_flat[2:4],
        s0=idm_params_flat[4:6],
        a=idm_params_flat[6:8],
        b=jnp.array([2.0, 2.0]),
        delta=jnp.array([4.0, 4.0]),
        length=jnp.array([5.0, 5.0]),
        rtime=jnp.array([0.2, 0.2])
    )
    env = BraxIDMEnv(num_vehicles=2, dt=0.1)
    init_pos = jnp.array([0.0, 20.0])
    target_pos = jnp.array([50.0, 70.0])
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, init_pos, target_pos, params)
    traj = env.rollout(state, max_steps=200)
    # 以最后一步距离目标的误差为损失
    final_state = traj[-1]
    loss = jnp.sum(jnp.abs(final_state.position - final_state.target_pos))
    return loss
'''
# 可直接用jax.grad/lax.scan等做端到端优化