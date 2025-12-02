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
import pandas as pd

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
    front_car_id: jnp.ndarray  # shape=(N,)
# ==========================================
# 2. 核心物理逻辑 (JAX Functional Style)
# ==========================================
def compute_idm_acc(
    v: jnp.ndarray,
    v_front: jnp.ndarray,
    dist_gap: jnp.ndarray,
    params: IDMParams,
    front_car_id: jnp.ndarray ,
    log_list: list = None,
    step_count: int = None,
    car_ids: list = None
) -> jnp.ndarray:
    """
    计算IDM加速度 (支持 vmap)
    增加front_car_id: 若有前车，free_acc为原始值；否则free_acc为固定值（如0）。
    idm params.v0*1.3,这里有问题    
    """
    # 判断是否有前车（front_car_id >= 0）
    has_front =front_car_id >= 0
    # 原始free_acc
    free_acc_orig0 = params.a * (1.0 - (v / params.v0) ** params.delta)
    free_acc_orig1 = params.a * (1.0 - (v / (params.v0*1.3)) ** params.delta)

    # 没有前车时free_acc
    free_acc = jnp.where(has_front, free_acc_orig1, free_acc_orig0)

    # 交互项
    delta_v = v - v_front
    s_star = params.s0 + v * params.T + (v * delta_v) / (2.0 * jnp.sqrt(params.a * params.b))
    interaction_acc = -params.a * (s_star / dist_gap) ** 2

    idmacc = free_acc + interaction_acc

    # 日志记录
    if log_list is not None:
        v_np = np.asarray(v)
        v_front_np = np.asarray(v_front)
        dist_gap_np = np.asarray(dist_gap)
        free_acc_np = np.asarray(free_acc)
        interaction_acc_np = np.asarray(interaction_acc)
        delta_v_np = np.asarray(delta_v)
        step = int(step_count) if step_count is not None else -1
        ids = car_ids if car_ids is not None else list(range(len(v_np)))
        front_ids = np.asarray(front_car_id) if front_car_id is not None else [None] * len(v_np)
        for i in range(len(v_np)):
            log_list.append({
                'step': step,
                'car_id': int(ids[i]),
                'front_car_id': int(front_ids[i]) if front_car_id is not None else None,
                'v': float(v_np[i]),
                'v_front': float(v_front_np[i]),
                'dist_gap': float(dist_gap_np[i]),
                'free_acc': float(free_acc_np[i]),
                'interaction_acc': float(interaction_acc_np[i]),
                'idm_acc': float(idmacc[i]),
                'delta_v': float(delta_v_np[i]),
                'a': float(np.asarray(params.a)[i]),
                'v0': float(np.asarray(params.v0)[i]),
                'T': float(np.asarray(params.T)[i]),
                's0': float(np.asarray(params.s0)[i]),
                'b': float(np.asarray(params.b)[i]),
                'delta': float(np.asarray(params.delta)[i]),
                'length': float(np.asarray(params.length)[i]),
                'rtime': float(np.asarray(params.rtime)[i]),
            })
    return idmacc


def compute_stopping_acc(
    v: jnp.ndarray,
    dist_to_target: jnp.ndarray,
    params: IDMParams
) -> jnp.ndarray:
    """
    计算为了在目标位置停止所需的加速度
    只有当距离目标小于 3 倍当前车速时，才开始计算停止加速度，否则返回0
    保持可微（避免if分支），用sigmoid平滑切换
    """
    # 距离目标的净距离
    net_dist = dist_to_target - params.s0

    # 阈值：3倍车速
    threshold = 2.0 * v

    # 平滑掩码，sigmoid在0附近切换
    stop_mask = jax.nn.sigmoid(10.0 * (threshold - net_dist))  # net_dist < threshold 时趋近1

    # 物理公式: v^2 = 2 * a * d  => a = -v^2 / (2d)
    req_acc = -(v ** 2) / (2.0 * jnp.maximum(net_dist, 0.1))

    # 限制最大减速度，防止数值爆炸，但允许紧急制动
    stop_acc = jnp.maximum(req_acc, -9.0)

    # 只有在掩码为1时才施加停止加速度，否则为0
    return stop_mask,stop_acc


# 距离目标位置越近的点为头车，为第一辆车
# 可用于仿真前对车辆排序，确保IDM跟车关系正确

def sort_by_target_distance(position: jnp.ndarray, target_pos: jnp.ndarray):
    """
    按照位置升序排序，返回排序后的索引。
    位置最大的为头车（索引0），依次类推。
    """
    dist =  target_pos-position
    sorted_idx = jnp.argsort(dist)
    #print(dist,sorted_idx)
    return sorted_idx

# ==========================================
# 3. Brax 风格环境定义
# ==========================================

class BraxIDMEnv:
    def __init__(self, num_vehicles: int = 2, dt: float = 0.1, red_light_pos: float = 100.0):
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.red_light_pos = red_light_pos  # 红灯位置

    def reset(self, rng: jnp.ndarray, init_pos: jnp.ndarray,init_vel: jnp.ndarray, params: IDMParams) -> EnvState:
        """初始化环境，所有车辆目标点为1000，支持初始速度设定"""
        target_pos = jnp.ones(self.num_vehicles) * 1000.0
        return EnvState(
            position=init_pos,
            velocity=init_vel,
            acceleration=jnp.zeros(self.num_vehicles),
            target_pos=target_pos,
            params=params,
            step_count=0,
            crashed=False,
            front_car_id=jnp.full((self.num_vehicles,), -1, dtype=jnp.int32)
        )


    def step(self, state: EnvState, action: Optional[jnp.ndarray] = None, idm_log_list: list = None) -> EnvState:
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

        # 恢复原始顺序
        inv_idx = jnp.argsort(idx)
        # JAX风格前车id计算：头车为-1，其余为前一个排序后idx
        front_car_id_sorted = jnp.full((len(idx),), -1, dtype=jnp.int32)
        front_car_id_sorted = front_car_id_sorted.at[1:].set(idx[:-1])
        # 恢复原始顺序
        front_car_id = jnp.empty_like(front_car_id_sorted)
        front_car_id = front_car_id.at[inv_idx].set(front_car_id_sorted)

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

        #头车前方无车，gap设为极大，前车速度设为0，头车的idx为0
        #index 0 is the front car of index 1
        pos_front = jnp.concatenate([jnp.array([1e9]),pos[0:-1]])
        vel_front = jnp.concatenate([jnp.array([0.0]),vel[0:-1]])
        # 跟车净距 = 前车头位置 - 本车头位置 - 本车车长
        gaps = pos_front - pos - p_sorted.length

        # IDM加速度
        acc_idm = compute_idm_acc(
            vel, vel_front, gaps, p_sorted,
            front_car_id=front_car_id_sorted,
            log_list=idm_log_list,
            step_count=state.step_count,
            car_ids=list(idx)
        )

        # 到目标点的距离
        dist_to_target = tgt - pos - p_sorted.length
        stop_mask2,acc_stop = compute_stopping_acc(vel, dist_to_target, p_sorted)

        # 取更强制动（停车优先）
        #target_mask = tgt < 1e8
        final_acc = jnp.where(stop_mask2, jnp.minimum(acc_idm, acc_stop), acc_idm)
        #final_acc = acc_idm
        # 平滑加速度
        alpha = 1.0 - jnp.exp(-p_sorted.rtime / self.dt)
        smoothed_acc = acc + alpha * (final_acc - acc)
        smoothed_acc = jnp.clip(smoothed_acc, -9.0, p_sorted.a)

        # 积分更新
        new_vel = jnp.maximum(vel + smoothed_acc * self.dt, 0.0)
        new_pos = pos + new_vel * self.dt

        # 碰撞检测：后车不能超过前车
        pos_diff = new_pos[1:] - new_pos[:-1]
        
        #前车减后车距离小于5.5,就是碰撞
        is_crashed = jnp.any(-pos_diff < 5.5)
        #rint(new_pos)
        #print(pos_diff)

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
            crashed=bool(is_crashed),
            front_car_id=front_car_id
        )
        return new_state

    def rollout(self, state: EnvState, max_steps: int = 200, idm_log_csv: str = None):
        """
        仿真直到所有车辆通过目标点且无碰撞，或达到最大步数。
        达成目标：所有车辆都已通过目标点（位置大于目标点），且未发生碰撞。
        新增：可选记录每步IDM输入和中间量到csv。
        """
        traj = []
        idm_log_list = [] if idm_log_csv is not None else None
        for _ in range(max_steps):
            traj.append(state)
            if state.crashed:
                break
            # 达成目标：所有车都已通过目标点
            arrived = jnp.all(state.position - state.target_pos > 0.0)
            if arrived:
                break
            state = self.step(state, idm_log_list=idm_log_list)
        # 仿真结束后保存csv
        if idm_log_csv is not None and idm_log_list is not None:
            import pandas as pd
            df = pd.DataFrame(idm_log_list)
            df.to_csv(idm_log_csv, index=False,float_format='%.2f')
            print(f"已保存IDM日志到 {idm_log_csv}")
        return traj

# ==========================================
# 4. 测试脚本与可视化
# ==========================================

def plot_traj(traj, save_gif=False, gif_path="idm_traj.gif", save_csv=False, csv_path="traj_log.csv", log_detail=False):
    """可视化轨迹并可选保存为gif和csv（先保存为jpg，再合成gif）"""
    output_dir = "brax_sim_frames"
    frames = []
    log_rows = []
    if save_gif:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for idx, state in enumerate(traj):
        # 记录每辆车的参数和状态，先按位置从小到大排序
        sort_idx = jnp.argsort(state.position)
        pos_sorted = state.position[sort_idx]
        vel_sorted = state.velocity[sort_idx]
        acc_sorted = state.acceleration[sort_idx]
        tgt_sorted = state.target_pos[sort_idx]
        params_sorted = IDMParams(
            v0=state.params.v0[sort_idx],
            T=state.params.T[sort_idx],
            s0=state.params.s0[sort_idx],
            a=state.params.a[sort_idx],
            b=state.params.b[sort_idx],
            delta=state.params.delta[sort_idx],
            length=state.params.length[sort_idx],
            rtime=state.params.rtime[sort_idx]
        )
        for i, (x, v, a, t) in enumerate(zip(pos_sorted, vel_sorted, acc_sorted, tgt_sorted)):
            # 计算与前车的距离和速度差
            if i < len(pos_sorted) - 1:
                gap = pos_sorted[i+1] - x - params_sorted.length[i]
                dv = v - vel_sorted[i+1]
            else:
                gap = float(100000)
                dv = float(0)
            row = {
                "step": state.step_count,
                "car_id": int(sort_idx[i]),
                "position": float(x),
                "velocity": float(v),
                "acceleration": float(a),
                "target_pos": float(t),
                "gap_to_front": float(gap),
                "dv_to_front": float(dv),
                "v0": float(params_sorted.v0[i]),
                "T": float(params_sorted.T[i]),
                "s0": float(params_sorted.s0[i]),
                "a_param": float(params_sorted.a[i]),
                "b_param": float(params_sorted.b[i]),
                "delta": float(params_sorted.delta[i]),
                "length": float(params_sorted.length[i]),
                "rtime": float(params_sorted.rtime[i]),
                "crashed": state.crashed
            }
            log_rows.append(row)
            if log_detail:
                print(row)
        # 可视化部分
        fig, ax = plt.subplots(figsize=(18, 2))
        ax.set_xlim(0, float(jnp.max(state.target_pos)) + 10)
        ax.set_ylim(-1, 1)
        for i, (x, v, t) in enumerate(zip(state.position, state.velocity, state.target_pos)):
            ax.plot([x], [0], 'o', label=f"car{i} pos={float(x):.2f} v={float(v):.2f}")
            ax.axvline(t, color='r', linestyle='--', alpha=0.5)
            
        ax.legend()
        ax.set_title(f"step={state.step_count} crashed={state.crashed}")
        plt.tight_layout()
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
    if save_csv:
        df = pd.DataFrame(log_rows)
        df.to_csv(csv_path, index=False,float_format='%.2f')
        print(f"已保存轨迹日志到 {csv_path}")

if __name__ == "__main__":
    # 参数可调
    params = IDMParams(
        v0=jnp.array([55.0/3.6, 55.0/3.6]),  # 两辆车的期望速度
        T=jnp.array([1, 1]),
        s0=jnp.array([2.0, 2.0]),
        a=jnp.array([2, 2]),
        b=jnp.array([6.0, 6]),
        delta=jnp.array([4.0, 4.0]),
        length=jnp.array([5.0, 5.0]),
        rtime=jnp.array([0.02, 0.02])
    )
    env = BraxIDMEnv(num_vehicles=2, dt=0.1, red_light_pos=900.0)
    # 初始位置和目标
    init_pos = jnp.array([50.0/3.6*1+5, 0])  # 车1在后，车0在前.idm适合与稳定的跟车阶段，不是追赶阶段
    init_vel = jnp.array([50.0/3.6, 50.0/3.6])  # 车1在后，车0在前
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, init_pos, init_vel,params)
    traj = env.rollout(state, max_steps=500, idm_log_csv="idm_step_log.csv")
    plot_traj(traj, save_gif=False, gif_path="idm_2cars_redlight.gif", save_csv=True, csv_path="traj_log.csv", log_detail=False)

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