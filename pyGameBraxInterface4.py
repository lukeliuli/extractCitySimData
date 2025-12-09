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
8.参考pyGameInterface3.py的写法，完成brax模型的编写。注意brax的环境编写方式和pyGameInterface3.py不一樣
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

import datetime
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
    red_light_pos: float       # 红灯位置
    red_light_state: bool      # 红灯是否为红(True)/绿(False)
    red_light_remaining: float # 红灯剩余时间(秒)
    time_to_vanish: jnp.ndarray # 每辆车通过红灯的时间，未通过为-1
    acc_stop: jnp.ndarray      # 停止加速度
    final_acc: jnp.ndarray     # 最终加速度
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
    net_dist = dist_to_target


    # 物理公式: v^2 = 2 * a * d  => a = -v^2 / (2d)
    req_acc = -(v ** 2) / (2.0 * jnp.maximum(net_dist, 0.01))

    # 限制最大减速度，防止数值爆炸，但允许紧急制动
    
    stop_acc = jnp.maximum(req_acc, -9.0)

    # 如果距离非常近，强制急停（JAX兼容写法）
    stop_acc = jnp.where(net_dist <= 0, -9.0, stop_acc)

    # 只有在掩码为1时才施加停止加速度，否则为0
    return stop_acc


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
    def __init__(self, num_vehicles: int = 2, dt: float = 0.1, red_light_pos: float = 100.0, red_light_duration: float = 30.0):
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.red_light_pos = red_light_pos  # 红灯位置
        self.red_light_duration = red_light_duration  # 红灯时长(秒)

    def reset(self, rng: jnp.ndarray, init_pos: jnp.ndarray,init_vel: jnp.ndarray, params: IDMParams) -> EnvState:
        """初始化环境，所有车辆目标点为1000，支持初始速度设定"""
        target_pos = jnp.ones(self.num_vehicles) * 300.0
        return EnvState(
            position=init_pos,
            velocity=init_vel,
            acceleration=jnp.zeros(self.num_vehicles),
            target_pos=target_pos,
            params=params,
            step_count=0,
            crashed=False,
            front_car_id=jnp.full((self.num_vehicles,), -1, dtype=jnp.int32),
            red_light_pos=self.red_light_pos,
            red_light_state=True,  # 初始为红灯
            red_light_remaining=self.red_light_duration,
            time_to_vanish=jnp.full((self.num_vehicles,), -1.0),
            acc_stop=jnp.zeros(self.num_vehicles),
            final_acc=jnp.zeros(self.num_vehicles)
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

        # 红灯逻辑：如果红灯为红，且车辆距离红灯小于一定阈值，则目标点临时设为红灯位置
        red_light_arr = jnp.ones_like(pos) * state.red_light_pos
        near_red = (red_light_arr - pos) < vel * 3.0  # 3秒内能到达红灯位置
        near_red2 = (red_light_arr - pos) < 30  # 30米内能到达红灯位置
        not_passed_red = pos < state.red_light_pos
        near_red2 = near_red2 | near_red2
        stop_mask = near_red2 & not_passed_red & state.red_light_state
        tgt = jnp.where(stop_mask, state.red_light_pos, tgt)

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
        dist_to_target = tgt - pos - p_sorted.length/2
        acc_stop = compute_stopping_acc(vel, dist_to_target, p_sorted)

        # 取更强制动（停车优先）
        #target_mask = tgt < 1e8
        final_acc = jnp.where(stop_mask, jnp.minimum(acc_idm, acc_stop), acc_idm)
        #final_acc = acc_idm
        # 平滑加速度
        alpha = jnp.exp(-p_sorted.rtime/10/self.dt)
        smoothed_acc = alpha*final_acc + (1-alpha) * acc
        smoothed_acc = jnp.clip(smoothed_acc, -9.0, p_sorted.a)
      

        # 积分更新
        new_vel = jnp.maximum(vel + smoothed_acc * self.dt, 0.0)
        new_pos = pos + new_vel * self.dt

        # 碰撞检测：后车不能超过前车
        pos_diff = new_pos[1:] - new_pos[:-1]
        
        #前车减后车距离小于5.5,就是碰撞
        is_crashed = jnp.any(-pos_diff < 5)
        #rint(new_pos)
        #print(pos_diff)

        # 红灯倒计时与状态切换
        red_light_remaining = state.red_light_remaining - self.dt if state.red_light_state else 0.0
        red_light_state = state.red_light_state
        if state.red_light_state and red_light_remaining <= 0:
            red_light_state = False  # 红灯变绿
            red_light_remaining = 0.0

        # 记录每辆车通过红灯的时间
        prev_time_to_vanish = state.time_to_vanish
        # 通过红灯条件：上一步没通过，这一步位置>=红灯位置
        passed_mask = (state.position < state.red_light_pos) & (new_pos[inv_idx] >= state.red_light_pos)
        new_time_to_vanish = jnp.where((prev_time_to_vanish < 0) & passed_mask, (state.step_count+1)*self.dt, prev_time_to_vanish)

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
            front_car_id=front_car_id,
            red_light_pos=state.red_light_pos,
            red_light_state=red_light_state,
            red_light_remaining=red_light_remaining,
            time_to_vanish=new_time_to_vanish,
            acc_stop=acc_stop[inv_idx],
            final_acc=final_acc[inv_idx]
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

        # 保存traj到csv
        traj_log = []
        for state in traj:
            for i in range(self.num_vehicles):
                traj_log.append({
                    "step": state.step_count,
                    "car_id": i,
                    "position": float(state.position[i]),
                    "velocity": float(state.velocity[i]),
                    "acceleration": float(state.acceleration[i]),
                    "target_pos": float(state.target_pos[i]),
                    "acc_stop": float(state.acc_stop[i]),
                    "final_acc": float(state.final_acc[i]),
                    "v0": float(state.params.v0[i]),
                    "T": float(state.params.T[i]),
                    "s0": float(state.params.s0[i]),
                    "a_param": float(state.params.a[i]),
                    "b_param": float(state.params.b[i]),
                    "delta": float(state.params.delta[i]),
                    "length": float(state.params.length[i]),
                    "rtime": float(state.params.rtime[i]),
                    "crashed": state.crashed,
                    "front_car_id": int(state.front_car_id[i]),
                    "red_light_pos": float(state.red_light_pos),
                    "red_light_state": state.red_light_state,
                    "red_light_remaining": float(state.red_light_remaining),
                    "time_to_vanish": float(state.time_to_vanish[i])
                })

        df_traj = pd.DataFrame(traj_log)
        now = datetime.datetime.now()
        filename = f"traj_log_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        df_traj.to_csv(filename, index=False, float_format='%.3f')
       
       
        if idm_log_csv is not None and idm_log_list is not None:
            filename = f"idm_log_{now.strftime('%Y%m%d_%H%M%S')}.csv"
            df = pd.DataFrame(idm_log_list)
            df.to_csv(filename, index=False,float_format='%.3f')
            #print(f"已保存IDM日志到 {idm_log_csv}")
        return traj

# ==========================================
# 4. 测试脚本与可视化
# ==========================================
# 在plot_traj中增加红灯状态和剩余时间的可视化
def plot_red_light(ax, red_light_pos, red_light_state, red_light_remaining):
    color = 'red' if red_light_state else 'green'
    ax.axvline(red_light_pos, color=color, linestyle='-', linewidth=3, alpha=0.6, label=f"RedLight ({color})")
    ax.text(red_light_pos + 1, 0.7, f"{'红灯' if red_light_state else '绿灯'}\n剩余{red_light_remaining:.1f}s",
            color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))

# 修改plot_traj调用
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
        acc_stop_sorted = state.acc_stop[sort_idx]
        final_acc_sorted = state.final_acc[sort_idx]
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
                "acc_stop": float(acc_stop_sorted[i]),
                "final_acc": float(final_acc_sorted[i]),
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
            # ax.axvline(t, color='r', linestyle='--', alpha=0.5) # 暂时注释掉目标位置显示，避免与红灯混淆

        # 增加红灯显示
        red_light_color = 'red' if state.red_light_state else 'green'
        ax.axvline(state.red_light_pos, color=red_light_color, linestyle='-', linewidth=2, label=f"Red Light Pos: {state.red_light_pos:.2f}")

        ax.legend(loc='upper left')
        title = f"Step: {state.step_count} | Crashed: {state.crashed} | Red Light: {'ON' if state.red_light_state else 'OFF'} | Time Left: {state.red_light_remaining:.1f}s"
        ax.set_title(title)
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


##########################################
# 测试二车idm模型 ,看环境仿真效果
##########################################
def test1():
    """测试二车idm模型 ,看环境仿真效果 """
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
    env = BraxIDMEnv(num_vehicles=2, dt=0.1, red_light_pos=100.0,red_light_duration=30.0)
    # 初始位置和目标
    init_pos = jnp.array([50.0/3.6*1+5, 0])  # 车1在后，车0在前.idm适合与稳定的跟车阶段，不是追赶阶段
    init_vel = jnp.array([50.0/3.6, 50.0/3.6])  # 车1在后，车0在前
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, init_pos, init_vel,params)
    traj = env.rollout(state, max_steps=500, idm_log_csv="idm_step_log.csv")
    plot_traj(traj, save_gif=False, gif_path="idm_2cars_redlight.gif", save_csv=True, csv_path="traj_log.csv", log_detail=False)
    # 输出每辆车通过红灯路口的时间
    print("每辆车通过红灯路口的时间（秒）：")
    for i, t in enumerate(traj[-1].time_to_vanish):
        print(f"car{i}: {float(t):.2f}")
#
#-------------------------------------------------------------- 
##主函数    
if __name__ == "__main__":
    test1()############### 测试二车idm模型 ,看环境仿真效果 ##################
