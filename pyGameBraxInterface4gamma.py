#准备测试brax,物理可微模型
# pyGameBraxInterface4beta.py，基于JAX和Flax实现的IDM交通流模型环境，支持多车仿真、红绿灯逻辑和数据记录功能。
# 该环境采用函数式编程风格，利用JAX的自动微分和JIT编译特性，实现高效的交通流仿真。
# 环境状态和车辆参数均定义为JAX的PyTree结构，便于与JAX生态系统集成。
# 现对于pyGameBraxInterface4.py进行完善和删除不适合GPU和TPU和并行部分，确保其功能完整且符合预期。


import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict, Any, Optional

import os
import numpy as np


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
    collision: jnp.ndarray             # 是否发生碰撞
    front_car_id: jnp.ndarray  # shape=(N,)
    red_light_pos: float       # 红灯位置
    red_light_state: bool      # 红灯是否为红(True)/绿(False)
    red_light_remaining: float # 红灯剩余时间(秒)
    time_to_vanish: jnp.ndarray # 每辆车通过红灯的时间，未通过为-1
    acc_stop: jnp.ndarray      # 停止加速度
    final_acc: jnp.ndarray     # 最终加速度
    v_front: jnp.ndarray       # 前车速度
    dist_gap: jnp.ndarray     # 与前车净距
    free_acc: jnp.ndarray     # IDM自由流加速度
    interaction_acc: jnp.ndarray # IDM交互项加速度
    dt: float                  # 时间步长
# ==========================================
# 2. 核心物理逻辑 (JAX Functional Style)
# ==========================================

#@jax.jit,因为一般只有2到6辆车，具体效果不好
def compute_idm_acc(
    v: jnp.ndarray,
    v_front: jnp.ndarray,
    dist_gap: jnp.ndarray,
    params: IDMParams,
    front_car_id: jnp.ndarray 
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

    return idmacc,free_acc,interaction_acc



#@jax.jit,因为一般只有2到6辆车，具体效果不好
def compute_stopping_acc(
    v: jnp.ndarray,
    dist_to_target: jnp.ndarray,
) -> jnp.ndarray:
  
   
    net_dist = dist_to_target

    # 物理公式: v^2 = 2 * a * d  => a = -v^2 / (2d)
    req_acc = -(v ** 2) / (2.0 * jnp.maximum(net_dist, 0.01))

    # 限制最大减速度，防止数值爆炸，但允许紧急制动
    stop_acc = jnp.maximum(req_acc, -9.0)

    # 如果距离非常近，强制急停（JAX兼容写法）
    stop_acc = jnp.where(net_dist <= 0, -9.0, stop_acc)
    return stop_acc
    
#@jax.jit,因为一般只有2到6辆车，具体效果不好
def step_pure(state: EnvState, num_vehicles: int, dt: float) -> EnvState:
    """
    单步仿真纯函数，所有依赖参数显式传递。
    """
    
    N = num_vehicles
    p = state.params

    idx = jnp.argsort(-state.position)
    pos = state.position[idx]
    vel = state.velocity[idx]
    acc = state.acceleration[idx]
    tgt = state.target_pos[idx]
    collision = state.collision[idx]
    p_sorted = IDMParams(
        v0=p.v0[idx], T=p.T[idx], s0=p.s0[idx], a=p.a[idx], b=p.b[idx],
        delta=p.delta[idx], length=p.length[idx], rtime=p.rtime[idx]
    )
    inv_idx = jnp.argsort(idx)
    front_car_id_sorted = jnp.full((len(idx),), -1, dtype=jnp.int32)
    front_car_id_sorted = front_car_id_sorted.at[1:].set(idx[:-1])
    front_car_id = jnp.empty_like(front_car_id_sorted)
    front_car_id = front_car_id.at[inv_idx].set(front_car_id_sorted)

    red_light_arr = jnp.ones_like(pos) * state.red_light_pos
    near_red = (red_light_arr - pos) < vel * 3.0
    near_red2 = (red_light_arr - pos) < 30
    not_passed_red = pos < state.red_light_pos
    near_red2 = near_red2 | near_red2
    stop_mask = near_red2 & not_passed_red & state.red_light_state
    tgt = jnp.where(stop_mask, state.red_light_pos, tgt)

    pos_front = jnp.concatenate([jnp.array([1e9]), pos[0:-1]])
    vel_front = jnp.concatenate([jnp.array([0.0]), vel[0:-1]])
    gaps = pos_front - pos - p_sorted.length

    acc_idm, free_acc, interaction_acc = compute_idm_acc(
        vel, vel_front, gaps, p_sorted,
        front_car_id=front_car_id_sorted
    )
    
    dist_to_target = tgt - pos - p_sorted.length / 2
    acc_stop = compute_stopping_acc(vel, dist_to_target)

    final_acc = jnp.where(stop_mask, jnp.minimum(acc_idm, acc_stop), acc_idm)
    alpha = jnp.exp(-p_sorted.rtime / 10 / dt)
    smoothed_acc = alpha * final_acc + (1 - alpha) * acc
    smoothed_acc = jnp.clip(smoothed_acc, -9.0, p_sorted.a)
    

    new_vel = jnp.maximum(vel + smoothed_acc * dt, 0.0)
    new_pos = pos + new_vel * dt

  

    pos_diff = new_pos[1:] - new_pos[:-1]
   
    collision = collision.at[1:].set(jnp.where(pos_diff < 5.5, 1.0, 0.0))

    # 对于虚拟占位车（速度<0），保持速度和位置不变
    is_virtual = (pos < -500) | (vel < 0) 
    new_vel = jnp.where(is_virtual, vel, new_vel)
    new_pos = jnp.where(is_virtual, pos, new_pos)
    collision = jnp.where(is_virtual, 0.0, collision)
    smoothed_acc = jnp.where(is_virtual, state.acceleration, smoothed_acc)
    front_car_id = jnp.where(is_virtual, state.front_car_id, front_car_id)
    acc_stop = jnp.where(is_virtual, state.acc_stop, acc_stop)
    final_acc = jnp.where(is_virtual, state.final_acc, final_acc)
    free_acc = jnp.where(is_virtual, state.free_acc,  free_acc)
   
    red_light_remaining = jnp.where(state.red_light_state, state.red_light_remaining - dt, 0.0)
    red_light_switch = (state.red_light_state) & (red_light_remaining <= 0)
    red_light_state = jnp.where(red_light_switch, False, state.red_light_state)
    red_light_remaining = jnp.where(red_light_switch, 0.0, red_light_remaining)
    

    prev_time_to_vanish = state.time_to_vanish
    passed_mask = (state.position < state.red_light_pos) & (new_pos[inv_idx] >= state.red_light_pos)
    new_time_to_vanish = jnp.where((prev_time_to_vanish < 0) & passed_mask, (state.step_count + 1) * dt, prev_time_to_vanish)
    
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
        collision=collision[inv_idx],
        front_car_id=front_car_id,
        red_light_pos=state.red_light_pos,
        red_light_state=red_light_state,
        red_light_remaining=red_light_remaining,
        time_to_vanish=new_time_to_vanish,
        acc_stop=acc_stop[inv_idx],
        final_acc=final_acc[inv_idx],
        v_front=vel_front[inv_idx],
        dist_gap=gaps[inv_idx],
        free_acc=free_acc[inv_idx],
        interaction_acc=interaction_acc[inv_idx],
        dt=state.dt
    )

    # 在 step_pure 末尾，返回 new_state 前：
    # 强制统一 dtype 为 float32，避免 JAX 认为结构不同
    new_state = EnvState(
        position=new_state.position.astype(jnp.float32),
        velocity=new_state.velocity.astype(jnp.float32),
        acceleration=new_state.acceleration.astype(jnp.float32),
        target_pos=new_state.target_pos.astype(jnp.float32),
        params=IDMParams(
            v0=new_state.params.v0.astype(jnp.float32),
            T=new_state.params.T.astype(jnp.float32),
            s0=new_state.params.s0.astype(jnp.float32),
            a=new_state.params.a.astype(jnp.float32),
            b=new_state.params.b.astype(jnp.float32),
            delta=new_state.params.delta.astype(jnp.float32),
            length=new_state.params.length.astype(jnp.float32),
            rtime=new_state.params.rtime.astype(jnp.float32),
        ),
        step_count=new_state.step_count,
        collision=new_state.collision.astype(jnp.float32),
        front_car_id=new_state.front_car_id,  # keep int32
        red_light_pos=new_state.red_light_pos.astype(jnp.float32),
        red_light_state=new_state.red_light_state,
        red_light_remaining=new_state.red_light_remaining.astype(jnp.float32),
        time_to_vanish=new_state.time_to_vanish.astype(jnp.float32),
        acc_stop=new_state.acc_stop.astype(jnp.float32),
        final_acc=new_state.final_acc.astype(jnp.float32),
        v_front=new_state.v_front.astype(jnp.float32),
        dist_gap=new_state.dist_gap.astype(jnp.float32),
        free_acc=new_state.free_acc.astype(jnp.float32),
        interaction_acc=new_state.interaction_acc.astype(jnp.float32),
        dt=new_state.dt.astype(jnp.float32),
    )

    return new_state




def rollout_pure(state: EnvState, num_vehicles: int, dt: float, max_steps: int = 200):
    """
    纯函数版 rollout，所有依赖参数显式传递。
    """

    def scan_step(state, _):
        new_state = step_pure(state, num_vehicles, dt)
        return new_state, state

    steps = max_steps
    _, traj_stacked = jax.lax.scan(scan_step, state, None, length=steps)
    traj = [jax.tree_util.tree_map(lambda x: x[i], traj_stacked) for i in range(steps)]
    # 不再提前 break，始终返回完整 traj
    return traj

def initial_env_state_pure(num_vehicles: int, dt: float, init_pos: jnp.ndarray, init_vel: jnp.ndarray, params: IDMParams,\
                            red_light_pos: float, red_light_duration: float) -> EnvState:
    target_pos = jnp.ones(num_vehicles, dtype=jnp.float32) * 300.0
    return EnvState(
        position=jnp.asarray(init_pos, dtype=jnp.float32),
        velocity=jnp.asarray(init_vel, dtype=jnp.float32),
        acceleration=jnp.zeros(num_vehicles, dtype=jnp.float32),
        target_pos=target_pos,
        params=IDMParams(
            v0=jnp.asarray(params.v0, dtype=jnp.float32),
            T=jnp.asarray(params.T, dtype=jnp.float32),
            s0=jnp.asarray(params.s0, dtype=jnp.float32),
            a=jnp.asarray(params.a, dtype=jnp.float32),
            b=jnp.asarray(params.b, dtype=jnp.float32),
            delta=jnp.asarray(params.delta, dtype=jnp.float32),
            length=jnp.asarray(params.length, dtype=jnp.float32),
            rtime=jnp.asarray(params.rtime, dtype=jnp.float32),
        ),
        step_count=0,
        collision=jnp.zeros(num_vehicles, dtype=jnp.float32),  # 注意：collision 应该是 bool？但你用 float，保持一致
        front_car_id=jnp.full((num_vehicles,), -1, dtype=jnp.int32),
        red_light_pos=jnp.float32(red_light_pos),  # scalar float is OK
        red_light_state=True,
        red_light_remaining=jnp.float32(red_light_duration),
        time_to_vanish=jnp.full((num_vehicles,), -1.0, dtype=jnp.float32),
        acc_stop=jnp.zeros(num_vehicles, dtype=jnp.float32),
        final_acc=jnp.zeros(num_vehicles, dtype=jnp.float32),
        v_front=jnp.full((num_vehicles,), -1.0, dtype=jnp.float32),
        dist_gap=jnp.full((num_vehicles,), -1.0, dtype=jnp.float32),
        free_acc=jnp.full((num_vehicles,), -1.0, dtype=jnp.float32),
        interaction_acc=jnp.full((num_vehicles,), -1.0, dtype=jnp.float32),
        dt=jnp.float32(dt),
    )



  
        


##########################################
# 测试二车idm模型 ,看环境仿真效果
##########################################
from jax import tree_util
import random
def test3():
    """测试N车idm模型 ,看环境仿真效果。 测试中"""
    batch_size = 100
    rng0= random.randint(0,1e9)
    rng = jax.random.PRNGKey(rng0)
    N_vehicle = 2
    # 随机生成100组参数
   
    init_pos0 = jax.random.uniform(rng, (batch_size, N_vehicle), minval=0.0, maxval=30.0)  # 例如在0~20米内随机
    init_vel0 = jax.random.uniform(rng, (batch_size, N_vehicle), minval=45/3.6,maxval=50/3.6)# 例如在0~10m/s内随机
    v0s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=50.0/3.6, maxval=60.0/3.6)
    Ts = jax.random.uniform(rng, (batch_size, N_vehicle), minval=0.8, maxval=1.8)
    s0s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=1.0, maxval=3.0)
    a_s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=1.0, maxval=3.0)
    b_s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=4.0, maxval=8.0)
    deltas = jnp.ones((batch_size, N_vehicle)) * 4.0
    lengths = jnp.ones((batch_size, N_vehicle)) * 5.0
    rtimes = jnp.ones((batch_size, N_vehicle)) * 0.02
    red_light_pos = 100.0
    red_light_duration = 30.0
    dt = 0.1
    


    states = []
    for i in range(batch_size):
        state = initial_env_state_pure(num_vehicles=N_vehicle, dt=dt, init_pos=init_pos0[i], init_vel=init_vel0[i], params=IDMParams(
            v0=v0s[i],
            T=Ts[i],
            s0=s0s[i],
            a=a_s[i],
            b=b_s[i],
            delta=deltas[i],
            length=lengths[i],
            rtime=rtimes[i]
        ),
        red_light_pos=red_light_pos,
        red_light_duration=red_light_duration)
        states.append(state)

    states = tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    # 由于rollout_pure返回的是list，不能直接vmap。我们只关心最后一个状态的time_to_vanish
    def get_time_to_vanish(state):
        traj = rollout_pure(state, num_vehicles=2, dt=0.1, max_steps=500)
        return traj[-1].time_to_vanish

    batched_time_to_vanish = jax.vmap(get_time_to_vanish)(states)
    # 计算每辆车通过红灯的平均时间（忽略未通过的车辆，即time_to_vanish < 0的不用计入平均值）
    mean_time = jnp.where(
        batched_time_to_vanish >= 0,
        batched_time_to_vanish,
        jnp.nan
    ).mean(axis=0)
    print("每辆车通过红灯的平均时间（秒）：")
    for i, t in enumerate(mean_time):
        print(f"car{i}: {float(t):.2f}")
    
def test4():
    """测试二车idm模型 ,看环境仿真效果。 测试中"""
    batch_size = 1
    rng0= random.randint(0,1e9)
    rng = jax.random.PRNGKey(rng0)
    # 随机生成100组参数
    N_vehicle = 20
    init_pos0 = jax.random.uniform(rng, (batch_size, N_vehicle), minval=5.0, maxval=30.0)  # 例如在0~20米内随机
    init_vel0 = jax.random.uniform(rng, (batch_size, N_vehicle), minval=45/3.6,maxval=50/3.6)# 例如在0~10m/s内随机
    v0s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=40.0/3.6, maxval=60.0/3.6)
    Ts = jax.random.uniform(rng, (batch_size, N_vehicle), minval=0.8, maxval=1.8)
    s0s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=1.0, maxval=3.0)
    a_s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=1.0, maxval=3.0)
    b_s = jax.random.uniform(rng, (batch_size, N_vehicle), minval=4.0, maxval=8.0)
    deltas = jnp.ones((batch_size, N_vehicle)) * 4.0
    lengths = jnp.ones((batch_size, N_vehicle)) * 5.0
    rtimes = jnp.ones((batch_size, N_vehicle)) * 0.02
    red_light_pos = 100.0
    red_light_duration = 20.0
    dt = 0.1
    
    init_pos0 = init_pos0.at[:, 8:N_vehicle].set(-1000.0)
    init_vel0 = init_vel0.at[:, 8:N_vehicle].set(-10.0)
    # 将第8到20辆车的位置设为-1000，速度设为-10
    states = []
    for i in range(batch_size):
        state = initial_env_state_pure(num_vehicles=N_vehicle, dt=dt, init_pos=init_pos0[i], init_vel=init_vel0[i], params=IDMParams(
            v0=v0s[i],
            T=Ts[i],
            s0=s0s[i],
            a=a_s[i],
            b=b_s[i],
            delta=deltas[i],
            length=lengths[i],
            rtime=rtimes[i]
        ),
        red_light_pos=red_light_pos,
        red_light_duration=red_light_duration)
        states.append(state)

    states = tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    # 由于rollout_pure返回的是list，不能直接vmap。我们只关心最后一个状态的time_to_vanish
    def get_time_to_vanish(state):
        traj = rollout_pure(state, num_vehicles=N_vehicle, dt=0.1, max_steps=500)
        return traj[-1].time_to_vanish

    batched_time_to_vanish = jax.vmap(get_time_to_vanish)(states)
    # 计算每辆车通过红灯的平均时间（忽略未通过的车辆，即time_to_vanish < 0的不用计入平均值）
    mean_time = jnp.where(
        batched_time_to_vanish >= 0,
        batched_time_to_vanish,
        jnp.nan
    ).mean(axis=0)
    print("每辆车通过红灯的平均时间（秒）：")
    for i, t in enumerate(mean_time):
        print(f"car{i}: {float(t):.2f}")
#
#-------------------------------------------------------------- 
##主函数    
import time
import os
from multiprocessing import Pool
from functools import partial
if __name__ == "__main__":
        # 设置设备
    
    jax.config.update('jax_platform_name', 'cpu')
    t0 = time.time()
    test3()
    t1 = time.time()
    print(f"test3:cpu: {t1-t0:.4f} s")
    


    t0 = time.time()
    test4()
    t1 = time.time()
    print(f"test4:cpu: {t1-t0:.4f} s")


    jax.config.update('jax_platform_name', 'gpu')
    t0 = time.time()
    test3()
    t1 = time.time()
    print(f"test3:gpu: {t1-t0:.4f} s")

    t0 = time.time()
    test4()
    t1 = time.time()
    print(f"test4:gpu: {t1-t0:.4f} s")




