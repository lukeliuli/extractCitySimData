import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
def simple_func(x):
    # 一个简单的逐元素运算
    return jnp.sin(x) + jnp.cos(x) * 2.0

def simple_func_np(x):
    # numpy 原生向量化
    return np.sin(x) + np.cos(x) * 2.0

def scan_func(carry, x):
    # 累加和
    return carry + simple_func(x), carry + simple_func(x)

def test_jax_performance(device='cpu'):
    # 设置设备
    if device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')
    elif device == 'gpu':
        jax.config.update('jax_platform_name', 'gpu')
    else:
        raise ValueError('device must be "cpu" or "gpu"')

    print(f"\n==== JAX 测试 on {device.upper()} ====")
    N = 10**6
    x = jnp.linspace(0, 100, N)
    x_np = np.linspace(0, 100, N)

    # 1. 普通for循环
    t0 = time.time()
    y_np_for = np.array([np.sin(xi) + np.cos(xi) * 2.0 for xi in x_np])
    t1 = time.time()
    print(f"普通 numpy for 循环: {t1-t0:.4f} s")

    # 2. numpy 原生向量化
    t0 = time.time()
    y_np_vec = simple_func_np(x_np)
    t1 = time.time()
    print(f"numpy 向量化: {t1-t0:.4f} s")

    # 3. JAX 原生
    t0 = time.time()
    y_jax = simple_func(x)
    y_jax.block_until_ready()
    t1 = time.time()
    print(f"JAX 原生: {t1-t0:.4f} s")

    # 4. JAX jit
    jit_func = jax.jit(simple_func)
    t0 = time.time()
    y_jit = jit_func(x)
    y_jit.block_until_ready()
    t1 = time.time()
    print(f"JAX jit: {t1-t0:.4f} s")

    # 5. JAX vmap
    vmap_func = jax.vmap(simple_func)
    t0 = time.time()
    y_vmap = vmap_func(x)
    y_vmap.block_until_ready()
    t1 = time.time()
    print(f"JAX vmap: {t1-t0:.4f} s")
    
    '''
    # 6. JAX scan
    t0 = time.time()
    carry_init = 0.0
    _, y_scan = jax.lax.scan(scan_func, carry_init, x)
    y_scan.block_until_ready()
    t1 = time.time()
    print(f"JAX scan: {t1-t0:.4f} s")

    # 7. JAX jit+scan
    jit_scan_func = jax.jit(lambda x: jax.lax.scan(scan_func, 0.0, x)[1])
    t0 = time.time()
    y_jit_scan = jit_scan_func(x)
    y_jit_scan.block_until_ready()
    t1 = time.time()
    print(f"JAX jit+scan: {t1-t0:.4f} s")
    '''
    print("结果一致性检查:", np.allclose(np.array(y_jax), y_np_vec, atol=1e-5))

    # ====== 微分对比 ======
    print("\n==== 微分/梯度计算对比 ====")
    # 1. numpy 数值微分
    t0 = time.time()
    dx = 1e-5
    grad_np = (simple_func_np(x_np + dx) - simple_func_np(x_np - dx)) / (2 * dx)
    t1 = time.time()
    print(f"numpy 数值微分: {t1-t0:.4f} s")

    # 2. JAX grad
    grad_func = jax.grad(lambda x: jnp.sum(simple_func(x)))
    t0 = time.time()
    grad_jax = grad_func(x)
    grad_jax.block_until_ready()
    t1 = time.time()
    print(f"JAX grad: {t1-t0:.4f} s")

    # 3. JAX jit+grad
    jit_grad_func = jax.jit(grad_func)
    t0 = time.time()
    grad_jit = jit_grad_func(x)
    grad_jit.block_until_ready()
    t1 = time.time()
    print(f"JAX jit+grad: {t1-t0:.4f} s")

    # 4. JAX vmap+grad（对每个元素求导）
    grad_elem_func = jax.vmap(jax.grad(lambda x: simple_func(x)))
    t0 = time.time()
    grad_vmap = grad_elem_func(x)
    grad_vmap.block_until_ready()
    t1 = time.time()
    print(f"JAX vmap+grad: {t1-t0:.4f} s")

    print("梯度一致性检查:", np.allclose(np.array(grad_jax), grad_np, atol=1e-4))
    
    # ====== JAX实现经典物理模型：简谐振子（弹簧振子） ======
    t0 = time.time()
    print("\n==== JAX 简谐振子时序模拟 ====")
    # 参数
    m = 1.0  # 质量
    k = 1.0  # 弹簧系数
    dt = 0.01
    steps = 10000

    def harmonic_oscillator_step(state, _):
        x, v = state
        a = -k * x / m
        v_new = v + a * dt
        x_new = x + v_new * dt
        return (x_new, v_new), (x_new, v_new)

    # 初始状态
    x0 = 1.0  # 初始位移
    v0 = 0.0  # 初始速度
    state0 = (x0, v0)

    # 用JAX scan进行时序模拟
    _, traj = jax.lax.scan(harmonic_oscillator_step, state0, None, length=steps)
    x_traj, v_traj = traj

    print("简谐振子末端位置: ", x_traj[-1])
    print("简谐振子末端速度: ", v_traj[-1])
    t1 = time.time()
    print(f"简谐振子: {t1-t0:.4f} s")
    # ====== JAX实现更复杂的经典物理模型：双摆系统 ======
    t0 = time.time()
    print("\n==== JAX 双摆系统时序模拟 ====")
    # 双摆参数
    m1, m2 = 1.0, 1.0  # 两个摆锤的质量
    l1, l2 = 1.0, 1.0  # 两个摆锤的长度
    g = 9.81           # 重力加速度

    def double_pendulum_step(state, _):
        theta1, omega1, theta2, omega2 = state

        delta = theta2 - theta1
        den1 = (m1 + m2) * l1 - m2 * l1 * jnp.cos(delta) ** 2
        den2 = (l2 / l1) * den1

        a1 = (m2 * l1 * omega1 ** 2 * jnp.sin(delta) * jnp.cos(delta) +
              m2 * g * jnp.sin(theta2) * jnp.cos(delta) +
              m2 * l2 * omega2 ** 2 * jnp.sin(delta) -
              (m1 + m2) * g * jnp.sin(theta1)) / den1

        a2 = (-m2 * l2 * omega2 ** 2 * jnp.sin(delta) * jnp.cos(delta) +
              (m1 + m2) * g * jnp.sin(theta1) * jnp.cos(delta) -
              (m1 + m2) * l1 * omega1 ** 2 * jnp.sin(delta) -
              (m1 + m2) * g * jnp.sin(theta2)) / den2

        omega1_new = omega1 + a1 * dt
        theta1_new = theta1 + omega1_new * dt
        omega2_new = omega2 + a2 * dt
        theta2_new = theta2 + omega2_new * dt

        return (theta1_new, omega1_new, theta2_new, omega2_new), (theta1_new, omega1_new, theta2_new, omega2_new)

    # 初始状态
    theta1_0 = jnp.pi / 2
    omega1_0 = 0.0
    theta2_0 = jnp.pi / 2
    omega2_0 = 0.0
    state0 = (theta1_0, omega1_0, theta2_0, omega2_0)

    # 用JAX scan进行时序模拟
    _, traj = jax.lax.scan(double_pendulum_step, state0, None, length=steps)
    theta1_traj, omega1_traj, theta2_traj, omega2_traj = traj

    print("双摆末端角度1: ", theta1_traj[-1])
    print("双摆末端角速度1: ", omega1_traj[-1])
    print("双摆末端角度2: ", theta2_traj[-1])
    print("双摆末端角速度2: ", omega2_traj[-1])
    t1 = time.time()
    print(f"双摆系统: {t1-t0:.4f} s")


if __name__ == "__main__":
    print("JAX 当前设备:", jax.devices())
    


    # 检查设备列表
    print("Available devices:", jax.devices())
    
    # 选择一个设备（如果有多个 GPU，可以选择特定的 GPU）
    device = jax.devices("gpu")[0]  # 如果你只有一个 GPU，这将是 [0]
    print("Selected device:", device)
    
    # 测试在 GPU 上运行一个简单的操作
    key = random.PRNGKey(0)
    x = random.normal(key, (1000, 1000))
    y = jax.device_put(x, device)  # 将数据放到 GPU 上
    result = jnp.dot(y, y)  # 在 GPU 上计算点积
    print("Result on GPU:", result)


    # CPU 测试
    test_jax_performance('cpu')
    # GPU 测试（如有GPU）
    try:
        test_jax_performance('gpu')
    except Exception as e:
        print("GPU 测试失败，原因：", e)


    print("\n所有测试完成。越复杂的物理系统计算，JAX的CPU和GPU差距不明显，尤其是在GPU上。")
    print("\n所有测试完成。越复杂越多粒子同时进行相同计算，JAX的CPU和GPU差距明显，尤其是在GPU上。")