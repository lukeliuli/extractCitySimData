import time
import jax
import jax.numpy as jnp
import numpy as np

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



if __name__ == "__main__":
    # CPU 测试
    test_jax_performance('cpu')
    # GPU 测试（如有GPU）
    try:
        test_jax_performance('gpu')
    except Exception as e:
        print("GPU 测试失败，原因：", e)
   