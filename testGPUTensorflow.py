import tensorflow as tf
import time
import numpy as np

# ====================== 1. 打印GPU信息 ======================
print("="*60)
print("TensorFlow 版本:", tf.__version__)
print("="*60)

# 列出所有可用设备
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"检测到 CPU 数量: {len(cpus)}")
print(f"检测到 GPU 数量: {len(gpus)}")

if gpus:
    for gpu in gpus:
        print("✅ 找到GPU设备:", gpu)
else:
    print("❌ 未找到GPU，将使用CPU运行")

print("="*60)

# ====================== 2. 持续30秒GPU计算 ======================
print("开始 GPU 压力测试，持续 30 秒...")
start_time = time.time()

# 创建大矩阵，强制GPU计算（越大越吃显存/算力）
a = tf.random.normal([1024, 1024])
b = tf.random.normal([1024, 1024])

# 循环计算矩阵乘法，持续30秒
while time.time() - start_time < 30:
    c = tf.matmul(a, b)  # 矩阵乘法（GPU高负载）
    d = tf.reduce_sum(c)
    # 每5秒打印一次提示
    if int(time.time() - start_time) % 5 == 0:
        print(f"已运行 {int(time.time()-start_time)}s / 30s")

print("="*60)
print("✅ GPU 测试完成！")
print("="*60)
