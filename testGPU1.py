import tensorflow as tf
import time

# Check TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPUs Available:")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("No GPUs found.")

# Test TensorFlow GPU functionality
# Define a simple computation: matrix multiplication
matrix_size = 10000

# Create random matrices
a = tf.random.normal([matrix_size, matrix_size])
b = tf.random.normal([matrix_size, matrix_size])


if gpus:
    with tf.device('/GPU:0'):
        start_time = time.time()
        c = tf.matmul(a, b)
        gpu_time = time.time() - start_time
        print(f"GPU computation time (larger matrices): {gpu_time:.4f} seconds")

# Measure time for CPU computation with larger matrices
with tf.device('/CPU:0'):
    start_time = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"CPU computation time (larger matrices): {cpu_time:.4f} seconds")

# Compare results with larger matrices
if gpus:
    print(f"GPU is {cpu_time / gpu_time:.2f}x faster than CPU with larger matrices")
