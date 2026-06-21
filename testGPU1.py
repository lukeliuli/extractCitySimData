import tensorflow as tf

# 查看是否有 GPU
print("TensorFlow 版本:", tf.__version__)
print("="*50)
print("可用 GPU 列表：", tf.config.list_physical_devices('GPU'))
print("可用 CPU 列表：", tf.config.list_physical_devices('CPU'))
print("="*50)