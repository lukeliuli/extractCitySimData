'''

参考modelsCollect3.py,做如下改进:
1.dt=0.5,提高速度
2.查查bug
'''

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
from jax import tree_util
import time
import os
import random
from pyGameBraxInterface4gamma import IDMParams, EnvState,initial_env_state_pure, rollout_pure,rollout_while
from sklearn.cluster import KMeans
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def format_tensor(tensor, decimals=1):
    """格式化张量到指定小数位数"""
    # 乘以 10^decimals，取整，再除以 10^decimals
    factor = tf.constant(10 ** decimals, dtype=tensor.dtype)
    rounded = tf.round(tensor * factor) / factor
    return rounded

# 1. 日志与参数配置
def setup_logger(log_path, debug=False):
    """配置日志记录器，同时输出到文件和控制台"""
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除旧的处理程序，避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理程序
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))

    # 控制台处理程序
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))

    # 添加处理程序到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    #v0: float
    #T: float
    #s0: float
    #a: float
    #b: float
    #delta: float
    #length: float
    #rtime: float

def get_param_bounds(num_types):
    """根据类别数量生成参数边界"""
    # [v0, T, s0, a, b, rtime]
    base_bounds = [
        (30/3.6, 65/3.6), # v0
        (0.5, 3.0),       # T
        (1.0, 3.0),       # s0
        (1.0, 3.0),       # a
        (1.0, 6.0),       # b
        (0.01, 3.0)       # rtime
    ]

    '''
        
    base_bounds = [
        (40/3.6, 60/3.6), # v0
        (0.5, 2.0),       # T
        (1.0, 3.0),       # s0
        (1.0, 3.0),       # a
        (1.0, 6.0),       # b
        (0.01, 2.0)       # rtime
    ]
    '''
    return np.array([base_bounds for _ in range(num_types)], dtype=np.float32)

# 2. Keras 模型定义 (优化)
def build_simple_resnet(input_dim, output_dim, unit=256, layNum=8):
    """构建一个优化的ResNet模型，调整了BN和ReLU的位置"""
    

    def resnet_block(x, units):
        shortcut = x

        # 第一个 Dense -> BN -> ReLU
        y = Dense(units)(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)

        # 第二个 Dense -> BN
        y = Dense(units)(y)
        y = BatchNormalization()(y)

        # 如果维度不匹配，使用1x1卷积调整shortcut
        if shortcut.shape[-1] != units:
            shortcut = Dense(units)(shortcut)

        # Add & 最后的ReLU
        y = Add()([shortcut, y])
        y = ReLU()(y)
        return y

    inp = Input(shape=(input_dim,))
    # 初始块: Dense -> BN -> ReLU
    x = Dense(unit)(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(layNum):
        x = resnet_block(x, unit)

    # 输出层使用sigmoid激活，将输出限制在(0, 1)范围，便于后续缩放
    out = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    return model



def run_batch_simulation2(nn_output_batch, raw_data_batch, param_bounds, num_types, columns=None,dt=0.5):
    """
    JAX纯函数版本，适用于vmap批量仿真。输入为单个样本，输出主车time_to_vanish。
    nn_output: (num_types*6,)
    raw_data: (features,)
    columns: list[str]
    param_bounds: (num_types, 6, 2)
    num_types: int
    """

    start_time = time.time()
    batch_size = nn_output_batch.shape[0]
    results = []
    statesAll = []

    for i in range(batch_size):
        nn_output = nn_output_batch[i].numpy()
        raw_data = raw_data_batch[i].numpy()
    
        # 1. 参数解码
        scaled_params = jnp.asarray(nn_output).reshape((num_types, 6))
        param_bounds = jnp.asarray(param_bounds)
        low = param_bounds[:, :, 0]
        high = param_bounds[:, :, 1]
        real_params = low + scaled_params * (high - low)
        # 插入常量delta=4.0, length=5.0
        def insert_constants(params):
            return jnp.concatenate([params[:5], jnp.array([4.0, 5.0]), params[5:]])
        idm_params_arr = jnp.stack([insert_constants(real_params[i]) for i in range(num_types)])
        #tf.print(batch_size,i,"IDM Params for types:\n", idm_params_arr)
        # 解析raw_data
        row_dict = {col: raw_data[i] for i, col in enumerate(columns)}
        intersection_pos = row_dict['intersection_pos']
        car_indices = []
        for i in range(20):
            pos_col = f'car_position_{i}'
            if pos_col in row_dict and row_dict[pos_col] != -1 and not jnp.isnan(row_dict[pos_col]):
                car_indices.append((intersection_pos - row_dict[pos_col], i))
            else:
                car_indices.append((intersection_pos - row_dict[pos_col]*random.randint(1000,2000), i))
                # 对于缺失车辆，给它一个负的较远的距离，确保排序靠后，距离红绿灯较远

        car_indices = sorted(car_indices)
        num_cars = 20
        def get_param(idx):
            vtype = idx if idx < num_types else num_types - 1
            return idm_params_arr[vtype]
        params_stack = jnp.stack([get_param(rank) for rank in range(num_cars)])
        v0 = params_stack[:, 0]
        T = params_stack[:, 1]
        s0 = params_stack[:, 2]
        a = params_stack[:, 3]
        b = params_stack[:, 4]
        delta = params_stack[:, 5]
        length = params_stack[:, 6]
        rtime = params_stack[:, 7]
        params = IDMParams(v0=v0, T=T, s0=s0, a=a, b=b, delta=delta, length=length, rtime=rtime)
        init_pos = jnp.array([row_dict[f'car_position_{i}'] for _, i in car_indices])
        init_speed = jnp.array([row_dict[f'car_speed_{i}'] for _, i in car_indices])
        main_car_pos_value = row_dict['main_car_position']
        main_car_rank = -1
        for idx, (_, i) in enumerate(car_indices):
            if jnp.abs(row_dict[f'car_position_{i}'] - main_car_pos_value) < 1e-3:
                main_car_rank = idx #idx为car_indices,init_pos和init_speed中的存储位置
                break

        red_light_pos = float(row_dict['intersection_pos'])
        red_light_duration = float(row_dict['redLightRemainingTime']) / 30.0#注意这里除以30了
        # 环境初始化
   
        state = initial_env_state_pure(num_vehicles=num_cars, dt=dt, init_pos=init_pos, init_vel=init_speed, params=params, red_light_pos=red_light_pos, red_light_duration=red_light_duration)
        statesAll.append((state, main_car_rank, num_cars))

    statesAll = tree_util.tree_map(lambda *xs: jnp.stack(xs), *statesAll)
    
    #使用rollout_pure计算，内部采用scan展开循环,但是发现scan效率不高,随着max_steps增加，效率下降明显
    def get_time_to_vanish1(states): 
        state, main_car_rank, num_cars = states
        traj = rollout_pure(state, num_vehicles=num_cars, dt=dt, max_steps=int(120/dt))
        tf.print("tra[-1].step_count:", traj[-1].step_count)
        return traj[-1].time_to_vanish[main_car_rank]#time_to_vanish计算时已经是秒了(step*0.1)
    
    #使用rollout_while计算，内部采用jax,while展开循环,可以提前终止
    def get_time_to_vanish2(states):
        # 替换原来的 rollout_pure 调用
        state, main_car_rank, num_cars = states
        final_state = rollout_while(state, num_vehicles=num_cars, dt=dt,  max_steps=int(120/dt))
        main_car_vanish_time = final_state.time_to_vanish[main_car_rank]
        #tf.print("final_state.step_count:", final_state.step_count)
        return main_car_vanish_time#time_to_vanish计算时已经是秒了(step*0.1)

    results = jax.vmap(get_time_to_vanish2)(statesAll)
    end_time = time.time()
    #tf.print(f"JAX SIM Batch sim time_to_vanish:\n {results}")
    tf.print(f"JAX SIM Batch simulation time (s): {end_time - start_time:.2f}")
    return tf.convert_to_tensor(results, dtype=tf.float32)
            
    

# -------------------------- 三种方案耗时测试函数 --------------------------
# 4. 主训练函数
def main(args):
    setup_logger(args.log_path, args.debug)
    logging.info(f"启动训练，参数: {args}")

    logging.info(f"从 {args.csv_path} 加载数据...")
    df = pd.read_csv(args.csv_path).dropna()
    
    #--------------------------------------------------------------------------------------------------------
    # 随机提取args.nC个样本，尽可能保证样本多样性
    numSamples = args.nC
    if len(df) > numSamples:
        # 先用KMeans聚类，保证多样性
        sample_features = df[[c for c in df.columns if 'car_position_' in c or 'car_speed_' in c]].values
        n_clusters = min(100, len(df) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(sample_features)
        sampled_indices = []
        for cluster in range(n_clusters):
            cluster_idx = np.where(cluster_labels == cluster)[0]
            if len(cluster_idx) > 0:
                # 每个簇随机采样一定数量
                n = max(1, int(numSamples / n_clusters))
                chosen = np.random.choice(cluster_idx, size=min(n, len(cluster_idx)), replace=False)
                sampled_indices.extend(chosen)
        # 如果不足1000个，再随机补齐
        if len(sampled_indices) < numSamples:
            remaining = list(set(range(len(df))) - set(sampled_indices))
            extra = np.random.choice(remaining, size=1000 - len(sampled_indices), replace=False)
            sampled_indices.extend(extra)
        sampled_indices = np.array(sampled_indices[:1000])
        df = df.iloc[sampled_indices].reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    #--------------------------------------------------------------------------------------------------------
    # 添加intersection_pos列
    lane_pos_map = {5: 53.05, 6: 53.13, 7: 53.30}
    df['intersection_pos'] = df['lane'].map(lane_pos_map)
    

    dt = args.dt
    logging.info(f"使用时间步长 dt={dt} 进行仿真")
  
    # --- 修改开始 ---
    # 明确区分特征列和传递给仿真的原始数据列
    feature_cols = [c for c in df.columns if ('car_position_' in c or 'car_speed_' in c or 'redLight' in c)]
    
    # 为仿真准备的列，确保没有重复
    # 我们需要所有特征列，以及一些额外的信息
    raw_cols_set = set(feature_cols)
    raw_cols_set.add('lane')
    raw_cols_set.add('intersection_pos')
    # 将主车位置重命名以避免冲突
    df.rename(columns={'car_position': 'main_car_position'}, inplace=True)
    raw_cols_set.add('main_car_position')
    
    raw_cols = sorted(list(raw_cols_set)) # 排序以保证列顺序一致

    X = df[feature_cols].values.astype(np.float32)
    y = (df['time_to_vanish'].values / 30.0).astype(np.float32)#注意已经除以30了
    raw_data_for_sim = df[raw_cols].values.astype(np.float32)
    # --- 修改结束 ---

    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
        X, y, raw_data_for_sim, test_size=args.test_size, random_state=42
    )


    # 创建 tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, raw_train))
    #train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, raw_val))
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    logging.info(f"数据准备完成: {len(X_train)} 训练样本, {len(X_val)} 验证样本")

    num_types = args.num_types
    output_dim = num_types * 6
    model = build_simple_resnet(X_train.shape[1], output_dim, args.unit, args.layNum)

    # 使用学习率调度器，开始时学习率较高，然后逐渐降低
    initial_learning_rate = args.lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,  # 每100步衰减
        decay_rate=0.99,  # 每次衰减到95%
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule)
   
    param_bounds = get_param_bounds(num_types)
  
    #raw_columns_list = tf.constant(raw_cols) # 将列名作为常量传递
    raw_columns_list = raw_cols
    logging.info(f"模型创建完成: {num_types} 个车辆类别, ResNet输出维度 {output_dim}")

    @tf.function
    def train_step(x_batch, y_batch, raw_batch):
        
        with tf.GradientTape() as tape:
            # ResNet forward pass
            nn_output = model(x_batch, training=True)
            
            # JAX simulation as a black-box function
            predicted_times = tf.py_function(
                func=lambda nn_out, raw: run_batch_simulation2(
                    nn_out, raw, param_bounds, num_types, columns=raw_columns_list,dt=dt
                ),
                inp=[nn_output, raw_batch],
                Tout=tf.float32
            )
            
            # 显式设置形状信息，帮助 TensorFlow 推断
            # predicted_times 来自JAX函数，需要明确设置形状
            batch_size = tf.shape(y_batch)[0]  # 获取批次大小
            predicted_times = tf.reshape(predicted_times, [batch_size])  # Reshape to [batch_size]
            y_batch = tf.reshape(y_batch, [batch_size])  # Reshape to [batch_size]

            # Loss computation with MSE + MAE for better convergence
            # Loss computation with MSE for stable gradients
            loss = tf.reduce_mean(tf.square(predicted_times - y_batch))
        
        # Backward pass and parameter update with gradient clipping
        grads = tape.gradient(loss, model.trainable_variables)
        # Apply gradient clipping to prevent gradient explosion
        clipped_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in grads]
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
        
        #tf.print("\n==================== Batch Training Info output:vanish Time ")
        #tf.print("Loss:", loss, summarize=-1)
        # predicted_times and y_batch are already 1D, so stack them directly
        #combined_tensor = tf.stack([predicted_times, y_batch], axis=0)  # Shape: [2, batch_size]
        #tf.print("Predicted & Actual times [Pred; Act]:\n", format_tensor(combined_tensor), summarize=-1)
        #tf.print("============================================================")
      
        return loss

    @tf.function
    def val_step(x_batch, y_batch, raw_batch):
        # ResNet forward pass
        nn_output = model(x_batch, training=False)

        # JAX simulation as a black-box function
        predicted_times = tf.py_function(
            func=lambda nn_out, raw: run_batch_simulation2(
                nn_out, raw, param_bounds, num_types, columns=raw_columns_list,dt=dt
            ),
            inp=[nn_output, raw_batch],
            Tout=tf.float32
        )

        # 显式设置形状信息，帮助 TensorFlow 推断
        # predicted_times 来自JAX函数，需要明确设置形状
        batch_size = tf.shape(y_batch)[0]  # 获取批次大小
        predicted_times = tf.reshape(predicted_times, [batch_size])  # Reshape to [batch_size]
        y_batch = tf.reshape(y_batch, [batch_size])  # Reshape to [batch_size]

        # Compute validation errors
        errors = predicted_times - y_batch

        return errors

    for epoch in range(args.epochs):
        logging.info(f"===== Epoch {epoch + 1}/{args.epochs} =====")
        
        epoch_loss_avg = tf.keras.metrics.Mean()

        # 统计batch总数
        total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        batch_idx = 0
        for x_batch, y_batch, raw_batch in train_dataset:
            batch_idx += 1
            batch_start_time = time.time()
            loss = train_step(x_batch, y_batch, raw_batch)
            batch_end_time = time.time()
            epoch_loss_avg.update_state(loss)
            # Logging outside tf.function, where .numpy() is available
            remaining_batches = total_batches - batch_idx
            logging.info(f"Batch {batch_idx}/{total_batches}, Loss: {loss.numpy():.4f}, \
                         Time: {batch_end_time - batch_start_time:.3f}s, Remaining: {remaining_batches}")


        logging.info(f"Epoch {epoch+1} 训练完成, 平均损失: {epoch_loss_avg.result().numpy():.4f}")

        # 每20个epoch进行一次验证
        if epoch % 20 == 0 or epoch == args.epochs - 1:  # 在第0, 20, 40,...个epoch以及最后一个epoch验证
            val_loss_avg = tf.keras.metrics.Mean()
            total_val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
            val_batch_idx = 0
            all_val_errors = []
            for x_batch, y_batch, raw_batch in val_dataset:
                val_batch_idx += 1
                val_batch_start_time = time.time()
                errors = val_step(x_batch, y_batch, raw_batch)
                errors_np = errors.numpy()
                all_val_errors.append(errors_np)
                val_batch_end_time = time.time()
                # 计算当前batch的损失
                val_loss = np.mean(np.square(errors_np))
                val_loss_avg.update_state(val_loss)
                remaining_val_batches = total_val_batches - val_batch_idx
                logging.info(f"Val Batch {val_batch_idx}/{total_val_batches}, Loss: {val_loss:.4f}, \
                             Time: {val_batch_end_time - val_batch_start_time:.3f}s, Remaining: {remaining_val_batches}")

            val_errors = np.concatenate([arr.flatten() for arr in all_val_errors])
            logging.info(f"Epoch {epoch+1} 验证完成 - 误差均值: {np.mean(val_errors):.4f}, RMSE: {np.sqrt(np.mean(np.square(val_errors))):.4f}")

            model_path = f"./tmpModes/model_epoch_{epoch+1}.h5"
            model.save(model_path)
            # Save validation errors
            np.save(f"./tmpModes/validation_errors_epoch_{epoch+1}.npy", val_errors)
            logging.info(f"模型已保存至 {model_path}")
        else:
            # 非验证epoch也保存模型，但不进行验证
            model_path = f"./tmpModes/model_epoch_{epoch+1}.h5"
            model.save(model_path)
            logging.info(f"模型已保存至 {model_path}")

    logging.info("训练全部完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Keras和交通仿真进行端到端模型训练")
    parser.add_argument('--csv_path', type=str, default='trainsamples_lane_5_6_7.csv', help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.9, help='验证集比例')
    parser.add_argument('--num_types', type=int, default=4, help='车辆类别数')
    parser.add_argument('--unit', type=int, default=128, help='ResNet隐藏层单元数')
    parser.add_argument('--layNum', type=int, default=8, help='ResNet块数量')
    parser.add_argument('--log_path', type=str, default='training_log.log', help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用Debug级别的日志信息')
    parser.add_argument('--dt', type=float, default=0.5, help='仿真时间步长')
    parser.add_argument('--nC', type=float, default=100, help='Kmeans聚类数量，用于样本多样性选择')
    args = parser.parse_args()
    main(args)

#python modelsCollect4.py --batch_size 16 --layNum 4
#python modelsCollect4.py --batch_size 32 --test_size 0.5 --epochs 100 --lr 0.005 --unit 256 --layNum 8 --dt 0.5







'''
wget', 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
'''