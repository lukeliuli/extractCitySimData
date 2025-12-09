'''

参考modelsCollect3.py,modelsCollect.py,用keras实现
1.将车辆分为4类，分别为排队第一辆车（头车），排队第二辆车，排队第三辆车，其他车辆
2.每类车辆使用都使用idm模型，但是参数不同（可变参数类型和范围参考modelCollect3）
3.基于modelsCollect2中的simpleResnet模型，基于输入数据，预测4类车辆参数
4.训练数据集生成方式参考modelsCollect.py
5.模型训练是先用simple Resnet预测4类车辆的idm参数,然后根据4类车辆的idm参数，仿真模型预测车辆time_to vanish时间。训练目标是最小化time_to vanish预测误差
6.训练不是用simple Resnet的预测time_t0_vanish，而是用仿真预测的time_to vanish时间和真实time_to_vanish时间做mse损失
7.注意保存仿真中间过程数据结果
8.设置选项，是否启用print调试信息
9.保存训练好的模型
10.代码结构清晰，便于后续维护和扩展.例如将车辆分为5,6类等，增加需要训练的参数
11.增加命令行参数，方便配置训练选项，例如是否启用print调试信息，训练轮数，学习率等
12.增加日志记录功能，记录训练过程中的重要信息，例如损失值变化，模型保存路径等
13.代码结果简单，不需要try except结构 ,不需要进行太多的错误处理
14.输入数据就是 csv_path = 'trainsamples_lane_5_6_7.csv' 
15.注意根据laneid,给出每条车道的intersection_pos
16.使用tqdm显示训练进度
17.注意仿真时，按照距离终点排序，将最近的3辆车分为1~3类，其余为4类，并为每辆车分配对应的IDM参数。
18.注意参考pyGameInterface3中的TrafficSimulator和VehicleParams类，仿真模型预测车辆time_to vanish时间
19.注意车辆的参数的变换范围参考modelsCollect3.py中的设置
'''

'''
1.重新优化所有函数，keras，tf GradientTape等只关注本身resnet模型的前向和反向传播，不涉及仿真细节
2.jax部分仅在仿真函数中使用，且仿真函数不参与梯度计算,jax仅作为一个黑盒仿真器使用，jax仅仅接受TF的输出参数做仿真，内部细节不涉及梯度计算
3. 整个链路为:keras模型前向传播->仿真函数（jax黑盒）->损失计算(仿真模型输出time_to_vanish与实际time_to_vanish)->keras模型参数更新(jax不参与梯度计算)

'''
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pyGameBraxInterface4 import IDMParams, BraxIDMEnv,EnvState


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
        (40/3.6, 60/3.6), # v0
        (0.5, 2.0),       # T
        (1.0, 3.0),       # s0
        (1.0, 3.0),       # a
        (1.0, 6.0),       # b
        (0.01, 1.0)       # rtime
    ]
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

# 3. 仿真函数 (单个样本)
def run_single_simulation(args_tuple):
    """为并行化设计的独立仿真函数"""
    # --- 修改开始 ---
    # 直接解包，columns 不再是 tensor
    nn_output_np, row_data_np, columns, param_bounds_np, num_types_np = args_tuple
    # --- 修改结束 ---
 
    row = pd.Series(row_data_np, index=columns)
   
    scaled_params = nn_output_np.reshape(num_types_np, 6)
    low, high = param_bounds_np[:, :, 0], param_bounds_np[:, :, 1]
    real_params = low + scaled_params * (high - low)

    idm_params_dict = {}
    for i in range(num_types_np):
        # real_params[i] is [v0, T, s0, a, b, rtime]
        # 我们需要 [v0, T, s0, a, b, delta, length, rtime]
        # 将常量 delta=4.0 和 length=5.0 插入
        params_with_constants = np.insert(real_params[i], 5, [4.0, 5.0]) 
        idm_params_dict[i] = params_with_constants
      
    intersection_pos = row['intersection_pos']
    car_indices = []

    #安装距离终点排序车辆
    for i in range(20):
        pos_col = f'car_position_{i}'
        if pos_col in row and row[pos_col] != -1 and not pd.isna(row[pos_col]):
            car_indices.append((intersection_pos - row[pos_col], i))
    car_indices.sort()
    
    # 根据车辆类别分配对应的IDM参数，组装为BraxIDMEnv需要的params
    num_cars = len(car_indices)
    params_dict = {k: [] for k in ['v0', 'T', 's0', 'a', 'b', 'delta', 'length', 'rtime']}
    for rank, (_, i) in enumerate(car_indices):
        vtype = rank if rank < num_types_np else num_types_np - 1
        car_params = idm_params_dict[vtype]
        params_dict['v0'].append(car_params[0])
        params_dict['T'].append(car_params[1])
        params_dict['s0'].append(car_params[2])
        params_dict['a'].append(car_params[3])
        params_dict['b'].append(car_params[4])
        params_dict['delta'].append(car_params[5])
        params_dict['length'].append(car_params[6])
        params_dict['rtime'].append(car_params[7])

    # 转换为jnp.array
    params = IDMParams(
        v0=jnp.array(params_dict['v0']),
        T=jnp.array(params_dict['T']),
        s0=jnp.array(params_dict['s0']),
        a=jnp.array(params_dict['a']),
        b=jnp.array(params_dict['b']),
        delta=jnp.array(params_dict['delta']),
        length=jnp.array(params_dict['length']),
        rtime=jnp.array(params_dict['rtime'])
    )
    
    # 找到主车在 car_indices 中的索引
    main_car_pos_value = row['main_car_position']
    main_car_rank = -1
    for idx, (_, i) in enumerate(car_indices):
        if abs(row[f'car_position_{i}'] - main_car_pos_value) < 1e-3:
            main_car_rank = idx
            break
    # main_car_rank 即主车在排序后的位置（距离终点最近的为0），如果未找到则为-1

    
    env = BraxIDMEnv(num_vehicles=len(car_indices), dt=0.1, red_light_pos=intersection_pos, red_light_duration=row['redLightRemainingTime']/30)
    init_pos = jnp.array([row[f'car_position_{i}'] for _, i in car_indices])
    init_speed = jnp.array([row[f'car_speed_{i}'] for _, i in car_indices]) 
    state = env.reset(jax.random.PRNGKey(0), init_pos, init_speed, params)
    traj = env.rollout(state, max_steps=1500, idm_log_csv="idm_step_log.csv")

    if main_car_rank == -1:
        # 主车未找到，返回一个较大的时间作为惩罚
        return np.float32(120.0)
    else:
        # 从traj.info['vanish_times']中获取主车的vanish_time
    # 从traj中获得各个时刻的envState，从最后一个时刻获得主车的time_to vanish时间
        env_states = traj  # traj是EnvState的列表
        # env_states 是 EnvState 的列表，最后一个元素包含最终的 time_to_vanish
        last_state = env_states[-1]
        vanish_times = last_state.time_to_vanish
        main_car_vanish_time = vanish_times[main_car_rank]
       
    for i, t in enumerate(traj[-1].time_to_vanish):
        print(f"car{i}: {float(t):.2f}")
   
        return np.float32(main_car_vanish_time)


def run_batch_simulation(nn_output_batch, raw_data_batch, param_bounds, num_types, columns=None):
    """串行运行批处理仿真"""
    if columns is None:
        raise ValueError("`columns` list was not provided to run_batch_simulation.")

    batch_size = nn_output_batch.shape[0]
    results = []

    for i in range(batch_size):
        # 将 Tensor 转换为 Numpy 数组，而 columns 已经是 Python 对象
        args_tuple = (
            nn_output_batch[i].numpy(), 
            raw_data_batch[i].numpy(), 
            columns,  # 直接传递 Python 列表
            param_bounds,  # 直接使用 numpy 数组
            num_types  # 直接使用整数值
        )
        result = run_single_simulation(args_tuple)
        results.append(result)

    results = np.array(results, dtype=np.float32)
    return tf.convert_to_tensor(results, dtype=tf.float32)
  

def simulation_layer_batch(nn_output, raw_data, param_bounds, num_types,columns_list):
    """批处理仿真层，包裹py_function"""
   
    func = lambda nn_out, r_data, p_bounds, n_types: run_batch_simulation(
        nn_out, r_data, p_bounds, n_types, columns=columns_list
    )

    predicted_times = tf.py_function(
        func=func,
        # 从 inp 中移除 'columns'
        inp=[nn_output, raw_data, param_bounds, num_types],
        Tout=tf.float32
    )
    # 确保输出有正确的形状

    predicted_times.set_shape([None])
    return predicted_times

# 4. 主训练函数
def main(args):
    setup_logger(args.log_path, args.debug)
    logging.info(f"启动训练，参数: {args}")

    logging.info(f"从 {args.csv_path} 加载数据...")
    df = pd.read_csv(args.csv_path).dropna()
    
    lane_pos_map = {5: 53.05, 6: 53.13, 7: 53.30}
    df['intersection_pos'] = df['lane'].map(lane_pos_map)
  
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
    y = (df['time_to_vanish'].values / 30.0).astype(np.float32)
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
    optimizer = Adam(learning_rate=args.lr)
   
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
                func=lambda nn_out, raw: run_batch_simulation(
                    nn_out, raw, param_bounds, num_types, columns=raw_columns_list
                ),
                inp=[nn_output, raw_batch],
                Tout=tf.float32
            )

            # Loss computation
            loss = tf.reduce_mean(tf.square(predicted_times - y_batch))
        
        # Backward pass and parameter update
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Use tf.print for debugging inside tf.function
        
        tf.print("Loss:", loss)
        tf.print("Predicted times:", predicted_times)
        tf.print("Actual times:", y_batch)
        
        return loss

    @tf.function
    def val_step(x_batch, y_batch, raw_batch):
        # ResNet forward pass
        nn_output = model(x_batch, training=False)

        # JAX simulation as a black-box function
        predicted_times = tf.py_function(
            func=lambda nn_out, raw: run_batch_simulation(
                nn_out, raw, param_bounds, num_types, columns=raw_columns_list
            ),
            inp=[nn_output, raw_batch],
            Tout=tf.float32
        )

        # Compute validation errors
        errors = predicted_times - y_batch
        return errors

    for epoch in range(args.epochs):
        logging.info(f"===== Epoch {epoch + 1}/{args.epochs} =====")
        
        epoch_loss_avg = tf.keras.metrics.Mean()

        # 不使用tqdm时，直接遍历数据集
        for x_batch, y_batch, raw_batch in train_dataset:
            loss = train_step(x_batch, y_batch, raw_batch)
            epoch_loss_avg.update_state(loss)
            # Logging outside tf.function, where .numpy() is available
            logging.info(f"Loss: {loss.numpy():.4f}")
            epoch_loss_avg.update_state(loss)

      

        logging.info(f"Epoch {epoch+1} 训练完成, 平均损失: {epoch_loss_avg.result().numpy():.4f}")

        all_val_errors = []
        for x_batch, y_batch, raw_batch in tqdm(val_dataset, desc=f"验证 Epoch {epoch+1}"):
            errors = val_step(x_batch, y_batch, raw_batch)
            all_val_errors.append(errors.numpy())
        
        val_errors = np.concatenate(all_val_errors)
        logging.info(f"Epoch {epoch+1} 验证完成 - 误差均值: {np.mean(val_errors):.4f}, RMSE: {np.sqrt(np.mean(np.square(val_errors))):.4f}")
        
        model_path = f"model_epoch_{epoch+1}.h5"
        model.save(model_path)
        np.save(f"validation_errors_epoch_{epoch+1}.npy", val_errors)
        logging.info(f"模型已保存至 {model_path}")

    logging.info("训练全部完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Keras和交通仿真进行端到端模型训练")
    parser.add_argument('--csv_path', type=str, default='trainsamples_lane_5_6_7.csv', help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1,help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.01, help='验证集比例')
    parser.add_argument('--num_types', type=int, default=4, help='车辆类别数')
    parser.add_argument('--unit', type=int, default=256, help='ResNet隐藏层单元数')
    parser.add_argument('--layNum', type=int, default=8, help='ResNet块数量')
    parser.add_argument('--log_path', type=str, default='training_log.log', help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用Debug级别的日志信息')
    
    args = parser.parse_args()
    main(args)









