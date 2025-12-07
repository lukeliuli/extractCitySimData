因为仿真物理模型无法可微，导致模型训练过程出现大问题，所以这个模型到此为止
'''

参考modelsCollect3.py,modelsCollect.py,用keras实现
1.将车辆分为4类，分别为排队第一辆车（头车），排队第二辆车，排队第三辆车，其他车辆
2.每类车辆使用都使用idm模型，但是参数不同（可变参数类型和范围参考modelCollect3）
3.基于modelsCollect2中的simpleResnet模型，基于输入数据，预测4类车辆参数
4.训练数据集生成方式参考modelsCollect.py
5.模型训练是先用simple Resnet预测4类车辆的idm参数,然后根据4类车辆的idm参数，仿真模型预测车辆time_to vanish时间。训练目标是最小化time_to vanish预测误差
6.训练不是用simple Resnet的预测time_t0_vanish，而是用仿真预测的time_to vanish时间和真实time_to vanish时间做mse损失
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
from concurrent.futures import ProcessPoolExecutor

# 确保可以导入 pyGameInterface3
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pyGameInterface3 import TrafficSimulator, VehicleParams

# 1. 日志与参数配置
def setup_logger(log_path, debug=False):
    """配置日志记录器，同时输出到文件和控制台"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

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
    main_car_pos = row['main_car_position']
    vehicles_to_add, car_indices, main_car_id = [], [], -1

    for i in range(20):
        pos_col = f'car_position_{i}'
        if pos_col in row and row[pos_col] != -1 and not pd.isna(row[pos_col]):
            car_indices.append((intersection_pos - row[pos_col], i))
    car_indices.sort()

    idm_params_per_car = {}
    for rank, (_, i) in enumerate(car_indices):
        vtype = rank if rank < num_types_np else num_types_np - 1
        vid = 100 + i
        current_car_pos = row[f'car_position_{i}']
        idm_params_per_car[vid] = VehicleParams(*idm_params_dict[vtype])
        vehicles_to_add.append({'id': vid, 'distance': current_car_pos, 'speed': row[f'car_speed_{i}']})
        if abs(current_car_pos - main_car_pos) < 0.1:
            main_car_id = vid
    
    if not vehicles_to_add: return np.float32(120.0)

    simulator = TrafficSimulator(default_params=list(idm_params_per_car.values())[0], time_step=0.1, intersection_pos=intersection_pos)
    simulator.add_vehicles(vehicles_to_add)
    simulator.batch_set_vehicle_params(idm_params_per_car)
    simulator.set_red_light(row['redLightRemainingTime'] / 30.0)
    df = simulator.run_simulation(max_duration=120)
    
    if main_car_id != -1 and not df.empty:
        passed = df[(df['id'] == main_car_id) & (df['has_passed'] == True)]
        if not passed.empty:
            return np.float32(passed.iloc[0]['time'])
    return np.float32(120.0)

def run_batch_simulation(nn_output_batch, raw_data_batch, param_bounds, num_types, columns=None):
    """串行运行批处理仿真"""
    if columns is None:
        raise ValueError("`columns` list was not provided to run_batch_simulation.")

    batch_size = nn_output_batch.shape[0]
    results = []
    
    for i in range(batch_size):
        # --- 修改开始 ---
        # 将 Tensor 转换为 Numpy 数组，而 columns 已经是 Python 对象
        args_tuple = (
            nn_output_batch[i].numpy(), 
            raw_data_batch[i].numpy(), 
            columns,  # 直接传递 Python 列表
            param_bounds.numpy(), 
            num_types.numpy()
        )
        # --- 修改结束 ---
        result = run_single_simulation(args_tuple)
        results.append(result)
 
    return np.array(results, dtype=np.float32)

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
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, raw_val))
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

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
            nn_output = model(x_batch, training=True)
            predicted_times = simulation_layer_batch(nn_output, raw_batch,  param_bounds, num_types,columns_list=raw_columns_list)
            loss = tf.reduce_mean(tf.square(predicted_times - y_batch))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x_batch, y_batch, raw_batch):
        nn_output = model(x_batch, training=False)
        predicted_times = simulation_layer_batch(nn_output, raw_batch,  param_bounds, num_types,columns_list=raw_columns_list)
        errors = predicted_times - y_batch
        return errors

    for epoch in range(args.epochs):
        logging.info(f"===== Epoch {epoch + 1}/{args.epochs} =====")
        
        epoch_loss_avg = tf.keras.metrics.Mean()
        pbar = tqdm(train_dataset, desc=f"训练 Epoch {epoch+1}")
        for x_batch, y_batch, raw_batch in pbar:
            loss = train_step(x_batch, y_batch, raw_batch)
            epoch_loss_avg.update_state(loss)
            pbar.set_postfix(loss=f"{epoch_loss_avg.result().numpy():.4f}")

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
    # 确保进程池被关闭
    global process_pool
    if process_pool:
        process_pool.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Keras和交通仿真进行端到端模型训练")
    parser.add_argument('--csv_path', type=str, default='trainsamples_lane_5_6_7.csv', help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--num_types', type=int, default=4, help='车辆类别数')
    parser.add_argument('--unit', type=int, default=256, help='ResNet隐藏层单元数')
    parser.add_argument('--layNum', type=int, default=8, help='ResNet块数量')
    parser.add_argument('--log_path', type=str, default='training_log.log', help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用Debug级别的日志信息')
    
    args = parser.parse_args()
    main(args)









