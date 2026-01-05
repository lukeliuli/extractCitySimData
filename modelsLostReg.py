'''

参考modelsCollect4.py,modelLostReg做如下改进:

1.识别模型改进：ResNet模型中，加入车辆丢失识别
'''
    #-----------------------------------------------------------------------------
    #数据库df,计算目标车的前面车辆的低速和静止状态下车辆与前车的距离以及头车与停止线的距离，并用car_position_i进行标记
    #基于softmax计算lost vehicle的概率
    #最终预测结果
    # 1.在ResNet的基础上，增加一个分支用于预测lost vehicle的概率
    # 2.或者基于简单回归分析进行lost vehicle的概率预测
    # 3.或者基于kmeans聚类分析进行lost vehicle的概率预测




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
import matplotlib.pyplot as plt

import time
import os
import random
from sklearn.cluster import KMeans
from tensorflow.keras.utils import Sequence
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



# 2. Keras 模型定义 (优化)
def build_simple_resnet(input_dim, output_dim, unit=256, layNum=8):
    """构建一个优化的ResNet模型，调整了BN和ReLU的位置
    对于二分类任务（是否丢失车辆），输出维度应为1，使用sigmoid激活函数
    """


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


    # 对于二分类任务，输出维度为1，使用sigmoid激活函数
    # 输出值在(0, 1)范围内，表示属于正类（车辆丢失）的概率
    # Dense(1, activation='sigmoid') 表示：创建一个全连接层，有1个神经元，使用sigmoid激活函数
    # sigmoid函数将输出压缩到0-1之间，可以解释为概率值
    out = Dense(1, activation='sigmoid')(x)  # 对于二分类，始终输出1个值，表示丢失概率
    model = Model(inputs=inp, outputs=out)
    return model


            
def genSamplesByRandomRemovingVehcile(df, remove_ratio=0.1):
    """
    根据上下文，读入csv文件，获得当前样本的下排队车辆有那些以及相应的位置和序号
    给出当前目标车对于的排队车辆的序号i
    同时实现数据增强：随机去掉一些车辆位置和速度数据，模拟传感器丢失数据的情况
    """
   
   
    remove_rows = []
    # 获取所有车辆位置列和速度列
    car_pos_cols = [col for col in df.columns if col.startswith('car_position_')]
    car_speed_cols = [col for col in df.columns if col.startswith('car_speed_')]

    # 按照位置排序，确定排队车辆的顺序
    results = []

    for idx, row in df.iterrows():
        # 获取当前样本的所有车辆位置
        positions = []
        for pos_col in car_pos_cols:
            pos_val = row[pos_col]
            if pos_val != -1 and not pd.isna(pos_val):  # 有效位置值
                # 提取车辆索引（从列名中获取）
                car_posi = int(pos_col.replace('car_position_', ''))
                positions.append((pos_val,car_posi))

        # 按位置排序（距离交叉口近的在前）
        positions.sort(key=lambda x: x[0],reverse=True)

        # 获取主车位置（目标车）
        main_car_pos = row['main_car_position'] if 'main_car_position' in row else row['car_position']

        # 找到主车在排队中的索引
        main_car_posque_idx = -1
        main_car_posi = -1
        for queue_idx, (pos, car_posi) in enumerate(positions):
            if abs(pos - main_car_pos) < 1e-3:  # 使用小的容差值比较浮点数
                mmain_car_posque_idx = queue_idx
                main_car_posi = car_posi
                break
      
        # 实现数据增强：随机移除一些车辆数据
        if remove_ratio > 0:
            # 计算要移除的车辆数量
            num_vehicles = len(positions)

            if num_vehicles <=2:
                #print('!',end='')
                
                continue
           
            num_to_remove = max(1, int(num_vehicles * remove_ratio))  # 至少移除1辆车

            # 随机选择要移除的车辆（但不能移除主车）
            available_indices = [i for i in range(len(positions)) if i != main_car_posque_idx]
            if len(available_indices) >= num_to_remove:
                indices_to_remove = random.sample(available_indices, num_to_remove)

                # 按降序排序以避免索引变化问题
                indices_to_remove.sort(reverse=True)

                # 移除选中的车辆
                for idx_to_remove in indices_to_remove:
                    if idx_to_remove < len(positions):
                        positions.pop(idx_to_remove)

                # 重新计算主车在新队列中的索引
                main_car_pos = row['main_car_position'] if 'main_car_position' in row else row['car_position']
                main_car_posque_idx  = -1
                for queue_idx, (pos, car_posi) in enumerate(positions):
                    if abs(pos - main_car_pos) < 1e-3:
                        main_car_posque_idx = queue_idx
                        main_car_posi = car_posi
                        break
                #print("After removal, main_car_posque_idx:", main_car_posque_idx)  
                
                # 2. indices_to_remove包含被移除车辆在原始positions中的索引,根据indices_to_remove，修改df的当前row,删除对应车辆的位置和速度数据
                for idx_to_remove in indices_to_remove:
                    if idx_to_remove < len(car_pos_cols):
                        car_pos_col = car_pos_cols[idx_to_remove]
                        car_speed_col = car_speed_cols[idx_to_remove]
                        row[car_pos_col] = -1
                        row[car_speed_col] = -1
                remove_rows.append(row)
                


                # 构建排队车辆信息
                queued_vehicles = {
                    'sample_idx': idx,
                    'main_car_position': main_car_pos,
                    'main_car_queue_idx': main_car_posque_idx,
                    'main_car_name_pos_i':  main_car_posi ,
                    'queued_vehicles': positions,  # [(position, original_index), ...]
                    'num_queued_vehicles': len(positions),
                    'removed_vehicles_count': len([col for col in car_pos_cols if row[col] != -1 and not pd.isna(row[col])]) - len(positions)
                }
                #print(queued_vehicles)
                results.append(queued_vehicles)

    df_missveh = pd.DataFrame(remove_rows, columns=df.columns)   
    return df_missveh,results

#--------------------------------------------------------------------------------------------------------
# 数据准备函数
def genDatasetVanishTime(df,test_size,batch_size):

    # 添加intersection_pos列
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
    
    raw_cols_set.add('main_car_position')
    
    raw_cols = sorted(list(raw_cols_set)) # 排序以保证列顺序一致

    X = df[feature_cols].values.astype(np.float32)
    y = (df['time_to_vanish'].values / 30.0).astype(np.float32)#注意已经除以30了
    raw_data_for_sim = df[raw_cols].values.astype(np.float32)
    # --- 修改结束 ---

    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
        X, y, raw_data_for_sim, test_size=test_size, random_state=42
    )


    # 创建 tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, raw_train))
    #train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, raw_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return X_train, X_val, y_train, y_val, raw_train, raw_val, train_dataset, val_dataset,raw_cols

#根据丢失车辆的情况，生成数据集。参考genDatasetVanishTime
#输出Y,为df['lost']为是否丢失车辆的标签0或者1
def genDatasetLost(df,test_size,batch_size):
    # 添加intersection_pos列
    lane_pos_map = {5: 53.05, 6: 53.13, 7: 53.30}
    df['intersection_pos'] = df['lane'].map(lane_pos_map)

    # 明确区分特征列和传递给仿真的原始数据列
    feature_cols = [c for c in df.columns if ('car_position_' in c or 'car_speed_' in c or 'redLight' in c)]

    # 为仿真准备的列，确保没有重复
    raw_cols_set = set(feature_cols)
    raw_cols_set.add('lane')
    raw_cols_set.add('intersection_pos')
    raw_cols_set.add('main_car_position')

    raw_cols = sorted(list(raw_cols_set)) # 排序以保证列顺序一致

    X = df[feature_cols].values.astype(np.float32)
    y = df['lost'].values.astype(np.float32)  # y is the 'lost' label (0 or 1)
    raw_data_for_sim = df[raw_cols].values.astype(np.float32)

    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
        X, y, raw_data_for_sim, test_size=test_size, random_state=42
    )

    # 创建 tf.data.Dataset - 只包含输入X和目标y，raw_data用于其他用途
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return X_train, X_val, y_train, y_val, raw_train, raw_val, train_dataset, val_dataset, raw_cols
    
    


# Generate dataset for both classification (lost flag) and regression (vanish time)
def genDatasetLostFlagVanishTime(df, test_size, batch_size):
    # Add intersection_pos column
    lane_pos_map = {5: 53.05, 6: 53.13, 7: 53.30}
    df['intersection_pos'] = df['lane'].map(lane_pos_map)

    # Identify feature columns
    feature_cols = [c for c in df.columns if ('car_position_' in c or 'car_speed_' in c or 'redLight' in c)]
    
    # Prepare raw data columns
    raw_cols_set = set(feature_cols)
    raw_cols_set.add('lane')
    raw_cols_set.add('intersection_pos')
    raw_cols_set.add('main_car_position')
    
    raw_cols = sorted(list(raw_cols_set))  # Sort to ensure consistent column order

    X = df[feature_cols].values.astype(np.float32)
    
    # Two targets: lost flag (classification) and vanish time (regression)
    y_lost = df['lost'].values.astype(np.float32)  # Binary classification target
    y_vanish = (df['time_to_vanish'].values / 30.0).astype(np.float32)  # Regression target, normalized
    
    raw_data_for_sim = df[raw_cols].values.astype(np.float32)

    # Split the data
    X_train, X_val, y_lost_train, y_lost_val, y_vanish_train, y_vanish_val, raw_train, raw_val = train_test_split(
        X, y_lost, y_vanish, raw_data_for_sim, test_size=test_size, random_state=42
    )

    # Create tf.data.Dataset with both targets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_lost_train, y_vanish_train)))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, (y_lost_val, y_vanish_val)))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return X_train, X_val, y_lost_train, y_lost_val, y_vanish_train, y_vanish_val, raw_train, raw_val, train_dataset, val_dataset, raw_cols

# ... existing code ...

def build_multi_task_resnet(input_dim, unit=256, layNum=8):
    """构建一个多任务ResNet模型，同时预测车辆丢失概率和消失时间
    """
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

    # 分支1: 预测车辆丢失概率 (二分类任务)
    lost_branch = Dense(1, activation='sigmoid', name='lost_prob')(x)  # 输出0-1之间的丢失概率
    
    # 分支2: 预测消失时间 (回归任务)
    vanish_branch = Dense(1, activation='linear', name='vanish_time')(x)  # 线性激活，输出时间值

    model = Model(inputs=inp, outputs=[lost_branch, vanish_branch])
    return model

# ... existing code ...
# ... existing code ...
# 4. 主训练函数
def main(args):

    setup_logger(args.log_path, args.debug)
    logging.info(f"启动训练，参数: {args}")
    logging.info(f"从 {args.csv_path} 加载数据...")
    df1 = pd.read_csv(args.csv_path).dropna()
    df1['lost'] = 0
    df1.rename(columns={'car_position': 'main_car_position'}, inplace=True)
    df1.rename(columns={'car_speed': 'main_car_speed'}, inplace=True)
    X_train1, X_val1, y_train1, y_val1, \
        raw_train1, raw_val1, train_dataset1, val_dataset1,\
            raw_cols1     = genDatasetLost(df1, args.test_size, args.batch_size)
            
            
    
    print(f"{'-'*100}")
    df_missveh,queued_info = genSamplesByRandomRemovingVehcile(df1, remove_ratio=0.1)
    print('len(queued_vehicles_removed):',len(queued_info)) 
    print(f"{'-'*100}")
   
    
    df_missveh['lost'] = 1
    X_train2, X_val2, y_train2, y_val2, \
        raw_train2, raw_val2, train_dataset2, val_dataset2, \
            raw_cols2 = genDatasetLost(df_missveh, args.test_size, args.batch_size  )


    df3 = pd.concat([df1, df_missveh]).drop_duplicates()

    X_train3, X_val3, y_train3, y_val3, \
        raw_train3, raw_val3, train_dataset3, val_dataset3, \
            raw_cols3 = genDatasetLost(df3, args.test_size, args.batch_size)
  
    logging.info(f"原始车辆数据准备完成: {len(X_train1)} 训练样本, {len(X_val1)} 验证样本")
    logging.info(f"丢失车辆数据准备完成: {len(X_train2)} 训练样本, {len(X_val2)} 验证样本")
    logging.info(f"合成车辆数据准备完成: {len(X_train3)} 训练样本, {len(X_val3)} 验证样本")



    ####################################################################################
    #-----------------------------------------------------------------------------
    #生成识别和回归双分支模型，预测同时识别和回归：车辆丢失概率和时间
     #----------------------------------------------------
    X_train4, X_val4, y_lost_train4, y_lost_val4, y_vanish_train4, y_vanish_val4,\
        raw_train4, raw_val4, \
            train_dataset4, val_dataset4, raw_cols4 = genDatasetLostFlagVanishTime(df3, args.test_size, args.batch_size)
    logging.info(f"双任务数据准备完成: {len(X_train4)} 训练样本, {len(X_val4)} 验证样本")
    model4_twoBranch = build_multi_task_resnet(X_train4.shape[1], args.unit, args.layNum)
    #---------------------------------------------------------------------------------------------
    # 多任务模型：同时预测车辆丢失概率和消失时间
    model4_twoBranch = build_multi_task_resnet(X_train3.shape[1], args.unit, args.layNum)

    # 使用学习率调度器
    initial_learning_rate = args.lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,  # 每100步衰减
        decay_rate=0.99,  # 每次衰减到99%
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # 编译多任务模型，为不同任务设置不同的损失函数和权重
    model4_twoBranch.compile(
        optimizer=optimizer,
        loss={
            'lost_prob': 'binary_crossentropy',  # 分类任务使用二元交叉熵
            'vanish_time': 'mse'  # 回归任务使用均方误差
        },
        loss_weights={
            'lost_prob': 10.0,  # 分类任务权重
            'vanish_time': 1.0  # 回归任务权重
        },
        metrics={
            'lost_prob': ['accuracy'],  # 分类任务的评估指标
            'vanish_time': ['mae']  # 回归任务的评估指标
        }
    )

    # 训练多任务模型
    history = model4_twoBranch.fit(
        train_dataset4,
        validation_data=val_dataset4,
        epochs=args.epochs,
        verbose=1
    )

    # 保存模型
    model4_twoBranch.save('resnet_model_twoBranch.h5')

    # 在验证集上进行预测
    predictions = model4_twoBranch.predict(X_val4)
    pred_lost, pred_vanish = predictions[0], predictions[1]

    # 分类任务评估
    # 将预测概率转换为二分类预测结果 (阈值设为0.5)
    y_pred_lost_binary = (pred_lost > 0.5).astype(int).flatten()
    y_true_lost_binary = y_lost_val4.astype(int)  # 假设y_val4是丢失标签
    
    # 计算准确率
    accuracy = np.mean(y_pred_lost_binary == y_true_lost_binary)
    print(f"Multi-task model - Lost vehicle detection accuracy on validation set: {accuracy:.4f}")
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true_lost_binary, y_pred_lost_binary)
    print("Confusion Matrix for Lost Vehicle Detection:")
    print(cm)
    
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Vehicle Lost', 'Vehicle Lost'], 
                yticklabels=['No Vehicle Lost', 'Vehicle Lost'])
    plt.title('Confusion Matrix for Lost Vehicle Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('modereg4_lost_confusion_matrix.png')
    plt.close()
    
    # 计算精确率、召回率和F1分数
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true_lost_binary, y_pred_lost_binary)
    recall = recall_score(y_true_lost_binary, y_pred_lost_binary)
    f1 = f1_score(y_true_lost_binary, y_pred_lost_binary)
    
    print(f"Precision for Lost Vehicle Detection: {precision:.4f}")
    print(f"Recall for Lost Vehicle Detection: {recall:.4f}")
    print(f"F1 Score for Lost Vehicle Detection: {f1:.4f}")
    
    # 回归任务评估
    # 假设y_val_vanish是消失时间标签（需要根据实际情况调整）
    # 这里使用原始的y_vanish_val，需要您在数据准备时提供
    # 计算MAE和MSE
   
    mae = np.mean(np.abs(pred_vanish.flatten() - y_vanish_val4))
    mse = np.mean((pred_vanish.flatten() - y_vanish_val4) ** 2)
    print(f"Multi-task model - Vanish time prediction MAE: {mae:.4f}")
    print(f"Multi-task model - Vanish time prediction MSE: {mse:.4f}")
    
    # 绘制预测结果对比图
    plt.figure(figsize=(12, 5))
    
    # 分类任务预测结果
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(pred_lost)), pred_lost, label='Predicted Lost Prob', alpha=0.7, marker='o', color='blue')
    plt.scatter(range(len(y_val3)), y_val3, label='True Lost Labels', alpha=0.7, marker='x', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Lost Vehicle Prediction')
    plt.legend()
    
    # 回归任务预测结果
   
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(pred_vanish)), pred_vanish, label='Predicted Vanish Time', alpha=0.7, marker='o', color='blue')
    plt.scatter(range(len(y_vanish_val4)), y_vanish_val4, label='True Vanish Time', alpha=0.7, marker='x', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Vanish Time Prediction')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('modereg4_predictions_and_true_values.png')
    plt.close()

    return

    #-----------------------------------------------------------------------------
    #生成识别模型 - 二分类任务（是否丢失车辆），输出维度为1
    output_dim = 1  # 二分类任务，输出为1维，表示车辆丢失的概率
    modelReg3 = build_simple_resnet(X_train3.shape[1], output_dim, args.unit, args.layNum)


    # 使用学习率调度器，开始时学习率较高，然后逐渐降低
    initial_learning_rate = args.lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,  # 每100步衰减
        decay_rate=0.99,  # 每次衰减到95%
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # 对于二分类任务，使用binary_crossentropy损失函数
    modelReg3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 自定义回调函数，每10个epoch才在验证集上评估并输出准确率
    class PeriodicValidationCallback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, validation_freq=10):
            super().__init__()
            self.validation_data = validation_data
            self.validation_freq = validation_freq

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.validation_freq == 0:  # 每10个epoch在验证集上评估
                val_result = self.model.evaluate(self.validation_data, verbose=0)
                val_loss, val_accuracy = val_result[0], val_result[1]
                print(f'Epoch {epoch + 1} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}')

    # 训练模型，verbose=1会输出每个epoch的loss和accuracy
    # 验证只在特定epoch进行，通过自定义回调实现
    validation_callback = PeriodicValidationCallback(val_dataset3, validation_freq=10)
    # 使用validation_freq参数控制验证频率，同时使用回调输出结果
    modelReg3.fit(train_dataset3, validation_data=val_dataset3, epochs=args.epochs,
                  verbose=1, callbacks=[validation_callback], validation_freq=10)

    # Save the model
    modelReg3.save('resnet_model_lostReg3.h5')
 
# ... existing code ...
    np.set_printoptions(suppress=True, precision=4)
    # Evaluate the model on the validation set
    predictions = modelReg3.predict(X_val3)
    # Convert prediction probabilities to binary classification results (threshold at 0.5)
    y_pred_binary = (predictions > 0.5).astype(int).flatten()
    y_true_binary = y_val3.astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_binary == y_true_binary)
    print(f"Model accuracy on validation set: {accuracy:.4f}")
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Vehicle Lost', 'Vehicle Lost'], 
                yticklabels=['No Vehicle Lost', 'Vehicle Lost'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('modereg3_confusion_matrix.png')
    plt.close()
    
    # Calculate precision, recall, and F1 score
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    

 

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Keras和交通仿真进行端到端模型训练")
    parser.add_argument('--csv_path', type=str, default='trainsamples_lane_5_6_7.csv', help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.9, help='验证集比例')
    parser.add_argument('--unit', type=int, default=128, help='ResNet隐藏层单元数')
    parser.add_argument('--layNum', type=int, default=8, help='ResNet块数量')
    parser.add_argument('--log_path', type=str, default='training_log.log', help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用Debug级别的日志信息')
    args = parser.parse_args()
    main(args)

#python modelsLostReg.py --batch_size 1280 --test_size 0.9 --epochs 100 --lr 0.005 --unit 256 --layNum 8 --dt 0.1


