# ===================== 系统库导入 =====================
import time
import os
import random
import sys
import argparse
import logging
import gc

# ===================== 第三方库导入 =====================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add, Dropout
from tensorflow.keras.optimizers import Adam, Adadelta, SGD, Adamax, RMSprop, AdamW
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ===================== 本地模块导入 =====================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tf_idm_simulation import tf_idm_simulation
from modelsLostReg import (
    genDatasetLost,
    genSamplesByRandomRemovingVehicle,
    genSamplesRemovingVehicleWithNum
)

# ===================== 全局常量定义（统一修改入口） =====================
# 车道位置映射
LANE_POS_MAP = {5: 53.05, 6: 53.13, 7: 53.30}

# IDM参数边界
BASE_BOUND_VEHICLE = [
    (30/3.6, 75/3.6),  # v0
    (0.1, 2.0),        # T
    (0.2, 1.0),        # s0
    (1.0, 6.0),        # a
    (1.0, 9.0),        # b
    (0.01, 1.0)        # rtime
]

# 保存目录常量
DIR_TMP_MODEL = "./tmpModes"
DIR_EVAL_MODEL0 = "./evaluation_results_model0"
DIR_EVAL_MODEL1 = "./evaluation_results_model1"

# 仿真相关常量
DEFAULT_DT = 0.5
DEFAULT_N_CLUSTERS = 100
MIN_GAP = 0.5  # 车辆最小间距
OFFSET_DISTANCE = 5.0  # 补全车辆位置偏移量
OVERSAMPLE_FACTOR = 2.0  # 丢失车辆样本过采样因子

# 日志格式
LOG_FORMAT = '%(asctime)s [%(levelname)s] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ===================== TensorFlow 配置 =====================
# 显存按需分配
def setup_tf_memory():
    """配置TensorFlow内存使用策略"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 可选：限制显存最大使用比例
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=5200)]
        # )

# 初始化TF配置
setup_tf_memory()

# ===================== 通用工具函数 =====================
def make_dir_safe(path: str):
    """安全创建目录，已存在不报错"""
    os.makedirs(path, exist_ok=True)

def get_car_pos_speed_cols(col_list):
    """统一提取car_position_、car_speed_开头列，多处复用"""
    pos_cols = [c for c in col_list if c.startswith('car_position_')]
    speed_cols = [c for c in col_list if c.startswith('car_speed_')]
    return pos_cols, speed_cols

def format_tensor(tensor, decimals=1):
    """格式化张量到指定小数位数"""
    factor = tf.constant(10 ** decimals, dtype=tensor.dtype)
    rounded = tf.round(tensor * factor) / factor
    return rounded

def force_clean_all_memory():
    """完善内存释放函数，清空tf缓存、垃圾回收"""
    # 清理TF会话
    tf.keras.backend.clear_session()
    if tf.config.list_physical_devices("GPU"):
        tf.config.experimental.reset_memory_stats("GPU:0")
    # 系统垃圾回收
    gc.collect()
    logging.info("内存清理完成")

def setup_logger(log_path, debug=False):
    """配置日志记录器，同时输出到文件和控制台"""
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清空已有处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def get_param_bounds(num_types):
    """根据类别数量生成参数边界"""
    return np.array([BASE_BOUND_VEHICLE for _ in range(num_types)], dtype=np.float32)

def generate_timestamp():
    """生成统一时间戳，减少系统调用"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def get_sample_indices(df, num_samples):
    """封装样本索引生成逻辑，避免重复计算"""
    if len(df) <= num_samples:
        return df.sample(frac=1, random_state=42).index.tolist()
    
    # KMeans聚类保证样本多样性
    sample_features = df[[c for c in df.columns if 'car_position_' in c or 'car_speed_' in c]].values
    n_clusters = min(DEFAULT_N_CLUSTERS, len(df) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
    cluster_labels = kmeans.fit_predict(sample_features)
    
    sampled_indices = []
    # 1/2 聚类抽样
    cluster_sample_num = max(1, int(num_samples / 2 / n_clusters))
    for cluster in range(n_clusters):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        if len(cluster_idx) > 0:
            chosen = np.random.choice(
                cluster_idx, 
                size=min(cluster_sample_num, len(cluster_idx)), 
                replace=False
            )
            sampled_indices.extend(chosen)
    
    # 1/2 lost加权抽样补齐
    if len(sampled_indices) < num_samples:
        remaining = list(set(range(len(df))) - set(sampled_indices))
        if remaining:
            lost_vals = df['lost'].iloc[remaining].fillna(0).astype(float).values
            weights = 1.0 + lost_vals * OVERSAMPLE_FACTOR
            probs = weights / np.sum(weights)
            
            extra_num = num_samples - len(sampled_indices)
            extra = np.random.choice(
                remaining, 
                size=extra_num, 
                replace=False, 
                p=probs
            )
            sampled_indices.extend(extra)
    
    # 截断到指定数量并去重
    sampled_indices = list(dict.fromkeys(sampled_indices))[:num_samples]
    return sampled_indices

# ===================== 网络模型定义 =====================
def build_simple_resnet(input_dim, output_dim, unit=256, layNum=8):
    """基础残差输出sigmoid，用于IDM参数解码"""
    def resnet_block(x, units):
        shortcut = x
        y = Dense(units)(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)

        y = Dense(units)(y)
        y = BatchNormalization()(y)

        if shortcut.shape[-1] != units:
            shortcut = Dense(units)(shortcut)

        y = Add()([shortcut, y])
        y = ReLU()(y)
        return y

    inp = Input(shape=(input_dim,))
    x = Dense(unit)(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(layNum):
        x = resnet_block(x, unit)

    out = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def build_simple_resnet2(input_dim, output_dim, unit=256, layNum=8):
    """带Dropout预激活残差块，防过拟合，用于IDM参数解码"""
    def resnet_block(x, units, dropout_rate=0.2):
        shortcut = x
        y = BatchNormalization()(x)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        y = Dropout(dropout_rate)(y)

        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        y = Dropout(dropout_rate)(y)

        if shortcut.shape[-1] != units:
            shortcut = Dense(units)(shortcut)

        y = Add()([shortcut, y])
        return y

    inp = Input(shape=(input_dim,))
    x = Dense(unit, kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(layNum):
        x = resnet_block(x, unit, dropout_rate=0.2)

    out = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def build_simple_resnet_regress(input_dim, output_dim, unit=256, layNum=8):
    """基础残差回归头，线性输出消失时间"""
    def resnet_block(x, units):
        shortcut = x
        y = Dense(units)(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)

        y = Dense(units)(y)
        y = BatchNormalization()(y)

        if shortcut.shape[-1] != units:
            shortcut = Dense(units)(shortcut)

        y = Add()([shortcut, y])
        y = ReLU()(y)
        return y

    inp = Input(shape=(input_dim,))
    x = Dense(unit)(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(layNum):
        x = resnet_block(x, unit)

    out = Dense(1, activation='linear', name='vanish_time')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def build_simple_resnet_regress2(input_dim, output_dim, unit=128, layNum=4):
    """轻量化带Dropout回归网络，直接预测消失时间"""
    def resnet_block(x, units, dropout_rate=0.2):
        shortcut = x
        y = BatchNormalization()(x)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        y = Dropout(dropout_rate)(y)

        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        y = Dropout(dropout_rate)(y)

        if shortcut.shape[-1] != units:
            shortcut = Dense(units)(shortcut)

        y = Add()([shortcut, y])
        return y

    inp = Input(shape=(input_dim,))
    x = Dense(unit, kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(layNum):
        x = resnet_block(x, unit, dropout_rate=0.2)

    out = Dense(1, activation='linear', name='vanish_time')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def rmse(y_true, y_pred):
    """自定义rmse损失指标"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ===================== 模型0训练：IDM仿真嵌套端到端训练 =====================
def train_model_mlp_cf(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt):
    """
    MLP+CF参数端到端训练
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param raw_train: 原始训练数据
    :param train_dataset: 训练数据集
    :param val_dataset: 验证数据集
    :param raw_cols: 原始数据列名
    :param args: 命令行参数
    :param dt: 仿真时间步长
    :return: 训练好的模型
    """
    num_types = args.num_types
    num_types2 = num_types + 1
    output_dim = num_types2 * 6

    # 构建模型
    model = build_simple_resnet2(X_train.shape[1], output_dim, args.unit, args.layNum)
    param_bounds = get_param_bounds(num_types)
    
    # 优化器配置
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr, decay_steps=100, decay_rate=0.99, staircase=True
    )
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)



    pos_idx_list = [i for i, c in enumerate(raw_cols) if c.startswith("car_position_")]
    speed_idx_list = [i for i, c in enumerate(raw_cols) if c.startswith("car_speed_")]
    idx_main_car = raw_cols.index("main_car_position")
    idx_inter = raw_cols.index("intersection_pos")
    idx_red = raw_cols.index("redLightRemainingTime")

    # 转为静态int张量，闭包捕获传入sim_wrapper
    tf_pos_idx = tf.constant(pos_idx_list, dtype=tf.int32)
    tf_speed_idx = tf.constant(speed_idx_list, dtype=tf.int32)
    tf_idx_main = tf.constant(idx_main_car, dtype=tf.int32)
    tf_idx_inter = tf.constant(idx_inter, dtype=tf.int32)
    tf_idx_red = tf.constant(idx_red, dtype=tf.int32)


    @tf.function(reduce_retracing=True)
    def train_step(x_batch, y_batch, raw_batch):
        """训练步（TF函数装饰，减少重追踪）"""
        with tf.GradientTape() as tape:
            nn_output = model(x_batch, training=True)
            # 仿真封装
            #def sim_wrapper(nn_out, raw):
            #    return tf_idm_simulation(
            #        nn_out, raw, param_bounds, num_types, 
            #        tf_pos_idx, tf_speed_idx, tf_idx_main, tf_idx_inter, tf_idx_red,
            #        dt=dt, go_flag=args.goffset
            #    )
            #predicted_times = tf.py_function(
            #    sim_wrapper, [nn_output, raw_batch], Tout=tf.float32
            #)

            predicted_times = tf_idm_simulation(
                    nn_output, raw_batch, param_bounds, num_types,
                    tf_pos_idx, tf_speed_idx, tf_idx_main, tf_idx_inter, tf_idx_red,
                    dt, args.goffset
                    )
            
            # 形状调整
            batch_size = tf.shape(y_batch)[0]
            predicted_times = tf.reshape(predicted_times, [batch_size])
            y_batch = tf.reshape(y_batch, [batch_size])
            loss = tf.reduce_mean(tf.square(predicted_times - y_batch))

        # 梯度裁剪与更新
        grads = tape.gradient(loss, model.trainable_variables)
        clipped_grads = [
            tf.clip_by_norm(g, 1.0) if g is not None else g 
            for g in grads
        ]
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def val_step(x_batch, y_batch, raw_batch):
        """验证步（TF函数装饰，减少重追踪）"""
        nn_output = model(x_batch, training=False)
        predicted_times = tf_idm_simulation(
                nn_output, raw_batch, param_bounds, num_types,
                tf_pos_idx, tf_speed_idx, tf_idx_main, tf_idx_inter, tf_idx_red,
                dt, args.goffset
                )
        # 形状调整
        batch_size = tf.shape(y_batch)[0]
        predicted_times = tf.reshape(predicted_times, [batch_size])
        y_batch = tf.reshape(y_batch, [batch_size])
        return predicted_times - y_batch

    # 在训练前做一次仿真函数的warm-up以触发tf.function编译，避免首个训练batch出现长时间编译延迟
    try:
        warm_batch_size = min(args.batch_size, X_train.shape[0])
        if warm_batch_size > 0:
            # 构造伪网络输出和原始数据的切片进行一次调用
            dummy_nn = tf.zeros((warm_batch_size, (num_types + 1) * 6), dtype=tf.float32)
            dummy_raw = tf.convert_to_tensor(raw_train[:warm_batch_size], dtype=tf.float32)
            # 调用一次以触发tf.function的编译/缓存
            _ = tf_idm_simulation(
                dummy_nn, dummy_raw, param_bounds, num_types,
                tf_pos_idx, tf_speed_idx, tf_idx_main, tf_idx_inter, tf_idx_red,
                dt, args.goffset
            )
    except Exception:
        # 如果warm-up失败，不阻止训练
        logging.warning("仿真warm-up失败，继续训练（非致命）")

    # 开始训练
    total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    for epoch in range(args.epochs):
        logging.info(f"===== Epoch {epoch + 1}/{args.epochs} =====")
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_idx = 0

        for x_batch, y_batch, raw_batch in train_dataset:
            batch_idx += 1
            t0 = time.time()
            loss = train_step(x_batch, y_batch, raw_batch)
            t1 = time.time()
            
            epoch_loss_avg.update_state(loss)
            rem_batches = total_batches - batch_idx
            logging.info(
                f"Batch {batch_idx}/{total_batches} | Loss: {loss.numpy():.4f} "
                f"| Time: {t1-t0:.2f}s | Remain: {rem_batches}"
            )

        # _epoch平均损失
        epoch_loss = epoch_loss_avg.result().numpy()
        logging.info(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

        # 验证逻辑
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            logging.info("===== 验证阶段开始 =====")
            val_errs = []
            val_loss_metric = tf.keras.metrics.Mean()

            for xb, yb, rb in val_dataset:
                errs = val_step(xb, yb, rb)
                enp = errs.numpy()
                val_errs.append(enp)
                val_loss_metric.update_state(np.mean(np.square(enp)))

            # 计算验证指标
            val_errs = np.concatenate([e.flatten() for e in val_errs])
            val_rmse = np.sqrt(np.mean(np.square(val_errs)))
            val_mae = np.mean(np.abs(val_errs))
            
            logging.info(
                f"Validation Results - RMSE: {val_rmse:.4f}, "
                f"MAE: {val_mae:.4f}, MSE: {val_loss_metric.result().numpy():.4f}"
            )

    # 模型保存
    make_dir_safe(DIR_TMP_MODEL)
    timestamp = generate_timestamp()
    save_path = f"{DIR_TMP_MODEL}/model0_{timestamp}_epoch_{epoch+1}.h5"
    model.save(save_path)
    logging.info(f"Model 0 saved to: {save_path}")

    return model

# ===================== 模型1训练：直接回归预测消失时间 =====================
def train_model_mlp_reg(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt, raw_val=None):
    """
    MLP直接回归预测消失时间
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param raw_train: 原始训练数据
    :param train_dataset: 训练数据集
    :param val_dataset: 验证数据集
    :param raw_cols: 原始数据列名
    :param args: 命令行参数
    :param dt: 仿真时间步长
    :param raw_val: 原始验证数据
    :return: 训练好的模型
    """
    logging.info("启动MLP直接回归模型训练（预测消失时间）")
    output_dim_vanish = 1
    
    # 构建模型
    model_vanish_reg = build_simple_resnet_regress2(
        X_train.shape[1], output_dim_vanish, args.unit, args.layNum
    )

    # 优化器配置
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr, decay_steps=100, decay_rate=0.99, staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    # 模型编译
    model_vanish_reg.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', rmse]
    )

    # 早停策略
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=100, 
        restore_best_weights=True, 
        verbose=1
    )

    # 模型训练
    model_vanish_reg.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        verbose=1,
        callbacks=[early_stop],
        shuffle=True
    )

    # 评估指标
    train_loss, train_mae, train_rmse = model_vanish_reg.evaluate(train_dataset, verbose=0)
    val_loss, val_mae, val_rmse = model_vanish_reg.evaluate(val_dataset, verbose=0)
    
    logging.info(
        f"Training Set - MAE: {train_mae:.4f}, MSE: {train_loss:.4f}, RMSE: {train_rmse:.4f}"
    )
    logging.info(
        f"Validation Set - MAE: {val_mae:.4f}, MSE: {val_loss:.4f}, RMSE: {val_rmse:.4f}"
    )

    # 模型保存
    make_dir_safe(DIR_TMP_MODEL)
    timestamp = generate_timestamp()
    model_path = f"{DIR_TMP_MODEL}/model1_reg_{timestamp}.h5"
    model_vanish_reg.save(model_path)
    logging.info(f"Regression model saved to: {model_path}")

    return model_vanish_reg

# ===================== 数据修补函数 =====================
def fix_missing_data(df, fix_type):
    """
    统一的数据修补入口函数
    :param df: 原始数据
    :param fix_type: 修补类型 0-不修补 1-直接补原始数据 2-前后车偏移补
    :return: 修补后的数据
    """
    if fix_type == 0:
        lost_count = len(df.index[df['lost'] == 1])
        logging.info(f"不修补数据，采样样本数: {len(df)}, 缺失样本数: {lost_count}")
        return df
    
    # 获取丢失样本索引
    lost_indices = df.index[df['lost'] > 0].tolist()
    logging.info(f"开始修补数据，共 {len(lost_indices)} 个缺失样本，修补类型: {fix_type}")

    if fix_type == 1:
        # 方法1：直接用原始数据补
        for idx in lost_indices:
            removed_vehs = df.at[idx, 'removed_vehicles']
            for car_pos_col, car_pos, car_pos_i in removed_vehs:
                # 补位置
                df.at[idx, f'car_position_{car_pos_i}'] = car_pos
                # 补速度（取前车速度）
                car_speed_i = max(0, car_pos_i - 1)
                df.at[idx, f'car_speed_{car_pos_i}'] = df.at[idx, f'car_speed_{car_speed_i}']

    elif fix_type == 2:
        # 方法2：前后车偏移补
        for idx in lost_indices:
            # 获取当前样本有效车辆位置和速度
            pos_cols, speed_cols = get_car_pos_speed_cols(df.columns)
            valid_positions = []
            valid_speeds = []
            
            for pos_col, speed_col in zip(pos_cols, speed_cols):
                pos_val = df.at[idx, pos_col]
                if pos_val != -1:
                    valid_positions.append(pos_val)
                    valid_speeds.append(df.at[idx, speed_col])
            
            if not valid_positions:
                continue

            # 按位置排序丢失车辆
            removed_vehs = df.at[idx, 'removed_vehicles']
            removed_sorted = sorted(removed_vehs, key=lambda x: x[1])

            for _, orig_car_pos, car_pos_i in removed_sorted:
                # 找最近车辆
                valid_arr = np.array(valid_positions)
                nearest_idx = np.argmin(np.abs(valid_arr - orig_car_pos))
                nearest_pos = valid_arr[nearest_idx]
                nearest_speed = valid_speeds[nearest_idx]

                # 计算新位置
                if orig_car_pos > nearest_pos:
                    new_pos = nearest_pos + OFFSET_DISTANCE
                else:
                    new_pos = nearest_pos - OFFSET_DISTANCE

                # 避免重叠
                if abs(new_pos - nearest_pos) < MIN_GAP:
                    new_pos = (
                        nearest_pos + MIN_GAP 
                        if orig_car_pos > nearest_pos 
                        else nearest_pos - MIN_GAP
                    )

                # 写入数据
                df.at[idx, f'car_position_{car_pos_i}'] = new_pos
                df.at[idx, f'car_speed_{car_pos_i}'] = nearest_speed

                # 更新有效列表
                valid_positions.append(new_pos)
                valid_speeds.append(nearest_speed)
                # 保持有序
                combined = sorted(zip(valid_positions, valid_speeds), key=lambda x: x[0])
                valid_positions, valid_speeds = zip(*combined) if combined else ([], [])
                valid_positions = list(valid_positions)
                valid_speeds = list(valid_speeds)

    logging.info(f"数据修补完成，共处理 {len(lost_indices)} 个缺失样本")
    return df

# ===================== 主函数 =====================
def main(args):
    """主训练流程"""
    # 初始化日志
    logger = setup_logger(args.log_path, args.debug)
    logger.info(f"训练启动，参数配置: {args}")
    
    # ===================== 1. 数据加载与预处理 =====================
    logger.info(f"从 {args.csv_path} 加载数据...")
    df1 = pd.read_csv(args.csv_path).dropna()
    df1['lost'] = 0
    df1['removed_vehicles'] = None
    df1.rename(columns={
        'car_position': 'main_car_position',
        'car_speed': 'main_car_speed'
    }, inplace=True)

    # ===================== 2. 生成缺失数据样本 =====================
    logger.info("生成缺失车辆样本...")
    # 生成不同数量丢失车辆的样本
    df_missveh_rn1, _, df_missveh2_rn1 = genSamplesRemovingVehicleWithNum(df1, num_to_remove=1)
    df_missveh_rn2, _, df_missveh2_rn2 = genSamplesRemovingVehicleWithNum(df1, num_to_remove=2)
    df_missveh_rn3, _, df_missveh2_rn3 = genSamplesRemovingVehicleWithNum(df1, num_to_remove=3)
    df_missveh_rn4, _, df_missveh2_rn4 = genSamplesRemovingVehicleWithNum(df1, num_to_remove=4)
    
    # 合并缺失样本
    df_step2_missveh2 = pd.concat([
        df_missveh2_rn1, df_missveh2_rn2, 
        df_missveh2_rn3, df_missveh2_rn4
    ], ignore_index=True)

    # ===================== 3. 样本合并与过滤 =====================
    # 选择训练验证模式
    if args.trainvalmode == 0:
        df_all = df1
    else:
        df_all = pd.concat([df1, df_step2_missveh2], ignore_index=True)
    
    # 添加路口位置列
    df_all['intersection_pos'] = df_all['lane'].map(LANE_POS_MAP)

    # 样本过滤（排队异常/丢失过多/消失时间过长）
    logger.info("开始样本过滤...")
    def count_queued_vehicles(row):
        """统计主车前方排队车辆数"""
        pos_cols, _ = get_car_pos_speed_cols(row.index)
        main_pos = row['main_car_position']
        queued_count = 0
        for col in pos_cols:
            pos = row[col]
            if pos != -1 and not pd.isna(pos) and pos < main_pos:
                queued_count += 1
        return queued_count

    # 计算排队车辆数
    df_all['queued_vehicles'] = df_all.apply(count_queued_vehicles, axis=1)
    
    # 过滤条件
    cond_queued = (df_all['queued_vehicles'] > 3) | (df_all['queued_vehicles'] < 1)
    cond_lost = df_all['lost'] >= 3
    cond_vanish = df_all['time_to_vanish'] > 35 * 30  # 还原原始时间
    
    # 执行过滤
    before_count = len(df_all)
    df_all = df_all[~(cond_queued | cond_lost | cond_vanish)].reset_index(drop=True)
    after_count = len(df_all)
    logger.info(
        f"样本过滤完成 - 过滤前: {before_count} 个, "
        f"过滤后: {after_count} 个, "
        f"删除: {before_count - after_count} 个"
    )

    # ===================== 4. 样本抽样 =====================
    logger.info(f"开始样本抽样，目标数量: {args.nC}")
    sampled_indices = get_sample_indices(df_all, args.nC)
    df_sampled = df_all.iloc[sampled_indices].reset_index(drop=True)
    logger.info(
        f"抽样完成 - 最终样本数: {len(df_sampled)}, "
        f"聚类抽样数: {min(len(sampled_indices)//2, args.nC//2)}, "
        f"加权补充数: {max(0, len(sampled_indices) - args.nC//2)}"
    )

    # ===================== 5. 数据修补 =====================
    df_fixed = fix_missing_data(df_sampled, args.fixdata)

    # ===================== 6. 数据集构建 =====================
    # 特征列和原始数据列
    feature_cols = [
        c for c in df_fixed.columns 
        if ('car_position_' in c or 'car_speed_' in c or 'redLight' in c)
    ]
    raw_cols_set = set(feature_cols)
    raw_cols_set.update([
        'lane', 'intersection_pos', 'main_car_position',
        'main_car_speed', 'lost'
    ])
    raw_cols = sorted(list(raw_cols_set))

    # 数据转换
    X = df_fixed[feature_cols].values.astype(np.float32)
    y = (df_fixed['time_to_vanish'].values / 30.0).astype(np.float32)
    raw_data_for_sim = df_fixed[raw_cols].values.astype(np.float32)

    # 划分训练验证集
    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
        X, y, raw_data_for_sim, 
        test_size=args.test_size, 
        random_state=42
    )

    logger.info(
        f"数据集构建完成 - 训练集: {len(X_train)} 样本, "
        f"验证集: {len(X_val)} 样本"
    )

    # ===================== 7. 模型训练 =====================
    dt = args.dt or DEFAULT_DT
    logger.info(f"开始模型训练，仿真时间步长: {dt}")

    if args.model == 0:
        # 模型0：MLP+CF端到端训练
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, raw_train))
        # 保持每个batch形状一致以减少tf.function retracing并降低编译开销
        train_dataset = train_dataset.cache().batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, raw_val))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_model_mlp_cf(
            X_train, y_train, raw_train,
            train_dataset, val_dataset, raw_cols,
            args, dt
        )

    elif args.model == 1:
        # 模型1：MLP直接回归
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_model_mlp_reg(
            X_train, y_train, raw_train,
            train_dataset, val_dataset, raw_cols,
            args, dt, raw_val=raw_val
        )

    # 清理内存
    force_clean_all_memory()
    logger.info("训练流程完成")

# ===================== 入口执行 =====================
if __name__ == "__main__":
    # 启动TF性能分析
    log_dir = "./profiler_records"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.profiler.experimental.server.start(6009)
    #tf.profiler.experimental.start(log_dir)

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="使用Keras和交通仿真进行端到端模型训练")
    parser.add_argument('--csv_path', type=str, default='trainsamples_lane_5_6_7.csv', 
                        help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.9, help='验证集比例')
    parser.add_argument('--num_types', type=int, default=6, help='车辆类别数')
    parser.add_argument('--unit', type=int, default=128, help='ResNet隐藏层单元数')
    parser.add_argument('--layNum', type=int, default=8, help='ResNet块数量')
    parser.add_argument('--log_path', type=str, default='training_log.log', help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用Debug级别的日志信息')
    parser.add_argument('--dt', type=float, default=DEFAULT_DT, help='仿真时间步长')
    parser.add_argument('--nC', type=int, default=1000, help='抽样样本数量')
    parser.add_argument('--model', type=int, default=0, help='0(MLP+CF),1(MLP+Regress)')
    parser.add_argument('--fixdata', type=int, default=0, help='0(不修补),1(原始数据补),2(前后车偏移补)')
    parser.add_argument('--goffset', type=int, default=1, help='仿真全局偏移参数开关')
    parser.add_argument('--trainvalmode', type=int, default=0, help='0(无丢失),1(有丢失)')
   

    args = parser.parse_args()
    main(args)

    # 停止性能分析
    #tf.profiler.experimental.stop()
    tf.profiler.experimental.server.stop()
