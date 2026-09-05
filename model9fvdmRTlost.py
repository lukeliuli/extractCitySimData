# ===================== 系统库导入 =====================
import time
import os
import random
import sys
import argparse
import logging
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'                 # 0=全部 1=关INFO 2=关INFO+WARNING 3=全关(含ERROR)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # 规避 protobuf GetPrototy
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
from tensorflow.keras.models import load_model
#tf.debugging.enable_check_numerics() 
# ===================== 本地模块导入 =====================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tf_fvdm_simulation import tf_fvdm_simulation

pd.options.mode.chained_assignment = None
from modelsLostReg import (
    genDatasetLost,
    genSamplesByRandomRemovingVehicle,
    genSamplesRemovingVehicleWithNum,
    genSamplesRemovingVehicleWithOneSlot,
    count_queued_vehicles
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

# FVDM参数边界,v0, alpha, lam, length, tanhP1, rtime
BASE_BOUND_VEHICLE = [
    (22, 36),          # v0
    (0.1, 1.0),        # alpha
    (0.5, 5.0),        # lam
    (1.0, 6.0),        # length
    (1.0, 9.0),        # tanhP1
    (0.01, 1.0)        # rtime
]
# w99参数边界
"""
| 参数 | 含义 | 原始文献/默认值范围 | 你的代码设置 | 问题所在 |
| :--- | :--- | :--- | :--- | :--- |
| CC0 | 停车间距 (Standstill Distance) | 1.0 ~ 2.0 m | (0.5, 2.5) | 基本合理 |
| CC1 | 车头时距 (Headway Time) | 0.5 ~ 2.0 s | (0.5, 2.5) | 基本合理 |
| CC2 | 跟车变量 (Following Variation) | 0.1 ~ 0.6 m/s² | (0.3, 1.5) | 上限偏高 |
| CC3 | 
| CC4 | 负的相对速度阈值 (Neg. "Following" Threshold) | -0.35 ~ 0.0 m/s | (0.5, 1.5) | 完全错误 |
| CC5 | 正的相对速度阈值 (Pos. "Following" Threshold) | 0.0 ~ 0.35 m/s | (1.6, 3.0) | 完全错误 |
| CC6 | 速度依赖性 (Speed Dependency) | 10.0 ~ 20.0 | (0.0, 0.5) | 量级错误 |
| CC7 | 加速度波动 (Oscillation Acceleration) | 0.1 ~ 0.5 m/s² | (0.5, 2.0) | 偏高 |
| CC8 | 启动加速度 (Standstill Acceleration) | 1.0 ~ 3.0 m/s² | (1.5, 3.0) | 基本合理 |
| CC9 | 高速加速度 (Acceleration at 80 km/h) | 0.5 ~ 1.5 m/s² | (0.8, 1.8) | 基本合理 |
"""
# FVDM参数边界（激活）：v0, alpha, lam, length, tanhP1, rtime
BASE_BOUND_VEHICLE = [
    (22, 36),          # v0 期望速度 [m/s]
    (0.1, 1.0),        # alpha 灵敏度系数
    (0.5, 5.0),        # lam 相对速度灵敏度
    (1.0, 6.0),        # length 车长 [m]
    (1.0, 9.0),        # tanhP1 期望速度曲线参数
    (0.01, 1.0)        # rtime 反应时间 [s]
]

'''
#2026-08-25 16:00:50 [INFO] - Validation Results - RMSE: 1.4807, MAE: 1.1386, MSE: 2.1931
BASE_BOUND_VEHICLE = [
    (1.0, 2.0),   # 0: CC0 (停车间距) [m]
    (0.5, 2.0),   # 1: CC1 (车头时距) [s]
    (0.1, 20),   # 2: CC2  SDX=ABX+CC2,安全距离附加值
    (0.5, 9.0),   # 3: CC3  following状态下的D值,对应的速度差的权重
    (-3.0, 0.0001), # 4: CC4 (负的相对速度阈值) [m/s]CLDV 为负
    (0.0001, 3.0),  # 5: CC5 (正的相对速度阈值) [m/s] OPDV，为正
    (1.0,9.0), # 6: CC6   following状态下的D值,对应的速度差的权重
    (2.0, 5.0),   # 7: CC7 (加速度波动) [m/s²]
    (1.0, 3.0),   # 8: CC8 (启动加速度) [m/s²]
    (0.5, 1.5)    # 9: CC9 (高速加速度) [m/s²]
]
'''
#2026-08-25 16:09:05 [INFO] - Batch 1/1 | Loss: 10.6058 | Time: 264.72s | Remain: 0
#2026-08-25 16:09:05 [INFO] - Epoch 1 Average Loss: 10.6058
#2026-08-25 16:09:05 [INFO] - Trainning Results (Real Time) - MSE: 10.6058, RMSE: 3.2567, MAE: 2.5527
#2026-08-25 16:09:05 [INFO] -===== 验证阶段开始 =====
#2026-08-25 16:09:36 [INFO] - Validation Results - RMSE: 1.8879, MAE: 1.3736, MSE: 3.5463
# FVDM参数边界（最终激活）：v0, alpha, lam, length, tanhP1, rtime
BASE_BOUND_VEHICLE = [
    (22, 36),          # v0 期望速度 [m/s]
    (0.1, 1.0),        # alpha 灵敏度系数
    (0.5, 5.0),        # lam 相对速度灵敏度
    (1.0, 6.0),        # length 车长 [m]
    (1.0, 9.0),        # tanhP1 期望速度曲线参数
    (0.01, 1.0)        # rtime 反应时间 [s]
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
RUN_START_TIME = ""
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

def setup_logger(args):
    """配置日志记录器，同时输出到文件和控制台"""
    global RUN_START_TIME 
    log_path = args.log_path
    debug = args.debug
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清空已有处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器
    timestamp = generate_timestamp()
    RUN_START_TIME = timestamp
    log_path = f"./tmpModes/fvdm_trainlog_{timestamp}_{args.epochs}_{args.model}_{args.trainvalmode}_{args.batch_size}_{args.fixdata}.log"
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
def plot_param_boxplots(long_df, save_dir, tag=""):
    """按参数画盒状图（横轴为 group：all_types / type_N / global），保存 PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    make_dir_safe(save_dir)

    def group_key(g):
        if g == "all_types":
            return (0, 0)
        if g == "global":
            return (2, 0)
        return (1, int(g.split("_")[1]))

    for p in long_df["param"].unique().tolist():
        sub = long_df[long_df["param"] == p]
        order = sorted(sub["group"].unique().tolist(), key=group_key)
        data = [sub.loc[sub["group"] == g, "value"].values for g in order]

        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(order)), 4))
        ax.boxplot(data)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=15)
        ax.set_title(p)
        ax.set_xlabel("group")
        ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = f"{save_dir}/fvdm_boxplot_{p}.png" if not tag else f"{save_dir}/fvdm_boxplot_{p}_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)

    logging.info(f"盒状图已保存到: {save_dir}")

def get_param_bounds(num_types):
    """根据类别数量生成参数边界"""
    return np.array([BASE_BOUND_VEHICLE for _ in range(num_types)], dtype=np.float32)


def compute_param_stats(arr, name, bin_num=50):
    """对一维参数数组做统计：均值/方差/标准差/众数比例/分位数/偏度/峰度/贴边比例"""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return dict(param=name, count=0)

    # 众数比例：连续值用直方图分箱，取频数最高箱的比例
    counts, edges = np.histogram(arr, bins=bin_num)
    k = int(np.argmax(counts))
    mode_prop = float(counts[k]) / n
    mode_lo, mode_hi = float(edges[k]), float(edges[k + 1])
    mode_center = 0.5 * (mode_lo + mode_hi)

    mean = float(np.mean(arr))
    var = float(np.var(arr))            # 方差
    std = float(np.std(arr))            # 标准差（ddof=0，与 var 一致）
    q0, q5, q25, q50, q75, q95, q100 = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100])

    # 偏度 / 峰度（Fisher，手写以避免 scipy 依赖）
    m2 = np.mean((arr - mean) ** 2)
    m3 = np.mean((arr - mean) ** 3)
    m4 = np.mean((arr - mean) ** 4)
    skew = m3 / (m2 ** 1.5) if m2 > 1e-12 else 0.0
    kurt = m4 / (m2 ** 2) - 3.0 if m2 > 1e-12 else 0.0

    # 贴边界比例（用于盒状图观察是否大量值被裁剪到 min/max）
    rng = q100 - q0
    prop_min = float(np.mean(np.isclose(arr, q0))) if rng > 0 else 0.0
    prop_max = float(np.mean(np.isclose(arr, q100))) if rng > 0 else 0.0

    return dict(
        param=name, count=n,
        mean=mean, var=var, std=std,
        mode_prop=mode_prop, mode_center=mode_center, mode_range=(mode_lo, mode_hi),
        min=q0, p5=q5, q1=q25, median=q50, q3=q75, p95=q95, max=q100,
        iqr=q75 - q25, skew=skew, kurt=kurt,
        prop_at_min=prop_min, prop_at_max=prop_max,
    )
def report_val_params_stats(real_params, scene_offset, tag=""):
    """统计 CC0-CC9（聚合 all_types + 按车型 type_0..N-1）及全局 2 个偏移，
    输出表格并导出盒状图长表 CSV"""
    param_names = ["v0", "alpha", "lam", "length", "tanhP1", "rtime"]
    num_types = real_params.shape[1]
    rows, long_records = [], []

    def _add_stats(vals, pname, group):
        st = compute_param_stats(vals, pname)
        st["group"] = group
        rows.append(st)
        for v in vals:
            long_records.append((pname, group, float(v)))

    # 1) CC0-CC9：所有车型聚合
    for i, pname in enumerate(param_names):
        _add_stats(real_params[..., i].ravel(), pname, "all_types")

    # 2) CC0-CC9：按车型分组
    for t in range(num_types):
        gname = f"type_{t}"
        for i, pname in enumerate(param_names):
            _add_stats(real_params[:, t, i].ravel(), pname, gname)

    # 3) 全局两个偏移
    if scene_offset is not None:
        red_time = (scene_offset[:, 0] * 2.0 - 1.0) * 2.0   # redlighttime_offset
        red_pos  = scene_offset[:, 3] * 2.0                  # redlightpos_offset
        _add_stats(red_time, "redlight_time_offset", "global")
        _add_stats(red_pos,  "redlight_pos_offset",  "global")

    df = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_records, columns=["param", "group", "value"])

    cols_show = ["group", "param", "count", "mean", "var", "std", "median", "q1", "q3",
                 "min", "max", "mode_prop", "mode_range", "skew", "kurt",
                 "prop_at_min", "prop_at_max"]
    logging.info(
        f"\n========== 验证集参数统计 ({tag}) ==========\n"
        + df[cols_show].round(4).to_string(index=False)
    )

    make_dir_safe(DIR_EVAL_MODEL0)
    stamp = generate_timestamp()
    df.to_csv(f"{DIR_EVAL_MODEL0}/fvdm_val_param_stats_{stamp}.csv", index=False)
    long_df.to_csv(f"{DIR_EVAL_MODEL0}/fvdm_val_param_boxplot_{stamp}.csv", index=False)
    logging.info(f"参数统计已导出: {DIR_EVAL_MODEL0}/fvdm_val_param_stats_{stamp}.csv")
    logging.info(f"盒状图数据已导出: {DIR_EVAL_MODEL0}/fvdm_val_param_boxplot_{stamp}.csv")
    return df, long_df

def generate_timestamp():
    """生成统一时间戳，减少系统调用"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def get_sample_indices(df, num_samples):
    """封装样本索引生成逻辑，避免重复计算"""
    if len(df) <= num_samples:
        return df.sample(frac=1, random_state=42).index.tolist()
    
    # KMeans聚类保证样本多样性
    sample_features = df[[c for c in df.columns if 'car_position_' in c or 'car_speed_' in c]].values
    n_clusters = min(DEFAULT_N_CLUSTERS, len(df) // 50)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
    cluster_labels = kmeans.fit_predict(sample_features)
    
    sampled_indices = []
    # 1/2 聚类抽样
    cluster_sample_num = max(1, int(num_samples / n_clusters))
    for cluster in range(n_clusters):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        if len(cluster_idx) > 0:
            chosen = np.random.choice(
                cluster_idx, 
                size=min(cluster_sample_num, len(cluster_idx)), 
                replace=False
            )
            sampled_indices.extend(chosen)
    logging.info(f"n_clusters:{n_clusters},kmeans_samples:{len(sampled_indices)}")
    # 补齐
    
    extra = []
    if len(sampled_indices) < num_samples:
        remaining = list(set(range(len(df))) - set(sampled_indices))
        if remaining:
            #lost_vals = df['lost'].iloc[remaining].fillna(0).astype(float).values
            #weights = 1.0 + lost_vals * OVERSAMPLE_FACTOR
            #probs = weights / np.sum(weights)
            
            extra_num = num_samples - len(sampled_indices)
            extra = np.random.choice(
                remaining, 
                size=extra_num, 
                replace=False, 
            )
            sampled_indices.extend(extra)
    
    
            # 截断到指定数量并去重
            #sampled_indices = list(dict.fromkeys(sampled_indices))[:num_samples]
    logging.info(f"Random_ReSamples:{len(extra)},samples:{len(sampled_indices)}")
            
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

def build_stable_resnet(input_dim, output_dim, unit=256, layNum=4):
    """稳定的后激活残差网络，防止 NaN"""
    def resnet_block(x, units):
        shortcut = x
        y = Dense(units, kernel_initializer='he_normal')(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        
        y = Dense(units, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y) # 注意：这里不加 ReLU，留给 Add 之后
        
        # 如果维度不匹配，用 1x1 卷积（这里用 Dense）对齐
        if shortcut.shape[-1] != units:
            shortcut = Dense(units, kernel_initializer='he_normal')(shortcut)
            shortcut = BatchNormalization()(shortcut) # shortcut 也过 BN
            
        y = Add()([shortcut, y])
        y = ReLU()(y) # 后激活
        return y

    inp = Input(shape=(input_dim,))
    x = Dense(unit, kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(layNum):
        x = resnet_block(x, unit)

    # 输出层
    out = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def build_simple_resnet2(input_dim, output_dim, unit=256, layNum=8):
    """带Dropout预激活残差块，防过拟合，用于IDM参数解码"""
    def resnet_block(x, units, dropout_rate=0.001):
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
    def resnet_block(x, units, dropout_rate=0.001):
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

def build_simple_resnet_regress3(input_dim, output_dim, unit=128, layNum=4):
    """轻量化带Dropout回归网络，直接丢失车辆的slot_multlabel预测"""
    def resnet_block(x, units,dropout_rate=0.001):
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

    out = Dense(output_dim, activation='sigmoid',name='missvehicles_slot_multlabel')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def rmse(y_true, y_pred):
    """自定义rmse损失指标"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ===================== 模型0训练：Wiedemann 99 仿真嵌套端到端训练 =====================

def train_model_mlp_cf(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt):
    """
    MLP+Wiedemann 99 参数端到端训练
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
    output_dim = num_types2 * 6  # FVDM需要6个参数

    # 构建模型
    model = build_simple_resnet2(X_train.shape[1], output_dim, args.unit, args.layNum)
    #model = build_stable_resnet(X_train.shape[1], output_dim, args.unit, args.layNum)

    param_bounds = get_param_bounds(num_types)
    
    # 优化器配置
    steps_per_epoch = max(1, len(X_train) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr, decay_steps=total_steps-10, decay_rate=0.99, staircase=True
    )
    #optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
    #optimizer = Adam(learning_rate=args.lr, clipnorm=1.0)
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5) #现阶段比较好
        


    #
    '''
    
    # 优化器配置：余弦退火
    steps_per_epoch = max(1, len(X_train) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=args.lr * 0.01
    )
    # 梯度裁剪已在train_step中用tf.clip_by_global_norm处理，此处不再重复clipnorm
    #optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
    logging.info(f"LR调度: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, base_lr={args.lr}")

   


    # 替换原有的 AdamW
    initial_lr = args.lr * 50  # SGD 一般需要比 Adam 大 10~100 倍，如 0.005 ~ 0.01
    momentum = 0.9
    weight_decay = 1e-4

    # 余弦退火调度（每轮 epoch 调整）
    cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=args.epochs  # 总 epoch 数作为周期
    )

    cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=args.epochs // 4,  # 每 1/4 总轮次重启一次
        t_mul=2.0, m_mul=0.5, alpha=0.0)

   # optimizer = SGD(learning_rate=cosine_schedule, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    '''



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
            nn_output = tf.clip_by_value(nn_output, clip_value_min=0.01, clip_value_max=0.99) 
            # 1. 检查是否有 NaN 或 Inf (如果有，说明前向传播就炸了)
            #has_nan = tf.reduce_any(tf.math.is_nan(nn_output))
            #has_inf = tf.reduce_any(tf.math.is_inf(nn_output))
            #tf.print("debug:nn_output 包含 NaN:", has_nan, " | 包含 Inf:", has_inf)

            # 2. 打印形状、均值、方差和前3个样本的输出，避免刷屏
            #tf.print("debug:nn_output 形状:", tf.shape(nn_output), " | 均值:", tf.reduce_mean(nn_output)," | 方差:", tf.math.reduce_variance(nn_output))
            #tf.print("debug:前3个样本输出:\n", nn_output[:3])

 
            predicted_times = tf_fvdm_simulation(
                    nn_output, raw_batch, param_bounds, num_types,
                    tf_pos_idx, tf_speed_idx, tf_idx_main, tf_idx_inter, tf_idx_red,
                    dt, args.goffset
                    )
           
            
            # 2. 对预测值取对数
            #safe_predicted_times = tf.maximum(predicted_times, 1e-7)
            #predicted_times = tf.math.log(safe_predicted_times)

            is_finite = tf.reduce_all(tf.math.is_finite(predicted_times))
            # 若存在 NaN/Inf，则用当前 batch 的均值（或固定常数）替代，避免梯度污染
            predicted_times = tf.cond(
                is_finite,
                lambda: predicted_times,
                lambda: tf.zeros_like(predicted_times)  # 或使用 tf.ones * 某个合理值
                )
            # 形状调整
            batch_size = tf.shape(y_batch)[0]
            predicted_times = tf.reshape(predicted_times, [batch_size])
            y_batch = tf.reshape(y_batch, [batch_size])


            loss = tf.reduce_mean(tf.square(predicted_times - y_batch))
            #loss = tf.reduce_mean(tf.keras.losses.huber(y_batch, predicted_times, delta=1.0))

            #tf.print("DEBUG Pred Mean:", tf.reduce_mean(predicted_times), 
            #    "Pred Var:", tf.math.reduce_variance(predicted_times),
            #   "Target Mean:", tf.reduce_mean(y_batch))


            # 梯度裁剪与更新
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)  # 全局范数裁剪更稳定
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
            #clipped_grads = [
            #    tf.clip_by_norm(g, 1.0) if g is not None else g 
            #    for g in grads
            #]
            #optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss,predicted_times

    @tf.function(reduce_retracing=True)
    def val_step(x_batch, y_batch, raw_batch):
        """验证步（TF函数装饰，减少重追踪）"""
        nn_output = model(x_batch, training=False)
        predicted_times = tf_fvdm_simulation(
                nn_output, raw_batch, param_bounds, num_types,
                tf_pos_idx, tf_speed_idx, tf_idx_main, tf_idx_inter, tf_idx_red,
                dt, args.goffset
                )
        # 形状调整
        batch_size = tf.shape(y_batch)[0]
        predicted_times = tf.reshape(predicted_times, [batch_size])
        #对预测值取对数
        #safe_predicted_times = tf.maximum(predicted_times, 1e-7)
        #predicted_times = tf.math.log(safe_predicted_times)
        
        y_batch = tf.reshape(y_batch, [batch_size])
        return predicted_times - y_batch,predicted_times,nn_output



    # 开始训练
    best_val_mae = float('inf')
    best_epoch = 0
    best_save_path = ""

    total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    for epoch in range(args.epochs):
        logging.info(f"===== Epoch {epoch + 1}/{args.epochs} =====")
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_idx = 0

        train_trues = []
        train_preds = []
        for x_batch, y_batch, raw_batch in train_dataset:
            #tf.debugging.assert_all_finite(x_batch, message="x_batch 输入数据包含 NaN 或 Inf!")
            
            batch_idx += 1
            t0 = time.time()
            loss,pred_y = train_step(x_batch, y_batch, raw_batch)
            t1 = time.time()
            
            epoch_loss_avg.update_state(loss)
            rem_batches = total_batches - batch_idx
            logging.info(
                f"Batch {batch_idx}/{total_batches} | Loss: {loss.numpy():.4f} "
                f"| Time: {t1-t0:.2f}s | Remain: {rem_batches}"
            )

            train_trues.append(y_batch.numpy())
            train_preds.append(pred_y.numpy())

      


        # _epoch平均损失
        epoch_loss = epoch_loss_avg.result().numpy()
        logging.info(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

        # 实际时间空间,计算验证指标
        train_preds = np.concatenate([p.flatten() for p in train_preds])
        train_trues = np.concatenate([t.flatten() for t in train_trues])

       
        
        errs= train_preds - train_trues
        train_mse = np.mean(np.square(errs))
        train_rmse = np.sqrt(train_mse)
        train_mae = np.mean(np.abs(errs))

        logging.info(
            f"Trainning Results (Real Time) - MSE: {train_mse:.4f}, "
            f"RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
   
        #--------------------------------------------- 验证逻辑
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logging.info("\n===== 验证阶段开始 =====")
            val_errs = []
            val_loss_metric = tf.keras.metrics.Mean() 
            val_real_params = []     # (batch, num_types, 10)
            val_scene_offset = []    # (batch, 10)
          
         
            for xb, yb, rb in val_dataset:
                errs,pred_y,nn_out = val_step(xb, yb, rb)
                enp = errs.numpy()
                val_errs.append(enp)
                val_loss_metric.update_state(np.mean(np.square(enp)))
                
                # ---- 参数反归一化采集（与仿真内部完全一致）----
                s = np.clip(nn_out.numpy(), 0.01, 0.99).reshape(-1, num_types + 1, 6)
                low = np.asarray(param_bounds, dtype=np.float32)[..., 0]
                high = np.asarray(param_bounds, dtype=np.float32)[..., 1]
                rp = low + s[:, :-1, :] * (high - low)
                rp = np.clip(rp, low, high)
                val_real_params.append(rp)
                val_scene_offset.append(s[:, -1, :])


            # 计算验证指标
            val_errs = np.concatenate([e.flatten() for e in val_errs])
            val_rmse = np.sqrt(np.mean(np.square(val_errs)))
            val_mae = np.mean(np.abs(val_errs))

            logging.info(
                f"Validation Results - RMSE: {val_rmse:.4f}, "
                f"MAE: {val_mae:.4f}, MSE: {val_loss_metric.result().numpy():.4f}\n\n"
            )

            # 保存验证集最小MAE的模型
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch + 1
                best_save_path = (
                    f"{DIR_TMP_MODEL}/fvdm_model0_{RUN_START_TIME}_{args.model}_{args.trainvalmode}_{args.batch_size}_{args.fixdata}"
                    f"_epoch_{best_epoch}_mae_{best_val_mae:.4f}.h5"
                )
                model.save(best_save_path)
                logging.info(f"New best model saved (min val MAE): {best_save_path}")

            # ---- 新最优模型：输出其参数分布统计 ----
            all_real_params = np.concatenate(val_real_params, axis=0)
            all_scene_offset = np.concatenate(val_scene_offset, axis=0)
            report_val_params_stats(all_real_params, all_scene_offset, tag=f"best_epoch{best_epoch}")
     
    

            gc.collect()

    # 模型保存
    make_dir_safe(DIR_TMP_MODEL)
    timestamp = generate_timestamp()
    save_path =  (
                    f"{DIR_TMP_MODEL}/fvdm_model0_{RUN_START_TIME}_{args.model}_{args.trainvalmode}_{args.batch_size}_{args.fixdata}"
                    f"_epoch_{best_epoch}_mae_{best_val_mae:.4f}.h5"
                )
    model.save(save_path)
    logging.info(f"Model 0 saved to: {save_path}")

    return model

# ===================== 模型1训练：直接回归预测消失时间 =====================
# ===================== 模型1训练：直接回归预测消失时间 =====================
from tensorflow.keras.callbacks import LambdaCallback


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

    best_val_mae = [float('inf')]   # 用 list 包裹，便于闭包修改

    def on_epoch_end(epoch,logs=None):
        if epoch % 5 != 1:
            return
        
        _, val_mae, _ = model_vanish_reg.evaluate(val_dataset, verbose=0)
        if val_mae < best_val_mae[0]:
            best_val_mae[0] = val_mae
            make_dir_safe(DIR_TMP_MODEL)
            save_path =  (
                f"{DIR_TMP_MODEL}/model1_reg_{RUN_START_TIME}_{args.epochs}_{args.trainvalmode}_{args.batch_size}_{args.fixdata}"
                f"_mae_{val_mae:.2f}.h5"
            )
            model_vanish_reg.save(save_path)
            logging.info(f"New best reg model saved (min val MAE)_mae_{val_mae:.2f}: {save_path}")

    cb = LambdaCallback(on_epoch_end=on_epoch_end)
   

    # 模型训练
    model_vanish_reg.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        verbose=1,
        callbacks=[cb],
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
    save_path =  (
                    f"{DIR_TMP_MODEL}/model1_reg_{RUN_START_TIME}_{args.epochs}_{args.trainvalmode}_{args.batch_size}_{args.fixdata}"
                    f"_mae_{val_mae:.2f}.h5"
                )
    model_vanish_reg.save(save_path)
    logging.info(f"Model 1 saved to: {save_path}")

    return model_vanish_reg

def make_missvehmultlabel_callback(model, val_dataset):
   
    def on_epoch_end(epoch, logs=None):

        if epoch%20 != 1:
            return 
        logging.info(f"\n Epoch {epoch+1} - 预测丢失车辆slot_multlabel验证指标计算中...")
      
    cb = LambdaCallback(on_epoch_end=on_epoch_end)
    
    return cb

from sklearn.metrics import multilabel_confusion_matrix, classification_report
def train_model_mlp_missonly(X_train, y_miss_train, raw_train, train_dataset, val_dataset, 
                             raw_cols, args, dt, raw_val=None):

    logging.info("启动MLP多标签分类模型训练（预测5个slot中哪些有丢失）") #数据集中移除车辆只能是car_position_1到5，对应位置0到4
    print("正样本总数:\n", np.sum(y_miss_train))
    print("每个类别的正样本数:\n", np.sum(y_miss_train, axis=0))
    print("每个类别正样本比例:\n", np.round(np.mean(y_miss_train, axis=0), 2))
    
    output_dim = y_miss_train.shape[1]  # 5个slot的多标签输出

    # 1. 构建模型（确保输出层 activation='sigmoid'）
    model = build_simple_resnet_regress3(
        X_train.shape[1], output_dim, args.unit, args.layNum)


    # 2. 优化器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr, decay_steps=100, decay_rate=0.99, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # 3. 编译（使用精简且适合多标签的指标）
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.AUC(multi_label=True, name='auc_per_class'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # 4. 回调（假设 make_missvehmultlabel_callback 内部已适配）
    callback = make_missvehmultlabel_callback(model, val_dataset)

    # 5. 训练
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        verbose=1,
        callbacks=[callback],
        shuffle=True
    )

    # 训练集评估
    train_loss, train_bin_acc, train_auc_per_class, train_precision, train_recall = model.evaluate(train_dataset, verbose=0)
    val_loss, val_bin_acc, val_auc_per_class, val_precision, val_recall = model.evaluate(val_dataset, verbose=0)




    # 计算 F1
    train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-7)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

    logging.info(
        f"Training   - Loss: {train_loss:.4f}, BinAcc: {train_bin_acc:.4f}, "
        f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}"
    )
    logging.info(
        f"Validation - Loss: {val_loss:.4f}, BinAcc: {val_bin_acc:.4f}, "
        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
    )


  

    


    # 获取验证集的预测概率和标签
    y_true_val = []
    y_pred_prob_val = []
    for xb, yb in val_dataset:
        y_true_val.append(yb.numpy())
        y_pred_prob_val.append(model.predict_on_batch(xb))

    y_true_val = np.concatenate(y_true_val, axis=0)      # shape: (n_samples, 5)
    y_pred_prob_val = np.concatenate(y_pred_prob_val, axis=0)

    # 将概率转换为 0/1 预测（阈值 0.5）
    y_pred_val = (y_pred_prob_val >= 0.5).astype(int)

    # 计算每个类别的混淆矩阵
    mcm = multilabel_confusion_matrix(y_true_val, y_pred_val)  # shape (20, 2, 2)

    # 打印每个类别的混淆矩阵
    for i in range(5):
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"Class {i:2d} | TP: {tp:4d}  FP: {fp:4d}  FN: {fn:4d}  TN: {tn:4d}")

    print("概率值 (前3样本, 前5类):\n", np.round(y_pred_prob_val[:10,], 2))
    print("二值预测 (前3样本, 前5类):\n", y_pred_val[:10,])
    print("真实标签 (前3样本, 前5类):\n", y_true_val[:10, ])

    # 7. 保存模型
    make_dir_safe(DIR_TMP_MODEL)
    timestamp = generate_timestamp()
    model_path = f"{DIR_TMP_MODEL}/model_missvehmultlabel_{timestamp}.h5"
    model.save(model_path)
    logging.info(f"多标签模型已保存至: {model_path}")

    return model
# ===================== 数据修补函数 =====================
def fix_missing_data(df, fix_type):
    """
    统一的数据修补入口函数
    :param df: 原始数据
    :param fix_type: 修补类型 0-不修补 1-直接补原始数据 2-前后车偏移补
    :return: 修补后的数据
    """
    if fix_type == 0:
        lost_count = len(df.index[df['lost'] > 0])
        logging.info(f"不修补数据，采样样本数: {len(df)}, 缺失样本数: {lost_count}")
        return df
    
    # 获取丢失样本索引
    lost_indices = df.index[df['lost'] > 0].tolist()
    logging.info(f"开始修补数据，共 {len(lost_indices)} 个缺失样本，修补类型: {fix_type}")
   
    # 方法1：直接前车-5
    if fix_type == 1:
        df_fixed = df
        for idx in lost_indices:
            removeVehCol = df.at[idx, 'removed_vehicles'] #car_pos_1~5
            removeIntidx =   df.at[idx,'removed_vehicles_intidx'] # 1~5
            speedlist = [df_fixed.at[idx, f'car_speed_{j}'] for j in range(0,20)]
            poslist = [df_fixed.at[idx, f'car_position_{j}'] for j in range(0,20)]
            poslist.insert(removeIntidx, poslist[removeIntidx]-5) # 例如插入原来的removeIntidx = 1的位置，那样原来car_position_1 变为2
            speedlist.insert(removeIntidx, speedlist[removeIntidx])# speedlist长度会增加到21吗
            for k in range(0,20):
                car_pos_col = f'car_position_{k}'
                car_speed_col = f'car_speed_{k}'
                df_fixed.at[idx,car_pos_col] = poslist[k]
                df_fixed.at[idx,car_speed_col] = speedlist[k]

    elif fix_type == 2:
        df_fixed = df
        for idx in lost_indices:
            removeVehCol = df.at[idx, 'removed_vehicles'] #car_pos_1~5
            removeIntidx =   df.at[idx,'removed_vehicles_intidx'] # 1~5
            speedlist = [df_fixed.at[idx, f'car_speed_{j}'] for j in range(0,20)]
            poslist = [df_fixed.at[idx, f'car_position_{j}'] for j in range(0,20)]
            posFront = poslist[removeIntidx]
            posBeh = poslist[removeIntidx-1]
            posValueinsert = (posFront+posBeh)/2
            poslist.insert(removeIntidx, posValueinsert) # 例如插入原来的removeIntidx = 1的位置，那样原来car_position_1 变为2
            speedlist.insert(removeIntidx, speedlist[removeIntidx])# speedlist长度会增加到21吗
            for k in range(0,20):
                car_pos_col = f'car_position_{k}'
                car_speed_col = f'car_speed_{k}'
                df_fixed.at[idx,car_pos_col] = poslist[k]
                df_fixed.at[idx,car_speed_col] = speedlist[k]

    logging.info(f"数据修补完成，共处理 {len(lost_indices)} 个缺失样本")
    return df


def compute_gap_and_dv(row):
    #数据中，已经假定car_position_0,位置最小，car_position_19位置最大。随着index增大，位置增大。最大只有20辆车，car_position_19假定最接近该车道消失线
    for i in range(19):
        pos_col = f"car_position_{i}"
        speed_col = f"car_speed_{i}"
        front_pos_col = f"car_position_{i+1}"
        front_speed_col = f"car_speed_{i+1}"
        
        pos = row[pos_col]
        front_pos = row[front_pos_col]
        if pos == -1:
            gap_col = f"car_gap_{i}"
            dv_col = f"car_dv_{i}"
            row[gap_col] = -1
            row[dv_col] = -1

        elif front_pos == -1:
            gap_col = f"car_gap_{i}"
            dv_col = f"car_dv_{i}"
            row[gap_col] = row['intersection_pos'] - pos
            row[dv_col] = row[speed_col]

        else:
           gap_col = f"car_gap_{i}"
           dv_col = f"car_dv_{i}"
           row[gap_col] = front_pos - pos
           row[dv_col] =  row[front_speed_col] - row[speed_col]

    if row["car_position_19"] == -1:      
        row["car_gap_19"]   = -1
        row["car_dv_19"]   =  -1
    else:
        row["car_gap_19"]   = row['intersection_pos'] - row["car_position_19"]
        row["car_dv_19"]   =  row["car_speed_19"]
    return row


#制作纯预测，模型从loadmodel加载后，直接调用vanish_prediction_log_real_time函数即可得到预测的消失时间
def vanish_prediction_realtime_cf(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger):
        X_vanish = df_fixed[feature_cols].values.astype(np.float32)
        y_vanish = (df_fixed['time_to_vanish'].values / 30.0).astype(np.float32)#注意这里训练样本的y/30,以秒为单位，后面还有log化
        #y_vanish_log = np.log(y_vanish) #注意y对数化了---------------------------------------------------------------------这里y对数化了
        raw_data_for_sim = df_fixed[raw_cols].values.astype(np.float32)
     
        mask = np.isfinite(X_vanish).all(axis=1)

        X_vanish, y_vanish, raw_data_for_sim = X_vanish[mask], y_vanish[mask], raw_data_for_sim[mask]

        test_dataset = tf.data.Dataset.from_tensor_slices((X_vanish, y_vanish, raw_data_for_sim))
        #根据上下文，用mlpw99cfModel预测test_dataset的消失时间,注意mlpw99cfModel里面有自定义的train_step和val_step函数       
        # 创建预测数据集
        test_dataset = tf.data.Dataset.from_tensor_slices((X_vanish, y_vanish, raw_data_for_sim))
        test_dataset = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)    

        vanish_predictions = []
        vanish_actual = []
        vanish_scaled = []  # 新增：收集缩放参数输出，用于参数统计
    
        for x_batch, y_batch, raw_batch in test_dataset:
            # 使用mlpw99cfModel的验证步逻辑进行预测
            nn_output = modelVanishPredict(x_batch, training=False)
            nn_output = tf.clip_by_value(nn_output, clip_value_min=0.01, clip_value_max=0.99)
            
            # 准备仿真参数
            num_types = args.num_types
            param_bounds = get_param_bounds(num_types)
            
            # 获取索引
            pos_idx_list = [i for i, c in enumerate(raw_cols) if c.startswith("car_position_")]
            speed_idx_list = [i for i, c in enumerate(raw_cols) if c.startswith("car_speed_")]
            idx_main_car = raw_cols.index("main_car_position")
            idx_inter = raw_cols.index("intersection_pos")
            idx_red = raw_cols.index("redLightRemainingTime")
            
            tf_pos_idx = tf.constant(pos_idx_list, dtype=tf.int32)
            tf_speed_idx = tf.constant(speed_idx_list, dtype=tf.int32)
            tf_idx_main = tf.constant(idx_main_car, dtype=tf.int32)
            tf_idx_inter = tf.constant(idx_inter, dtype=tf.int32)
            tf_idx_red = tf.constant(idx_red, dtype=tf.int32)
            
            # 执行Wiedemann仿真
            predicted_times = tf_fvdm_simulation(
                nn_output, raw_batch, param_bounds, num_types,
                tf_pos_idx, tf_speed_idx, tf_idx_main, tf_idx_inter, tf_idx_red,
                args.dt, args.goffset
            )
            
            # 处理预测结果
            batch_size = tf.shape(y_batch)[0]
            predicted_times = tf.reshape(predicted_times, [batch_size])
            predicted_times = tf.clip_by_value(predicted_times, 1e-7, 1e7)
            
            vanish_predictions.append(predicted_times.numpy())
            vanish_actual.append(y_batch.numpy())
            vanish_scaled.append(nn_output.numpy())  # 新增：收集缩放输出
        
        # 计算评估指标
        vanish_preds = np.concatenate(vanish_predictions)
        vanish_true = np.concatenate(vanish_actual)
        vanish_preds_real = vanish_preds
        vanish_true_real = vanish_true
        
        mse_real = np.mean(np.square(vanish_preds_real - vanish_true_real))
        rmse_real = np.sqrt(mse_real)
        mae_real = np.mean(np.abs(vanish_preds_real - vanish_true_real))        
       
        logger.info(f"预测结果 (实际时间) - MSE: {mse_real:.4f}, RMSE: {rmse_real:.4f}, MAE: {mae_real:.4f}")
                # ===================== 参数分布统计（不训练，仅统计）=====================
        num_types = args.num_types
        param_bounds = get_param_bounds(num_types)
        low  = np.asarray(param_bounds, dtype=np.float32)[..., 0]   # (num_types, 10)
        high = np.asarray(param_bounds, dtype=np.float32)[..., 1]

        all_scaled = np.concatenate(vanish_scaled, axis=0)         # (N, num_types+1, 10)
        scaled0 = all_scaled.reshape(-1, num_types + 1, 6)
        scaled_params = scaled0[:, :-1, :]      # (N, num_types, 10) -> CC0~CC9
        scene_offset   = scaled0[:, -1, :]      # (N, 10) -> 全局偏移

        real_params = np.clip(low + scaled_params * (high - low), low, high)

        df_stats, long_df = report_val_params_stats(
            real_params, scene_offset, tag="realtime_cf"
        )
        plot_param_boxplots(long_df, DIR_EVAL_MODEL0, tag="realtime_cf")
        

def vanish_prediction_realtime_reg(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger):
        X_vanish = df_fixed[feature_cols].values.astype(np.float32)
        y_vanish = (df_fixed['time_to_vanish'].values / 30.0).astype(np.float32)#注意这里训练样本的y/30,以秒为单位，后面还有log化
        #y_vanish_log = np.log(y_vanish) #注意y对数化了---------------------------------------------------------------------这里y对数化了
        raw_data_for_sim = df_fixed[raw_cols].values.astype(np.float32)
     
        mask = np.isfinite(X_vanish).all(axis=1)

        X_vanish, y_vanish, raw_data_for_sim = X_vanish[mask], y_vanish[mask], raw_data_for_sim[mask]

        val_dataset = tf.data.Dataset.from_tensor_slices((X_vanish, y_vanish))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        val_loss, val_mae, val_rmse = modelVanishPredict.evaluate(val_dataset, verbose=0)
     
        logger.info(
            f"Validation Set - MAE: {val_mae:.4f}, MSE: {val_loss:.4f}, RMSE: {val_rmse:.4f}"
        )

        

     
# ===================== 主函数 =====================
#import swifter
def main(args):
    """主训练流程"""
    # 初始化日志
    logger = setup_logger(args)
    logger.info(f"训练启动，参数配置: {args}")
    
    # ===================== 1. 数据加载与预处理 =====================
    logger.info(f"从 {args.csv_path} 加载数据...")
    df1 = pd.read_csv(args.csv_path).dropna()
    df1['lost'] = 0
    df1['removed_vehicles'] = [[] for _ in range(len(df1))]
    df1.rename(columns={
        'car_position': 'main_car_position',
        'car_speed': 'main_car_speed'
    }, inplace=True)
    df1['intersection_pos'] = df1['lane'].map(LANE_POS_MAP)

    ##########
    #修补一个bug,
    #1.所有的car_position_0到19，以及main_car_position,开始都认为是起点距离车道线起始线的距离
    #2.事实上，所有的car_position_0到19，以及main_car_position都是距离车道终点的距离
    #3.修补后，所有的car_position_0到19，以及main_car_position都是起点距离车道线起始线的距离
    if 1:
            logger.info(f"修补一个数据bug:所有的car_position是车辆距离车道线起始线的距离")
            inter_pos = df1['intersection_pos']
            df1['main_car_position'] = inter_pos - df1['main_car_position']

            for jj in range(20):
                col_name = f"car_position_{jj}"
                pos = df1[col_name]
                # 只在 pos > 0 时用 inter_pos - pos，其余（-1/NaN）保持原值
                df1[col_name] = pos.where(pos <= 0, inter_pos - pos)
    
    # 添加路口位置列,以及其他列
    miss_outdim =5 #根据数据集，移除车辆位置只能是car_position_1到5，对应位置0到4
    #df1['intersection_pos'] = df1['lane'].map(LANE_POS_MAP)
    df1['queued_vehicles']  = df1.apply(count_queued_vehicles, axis=1)
    df1['removed_vehicles_intidx']= [[] for _ in range(len(df1))]
    df1['removed_vehicles_multlabel']=  [[0]*miss_outdim for _ in range(len(df1))]
    for i in range(20):
        df1[f"car_gap_{i}"] = -1.0
        df1[f"car_dv_{i}"] = -1.0

    #for i in range(len(df1)):   
    #    print(i,len(df1))
    #    row = df1.iloc[i]
    #    row1 = compute_gap_and_dv(row)
    #    df1.iloc[i] = row1    
    df1 = df1.apply(compute_gap_and_dv, axis=1)    
    
    #df1 = df1.swifter.apply(compute_gap_and_dv, axis=1)    
    # ===================== 2. 生成缺失数据样本 =====================
    # ===================== 3. 样本合并与过滤 =====================
    # 选择训练验证模式
    if args.trainvalmode == 0:
        df_all = df1
    else:
        logger.info("生成缺失车辆样本...")
        # 生成不同数量丢失车辆的样本
        #print(df1.columns)
        df_missveh1 =  genSamplesRemovingVehicleWithOneSlot(df1)
       
       
        # 合并缺失样本
        #df_step2_missveh2 = pd.concat([
        #    df_missveh2_rn1, df_missveh2_rn2, 
        #    df_missveh2_rn3, df_missveh2_rn4
        #], ignore_index=True)
        #df_all = pd.concat([df1, df_missveh1], ignore_index=True)
        df_all = df_missveh1 
    
    
  

    # 样本过滤（排队异常/丢失过多/消失时间过长）
    logger.info("开始样本过滤...")
  

    
    # 过滤条件
    cond_queued = (df_all['queued_vehicles'] > 5) | (df_all['queued_vehicles'] == 0)
    cond_lost = df_all['lost'] >= 3
    cond_vanish = df_all['time_to_vanish'] > 30 * 30  # 还原原始时间
    cond_redTime = df_all['time_to_vanish'] < df_all['redLightRemainingTime']
    # 执行过滤
    before_count = len(df_all)
    df_all = df_all[~(cond_queued | cond_lost | cond_vanish | cond_redTime)].reset_index(drop=True)
    after_count = len(df_all)
    logger.info(
        f"样本过滤完成 - 过滤前: {before_count} 个, "
        f"过滤后: {after_count} 个, "
        f"删除: {before_count - after_count} 个"
    )

 # ===================== 4. 样本抽样 =====================
    if args.nC > len(df_all) or args.nC <= 0:
        logger.warning(
            f"目标样本数 {args.nC} 超过可用样本数 {len(df_all)} 或不合法，"
            f"将使用全部样本进行训练"
        )
        args.nC = len(df_all)
        df_sampled = df_all.copy()
    else:
        logger.info(f"开始样本抽样，样本目标数量: {args.nC}")
        sampled_indices = get_sample_indices(df_all, args.nC)
        df_sampled = df_all.iloc[sampled_indices].reset_index(drop=True)
        logger.info(
            f"抽样完成 - 最终样本数: {len(df_sampled)}"
        )


    # ===================== 5. 数据修补 =====================
    # ===================== 5. 数据修补 =====================

 
    df_fixed = df_sampled.copy()
    
    # ===================== 6. 数据集构建 =====================
    # 特征列和原始数据列
    # ===================== 6. 数据集构建 =====================
    # 特征列和原始数据列
    feature_cols = [f"car_position_{i}" for i in range(20)] + [f"car_speed_{i}" for i in range(20)]
    
    feature_cols.append('intersection_pos')
    feature_cols.append('lane')
    feature_cols.append('main_car_position')
    feature_cols.append('main_car_speed')
    feature_cols.append('queued_vehicles')
    feature_cols.append('redLightRemainingTime')


    feature_colsTmp = [f"car_gap_{i}" for i in range(20)] + [f"car_dv_{i}" for i in range(20)]
    #feature_cols.extend(feature_colsTmp)
    

    raw_cols = feature_cols
    # 数据转换
    X = df_fixed[feature_cols].values.astype(np.float32)
    y = (df_fixed['time_to_vanish'].values / 30.0).astype(np.float32)#注意这里训练样本的y/30,以秒为单位，后面还有log化
    raw_data_for_sim = df_fixed[raw_cols].values.astype(np.float32)

    #y = np.log(y) #注意y对数化了---------------------------------------------------------------------这里y对数化了
    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
    X, y, raw_data_for_sim, 
    test_size=args.test_size, 
    random_state=42
    )
    
    if args.trainvalmode == 1:
        XMiss = X
        yMissmultlabel = np.stack(df_fixed['removed_vehicles_multlabel'].values).astype(np.float32)
        mask = np.isfinite(X).all(axis=1) 
        print(f"清理完成：删除了 {(~mask).sum()} 条包含 NaN/Inf 的脏数据")
    
        # 3. 应用掩码过滤（同步过滤三个数组，保持维度一致）
        XMiss, yMissmultlabel, raw_data_for_sim = XMiss[mask], yMissmultlabel[mask], raw_data_for_sim[mask]

        # 划分训练验证集
        Xmiss_train, Xmiss_val, ymiss_train, ymiss_val, rawmiss_train, rawmiss_val = train_test_split(
            XMiss, yMissmultlabel, raw_data_for_sim, 
            test_size=args.test_size, 
            random_state=12
        )

        logger.info(
            f"slotLost数据集构建完成 - 训练集: {len(Xmiss_train)} 样本, "
            f"验证集: {len(Xmiss_val)} 样本"
        )
        
        
    if args.trainvalmode == 0:
        # 1. 生成掩码：X 的所有特征有限 且 y 有限
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)

        # 2. 打印清理结果
        print(f"清理完成：删除了 {(~mask).sum()} 条包含 NaN/Inf 的脏数据")

        # 3. 应用掩码过滤（同步过滤三个数组，保持维度一致）
        X, y, raw_data_for_sim = X[mask], y[mask], raw_data_for_sim[mask]

        # 划分训练验证集
        X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
            X, y, raw_data_for_sim, 
            test_size=args.test_size, 
            random_state=42
        )

        logger.info(
            f"vanishTime数据集构建完成 - 训练集: {len(X_train)} 样本, "
            f"验证集: {len(X_val)} 样本"
        )

    # ===================== 7. 模型训练 =====================
    dt = args.dt or DEFAULT_DT
    logger.info(f"开始模型训练，仿真时间步长: {dt}")

    if args.model == 0:
        # 模型0：MLP+CF端到端训练
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, raw_train))
        # 保持每个batch形状一致以减少tf.function retracing并降低编译开销
        #train_dataset = train_dataset.cache().batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, raw_val))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

       

        train_model_mlp_cf(
            X_train, y_train, raw_train,
            train_dataset, val_dataset, raw_cols,
            args, dt
        )

    elif args.model == 1:
        # 模型1：MLP直接回归
        args.batch_size = X_train.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_model_mlp_reg(
            X_train, y_train, raw_train,
            train_dataset, val_dataset, raw_cols,
            args, dt, raw_val=raw_val
        )
    elif args.model == 2:
        # 模型2：MLP+预测丢失slot的multlabel
        #Xmiss_train, Xmiss_val, ymiss_train, ymiss_val, rawmiss_train, rawmiss_val 
        args.batch_size =Xmiss_train.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices((Xmiss_train, ymiss_train))
        train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((Xmiss_val, ymiss_val))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_model_mlp_missonly(
            Xmiss_train, ymiss_train, rawmiss_train,
            train_dataset, val_dataset, raw_cols,
            args, dt, raw_val=rawmiss_val
        )
    elif args.model == 3 or args.model == 4:
        #只预测，不训练，最终实现slot和vanish的预测，slot预测使用训练好的mlp_multlabel模型，vanish预测使用训练好的mlp_cf模型
        
        if args.model == 3:
            model_path = f"./tmpModes/fvdm_model0_20260831_104110_0_0_300_0_epoch_31_mae_1.2668.h5"
            mlpcfModel = load_model(model_path)
            modelVanishPredict = mlpcfModel
            logger.info(f"加载CF模型成功")


        if args.model == 4:
            model_pathT = f"./tmpModes/model1_reg_20260826_130038_1000_0_1172_0_mae_1.30.h5"
            mlpw99regModel = load_model(model_pathT,custom_objects={'rmse': rmse})
            modelVanishPredict = mlpw99regModel
            logger.info(f"加载回归模型成功")
                       

        if args.fixdata == 1 or args.fixdata == 2:#用原始位置数据不需要识别
            df_fixed = fix_missing_data(df_sampled, args.fixdata)
            logger.info("args.fixdata == 1 or 2, 启用原始数据修补，使用原始数据进行slot位置修补和数据修补")
            if args.model == 3:
                vanish_prediction_realtime_cf(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger)
            if args.model == 4:
                vanish_prediction_realtime_reg(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger)
            
        elif args.fixdata == 3:
            # 1. 首先使用missModel预测哪些slot有丢失车辆
            # 2. 根据预测结果确定需要修补的位置
            # 3. 使用前后车偏移方法进行数据修补
            model_path = f"./tmpModes/model_missvehmultlabel_20260903_143007.h5"
            missModel = load_model(model_path)

            
            X_miss_input = df_sampled[feature_cols].values.astype(np.float32)
            yMissmultlabel = np.stack(df_sampled['removed_vehicles_multlabel'].values).astype(np.float32)
            mask = np.isfinite(X_miss_input).all(axis=1)
            X_miss_input,yMissmultlabel = X_miss_input[mask],yMissmultlabel[mask]   
            
            # 预测丢失slot
            miss_pred_dataset = tf.data.Dataset.from_tensor_slices((X_miss_input,yMissmultlabel))
            miss_pred_dataset = miss_pred_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

           
            miss_pred_probs = []
            miss_true_labels = []
            for x_batch,y_batch in miss_pred_dataset:
                pred_prob = missModel.predict_on_batch(x_batch)
                miss_pred_probs.append(np.asarray(pred_prob))
                miss_true_labels.append(y_batch.numpy())
            
            miss_pred_probs = np.concatenate(miss_pred_probs, axis=0)
            miss_true_labels = np.concatenate(miss_true_labels, axis=0)
            miss_pred_binary = (miss_pred_probs >= 0.5).astype(int)
            
            # 根据预测结果修补数据
            df_fixed = df_sampled[mask].copy().reset_index(drop=True)
            for idx, pred_labels in enumerate(miss_pred_binary):
                # 对于每个预测为1的slot，查找并修补数据，只会有一个
                #注意这里根据模型和数据，slot_i 智能是0,1,2,3,4,5其中之一.而且只会有一个为1. 
                for slot_i in range(pred_labels.shape[0]):
                    if pred_labels[slot_i] == 1:
                        #slot 0,1,2,3,4 对应car_position0前之间
                        # 前提car_position0~19已经按照从小到大排列                  
                       
                        speedlist = [df_fixed.at[idx, f'car_speed_{j}'] for j in range(0,20)]
                        poslist = [df_fixed.at[idx, f'car_position_{j}'] for j in range(0,20)]
                        poslist.insert(slot_i, poslist[slot_i]+8) # slot_i如果为0，就插到index为0的位置，也就是slot_i.
                        speedlist.insert(slot_i, speedlist[slot_i])
                        for k in range(0,20):
                            car_pos_col = f'car_position_{k}'
                            car_speed_col = f'car_speed_{k}'
                            df_fixed.at[idx,car_pos_col] = poslist[k]
                            df_fixed.at[idx,car_speed_col] = speedlist[k]
                        break #对于每个预测为1的slot，查找并修补数据，只会有一个
    
            
            logger.info(f"模型预测修补完成，处理了 {len(df_fixed)} 个样本")
            
            # 然后使用修补后的数据进行消失时间预测
            if args.model == 3:
                vanish_prediction_realtime_cf(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger)
            if args.model == 4:
                vanish_prediction_realtime_reg(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger)
            
           

        else:
            logger.info("args.fixdata 参数不合法 or args.fixdata == 0, 不进行数据修补，直接使用原始数据进行预测") 
            df_fixed = df_sampled  
            if args.model == 3:
                vanish_prediction_realtime_cf(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger)
            if args.model == 4:
                vanish_prediction_realtime_reg(modelVanishPredict,df_fixed,feature_cols,raw_cols, args,logger)
            
        

            


    # 清理内存
    force_clean_all_memory()
    logger.info("训练流程完成")

# ===================== 入口执行 =====================
if __name__ == "__main__":
    # 启动TF性能分析
    #log_dir = "./profiler_records"
    #os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    #tf.profiler.experimental.server.start(6009)
    #tf.profiler.experimental.start(log_dir)

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="使用Keras和交通仿真进行端到端模型训练")
    parser.add_argument('--csv_path', type=str, default='trainsamples_lane_5_6_7.csv', 
                        help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.9, help='验证集比例')
    parser.add_argument('--num_types', type=int, default=4, help='车辆类别数')
    parser.add_argument('--unit', type=int, default=128, help='ResNet隐藏层单元数')
    parser.add_argument('--layNum', type=int, default=8, help='ResNet块数量')
    parser.add_argument('--log_path', type=str, default='training_log.log', help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用Debug级别的日志信息')
    parser.add_argument('--dt', type=float, default=DEFAULT_DT, help='仿真时间步长')
    parser.add_argument('--nC', type=int, default=1000, help='抽样样本数量')
    parser.add_argument('--model', type=int, default=0, help='0(MLP+CF),1(MLP+Regress),2(MLP+预测丢失slot的multlabel),3(丢失slot+vanish时间联合)')
    parser.add_argument('--fixdata', type=int, default=0, help='0(不修补),1(原始数据补),2(前后车偏移补),3 模型预测修补，前后车偏移补')
    parser.add_argument('--goffset', type=int, default=1, help='仿真全局偏移参数开关')
    parser.add_argument('--trainvalmode', type=int, default=0, help='0(无丢失,只有vanish),1(有丢失,有misss数据)')
   

    args = parser.parse_args()
    main(args)

    # 停止性能分析
    #tf.profiler.experimental.stop()
    #tf.profiler.experimental.server.stop()
