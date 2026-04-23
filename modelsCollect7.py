'''

1.实现如果有车辆丢失，模型自己加入丢失车辆进行仿真，保证每次仿真

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
from tensorflow.keras.optimizers import Adam,Adadelta,SGD,Adamax, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
from jax import tree_util
import time
import os
import random
from pyGameBraxInterface4delta import IDMParams, EnvState,initial_env_state_pure, rollout_pure,rollout_while
from sklearn.cluster import KMeans
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
from tensorflow import keras
import gc
import jax.numpy as jnp
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


from jax.lib import xla_client
def force_clean_all_memory():
    """硬核清理：JAX+Keras+系统内存，彻底释放无效占用"""
    # 1. 清理Keras计算图
    keras.backend.clear_session()
    # 3. 强制解绑所有JAX张量引用（核心！解决释放无效）
    jnp.zeros((1,)).delete()
    # 4. 手动触发系统垃圾回收
    gc.collect()
    gc.collect()  # 两次回收更彻底
    
    """
    JAX 0.3.25 专属强制清理
    清理：JAX缓存 + TensorFlow显存 + 系统内存
    """
 
    # 2. 清理TensorFlow显存（TF和JAX共用GPU，必须清！）
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

       
    
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
#############################################################################################核心变量
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

# 2. Keras 模型定义 (优化)
def build_simple_resnet2(input_dim, output_dim, unit=256, layNum=8):
    """构建一个优化的ResNet模型，调整了BN和ReLU的位置"""
    """
    优化版 ResNet 回归模型
    改进点：
    1. 加入 Dropout 防止过拟合
    2. 优化残差块结构（Pre-Activation，训练更稳）
    3. 降低神经元 + 减少层数，适配小数据集
    4. 每一层都做归一化 + 正则化
    """

    def resnet_block(x, units, dropout_rate=0):
        shortcut = x

        # 🔥 优化版残差块（BN -> ReLU -> Dense，训练更稳定）
        y = BatchNormalization()(x)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        
        # ✅ 加入 Dropout，核心防过拟合
        y = Dropout(dropout_rate)(y)

        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        y = Dropout(dropout_rate)(y)

        # 维度匹配
        if shortcut.shape[-1] != units:
            shortcut = Dense(units)(shortcut)

        # 残差连接
        y = Add()([shortcut, y])
        return y

    # 输入层
    inp = Input(shape=(input_dim,))
    
    # 初始层
    x = Dense(unit, kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = Dropout(0.2)(x)  # 初始层也加 Dropout

    # 残差层（层数减少，防止过拟合）
    for _ in range(layNum):
        x = resnet_block(x, unit, dropout_rate=0.2)

    # 输出层使用sigmoid激活，将输出限制在(0, 1)范围，便于后续缩放
    out = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    return model
##回归模型1
def build_simple_resnet_regress(input_dim, output_dim, unit=256, layNum=8):


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


 
    out = Dense(1, activation='linear', name='vanish_time')(x)  # 线性激活，输出时间值
    model = Model(inputs=inp, outputs=out)
    return model

###############################################################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add, Dropout

def build_simple_resnet_regress2(input_dim, output_dim, unit=128, layNum=4):
    """
    优化版 ResNet 回归模型
    改进点：
    1. 加入 Dropout 防止过拟合
    2. 优化残差块结构（Pre-Activation，训练更稳）
    3. 降低神经元 + 减少层数，适配小数据集
    4. 每一层都做归一化 + 正则化
    """

    def resnet_block(x, units, dropout_rate=0.2):
        shortcut = x

        # 🔥 优化版残差块（BN -> ReLU -> Dense，训练更稳定）
        y = BatchNormalization()(x)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        
        # ✅ 加入 Dropout，核心防过拟合
        y = Dropout(dropout_rate)(y)

        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Dense(units, kernel_initializer='he_normal')(y)
        y = Dropout(dropout_rate)(y)

        # 维度匹配
        if shortcut.shape[-1] != units:
            shortcut = Dense(units)(shortcut)

        # 残差连接
        y = Add()([shortcut, y])
        return y

    # 输入层
    inp = Input(shape=(input_dim,))
    
    # 初始层
    x = Dense(unit, kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = Dropout(0.2)(x)  # 初始层也加 Dropout

    # 残差层（层数减少，防止过拟合）
    for _ in range(layNum):
        x = resnet_block(x, unit, dropout_rate=0.2)

    # 输出层（回归任务）
    out = Dense(1, activation='linear', name='vanish_time')(x)

    model = Model(inputs=inp, outputs=out)
    return model




def run_batch_simulation2(nn_output_batch, raw_data_batch, param_bounds, num_types, columns=None,dt=0.5,args=None):
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
    
        # 1. 参数解码numType =4+1;4为4类IDM参数，1为全局偏移量
        scaled_params0 = jnp.asarray(nn_output).reshape((num_types+1, 6))
        scaled_params = scaled_params0[:-1, :]  # 去掉最后一行全局偏移量
        param_bounds = jnp.asarray(param_bounds)
        low = param_bounds[:, :, 0]
        high = param_bounds[:, :, 1]
        real_params = low + scaled_params * (high - low)

        scence_offset = scaled_params0[-1, :]  # 最后一行全局场景偏移量
        redlighttime_offset,\
        redlightpos2vanishpos_offset,\
        vehpos_offset,\
        redlightpos_offset,\
        vanishtime_offset,\
        distgap_offset = scence_offset
        
        #######################################################################
        ################################################################################################核心变换
        #######################################################################
        if args.goffset == 1:
            redlighttime_offset = (-1.0+redlighttime_offset*2.0)*2.0
            redlightpos2vanishpos_offset = redlightpos2vanishpos_offset*20
            vehpos_offset = (-1.0+vehpos_offset*2.0)*1
            redlightpos_offset = redlightpos_offset*2
            #vanishtime_offset = (-1.0+vanishtime_offset*2.0)*1.0
            vanishtime_offset = (-1.0+vanishtime_offset*2.0)*0.5 #结果变好的核心改变
            distgap_offset = (-1.0+distgap_offset*2.0)*1
        else:
            redlighttime_offset = (-1.0+redlighttime_offset*2.0)*0.0
            redlightpos2vanishpos_offset = redlightpos2vanishpos_offset*0
            vehpos_offset = (-1.0+vehpos_offset*2.0)*0
            redlightpos_offset = redlightpos_offset*0
            #vanishtime_offset = (-1.0+vanishtime_offset*2.0)*0.0
            vanishtime_offset = (-1.0+vanishtime_offset*2.0)*0.0 #结果变好的核心改变
            distgap_offset = (-1.0+distgap_offset*2.0)*0
            
        scence_offset = redlighttime_offset,\
                            redlightpos2vanishpos_offset,\
                            vehpos_offset,\
                            redlightpos_offset,\
                            vanishtime_offset,\
                            distgap_offset
        
        def insert_constants(params):
            return jnp.concatenate([params[:5], jnp.array([4.0, 4.0]), params[5:]])
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
                #######################################################################
                ###核心变换，注意这里车辆位置为intersection_pos - row_dict[pos_col，而不是距离距离车道终点的位置
                #######################################################################
               
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
   
        state = initial_env_state_pure(num_vehicles=num_cars, dt=dt, init_pos=init_pos, init_vel=init_speed, params=params, \
                                        red_light_pos=red_light_pos, red_light_duration=red_light_duration, scence_offset=scence_offset)
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
        main_car_vanish_time = final_state.time_to_vanish[main_car_rank]+final_state.vanishtime_offset
        #tf.print("final_state.step_count:", final_state.step_count)
        return main_car_vanish_time#time_to_vanish计算时已经是秒了(step*0.1)

    results = jax.vmap(get_time_to_vanish2)(statesAll)
    del statesAll, idm_params_arr, scaled_params0, real_params
    end_time = time.time()
    #tf.print(f"JAX SIM Batch sim time_to_vanish:\n {results}")
    tf.print(f"JAX SIM Batch simulation time (s): {end_time - start_time:.2f}")
    return tf.convert_to_tensor(results, dtype=tf.float32)
            
    


#------------------------------------------------------------------------------------------
from modelsLostReg import genDatasetLost
from modelsLostReg import genSamplesByRandomRemovingVehicle
def main(args):
    test_size = args.test_size
    batch_size = args.batch_size
    
    # ==============================================
    # 🚕：第一步，读入原始数据，加入lost和removed_vehicles列（无数据）,并生成训练数据集和验证数据集合
    # ==============================================
    

    setup_logger(args.log_path, args.debug)
    logging.info(f"启动训练，参数: {args}")
    logging.info(f"从 {args.csv_path} 加载数据...")
 
    df1 = pd.read_csv(args.csv_path).dropna()
    df1['lost'] = 0
    df1['removed_vehicles'] = None   
    df1.rename(columns={'car_position': 'main_car_position'}, inplace=True)
    df1.rename(columns={'car_speed': 'main_car_speed'}, inplace=True)

    X_train1, X_val1, y_train1, y_val1, \
        raw_train1, raw_val1, train_dataset1, val_dataset1,\
            raw_cols1     = genDatasetLost(df1, test_size, batch_size)#简单生成数据，没有去掉车
    df_step1 = df1.copy()

    # ==============================================
    # 🚕：第二步，随机去掉部分车辆，生成缺失数据样本，加入lost和removed_vehicles列,并生成训练数据集和验证数据集合
    # ==============================================

    print(f"{'-'*100}")
    ## 生成数据，去掉车,其中df_missveh2，加入列df_missveh：['removed_vehicles'] = removed_vehicles_posi
    df_missveh,queued_info,df_missveh2 = genSamplesByRandomRemovingVehicle(df1, remove_ratio=0.6)
    
    print('len(queued_vehicles_removed):',len(queued_info)) 
   
    print(f"{'-'*100}")
    
    #调试用，检测一下df_missveh2中，丢失的车辆位置是否正确
    '''
    print(df_missveh.iloc[0])
    print(df_missveh2.iloc[0])
    for idx, row in df_missveh2.iterrows():
        print(f"{'-'*5}") 
        removed_vehicles = row['removed_vehicles']
        print(removed_vehicles)   
        for val in removed_vehicles:
            if val is not None:
                pos_col,pos,car_posi = val  # Extract position from tuple (pos_col,pos,car_posi)
                print(f"Sample car_pos_{car_posi}: Checking removed vehicle at position {pos:.2f}")

    print(f"{'-'*100}")   
    '''
    #调试结束
   
    X_train2, X_val2, y_train2, y_val2, \
        raw_train2, raw_val2, train_dataset2, val_dataset2, \
            raw_cols2 = genDatasetLost(df_missveh2, args.test_size, args.batch_size  )

    df_step2_missveh2 = df_missveh2.copy()#df_missveh的['removed_vehicles']为None,df_missveh2['removed_vehicles']为具体删除车辆的信息位置和名称
    df_step2_missveh = df_missveh.copy()
    
    
        
        
        
    
    # ==============================================
    # 🚕：第三步，将两部分数据合并，加入intersection_pos列，并随机抽样1000个样本，保证样本多样性
    # ==============================================
    #parser.add_argument('--trainvalMode', type=int, default=0, help='0(训练验证都无丢失),1(训练验证都有丢失)')
    #df_step1,df_step2_missveh2
    if args.trainvalmode == 0:
         df_all = df1
    if args.trainvalmode == 1:
         df_all = pd.concat([df1, df_missveh2], ignore_index=True)
            
  
            
   



            
    
        
        
    lane_pos_map = {5: 53.05, 6: 53.13, 7: 53.30}
    df_all['intersection_pos'] = df_all['lane'].map(lane_pos_map)
    print(df_all.iloc[0])
    
    #--------------------------------------------------------------------------------------------------------
    # 随机提取args.nC个样本，尽可能保证样本多样性
    df = df_all.copy()
    numSamples = args.nC
    print(f"原始数据样本数: {len(df)}，准备随机抽样 {numSamples} 个样本进行训练和验证...")
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
        # 如果不足numSamples个，再随机补齐
        if len(sampled_indices) < numSamples:
            remaining = list(set(range(len(df))) - set(sampled_indices))
            extra = np.random.choice(remaining, size=numSamples - len(sampled_indices), replace=False)
            sampled_indices.extend(extra)
        sampled_indices = np.array(sampled_indices[:numSamples])  # 最终保留numSamples个样本
        df = df.iloc[sampled_indices].reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)


    df_step3_all = df_all.copy()
    df_step3_allSampled = df.copy()

    
    if args.trainvalmode == 3:
        df_all = pd.concat([df1, df_missveh2], ignore_index=True)
       

    dt = args.dt
    logging.info(f"使用时间步长 dt={dt} 进行仿真。采样样本数为:{len(df)}")
    
    # ==============================================
    # 🚕：第四步，处理是否修补数据
    # ==============================================

    #方法0，不补。用于比较
    if args.fixdata == 0:
        lost_indices = df.index[df['lost'] == 1].tolist()
        print(f'方法0不修补数据，直接仿真。采样样本数为:{len(df)}，缺失样本数{len(lost_indices)}')
    #-----------------------------------------------------------------------------------------------
    #识别当前样本是否有缺失数据
    #1.采用简单方法判别，就是与前车的距离是否大于某个阈值，比如5米，就认为中间有车丢失
    #2.采用机器学习方法判别，训练一个二分类模型，输入为
    #3.或者直接使用df中的'lost'列，1表示有车丢失，0表示没有车丢失。用[removed_vehicles]列进行辅助判别

    #缺失数据修改start-----------------------------------------------------------------------------------------------
    # 方法1. 计算每辆车与前车的距离(没有完成)
    if 0:
        lost_indices = df.index[df['lost'] == 1].tolist()
        car_pos_cols = [col for col in df.columns if col.startswith('car_position_')]
        front_car_ids = [{} for _ in range(df.shape[0])]  # 存储每个样本的前车ID映射
        car_gaps = [{} for _ in range(df.shape[0])] # 存储每个样本的车间距映射
        for idx in lost_indices:
            data = df.loc[idx]
            data_car_positions = data[car_pos_cols]
            # 获取当前样本的所有车辆位置和对应的列名
            valid = []
            for i, pos in enumerate(data_car_positions):
                col_name = car_pos_cols[i]
                if pos != -1:
                    valid.append((pos, col_name))
            
            # 按位置从大到小排序
            sorted_valid = sorted(valid, key=lambda x: -x[0])
            for idx2 in range(1, len(sorted_valid)):
                curr_id = sorted_valid[idx2][1]  # 获取id = col_name
                prev_id = sorted_valid[idx2 - 1][1]
                cur_pos =  sorted_valid[idx2][0]
                prev_pos = sorted_valid[idx2 - 1][0]
                front_car_ids[idx][curr_id] = prev_id
                car_gaps[idx][curr_id] = prev_pos - cur_pos

       
    # 方法2. 直接用原来的数据补,速度靠前车速度,简单点
    if args.fixdata == 1:
        print('方法1修补数据')
        lost_indices = df.index[df['lost'] == 1].tolist()
        for idx in lost_indices:
            removed_vehs = df.at[idx, 'removed_vehicles']#(丢失车辆命名,丢失车辆位置,车辆命名i的int值)
            for i in range(len(removed_vehs)):
                car_pos_col,car_pos,car_pos_i = removed_vehs[i]
                df.at[idx, f'car_position_{car_pos_i}'] = car_pos
            
                car_speed_i = max(0,car_pos_i-1)#一般而言，前车(小i值)一般有，car_pos样本中一般都是距离红灯距离，car_pos_i越小距离红灯越近（不绝对）
                df.at[idx, f'car_speed_{car_pos_i}'] = df.at[idx, f'car_speed_{car_speed_i}']




    # 方法2. 根据'lost'和'removed_vehicles'列，车辆丢失的位置,前车-5，或者后车+5
    if args.fixdata == 2:
        print('方法2修补数据,+—5前后车')
        lost_indices = df.index[df['lost'] == 1].tolist()
        if len(lost_indices) >0:       
            for idx in lost_indices:
                car_positions1 = df.loc[idx, [c for c in df.columns if c.startswith('car_position_')]].values
                car_speed1 = df.loc[idx, [c for c in df.columns if c.startswith('car_speed_')]].values

                car_positions1 = [c for c in car_positions1 if c != -1]
                car_speed1 =  [c for c in car_speed1 if c != -1]


                removed_vehs = df.at[idx, 'removed_vehicles']#(丢失车辆命名,丢失车辆位置,车辆命名i的int值)
                for i in range(len(removed_vehs)):
                    car_pos_col,car_pos,car_pos_i = removed_vehs[i]
                    car_positions1 = np.array(car_positions1)  
                    tmp = car_positions1-car_pos

                    indexTmp = np.argmin(np.abs(tmp))
                    #车辆丢失的位置,前车+5，或者后车-5
                    if car_pos > car_positions1[indexTmp]:#car_pos样本中一般都是距离红灯距离
                        car_pos_pTmp= car_positions1[indexTmp]+5.0
                    else:
                        car_pos_pTmp= car_positions1[indexTmp]-5.0

                    df.at[idx, f'car_position_{car_pos_i}'] = car_pos_pTmp#直接用最近车位置的+或者-代替
                    df.at[idx, f'car_speed_{car_pos_i}'] = car_speed1[indexTmp]#直接用最近车的速度代替
        

        

            

                

    df_step4_lostfilled = df.copy()

    #第四步，处理缺失数据，补车。缺失数据修改end-----------------------------------------------------------------------------------------------
    
    
    
    
    # ==============================================
    # 🚕：第五步，开始处理数据，准备训练数据集和验证数据集
    # ==============================================
    # 明确区分特征列和传递给仿真的原始数据列
    feature_cols = [c for c in df.columns if ('car_position_' in c or 'car_speed_' in c or 'redLight' in c)]
    #feature_cols.add('main_car_position')#这里特征没有加入 main_car_position,原因是采用模拟方法，
    #下面raw_cols_set加入了main_car_position，会提取相应主车的最终时间Y.而直接黑盒预测需要加入main_car_position
    # 为仿真准备的列，确保没有重复
    # 我们需要所有特征列，以及一些额外的信息
    raw_cols_set = set(feature_cols)
    raw_cols_set.add('lane')
    raw_cols_set.add('intersection_pos')
    raw_cols_set.add('main_car_position')
    raw_cols_set.add('lost')
    raw_cols = sorted(list(raw_cols_set)) # 排序以保证列顺序一致

    X = df[feature_cols].values.astype(np.float32)
    y = (df['time_to_vanish'].values / 30.0).astype(np.float32)#注意已经除以30了,仿真中还会对redLight，car_pos进行进一步处理
    raw_data_for_sim = df[raw_cols].values.astype(np.float32)
    
    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
        X, y, raw_data_for_sim, test_size=args.test_size, random_state=42
    )
    
    


   
    # ==============================================
    # 💎 # 功能：mlp+cf参数+全局参数 模型构建 + 训练 + 验证 + 保存，注意需要1,2,3,4,5步，
    # ==============================================
    
    if args.model == 0:
        # 创建 tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, raw_train))
        #train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, raw_val))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        logging.info(f"数据准备完成: {len(X_train)} 训练样本, {len(X_val)} 验证样本")
        logging.info(f"数据准备完成: X_train.shape:{X_train.shape}, y_train.shape:{y_train.shape}")
        logging.info(f"数据准备完成: X_val.shape:{X_val.shape}, y_val.shape:{y_val.shape}")
        #mlp+cf参数+全局参数 模型构建 + 训练 + 验证 + 保存
        train_model(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt)
    

    
    
    # ==============================================
    # 💎 # 功能：mlp+regress,直接预测vansishTime, 模型构建 + 训练 + 验证 + 保存
    # ==============================================
    if args.model == 1:
        # 创建 tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        #train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        logging.info(f"数据准备完成: {len(X_train)} 训练样本, {len(X_val)} 验证样本")
        logging.info(f"数据准备完成: X_train.shape:{X_train.shape}, y_train.shape:{y_train.shape}")
        logging.info(f"数据准备完成: X_val.shape:{X_val.shape}, y_val.shape:{y_val.shape}")
        #mlp+regress参数+全局参数 模型构建 + 训练 + 验证 + 保存
        train_model2(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt, raw_val=raw_val)
  






# ==============================================
# 💎【代码块1 已封装成独立函数】
# 功能：mlp+cf参数+全局参数 模型构建 + 训练 + 验证 + 保存
# ==============================================
def train_model(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt):
    num_types = args.num_types
    num_types2 = num_types + 1
    output_dim = num_types2 * 6

    # 构建模型
    model = build_simple_resnet2(X_train.shape[1], output_dim, args.unit, args.layNum)
    param_bounds = get_param_bounds(num_types)
    raw_columns_list = raw_cols

    # 为后续按丢失车辆数分组评估，确定 'lost' 在 raw_columns_list 中的索引（若不存在则为 None）
    idx_lost = raw_columns_list.index('lost') if 'lost' in raw_columns_list else None

    # 学习率调度器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr, decay_steps=100, decay_rate=0.99, staircase=True
    )
    optimizer = RMSprop(learning_rate=lr_schedule)

    # 训练步
    @tf.function
    def train_step(x_batch, y_batch, raw_batch):
        with tf.GradientTape() as tape:
            nn_output = model(x_batch, training=True)
            predicted_times = tf.py_function(
                func=lambda nn_out, raw: run_batch_simulation2(
                    nn_out, raw, param_bounds, num_types, columns=raw_columns_list, dt=dt,args=args
                ),
                inp=[nn_output, raw_batch],
                Tout=tf.float32
            )
            batch_size = tf.shape(y_batch)[0]
            predicted_times = tf.reshape(predicted_times, [batch_size])
            y_batch = tf.reshape(y_batch, [batch_size])
            loss = tf.reduce_mean(tf.square(predicted_times - y_batch))

        grads = tape.gradient(loss, model.trainable_variables)
        clipped_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
        del nn_output, predicted_times, grads, clipped_grads
        return loss

    # 验证步
    @tf.function
    def val_step(x_batch, y_batch, raw_batch):
        nn_output = model(x_batch, training=False)
        predicted_times = tf.py_function(
            func=lambda nn_out, raw: run_batch_simulation2(
                nn_out, raw, param_bounds, num_types, columns=raw_columns_list, dt=dt,args=args
            ),
            inp=[nn_output, raw_batch],
            Tout=tf.float32
        )
        batch_size = tf.shape(y_batch)[0]
        predicted_times = tf.reshape(predicted_times, [batch_size])
        y_batch = tf.reshape(y_batch, [batch_size])
        return predicted_times - y_batch

    # 主训练循环
    for epoch in range(args.epochs):
        logging.info(f"===== Epoch {epoch + 1}/{args.epochs} =====")
        epoch_loss_avg = tf.keras.metrics.Mean()
        total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        batch_idx = 0

        for x_batch, y_batch, raw_batch in train_dataset:
            batch_idx += 1
            t0 = time.time()
            loss = train_step(x_batch, y_batch, raw_batch)
            t1 = time.time()
            epoch_loss_avg.update_state(loss)
            rem = total_batches - batch_idx
            logging.info(f"Batch {batch_idx}/{total_batches} | Loss: {loss.numpy():.4f} | {t1-t0:.2f}s | Remain:{rem}")

        logging.info(f"Epoch {epoch+1} Avg Loss: {epoch_loss_avg.result().numpy():.4f}")

        # 验证
        if epoch % 3 == 0 or epoch == args.epochs - 1:
            val_errs = []
            val_loss = tf.keras.metrics.Mean()
            for xb, yb, rb in val_dataset:
                errs = val_step(xb, yb, rb)
                enp = errs.numpy()
                val_errs.append(enp)
                val_loss.update_state(np.mean(np.square(enp)))

            val_errs = np.concatenate([e.flatten() for e in val_errs])
            rmse = np.sqrt(np.mean(np.square(val_errs)))
            mae = np.mean(np.abs(val_errs))
            logging.info(f"Val RMSE: {rmse:.4f},Val Mae:{mae:.4f}")
            #np.save(f"./tmpModes/val_err_epoch_{epoch+1}.npy", val_errs)

        # 保存模型
       # os.makedirs("./tmpModes", exist_ok=True)
       # model.save(f"./tmpModes/model_epoch_{epoch+1}.h5")
        logging.info(f"Model saved: model_epoch_{epoch+1}.h5")
        force_clean_all_memory()

    logging.info("✅ 训练完成！")

    # =================
    # 功能：对验证集中按缺失车辆数（0,1,2,3,4+）进行单独评估，看看模型在不同丢失程度上的表现
    # df['lost'] 代表丢失车辆数量，df['removed_vehicles'] 代表丢失车辆的信息（位置和命名）
    # ================
   

    # buckets: 0,1,2,3 表示 3 表示 3+辆丢失
    buckets = {0: [], 1: [], 2: [], 3: []}
    # 为最终的按丢失车辆分组评估重置错误收集器，
    # 避免之前训练/验证阶段将 `val_errs` 设为 numpy 数组后再次调用 `.append` 抛错
    val_errs = []
    val_loss = tf.keras.metrics.Mean()

    for xb, yb, rb in val_dataset:
        errs = val_step(xb, yb, rb)
        enp = errs.numpy().flatten()
        val_errs.append(enp)
        val_loss.update_state(np.mean(np.square(enp)))

        # 使用 raw 中的 'lost' 列进行分组
    

        try:
            rb_np = rb.numpy()
        except Exception:
            rb_np = np.array(rb)

        # 如果 idx_lost 未定义或不在 raw columns 中，跳过分组统计
        if idx_lost is None:
            continue

        # 取出每个样本的 lost 值（兼容 rb_np 为 1D 或 2D 的情况）
        if rb_np.ndim == 1:
            # 单样本情况，rb_np 长度等于列数
            lost_vals = np.array([rb_np[idx_lost]]).astype(np.int32)
        else:
            lost_vals = rb_np[:, idx_lost].astype(np.int32)
        for i, m in enumerate(lost_vals):
            key = int(m) if int(m) <= 2 else 3
            buckets[key].append(float(enp[i]))

    # 计算并输出每个分组的指标
    for k in sorted(buckets.keys()):
        arr = np.array(buckets[k])
        cnt = arr.size
        if cnt > 0:
            rmse_k = np.sqrt(np.mean(np.square(arr)))
            mae_k = np.mean(np.abs(arr))
        else:
            rmse_k = float('nan')
            mae_k = float('nan')
        name = f"{k}" if k < 3 else "3+"
        logging.info(f"Val Missing={name}: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")
        print(f"Val Missing={name}: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")

    return model

# ==============================================
# 💎【代码块1结束
# ==============================================


# ==============================================
# 💎【代码块2 】
# 功能：mlp直接回归 模型构建 + 训练 + 验证 + 保存
# ==============================================
from tensorflow.keras.callbacks import EarlyStopping
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def train_model2(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt, raw_val=None):
    print("train_model2：mlp预测vanishTime 直接回归任务")
    logging.info("启动 MLP 直接回归模型训练（预测消失时间）")
   
    # 构建回归模型，用于预测消失时间
    output_dim_vanish = 1  # 回归任务，预测消失时间
    model_vanish_reg = build_simple_resnet_regress2(X_train.shape[1], output_dim_vanish, args.unit, args.layNum)

    # 使用学习率调度器
    initial_learning_rate = args.lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,  # 每100步衰减
        decay_rate=0.99,  # 每次衰减到99%
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule)
    #optimizer = RMSprop(learning_rate=lr_schedule)

    # 对于回归任务，使用均方误差损失函数
    model_vanish_reg.compile(optimizer=optimizer, loss='mse', metrics=['mae',rmse])
    
    
    early_stop = EarlyStopping(
            monitor='loss',
            patience=100,
            restore_best_weights=True,
            verbose=1
        )
      #model.fit(
      #  X_train, y_train,
      #  validation_data=(X_val, y_val),
      #  epochs=150,
      #  batch_size=8,
      #  callbacks=[early_stop],  # 加上这个
      #  shuffle=True
            #)
  

    # 训练回归模型
    model_vanish_reg.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs,verbose=1,callbacks=[early_stop],shuffle=True)


    
   # ===================== 1. 训练完成 → 评估训练集/验证集指标 =====================
    print("\n" + "="*60)
    print("模型最终评估指标")
    print("="*60)

    # 评估 训练集
    train_loss, train_mae,train_rmse = model_vanish_reg.evaluate(train_dataset, verbose=0)
    train_mse = train_loss  # MSE损失 = 损失值
    
    # 评估 验证集
    val_loss, val_mae,val_rmse = model_vanish_reg.evaluate(val_dataset, verbose=0)
    val_mse = val_loss
   

    # ===================== 6. 打印+日志输出指标 =====================
    # 训练集结果
    print(f"【训练集】 MAE: {train_mae:.4f} | MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f}")
    # 验证集结果
    print(f"【验证集】 MAE: {val_mae:.4f} | MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f}")
    
    logging.info(f"训练集评估 - MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
    logging.info(f"验证集评估 - MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")

    # ===================== 7. 保存模型（可选） =====================
    os.makedirs("./tmpModes", exist_ok=True)
    model_path = "./tmpModes/mlp_regression_model.h5"
    model_vanish_reg.save(model_path)
    logging.info(f"回归模型已保存至: {model_path}")

    logging.info("train_model2：mlp reg 训练完成！")

    ################################
    # 按丢失车辆数对验证集进行单独评估（0,1,2,3+，其中3表示3+）
    ###############################
   
    idx_lost = raw_cols.index('lost') if 'lost' in raw_cols else None
    raw_val_np = np.asarray(raw_val)
    print(f"raw_val_np.shape={raw_val_np.shape}, idx_lost={idx_lost}")

    # 获取验证集真实值（从 val_dataset 或者 raw_val 旁路收集）
    y_true_list = []
    for xb, yb in val_dataset:
        y_true_list.append(yb.numpy().reshape(-1))
   
    y_true = np.concatenate(y_true_list)
  

    # 预测验证集（与 val_dataset 保持相同顺序）
    y_pred = model_vanish_reg.predict(val_dataset)
    y_pred = np.asarray(y_pred).reshape(-1)

    # 残差（预测 - 真实）
    residuals = y_pred - y_true

    # 分组统计
    buckets = {0: [], 1: [], 2: [], 3: []}
    n_res = residuals.shape[0]
    for i in range(n_res):
        lost_val = int(raw_val_np[i, idx_lost])
        key = lost_val if lost_val <= 2 else 3
        buckets[key].append(float(residuals[i]))

    # 输出每个分组的 RMSE/MAE
    for k in sorted(buckets.keys()):
        arr = np.array(buckets[k])
        cnt = arr.size
        if cnt > 0:
            rmse_k = np.sqrt(np.mean(np.square(arr)))
            mae_k = np.mean(np.abs(arr))
        else:
            rmse_k = float('nan')
            mae_k = float('nan')
        name = f"{k}" if k < 3 else "3+"
        logging.info(f"Regress Val Missing={name}: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")
        print(f"Regress Val Missing={name}: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")

    return model_vanish_reg

# ==============================================
# 💎【代码块1结束
# ==============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Keras和交通仿真进行端到端模型训练")
    parser.add_argument('--csv_path', type=str, default='trainsamples_lane_5_6_7.csv', help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.9, help='验证集比例')
    parser.add_argument('--num_types', type=int, default=6, help='车辆类别数')
    parser.add_argument('--unit', type=int, default=128, help='ResNet隐藏层单元数')
    parser.add_argument('--layNum', type=int, default=8, help='ResNet块数量')
    parser.add_argument('--log_path', type=str, default='training_log.log', help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用Debug级别的日志信息')
    parser.add_argument('--dt', type=float, default=0.5, help='仿真时间步长')
    parser.add_argument('--nC', type=int, default=100, help='Kmeans聚类数量，用于样本多样性选择')
    parser.add_argument('--model', type=int, default=0, help='0(mlp+jax),1(mlp+regress)')
    parser.add_argument('--fixdata', type=int, default=0, help='0(不修补),1(直接用原始数据修补),2(前后车+-5位置进行修补)')
    parser.add_argument('--goffset', type=int, default=1, help='0(jax模型中全局偏移参数为0),1(jax模型中全局偏移参数为1，默认1)')
    parser.add_argument('--trainvalmode', type=int, default=0, help='0(训练验证都无丢失),1(训练验证都有丢失)')
    args = parser.parse_args()
    main(args)

#python modelsCollect4.py --batch_size 16 --layNum 4
#python modelsCollect7.py --batch_size 32 --test_size 0.5 --epochs 150 --lr 0.00005 --unit 256 --layNum 16 --dt 0.1 --nC 500 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1
#mae3.8