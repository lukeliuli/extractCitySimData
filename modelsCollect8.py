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
jax.config.update('jax_platform_name', 'cpu')
os.environ["JAX_PLATFORMS"] = "cpu"
# 可选：禁用GPU相关的JAX功能
os.environ["JAX_ENABLE_GPU"] = "0"

from jax.lib import xla_client
def force_clean_all_memory():
    """硬核清理：JAX+Keras+系统内存，彻底释放无效占用"""
    # 1. 清理Keras计算图
    #keras.backend.clear_session()
    # 3. 强制解绑所有JAX张量引用（核心！解决释放无效）
    #jnp.zeros((1,)).delete()
    # 4. 手动触发系统垃圾回收
    #gc.collect()
    #gc.collect()  # 两次回收更彻底
    
    """
    JAX 0.3.25 专属强制清理
    清理：JAX缓存 + TensorFlow显存 + 系统内存
    """
 
    # 2. 清理TensorFlow显存（TF和JAX共用GPU，必须清！）
    #tf.keras.backend.clear_session()
    #tf.compat.v1.reset_default_graph()

       
    
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
        (30/3.6, 75/3.6), # v0
        (0.1, 2.0),       # T
        (0.2, 1.0),       # s0
        (1.0, 6.0),       # a
        (1.0, 9.0),       # b
        (0.01, 1.0)       # rtime
     
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
            redlightpos2vanishpos_offset = redlightpos2vanishpos_offset*8
            vehpos_offset = (-1.0+vehpos_offset*2.0)*2
            redlightpos_offset = redlightpos_offset*2
            #vanishtime_offset = (-1.0+vanishtime_offset*2.0)*1.0
            vanishtime_offset = (-1.0+vanishtime_offset*2.0)*2 #结果变好的核心改变
            distgap_offset = (-1.0+distgap_offset*2.0)*2
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
from modelsLostReg import genSamplesByRandomRemovingVehicle,genSamplesRemovingVehicleWithNum
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
    #df_missveh,queued_info,df_missveh2 = genSamplesByRandomRemovingVehicle(df1, remove_ratio=0.6)
    df_missveh_rn1,queued_info_rn1,df_missveh2_rn1  = genSamplesRemovingVehicleWithNum(df1, num_to_remove = 1)
    df_missveh_rn2,queued_info_rn2,df_missveh2_rn2 = genSamplesRemovingVehicleWithNum(df1, num_to_remove = 2)
    df_missveh_rn3,queued_info_rn3,df_missveh2_rn3  = genSamplesRemovingVehicleWithNum(df1, num_to_remove = 3)
    df_missveh_rn4,queued_info_rn4,df_missveh2_rn4  = genSamplesRemovingVehicleWithNum(df1, num_to_remove = 4)
    
    df_step2_missveh2 = pd.concat([df_missveh2_rn1, df_missveh2_rn1, df_missveh2_rn1, df_missveh2_rn4], ignore_index=True)
   
    
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
            raw_cols2 = genDatasetLost(df_step2_missveh2, args.test_size, args.batch_size  )

    df_step2_missveh2_backup = df_step2_missveh2.copy()#df_missveh的['removed_vehicles']为None,df_missveh2['removed_vehicles']为具体删除车辆的信息位置和名称
  
    
    
        
        
        
    
    # ==============================================
    # 🚕：第三步，将两部分数据合并，人工删除样本，加入intersection_pos列，并随机抽样1000个样本，保证样本多样性
    # ==============================================
    #parser.add_argument('--trainvalMode', type=int, default=0, help='0(训练验证都无丢失),1(训练验证都有丢失)')
    #df_step1,df_step2_missveh2
    if args.trainvalmode == 0:
         df_all = df1
    if args.trainvalmode == 1:
         df_all = pd.concat([df1, df_step2_missveh2], ignore_index=True)
            
  
            
   



            
    
        
        
    lane_pos_map = {5: 53.05, 6: 53.13, 7: 53.30}
    df_all['intersection_pos'] = df_all['lane'].map(lane_pos_map)
    #print(df_all.iloc[0])
    
    df_step3_noHumanRemove = df_all.copy()
    #按照人工结果分析，将主车前方排队车辆大于等于4，小于1；lost车辆大于等于3；vanish_time 大于35秒的样本，从样本中删除
    df = df_all 
    def count_queued_vehicles(row):
        """统计主车前方排队的有效车辆数（位置≠-1）"""
        # 获取所有车辆位置列
        car_pos_cols = [col for col in row.index if col.startswith('car_position_')]
        # 主车位置
        main_car_pos = row['main_car_position']
        # 统计主车前方（位置 < 主车位置，且有效）的车辆数
        queued_count = 0
        for col in car_pos_cols:
            pos = row[col]
            if pos != -1 and not pd.isna(pos) and pos < main_car_pos:
                queued_count += 1
        return queued_count

    # 计算排队车辆数
    df['queued_vehicles'] = df.apply(count_queued_vehicles, axis=1)

    # 2. 定义过滤条件
    # 条件1：排队车辆数 ≥4 或 <1
    cond_queued = (df['queued_vehicles'] > 3) | (df['queued_vehicles'] < 1)
    # 条件2：丢失车辆数 ≥3（lost列）
    cond_lost = df['lost'] >= 3
    # 条件3：消失时间 >35秒（注意原数据已除以30，需还原后判断）
    cond_vanish = (df['time_to_vanish']) > 35*30

    # 3. 合并过滤条件（满足任一条件则删除）
    filter_cond = cond_queued | cond_lost | cond_vanish

    # 4. 执行过滤
    before_count = len(df)
    df = df[~filter_cond].reset_index(drop=True)
    after_count = len(df)

    # 打印过滤日志
    logging.info(f"样本过滤：删除排队车辆异常/丢失车辆过多/消失时间过长样本 {before_count - after_count} 个")
    logging.info(f"过滤前样本数：{before_count}，过滤后样本数：{after_count}")

    # 清理临时列
    #df = df.drop(columns=['queued_vehicles'])
    df_step3_humanRemoved  = df.copy()

    df_all =  df.copy()
    #--------------------------------------------------------------------------------------------------------
    # 随机提取args.nC个样本，尽可能保证样本多样性
  
    numSamples = args.nC
    print(f"原始数据样本数: {len(df)}，准备随机抽样 {numSamples} 个样本进行训练和验证...")
    if len(df) > numSamples:
        # 先用KMeans聚类，保证多样性
        sample_features = df[[c for c in df.columns if 'car_position_' in c or 'car_speed_' in c]].values
        n_clusters = min(100, len(df) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
        cluster_labels = kmeans.fit_predict(sample_features)
        sampled_indices = []

        #1/2KMeans聚类抽样，1/2lost加权抽样
        for cluster in range(n_clusters):
            cluster_idx = np.where(cluster_labels == cluster)[0]
            if len(cluster_idx) > 0:
                # 每个簇随机采样一定数量
                n = max(1, int(numSamples/2/ n_clusters))
                chosen = np.random.choice(cluster_idx, size=min(n, len(cluster_idx)), replace=False)
                sampled_indices.extend(chosen)
        sampleKeamNum = len(sampled_indices)
        # 如果不足numSamples个，再基于 lost 加权概率补齐（偏重 lost 值大的样本）
        if len(sampled_indices) < numSamples:
            remaining = list(set(range(len(df))) - set(sampled_indices))
            if len(remaining) > 0:
                oversample_factor = 2.0
                lost_rem = df['lost'].iloc[remaining].fillna(0).astype(float).values
                weights_rem = 1.0 + lost_rem * oversample_factor
                probs_rem = weights_rem / np.sum(weights_rem)
                
                extra = list(np.random.choice(remaining, size=(numSamples - len(sampled_indices)), replace=False, p=probs_rem))
               
                sampled_indices.extend(extra)
        
        sampled_indices = np.array(sampled_indices[:numSamples])  # 最终保留numSamples个样本
        sampleLostWProNum = len(sampled_indices) - sampleKeamNum
        df = df.iloc[sampled_indices].reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)


   
    df_step3_allSampled = df.copy()
    logging.info(f"抽样样本数: {len(sampled_indices)}.KMeans聚类抽样了{sampleKeamNum}个样本,lost加权抽样补充了{sampleLostWProNum}个样本。")
    
    

       

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
        lost_indices = df.index[df['lost'] > 0].tolist()
        for idx in lost_indices:
            removed_vehs = df.at[idx, 'removed_vehicles']#(丢失车辆命名,丢失车辆位置,车辆命名i的int值)
            for i in range(len(removed_vehs)):
                car_pos_col,car_pos,car_pos_i = removed_vehs[i]
                df.at[idx, f'car_position_{car_pos_i}'] = car_pos
            
                car_speed_i = max(0,car_pos_i-1)#一般而言，前车(小i值)一般有，car_pos样本中一般都是距离红灯距离，car_pos_i越小距离红灯越近（不绝对）
                df.at[idx, f'car_speed_{car_pos_i}'] = df.at[idx, f'car_speed_{car_speed_i}']




    # 方法2. 根据'lost'和'removed_vehicles'列，车辆丢失的位置,前车-5，或者后车+5
    if args.fixdata == 2:
        print('方法2修补数据：根据前后车位置动态插入（偏移量±5米）')
        lost_indices = df.index[df['lost'] > 0].tolist()          # 修正：去掉多余的空格和数字1
        offset = 5.0                                             # 偏移量（前车+5，后车-5）
        min_gap = 0.5                                            # 避免重叠的最小间距

        for idx in lost_indices:
            # 收集当前帧所有有效车辆的位置与速度（排除 -1）
            car_pos_cols = [c for c in df.columns if c.startswith('car_position_')]
            car_speed_cols = [c for c in df.columns if c.startswith('car_speed_')]
            valid_positions = []
            valid_speeds = []
            for pos_col, speed_col in zip(car_pos_cols, car_speed_cols):
                pos = df.at[idx, pos_col]
                if pos != -1:
                    valid_positions.append(pos)
                    valid_speeds.append(df.at[idx, speed_col])

            # 如果没有有效车辆，无法推断，跳过当前帧
            if not valid_positions:
                continue

            removed_vehs = df.at[idx, 'removed_vehicles']         # 每个元素: (_, orig_car_pos, car_pos_i)
            # 按丢失车辆的原始位置排序，保证插入顺序合理
            removed_sorted = sorted(removed_vehs, key=lambda x: x[1])

            for _, orig_car_pos, car_pos_i in removed_sorted:
                # 在当前有效车辆中寻找与原始位置最接近的车辆
                valid_arr = np.array(valid_positions)
                nearest_idx = np.argmin(np.abs(valid_arr - orig_car_pos))
                nearest_pos = valid_arr[nearest_idx]
                nearest_speed = valid_speeds[nearest_idx]

                # 根据原始位置与最近车的关系决定新位置
                if orig_car_pos > nearest_pos:          # 丢失车辆更靠近红灯（前车）
                    new_pos = nearest_pos + offset
                else:                                   # 丢失车辆更远离红灯（后车）
                    new_pos = nearest_pos - offset

                # 避免新位置与最近车过于接近（防止重叠）
                if abs(new_pos - nearest_pos) < min_gap:
                    new_pos = nearest_pos + min_gap if orig_car_pos > nearest_pos else nearest_pos - min_gap

                # 写入数据帧
                df.at[idx, f'car_position_{car_pos_i}'] = new_pos
                df.at[idx, f'car_speed_{car_pos_i}'] = nearest_speed

                # 将新插入的车辆加入有效列表，供后续丢失车辆参考
                valid_positions.append(new_pos)
                valid_speeds.append(nearest_speed)
                # 保持有效列表有序，以便下次查找最近车
                combined = sorted(zip(valid_positions, valid_speeds), key=lambda x: x[0])
                valid_positions, valid_speeds = zip(*combined) if combined else ([], [])
                valid_positions = list(valid_positions)
                valid_speeds = list(valid_speeds)
        

        

            

                

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
    raw_cols_set.add('main_car_speed')  # ✅ 新增：加入主车速度到原始数据列
    raw_cols_set.add('lost')
    raw_cols = sorted(list(raw_cols_set)) # 排序以保证列顺序一致

    X = df[feature_cols].values.astype(np.float32)
    y = (df['time_to_vanish'].values / 30.0).astype(np.float32)#-----------------------------------------------------注意已经除以30了,仿真中还会对redLight，car_pos进行进一步处理，数据Y
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
        train_dataset = train_dataset.cache() 
        train_dataset = train_dataset.cache() 
        train_model2(X_train, y_train, raw_train, train_dataset, val_dataset, raw_cols, args, dt, raw_val=raw_val)
  






# ==============================================
# 💎【代码块1 已封装成独立函数】
# 功能：mlp+cf参数+全局参数 模型构建 + 训练 + 验证 + 保存
# ==============================================
from tensorflow.keras.optimizers import AdamW
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
    #optimizer = RMSprop(learning_rate=lr_schedule)
    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-5)
    
    #optimizer = Adam(learning_rate=1e-3)
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
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            logging.info("=================================================================验证开始")
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
       # logging.info(f"Model saved: model_epoch_{epoch+1}.h5")
       # force_clean_all_memory()

    logging.info("训练完成！")
    os.makedirs("./tmpModes", exist_ok=True)
    model.save(f"./tmpModes/model0_epoch_{epoch+1}.h5")
    logging.info(f"Model saved: model_epoch_{epoch+1}.h5")
    

    
     ###########################################################################
    # ✅ 新增：基于验证集合，对train_model全维度综合评估模块
    # 分析：丢失车辆数、红灯剩余时间、排队车辆数、vanishTime大小、主车位置，以及主车位置处于主车速度的值，对误差的影响
    # 输出：单因素影响分析 + 多因素联合最优区间（含联合概率）
    ###########################################################################
           

    logging.info("="*100)
    logging.info("🚀 开始train_model（模型0：MLP+JAX仿真）全维度综合误差影响分析")
    logging.info("="*100)
    
    # --------------------- 1. 收集所有验证集样本的完整数据 ---------------------
    print("\n正在收集验证集完整数据...")
    all_samples_data = []
    
    # 获取所有需要的列索引（复用已定义的变量，新增主车速度索引）
    idx_main_car_speed = raw_columns_list.index('main_car_speed') if 'main_car_speed' in raw_columns_list else None
    idx_main_car_pos = raw_columns_list.index('main_car_position') if 'main_car_position' in raw_columns_list else None
    idx_intersection_pos = raw_columns_list.index('intersection_pos') if 'intersection_pos' in raw_columns_list else None
    car_pos_cols = [col for col in raw_columns_list if col.startswith('car_position_')]
    idx_car_pos_list = [raw_columns_list.index(col) for col in car_pos_cols]
    idx_redlight = raw_columns_list.index('redLightRemainingTime') if 'redLightRemainingTime' in raw_columns_list else None
    
    # 重新遍历验证集，收集所有样本数据（独立计算，不复用历史结果）
    for xb, yb, rb in val_dataset:
        # 计算残差（预测值 - 真实值）
        residuals_batch = val_step(xb, yb, rb).numpy().flatten()
        y_true_batch = yb.numpy().flatten()
        rb_np = rb.numpy()
        batch_size = len(residuals_batch)
        
        for i in range(batch_size):
            sample_raw = rb_np[i]
            y_true = y_true_batch[i]
            residual = residuals_batch[i]
            
            # 提取基础特征（空值保护）
            lost = int(sample_raw[idx_lost]) if (idx_lost is not None and not np.isnan(sample_raw[idx_lost])) else -1
            redlight_sec = sample_raw[idx_redlight]/30.0 if (idx_redlight is not None and not np.isnan(sample_raw[idx_redlight])) else -1
            main_pos = sample_raw[idx_main_car_pos] if (idx_main_car_pos is not None and not np.isnan(sample_raw[idx_main_car_pos])) else -1
            main_speed = sample_raw[idx_main_car_speed] if (idx_main_car_speed is not None and not np.isnan(sample_raw[idx_main_car_speed])) else -1
            inter_pos = sample_raw[idx_intersection_pos] if (idx_intersection_pos is not None and not np.isnan(sample_raw[idx_intersection_pos])) else -1
            
            # ✅ 新增：计算主车到红灯的预计到达时间（核心因素）
            if main_speed > 1e-6 and main_pos != -1 and inter_pos != -1 and inter_pos > main_pos:
                main_pos_to_red_time = (inter_pos - main_pos) / main_speed
            else:
                main_pos_to_red_time = -1.0  # 无效值标记
            
            # 计算排队车辆数
            queue_count = 0
            if idx_main_car_pos is not None and idx_intersection_pos is not None:
                for idx in idx_car_pos_list:
                    car_pos = sample_raw[idx]
                    if not np.isnan(car_pos) and car_pos != -1 and main_pos < car_pos < inter_pos:
                        queue_count += 1
            
            # 存储样本数据
            all_samples_data.append({
                'lost': lost,
                'redlight_sec': redlight_sec,
                'queue_count': queue_count,
                'vanish_time_true': y_true,
                'main_car_pos': main_pos,
                'main_car_speed': main_speed,
                'main_pos_to_red_time': main_pos_to_red_time,
                'residual': residual,
                'abs_error': abs(residual)
            })
    
    # 转换为DataFrame方便分析（空值保护）
    df_eval = pd.DataFrame(all_samples_data)
    total_samples = len(df_eval)
    
    if total_samples == 0:
        logging.error("❌ 验证集样本为空，无法进行综合评估")
        print("❌ 验证集样本为空，无法进行综合评估")
        return model
        
    logging.info(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    print(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    
    # --------------------- 2. 单因素影响分析 ---------------------
    logging.info("\n" + "="*80)
    logging.info("📊 单因素对预测误差的影响分析（按MAE升序排列）")
    logging.info("="*80)
    
    # 定义各因素的分箱规则（与train_model2完全一致，便于对比）
    bin_rules = {
        'lost': {
            'bins': [-1, 0, 1, 2, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3+辆']
        },
        'redlight_sec': {
            'bins': [0, 3, 6, 9, 12, 15, 18, 21, 24, float('inf')],
            'labels': ['0-3s', '3-6s', '6-9s', '9-12s', '12-15s', '15-18s', '18-21s', '21-24s', '>24s']
        },
        'queue_count': {
            'bins': [-1, 0, 1, 2, 3, 4, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3辆', '4辆', '5+辆']
        },
        'vanish_time_true': {
            'bins': [0, 5, 10, 15, 20, 25, float('inf')],
            'labels': ['0-5s', '5-10s', '10-15s', '15-20s', '20-25s', '>25s']
        },
        'main_car_pos': {
            'bins': [0, 40, 45, 50, 53, float('inf')],
            'labels': ['<40m', '40-45m', '45-50m', '50-53m', '>53m']
        },
        'main_car_speed': {
            'bins': [0, 5, 10, 15, 20, float('inf')],
            'labels': ['0-5m/s', '5-10m/s', '10-15m/s', '15-20m/s', '>20m/s']
        },
        'main_pos_to_red_time': {
            'bins': [-1, 2, 4, 6, 8, 10, float('inf')],
            'labels': ['<2s', '2-4s', '4-6s', '6-8s', '8-10s', '>10s']
        }
    }
    
    # 存储各因素的分析结果
    factor_results = {}
    
    for factor, rule in bin_rules.items():
        if factor not in df_eval.columns:
            continue
            
        # 分箱（空值处理）
        df_eval[f'{factor}_bin'] = pd.cut(
            df_eval[factor], 
            bins=rule['bins'], 
            labels=rule['labels'],
            include_lowest=True
        )
        
        # 计算各分箱的统计量
        bin_stats = df_eval.groupby(f'{factor}_bin', observed=False).agg({
            'abs_error': ['count', 'mean', 'std'],
            'residual': lambda x: np.sqrt(np.mean(np.square(x)))
        })
        
        # 重命名列并四舍五入
        bin_stats.columns = ['样本数', 'MAE', '误差标准差', 'RMSE']
        bin_stats['MAE'] = bin_stats['MAE'].apply(lambda x: round(x, 4))
        bin_stats['误差标准差'] = bin_stats['误差标准差'].apply(lambda x: round(x, 4))
        bin_stats['RMSE'] = bin_stats['RMSE'].apply(lambda x: round(x, 4))
        bin_stats['样本占比(%)'] = (bin_stats['样本数'] / total_samples * 100).apply(lambda x: round(x, 2))
        
        # 按MAE升序排列
        bin_stats = bin_stats.sort_values('MAE')
        
        # 存储结果
        factor_results[factor] = bin_stats
        
        # 输出结果
        logging.info(f"\n--- {factor} 对误差的影响 ---")
        logging.info(bin_stats.to_string())
        print(f"\n--- {factor} 对误差的影响 ---")
        print(bin_stats.to_string())
    
    # --------------------- 3. 多因素联合最优区间分析（含联合概率） ---------------------
    logging.info("\n" + "="*80)
    logging.info("🎯 多因素联合最优区间分析（误差最小且样本量充足）")
    logging.info("="*80)
    
    # 第一步：从单因素分析中提取每个因素的低误差区间（MAE < 整体平均MAE）
    overall_mae = round(df_eval['abs_error'].mean(), 4)
    overall_rmse = round(np.sqrt(np.mean(np.square(df_eval['residual']))), 4)
    logging.info(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    print(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    
    # 提取每个因素的低误差区间
    low_error_intervals = {}
    for factor, stats in factor_results.items():
        low_bins = stats[stats['MAE'] < overall_mae].index.tolist()
        low_error_intervals[factor] = low_bins
        logging.info(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
        print(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
    
    # 第二步：筛选同时满足所有低误差区间的样本（严格条件）
    strict_mask = pd.Series([True]*len(df_eval))
    for factor, bins in low_error_intervals.items():
        strict_mask &= df_eval[f'{factor}_bin'].isin(bins)
    
    strict_samples = df_eval[strict_mask]
    strict_count = len(strict_samples)
    strict_prob = round(strict_count / total_samples * 100, 2)
    strict_mae = round(strict_samples['abs_error'].mean(), 4) if strict_count > 0 else float('nan')
    strict_rmse = round(np.sqrt(np.mean(np.square(strict_samples['residual']))), 4) if strict_count > 0 else float('nan')
    
    # 第三步：筛选满足核心因素低误差区间的样本（宽松条件）
    # 核心因素：丢失车辆数 + 排队车辆数 + 红灯时间 + 预计到达时间
    core_factors = ['lost', 'queue_count', 'redlight_sec', 'main_pos_to_red_time']
    loose_mask = pd.Series([True]*len(df_eval))
    for factor in core_factors:
        if factor in low_error_intervals:
            loose_mask &= df_eval[f'{factor}_bin'].isin(low_error_intervals[factor])
    
    loose_samples = df_eval[loose_mask]
    loose_count = len(loose_samples)
    loose_prob = round(loose_count / total_samples * 100, 2)
    loose_mae = round(loose_samples['abs_error'].mean(), 4) if loose_count > 0 else float('nan')
    loose_rmse = round(np.sqrt(np.mean(np.square(loose_samples['residual']))), 4) if loose_count > 0 else float('nan')
    
    # 第四步：输出最优区间总结
    logging.info("\n" + "-"*60)
    logging.info("📈 综合评估结果总结")
    logging.info("-"*60)
    
    logging.info(f"\n✅ 严格最优区间（同时满足所有因素低误差）：")
    for factor in low_error_intervals.keys():
        logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {strict_prob}% ({strict_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {strict_mae:.4f}s, 平均RMSE: {strict_rmse:.4f}s")
    if not np.isnan(strict_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - strict_mae)/overall_mae*100, 2)}%")
    
    logging.info(f"\n✅ 实用最优区间（满足核心因素低误差，覆盖更多样本）：")
    for factor in core_factors:
        if factor in low_error_intervals:
            logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {loose_prob}% ({loose_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {loose_mae:.4f}s, 平均RMSE: {loose_rmse:.4f}s")
    if not np.isnan(loose_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - loose_mae)/overall_mae*100, 2)}%")
    
    # --------------------- 4. 多因素联合最差区间分析 ---------------------
    logging.info("\n" + "="*80)
    logging.info("⚠️  多因素联合最差区间分析（误差最大且样本量充足）")
    logging.info("="*80)
    
    # 提取每个因素的高误差区间（MAE > 整体平均MAE的1.5倍）
    high_error_threshold = overall_mae * 1.5
    high_error_intervals = {}
    for factor, stats in factor_results.items():
        high_bins = stats[stats['MAE'] > high_error_threshold].index.tolist()
        high_error_intervals[factor] = high_bins
        logging.info(f"{factor} 高误差区间(MAE>{high_error_threshold:.4f}): {high_bins}")
        print(f"{factor} 高误差区间(MAE>{high_error_threshold:.4f}): {high_bins}")
    
    # 筛选同时满足所有高误差区间的样本（严格最差条件）
    worst_strict_mask = pd.Series([True]*len(df_eval))
    for factor, bins in high_error_intervals.items():
        if bins:  # 只有当该因素有高误差区间时才加入筛选
            worst_strict_mask &= df_eval[f'{factor}_bin'].isin(bins)
    
    worst_strict_samples = df_eval[worst_strict_mask]
    worst_strict_count = len(worst_strict_samples)
    worst_strict_prob = round(worst_strict_count / total_samples * 100, 2)
    worst_strict_mae = round(worst_strict_samples['abs_error'].mean(), 4) if worst_strict_count > 0 else float('nan')
    worst_strict_rmse = round(np.sqrt(np.mean(np.square(worst_strict_samples['residual']))), 4) if worst_strict_count > 0 else float('nan')
    
    # 筛选满足核心因素高误差区间的样本（宽松最差条件）
    worst_loose_mask = pd.Series([True]*len(df_eval))
    for factor in core_factors:
        if factor in high_error_intervals and high_error_intervals[factor]:
            worst_loose_mask &= df_eval[f'{factor}_bin'].isin(high_error_intervals[factor])
    
    worst_loose_samples = df_eval[worst_loose_mask]
    worst_loose_count = len(worst_loose_samples)
    worst_loose_prob = round(worst_loose_count / total_samples * 100, 2)
    worst_loose_mae = round(worst_loose_samples['abs_error'].mean(), 4) if worst_loose_count > 0 else float('nan')
    worst_loose_rmse = round(np.sqrt(np.mean(np.square(worst_loose_samples['residual']))), 4) if worst_loose_count > 0 else float('nan')
    
    # 输出最差区间总结
    logging.info(f"\n❌ 严格最差区间（同时满足所有因素高误差）：")
    for factor in high_error_intervals.keys():
        logging.info(f"   - {factor}: {high_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {worst_strict_prob}% ({worst_strict_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {worst_strict_mae:.4f}s, 平均RMSE: {worst_strict_rmse:.4f}s")
    if not np.isnan(worst_strict_mae) and overall_mae > 0:
        logging.info(f"   - 误差升高幅度: {round((worst_strict_mae - overall_mae)/overall_mae*100, 2)}%")
    
    logging.info(f"\n❌ 实用最差区间（满足核心因素高误差，覆盖更多样本）：")
    for factor in core_factors:
        if factor in high_error_intervals:
            logging.info(f"   - {factor}: {high_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {worst_loose_prob}% ({worst_loose_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {worst_loose_mae:.4f}s, 平均RMSE: {worst_loose_rmse:.4f}s")
    if not np.isnan(worst_loose_mae) and overall_mae > 0:
        logging.info(f"   - 误差升高幅度: {round((worst_loose_mae - overall_mae)/overall_mae*100, 2)}%")
    
    # 输出误差最大的前10个单因素区间
    logging.info(f"\n⚠️  误差最大的前10个单因素区间：")
    all_bins = []
    for factor, stats in factor_results.items():
        for bin_name, row in stats.iterrows():
            all_bins.append({
                '因素': factor,
                '区间': bin_name,
                'MAE': row['MAE'],
                '样本数': row['样本数']
            })
    df_all_bins = pd.DataFrame(all_bins).sort_values('MAE', ascending=False).head(10)
    logging.info(df_all_bins.to_string(index=False))
    print(df_all_bins.to_string(index=False))
    
    logging.info("\n" + "="*100)
    logging.info("✅ train_model（模型0）全维度综合评估完成")
    logging.info("="*100)
    
    ###########################################################################
    # ✅ 所有评估结果自动保存为CSV文件（与模型1结果分开存储）
    ###########################################################################
    logging.info("\n" + "="*80)
    logging.info("💾 正在保存模型0评估结果到CSV文件...")
    logging.info("="*80)
    
    # 创建模型0专属结果保存目录，避免与模型1覆盖
    result_dir = "./evaluation_results_model0"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加时间戳避免覆盖
    
    # 1. 保存所有验证集样本的详细数据
    all_samples_path = f"{result_dir}/model0_all_samples_evaluation_{timestamp}.csv"
    df_eval.to_csv(all_samples_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 所有样本详细数据已保存: {all_samples_path}")
    
    # 2. 保存每个单因素的分箱统计结果
    for factor, stats in factor_results.items():
        factor_path = f"{result_dir}/model0_factor_{factor}_{timestamp}.csv"
        stats.to_csv(factor_path, encoding="utf-8-sig")
        logging.info(f"✅ {factor} 单因素分析结果已保存: {factor_path}")
    
    # 3. 保存整体评估总结和最优/最差区间
    summary_data = [
        {"指标": "整体平均MAE(s)", "值": overall_mae},
        {"指标": "整体平均RMSE(s)", "值": overall_rmse},
        {"指标": "严格最优区间样本数", "值": strict_count},
        {"指标": "严格最优区间样本占比(%)", "值": strict_prob},
        {"指标": "严格最优区间平均MAE(s)", "值": strict_mae},
        {"指标": "严格最优区间平均RMSE(s)", "值": strict_rmse},
        {"指标": "严格最优区间误差降低幅度(%)", "值": round((overall_mae - strict_mae)/overall_mae*100, 2) if not np.isnan(strict_mae) else float('nan')},
        {"指标": "实用最优区间样本数", "值": loose_count},
        {"指标": "实用最优区间样本占比(%)", "值": loose_prob},
        {"指标": "实用最优区间平均MAE(s)", "值": loose_mae},
        {"指标": "实用最优区间平均RMSE(s)", "值": loose_rmse},
        {"指标": "实用最优区间误差降低幅度(%)", "值": round((overall_mae - loose_mae)/overall_mae*100, 2) if not np.isnan(loose_mae) else float('nan')},
        {"指标": "严格最差区间样本数", "值": worst_strict_count},
        {"指标": "严格最差区间样本占比(%)", "值": worst_strict_prob},
        {"指标": "严格最差区间平均MAE(s)", "值": worst_strict_mae},
        {"指标": "严格最差区间平均RMSE(s)", "值": worst_strict_rmse},
        {"指标": "严格最差区间误差升高幅度(%)", "值": round((worst_strict_mae - overall_mae)/overall_mae*100, 2) if not np.isnan(worst_strict_mae) else float('nan')},
        {"指标": "实用最差区间样本数", "值": worst_loose_count},
        {"指标": "实用最差区间样本占比(%)", "值": worst_loose_prob},
        {"指标": "实用最差区间平均MAE(s)", "值": worst_loose_mae},
        {"指标": "实用最差区间平均RMSE(s)", "值": worst_loose_rmse},
        {"指标": "实用最差区间误差升高幅度(%)", "值": round((worst_loose_mae - overall_mae)/overall_mae*100, 2) if not np.isnan(worst_loose_mae) else float('nan')}
    ]
    
    # 添加各因素低误差区间到总结
    for factor, intervals in low_error_intervals.items():
        summary_data.append({
            "指标": f"{factor} 低误差区间",
            "值": ", ".join(intervals)
        })
    
    # 添加各因素高误差区间到总结
    for factor, intervals in high_error_intervals.items():
        summary_data.append({
            "指标": f"{factor} 高误差区间",
            "值": ", ".join(intervals)
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = f"{result_dir}/model0_overall_evaluation_summary_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 整体评估总结已保存: {summary_path}")
    
    # 4. 保存误差最大的前10个区间
    top10_error_path = f"{result_dir}/model0_top10_high_error_bins_{timestamp}.csv"
    df_all_bins.to_csv(top10_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 误差最大前10区间已保存: {top10_error_path}")
    
    # 5. 保存低误差区间汇总表
    low_error_summary = []
    for factor, intervals in low_error_intervals.items():
        for interval in intervals:
            bin_stats = factor_results[factor].loc[interval]
            low_error_summary.append({
                "因素": factor,
                "区间": interval,
                "样本数": bin_stats["样本数"],
                "样本占比(%)": bin_stats["样本占比(%)"],
                "MAE(s)": bin_stats["MAE"],
                "RMSE(s)": bin_stats["RMSE"]
            })
    df_low_error = pd.DataFrame(low_error_summary)
    low_error_path = f"{result_dir}/model0_low_error_intervals_summary_{timestamp}.csv"
    df_low_error.to_csv(low_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 低误差区间汇总表已保存: {low_error_path}")
    
    # 6. 保存高误差区间汇总表
    high_error_summary = []
    for factor, intervals in high_error_intervals.items():
        for interval in intervals:
            bin_stats = factor_results[factor].loc[interval]
            high_error_summary.append({
                "因素": factor,
                "区间": interval,
                "样本数": bin_stats["样本数"],
                "样本占比(%)": bin_stats["样本占比(%)"],
                "MAE(s)": bin_stats["MAE"],
                "RMSE(s)": bin_stats["RMSE"]
            })
    df_high_error = pd.DataFrame(high_error_summary)
    high_error_path = f"{result_dir}/model0_high_error_intervals_summary_{timestamp}.csv"
    df_high_error.to_csv(high_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 高误差区间汇总表已保存: {high_error_path}")
    
    logging.info(f"\n🎉 模型0所有评估结果已保存到目录: {os.path.abspath(result_dir)}")
    print(f"\n🎉 模型0所有评估结果已保存到目录: {os.path.abspath(result_dir)}")

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
        
        
 
    ###############################
    # 新增：按剩余红灯时间（3秒间隔）分组评估验证集误差
    # 修复：val_dataset只有 (xb, yb)，无rb；独立遍历，不复用任何数据
    ###############################
    logging.info("=================================================================按红灯时间分组验证")

    # 1. 找到红灯时间列索引
    idx_redlight = raw_cols.index('redLightRemainingTime') if 'redLightRemainingTime' in raw_cols else None
    raw_val_np = np.asarray(raw_val)  # 原始验证集（与val_dataset顺序完全一致）

    if idx_redlight is not None:
        # 定义3秒间隔分箱
        redlight_bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, float('inf')]
        redlight_labels = [f"{redlight_bins[i]}-{redlight_bins[i+1]}s" for i in range(len(redlight_bins)-2)] + [f">{redlight_bins[-2]}s"]
        redlight_buckets = {label: [] for label in redlight_labels}

        # 2. 独立遍历验证集（只有xb, yb，无rb！），用索引对应原始数据
        sample_index = 0  # 全局样本索引，对应raw_val_np的行
        for xb, yb in val_dataset:
            # --------------------- 独立计算，不复用任何前置数据 ---------------------
            # 真实值
            y_true_batch = yb.numpy().reshape(-1)
            # 模型预测
            y_pred_batch = model_vanish_reg.predict(xb)
            y_pred_batch = np.asarray(y_pred_batch).reshape(-1)
            # 残差
            residuals_batch = y_pred_batch - y_true_batch
            # 获取当前batch的样本数量
            batch_size = len(y_true_batch)

            # --------------------- 从原始数据提取红灯时间（和lost列逻辑一致） ---------------------
            # 按索引提取当前batch对应的红灯剩余时间
            red_seconds_batch = raw_val_np[sample_index : sample_index + batch_size, idx_redlight]
            red_seconds_batch = red_seconds_batch.astype(float)/30                               ##################模型mlp这里除以30

            # --------------------- 分组存储误差 ---------------------
            for err, sec in zip(residuals_batch, red_seconds_batch):
                for i in range(len(redlight_bins) - 1):
                    if redlight_bins[i] <= sec < redlight_bins[i+1]:
                        redlight_buckets[redlight_labels[i]].append(float(err))
                        break

            # 更新全局索引
            sample_index += batch_size

        # 3. 计算并输出误差指标（格式统一）
        logging.info("===== 按剩余红灯时间（3秒间隔）验证误差 =====")
        print("===== 按剩余红灯时间（3秒间隔）验证误差 =====")
        for label in redlight_labels:
            arr = np.array(redlight_buckets[label])
            cnt = arr.size
            if cnt > 0:
                rmse_k = np.sqrt(np.mean(np.square(arr)))
                mae_k = np.mean(np.abs(arr))
            else:
                rmse_k = float('nan')
                mae_k = float('nan')

            logging.info(f"Regress Val RedLight={label}: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")
            print(f"Regress Val RedLight={label}: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")
    else:
        logging.warning("未找到 redLightRemainingTime 列，跳过红灯时间分组评估")
        print("未找到 redLightRemainingTime 列，跳过红灯时间分组评估")
        
        
        
    
    ###############################
    # 新增：按主车前排队车辆数目分组评估验证集误差
    # 规则：主车与红灯之间的有效车辆 = 排队车辆数
    # 分组：0辆、1辆、2辆、3辆、4+辆
    # 独立遍历，不复用任何数据
    ###############################
    logging.info("=================================================================按主车前排队车辆数分组验证")

    # 1. 定义所需列索引（主车位置、红灯位置、所有车辆位置）
    idx_main_car_pos = raw_cols.index('main_car_position') if 'main_car_position' in raw_cols else None
    idx_red_pos = raw_cols.index('intersection_pos') if 'intersection_pos' in raw_cols else None
    car_pos_cols = [col for col in raw_cols if col.startswith('car_position_')]
    idx_car_pos_list = [raw_cols.index(col) for col in car_pos_cols]

    # 2. 定义分组规则：0,1,2,3,4+
    queue_bins = [0, 1, 2, 3, 4, float('inf')]
    queue_labels = [f"{queue_bins[i]}" for i in range(len(queue_bins)-2)] + [f">{queue_bins[-2]}"]
    queue_buckets = {label: [] for label in queue_labels}

    # 3. 校验关键列是否存在
    if idx_main_car_pos is not None and idx_red_pos is not None and len(idx_car_pos_list) > 0:
        # 独立遍历验证集，不复用任何前置数据
        sample_index = 0
        for xb, yb in val_dataset:
            # 独立计算：真实值/预测值/残差
            y_true_batch = yb.numpy().reshape(-1)
            y_pred_batch = model_vanish_reg.predict(xb)
            y_pred_batch = np.asarray(y_pred_batch).reshape(-1)
            residuals_batch = y_pred_batch - y_true_batch
            batch_size = len(y_true_batch)

            # 独立提取当前batch原始数据
            batch_raw = raw_val_np[sample_index : sample_index + batch_size]
            
            # 遍历batch内每个样本，计算排队车辆数
            for i in range(batch_size):
                sample_raw = batch_raw[i]
                main_pos = sample_raw[idx_main_car_pos]       # 主车位置
                red_pos = sample_raw[idx_red_pos]             # 红灯/路口位置
                err = residuals_batch[i]

                # 计算：主车前排队车辆数（车辆位置在 主车~红灯 之间，且有效≠-1）
                queue_count = 0
                for idx in idx_car_pos_list:
                    car_pos = sample_raw[idx]
                    if car_pos != -1 and main_pos < car_pos < red_pos:
                        queue_count += 1

                # 匹配分组
                for bin_idx in range(len(queue_bins)-1):
                    if queue_bins[bin_idx] <= queue_count < queue_bins[bin_idx+1]:
                        queue_buckets[queue_labels[bin_idx]].append(float(err))
                        break

            # 更新样本索引
            sample_index += batch_size

        # 4. 计算并输出误差指标（格式与原有代码完全统一）
        logging.info("===== 按主车前排队车辆数验证误差 =====")
        print("===== 按主车前排队车辆数验证误差 =====")
        for label in queue_labels:
            arr = np.array(queue_buckets[label])
            cnt = arr.size
            if cnt > 0:
                rmse_k = np.sqrt(np.mean(np.square(arr)))
                mae_k = np.mean(np.abs(arr))
            else:
                rmse_k = float('nan')
                mae_k = float('nan')

            logging.info(f"Regress Val Queue={label}辆: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")
            print(f"Regress Val Queue={label}辆: count={cnt}, RMSE={rmse_k:.4f}, MAE={mae_k:.4f}")
    else:
        logging.warning("未找到主车位置/红灯位置/车辆位置列，跳过排队车辆数分组评估")
        print("未找到主车位置/红灯位置/车辆位置列，跳过排队车辆数分组评估")
        
     ###############################
     #给出代码，对train_model2进行评估：给出丢失车辆数，红灯剩余时间，主车前面排队车辆数目，vanishTime大小，main_car_pos/main_car_speed，对vanishTime预测的误差大小的影响
     # 并综合结果，给出误差比较小的以上各个参数区间，注意要综合评估，最好是联合概率
    ###############################   
  ###########################################################################
    # ✅ 新增：train_model2全维度综合评估模块（已修复round语法错误+空值保护）
    # 分析：丢失车辆数、红灯剩余时间、排队车辆数、vanishTime大小、主车位置、主车速度对误差的影响
    # 输出：单因素影响分析 + 多因素联合最优区间（含联合概率）
    ###########################################################################
    logging.info("="*100)
    logging.info("🚀 开始train_model2全维度综合误差影响分析")
    logging.info("="*100)
    
    # --------------------- 1. 收集所有验证集样本的完整数据 ---------------------
    print("\n正在收集验证集完整数据...")
    all_samples_data = []
    
    # 获取所有需要的列索引
    idx_lost = raw_cols.index('lost') if 'lost' in raw_cols else None
    idx_redlight = raw_cols.index('redLightRemainingTime') if 'redLightRemainingTime' in raw_cols else None
    idx_main_car_pos = raw_cols.index('main_car_position') if 'main_car_position' in raw_cols else None
    idx_main_car_speed = raw_cols.index('main_car_speed') if 'main_car_speed' in raw_cols else None
    idx_intersection_pos = raw_cols.index('intersection_pos') if 'intersection_pos' in raw_cols else None
    car_pos_cols = [col for col in raw_cols if col.startswith('car_position_')]
    idx_car_pos_list = [raw_cols.index(col) for col in car_pos_cols]
    
    # 重新遍历验证集，收集所有样本数据（独立计算，不复用历史结果）
    sample_index = 0
    for xb, yb in val_dataset:
        y_true_batch = yb.numpy().reshape(-1)
        y_pred_batch = model_vanish_reg.predict(xb, verbose=0).reshape(-1)
        residuals_batch = y_pred_batch - y_true_batch
        batch_size = len(y_true_batch)
        
        batch_raw = raw_val_np[sample_index : sample_index + batch_size]
        
        for i in range(batch_size):
            sample_raw = batch_raw[i]
            y_true = y_true_batch[i]
            y_pred = y_pred_batch[i]
            residual = residuals_batch[i]
            
            # 提取基础特征（空值保护）
            lost = int(sample_raw[idx_lost]) if (idx_lost is not None and not np.isnan(sample_raw[idx_lost])) else -1
            redlight_sec = sample_raw[idx_redlight]/30.0 if (idx_redlight is not None and not np.isnan(sample_raw[idx_redlight])) else -1
            main_pos = sample_raw[idx_main_car_pos] if (idx_main_car_pos is not None and not np.isnan(sample_raw[idx_main_car_pos])) else -1
            main_speed = sample_raw[idx_main_car_speed] if (idx_main_car_speed is not None and not np.isnan(sample_raw[idx_main_car_speed])) else -1
            inter_pos = sample_raw[idx_intersection_pos] if (idx_intersection_pos is not None and not np.isnan(sample_raw[idx_intersection_pos])) else -1
            
            # 计算排队车辆数
            queue_count = 0
            if idx_main_car_pos is not None and idx_intersection_pos is not None:
                for idx in idx_car_pos_list:
                    car_pos = sample_raw[idx]
                    if not np.isnan(car_pos) and car_pos != -1 and main_pos < car_pos < inter_pos:
                        queue_count += 1
            
            # 存储样本数据
            all_samples_data.append({
                'lost': lost,
                'redlight_sec': redlight_sec,
                'queue_count': queue_count,
                'vanish_time_true': y_true,
                'main_car_pos': main_pos,
                'main_car_speed': main_speed,
                'residual': residual,
                'abs_error': abs(residual)
            })
        
        sample_index += batch_size
    
    # 转换为DataFrame方便分析（空值保护）
    df_eval = pd.DataFrame(all_samples_data)
    total_samples = len(df_eval)
    
    if total_samples == 0:
        logging.error("❌ 验证集样本为空，无法进行综合评估")
        print("❌ 验证集样本为空，无法进行综合评估")
        return model_vanish_reg
        
    logging.info(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    print(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    
    # --------------------- 2. 单因素影响分析 ---------------------
    logging.info("\n" + "="*80)
    logging.info("📊 单因素对预测误差的影响分析（按MAE升序排列）")
    logging.info("="*80)
    
    # 定义各因素的分箱规则
    bin_rules = {
        'lost': {
            'bins': [-1, 0, 1, 2, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3+辆']
        },
        'redlight_sec': {
            'bins': [0, 3, 6, 9, 12, 15, 18, 21, 24, float('inf')],
            'labels': ['0-3s', '3-6s', '6-9s', '9-12s', '12-15s', '15-18s', '18-21s', '21-24s', '>24s']
        },
        'queue_count': {
            'bins': [-1, 0, 1, 2, 3, 4, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3辆', '4辆', '5+辆']
        },
        'vanish_time_true': {
            'bins': [0, 5, 10, 15, 20, 25, float('inf')],
            'labels': ['0-5s', '5-10s', '10-15s', '15-20s', '20-25s', '>25s']
        },
        'main_car_pos': {
            'bins': [0, 40, 45, 50, 53, float('inf')],
            'labels': ['<40m', '40-45m', '45-50m', '50-53m', '>53m']
        },
        'main_car_speed': {
            'bins': [0, 5, 10, 15, 20, float('inf')],
            'labels': ['0-5m/s', '5-10m/s', '10-15m/s', '15-20m/s', '>20m/s']
        }
    }
    
    # 存储各因素的分析结果
    factor_results = {}
    
    for factor, rule in bin_rules.items():
        if factor not in df_eval.columns:
            continue
            
        # 分箱（空值处理）
        df_eval[f'{factor}_bin'] = pd.cut(
            df_eval[factor], 
            bins=rule['bins'], 
            labels=rule['labels'],
            include_lowest=True
        )
        
        # 计算各分箱的统计量
        bin_stats = df_eval.groupby(f'{factor}_bin', observed=False).agg({
            'abs_error': ['count', 'mean', 'std'],
            'residual': lambda x: np.sqrt(np.mean(np.square(x)))
        })
        
        # 重命名列并四舍五入（修复：使用Python原生round函数）
        bin_stats.columns = ['样本数', 'MAE', '误差标准差', 'RMSE']
        bin_stats['MAE'] = bin_stats['MAE'].apply(lambda x: round(x, 4))
        bin_stats['误差标准差'] = bin_stats['误差标准差'].apply(lambda x: round(x, 4))
        bin_stats['RMSE'] = bin_stats['RMSE'].apply(lambda x: round(x, 4))
        bin_stats['样本占比(%)'] = (bin_stats['样本数'] / total_samples * 100).apply(lambda x: round(x, 2))
        
        # 按MAE升序排列
        bin_stats = bin_stats.sort_values('MAE')
        
        # 存储结果
        factor_results[factor] = bin_stats
        
        # 输出结果
        logging.info(f"\n--- {factor} 对误差的影响 ---")
        logging.info(bin_stats.to_string())
        print(f"\n--- {factor} 对误差的影响 ---")
        print(bin_stats.to_string())
    
    # --------------------- 3. 多因素联合最优区间分析（含联合概率） ---------------------
    logging.info("\n" + "="*80)
    logging.info("🎯 多因素联合最优区间分析（误差最小且样本量充足）")
    logging.info("="*80)
    
    # 第一步：从单因素分析中提取每个因素的低误差区间（MAE < 整体平均MAE）
    overall_mae = round(df_eval['abs_error'].mean(), 4)  # ✅ 修复：使用Python原生round函数
    overall_rmse = round(np.sqrt(np.mean(np.square(df_eval['residual']))), 4)  # ✅ 修复
    logging.info(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    print(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    
    # 提取每个因素的低误差区间
    low_error_intervals = {}
    for factor, stats in factor_results.items():
        low_bins = stats[stats['MAE'] < overall_mae].index.tolist()
        low_error_intervals[factor] = low_bins
        logging.info(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
        print(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
    
    # 第二步：筛选同时满足所有低误差区间的样本（严格条件）
    strict_mask = pd.Series([True]*len(df_eval))
    for factor, bins in low_error_intervals.items():
        strict_mask &= df_eval[f'{factor}_bin'].isin(bins)
    
    strict_samples = df_eval[strict_mask]
    strict_count = len(strict_samples)
    strict_prob = round(strict_count / total_samples * 100, 2)  # ✅ 修复
    strict_mae = round(strict_samples['abs_error'].mean(), 4) if strict_count > 0 else float('nan')  # ✅ 修复
    strict_rmse = round(np.sqrt(np.mean(np.square(strict_samples['residual']))), 4) if strict_count > 0 else float('nan')  # ✅ 修复
    
    # 第三步：筛选满足核心因素（丢失车辆数+排队车辆数+红灯时间）低误差区间的样本（宽松条件）
    core_factors = ['lost', 'queue_count', 'redlight_sec']
    loose_mask = pd.Series([True]*len(df_eval))
    for factor in core_factors:
        if factor in low_error_intervals:
            loose_mask &= df_eval[f'{factor}_bin'].isin(low_error_intervals[factor])
    
    loose_samples = df_eval[loose_mask]
    loose_count = len(loose_samples)
    loose_prob = round(loose_count / total_samples * 100, 2)  # ✅ 修复
    loose_mae = round(loose_samples['abs_error'].mean(), 4) if loose_count > 0 else float('nan')  # ✅ 修复
    loose_rmse = round(np.sqrt(np.mean(np.square(loose_samples['residual']))), 4) if loose_count > 0 else float('nan')  # ✅ 修复
    
    # 第四步：输出最优区间总结
    logging.info("\n" + "-"*60)
    logging.info("📈 综合评估结果总结")
    logging.info("-"*60)
    
    logging.info(f"\n✅ 严格最优区间（同时满足所有因素低误差）：")
    for factor in low_error_intervals.keys():
        logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {strict_prob}% ({strict_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {strict_mae:.4f}s, 平均RMSE: {strict_rmse:.4f}s")
    if not np.isnan(strict_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - strict_mae)/overall_mae*100, 2)}%")  # ✅ 修复
    
    logging.info(f"\n✅ 实用最优区间（满足核心因素低误差，覆盖更多样本）：")
    for factor in core_factors:
        if factor in low_error_intervals:
            logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {loose_prob}% ({loose_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {loose_mae:.4f}s, 平均RMSE: {loose_rmse:.4f}s")
    if not np.isnan(loose_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - loose_mae)/overall_mae*100, 2)}%")  # ✅ 修复
    
    # 第五步：输出误差最大的区间（用于模型改进）
    logging.info(f"\n⚠️  误差最大的前5个单因素区间：")
    all_bins = []
    for factor, stats in factor_results.items():
        for bin_name, row in stats.iterrows():
            all_bins.append({
                '因素': factor,
                '区间': bin_name,
                'MAE': row['MAE'],
                '样本数': row['样本数']
            })
    df_all_bins = pd.DataFrame(all_bins).sort_values('MAE', ascending=False).head(5)
    logging.info(df_all_bins.to_string(index=False))
    print(df_all_bins.to_string(index=False))
    
    logging.info("\n" + "="*100)
    logging.info("✅ train_model2全维度综合评估完成")
    logging.info("="*100)
    
    ###########################################################################
    # ✅ 新增：所有评估结果自动保存为CSV文件
    ###########################################################################
    logging.info("\n" + "="*80)
    logging.info("💾 正在保存评估结果到CSV文件...")
    logging.info("="*80)
    
    # 创建结果保存目录
    result_dir = "./evaluation_results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加时间戳避免覆盖
    
    # 1. 保存所有验证集样本的详细数据
    all_samples_path = f"{result_dir}/all_samples_evaluation_{timestamp}.csv"
    df_eval.to_csv(all_samples_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 所有样本详细数据已保存: {all_samples_path}")
    
    # 2. 保存每个单因素的分箱统计结果
    for factor, stats in factor_results.items():
        factor_path = f"{result_dir}/factor_{factor}_{timestamp}.csv"
        stats.to_csv(factor_path, encoding="utf-8-sig")
        logging.info(f"✅ {factor} 单因素分析结果已保存: {factor_path}")
    
    # 3. 保存整体评估总结和最优区间
    summary_data = [
        {"指标": "整体平均MAE(s)", "值": overall_mae},
        {"指标": "整体平均RMSE(s)", "值": overall_rmse},
        {"指标": "严格最优区间样本数", "值": strict_count},
        {"指标": "严格最优区间样本占比(%)", "值": strict_prob},
        {"指标": "严格最优区间平均MAE(s)", "值": strict_mae},
        {"指标": "严格最优区间平均RMSE(s)", "值": strict_rmse},
        {"指标": "严格最优区间误差降低幅度(%)", "值": round((overall_mae - strict_mae)/overall_mae*100, 2) if not np.isnan(strict_mae) else float('nan')},
        {"指标": "实用最优区间样本数", "值": loose_count},
        {"指标": "实用最优区间样本占比(%)", "值": loose_prob},
        {"指标": "实用最优区间平均MAE(s)", "值": loose_mae},
        {"指标": "实用最优区间平均RMSE(s)", "值": loose_rmse},
        {"指标": "实用最优区间误差降低幅度(%)", "值": round((overall_mae - loose_mae)/overall_mae*100, 2) if not np.isnan(loose_mae) else float('nan')}
    ]
    
    # 添加各因素低误差区间到总结
    for factor, intervals in low_error_intervals.items():
        summary_data.append({
            "指标": f"{factor} 低误差区间",
            "值": ", ".join(intervals)
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = f"{result_dir}/overall_evaluation_summary_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 整体评估总结已保存: {summary_path}")
    
    # 4. 保存误差最大的前5个区间
    top5_error_path = f"{result_dir}/top5_high_error_bins_{timestamp}.csv"
    df_all_bins.to_csv(top5_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 误差最大前5区间已保存: {top5_error_path}")
    
    # 5. 保存低误差区间汇总表
    low_error_summary = []
    for factor, intervals in low_error_intervals.items():
        for interval in intervals:
            bin_stats = factor_results[factor].loc[interval]
            low_error_summary.append({
                "因素": factor,
                "区间": interval,
                "样本数": bin_stats["样本数"],
                "样本占比(%)": bin_stats["样本占比(%)"],
                "MAE(s)": bin_stats["MAE"],
                "RMSE(s)": bin_stats["RMSE"]
            })
    df_low_error = pd.DataFrame(low_error_summary)
    low_error_path = f"{result_dir}/low_error_intervals_summary_{timestamp}.csv"
    df_low_error.to_csv(low_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 低误差区间汇总表已保存: {low_error_path}")
    
    logging.info(f"\n🎉 所有评估结果已保存到目录: {os.path.abspath(result_dir)}")
    print(f"\n🎉 所有评估结果已保存到目录: {os.path.abspath(result_dir)}")
        
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
    parser.add_argument('--nC', type=int, default=1000, help='Kmeans聚类数量，用于样本多样性选择')
    parser.add_argument('--model', type=int, default=0, help='0(mlp+jax),1(mlp+regress)')
    parser.add_argument('--fixdata', type=int, default=0, help='0(不修补),1(直接用原始数据修补),2(前后车+-5位置进行修补)')
    parser.add_argument('--goffset', type=int, default=1, help='0(jax模型中全局偏移参数为0),1(jax模型中全局偏移参数为1，默认1)')
    parser.add_argument('--trainvalmode', type=int, default=0, help='0(训练验证都无丢失),1(训练验证都有丢失)')
    parser.add_argument('--vehparamgrp', type=int, default=4, help='vehparamgrp,1.2.3.4(跟车模型几类参数)')
    args = parser.parse_args()
    main(args)

#python modelsCollect4.py --batch_size 16 --layNum 4
#python modelsCollect7.py --batch_size 32 --test_size 0.5 --epochs 150 --lr 0.00005 --unit 256 --layNum 16 --dt 0.1 --nC 500 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1
#python modelsCollect7.py --batch_size 100 --test_size 0.95 --epochs 150 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 10000 --model 1 --fixdata 0 --trainvalmode 1 --goffset 1
#python modelsCollect7.py --batch_size 100 --test_size 0.85 --epochs 150 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 1 --goffset 1
#mae3.8


###########################################################################
# ✅ 升级：train_model2全维度综合评估模块
# 新增分析因素：主车到红灯的预计到达时间 = (红灯位置 - 主车位置) / 主车速度
# 输出：单因素影响分析 + 多因素联合最优区间 + 多因素联合最差区间
###########################################################################
def trainmode2_analyzeall(model_vanish_reg, val_dataset,raw_cols):

    logging.info("="*100)
    logging.info("🚀 开始train_model2全维度综合误差影响分析")
    logging.info("="*100)
    
    # --------------------- 1. 收集所有验证集样本的完整数据 ---------------------
    print("\n正在收集验证集完整数据...")
    all_samples_data = []
    
    # 获取所有需要的列索引
    idx_lost = raw_cols.index('lost') if 'lost' in raw_cols else None
    idx_redlight = raw_cols.index('redLightRemainingTime') if 'redLightRemainingTime' in raw_cols else None
    idx_main_car_pos = raw_cols.index('main_car_position') if 'main_car_position' in raw_cols else None
    idx_main_car_speed = raw_cols.index('main_car_speed') if 'main_car_speed' in raw_cols else None
    idx_intersection_pos = raw_cols.index('intersection_pos') if 'intersection_pos' in raw_cols else None
    car_pos_cols = [col for col in raw_cols if col.startswith('car_position_')]
    idx_car_pos_list = [raw_cols.index(col) for col in car_pos_cols]
    
    # 重新遍历验证集，收集所有样本数据（独立计算，不复用历史结果）
    sample_index = 0
    for xb, yb in val_dataset:
        y_true_batch = yb.numpy().reshape(-1)
        y_pred_batch = model_vanish_reg.predict(xb, verbose=0).reshape(-1)
        residuals_batch = y_pred_batch - y_true_batch
        batch_size = len(y_true_batch)
        
        batch_raw = raw_val_np[sample_index : sample_index + batch_size]
        
        for i in range(batch_size):
            sample_raw = batch_raw[i]
            y_true = y_true_batch[i]
            y_pred = y_pred_batch[i]
            residual = residuals_batch[i]
            
            # 提取基础特征（空值保护）
            lost = int(sample_raw[idx_lost]) if (idx_lost is not None and not np.isnan(sample_raw[idx_lost])) else -1
            redlight_sec = sample_raw[idx_redlight]/30.0 if (idx_redlight is not None and not np.isnan(sample_raw[idx_redlight])) else -1
            main_pos = sample_raw[idx_main_car_pos] if (idx_main_car_pos is not None and not np.isnan(sample_raw[idx_main_car_pos])) else -1
            main_speed = sample_raw[idx_main_car_speed] if (idx_main_car_speed is not None and not np.isnan(sample_raw[idx_main_car_speed])) else -1
            inter_pos = sample_raw[idx_intersection_pos] if (idx_intersection_pos is not None and not np.isnan(sample_raw[idx_intersection_pos])) else -1
            
            # ✅ 新增：计算主车到红灯的预计到达时间（核心因素）
            if main_speed > 1e-6 and main_pos != -1 and inter_pos != -1 and inter_pos > main_pos:
                main_pos_to_red_time = (inter_pos - main_pos) / main_speed
            else:
                main_pos_to_red_time = -1.0  # 无效值标记
            
            # 计算排队车辆数
            queue_count = 0
            if idx_main_car_pos is not None and idx_intersection_pos is not None:
                for idx in idx_car_pos_list:
                    car_pos = sample_raw[idx]
                    if not np.isnan(car_pos) and car_pos != -1 and main_pos < car_pos < inter_pos:
                        queue_count += 1
            
            # 存储样本数据（新增预计到达时间）
            all_samples_data.append({
                'lost': lost,
                'redlight_sec': redlight_sec,
                'queue_count': queue_count,
                'vanish_time_true': y_true,
                'main_car_pos': main_pos,
                'main_car_speed': main_speed,
                'main_pos_to_red_time': main_pos_to_red_time,  # ✅ 新增
                'residual': residual,
                'abs_error': abs(residual)
            })
        
        sample_index += batch_size
    
    # 转换为DataFrame方便分析（空值保护）
    df_eval = pd.DataFrame(all_samples_data)
    total_samples = len(df_eval)
    
    if total_samples == 0:
        logging.error("❌ 验证集样本为空，无法进行综合评估")
        print("❌ 验证集样本为空，无法进行综合评估")
        return model_vanish_reg
        
    logging.info(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    print(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    
    # --------------------- 2. 单因素影响分析 ---------------------
    logging.info("\n" + "="*80)
    logging.info("📊 单因素对预测误差的影响分析（按MAE升序排列）")
    logging.info("="*80)
    
    # 定义各因素的分箱规则（新增预计到达时间）
    bin_rules = {
        'lost': {
            'bins': [-1, 0, 1, 2, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3+辆']
        },
        'redlight_sec': {
            'bins': [0, 3, 6, 9, 12, 15, 18, 21, 24, float('inf')],
            'labels': ['0-3s', '3-6s', '6-9s', '9-12s', '12-15s', '15-18s', '18-21s', '21-24s', '>24s']
        },
        'queue_count': {
            'bins': [-1, 0, 1, 2, 3, 4, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3辆', '4辆', '5+辆']
        },
        'vanish_time_true': {
            'bins': [0, 5, 10, 15, 20, 25, float('inf')],
            'labels': ['0-5s', '5-10s', '10-15s', '15-20s', '20-25s', '>25s']
        },
        'main_car_pos': {
            'bins': [0, 40, 45, 50, 53, float('inf')],
            'labels': ['<40m', '40-45m', '45-50m', '50-53m', '>53m']
        },
        'main_car_speed': {
            'bins': [0, 5, 10, 15, 20, float('inf')],
            'labels': ['0-5m/s', '5-10m/s', '10-15m/s', '15-20m/s', '>20m/s']
        },
        'main_pos_to_red_time': {  # ✅ 新增：预计到达时间分箱规则
            'bins': [-1, 2, 4, 6, 8, 10, float('inf')],
            'labels': ['<2s', '2-4s', '4-6s', '6-8s', '8-10s', '>10s']
        }
    }
    
    # 存储各因素的分析结果
    factor_results = {}
    
    for factor, rule in bin_rules.items():
        if factor not in df_eval.columns:
            continue
            
        # 分箱（空值处理）
        df_eval[f'{factor}_bin'] = pd.cut(
            df_eval[factor], 
            bins=rule['bins'], 
            labels=rule['labels'],
            include_lowest=True
        )
        
        # 计算各分箱的统计量
        bin_stats = df_eval.groupby(f'{factor}_bin', observed=False).agg({
            'abs_error': ['count', 'mean', 'std'],
            'residual': lambda x: np.sqrt(np.mean(np.square(x)))
        })
        
        # 重命名列并四舍五入
        bin_stats.columns = ['样本数', 'MAE', '误差标准差', 'RMSE']
        bin_stats['MAE'] = bin_stats['MAE'].apply(lambda x: round(x, 4))
        bin_stats['误差标准差'] = bin_stats['误差标准差'].apply(lambda x: round(x, 4))
        bin_stats['RMSE'] = bin_stats['RMSE'].apply(lambda x: round(x, 4))
        bin_stats['样本占比(%)'] = (bin_stats['样本数'] / total_samples * 100).apply(lambda x: round(x, 2))
        
        # 按MAE升序排列
        bin_stats = bin_stats.sort_values('MAE')
        
        # 存储结果
        factor_results[factor] = bin_stats
        
        # 输出结果
        logging.info(f"\n--- {factor} 对误差的影响 ---")
        logging.info(bin_stats.to_string())
        print(f"\n--- {factor} 对误差的影响 ---")
        print(bin_stats.to_string())
    
    # --------------------- 3. 多因素联合最优区间分析（含联合概率） ---------------------
    logging.info("\n" + "="*80)
    logging.info("🎯 多因素联合最优区间分析（误差最小且样本量充足）")
    logging.info("="*80)
    
    # 第一步：从单因素分析中提取每个因素的低误差区间（MAE < 整体平均MAE）
    overall_mae = round(df_eval['abs_error'].mean(), 4)
    overall_rmse = round(np.sqrt(np.mean(np.square(df_eval['residual']))), 4)
    logging.info(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    print(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    
    # 提取每个因素的低误差区间
    low_error_intervals = {}
    for factor, stats in factor_results.items():
        low_bins = stats[stats['MAE'] < overall_mae].index.tolist()
        low_error_intervals[factor] = low_bins
        logging.info(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
        print(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
    
    # 第二步：筛选同时满足所有低误差区间的样本（严格条件）
    strict_mask = pd.Series([True]*len(df_eval))
    for factor, bins in low_error_intervals.items():
        strict_mask &= df_eval[f'{factor}_bin'].isin(bins)
    
    strict_samples = df_eval[strict_mask]
    strict_count = len(strict_samples)
    strict_prob = round(strict_count / total_samples * 100, 2)
    strict_mae = round(strict_samples['abs_error'].mean(), 4) if strict_count > 0 else float('nan')
    strict_rmse = round(np.sqrt(np.mean(np.square(strict_samples['residual']))), 4) if strict_count > 0 else float('nan')
    
    # 第三步：筛选满足核心因素低误差区间的样本（宽松条件）
    # 核心因素：丢失车辆数 + 排队车辆数 + 红灯时间 + 预计到达时间（新增）
    core_factors = ['lost', 'queue_count', 'redlight_sec', 'main_pos_to_red_time']
    loose_mask = pd.Series([True]*len(df_eval))
    for factor in core_factors:
        if factor in low_error_intervals:
            loose_mask &= df_eval[f'{factor}_bin'].isin(low_error_intervals[factor])
    
    loose_samples = df_eval[loose_mask]
    loose_count = len(loose_samples)
    loose_prob = round(loose_count / total_samples * 100, 2)
    loose_mae = round(loose_samples['abs_error'].mean(), 4) if loose_count > 0 else float('nan')
    loose_rmse = round(np.sqrt(np.mean(np.square(loose_samples['residual']))), 4) if loose_count > 0 else float('nan')
    
    # 第四步：输出最优区间总结
    logging.info("\n" + "-"*60)
    logging.info("📈 综合评估结果总结")
    logging.info("-"*60)
    
    logging.info(f"\n✅ 严格最优区间（同时满足所有因素低误差）：")
    for factor in low_error_intervals.keys():
        logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {strict_prob}% ({strict_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {strict_mae:.4f}s, 平均RMSE: {strict_rmse:.4f}s")
    if not np.isnan(strict_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - strict_mae)/overall_mae*100, 2)}%")
    
    logging.info(f"\n✅ 实用最优区间（满足核心因素低误差，覆盖更多样本）：")
    for factor in core_factors:
        if factor in low_error_intervals:
            logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {loose_prob}% ({loose_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {loose_mae:.4f}s, 平均RMSE: {loose_rmse:.4f}s")
    if not np.isnan(loose_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - loose_mae)/overall_mae*100, 2)}%")
    
    # --------------------- 4. 多因素联合最差区间分析（新增） ---------------------
    logging.info("\n" + "="*80)
    logging.info("⚠️  多因素联合最差区间分析（误差最大且样本量充足）")
    logging.info("="*80)
    
    # 提取每个因素的高误差区间（MAE > 整体平均MAE的1.5倍）
    high_error_threshold = overall_mae * 1.5
    high_error_intervals = {}
    for factor, stats in factor_results.items():
        high_bins = stats[stats['MAE'] > high_error_threshold].index.tolist()
        high_error_intervals[factor] = high_bins
        logging.info(f"{factor} 高误差区间(MAE>{high_error_threshold:.4f}): {high_bins}")
        print(f"{factor} 高误差区间(MAE>{high_error_threshold:.4f}): {high_bins}")
    
    # 筛选同时满足所有高误差区间的样本（严格最差条件）
    worst_strict_mask = pd.Series([True]*len(df_eval))
    for factor, bins in high_error_intervals.items():
        if bins:  # 只有当该因素有高误差区间时才加入筛选
            worst_strict_mask &= df_eval[f'{factor}_bin'].isin(bins)
    
    worst_strict_samples = df_eval[worst_strict_mask]
    worst_strict_count = len(worst_strict_samples)
    worst_strict_prob = round(worst_strict_count / total_samples * 100, 2)
    worst_strict_mae = round(worst_strict_samples['abs_error'].mean(), 4) if worst_strict_count > 0 else float('nan')
    worst_strict_rmse = round(np.sqrt(np.mean(np.square(worst_strict_samples['residual']))), 4) if worst_strict_count > 0 else float('nan')
    
    # 筛选满足核心因素高误差区间的样本（宽松最差条件）
    worst_loose_mask = pd.Series([True]*len(df_eval))
    for factor in core_factors:
        if factor in high_error_intervals and high_error_intervals[factor]:
            worst_loose_mask &= df_eval[f'{factor}_bin'].isin(high_error_intervals[factor])
    
    worst_loose_samples = df_eval[worst_loose_mask]
    worst_loose_count = len(worst_loose_samples)
    worst_loose_prob = round(worst_loose_count / total_samples * 100, 2)
    worst_loose_mae = round(worst_loose_samples['abs_error'].mean(), 4) if worst_loose_count > 0 else float('nan')
    worst_loose_rmse = round(np.sqrt(np.mean(np.square(worst_loose_samples['residual']))), 4) if worst_loose_count > 0 else float('nan')
    
    # 输出最差区间总结
    logging.info(f"\n❌ 严格最差区间（同时满足所有因素高误差）：")
    for factor in high_error_intervals.keys():
        logging.info(f"   - {factor}: {high_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {worst_strict_prob}% ({worst_strict_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {worst_strict_mae:.4f}s, 平均RMSE: {worst_strict_rmse:.4f}s")
    if not np.isnan(worst_strict_mae) and overall_mae > 0:
        logging.info(f"   - 误差升高幅度: {round((worst_strict_mae - overall_mae)/overall_mae*100, 2)}%")
    
    logging.info(f"\n❌ 实用最差区间（满足核心因素高误差，覆盖更多样本）：")
    for factor in core_factors:
        if factor in high_error_intervals:
            logging.info(f"   - {factor}: {high_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {worst_loose_prob}% ({worst_loose_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {worst_loose_mae:.4f}s, 平均RMSE: {worst_loose_rmse:.4f}s")
    if not np.isnan(worst_loose_mae) and overall_mae > 0:
        logging.info(f"   - 误差升高幅度: {round((worst_loose_mae - overall_mae)/overall_mae*100, 2)}%")
    
    # 输出误差最大的前10个单因素区间（扩展到前10个）
    logging.info(f"\n⚠️  误差最大的前10个单因素区间：")
    all_bins = []
    for factor, stats in factor_results.items():
        for bin_name, row in stats.iterrows():
            all_bins.append({
                '因素': factor,
                '区间': bin_name,
                'MAE': row['MAE'],
                '样本数': row['样本数']
            })
    df_all_bins = pd.DataFrame(all_bins).sort_values('MAE', ascending=False).head(10)
    logging.info(df_all_bins.to_string(index=False))
    print(df_all_bins.to_string(index=False))
    
    logging.info("\n" + "="*100)
    logging.info("✅ train_model2全维度综合评估完成")
    logging.info("="*100)
    
    ###########################################################################
    # ✅ 所有评估结果自动保存为CSV文件（自动包含新增因素）
    ###########################################################################
    logging.info("\n" + "="*80)
    logging.info("💾 正在保存评估结果到CSV文件...")
    logging.info("="*80)
    
    # 创建结果保存目录
    result_dir = "./evaluation_results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加时间戳避免覆盖
    
    # 1. 保存所有验证集样本的详细数据（包含新增的预计到达时间）
    all_samples_path = f"{result_dir}/all_samples_evaluation_{timestamp}.csv"
    df_eval.to_csv(all_samples_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 所有样本详细数据已保存: {all_samples_path}")
    
    # 2. 保存每个单因素的分箱统计结果（包含新增的预计到达时间）
    for factor, stats in factor_results.items():
        factor_path = f"{result_dir}/factor_{factor}_{timestamp}.csv"
        stats.to_csv(factor_path, encoding="utf-8-sig")
        logging.info(f"✅ {factor} 单因素分析结果已保存: {factor_path}")
    
    # 3. 保存整体评估总结和最优/最差区间
    summary_data = [
        {"指标": "整体平均MAE(s)", "值": overall_mae},
        {"指标": "整体平均RMSE(s)", "值": overall_rmse},
        {"指标": "严格最优区间样本数", "值": strict_count},
        {"指标": "严格最优区间样本占比(%)", "值": strict_prob},
        {"指标": "严格最优区间平均MAE(s)", "值": strict_mae},
        {"指标": "严格最优区间平均RMSE(s)", "值": strict_rmse},
        {"指标": "严格最优区间误差降低幅度(%)", "值": round((overall_mae - strict_mae)/overall_mae*100, 2) if not np.isnan(strict_mae) else float('nan')},
        {"指标": "实用最优区间样本数", "值": loose_count},
        {"指标": "实用最优区间样本占比(%)", "值": loose_prob},
        {"指标": "实用最优区间平均MAE(s)", "值": loose_mae},
        {"指标": "实用最优区间平均RMSE(s)", "值": loose_rmse},
        {"指标": "实用最优区间误差降低幅度(%)", "值": round((overall_mae - loose_mae)/overall_mae*100, 2) if not np.isnan(loose_mae) else float('nan')},
        {"指标": "严格最差区间样本数", "值": worst_strict_count},
        {"指标": "严格最差区间样本占比(%)", "值": worst_strict_prob},
        {"指标": "严格最差区间平均MAE(s)", "值": worst_strict_mae},
        {"指标": "严格最差区间平均RMSE(s)", "值": worst_strict_rmse},
        {"指标": "严格最差区间误差升高幅度(%)", "值": round((worst_strict_mae - overall_mae)/overall_mae*100, 2) if not np.isnan(worst_strict_mae) else float('nan')},
        {"指标": "实用最差区间样本数", "值": worst_loose_count},
        {"指标": "实用最差区间样本占比(%)", "值": worst_loose_prob},
        {"指标": "实用最差区间平均MAE(s)", "值": worst_loose_mae},
        {"指标": "实用最差区间平均RMSE(s)", "值": worst_loose_rmse},
        {"指标": "实用最差区间误差升高幅度(%)", "值": round((worst_loose_mae - overall_mae)/overall_mae*100, 2) if not np.isnan(worst_loose_mae) else float('nan')}
    ]
    
    # 添加各因素低误差区间到总结
    for factor, intervals in low_error_intervals.items():
        summary_data.append({
            "指标": f"{factor} 低误差区间",
            "值": ", ".join(intervals)
        })
    
    # 添加各因素高误差区间到总结
    for factor, intervals in high_error_intervals.items():
        summary_data.append({
            "指标": f"{factor} 高误差区间",
            "值": ", ".join(intervals)
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = f"{result_dir}/overall_evaluation_summary_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 整体评估总结已保存: {summary_path}")
    
    # 4. 保存误差最大的前10个区间
    top10_error_path = f"{result_dir}/top10_high_error_bins_{timestamp}.csv"
    df_all_bins.to_csv(top10_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 误差最大前10区间已保存: {top10_error_path}")
    
    # 5. 保存低误差区间汇总表
    low_error_summary = []
    for factor, intervals in low_error_intervals.items():
        for interval in intervals:
            bin_stats = factor_results[factor].loc[interval]
            low_error_summary.append({
                "因素": factor,
                "区间": interval,
                "样本数": bin_stats["样本数"],
                "样本占比(%)": bin_stats["样本占比(%)"],
                "MAE(s)": bin_stats["MAE"],
                "RMSE(s)": bin_stats["RMSE"]
            })
    df_low_error = pd.DataFrame(low_error_summary)
    low_error_path = f"{result_dir}/low_error_intervals_summary_{timestamp}.csv"
    df_low_error.to_csv(low_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 低误差区间汇总表已保存: {low_error_path}")
    
    # 6. 保存高误差区间汇总表（新增）
    high_error_summary = []
    for factor, intervals in high_error_intervals.items():
        for interval in intervals:
            bin_stats = factor_results[factor].loc[interval]
            high_error_summary.append({
                "因素": factor,
                "区间": interval,
                "样本数": bin_stats["样本数"],
                "样本占比(%)": bin_stats["样本占比(%)"],
                "MAE(s)": bin_stats["MAE"],
                "RMSE(s)": bin_stats["RMSE"]
            })
    df_high_error = pd.DataFrame(high_error_summary)
    high_error_path = f"{result_dir}/high_error_intervals_summary_{timestamp}.csv"
    df_high_error.to_csv(high_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 高误差区间汇总表已保存: {high_error_path}")
    
    logging.info(f"\n🎉 所有评估结果已保存到目录: {os.path.abspath(result_dir)}")
    print(f"\n🎉 所有评估结果已保存到目录: {os.path.abspath(result_dir)}")
        
    
    
        
    ###########################################################################
    # ✅ 新增：train_model（模型0）全维度综合评估模块
    # 分析：丢失车辆数、红灯剩余时间、排队车辆数、vanishTime大小、主车位置、主车速度、主车到红灯预计到达时间
    # 输出：单因素影响分析 + 多因素联合最优区间 + 多因素联合最差区间 + CSV结果保存
    ###########################################################################
def trainmode_analyzeall(model, val_dataset,raw_columns_list):
       

    logging.info("="*100)
    logging.info("🚀 开始train_model（模型0：MLP+JAX仿真）全维度综合误差影响分析")
    logging.info("="*100)
    
    # --------------------- 1. 收集所有验证集样本的完整数据 ---------------------
    print("\n正在收集验证集完整数据...")
    all_samples_data = []
    
    # 获取所有需要的列索引（复用已定义的变量，新增主车速度索引）
    idx_main_car_speed = raw_columns_list.index('main_car_speed') if 'main_car_speed' in raw_columns_list else None
    
    # 重新遍历验证集，收集所有样本数据（独立计算，不复用历史结果）
    for xb, yb, rb in val_dataset:
        # 计算残差（预测值 - 真实值）
        residuals_batch = val_step(xb, yb, rb).numpy().flatten()
        y_true_batch = yb.numpy().flatten()
        rb_np = rb.numpy()
        batch_size = len(residuals_batch)
        
        for i in range(batch_size):
            sample_raw = rb_np[i]
            y_true = y_true_batch[i]
            residual = residuals_batch[i]
            
            # 提取基础特征（空值保护）
            lost = int(sample_raw[idx_lost]) if (idx_lost is not None and not np.isnan(sample_raw[idx_lost])) else -1
            redlight_sec = sample_raw[idx_redlight]/30.0 if (idx_redlight is not None and not np.isnan(sample_raw[idx_redlight])) else -1
            main_pos = sample_raw[idx_main_car_pos] if (idx_main_car_pos is not None and not np.isnan(sample_raw[idx_main_car_pos])) else -1
            main_speed = sample_raw[idx_main_car_speed] if (idx_main_car_speed is not None and not np.isnan(sample_raw[idx_main_car_speed])) else -1
            inter_pos = sample_raw[idx_intersection_pos] if (idx_intersection_pos is not None and not np.isnan(sample_raw[idx_intersection_pos])) else -1
            
            # ✅ 新增：计算主车到红灯的预计到达时间（核心因素）
            if main_speed > 1e-6 and main_pos != -1 and inter_pos != -1 and inter_pos > main_pos:
                main_pos_to_red_time = (inter_pos - main_pos) / main_speed
            else:
                main_pos_to_red_time = -1.0  # 无效值标记
            
            # 计算排队车辆数
            queue_count = 0
            if idx_main_car_pos is not None and idx_intersection_pos is not None:
                for idx in idx_car_pos_list:
                    car_pos = sample_raw[idx]
                    if not np.isnan(car_pos) and car_pos != -1 and main_pos < car_pos < inter_pos:
                        queue_count += 1
            
            # 存储样本数据
            all_samples_data.append({
                'lost': lost,
                'redlight_sec': redlight_sec,
                'queue_count': queue_count,
                'vanish_time_true': y_true,
                'main_car_pos': main_pos,
                'main_car_speed': main_speed,
                'main_pos_to_red_time': main_pos_to_red_time,
                'residual': residual,
                'abs_error': abs(residual)
            })
    
    # 转换为DataFrame方便分析（空值保护）
    df_eval = pd.DataFrame(all_samples_data)
    total_samples = len(df_eval)
    
    if total_samples == 0:
        logging.error("❌ 验证集样本为空，无法进行综合评估")
        print("❌ 验证集样本为空，无法进行综合评估")
        return model
        
    logging.info(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    print(f"共收集到 {total_samples} 个验证集样本用于综合评估")
    
    # --------------------- 2. 单因素影响分析 ---------------------
    logging.info("\n" + "="*80)
    logging.info("📊 单因素对预测误差的影响分析（按MAE升序排列）")
    logging.info("="*80)
    
    # 定义各因素的分箱规则（与train_model2完全一致，便于对比）
    bin_rules = {
        'lost': {
            'bins': [-1, 0, 1, 2, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3+辆']
        },
        'redlight_sec': {
            'bins': [0, 3, 6, 9, 12, 15, 18, 21, 24, float('inf')],
            'labels': ['0-3s', '3-6s', '6-9s', '9-12s', '12-15s', '15-18s', '18-21s', '21-24s', '>24s']
        },
        'queue_count': {
            'bins': [-1, 0, 1, 2, 3, 4, float('inf')],
            'labels': ['0辆', '1辆', '2辆', '3辆', '4辆', '5+辆']
        },
        'vanish_time_true': {
            'bins': [0, 5, 10, 15, 20, 25, float('inf')],
            'labels': ['0-5s', '5-10s', '10-15s', '15-20s', '20-25s', '>25s']
        },
        'main_car_pos': {
            'bins': [0, 40, 45, 50, 53, float('inf')],
            'labels': ['<40m', '40-45m', '45-50m', '50-53m', '>53m']
        },
        'main_car_speed': {
            'bins': [0, 5, 10, 15, 20, float('inf')],
            'labels': ['0-5m/s', '5-10m/s', '10-15m/s', '15-20m/s', '>20m/s']
        },
        'main_pos_to_red_time': {
            'bins': [-1, 2, 4, 6, 8, 10, float('inf')],
            'labels': ['<2s', '2-4s', '4-6s', '6-8s', '8-10s', '>10s']
        }
    }
    
    # 存储各因素的分析结果
    factor_results = {}
    
    for factor, rule in bin_rules.items():
        if factor not in df_eval.columns:
            continue
            
        # 分箱（空值处理）
        df_eval[f'{factor}_bin'] = pd.cut(
            df_eval[factor], 
            bins=rule['bins'], 
            labels=rule['labels'],
            include_lowest=True
        )
        
        # 计算各分箱的统计量
        bin_stats = df_eval.groupby(f'{factor}_bin', observed=False).agg({
            'abs_error': ['count', 'mean', 'std'],
            'residual': lambda x: np.sqrt(np.mean(np.square(x)))
        })
        
        # 重命名列并四舍五入
        bin_stats.columns = ['样本数', 'MAE', '误差标准差', 'RMSE']
        bin_stats['MAE'] = bin_stats['MAE'].apply(lambda x: round(x, 4))
        bin_stats['误差标准差'] = bin_stats['误差标准差'].apply(lambda x: round(x, 4))
        bin_stats['RMSE'] = bin_stats['RMSE'].apply(lambda x: round(x, 4))
        bin_stats['样本占比(%)'] = (bin_stats['样本数'] / total_samples * 100).apply(lambda x: round(x, 2))
        
        # 按MAE升序排列
        bin_stats = bin_stats.sort_values('MAE')
        
        # 存储结果
        factor_results[factor] = bin_stats
        
        # 输出结果
        logging.info(f"\n--- {factor} 对误差的影响 ---")
        logging.info(bin_stats.to_string())
        print(f"\n--- {factor} 对误差的影响 ---")
        print(bin_stats.to_string())
    
    # --------------------- 3. 多因素联合最优区间分析（含联合概率） ---------------------
    logging.info("\n" + "="*80)
    logging.info("🎯 多因素联合最优区间分析（误差最小且样本量充足）")
    logging.info("="*80)
    
    # 第一步：从单因素分析中提取每个因素的低误差区间（MAE < 整体平均MAE）
    overall_mae = round(df_eval['abs_error'].mean(), 4)
    overall_rmse = round(np.sqrt(np.mean(np.square(df_eval['residual']))), 4)
    logging.info(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    print(f"\n整体平均MAE: {overall_mae:.4f}s, 整体平均RMSE: {overall_rmse:.4f}s")
    
    # 提取每个因素的低误差区间
    low_error_intervals = {}
    for factor, stats in factor_results.items():
        low_bins = stats[stats['MAE'] < overall_mae].index.tolist()
        low_error_intervals[factor] = low_bins
        logging.info(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
        print(f"{factor} 低误差区间(MAE<{overall_mae}): {low_bins}")
    
    # 第二步：筛选同时满足所有低误差区间的样本（严格条件）
    strict_mask = pd.Series([True]*len(df_eval))
    for factor, bins in low_error_intervals.items():
        strict_mask &= df_eval[f'{factor}_bin'].isin(bins)
    
    strict_samples = df_eval[strict_mask]
    strict_count = len(strict_samples)
    strict_prob = round(strict_count / total_samples * 100, 2)
    strict_mae = round(strict_samples['abs_error'].mean(), 4) if strict_count > 0 else float('nan')
    strict_rmse = round(np.sqrt(np.mean(np.square(strict_samples['residual']))), 4) if strict_count > 0 else float('nan')
    
    # 第三步：筛选满足核心因素低误差区间的样本（宽松条件）
    # 核心因素：丢失车辆数 + 排队车辆数 + 红灯时间 + 预计到达时间
    core_factors = ['lost', 'queue_count', 'redlight_sec', 'main_pos_to_red_time']
    loose_mask = pd.Series([True]*len(df_eval))
    for factor in core_factors:
        if factor in low_error_intervals:
            loose_mask &= df_eval[f'{factor}_bin'].isin(low_error_intervals[factor])
    
    loose_samples = df_eval[loose_mask]
    loose_count = len(loose_samples)
    loose_prob = round(loose_count / total_samples * 100, 2)
    loose_mae = round(loose_samples['abs_error'].mean(), 4) if loose_count > 0 else float('nan')
    loose_rmse = round(np.sqrt(np.mean(np.square(loose_samples['residual']))), 4) if loose_count > 0 else float('nan')
    
    # 第四步：输出最优区间总结
    logging.info("\n" + "-"*60)
    logging.info("📈 综合评估结果总结")
    logging.info("-"*60)
    
    logging.info(f"\n✅ 严格最优区间（同时满足所有因素低误差）：")
    for factor in low_error_intervals.keys():
        logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {strict_prob}% ({strict_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {strict_mae:.4f}s, 平均RMSE: {strict_rmse:.4f}s")
    if not np.isnan(strict_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - strict_mae)/overall_mae*100, 2)}%")
    
    logging.info(f"\n✅ 实用最优区间（满足核心因素低误差，覆盖更多样本）：")
    for factor in core_factors:
        if factor in low_error_intervals:
            logging.info(f"   - {factor}: {low_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {loose_prob}% ({loose_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {loose_mae:.4f}s, 平均RMSE: {loose_rmse:.4f}s")
    if not np.isnan(loose_mae) and overall_mae > 0:
        logging.info(f"   - 误差降低幅度: {round((overall_mae - loose_mae)/overall_mae*100, 2)}%")
    
    # --------------------- 4. 多因素联合最差区间分析 ---------------------
    logging.info("\n" + "="*80)
    logging.info("⚠️  多因素联合最差区间分析（误差最大且样本量充足）")
    logging.info("="*80)
    
    # 提取每个因素的高误差区间（MAE > 整体平均MAE的1.5倍）
    high_error_threshold = overall_mae * 1.5
    high_error_intervals = {}
    for factor, stats in factor_results.items():
        high_bins = stats[stats['MAE'] > high_error_threshold].index.tolist()
        high_error_intervals[factor] = high_bins
        logging.info(f"{factor} 高误差区间(MAE>{high_error_threshold:.4f}): {high_bins}")
        print(f"{factor} 高误差区间(MAE>{high_error_threshold:.4f}): {high_bins}")
    
    # 筛选同时满足所有高误差区间的样本（严格最差条件）
    worst_strict_mask = pd.Series([True]*len(df_eval))
    for factor, bins in high_error_intervals.items():
        if bins:  # 只有当该因素有高误差区间时才加入筛选
            worst_strict_mask &= df_eval[f'{factor}_bin'].isin(bins)
    
    worst_strict_samples = df_eval[worst_strict_mask]
    worst_strict_count = len(worst_strict_samples)
    worst_strict_prob = round(worst_strict_count / total_samples * 100, 2)
    worst_strict_mae = round(worst_strict_samples['abs_error'].mean(), 4) if worst_strict_count > 0 else float('nan')
    worst_strict_rmse = round(np.sqrt(np.mean(np.square(worst_strict_samples['residual']))), 4) if worst_strict_count > 0 else float('nan')
    
    # 筛选满足核心因素高误差区间的样本（宽松最差条件）
    worst_loose_mask = pd.Series([True]*len(df_eval))
    for factor in core_factors:
        if factor in high_error_intervals and high_error_intervals[factor]:
            worst_loose_mask &= df_eval[f'{factor}_bin'].isin(high_error_intervals[factor])
    
    worst_loose_samples = df_eval[worst_loose_mask]
    worst_loose_count = len(worst_loose_samples)
    worst_loose_prob = round(worst_loose_count / total_samples * 100, 2)
    worst_loose_mae = round(worst_loose_samples['abs_error'].mean(), 4) if worst_loose_count > 0 else float('nan')
    worst_loose_rmse = round(np.sqrt(np.mean(np.square(worst_loose_samples['residual']))), 4) if worst_loose_count > 0 else float('nan')
    
    # 输出最差区间总结
    logging.info(f"\n❌ 严格最差区间（同时满足所有因素高误差）：")
    for factor in high_error_intervals.keys():
        logging.info(f"   - {factor}: {high_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {worst_strict_prob}% ({worst_strict_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {worst_strict_mae:.4f}s, 平均RMSE: {worst_strict_rmse:.4f}s")
    if not np.isnan(worst_strict_mae) and overall_mae > 0:
        logging.info(f"   - 误差升高幅度: {round((worst_strict_mae - overall_mae)/overall_mae*100, 2)}%")
    
    logging.info(f"\n❌ 实用最差区间（满足核心因素高误差，覆盖更多样本）：")
    for factor in core_factors:
        if factor in high_error_intervals:
            logging.info(f"   - {factor}: {high_error_intervals[factor]}")
    logging.info(f"   - 样本占比: {worst_loose_prob}% ({worst_loose_count}/{total_samples})")
    logging.info(f"   - 平均MAE: {worst_loose_mae:.4f}s, 平均RMSE: {worst_loose_rmse:.4f}s")
    if not np.isnan(worst_loose_mae) and overall_mae > 0:
        logging.info(f"   - 误差升高幅度: {round((worst_loose_mae - overall_mae)/overall_mae*100, 2)}%")
    
    # 输出误差最大的前10个单因素区间
    logging.info(f"\n⚠️  误差最大的前10个单因素区间：")
    all_bins = []
    for factor, stats in factor_results.items():
        for bin_name, row in stats.iterrows():
            all_bins.append({
                '因素': factor,
                '区间': bin_name,
                'MAE': row['MAE'],
                '样本数': row['样本数']
            })
    df_all_bins = pd.DataFrame(all_bins).sort_values('MAE', ascending=False).head(10)
    logging.info(df_all_bins.to_string(index=False))
    print(df_all_bins.to_string(index=False))
    
    logging.info("\n" + "="*100)
    logging.info("✅ train_model（模型0）全维度综合评估完成")
    logging.info("="*100)
    
    ###########################################################################
    # ✅ 所有评估结果自动保存为CSV文件（与模型1结果分开存储）
    ###########################################################################
    logging.info("\n" + "="*80)
    logging.info("💾 正在保存模型0评估结果到CSV文件...")
    logging.info("="*80)
    
    # 创建模型0专属结果保存目录，避免与模型1覆盖
    result_dir = "./evaluation_results_model0"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加时间戳避免覆盖
    
    # 1. 保存所有验证集样本的详细数据
    all_samples_path = f"{result_dir}/model0_all_samples_evaluation_{timestamp}.csv"
    df_eval.to_csv(all_samples_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 所有样本详细数据已保存: {all_samples_path}")
    
    # 2. 保存每个单因素的分箱统计结果
    for factor, stats in factor_results.items():
        factor_path = f"{result_dir}/model0_factor_{factor}_{timestamp}.csv"
        stats.to_csv(factor_path, encoding="utf-8-sig")
        logging.info(f"✅ {factor} 单因素分析结果已保存: {factor_path}")
    
    # 3. 保存整体评估总结和最优/最差区间
    summary_data = [
        {"指标": "整体平均MAE(s)", "值": overall_mae},
        {"指标": "整体平均RMSE(s)", "值": overall_rmse},
        {"指标": "严格最优区间样本数", "值": strict_count},
        {"指标": "严格最优区间样本占比(%)", "值": strict_prob},
        {"指标": "严格最优区间平均MAE(s)", "值": strict_mae},
        {"指标": "严格最优区间平均RMSE(s)", "值": strict_rmse},
        {"指标": "严格最优区间误差降低幅度(%)", "值": round((overall_mae - strict_mae)/overall_mae*100, 2) if not np.isnan(strict_mae) else float('nan')},
        {"指标": "实用最优区间样本数", "值": loose_count},
        {"指标": "实用最优区间样本占比(%)", "值": loose_prob},
        {"指标": "实用最优区间平均MAE(s)", "值": loose_mae},
        {"指标": "实用最优区间平均RMSE(s)", "值": loose_rmse},
        {"指标": "实用最优区间误差降低幅度(%)", "值": round((overall_mae - loose_mae)/overall_mae*100, 2) if not np.isnan(loose_mae) else float('nan')},
        {"指标": "严格最差区间样本数", "值": worst_strict_count},
        {"指标": "严格最差区间样本占比(%)", "值": worst_strict_prob},
        {"指标": "严格最差区间平均MAE(s)", "值": worst_strict_mae},
        {"指标": "严格最差区间平均RMSE(s)", "值": worst_strict_rmse},
        {"指标": "严格最差区间误差升高幅度(%)", "值": round((worst_strict_mae - overall_mae)/overall_mae*100, 2) if not np.isnan(worst_strict_mae) else float('nan')},
        {"指标": "实用最差区间样本数", "值": worst_loose_count},
        {"指标": "实用最差区间样本占比(%)", "值": worst_loose_prob},
        {"指标": "实用最差区间平均MAE(s)", "值": worst_loose_mae},
        {"指标": "实用最差区间平均RMSE(s)", "值": worst_loose_rmse},
        {"指标": "实用最差区间误差升高幅度(%)", "值": round((worst_loose_mae - overall_mae)/overall_mae*100, 2) if not np.isnan(worst_loose_mae) else float('nan')}
    ]
    
    # 添加各因素低误差区间到总结
    for factor, intervals in low_error_intervals.items():
        summary_data.append({
            "指标": f"{factor} 低误差区间",
            "值": ", ".join(intervals)
        })
    
    # 添加各因素高误差区间到总结
    for factor, intervals in high_error_intervals.items():
        summary_data.append({
            "指标": f"{factor} 高误差区间",
            "值": ", ".join(intervals)
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = f"{result_dir}/model0_overall_evaluation_summary_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 整体评估总结已保存: {summary_path}")
    
    # 4. 保存误差最大的前10个区间
    top10_error_path = f"{result_dir}/model0_top10_high_error_bins_{timestamp}.csv"
    df_all_bins.to_csv(top10_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 误差最大前10区间已保存: {top10_error_path}")
    
    # 5. 保存低误差区间汇总表
    low_error_summary = []
    for factor, intervals in low_error_intervals.items():
        for interval in intervals:
            bin_stats = factor_results[factor].loc[interval]
            low_error_summary.append({
                "因素": factor,
                "区间": interval,
                "样本数": bin_stats["样本数"],
                "样本占比(%)": bin_stats["样本占比(%)"],
                "MAE(s)": bin_stats["MAE"],
                "RMSE(s)": bin_stats["RMSE"]
            })
    df_low_error = pd.DataFrame(low_error_summary)
    low_error_path = f"{result_dir}/model0_low_error_intervals_summary_{timestamp}.csv"
    df_low_error.to_csv(low_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 低误差区间汇总表已保存: {low_error_path}")
    
    # 6. 保存高误差区间汇总表
    high_error_summary = []
    for factor, intervals in high_error_intervals.items():
        for interval in intervals:
            bin_stats = factor_results[factor].loc[interval]
            high_error_summary.append({
                "因素": factor,
                "区间": interval,
                "样本数": bin_stats["样本数"],
                "样本占比(%)": bin_stats["样本占比(%)"],
                "MAE(s)": bin_stats["MAE"],
                "RMSE(s)": bin_stats["RMSE"]
            })
    df_high_error = pd.DataFrame(high_error_summary)
    high_error_path = f"{result_dir}/model0_high_error_intervals_summary_{timestamp}.csv"
    df_high_error.to_csv(high_error_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 高误差区间汇总表已保存: {high_error_path}")
    
    logging.info(f"\n🎉 模型0所有评估结果已保存到目录: {os.path.abspath(result_dir)}")
    print(f"\n🎉 模型0所有评估结果已保存到目录: {os.path.abspath(result_dir)}")
        
    return model












#python modelsCollect7.py --batch_size 100 --test_size 0.85 --epochs 180 --lr 0.0005 --unit 256 --layNum 256 --dt 0.1 --nC 2000 --model 0 --fixdata 2 --trainvalmode 1 --goffset 1 2>&1 | tee model0_fixedata2_humanRevmove_output.log


#python modelsCollect7.py --batch_size 100 --test_size 0.95 --epochs 150 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 10000 --model 1 --fixdata 2 --trainvalmode 1 --goffset 1 2>&1 | tee model1_fixedata2_humanRevmove_output.log