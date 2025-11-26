
'''

生成一个函数：基于python模拟退火算法，搜索所有车辆跟车模型的最佳参数
1.车辆IDM需要搜索最佳参数:
安全车头时距 T,取值范围为（0.5,2）
期望速度 (m/s) v0: 取值范围为（40/3.6，60/3.6）
最小跟车距离为 s0， 取值范围为（1，3）
最大加速度 (m/s²) a,取值范围为（1，3）
舒适减速度 (m/s²) a,取值范围为（1，6）
反应时间 (s): float = 0.01 取值范围为（0.01，1）
2.参考函数model_with_SimulCal_SimpleResnet中的输入样本处理和输出结果，不用神经网络，用模拟退火以及其他2种以上最优化方法，用于对比
3.参考modelsCollect2.py中代码
4.必须要用SciPy中的模拟退火算法实现，以及另外2种以上全局最优化方法实现
5.给出最优参数下预测结果与实际结果的误差的均值和方差，以及其他统计指标，用于评估最优参数的性能
'''

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing, differential_evolution, shgo
from tqdm import tqdm
import time

# 假设 pyGameInterface2 在同一目录下，或者在 python path 中
# 如果 pyGameInterface2.py 在上级目录或其他位置，请调整 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyGameInterface2 import TrafficSimulator, VehicleParams
from scipy.stats import pearsonr
def calculate_simulation_rmse(params_array, data_samples):
    """
    目标函数：计算给定参数下的均方根误差 (RMSE)。
    SciPy 的优化算法会尝试最小化这个函数的返回值。
    
    Args:
        params_array: list/array, 顺序为 [v0, T, s0, a, b, rtime]
        data_samples: DataFrame, 用于评估的样本数据
    """
    # 1. 解包参数
    v0, T, s0, a, b, rtime = params_array
    
    # 2. 构建参数对象
    # 注意：pyGameInterface2 中的 VehicleParams 需要支持 rtime 字段
    # 如果 VehicleParams 定义中没有 rtime，请确保在 pyGameInterface2.py 中添加了该字段
    vehicle_params = VehicleParams(
        v0=v0,
        T=T,
        s0=s0,
        a=a,
        b=b,
        delta=4.0, # 固定值
        length=5.0, # 固定值
        rtime=rtime 
    )

    errors_sq = []
    sim_times = []
    true_times = []
    
    # 3. 遍历样本进行模拟
    # 为了优化速度，这里不使用 tqdm，因为会被优化器频繁调用
    for _, row in data_samples.iterrows():
        # 初始化模拟器
        print(f"Simulating sample with carID {row['carID']}...\n",row)
        simulator = TrafficSimulator(vehicle_params, time_step=0.1, intersection_pos=1000.0)
        
        # 设置红灯 (数据单位通常是帧，需转换为秒，假设30fps)
        red_light_sec = row['redLightRemainingTime'] / 30.0
        simulator.set_red_light(red_light_sec)
        
        car_id = row['carID']
        distance = row['car_position']
        speed = row['car_speed']
        landid = row['lane']

        end_of_lane_coordsNow = {
        5: (53.04760881,54.77239228),
        6: (53.13174459,57.71714455),
        7: (53.30001614,61.79772985)}
        intersection_pos = end_of_lane_coordsNow[landid][0]  # Use x-coordinate as intersection position
        simulator.intersection_pos = intersection_pos

        # Add vehicles from the sample
        for i in range(20):
            pos_col = f'car_position_{i}'
            speed_col = f'car_speed_{i}'
            if pos_col in row and row[pos_col] != -1:
            # Assuming a unique ID for these other cars is needed for the simulator
            # Using a large number + index to avoid collision with main carID

                other_car_id = 100 + i 
                simulator.add_vehicles([
                    {'id': other_car_id,'distance': row[pos_col],'speed': row[speed_col]}
                ])
                
                if row[pos_col] == distance:
                    main_car_id = other_car_id  # Identify the main car's ID

        
        #2.运行模拟，记录每次模拟main_car_id的首次has_passed为True时的time,并记录
        #recordD的数据是time	red_light_remaining	id	distance	speed	acceleration	has_passed	waiting_time


        recordDF = simulator.run_simulation(max_duration=100)

       
        main_car_data = recordDF[recordDF['id'] == main_car_id]
        passed_data = main_car_data[main_car_data['has_passed'] == True]
        if passed_data.empty:
            # 如果车辆未通过路口，给予一个较大的惩罚值
            errors_sq.append((1000.0 - (row['time_to_vanish'] / 30.0)) ** 2)
            continue
        sim_time = passed_data.iloc[0]['time']  # Get the first time where has_passed is True
        true_time = row['time_to_vanish'] / 30.0
        errors_sq.append((sim_time - true_time) ** 2)
        sim_times.append(sim_time)
        true_times.append(true_time)

   
    if len(sim_times) > 1 and len(true_times) > 1:
        print(f"预测时间 (Simulated Times): {[f'{t:.2f}' for t in sim_times]}")
        print(f"真实时间 (True Times):       {[f'{t:.2f}' for t in true_times]}")
    rmse = np.sqrt(np.mean(errors_sq))
    return rmse

def evaluate_performance(best_params, test_samples, method_name):
    """
    在测试集上评估最优参数，计算详细统计指标
    """
    print(f"\n>>> 正在评估 [{method_name}] 的最优参数性能...")
    
    errors = []
    sim_times = []
    true_times = []
    
    # 解包参数
    v0, T, s0, a, b, rtime = best_params
    vehicle_params = VehicleParams(v0=v0, T=T, s0=s0, a=a, b=b, rtime=rtime)
    
    for _, row in test_samples.iterrows():
        simulator = TrafficSimulator(vehicle_params, time_step=0.1, intersection_pos=1000.0)
        
        # 设置红灯 (数据单位通常是帧，需转换为秒，假设30fps)
        red_light_sec = row['redLightRemainingTime'] / 30.0
        simulator.set_red_light(red_light_sec)
        
        car_id = row['carID']
        distance = row['car_position']
        speed = row['car_speed']
        landid = row['lane']

        
        vehicle_list = []
        main_car_id = -1
        for i in range(20):
            pos_col = f'car_position_{i}'
            speed_col = f'car_speed_{i}'
            if pos_col in row and row[pos_col] != -1 and not pd.isna(row[pos_col]):
                curr_id = 100 + i
                vehicle_list.append({'id': curr_id, 'distance': row[pos_col], 'speed': row[speed_col]})
                if abs(row[pos_col] - row['car_position']) < 0.1:
                    main_car_id = curr_id
        
      
        simulator.add_vehicles(vehicle_list)


        end_of_lane_coordsNow = {
        5: (53.04760881,54.77239228),
        6: (53.13174459,57.71714455),
        7: (53.30001614,61.79772985)}
        intersection_pos = end_of_lane_coordsNow[landid][0]  # Use x-coordinate as intersection position
        simulator.intersection_pos = intersection_pos

        
        df = simulator.run_simulation(max_duration=120)
     
        passed = df[(df['id'] == main_car_id) & (df['has_passed'] == True)]
           
        sim_t = passed.iloc[0]['time']
        true_t = row['time_to_vanish'] / 30.0
        
        errors.append(sim_t - true_t)
        sim_times.append(sim_t)
        true_times.append(true_t)
    
    errors = np.array(errors)

    # 可以在这里添加更多评估指标，例如相关系数
    
    # 确保 sim_times 和 true_times 长度大于1以计算相关性
    if len(sim_times) > 1 and len(true_times) > 1:
      
        print(f"预测时间 (Simulated Times): {[f'{t:.2f}' for t in sim_times]}")
        print(f"真实时间 (True Times):       {[f'{t:.2f}' for t in true_times]}")


    if len(errors) > 0:
        print(f"--- {method_name} 评估结果 ---")
        print(f"最优参数: v0={v0:.2f}, T={T:.2f}, s0={s0:.2f}, a={a:.2f}, b={b:.2f}, rtime={rtime:.3f}")
        print(f"误差均值 (Mean Error): {np.mean(errors):.4f} s")
        print(f"误差方差 (Variance): {np.var(errors):.4f}")
        print(f"平均绝对误差 (MAE): {np.mean(np.abs(errors)):.4f} s")
        print(f"均方根误差 (RMSE): {np.sqrt(np.mean(errors**2)):.4f} s")
        print(f"最大误差: {np.max(np.abs(errors)):.4f} s")
    else:
        print("警告：测试集评估未产生有效数据。")

def optimize_idm_params_scipy():
    """
    主函数：使用 SciPy 的三种全局优化算法搜索最佳 IDM 参数
    """
    print("="*30)
    print("开始 IDM 参数全局优化 (SciPy)")
    print("="*30)

    # 1. 加载数据
    try:
        # 假设文件名，请根据实际情况修改
        csv_path = 'trainsamples_lane_5_6_7.csv' 
        if not os.path.exists(csv_path):
            print(f"错误：找不到文件 {csv_path}")
            return
        
        full_df = pd.read_csv(csv_path)
   
        
        # 划分训练集（用于优化）和测试集（用于评估）
        # 为了优化速度，训练集取较小样本 (例如 30 个)，测试集取 100 个
        train_df = full_df.sample(n=50, random_state=42)
        test_df = full_df.drop(train_df.index).sample(n=100, random_state=99)
        
        print(f"数据加载完成: 训练样本 {len(train_df)}, 测试样本 {len(test_df)}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 定义参数边界 (Bounds)
    # 顺序: [v0, T, s0, a, b, rtime]
    # v0: (40/3.6, 60/3.6) -> (11.11, 16.67)
    bounds = [
        (50/3.6, 60/3.6), # v0
        (0.5, 1.0),       # T
        (1.0, 3.0),       # s0
        (2.0, 3.0),       # a
        (2.0, 6.0),       # b
        (0.01, 1.0)       # rtime
    ]
    
    print(f"参数边界: {bounds}")

    # 3. 执行优化算法
    
    # --- 方法 A: 模拟退火 (Dual Annealing) ---
    start_time = time.time()
    print("\n[1/3] 正在运行 SciPy Dual Annealing (模拟退火)...")
    
    # 创建一个 tqdm 进度条来监控迭代
    progress_bar = tqdm(total=50, desc="Dual Annealing", unit="iter")

    def callback(x, f, context):
        """
        回调函数，用于更新进度条并打印当前最佳结果。
        x: 当前最优参数
        f: 当前最优函数值 (loss)
        context: 优化器上下文 ('nit', 'nfev' 等)
        """
        print(f"当前最佳参数: v0={x[0]:.2f}, T={x[1]:.2f}, s0={x[2]:.2f}, a={x[3]:.2f}, b={x[4]:.2f}, rtime={x[5]:.3f} | Loss: {f:.4f}")
        print(f"当前context: {context}")
        print("-"*50)

        progress_bar.update(1) # 每次迭代更新进度条
        progress_bar.set_postfix(loss=f"{f:.4f}")
        

    res_da = dual_annealing(
        func=calculate_simulation_rmse, 
        bounds=bounds, 
        args=(train_df,),
        maxiter=50,  # 迭代次数，根据时间调整
        seed=42,
        callback=callback
    )
  
    print(f"Dual Annealing 完成，耗时 {time.time()-start_time:.1f}s")
    print(f"  -> 最佳 Loss: {res_da.fun:.4f}")
    evaluate_performance(res_da.x, test_df, "Dual Annealing")
    return 
    # --- 方法 B: 差分进化 (Differential Evolution) ---
    print("\n[2/3] 正在运行 SciPy Differential Evolution (差分进化)...")
    start_time = time.time()
    res_de = differential_evolution(
        func=calculate_simulation_rmse,
        bounds=bounds,
        args=(train_df,),
        maxiter=10, # 这里的 maxiter 是指代数
        popsize=5,  # 种群大小
        seed=42,
        workers=1   # 如果模拟器不是线程安全的，设为1；如果是，设为-1并行
    )
    print(f"Differential Evolution 完成，耗时 {time.time()-start_time:.1f}s")
    print(f"  -> 最佳 Loss: {res_de.fun:.4f}")
    evaluate_performance(res_de.x, test_df, "Differential Evolution")

    # --- 方法 C: SHGO (Simplicial Homology Global Optimization) ---
    print("\n[3/3] 正在运行 SciPy SHGO (单纯形同调全局优化)...")
    # SHGO 适合低维边界约束问题
    start_time = time.time()
    res_shgo = shgo(
        func=calculate_simulation_rmse,
        bounds=bounds,
        args=(train_df,),
        options={'maxiter': 3, 'f_min': 0} # 限制迭代以节省时间
    )
    print(f"SHGO 完成，耗时 {time.time()-start_time:.1f}s")
    print(f"  -> 最佳 Loss: {res_shgo.fun:.4f}")
    evaluate_performance(res_shgo.x, test_df, "SHGO")

if __name__ == "__main__":
    sys.stdout = open('output.log', 'w',encoding='utf-8')  # 屏蔽输出以加快速度
    optimize_idm_params_scipy()