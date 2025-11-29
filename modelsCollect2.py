
from pyGameInterface2 import TrafficSimulator, VehicleParams
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import math

'''
采用全局搜索模型（如模拟退火），对车辆跟随模型进行参数优化。不使用神经网络。
'''
# =============================================================================
# IDM 参数优化模块
# =============================================================================

# 1. 定义目标函数 (Cost Function)
def calculate_simulation_error(params_dict, data_samples, sim_num_per_sample=1):
    """
    计算给定IDM参数下，模拟结果与真实数据之间的平均误差。
    这是所有优化算法要最小化的目标。

    Args:
        params_dict (dict): 包含IDM参数的字典 (v0, T, s0, a, b)。
        data_samples (pd.DataFrame): 用于评估的样本数据集。
        sim_num_per_sample (int): 为减少随机性，每个样本运行模拟的次数。

    Returns:
        float: 均方根误差 (RMSE)，作为成本值。
    """
    errors = []
    for _, row in data_samples.iterrows():
        
        time_to_vanish_sim_list = []
        for _ in range(sim_num_per_sample):

            # 使用传入的参数创建参数对象
            params = VehicleParams(
                v0=params_dict['v0'],
                T=params_dict['T'],
                s0=params_dict['s0'],
                a=params_dict['a'],
                b=params_dict['b']
            )
            simulator = TrafficSimulator(params, time_step=0.1, intersection_pos=1000.0)

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
        
         

            # 从样本行设置模拟器初始状态
            redLightRemainingTime = row['redLightRemainingTime'] / 30
            simulator.set_red_light(redLightRemainingTime)
            
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


            # 运行模拟
            recordDF = simulator.run_simulation(max_duration=120)
            
            # 计算模拟的 time_to_vanish
            if not recordDF.empty and main_car_id != -1:
                main_car_data = recordDF[recordDF['id'] == main_car_id]
                passed_data = main_car_data[main_car_data['has_passed'] == True]
                if not passed_data.empty:
                    time_to_vanish_sim = passed_data.iloc[0]['time']
                    time_to_vanish_sim_list.append(time_to_vanish_sim)

        # 如果模拟成功，计算误差
        if time_to_vanish_sim_list:
            mean_sim_time = np.mean(time_to_vanish_sim_list)
            true_time = row['time_to_vanish']
            errors.append((mean_sim_time - true_time) ** 2)

    # 返回均方根误差
    if not errors:
        return float('inf') # 如果没有成功模拟，返回无穷大成本
    return np.sqrt(np.mean(errors))


# 2. 定义优化算法
def simulated_annealing(param_bounds, data_samples, max_iter=100, initial_temp=100, cooling_rate=0.95):
    """使用模拟退火算法寻找最优参数"""
    print("\n--- 开始模拟退火优化 ---")
    current_params = {key: random.uniform(low, high) for key, (low, high) in param_bounds.items()}
    current_cost = calculate_simulation_error(current_params, data_samples)
    best_params, best_cost = current_params, current_cost
    temp = initial_temp

    for i in tqdm(range(max_iter), desc="模拟退火"):
        # 生成邻近解
        neighbor_params = current_params.copy()
        param_to_change = random.choice(list(param_bounds.keys()))
        low, high = param_bounds[param_to_change]
        neighbor_params[param_to_change] = np.clip(
            random.gauss(current_params[param_to_change], (high - low) * 0.1), low, high
        )
        
        neighbor_cost = calculate_simulation_error(neighbor_params, data_samples)
        
        # 决定是否接受新解
        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
            current_params, current_cost = neighbor_params, neighbor_cost
            if current_cost < best_cost:
                best_params, best_cost = current_params, current_cost
        
        temp *= cooling_rate
        print(f"迭代 {i+1}: 当前成本={current_cost:.4f}, 最佳成本={best_cost:.4f}, 温度={temp:.2f}")

    return best_params, best_cost

def genetic_algorithm(param_bounds, data_samples, pop_size=20, generations=10, mutation_rate=0.1):
    """使用遗传算法寻找最优参数"""
    print("\n--- 开始遗传算法优化 ---")
    # 初始化种群
    population = [{key: random.uniform(low, high) for key, (low, high) in param_bounds.items()} for _ in range(pop_size)]
    
    for gen in tqdm(range(generations), desc="遗传算法"):
        # 计算适应度
        costs = [calculate_simulation_error(ind, data_samples) for ind in population]
        
        # 选择
        sorted_indices = np.argsort(costs)
        population = [population[i] for i in sorted_indices]
        costs = [costs[i] for i in sorted_indices]
        
        print(f"代 {gen+1}: 最佳成本={costs[0]:.4f}")

        # 保留最优的一部分
        elite_size = int(pop_size * 0.2)
        new_population = population[:elite_size]
        
        # 交叉和变异
        while len(new_population) < pop_size:
            p1, p2 = random.choices(population[:elite_size*2], k=2) # 从较优个体中选择父母
            child = {key: (p1[key] + p2[key]) / 2 for key in param_bounds}
            
            # 变异
            if random.random() < mutation_rate:
                param_to_mutate = random.choice(list(param_bounds.keys()))
                low, high = param_bounds[param_to_mutate]
                child[param_to_mutate] = np.clip(random.gauss(child[param_to_mutate], (high - low) * 0.1), low, high)
            
            new_population.append(child)
        
        population = new_population

    final_costs = [calculate_simulation_error(ind, data_samples) for ind in population]
    best_idx = np.argmin(final_costs)
    return population[best_idx], final_costs[best_idx]

def random_search(param_bounds, data_samples, max_iter=100):
    """使用随机搜索寻找最优参数"""
    print("\n--- 开始随机搜索优化 ---")
    best_params = None
    best_cost = float('inf')

    for _ in tqdm(range(max_iter), desc="随机搜索"):
        params = {key: random.uniform(low, high) for key, (low, high) in param_bounds.items()}
        cost = calculate_simulation_error(params, data_samples)
        if cost < best_cost:
            best_cost, best_params = cost, params
            print(f"新最佳成本: {best_cost:.4f}")
            
    return best_params, best_cost

# 3. 主执行函数
def find_optimal_idm_params():
    """
    运行多种优化算法来寻找IDM模型的最优参数，并评估结果。
    """
    # 定义参数搜索范围
    param_bounds = {
        'v0': (40 / 3.6, 60 / 3.6),
        'T': (0.5, 2.0),
        's0': (1.0, 3.0),
        'a': (1.0, 3.0),
        'b': (1.0, 6.0)
    }
    
    # 加载数据并选择一个小子集进行优化（否则太慢）
    df = pd.read_csv('trainsamples_lane_5_6_7.csv')
    # 为保证可复现性和样本多样性，随机选择但固定种子
    data_samples = df.sample(n=2000, random_state=42) 
    
    # 运行优化算法
    sa_params, sa_cost = simulated_annealing(param_bounds, data_samples, max_iter=50)
    ga_params, ga_cost = genetic_algorithm(param_bounds, data_samples, pop_size=10, generations=5)
    rs_params, rs_cost = random_search(param_bounds, data_samples, max_iter=50)

    results = {
        "模拟退火": {"params": sa_params, "cost": sa_cost},
        "遗传算法": {"params": ga_params, "cost": ga_cost},
        "随机搜索": {"params": rs_params, "cost": rs_cost}
    }

    print("\n" + "="*20 + " 优化结果总结 " + "="*20)
    for name, result in results.items():
        print(f"\n--- {name} ---")
        print(f"最佳成本 (RMSE): {result['cost']:.4f}")
        print("最佳参数:")
        for p, v in result['params'].items():
            print(f"  {p}: {v:.4f}")

    # 使用测试集评估最优参数的性能
    print("\n" + "="*20 + " 最优参数性能评估 " + "="*20)
    test_samples = df.drop(data_samples.index).sample(n=50, random_state=24) # 使用不同的样本作为测试集
    
    for name, result in results.items():
        errors = []
        sim_times = []
        true_times = []
        
        # 重新计算误差以获取详细统计信息
        for _, row in tqdm(test_samples.iterrows(), desc=f"评估 {name}"):
            params = VehicleParams(**result['params'])
            simulator = TrafficSimulator(params, time_step=0.1, intersection_pos=1000.0)
            
            redLightRemainingTime = row['redLightRemainingTime'] / 30
            simulator.set_red_light(redLightRemainingTime)
            
            main_car_id = -1
            vehicle_list = []
            for i in range(20):
                pos_col = f'car_position_{i}'
                speed_col = f'car_speed_{i}'
                if pos_col in row and row[pos_col] != -1:
                    other_car_id = 100 + i
                    vehicle_list.append({'id': other_car_id, 'distance': row[pos_col], 'speed': row[speed_col]})
                    if row[pos_col] == row['car_position']:
                        main_car_id = other_car_id
            simulator.add_vehicles(vehicle_list)

            recordDF = simulator.run_simulation(max_duration=120)
            
            if not recordDF.empty and main_car_id != -1:
                main_car_data = recordDF[recordDF['id'] == main_car_id]
                passed_data = main_car_data[main_car_data['has_passed'] == True]
                if not passed_data.empty:
                    sim_time = passed_data.iloc[0]['time']
                    true_time = row['time_to_vanish'] / 30
                    errors.append(sim_time - true_time)
                    sim_times.append(sim_time)
                    true_times.append(true_time)

        if errors:
            errors = np.array(errors)
            print(f"\n--- {name} 在测试集上的性能 ---")
            print(f"  误差均值 (Mean Error): {np.mean(errors):.4f}")
            print(f"  误差方差 (Variance of Error): {np.var(errors):.4f}")
            print(f"  平均绝对误差 (MAE): {np.mean(np.abs(errors)):.4f}")
            print(f"  均方根误差 (RMSE): {np.sqrt(np.mean(errors**2)):.4f}")
        else:
            print(f"\n--- {name} 在测试集上评估失败 ---")


if __name__ == "__main__":
    
    # 注意：为了防止日志过长，建议将优化过程的输出重定向到文件
    # with open('optimization_log.txt', 'w', encoding='utf-8') as f:
    #     sys.stdout = f
    #     find_optimal_idm_params()
    # sys.stdout = sys.__stdout__
    # print("优化完成，详情请见 optimization_log.txt")

    # 直接在控制台运行
    find_optimal_idm_params()
    
    # 以下是旧的函数调用，可以注释掉
    # sys.stdout = open('output.log', 'w', encoding='utf-8') 
    # model_simpleResnet0(unit=256,layNum=10,batch_size=640*20,epochs=500)
    # model_with_MCDDropout(unit=256, layNum=10, batch_size=640*20, epochs=500, test_size=0.9, mc_samples=10)
    # model_with_ensemble(unit=256, layNum=10, batch_size=640*20, epochs=500, test_size=0.9, ensemble_size=5) 
    # model_with_SimulCal_SimpleResnet(unit=256, layNum=10, batch_size=64, epochs=50, test_size=0.9,simNum=10)

    还没有运行测试过
