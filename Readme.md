## 文档结构概览

本项目文档主要分为以下几个部分：

1. **项目简介**：介绍项目的目标和应用场景。
2. **文件目录与功能说明**：详细列举各个脚本文件及其主要函数的功能。
3. **数据说明**：包括数据来源、格式、结构及其在各脚本中的应用方式。
4. **地图与场景细节**：描述仿真地图的具体信息和交通流特征。
5. **模型与算法说明**：对所用的物理模型、统计模型和深度学习模型进行归纳总结。
6. **常见问题与扩展讨论**：针对实际应用中遇到的问题，提出解决思路和方法。

---

## 项目简介

`extractCitySimData` 项目旨在从 CitySim 仿真环境中提取交通流数据，生成用于交通流建模与智能交通系统训练的数据集。数据既可用于 SUMO 仿真，也可用于排队过路口模型和世界模型的训练与评估。

---

## 文件目录与功能说明

### 主要脚本及其功能

- **test0_static_features.py**
    - 提取地图、道路、车辆等静态特征，生成车道与轨迹点的对应关系，并可视化轨迹分布和外围盒。
    - 主要函数：
        - `extract_traj_data`：提取车辆轨迹，生成轨迹图像，分析车辆分布与轨迹范围。
        - `extract_lane_traj_data`：针对每个车道，绘制车辆中心点分布。
        - `extract_one_lane_redlight_data`：分析特定车道红灯期间车辆停留时间。
        - `create_gif_of_car_positions`：生成车辆位置变化的动态图（GIF）。

- **test1.py**
    - 实现基础的数据提取，聚焦于单条道路的数据处理。
    - 主要函数：
        - `extract_laneid_data`：按车道ID提取CSV数据。
        - `gen_samples`：生成红灯状态下的车辆队列数据，计算车辆特征，统计绿灯期间车辆通过停止线的时间。

- **test2.py**
    - 整合多源数据，转换为适合 Keras 训练的数据格式，并保存为 CSV/PKL。
    - 主要函数：
        - `prepare_data`：读取并筛选每帧数据，形成训练样本。
        - `train_and_evaluate_model`：基于 ResNet 结构进行模型训练与评估。

- **modelCollect.py**
    - 集成多种神经网络模型，支持不同训练与评估方式。
    - 主要函数：
        - `model_simpleResnet0`：基础 ResNet 训练与评估。
        - `model_with_MCDDropout`：基于 MCDropout 的不确定性建模。
        - `model_with_ensemble`：集成学习方法，多模型采样训练与评估。

- **modelCollect2.py**
    - 针对车辆跟随模型参数优化，采用全局搜索算法。
    - 主要函数：
        - `simulated_annealing`：模拟退火算法，已调试通过。
        - `genetic_algorithm`：遗传算法，待完善。
        - `random_search`：随机搜索，待完善。
    - 注：采用 pyGameInterface2，所有车辆参数一致。

- **modelCollect4.py**
    - 针对每辆车独立参数的 IDM 跟车模型，探索可微分物理模型的局限性。
    - 注：采用 pyGameInterface4beta，每辆车参数独立。
    - 因为仿真物理模型无法可微，导致模型训练过程出现大问题，所以这个模型到此为止

- **modelCollect3.py** 已经跑通
    > - 针对每辆车独立参数的 IDM 跟车模型，探索可微分物理模型的局限性。
    > - 所有函数都针对jax优化,输入输出都是,jnp.ndarray
    > - 采用 pyGameBraxInterface4gamma，全部是函数，不用类（还用数据类）
    > - 结果不好，因为迅训练过程中物理模型的LOSS变化不大。
    > - nohup python modelsCollect3.py --batch_size 1280  --test_size 0.8 --epochs 100 --lr 0.0005 --unit 256 --layNum 8 &
    > - 主要函数：
        - run_single_simulation1  为并行化设计的独立仿真函数,用于以pool多进程的run_batch_simulation1调用
        - run_batch_simulation1  pool多进程的进行仿真（不用jax,只用CPU）
        - run_batch_simulation2  JAX纯函数版本，适用于vmap批量仿真。输入为batch样本，输出主车time_to_vanish。
            - 调用 initial_env_state_pure和rollout_while等纯函数，进行jax batch函数仿真
            - 

- **pyGameBraxInterface4gamma**
    - jax可微的仿真模型，全部用initial_env_state_pure，rollout_while 纯函数和jax优化
    - 核心函数为 initial_env_state_pure，rollout_while，ompute_idm_acc，compute_stopping_acc，step_pure
    - 其他函数为中间过程和测试

- **pyGameBraxInterface4beta**
    - 不使用jax的特性，jnp就是np,方便调试和多进程仿真。


## 数据说明

- **数据来源**：  
    - [UCF-SST-CitySim1-Dataset](https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset)
    - [数据格式说明](https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset/wiki)

- **数据结构**：
    - 车道起止坐标、车辆队列数据、车辆特征、绿灯通过数据等，均以 CSV 格式存储，便于后续处理与分析。

---

## 地图与场景细节

- **IntersectionA-01.csv**：
    - 车道 0,1,2 无红绿灯，车辆速度较高。
    - 车道 3-7 含红绿灯，部分为左转道，红绿灯时长各异。

---

## 模型与算法归纳总结

- **物理模型**：如匀加速模型、IDM 跟车模型，适用于已知动力学规律的场景。
- **统计模型**：对粒子（车辆）最终分布概率建模，适合参数未知但分布已知的情况。
- **深度学习模型**：如 ResNet、Transformer，可学习复杂的时空关系和非线性动力学。

---

## 函数功能归纳

- 数据提取与预处理：实现从原始仿真数据到结构化训练样本的自动化转换。
- 特征工程与可视化：支持静态与动态特征的提取、分析与可视化。
- 模型训练与评估：集成多种神经网络结构，支持不同训练策略与评估方法。
- 参数优化与搜索：提供多种全局优化算法，提升物理模型的拟合能力。

---

## 常见问题与扩展讨论

### 问题：如何结合 Transformer 与物理/统计模型进行粒子状态预测？

#### 1. 已知匀加速或速度概率模型
- 可将物理模型的预测结果作为 Transformer 的输入特征或辅助标签，提升模型泛化能力。
- 代码实现示例（伪代码）：
        ```python
        # 物理模型预测
        physics_pred = simple_physics_model(state, params)
        # Transformer 输入
        transformer_input = torch.cat([raw_features, physics_pred], dim=-1)
        output = transformer(transformer_input)
        ```

#### 2. 已知最终分布概率但参数未知
- 可用统计模型生成伪标签或作为损失函数的正则项，引导 Transformer 学习分布特征。
- 代码实现示例（伪代码）：
        ```python
        # 统计分布约束
        loss = mse_loss(predicted, target) + kl_divergence(predicted_distribution, empirical_distribution)
        ```

#### 3. 综合方法
- 采用多任务学习，将物理模型、统计模型与 Transformer 联合训练，互为补充。
- 结构示意：
        - 输入：原始特征 + 物理模型输出
        - 损失：预测误差 + 分布约束
        - 输出：粒子状态序列或最终分布

---

## 总结

本项目通过多脚本协作，实现了从仿真数据提取、特征工程、模型训练到参数优化的完整流程。各模块功能清晰，便于扩展和维护。针对不同建模需求，支持物理、统计与深度学习多种方法，并可灵活组合以提升预测精度和泛化能力。

## extractCitySimData 
    - 用于提取CitySim里面的数据，形成仿真和训练数据集
    - 仿真为Sumo,数据集为用于排队过路口模型+世界模型
  
## 文件目录
    - test2.py 将test1生成的各种数据文件，整合成输入到keras进行训练的格式，并保存为csv,pkl
    - test1.py 第一个数据提取代码，简单只提取一条道路
    - test0_static_features.py,获得地图、道路、轨迹、车辆一些静态特征数据
      - 已经生成所有车道一一对应的轨迹点
      - 已经生成所有轨迹对应的外围盒。（整个地图是4K图像，3840x2160）
#### test0_static_features.py 中所有函数的简要功能解释：
- **extract_traj_data**:提取车辆轨迹数据，生成轨迹图像并绘制凸包和边界框，用于分析车辆分布和轨迹范围。
- **extract_lane_traj_data**:提取指定车道的车辆轨迹数据，并为每个车道绘制车辆中心点的分布图。
- **extract_one_lane_redlight_data**:分析特定车道的车辆数据，提取满足速度和距离条件的车辆停留时间（红灯时间）。
- **create_gif_of_car_positions**:根据车辆轨迹数据生成多帧图像，并将其合成为GIF，用于动态展示车辆在车道上的位置变化。
- 
#### tes1.py 中所有函数的简要功能解释：
- **extract_laneid_data**:读取CSV文件并提取,根据ID特定道路的数据。
- **gen_samples**:  
        - 1.获得红灯状态而且停车线有车辆时的每一帧数据，以及对应红灯剩余时间
        - 2.根据每一帧数据和对应的红灯剩余时间，获得当前车道每一辆车的位置和速度，并且按照到停止线的距离进行排序
        - 3.根据frameNum和laneID，生成一个记录，获得道路上的所有车辆根据carID,搜索和计算一下特征
        - 4.计算绿灯持续时间，并在绿灯期间计算每辆车经过停止线的时间及相关信息，并排序GIT
        - 
#### tes2.py 中所有函数的简要功能解释：
- **prepare_data**:IntersectionA-01-trainsamples读入每个车道每帧的类似的数据，并筛选特征和形成训练数据
- **train_and_evaluate_model**:根据形成的训练数据，采用最简单的resnet进行训练和评估

#### modelCollect.py 中所有函数的简要功能解释：
- **model_simpleResnet0** 与test2的train_and_evaluate_model一样，采用最简单的resnet进行训练和评估被test2调用进行训练的评估。
- **model_with_MCDDropout** 与test2的train_and_evaluate_model一样，采用最简单的MCDrop进行随机采样的进行训练和评估被test2调用进行训练的评估。
- **#model_with_ensemble** 与test2的train_and_evaluate_model一样，采用集成模型的进行多模型采样的进行训练和评估被test2调用进行训练的评估。
- 
#### modelCollect2.py 中所有函数的简要功能解释（采用全局搜索模型，对车辆跟随模型进行参数优化）：
- **simulated_annealing** 模拟退火（现阶段调试成功）
- **genetic_algorithm** 基因算法（没有调试成功）
- **random_search** 随机搜索（没有调试成功）
- **注意**，以上算法采用pyGameInterface2，也就是所有车辆采用相同参数的一个idm跟车模型
  
#### modelCollect4.py 
-- 因为仿真物理模型无法可微，导致模型训练过程出现大问题，所以这个模型到此为止
-- 采用采用pyGameInterface3，每一个车辆都有一个专有参数的idm跟车模型


## 基本信息
    - 数据在 https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset
    - 数据CSV格式在 https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset/wiki
## 地图细节
#### 1. IntersectionA-01.csv
    - IntersectionA-01.csv 的车道0,1,2 没有红绿灯，所有车辆速度大于15
    - lane 3,4,5,6,7为从左向右的道路，有红绿灯,lane 3,4 和 5,6，7 是左转向道，红绿灯时间不一样

## 数据文件的数据结构
#### test1的数据结果和输出文件的数据结构
- 车道的开始和结束坐标
    ~~~
     start_of_lane_coordsNow = {
        5: (101, 1302),
        6: (100, 1410),
        7: (95, 1498)
    }
    end_of_lane_coordsNow = {
        5: (1261, 1302),
        6: (1263, 1372),
        7: (1267, 1469)
    }
    ~~~
  
-   lane_5_frame_0_queue_data.csv的格式
    ~~~
    record = {
            'frameNum': frame,
            'redLightRemainingTime': red_light_remaining_time[frame],
            'carIds': car_ids,
            'carPositions': car_positions,
            'carSpeeds': car_speeds
            }
    ~~~
- lane_5_car_features.csv的格式
    ~~~
    'carId': car_id,
    'maxSpeed': max_speed,
    'minSpeed': min_speed,
    'avgSpeed': avg_speed,
    'maxAcceleration': max_acceleration,
    'minAcceleration': min_acceleration,
    'crossingFrame1': crossing_frame1 if not np.isnan(crossing_frame1) else None,
    'crossingFrame2_vanish': crossing_frame2_vanish     
    ~~~ 
- lane_5_green_light_data.csv的格式
    ~~~
    'carId': car_id,
    'greenStartFrame': green_start,
    'crossingFrame': crossing_frame if not np.isnan(crossing_frame) else None,
    'initialPositionX': initial_position[0],
    'initialPositionY': initial_position[1],
    'initialSpeed': initial_speed
    ~~~

    
##***提出问题***
1.基于tranformer对物理模型进行预测，类似预测多个粒子的运动状态，给出相应的原理和代码。如果出现以下情况，1.如果已经知道一个简单的匀加速或者速度概率模型对粒子状态进行预测。2.基于统计模型，对粒子最终（而不是中间过程）分布概率已经建模，但是不知道具体参数。如何综合tranformer和1，2两种情况进行预测