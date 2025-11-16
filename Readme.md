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

#### modelCollect 中所有函数的简要功能解释：
- **model_simpleResnet0** 与test2的train_and_evaluate_model一样，采用最简单的resnet进行训练和评估被test2调用进行训练的评估。
- **model_with_MCDDropout** 与test2的train_and_evaluate_model一样，采用最简单的MCDrop进行随机采样的进行训练和评估被test2调用进行训练的评估。
- **#model_with_ensemble** 与test2的train_and_evaluate_model一样，采用集成模型的进行多模型采样的进行训练和评估被test2调用进行训练的评估。

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