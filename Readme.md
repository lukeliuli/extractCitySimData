## extractCitySimData 
    - 用于提取CitySim里面的数据，形成仿真和训练数据集
    - 仿真为Sumo,数据集为用于排队过路口模型+世界模型
  
## 文件目录
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
  
## 基本信息
    - 数据在 https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset
    - 数据CSV格式在 https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset/wiki
## 地图细节
#### 1. IntersectionA-01.csv
    - IntersectionA-01.csv 的车道0,1,2 没有红绿灯，所有车辆速度大于15
    - lane 3,4,5,6,7为从左向右的道路，有红绿灯,lane 3,4 和 5,6，7 是左转向道，红绿灯时间不一样
