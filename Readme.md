## extractCitySimData 
    - 用于提取CitySim里面的数据，形成仿真和训练数据集
    - 仿真为Sumo,数据集为用于排队过路口模型+世界模型
  
## 文件目录
    - test1.py 第一个数据提取代码，简单只提取一条道路
    - test0_static_features.py,获得地图、道路、轨迹、车辆一些静态特征数据
      - 已经生成所有车道一一对应的轨迹点
      - 已经生成所有轨迹对应的外围盒。（整个地图是4K图像，3840x2160）
  
## 基本信息
    - 数据在 https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset
    - 数据CSV格式在 https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset/wiki
