

## 项目概述

`extractCitySimData` 是一个用于从 CitySim 仿真环境中提取交通流数据并构建智能交通系统训练数据集的综合性项目。该项目旨在为交通流建模、排队过路口模型和世界模型提供高质量的数据支持，同时探索深度学习在车辆丢失检测与消失时间预测中的应用。

## 核心功能模块

### 数据提取与预处理

#### test0_static_features.py
- **功能**：提取地图、道路、车辆等静态特征，生成车道与轨迹点的对应关系，并可视化轨迹分布。
- **主要函数**：
  - [extract_traj_data](file://e:\extractCitySimData\test0_static_features.py#L16-L94)：提取车辆轨迹数据，生成轨迹图像并绘制凸包和边界框。
  - [extract_lane_traj_data](file://e:\extractCitySimData\test0_static_features.py#L101-L159)：针对每个车道，绘制车辆中心点分布。
  - [extract_one_lane_redlight_data](file://e:\extractCitySimData\test0_static_features.py#L318-L470)：分析特定车道红灯期间车辆停留时间。
  - [create_gif_of_car_positions](file://e:\extractCitySimData\test0_static_features.py#L476-L556)：生成车辆位置变化的动态图（GIF）。

#### test1.py
- **功能**：实现基础的数据提取，聚焦于单条道路的数据处理。
- **主要函数**：
  - [extract_laneid_data](file://e:\extractCitySimData\test1.py#L10-L27)：按车道ID提取CSV数据。
  - [gen_samples](file://e:\extractCitySimData\test1.py#L175-L360)：生成红灯状态下的车辆队列数据，计算车辆特征，统计绿灯期间车辆通过停止线的时间。

#### test2.py
- **功能**：整合多源数据，转换为适合 Keras 训练的数据格式，并保存为 CSV/PKL。
- **主要函数**：
  - [prepare_data](file://e:\extractCitySimData\test2.py#L46-L141)：读取并筛选每帧数据，形成训练样本。
  - [train_and_evaluate_model](file://e:\extractCitySimData\test2.py#L158-L237)：基于 ResNet 结构进行模型训练与评估。

### 模型训练与评估

#### modelCollect.py
- **功能**：集成多种神经网络模型，支持不同训练与评估方式。
- **主要函数**：
  - [model_simpleResnet0](file://e:\extractCitySimData\modelsCollect.py#L257-L336)：基础 ResNet 训练与评估。
  - [model_with_MCDDropout](file://e:\extractCitySimData\modelsCollect.py#L35-L151)：基于 MCDropout 的不确定性建模。
  - [model_with_ensemble](file://e:\extractCitySimData\modelsCollect.py#L161-L252)：集成学习方法，多模型采样训练与评估。

#### modelCollect2.py
- **功能**：针对车辆跟随模型参数优化，采用全局搜索算法。
- **主要函数**：
  - [simulated_annealing](file://e:\extractCitySimData\modelsCollect2.py#L104-L133)：模拟退火算法。
  - [genetic_algorithm](file://e:\extractCitySimData\modelsCollect2.py#L135-L173)：遗传算法（待完善）。
  - [random_search](file://e:\extractCitySimData\modelsCollect2.py#L175-L188)：随机搜索（待完善）。

#### modelCollect3.py
- **功能**：针对每辆车独立参数的 IDM 跟车模型，探索可微分物理模型的局限性。
- **主要函数**：
  - [run_single_simulation1](file://e:\extractCitySimData\modelsCollect3.py#L155-L243)：为并行化设计的独立仿真函数。
  - [run_batch_simulation1](file://e:\extractCitySimData\modelsCollect3.py#L341-L372)：pool多进程的进行仿真。
  - [run_batch_simulation2](file://e:\extractCitySimData\modelsCollect3.py#L246-L338)：JAX纯函数版本，适用于vmap批量仿真。

#### pyGameBraxInterface4gamma.py
- **功能**：jax可微的仿真模型，全部用initial_env_state_pure，rollout_while 纯函数和jax优化。
- **核心函数**：
  - [initial_env_state_pure](file://e:\extractCitySimData\pyGameBraxInterface4gamma.py#L312-L344)：初始化环境状态。
  - [rollout_while](file://e:\extractCitySimData\pyGameBraxInterface4gamma.py#L257-L279)：使用 jax.lax.while_loop 实现动态终止的 rollout。
  - [compute_idm_acc](file://e:\extractCitySimData\pyGameBraxInterface4gamma.py#L63-L90)：计算IDM加速度。
  - [compute_stopping_acc](file://e:\extractCitySimData\pyGameBraxInterface4gamma.py#L95-L111)：计算停车加速度。
  - [step_pure](file://e:\extractCitySimData\pyGameBraxInterface4gamma.py#L114-L254)：单步仿真纯函数。

#### modelLostReg.py
- **功能**：实现车辆漏检的数据集生成，以及基于模型进行识别。
- **主要函数**：
  - [genSamplesByRandomRemovingVehcile](file://e:\extractCitySimData\modelsLostReg.py#L164-L265)：随机删除车辆，生成df数据集。
  - [genDatasetVanishTime](file://e:\extractCitySimData\modelsLostReg.py#L269-L310)：生成用于训练vanishTime回归模型训练和验证数据集。
  - [genDatasetLost](file://e:\extractCitySimData\modelsLostReg.py#L314-L345)：生成用于训练lostFlag二分法识别模型训练和验证数据集。
  - [genDatasetLostFlagVanishTime](file://e:\extractCitySimData\modelsLostReg.py#L351-L387)：生成用于识别和回归双分支模型训练和验证数据集。
  - [build_multi_task_resnet](file://e:\extractCitySimData\modelsLostReg.py#L391-L431)：基于keras生成识别和回归双分支模型。
  - [build_simple_resnet_regress](file://e:\extractCitySimData\modelsLostReg.py#L125-L162)：基于keras生成回归模型。
  - [build_simple_resnet](file://e:\extractCitySimData\modelsLostReg.py#L80-L123)：基于keras生成识别模型。
  - [main](file://e:\extractCitySimData\modelsLostReg.py#L436-L789)：主训练函数， orchestrates the entire process.

### 数据说明

- **数据来源**：UCF-SST-CitySim1-Dataset
- **数据结构**：
  - 车道起止坐标、车辆队列数据、车辆特征、绿灯通过数据等，均以 CSV 格式存储。
  - 示例数据文件包括 `lane_5_frame_0_queue_data.csv`、[lane_5_car_features.csv](file://e:\extractCitySimData\IntersectionA-01-trainsamples\lane_5_car_features.csv)、[lane_5_green_light_data.csv](file://e:\extractCitySimData\IntersectionA-01-trainsamples\lane_5_green_light_data.csv)。

### 地图与场景细节

- **IntersectionA-01.csv**：
  - 车道0,1,2 没有红绿灯，所有车辆速度大于15。
  - lane 3,4,5,6,7为从左向右的道路，有红绿灯，lane 3,4 和 5,6，7 是左转向道，红绿灯时间不一样。

### 模型与算法说明

#### 物理模型
- **IDM模型**：用于描述车辆跟车行为，参数包括期望速度、安全车头时距、静止安全距离、最大加速度、舒适减速度、加速度指数、车长和反应时间。

#### 统计模型
- **ResNet模型**：用于车辆丢失检测和消失时间预测，包含单任务分类、单任务回归和多任务学习三种模式。

#### 深度学习模型
- **MCDropout**：基于蒙特卡洛Dropout的不确定性建模。
- **集成学习**：通过多个模型的集成提高预测性能。

### 常见问题与扩展讨论

- **问题**：如何处理车辆丢失情况？
  - **解决方案**：通过随机删除车辆位置和速度数据，模拟传感器丢失数据的情况，增强模型鲁棒性。

- **问题**：如何提高模型训练效率？
  - **解决方案**：使用JAX的自动微分和JIT编译特性，实现高效的交通流仿真。

- **问题**：如何评估模型性能？
  - **解决方案**：使用准确率、精确率、召回率、F1分数、均方误差（MSE）、均方根误差（RMSE）等指标进行综合评估。

### 使用说明

- **运行命令**：
  ```bash
  python modelsLostReg.py --batch_size 1280 --test_size 0.5 --epochs 500 --lr 0.005 --unit 256 --layNum 8