import pandas as pd
import numpy as np
import os
import random
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pickle
import csv

##############
'''
1.原理简单介绍
    - 获得每一帧（每个记录）当前车道场景数据，预测车辆通过路口的时间
    - 获得每一帧（每个记录）当前车道场景数据，预测下一辆车通过路口的时间，并进行自回归分析预测下下辆车通过路口的时间
    - 获得每一帧（每个记录）当前车道场景数据，基于SUMO模型预测车辆通过路口的时间和过程状态 ？？？
    - 获得每一帧（每个记录）当前车道场景数据，端对端预测车辆通过路口的时间和过程状态（GMM） ？？？
    - 获得每一帧（每个记录）当前车道场景数据，TRACE框架端预测车辆通过路口的时间和过程状态（GMM） ？？？
    - NeRF,物理约束神经网络？？，MLPODE
        

'''
##########################################################################################################################
'''
+ 输入处理
1.已经知道车道5,6,7的开始和结束位置,从../myData/lane_5_frame_0_queue_data.csv等文件中提取每一帧的车道数据
##
'''
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
'''
# 从E:\myData\IntersectionA-01-trainsamples读入每个5,6,7车道每帧的类似lane_5_frame_0_queue_data.csv的数据，注数据格式如下
    record = {
            'frameNum': frame,
            'redLightRemainingTime': red_light_remaining_time[frame],
            'carIds': car_ids,
            'carPositions': car_positions,
            'carSpeeds': car_speeds
                }
    其中carIds,carPositions,carSpeeds均为列表,需要全部读取
'''

lane_data_path = "E:/myData/IntersectionA-01-trainsamples"
trainsamples = {}
lane_data = {}
for lane in start_of_lane_coordsNow.keys():
    lane_data[lane] = []
    lane_files = [f for f in os.listdir(lane_data_path) if f.startswith(f"lane_{lane}_frame_") and f.endswith("_queue_data.csv")]
    for file in lane_files:
        file_path = os.path.join(lane_data_path, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            frameNum = int(df['frameNum'].values[0])
            car_ids = list(map(int, df['carIds'].values[0].strip("[]").split(',')))
            car_speeds = list(map(float, df['carSpeeds'].values[0].strip("[]").split(',')))
            redLightRemainingTime =int(df['redLightRemainingTime'].values[0])
          
    
            # car_positions 为字符串，内容为[[1135, 1470], [878, 1475], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]，转换为整数的序列
            car_positions = eval(df['carPositions'].values[0])  # Convert string to list of lists
            car_positions = np.array(car_positions)
            car_positions2 = np.where(car_positions[:, 0] == -1, -1, end_of_lane_coordsNow[lane][0] - car_positions[:, 0])
            
            #根据lane,从类似lane_5_car_features.csv文件读取carid对应的车辆特征，并转换为np.array格式,存在对应的 dict中
            car_features_file = f"E:/myData/IntersectionA-01-trainsamples/lane_{lane}_car_features.csv"
            car_features_df = pd.read_csv(car_features_file)
            crossing_frame2_vanish = {}
            min_speed = {}
            for car_id in car_ids:
                feature_row = car_features_df[car_features_df['carId'] == car_id]
                if not feature_row.empty:
                    crossing_frame2_vanish[car_id] = feature_row['crossingFrame2_vanish'].values[0]  # Extract crossingFrame2_vanish
                    min_speed[car_id] = feature_row['minSpeed'].values[0] 
                else:
                    crossing_frame2_vanish[car_id] = -1  # Default to None if not found
                    min_speed[car_id] = -1
            
            car_features_file = f"E:/myData/IntersectionA-01-trainsamples/lane_{lane}_car_features.csv"
            car_features_df = pd.read_csv(car_features_file)
            laneMaxSpeed = np.percentile(car_features_df['maxSpeed'], 95)  # Use the 95th percentile of maxSpeed as lane max speed
            laneAvgSpeed = np.percentile(car_features_df['avgSpeed'], 50)  # Use the 60th percentile of maxSpeed as lane max speed
            maxAcceleration = np.percentile(car_features_df['maxAcceleration'], 95)  # Use the 95th percentile of maxAcceleration as lane max acceleration
            #打印检查读取的数据                    
            
            
            
            print(" file_path:",file_path)
            #print(" car_ids:", car_ids)
            #print(" car_speeds:", car_speeds)
            #print(" frameNum:", frameNum)
            #print(" redLightRemainingTime:", redLightRemainingTime)
            ##print(" car_positions:", car_positions)
            #print(" car_positions2:", car_positions2)
            
            #从car_ids,car_positions2,car_speeds提取每个car_id对应的位置和速度，结合car_positions2，car_speeds
            
            for idx, car_id in enumerate(car_ids):
                if car_id != -1:  # Ensure valid car_id
                    tmp0 = np.array([car_id, lane,frameNum])  # 3
                    tmp1 = np.array([car_positions2[idx], car_speeds[idx],redLightRemainingTime])#3
                    tmp2 = car_positions2#20
                    tmp3 = np.array(car_speeds)#20
                    tmp4 =  [laneMaxSpeed, laneAvgSpeed, maxAcceleration]#3
                    tmpY1 =  np.array([crossing_frame2_vanish[car_id]-frameNum,min_speed[car_id]])#2
                    record = {
                        'name': ['carID','lane','frameNum'] +\
                                        ['car_position', 'car_speed','redLightRemainingTime'] + \
                                        [f'car_position_{i}' for i in range(len(car_positions2))] + \
                                        [f'car_speed_{i}' for i in range(len(car_speeds))] + \
                                        ['laneMaxSpeed', 'laneAvgSpeed', 'maxAcceleration']+\
                                        ['time_to_vanish','min_speed'],
                        'vector': np.concatenate([tmp0,tmp1, tmp2, tmp3,tmp4,tmpY1]),                    
                    }
                    trainsamples[f"{lane}_{car_id}_{frameNum}"] = record
                       
#trainsamples字典中存储了所有车道的每一帧的车辆数据，键为"{lane}_{car_id}_{frameNum}"，值为对应的记录字典
print("总共提取的训练样本数量:", len(trainsamples))#
#保存trainsamples到文件

with open('E:/myCodes/extractCitySIMData/extractCitySimData/trainsamples_lane_5_6_7.pkl', 'wb') as f:
    pickle.dump(trainsamples, f) 

# 转换保存dict类的trainsamples的vector到pandas,并保存为CSV文件
# 其中表头为trainsamples的name

# Extract the header from the first record in trainsamples
header = list(trainsamples.values())[0]['name']
print(header)
# Prepare data for DataFrame
data = []
for record in trainsamples.values():
    data.append(record['vector'])

# Create DataFrame
df = pd.DataFrame(data, columns=header)

# Save DataFrame to CSV
df.to_csv('E:/myCodes/extractCitySIMData/extractCitySimData/trainsamples_lane_5_6_7.csv', index=False, encoding='utf-8')
