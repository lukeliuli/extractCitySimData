import pandas as pd
import numpy as np
import os
import random
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pickle
import csv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import numpy as np

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
'''
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


def prepare_data(lane_data_path,start_of_lane_coordsNow,end_of_lane_coordsNow):
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
                car_features_file = f"IntersectionA-01-trainsamples/lane_{lane}_car_features.csv"
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
                
                car_features_file = f"IntersectionA-01-trainsamples/lane_{lane}_car_features.csv"
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

    with open('trainsamples_lane_5_6_7.pkl', 'wb') as f:
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
    df.to_csv('trainsamples_lane_5_6_7.csv', index=False, encoding='utf-8')

#######################################################################################################################################

#读入上面生成的数据，提取除了'carID','lane','frameNum'以外数据，然后目标为time_to_vanish减去当前样本的frameNum,以及'min_speed'，生成训练数据生成器，
#并基于keras进行简单的toyh回归训练,模型包括10层resnet，全连接mlp等等模型,
#进一步生成一个函数，生成的训练数据进行5,10,15帧的滚动预测，进行误差分析。
#进一步生成一个函数，基于训练好的数据，进行端对端的车辆通过路口时间和最小速度进行均值和方差预测。

#进一步生成一个函数，基于训练好的模型，进行端对端的车辆通过路口时间预测。
#进一步生成一个函数，基于训练好的模型，进行端对端的车辆通过路口时间和过程状态预测。
#进一步生成一个函数，基于训练好的模型，进行端对端的车辆通过路口时间和过程状态预测，并进行不确定性分析（GMM）。
#进一步生成一个函数，基于训练好的模型，进行端对端的车辆通过路口时间和过程状态预测，并进行TRACE分析。
#进一步生成一个函数，基于训练好的模型，进行端对端的车辆通过路口时间和过程状态预测，并进行NeRF分析。
#进一步生成一个函数，基于训练好的模型，进行端对端的车辆通过路口时间和过程状态预测，并进行物理约束神经网络分析（MLPODE）。
#进一步生成一个函数，基于训练好的模型，进行端对端的车辆通过路口时间和最小速度进行均值和方差预测。
#进一步生成一个函数，基于训练好的数据，如何结合微观交通模型进行联合预测。
def train_and_evaluate_model(unit=256, layNum=10, batch_size=64, epochs=50):

    df = pd.read_csv('trainsamples_lane_5_6_7.csv')

    # Prepare the data
    features = df.drop(columns=['carID', 'lane', 'frameNum', 'time_to_vanish', 'min_speed']).values
    targets = df[['time_to_vanish']].copy()
    targets['time_to_vanish'] = targets['time_to_vanish']  # Adjust time_to_vanish by subtracting frameNum，数据准备阶段已经减去当前frameNum
    targets = targets.values/30  # Normalize targets to minutes assuming 30 FPS

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.5, random_state=42)

    # Define a data generator
    class DataGenerator(Sequence):
        def __init__(self, X, y, batch_size=32):
            self.X = X
            self.y = y
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))

        def __getitem__(self, idx):
            batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_X, batch_y

    train_gen = DataGenerator(X_train, y_train, batch_size=batch_size)
    val_gen = DataGenerator(X_val, y_val, batch_size=batch_size)

    # Define a ResNet block
    def resnet_block(x, units):
        shortcut = x
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = ReLU()(x)
        return x

    # Build the model
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(unit)(input_layer)
    x = ReLU()(x)
    for _ in range(layNum):  # 10 ResNet blocks
        x = resnet_block(x, units=unit)
    output_layer = Dense(1)(x)  # Predict adjusted time_to_vanish

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Train the model
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save the model
    model.save('resnet_model_no_min_speed.h5')
    np.set_printoptions(suppress=True, precision=4)
    # Evaluate the model on the validation set
    predictions = model.predict(X_val)
    differences = predictions - y_val

    # Calculate mean and variance of the differences
    mean_difference = np.mean(differences, axis=0)
    variance_difference = np.var(differences, axis=0)

    print("Mean Difference (Prediction - True):", mean_difference)
    print("Variance of Differences:", variance_difference)
    # Plot predictions vs true values with different markers
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(predictions)), predictions, label='Predictions', alpha=0.7, marker='o', color='blue')
    plt.scatter(range(len(y_val)), y_val, label='True Values', alpha=0.7, marker='x', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Predictions and True Values')
    plt.legend()
    plt.savefig('predictions_and_true_values.png')
    plt.close()

if __name__ == "__main__":

    start_of_lane_coordsNow = {
        5: (4.24885685,55.9082253),
        6: (4.20678896,59.31572436),
        7: (3.99644951,63.01769865)
    }
    end_of_lane_coordsNow = {
        5: (53.04760881,54.77239228),
        6: (53.13174459,57.71714455),
        7: (53.30001614,61.79772985)
    }

    lane_data_path = "IntersectionA-01-trainsamples"
    prepare_data(lane_data_path,start_of_lane_coordsNow,end_of_lane_coordsNow)
    train_and_evaluate_model(unit=32,layNum=10,batch_size=640*20,epochs=500)#就trainsamples_lane_5_6_7.csv结论太好
                                                                            #不需要太好模型
    #train_and_evaluate_model(unit=256,layNum=30,batch_size=640*20,epochs=500)
    