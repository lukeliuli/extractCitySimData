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
from tensorflow.keras.layers import Dropout

############################################
##收集和实现各种模型，用于测试和比较
############################################

##优化resetnet模型， 采用分数位回归，估计5%，25%，50%分位数，75%，95%分位数的数据
##1.数据准备阶段，输出数据需要准备，5%，25%，50%分位数，75%，95%分位数的数据
##2.对于当前样本，收集原始数据集中相同carID,lane,收集前后60个frameNum（注意是frameNum-1，frameNum-2，frameNum，frameNum+1的数据，不是样本的序号），
# 对于收集的60个数据样本，输出收集的time_to_vanish的5%，25%，50%分位数，75%，95%分位数的数据，作为训练和预测的目标值
####
# ~~~暂时放弃分位数，因为样本中输出是固定不变的，无法体现分位数的意义~~~


##优化resetnet模型,在训练和预测时都开启Dropout。预测时进行多次前向传播（ Monte Carlo Sampling），
# 用多次预测结果的均值和方差作为最终预测和不确定性估,并于样本集的预测结果的均值和方差进行对比
##优化resetnet模型,在训练和预测时都开启Dropout。预测时进行多次前向传播（ Monte Carlo Sampling），
# 用多次预测结果的均值和方差作为最终预测和不确定性估,并于样本集的预测结果的均值和方差进行对比

def model_with_MCDDropout(unit=256, layNum=10, batch_size=64, epochs=50, test_size=0.9, mc_samples=100):
    df = pd.read_csv('trainsamples_lane_5_6_7.csv')

    # Prepare the data
    features = df.drop(columns=['carID', 'lane', 'frameNum', 'time_to_vanish', 'min_speed']).values
    targets = df[['time_to_vanish']].copy()
    targets['time_to_vanish'] = targets['time_to_vanish']
    targets = targets.values / 30  # Normalize targets to minutes assuming 30 FPS

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=test_size, random_state=42)

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

    # Define a ResNet block with Dropout

    def resnet_block(x, units, dropout_rate=0.1):
        shortcut = x
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout layer
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout layer
        return x

    # Build the model with Dropout
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(unit)(input_layer)
    x = ReLU()(x)
    for _ in range(layNum):
        x = resnet_block(x, units=unit)
    output_layer = Dense(1)(x)  # Predict adjusted time_to_vanish

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Train the model
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save the model
    model.save('resnet_model_with_uncertainty.h5')
    np.set_printoptions(suppress=True, precision=4)

    # Monte Carlo Sampling for uncertainty estimation with forced dropout activation
    def monte_carlo_predictions(model, X, mc_samples):
        f_model = Model(inputs=model.input, outputs=model.output)
        # Force dropout layers to be active during prediction
        preds = np.array([
            f_model(X, training=True).numpy()  # Set training=True to activate dropout during inference
            for _ in range(mc_samples)
        ])
        return preds

    # Perform Monte Carlo Sampling
    mc_preds = monte_carlo_predictions(model, X_val, mc_samples)
    mean_preds = mc_preds.mean(axis=0)
    var_preds = mc_preds.var(axis=0)
    #print("mc_samples.shape:", mc_preds.shape)
    #print("X_val.shape:", X_val.shape)
    #print("mc_samples[:, 0, 0]:", mc_preds[:, 0, 0])
    #print("mc_samples[:, 1, 0]:", mc_preds[:, 1, 0])
    #print("Mean Predictions:", mean_preds.ravel())
    #print("Variance of Predictions:", var_preds.ravel())
        # Plot mean_preds and var_preds
    plt.figure(figsize=(10, 6))
    plt.plot(mean_preds, label='Mean Predictions', color='blue')
    plt.plot(var_preds, label='Variance of Predictions', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Mean Predictions and Variance of Predictions')
    plt.legend()
    plt.savefig('mean_and_variance_predictions.png')
    plt.close()


    # Calculate differences
    differences = mean_preds - y_val

    # Calculate mean and variance of the differences
    mean_difference = np.mean(differences, axis=0)
    variance_difference = np.var(differences, axis=0)

    print("Mean Difference (Prediction - True):", mean_difference)
    print("Variance of Differences:", variance_difference)
    print("Mean Prediction:", mean_preds)
    print("var_preds:", var_preds)
    
    # Plot predictions vs true values with uncertainty
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(mean_preds)), mean_preds.ravel(), yerr=np.sqrt(var_preds).ravel(), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.scatter(range(len(y_val)), y_val, label='True Values', alpha=0.7, marker='x', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Predictions with Uncertainty and True Values')
    plt.legend()
    plt.savefig('predictions_with_uncertainty.png')
    plt.close()
    




##修改和优化model_with_ensemble函数,采用深度集成法 用不同的随机初始化训练同一个模型的多个副本。对于同一个输入，每个模型会给出一个预测。将这些预测集合起来，计算均值和方差，
# 用多次预测结果的均值和方差作为最终预测和不确定性估,并于样本集的预测结果的均值和方差进行对比
# 

def model_with_ensemble(unit=256, layNum=10, batch_size=64, epochs=50, test_size=0.9, ensemble_size=5):
    df = pd.read_csv('trainsamples_lane_5_6_7.csv')

    # Prepare the data
    features = df.drop(columns=['carID', 'lane', 'frameNum', 'time_to_vanish', 'min_speed']).values
    targets = df[['time_to_vanish']].copy()
    targets['time_to_vanish'] = targets['time_to_vanish']
    targets = targets.values / 30  # Normalize targets to minutes assuming 30 FPS

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=test_size, random_state=42)

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
    def resnet_block(x, units, dropout_rate=0.1):
        shortcut = x
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout layer
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout layer
        return x
    # Function to build a single model
    def build_model():
        input_layer = Input(shape=(X_train.shape[1],))
        x = Dense(unit)(input_layer)
        x = ReLU()(x)
        for _ in range(layNum):
            x = resnet_block(x, units=unit)
        output_layer = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    # Train ensemble models
    ensemble_models = []
    for i in range(ensemble_size):
        print(f"Training model {i + 1}/{ensemble_size}")
        model = build_model()
        model.fit(train_gen, validation_data=val_gen, epochs=epochs)
        ensemble_models.append(model)

    # Save ensemble models
    for i, model in enumerate(ensemble_models):
        model.save(f'ensemble_model_{i}.h5')

    # Ensemble predictions
    predictions = np.array([model.predict(X_val) for model in ensemble_models])
    mean_preds = predictions.mean(axis=0)
    var_preds = predictions.var(axis=0)

    # Calculate differences
    differences = mean_preds - y_val

    # Calculate mean and variance of the differences
    mean_difference = np.mean(differences, axis=0)
    variance_difference = np.var(differences, axis=0)

    print("Mean Difference (Prediction - True):", mean_difference)
    print("Variance of Differences:", variance_difference)

    # Plot predictions vs true values with uncertainty
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(mean_preds)), mean_preds.ravel(), yerr=np.sqrt(var_preds).ravel(), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.scatter(range(len(y_val)), y_val, label='True Values', alpha=0.7, marker='x', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Ensemble Predictions with Uncertainty and True Values')
    plt.legend()
    plt.savefig('ensemble_predictions_with_uncertainty.png')
    plt.close()



##简单的resnet模型，仅估计均值
def model_simpleResnet0(unit=256, layNum=10, batch_size=64, epochs=50,test_size=0.9):

    df = pd.read_csv('trainsamples_lane_5_6_7.csv')

    # Prepare the data
    features = df.drop(columns=['carID', 'lane', 'frameNum', 'time_to_vanish', 'min_speed']).values
    targets = df[['time_to_vanish']].copy()
    targets['time_to_vanish'] = targets['time_to_vanish']  # Adjust time_to_vanish by subtracting frameNum，数据准备阶段已经减去当前frameNum
    targets = targets.values/30  # Normalize targets to minutes assuming 30 FPS

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=test_size, random_state=42)

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

##双模型协同校准，把物理仿真模型的预测结果作为输入特征之一，和其他特征一起输入到resnet模型中进行训练和预测
##训练是损失函数同时包括与真实结果误差和与物理仿真模型预测结果的误差
##进一步根据仿真结果给出结果分布概率，预测进行蒙特卡洛仿真，并对仿真结果和结果分布概率进行加权融合校正（后期加入后验证概率校准的理论和方法）
##看能不能用AI编写一个跟车小有小车的控制器，进行跟车控制

if __name__ == "__main__":

    #model_simpleResnet0(unit=256,layNum=30,batch_size=640*20,epochs=500)
    #model_with_MCDDropout(unit=256, layNum=10, batch_size=64, epochs=50, test_size=0.9, mc_samples=10)
    #model_with_ensemble(unit=256, layNum=10, batch_size=64, epochs=50, test_size=0.9, ensemble_size=5) 
