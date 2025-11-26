import pandas as pd

import numpy as np
import os
import random
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull
import sys
import matplotlib.pyplot as plt
import imageio.v2 as imageio

######################################################################################
######################################################################################
# 读取CSV文件并提取特定道路的外围盒数据
#extract_traj_data函数的说明

def extract_traj_data(file_path,laneId=[1,2,3]):
    
    """
    提取车辆轨迹数据并生成可视化图像。
    参数:
        file_path (str): 输入CSV文件的路径，包含车辆轨迹数据。
        laneId (list, 可选): 要分析的车道ID列表，默认为 [1, 2, 3]。
    功能:
        1. 从指定的CSV文件中读取车辆轨迹数据。
        2. 根据指定的车道ID过滤数据。
        3. 为不同的车道分配颜色并绘制车辆中心点的散点图。
        4. 计算车辆轨迹的凸包和边界框，并在图像中绘制。
        5. 保存生成的轨迹图像到当前文件夹中。
    返回:
        tuple: 包含以下内容的元组:
            - df1 (DataFrame): 过滤后的车辆轨迹数据。
            - points (ndarray): 车辆中心点的坐标数组。
            - hull_points (ndarray): 凸包顶点的坐标数组。
    """

    # Read the CSV file
    df = pd.read_csv(file_path)
    # Select only the required columns
    columns_to_analyze = ['frameNum', 'carId', 'carCenterX', 'carCenterY', 
                          'headX', 'headY', 'tailX', 'tailY', 'speed', 'laneId']
    df1= df[columns_to_analyze]
    # Filter data based on the specified lane IDs
    df1 = df1[df1['laneId'].isin(laneId)]

    # Assign colors to different lane IDs
    lane_colors = {lane: plt.cm.tab10(i) for i, lane in enumerate(laneId)}

    # Plot points with different colors for each lane ID
    plt.figure()
    for lane in laneId:
        lane_data = df1[df1['laneId'] == lane]
        plt.scatter(lane_data['carCenterX'], lane_data['carCenterY'], 
                    color=lane_colors[lane], label=f"Lane {lane}", s=10)

    #只要取车的轨迹数据
    points = df1[['carCenterX', 'carCenterY']].values

    hull = ConvexHull(points)

    # Extract the vertices of the convex hull
    hull_points = points[hull.vertices]
    # Calculate the minimum and maximum X and Y coordinates
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    print(f"Minimum X: {min_x}, Minimum Y: {min_y}")
    print(f"Maximum X: {max_x}, Maximum Y: {max_y}")

    plt.figure()
    #plt.plot(df1['carCenterX'], df1['carCenterY'], label='Car Center')
    # Draw the convex hull
    plt.xlim(0, 3840)
    plt.ylim(0, 2160)

    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', label='Convex Hull')
    # Draw the bounding box
    bounding_box_x = [min_x, max_x, max_x, min_x, min_x]
    bounding_box_y = [min_y, min_y, max_y, max_y, min_y]
    plt.plot(bounding_box_x, bounding_box_y, 'g-', label='Bounding Box')
    plt.title(f"Trajectory of Car")
    plt.xlabel("X Coordinate (Image)")
    plt.ylabel("Y Coordinate (Image)")
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # 反转y轴
    plt.show()
    # Save the plot as an image in the current folder
    output_image_path = "trajectory_plot.jpg"

    #output_image_path = os.path.join(os.path.dirname(file_path), "trajectory_plot.png")
    plt.savefig(output_image_path)
    print(f"Plot saved as {output_image_path}")
    plt.close()
    return df1, points, hull_points

######################################################################################
######################################################################################
# Extract trajectory data for a specific lane
#生成车道的轨迹图

def extract_lane_traj_data(file_path,laneId=[1,2,3]):
    """
    提取并可视化指定车道的车辆轨迹数据。
    参数:
        file_path (str): CSV文件的路径，包含车辆轨迹数据。
        laneId (list, 可选): 要分析的车道ID列表，默认为[1, 2, 3]。
    功能:
        1. 从指定的CSV文件中读取车辆轨迹数据。
        2. 筛选出指定车道ID的数据。
        3. 为每个车道分配不同的颜色，支持多达50种颜色。
        4. 绘制车辆轨迹图，显示不同车道的车辆中心点分布。
    注意:
        - 图像的X轴和Y轴分别表示图像中的坐标。
        - Y轴方向被反转以匹配图像坐标系。
        - 图例显示每个车道的颜色和对应的车道ID。
    返回:
        无返回值。函数直接显示绘制的图像。
    """

    # Read the CSV file
    df = pd.read_csv(file_path)
    # Select only the required columns
    columns_to_analyze = ['frameNum', 'carId', 'carCenterX', 'carCenterY', 
                          'headX', 'headY', 'tailX', 'tailY', 'speed', 'laneId']
    df1= df[columns_to_analyze]
    # Filter data based on the specified lane IDs
    df1 = df1[df1['laneId'].isin(laneId)]

    # Assign colors to different lane IDs, supporting more than 50 colors
    color_map = plt.cm.get_cmap('tab20', 50)  # Use 'tab20' colormap with 50 distinct colors
    lane_colors = {lane: color_map(i % 50) for i, lane in enumerate(laneId)}

    # Add a random offset to the color index for each lane
    lane_colors = {lane: color_map((i + random.randint(0, 50)) % 50) for i, lane in enumerate(laneId)}


    
    # Plot points with different colors for each lane ID
    plt.figure()
    plt.xlim(0, 3840) #4K图像
    plt.ylim(0, 2160)
    plt.title(f"Trajectory of different lanes,35 lanes,lane35 is the center area")
    plt.xlabel("X Coordinate (Image)")
    plt.ylabel("Y Coordinate (Image)")
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # 反转y轴
 
    
    for lane in laneId:
        lane_data = df1[df1['laneId'] == lane]
        plt.scatter(lane_data['carCenterX'], lane_data['carCenterY'], 
                    color=lane_colors[lane], label=f"Lane {lane}", s=10)
    plt.legend(ncol=3, fontsize='small', loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.legend()
    plt.show()
    plt.close()

    return

######################################################################################
######################################################################################
#提取每条道路的红灯时间，也就是道路终点有车停的时间

#根据数据提取制定laneID道路的开始位置和结束位置，以及车道的方向.
#1.由feet转换为米 
# 分析['frameNum', 'carId', 'carCenterXft', 'carCenterYft', 'headXft','headYft','tailXft','tailXft','speed', 'heading','laneId']
#2. 计算车道的方向
#3. 计算车道的起点和终点
#4. 计算每辆车每个时刻距离车道终点的距离
#5. 计算每辆车在距离车道终点3米内，速度小于1米每秒的时间段
#6. 计算每辆车在车道内的持续时间
#7. 计算每辆车在速度小于1米每秒的时间段
# save all the results to a CSV file
def extract_one_lane_redlight_dataBymeters(file_path, laneId=1):
    """
    从CSV文件中提取并分析特定车道的数据，单位为英尺，并将其转换为米。
    该函数旨在识别在车道末端附近满足特定速度和距离条件的车辆，并将结果保存到CSV文件中。

    参数:
    -----------
    file_path : str
        包含交通数据的CSV文件路径，数据单位为英尺。
    laneId : int, optional
        要分析的车道ID（默认值为1）。

    功能:
    1. 读取CSV文件并选择相关列。
    2. 将坐标和速度从英尺/秒转换为米/秒。
    3. 根据车道中车辆的行驶方向确定车道的起点和终点。
    4. 计算每辆车到车道终点的距离。
    5. 筛选出距离车道终点3米以内且速度小于1米/秒的车辆。
    6. 计算每辆车在车道内的总停留时间。
    7. 计算每辆车在低速（<1 m/s）状态下的总持续时间。
    8. 将分析结果（包括车道信息、车辆持续时间、低速持续时间和满足红灯条件的车辆）保存到CSV文件中。

    返回:
    --------
  
    car_durations : pandas.DataFrame
        一个DataFrame，包含每辆车在车道末端附近满足条件的持续时间（以帧为单位）。
        包括以下列：
        - 'min': 车辆首次满足条件的帧号。
        - 'max': 车辆最后一次满足条件的帧号。
        - 'duration': 车辆满足条件的总持续时间（以帧为单位）。
    filtered_vehicles : pandas.DataFrame
        一个DataFrame，包含满足条件的车辆的详细信息，条件为距离车道末端在阈值范围内，
        且速度低于指定阈值。包括以下列：
        - 'frameNum': 帧号。
        - 'carId': 车辆ID。
        - 'speed': 车辆速度。
        - 'distance_to_end': 车辆到车道末端的距离。
    """
    # 1. 读取CSV文件并转换单位
    df = pd.read_csv(file_path)
    
    # 选择分析所需的列
    columns_to_analyze = ['frameNum', 'carId', 'carCenterXft', 'carCenterYft', 
                          'headXft', 'headYft', 'tailXft', 'tailYft', 
                          'speed', 'heading', 'laneId']
    df = df[columns_to_analyze]

    # 按指定的laneId筛选数据
    lane_data = df[df['laneId'] == laneId].copy()
    if lane_data.empty:
        print(f"No data found for laneId {laneId}")
        return

    # 将英尺转换为米 (1 foot = 0.3048 meters)
    ft_to_m = 0.3048
    for col in ['carCenterXft', 'carCenterYft', 'headXft', 'headYft', 'tailXft', 'tailYft']:
        lane_data[col.replace('ft', '_m')] = lane_data[col] * ft_to_m
    
    # 假设速度单位是Miles per Hour，转换为米/秒
    mph_to_mps = 0.44704
    lane_data['speed_mps'] = lane_data['speed'] * mph_to_mps
 

    # 2. 计算车道方向
    # 使用第一辆车的数据来确定方向
    first_vehicle = lane_data.iloc[0]
    direction_vector = np.array([first_vehicle['headX_m'] - first_vehicle['tailX_m'], 
                                 first_vehicle['headY_m'] - first_vehicle['tailY_m']])
    
    # 3. 计算车道的起点和终点
    # 根据车流方向确定起点和终点
    # 我们假设车流方向与坐标轴大致对齐
    if abs(direction_vector[0]) > abs(direction_vector[1]): # 主要沿X轴移动
        if direction_vector[0] > 0: # X正方向
            start_idx = lane_data['carCenterX_m'].idxmin()
            end_idx = lane_data['carCenterX_m'].idxmax()
        else: # X负方向
            start_idx = lane_data['carCenterX_m'].idxmax()
            end_idx = lane_data['carCenterX_m'].idxmin()
    else: # 主要沿Y轴移动
        if direction_vector[1] > 0: # Y正方向
            start_idx = lane_data['carCenterY_m'].idxmin()
            end_idx = lane_data['carCenterY_m'].idxmax()
        else: # Y负方向
            start_idx = lane_data['carCenterY_m'].idxmax()
            end_idx = lane_data['carCenterY_m'].idxmin()

    start_of_lane = lane_data.loc[start_idx, ['carCenterX_m', 'carCenterY_m']].values
    end_of_lane = lane_data.loc[end_idx, ['carCenterX_m', 'carCenterY_m']].values

    # 4. 计算每辆车到车道终点的距离
    lane_data['distance_to_end_m'] = np.sqrt(
        (lane_data['carCenterX_m'] - end_of_lane[0])**2 +
        (lane_data['carCenterY_m'] - end_of_lane[1])**2
    )

    # 5. 筛选满足满足红灯条件下在红灯附件停止的车辆（也就是头车）
    red_light_condition = (lane_data['distance_to_end_m'] < 6) & (lane_data['speed_mps'] < 1)
    red_light_vehicles = lane_data[red_light_condition]
    red_light_vehicles = red_light_vehicles [['frameNum', 'carId', 'speed_mps', 'distance_to_end_m']]
    print(f"Lane {laneId} - Vehicles near end with low speed:\n",red_light_vehicles [['frameNum', 'carId', 'speed_mps', 'distance_to_end_m']][:5])
    print(f"Lane {laneId} - red_light_vehicles carids:",red_light_vehicles['carId'].unique())
    # 6. 计算满足红灯条件下在红灯附件停止的车辆（也就是头车）,在车道内的持续时间
    car_total_duration = red_light_vehicles.groupby('carId')['frameNum'].agg(['min', 'max'])
    car_total_duration['duration_frames'] = car_total_duration['max'] - car_total_duration['min']
    #car_total_duration.rename(columns={'min': 'entry_frame', 'max': 'exit_frame'}, inplace=True)

    # 7. 计算每辆车低速行驶的时间
    low_speed_vehicles = lane_data[lane_data['speed_mps'] < 1]
    low_speed_duration = low_speed_vehicles.groupby('carId').size().reset_index(name='low_speed_frames')

    # 8. 保存每个结果到不同的CSV文件
    # 虽然下面的代码是保存到Excel的不同sheet，但概念上是分开保存结果
    output_filename_base = f"lane_{laneId}_analysis_by_meter"
    
    # 保存车道信息
    lane_info_df = pd.DataFrame({
        'laneId': [laneId],
        'start_of_lane_m': [str(start_of_lane)],
        'end_of_lane_m': [str(end_of_lane)]
    })
    # lane_data只保存frameNum, carId, carCenterX_m, carCenterY_m, speed_mps, distance_to_end_m,speed
    columns_to_save = ['frameNum', 'carId', 'carCenterX_m', 'carCenterY_m', 'speed_mps', 'distance_to_end_m', 'speed']
    lane_data[columns_to_save].to_csv(f"{output_filename_base}_all_data.csv", index=False)

    lane_info_df.to_csv(f"{output_filename_base}_lane_info.csv", index=False)

    # 保存车辆总持续时间
    car_total_duration.to_csv(f"{output_filename_base}_total_duration.csv")

    # 保存车辆低速持续时间
    low_speed_duration.to_csv(f"{output_filename_base}_low_speed_duration.csv", index=False)

    # 保存满足红灯条件的车辆帧
    red_light_vehicles.to_csv(f"{output_filename_base}_red_light_frames.csv", index=False)

    print(f"Analysis files for lane {laneId} saved with base name {output_filename_base}")
    
   
    return car_total_duration,red_light_vehicles

##废弃，因为车辆的位置都是以像素为单位的，重写了一个函数以米作为单位
def extract_one_lane_redlight_data(file_path, laneId=1):
    """
    从CSV文件中提取并分析特定车道的数据，识别在车道末端附近满足特定速度和距离条件的车辆。
    参数:
    -----------
    file_path : str
        包含交通数据的CSV文件路径。
    laneId : int, optional
        要分析的车道ID（默认值为1）。
    返回值:
    --------
    car_durations : pandas.DataFrame
        一个DataFrame，包含每辆车在车道末端附近满足条件的持续时间（以帧为单位）。
        包括以下列：
        - 'min': 车辆首次满足条件的帧号。
        - 'max': 车辆最后一次满足条件的帧号。
        - 'duration': 车辆满足条件的总持续时间（以帧为单位）。
    filtered_vehicles : pandas.DataFrame
        一个DataFrame，包含满足条件的车辆的详细信息，条件为距离车道末端在阈值范围内，
        且速度低于指定阈值。包括以下列：
        - 'frameNum': 帧号。
        - 'carId': 车辆ID。
        - 'speed': 车辆速度。
        - 'distance_to_end': 车辆到车道末端的距离。
    注意:
    ------
    - 函数根据车道中第一辆车的头部和尾部位置计算车道方向。
    - 根据车道方向确定车道的起点和终点。
    - 根据车辆到车道末端的距离（默认阈值：118像素）和速度（默认阈值：5 m/s）过滤车辆。
    - 函数会打印中间结果，包括车道方向、车道起点和终点坐标，以及过滤后的车辆数据。
    示例:
    --------
    car_durations, filtered_vehicles = extract_one_lane_redlight_data("traffic_data.csv", laneId=2)
    """
 # Read the CSV file
    df = pd.read_csv(file_path)
    # Select only the required columns
    columns_to_analyze = ['frameNum', 'carId', 'carCenterX', 'carCenterY', 
                          'headX', 'headY', 'tailX', 'tailY', 'speed', 'heading','laneId']
    df1= df[columns_to_analyze]
    # Filter data based on the specified lane IDs
    #df1 = df1[df1['laneId'].isin(laneId)]

    # Assign colors to different lane IDs, supporting more than 50 colors
    lane_data = df1[df1['laneId'] == laneId]

    #获得每辆车的方向，并对应每条车道的方向，以及确定车道的起始点
    # Calculate the direction of the lane based on the head and tail coordinates
    # Extract the head and tail positions of one vehicle
    veh_data = lane_data.iloc[0]  # Select the first vehicle in the lane
    head_position = np.array([veh_data['headX'], veh_data['headY']])
    tail_position = np.array([veh_data['tailX'], veh_data['tailY']])
    # Calculate the distance between head and tail for each vehicle
    #lane_data['head_tail_distance'] = np.sqrt(
    #    (lane_data['headX'] - lane_data['tailX'])**2 +
    #    (lane_data['headY'] - lane_data['tailY'])**2
    #)

    # Calculate statistics for the head-tail distance
    #head_tail_distance_stats = lane_data['head_tail_distance'].describe()
    #print("Head-Tail Distance Statistics:")
    #print(head_tail_distance_stats)
    # Calculate the direction vector of the vehicle
    vehicle_direction = head_position - tail_position
    vehicle_direction = vehicle_direction / np.linalg.norm(vehicle_direction)  # Normalize the direction vector
    
    # Calculate the direction of the lane based on the vehicle direction
    # Convert the vehicle direction vector to an angle in degrees (0-360)
    #事实上不需要计算，车的朝向已经给出
    angle = np.degrees(np.arctan2(vehicle_direction[1], vehicle_direction[0]))
    if angle < 0:
        angle += 360  # Ensure the angle is in the range [0, 360]
    lane_direction = angle
    print(f"Lane {laneId} direction vector: {angle}")
    # Determine the start and end points of the lane based on the direction
    if 0 <= lane_direction < 90:  # Positive X and Y direction
        start_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].min()]
        end_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].max()]
    elif 90 <= lane_direction < 180:  # Negative X and Positive Y direction
        start_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].max()]
        end_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].min()]
    elif 180 <= lane_direction < 270:  # Negative X and Y direction
        start_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].max()]
        end_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].min()]
    else:  # Positive X and Negative Y direction
        start_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].min()]
        end_of_lane_data = lane_data[lane_data['carCenterX'] == lane_data['carCenterX'].max()]
   
    start_of_lane_coords1 = start_of_lane_data[['carCenterX', 'carCenterY']].values
    end_of_lane_coords1 = end_of_lane_data[['carCenterX', 'carCenterY']].values

    print(f"Start of lane {laneId} coordinates: {start_of_lane_coords1}")
    print(f"End of lane {laneId} coordinates: {end_of_lane_coords1}")
 
    
    # Extract the coordinates for the start and end points
    start_of_lane_coords = start_of_lane_data[['carCenterX', 'carCenterY']].iloc[0].values
    end_of_lane_coords = end_of_lane_data[['carCenterX', 'carCenterY']].iloc[0].values

    print(f"Start of lane {laneId} coordinates: {start_of_lane_coords}")
    print(f"End of lane {laneId} coordinates: {end_of_lane_coords}")

    
    #根据上下文，lanedata里面是一段时间内车辆的位置和速度以及对应的时间，
    #给出距离车道终点3米内，速度小于1米每秒的车辆以及对应的时间（也就是frameNum）
    # Filter vehicles within 3 meters of the lane end and with speed less than 1 m/s
    threshold_distance = 118  # Distance threshold in  pixel,统计车长平均值为118，
    speed_threshold = 5  # Speed threshold in meters per second 

    # Calculate the distance of each vehicle from the end of the lane
    X = lane_data['carCenterX'].values
    Y = lane_data['carCenterY'].values
  
    distance_to_end = np.sqrt(
        (X - end_of_lane_coords[0])**2 +
        (Y - end_of_lane_coords[1])**2
    )
   

    lane_data = lane_data.copy()  # Create a copy to avoid SettingWithCopyWarning
    lane_data.loc[:, 'distance_to_end'] = distance_to_end
    #print(lane_data[['frameNum', 'carId', 'carCenterX', 'carCenterY', 'speed', 'heading','distance_to_end']])
    
    # Calculate statistics for 'speed' and 'distance_to_end'
    #speed_stats = lane_data['speed'].describe(percentiles=[0.01,0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    #distance_stats = lane_data['distance_to_end'].describe(percentiles=[0.01,0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    #print("Speed Statistics:")
    #print(speed_stats)
    #print("\nDistance to End Statistics:")
    #print(distance_stats)
    # Filter vehicles meeting the criteria
    filtered_vehicles = lane_data[
        (lane_data['distance_to_end'] <= threshold_distance) &
        (lane_data['speed'] < speed_threshold)
    ]

    # Extract the frame numbers and vehicle IDs
    result = filtered_vehicles[['frameNum', 'carId','speed', 'distance_to_end']]
    print("Vehicles within 118 pixel of the lane end and speed < 1 m/s:")
    
    #print(filtered_vehicles[['frameNum', 'carId', 'carCenterX', 'carCenterY', 'speed', 'heading','distance_to_end']])
   
    
 
# Group by carId and calculate the duration for each car
    car_durations = filtered_vehicles.groupby('carId')['frameNum'].agg(['min', 'max'])
    car_durations['duration'] = car_durations['max'] - car_durations['min']
    print(f"laneID:{laneId} Duration of each carId:")
    print(car_durations)
    
    
    return car_durations,filtered_vehicles

######################################################################################
######################################################################################
#生成车道的轨迹GIF图图

def create_gif_of_car_positions(file_path, laneIds=[4, 5, 6, 7],redConfig=None):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select only the required columns
    columns_to_analyze = ['frameNum', 'carId', 'carCenterX', 'carCenterY', 
                          'headX', 'headY', 'tailX', 'tailY', 'heading', 'speed', 'laneId']
    df = df[columns_to_analyze]

    # Filter data for the specified laneIds
    lane_data = df[df['laneId'].isin(laneIds)]


       # Get all carCenterX and carCenterY points
    points = lane_data[['carCenterX', 'carCenterY']].values

    # Compute the convex hull of the points
    hull = ConvexHull(points)

    # Extract the vertices of the convex hull
    hull_points = points[hull.vertices]

    



    # Get unique frame numbers
    frame_nums = lane_data['frameNum'].unique()
    frame_nums.sort()

    # List to store file paths of generated images
    image_files = []

    # Generate individual frame images
    #frame_nums = frame_nums[0:8000]  # Optionally, reduce the number of frames for the GIF
    for frame in frame_nums:
        print('frame_num:',frame)
        if frame % 10 != 0:
            continue  # Skip frames to reduce the number of images

        plt.figure(figsize=(10, 6))
        plt.title(f"Car Positions in Lanes {laneIds} - Frame {frame}")
        plt.xlabel("X Coordinate (Image)")
        plt.ylabel("Y Coordinate (Image)")
        plt.gca().invert_yaxis()  # Reverse y-axis
        plt.grid()
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label='Convex Hull')

        # Check if the current frame is within any red light interval
        if redConfig is not None:
            merged_redlight_data = redConfig
            is_red_light = any((frame >= row['min'] and frame <= row['max']) for _, row in merged_redlight_data.iterrows())
            light_status = "Red Light" if is_red_light else "Green Light"
            plt.title(f"Car Positions in Lanes {laneIds} - Frame {frame} ({light_status})")

        # Plot car positions for each lane
        for lane_id in laneIds:
            lane_frame_data = lane_data[(lane_data['frameNum'] == frame) & (lane_data['laneId'] == lane_id)]
            if not lane_frame_data.empty:
                plt.scatter(lane_frame_data['carCenterX'], lane_frame_data['carCenterY'], label=f"Lane {lane_id}", s=10)
                for _, row in lane_frame_data.iterrows():
                    plt.text(row['carCenterX'], row['carCenterY'], str(row['carId']), fontsize=8, ha='right')

        plt.legend(ncol=3, fontsize='small', loc='upper right', bbox_to_anchor=(1.1, 1.05))

        # Save the frame as an image
        os.makedirs('frames', exist_ok=True)
        frame_file = f"frames/multilanes_{'_'.join(map(str, laneIds))}_frame_{frame}.png"
        plt.savefig(frame_file, format='png')
        image_files.append(frame_file)
        plt.close()

    # Create a GIF from the saved images
    os.makedirs('gifs', exist_ok=True)
    gif_output_path = f"gifs/lanes_{'_'.join(map(str, laneIds))}_car_positions.gif"
    with imageio.get_writer(gif_output_path, mode='I', fps=5) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"GIF saved as {gif_output_path}")



# Example usage

# Redirect all print statements to a file
if 0:
    log_file_path = "output_log.txt"
    sys.stdout = open(log_file_path, "w+")

if 0:#生成车道的轨迹图
    file_path = 'E:\myData\IntersectionA-01.csv'
    #extract_traj_data(file_path)
    laneIDs = [1,2,3,4,5,6,7,8,9,10,11,12]
    laneIDs = list(range(0, 36))
    laneIDs = [0,1,2,3,4,5,6,7,8]
    extract_lane_traj_data(file_path,laneId=laneIDs)

if 1:#分析交通图,获得每条车道的红灯时间或者lane5,6,7车道的红灯时间
    file_path = 'E:\myData\IntersectionA-01.csv'
    #extract_traj_data(file_path)
    
    #车道0123，没有交通灯，4,5,6,7有交通灯,4是左转向道，5,6,7是直行道
    laneIDs = [5,6,7]
    combined_redlight_data = []

    for laneID in laneIDs:
        #car_durations, filtered_vehicles = extract_one_lane_redlight_data(file_path, laneId=laneID)
        car_durations, filtered_vehicles = extract_one_lane_redlight_dataBymeters(file_path, laneId=laneID)
        combined_redlight_data.append(car_durations)

    # Combine red light durations for lanes 5, 6, and 7
    combined_redlight_data = pd.concat(combined_redlight_data)

    #print(f"Combined red light durations for lanes{laneIDs}:")
    #print(combined_redlight_data)


    combined_redlight_data = combined_redlight_data.groupby('carId').agg({'min': 'min', 'max': 'max'})
    combined_redlight_data['duration'] = combined_redlight_data['max'] - combined_redlight_data['min']

    print(f"Combined red light durations for lanes {laneIDs}:")
    print(combined_redlight_data)


    # Merge overlapping time intervals
    combined_redlight_data = combined_redlight_data.sort_values(by='min')
    merged_intervals = []
    current_start, current_end = None, None

    for _, row in combined_redlight_data.iterrows():
        start, end = row['min'], row['max']
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:  # Overlapping intervals
            current_end = max(current_end, end)
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end

    if current_start is not None:
        merged_intervals.append((current_start, current_end))

    # Convert merged intervals back to a DataFrame
    merged_redlight_data = pd.DataFrame(merged_intervals, columns=['min', 'max'])
    merged_redlight_data['duration'] = merged_redlight_data['max'] - merged_redlight_data['min']

    print(f"Combined red light durations for lanes {laneIDs}:")
    print(merged_redlight_data)
    

if 0:#生成车道的轨迹GIF图图
    file_path = 'E:\myData\IntersectionA-01.csv'
    #extract_traj_data(file_path)
    laneIDs = [1,2,3,4,5,6,7,8,9,10,11,12]
    laneIDs = list(range(0, 36))
    laneIDs = [0,1,2,3,4,5,6,7,8]
    laneIDs = [6,7,8]
    create_gif_of_car_positions(file_path,laneIds=laneIDs,redConfig=merged_redlight_data)



   