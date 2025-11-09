import pandas as pd
import numpy as np
import os
import random
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


# 读取CSV文件并提取特定道路的数据
def extract_laneid_data(file_path, output_dir):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique city names
    laneIds = df['laneId'].unique()
    
    for laneId in laneIds:
        # Filter data for the current city
        laneId_data = df[df['laneId'] == laneId]
        
        # Save to a new CSV file named after the city
        laneId_file_path = os.path.join(output_dir, f"{laneId}_data.csv")
        laneId_data.to_csv(laneId_file_path, index=False)
        print(f"Data for laneid:{laneId} saved to {laneId_file_path}")


# 分析并绘制特定车辆的在制定道路上的轨迹数据
def analyze_and_plot_lane_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Filter data for the specified laneId
    lane_data = df
    
    # Select only the required columns
    columns_to_analyze = ['frameNum', 'carId', 'carCenterX', 'carCenterY', 
                          'headX', 'headY', 'tailX', 'tailY', 'speed', 'laneId']
    lane_data = lane_data[columns_to_analyze]
    
    lane_id = lane_data['laneId'].iloc[0]
    print(f"Analyzing data for laneId: {lane_id}")

    
    # Get unique carIds in the lane
    car_ids = lane_data['carId'].unique()
    for  selected_car_id in car_ids:
        # Extract trajectory data for the selected car
        car_data = lane_data[lane_data['carId'] == selected_car_id]
        
        # Plot the trajectory using frameNum as the index
        
        plt.figure(figsize=(10, 6))
        plt.plot(car_data['carCenterX'], car_data['carCenterY'], label='Car Center')
        #plt.plot(car_data['headX'], car_data['headY'], label='Car Head')
        #plt.plot(car_data['tailX'], car_data['tailY'], label='Car Tail')
        plt.title(f"Trajectory of Car {selected_car_id} in Lane {lane_id}")
        plt.xlabel("X Coordinate (Image)")
        plt.ylabel("Y Coordinate (Image)")
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()  # 反转y轴
        #plt.show()
        
        # Save the plot as a JPG file in the current directory
        os.makedirs('trajs', exist_ok=True)
        output_filename = f"trajs\car_{selected_car_id}_lane_{lane_id}_trajectory.jpg"
        plt.savefig(output_filename, format='jpg')
        print(f"Plot saved as {output_filename}")
        plt.close()

# Create a GIF of car positions for each frameNum using imageio
import imageio.v2 as imageio
def create_gif_of_car_positions(file_path):
 

    # Read the CSV file
    df = pd.read_csv(file_path)
    # Filter data for the specified laneId
    lane_data = df

    # Select only the required columns
    columns_to_analyze = ['frameNum', 'carId', 'carCenterX', 'carCenterY', 
                          'headX', 'headY', 'tailX', 'tailY', 'speed', 'laneId']
    lane_data = lane_data[columns_to_analyze]
    lane_id = lane_data['laneId'].iloc[0]
    print(f"Analyzing data for laneId: {lane_id}")

    # Get all carCenterX and carCenterY points
    points = lane_data[['carCenterX', 'carCenterY']].values

    # Compute the convex hull of the points
    hull = ConvexHull(points)

    # Extract the vertices of the convex hull
    hull_points = points[hull.vertices]

    

    # Plot the points and the minimum bounding rectangle
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], label='Car Center Points', color='blue')
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label='Convex Hull')
    plt.gca().invert_yaxis()  # Reverse y-axis



    plt.title(f"Car Center Points and Bounding Rectangle in Lane {lane_id}")
    plt.xlabel("X Coordinate (Image)")
    plt.ylabel("Y Coordinate (Image)")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
  
  
    # Get unique frame numbers
    frame_nums = lane_data['frameNum'].unique()
    frame_nums.sort()

    # List to store file paths of generated images
    image_files = []

    # Generate individual frame images
    frame_nums = frame_nums[0:2000]  # Optionally, reduce the number of frames for the GIF
    for frame in frame_nums:
        if frame % 10 != 0:
            continue  # Skip frames to reduce the number of images

        plt.figure(figsize=(10, 6))
        plt.title(f"Car Positions in Lane {lane_id} - Frame {frame}")
        plt.xlabel("X Coordinate (Image)")
        plt.ylabel("Y Coordinate (Image)")
        plt.gca().invert_yaxis()  # Reverse y-axis
        plt.grid()

        # 先画所有路径点和凸包
        #plt.scatter(points[:, 0], points[:, 1], label='Car Center Points', color='lightgray')
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label='Convex Hull')

        # Filter data for the current frame
        frame_data = lane_data[lane_data['frameNum'] == frame]

        # Plot car positions and annotate with car IDs
        for _, row in frame_data.iterrows():
            plt.scatter(row['carCenterX'], row['carCenterY'], label=f"Car {row['carId']}")
            plt.text(row['carCenterX'], row['carCenterY'], str(row['carId']), fontsize=8, ha='right')

        # Save the frame as an image
        os.makedirs('frames', exist_ok=True)
        frame_file = f"frames/lane_{lane_id}_frame_{frame}.png"
        plt.savefig(frame_file, format='png')
        image_files.append(frame_file)
        plt.close()

    # Create a GIF from the saved images
    os.makedirs('gifs', exist_ok=True)
    gif_output_path = f"gifs/lane_{lane_id}_car_positions.gif"
    with imageio.get_writer(gif_output_path, mode='I', fps=5) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"GIF saved as {gif_output_path}")

    # Optionally, clean up the individual frame images
    #for image_file in image_files:
    #    os.remove(image_file)

####################################################################################
####################################################################################
##生成用于分析和训练的车道数据样本，数据集

def gen_samples(input_csv,output_dir,laneIds,redlightConfig,start_of_lane_coords,end_of_lane_coords):

    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    #laneIds = df['laneId'].unique()
    for laneId in laneIds:
        lane_data = df[df['laneId'] == laneId]
        lane_file_path = os.path.join(output_dir, f"{laneId}_data.csv")
        lane_data.to_csv(lane_file_path, index=False)
        print(f"Data for laneid:{laneId} saved to {lane_file_path}")
        #1.获得红灯状态而且停车线有车辆时的每一帧数据，以及对应红灯剩余时间
        #2.根据每一帧数据和对应的红灯剩余时间，获得当前车道每一辆车的位置和速度，并且按照到停止线的距离进行排序
        #3.根据每一帧数据和对应的红灯剩余时间，获得道路上的所有车辆根据carID搜索整个出现时间的速度变换速度（如最大速度，平均速度，最小速度）
        #还包括最大加速度、最小加速度、经过停止线的时间以及与前车的距离，与前车的距离变化率。以及红灯变绿灯后，每一辆车的起步时间和起步加速度
        #还包括每辆车经过停止线的时间，前车经过停止线的时间和后车经过停止线的时间以及之间的时间差。
        #4.计算红灯变绿灯后，根据每辆车经过停止线的时间，前车经过停止线的时间和后车经过停止线的时间以及之间的时间差。
        # 加上车辆的位置和速度，进行多维高斯或者log建模对排队车辆进行过停止线的时间进行预测
        #5.根据车辆的位置和速度，本车道的历史数据和前5秒的速度，基于跟车动态模型，预测每辆车从开始位置到停止线的所有状态信息，以及停止线的时间进行预测
        #6.结合神经网络模型、概率参数模型、跟车动态模型，综合提高预测的结果的正确性   
        
        ##---------------------------------------------------------------------------------------------------
        ####1.获得红灯状态而且停车线有车辆时的每一帧数据，以及对应红灯剩余时间
        red_light_frames = []
        red_light_remaining_time = {}

        for _, row in redlightConfig.iterrows():
            for frame in range(row['min'], row['max'] + 1):
                red_light_frames.append(frame)
                red_light_remaining_time[frame] = row['max'] - frame + 1

        # Filter the dataset for frames during red light
        red_light_data = lane_data[lane_data['frameNum'].isin(red_light_frames)]

        # Add a column for red light remaining time
        red_light_data['redLightRemainingTime'] = red_light_data['frameNum'].map(red_light_remaining_time)

        #---------------------------------------------------------------------------------------------------
        ####2.根据每一帧数据和对应的红灯剩余时间，获得当前车道每一辆车的位置和速度，并且按照到停止线的距离进行排序
        # Get the stop line coordinates for the current lane
        stop_line_x, stop_line_y = end_of_lane_coords[laneId]

        # Calculate the distance of each car to the stop line
        red_light_data['distanceToStopLine'] = red_light_data.apply(
            lambda row: np.sqrt((row['carCenterX'] - stop_line_x) ** 2 + (row['carCenterY'] - stop_line_y) ** 2),
            axis=1
        )

        #对每个red_light_data的每个frameNum生成一个记录并保存为csv文件,其中每个记录包括:
        #frameNum,redLightRemainingTime,当前车道的按照到停止线距离从小到大排列的所有车辆的carid,位置和速度,总共20辆车，不足补0，有多的取前20辆
        # Group data by frameNum
        grouped = red_light_data.groupby('frameNum')

        # Prepare output directory for the generated CSV files
        os.makedirs(output_dir, exist_ok=True)

        for frame, frame_data in grouped:
            # Sort cars by distance to the stop line
            sorted_cars = frame_data.sort_values(by='distanceToStopLine')

            # Extract car IDs, positions, and speeds
            car_ids = sorted_cars['carId'].tolist()
            car_positions = sorted_cars[['carCenterX', 'carCenterY']].values.tolist()
            car_speeds = sorted_cars['speed'].tolist()

            # Limit to 20 cars, pad with zeros if fewer
            max_cars = 20
            car_ids = car_ids[:max_cars] + [-1] * (max_cars - len(car_ids))
            car_positions = car_positions[:max_cars] + [[-1, -1]] * (max_cars - len(car_positions))
            car_speeds = car_speeds[:max_cars] + [-1] * (max_cars - len(car_speeds))

            # Create a record for the current frame
            record = {
            'frameNum': frame,
            'redLightRemainingTime': red_light_remaining_time[frame],
            'carIds': car_ids,
            'carPositions': car_positions,
            'carSpeeds': car_speeds
            }

            # Save the record to a CSV file
            output_file = os.path.join(output_dir, f"lane_{laneId}_frame_{frame}_queue_data.csv")
            pd.DataFrame([record]).to_csv(output_file, index=False)
            print(f"Saved frame {frame} data to {output_file}")

        #---------------------------------------------------------------------------------------------------
        ####3.根据frameNum和laneID，生成一个记录，获得道路上的所有车辆根据carID,搜索和计算一下特征
        # 车道的每辆车的最大速度，最小速度，平均速度，最大加速度，最小加速度，经过停止线的时间frameNum,
        # Group data by carId
        car_grouped = lane_data.groupby('carId')

        # Prepare a list to store the records
        car_features = []

        for car_id, car_data in car_grouped:
            # Calculate speed statistics
            max_speed = car_data['speed'].max()
            min_speed = car_data['speed'].min()
            avg_speed = car_data['speed'].mean()

            # Calculate acceleration statistics
            car_data = car_data.sort_values(by='frameNum')
            car_data['acceleration'] = car_data['speed'].diff() / car_data['frameNum'].diff()
            max_acceleration = car_data['acceleration'].max()
            min_acceleration = car_data['acceleration'].min()

            # Determine the frameNum when the car crosses the stop line
            stop_line_x, stop_line_y = end_of_lane_coords[laneId]
            car_data['distanceToStopLine'] = car_data.apply(
            lambda row: np.sqrt((row['carCenterX'] - stop_line_x) ** 2 + (row['carCenterY'] - stop_line_y) ** 2),
            axis=1
            )
            crossing_frame1 = car_data[car_data['distanceToStopLine'] <= 120]['frameNum'].max()  # Threshold for crossing
            crossing_frame2_vanish = car_data['frameNum'].max()  # Threshold for crossing，如果没有经过停止线，就取最后一帧

            # Create a record for the car
            car_features.append({
            'carId': car_id,
            'maxSpeed': max_speed,
            'minSpeed': min_speed,
            'avgSpeed': avg_speed,
            'maxAcceleration': max_acceleration,
            'minAcceleration': min_acceleration,
            'crossingFrame1': crossing_frame1 if not np.isnan(crossing_frame1) else None,
            'crossingFrame2_vanish': crossing_frame2_vanish
            })

        # Save the car features to a CSV file
        car_features_df = pd.DataFrame(car_features)
        output_file = os.path.join(output_dir, f"lane_{laneId}_car_features.csv")
        car_features_df.to_csv(output_file, index=False)
        print(f"Saved car features for lane {laneId} to {output_file}")
        
        #---------------------------------------------------------------------------------------------------
        ####4.计算绿灯持续时间，并在绿灯期间计算每辆车经过停止线的时间及相关信息，并排序
        # Identify green light periods based on red light configuration
        green_light_periods = []
        for i in range(len(redlightConfig) - 1):
            green_start = redlightConfig.iloc[i]['max'] + 1
            green_end = redlightConfig.iloc[i + 1]['min'] - 1
            green_light_periods.append((green_start, green_end))

        # Prepare a list to store green light data
        green_light_data = []

        for green_start, green_end in green_light_periods:
            # Filter data for the green light period
            green_period_data = lane_data[(lane_data['frameNum'] >= green_start) & (lane_data['frameNum'] <= green_end)]

            # Group data by carId
            car_grouped = green_period_data.groupby('carId')

            for car_id, car_data in car_grouped:
            # Determine the frameNum when the car crosses the stop line
                car_data = car_data.sort_values(by='frameNum')
                car_data['distanceToStopLine'] = car_data.apply(
                    lambda row: np.sqrt((row['carCenterX'] - stop_line_x) ** 2 + (row['carCenterY'] - stop_line_y) ** 2),
                    axis=1
                )
                crossing_frame = car_data[car_data['distanceToStopLine'] <= 120]['frameNum'].min()  # Threshold for crossing

                # Get the car's position and speed at the start of the green light
                initial_data = car_data[car_data['frameNum'] == green_start]
                if not initial_data.empty:
                    initial_position = (initial_data.iloc[0]['carCenterX'], initial_data.iloc[0]['carCenterY'])
                    initial_speed = initial_data.iloc[0]['speed']
                else:
                    initial_position = (-1, -1)  # Car not in lane at green light start
                    initial_speed = -1

                # Append the data for the car
                green_light_data.append({
                    'carId': car_id,
                    'greenStartFrame': green_start,
                    'crossingFrame': crossing_frame if not np.isnan(crossing_frame) else None,
                    'initialPositionX': initial_position[0],
                    'initialPositionY': initial_position[1],
                    'initialSpeed': initial_speed
                })

        # Save the green light data to a CSV file
        green_light_df = pd.DataFrame(green_light_data)
        output_file = os.path.join(output_dir, f"lane_{laneId}_green_light_data.csv")
        green_light_df.to_csv(output_file, index=False)
        print(f"Saved green light data for lane {laneId} to {output_file}")

# Example usage
##获得从数据中，提取每个道路样本数据
if 0:
    file_path = 'E:\myData\IntersectionA-01.csv'
    output_dir = 'E:\myData\IntersectionA-01-output'
    extract_laneid_data(file_path, output_dir)

##获得道路样本数据的图（test0也有）
if 0:
    file_path = 'E:\\myData\\IntersectionA-01-output\\4_data.csv'  # Replace with the path to your CSV file
    analyze_and_plot_lane_data(file_path)

##获得道路样本数据的DIF图（test0也有）
if 0:
    file_path = 'E:\\myData\\IntersectionA-01-output\\4_data.csv'  # Replace with the path to your CSV file
    create_gif_of_car_positions(file_path)

if 1:
    #基本参数见test0 中函数extract_one_lane_redlight_data 获取output_log.txt的输出
    input_csv = 'E:\myData\IntersectionA-01.csv'
    file_path = 'E:\myData\IntersectionA-01.csv'
    output_dir = 'E:\myData\IntersectionA-01-trainsamples'

    '''
    Combined red light durations for lanes [5, 6, 7]:
        min   max  duration
    0     0   879       879
    1  3730  6592      2862
    2  8255  8999       744
    '''
    redlightConfigNow = pd.DataFrame({
        'min': [0, 3730, 8255],
        'max': [879, 6592 , 8999]
    })
   
    '''
    #车道的起点和终点坐标

    Start of lane 7 coordinates: [  95 1498]
    End of lane 7 coordinates: [1267 1469]
    Start of lane 6 coordinates: [ 100 1410]
    End of lane 6 coordinates: [1263 1372]
    Start of lane 5 coordinates: [ 101 1329]
    End of lane 5 coordinates: [1261 1302]
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

    laneIds=[5,6,7]
    gen_samples(input_csv,output_dir,laneIds,redlightConfigNow,start_of_lane_coordsNow,end_of_lane_coordsNow)
    '''
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
 
    '''    