import pandas as pd
import numpy as np
import os
import random
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


# 读取CSV文件并提取特定道路的数据
def extract_traj_data(file_path,laneId=[1,2,3]):
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

# Extract trajectory data for a specific lane
def extract_lane_traj_data(file_path,laneId=[1,2,3]):
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

# Example usage
if 1:
    file_path = 'E:\myData\IntersectionA-01.csv'
    #extract_traj_data(file_path)
    laneIDs = [1,2,3,4,5,6,7,8,9,10,11,12]
    laneIDs = list(range(0, 36))
    extract_lane_traj_data(file_path,laneId=laneIDs)
   