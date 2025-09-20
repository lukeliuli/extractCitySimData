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

# Example usage



# Example usage
if 0:
    file_path = 'E:\myData\IntersectionA-01.csv'
    output_dir = 'E:\myData\IntersectionA-01-output'
    extract_laneid_data(file_path, output_dir)
if 0:
    file_path = 'E:\\myData\\IntersectionA-01-output\\4_data.csv'  # Replace with the path to your CSV file
    analyze_and_plot_lane_data(file_path)

if 1:
    file_path = 'E:\\myData\\IntersectionA-01-output\\4_data.csv'  # Replace with the path to your CSV file
    create_gif_of_car_positions(file_path)

