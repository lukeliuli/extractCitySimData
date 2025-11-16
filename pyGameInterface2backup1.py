import math
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys

@dataclass
class VehicleParams:
    """IDM模型参数配置"""
    v0: float = 50 / 3.6  # 期望速度 (m/s)
    T: float = 1       # 安全车头时距 (s)
    s0: float = 2.0       # 最小跟车距离 (m)
    a: float = 1.0        # 最大加速度 (m/s²)
    b: float = 2.0        # 舒适减速度 (m/s²)
    delta: float = 4.0    # 加速度指数
    length: float = 5.0   # 车辆长度 (m)

@dataclass
class VehicleState:
    """车辆状态"""
    id: int
    distance: float  # 车辆前端在坐标轴上的位置 (m)
    speed: float     # 当前速度 (m/s)
    acceleration: float = 0.0
    has_passed: bool = False
    waiting_time: float = 0.0

class TrafficSimulator:
    def __init__(self, params: VehicleParams, time_step: float = 0.1, intersection_pos: float = 1000.0):
        self.params = params
        self.time_step = time_step
        self.vehicles: List[VehicleState] = []
        self.red_light_remaining: float = 0
        self.max_vehicles: int = 20
        self.simulation_history: List[Dict] = []
        self.intersection_pos = intersection_pos
        self.total_time_simulated = 0.0

    def add_vehicles(self, vehicle_data: List[Dict]):
        """添加车辆到模拟器"""
        for data in vehicle_data:
            if len(self.vehicles) >= self.max_vehicles:
                print(f"警告：车道已满（最多{self.max_vehicles}辆），无法添加更多车辆。")
                break
            if data['distance'] >= self.intersection_pos:
                print(f"警告：车辆 {data['id']} 的初始位置 {data['distance']}m 已在或超过路口 {self.intersection_pos}m，不予添加。")
                continue
            self.vehicles.append(VehicleState(
                id=data['id'],
                distance=data['distance'],
                speed=data['speed']
            ))
        # 车辆始终按位置从后到前排序
        self.vehicles.sort(key=lambda v: v.distance)

    def set_red_light(self, remaining_time: float):
        """设置红灯剩余时间"""
        self.red_light_remaining = remaining_time

    def idm_acceleration(self, vehicle: VehicleState, front_vehicle: Optional[VehicleState] = None) -> float:
        """智能驾驶员模型（IDM）计算加速度"""
        free_acc = self.params.a * (1 - (vehicle.speed / self.params.v0) ** self.params.delta)

        if front_vehicle is None:
            return free_acc

        gap = front_vehicle.distance - vehicle.distance - self.params.length
        if gap < self.params.s0:
            return -9.0  # 紧急制动

        headway = gap / max(vehicle.speed, 0.01) # 避免除以零
        if headway > 3.0: # 大于3倍车头时距，自由行驶
             return free_acc

        desired_gap = self.params.s0 + max(0, vehicle.speed * self.params.T + (vehicle.speed * (vehicle.speed - front_vehicle.speed)) / (2 * math.sqrt(self.params.a * self.params.b)))
        
        # 防止gap过小导致加速度爆炸
        if desired_gap > gap:
             return -self.params.b * (desired_gap / max(gap, 0.1))

        follow_acc = self.params.a * (1 - (vehicle.speed / self.params.v0) ** self.params.delta - (desired_gap / gap) ** 2)
        return follow_acc

    def _get_stopping_acceleration_for_light(self, vehicle: VehicleState) -> float:
        """计算在红灯前停车所需的加速度"""
        dist_to_light = self.intersection_pos - vehicle.distance - self.params.length
        if dist_to_light <= self.params.s0: # 如果已经很近，就紧急刹车
            return -9.0

        # 目标是在路口停车线前以0速度停下
        # v_f^2 = v_i^2 + 2*a*d => a = -v_i^2 / (2*d)
        required_acc = -(vehicle.speed ** 2) / (2 * dist_to_light)
        
        # 使用舒适减速度和所需减速度中更安全（更小）的一个
        return max(required_acc, -self.params.b)

    def update_vehicle_state(self, vehicle: VehicleState, front_vehicle: Optional[VehicleState]):
        """更新单个车辆的状态"""
        if self.red_light_remaining > 0:
            # 红灯逻辑
            dist_to_light = self.intersection_pos - vehicle.distance
            
            # 检查是否需要开始为红灯减速
            stopping_dist_needed = vehicle.speed**2 / (2 * self.params.b)
            
            if front_vehicle:
                # 前方有车，按跟车模型行驶
                vehicle.acceleration = self.idm_acceleration(vehicle, front_vehicle)
            elif dist_to_light <= stopping_dist_needed + self.params.s0:
                # 前方无车，且进入需要减速的区域
                vehicle.acceleration = self._get_stopping_acceleration_for_light(vehicle)
            else:
                # 离路口还远，自由行驶
                vehicle.acceleration = self.idm_acceleration(vehicle, None)

            if vehicle.speed < 0.1 and dist_to_light < 2 * self.params.s0:
                 vehicle.waiting_time += self.time_step

        else:
            # 绿灯逻辑
            vehicle.acceleration = self.idm_acceleration(vehicle, front_vehicle)

        # 更新速度和位置
        vehicle.speed += vehicle.acceleration * self.time_step
        vehicle.speed = max(0, vehicle.speed)
        
        # 检查车辆是否会与前车重叠
        if front_vehicle:
            max_dist = front_vehicle.distance - self.params.length - self.params.s0
            vehicle.distance = min(vehicle.distance + vehicle.speed * self.time_step, max_dist)
        else:
            vehicle.distance += vehicle.speed * self.time_step

        # 更新是否通过路口状态
        if not vehicle.has_passed and vehicle.distance >= self.intersection_pos:
            vehicle.has_passed = True
            print(f"*** 车辆 {vehicle.id} 在时间 {self.total_time_simulated:.1f}s 通过路口！ ***")

    def find_front_vehicle(self, current_vehicle_index: int) -> Optional[VehicleState]:
        """找到当前车辆正前方的车辆"""
        if current_vehicle_index + 1 < len(self.vehicles):
            return self.vehicles[current_vehicle_index + 1]
        return None

    def simulate_step(self):
        """模拟一个时间步"""
        self.total_time_simulated += self.time_step
        if self.red_light_remaining > 0:
            self.red_light_remaining = max(0, self.red_light_remaining - self.time_step)

        # 从最前面的车开始更新，避免后车“跳”到前车位置
        for i in range(len(self.vehicles) - 1, -1, -1):
            vehicle = self.vehicles[i]
            front_vehicle = self.find_front_vehicle(i)
            self.update_vehicle_state(vehicle, front_vehicle)

        # 记录历史
        self._log_step()

    def _log_step(self):
        """记录当前时间步的状态"""
        step_info = {
            'time': self.total_time_simulated,
            'red_light_remaining': self.red_light_remaining,
            'vehicles': []
        }
        print(f"\n--- 时间: {self.total_time_simulated:.1f}s | 红灯剩余: {self.red_light_remaining:.1f}s ---")
        print(f"{'ID':<5}{'位置 (m)':<12}{'速度 (m/s)':<12}{'加速度 (m/s²)':<15}{'等待时间 (s)':<15}{'是否通过':<10}")
        
        for vehicle in self.vehicles:
            state_str = (f"{vehicle.id:<5}"
                         f"{vehicle.distance:<12.2f}"
                         f"{vehicle.speed:<12.2f}"
                         f"{vehicle.acceleration:<15.2f}"
                         f"{vehicle.waiting_time:<15.2f}"
                         f"{str(vehicle.has_passed):<10}")
            print(state_str)
            step_info['vehicles'].append(vars(vehicle).copy())
        self.simulation_history.append(step_info)


    def run_simulation(self, max_duration: int = 100):
        """运行完整模拟"""
        print("="*20 + " 开始交通模拟 " + "="*20)
        while self.total_time_simulated < max_duration:
            self.simulate_step()
            # 如果所有车都已远离路口，可以提前结束
            if all(v.distance > self.intersection_pos + 200 for v in self.vehicles):
                print("\n所有车辆已远离路口，模拟结束。")
                break
        print("\n" + "="*20 + " 模拟结束 " + "="*20)


# 测试用例
if __name__ == "__main__":
    # 将输出重定向到文件，避免控制台刷屏
    with open('output.log', 'w', encoding='utf-8') as f:
        sys.stdout = f
        params = VehicleParams()
        simulator = TrafficSimulator(params, time_step=0.1, intersection_pos=1000.0)
        
        # 测试用例：多车排队，包含近车和远车
        simulator.add_vehicles([
            {'id': 1, 'distance': 950, 'speed': 10},
            {'id': 2, 'distance': 920, 'speed': 12},
            {'id': 3, 'distance': 850, 'speed': 15},
            {'id': 4, 'distance': 700, 'speed': 20},
        ])
        
        simulator.set_red_light(20)
        simulator.run_simulation(max_duration=100)

    # 恢复标准输出并打印最终结果
    sys.stdout = sys.__stdout__
    print("模拟完成，详细日志已写入 output.log 文件。")


