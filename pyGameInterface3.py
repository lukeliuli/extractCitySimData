import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import sys
import os
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt

@dataclass
class VehicleParams:
    """IDM模型参数配置"""
    v0: float
    T: float
    s0: float
    a: float
    b: float
    delta: float
    length: float
    rtime: float

@dataclass
class VehicleState:
    id: int
    distance: float
    speed: float
    acceleration: float = 0.0
    has_passed: bool = False
    waiting_time: float = 0.0
    params: VehicleParams = None  # 每辆车独立的参数
    # Removed VehicleClassifier related code and functionality


class TrafficSimulator:
    def __init__(self, default_params: VehicleParams, time_step: float = 0.1, intersection_pos: float = 1000.0, num_types: int = 4):
        self.default_params = default_params  # 默认参数
        self.time_step = time_step
        self.vehicles: List[VehicleState] = []
        self.red_light_remaining: float = 0
        self.max_vehicles: int = 20
        self.simulation_history: List[Dict] = []
        self.intersection_pos = intersection_pos
        self.total_time_simulated = 0.0
        self.visualization = 0
        self.num_types = num_types
        self.vehicle_params_dict: Dict[int, VehicleParams] = {}  # {车辆id: VehicleParams}

    def add_vehicles(self, vehicle_data: List[Dict]):
        for data in vehicle_data:
            if len(self.vehicles) >= self.max_vehicles:
                print(f"警告：车道已满（最多{self.max_vehicles}辆），无法添加更多车辆。")
                break
            if data['distance'] >= self.intersection_pos:
                print(f"警告：车辆 {data['id']} 的初始位置 {data['distance']}m 已在或超过路口 {self.intersection_pos}m，不予添加。")
                continue
            # 默认参数
            params = self.vehicle_params_dict.get(data['id'], self.default_params)
            self.vehicles.append(VehicleState(
                id=data['id'],
                distance=data['distance'],
                speed=data['speed'],
                params=params
            ))
        self.vehicles.sort(key=lambda v: v.distance)

    def set_red_light(self, remaining_time: float):
        self.red_light_remaining = remaining_time

    def set_vehicle_params(self, vid: int, params: VehicleParams):
        """接口：为指定车辆id设置参数"""
        self.vehicle_params_dict[vid] = params
        for v in self.vehicles:
            if v.id == vid:
                v.params = params

    def batch_set_vehicle_params(self, params_dict: Dict[int, VehicleParams]):
        """接口：批量设置车辆参数"""
        self.vehicle_params_dict.update(params_dict)
        for v in self.vehicles:
            if v.id in params_dict:
                v.params = params_dict[v.id]

    def idm_acceleration(self, vehicle: VehicleState, front_vehicle: Optional[VehicleState]) -> float:
        params = vehicle.params if vehicle.params else self.default_params
        free_acc = params.a * (1 - (vehicle.speed / params.v0) ** params.delta)
        if front_vehicle is None:
            return free_acc
        gap = front_vehicle.distance - vehicle.distance - params.length
        if gap < params.s0:
            return -9.0
        headway = gap / max(vehicle.speed, 0.01)
        if headway > 3.0:
            return free_acc
        desired_gap = params.s0 + max(0, vehicle.speed * params.T + (vehicle.speed * (vehicle.speed - front_vehicle.speed)) / (2 * math.sqrt(params.a * params.b)))
        if desired_gap > gap:
            return -params.b * (desired_gap / max(gap, 0.1))
        follow_acc = params.a * (1 - (vehicle.speed / params.v0) ** params.delta - (desired_gap / gap) ** 2)
        return follow_acc

    def _get_stopping_acceleration_for_light(self, vehicle: VehicleState) -> float:
        params = vehicle.params if vehicle.params else self.default_params
        dist_to_light = self.intersection_pos - vehicle.distance - params.length
        if dist_to_light <= params.s0:
            return -9.0
        required_acc = -(vehicle.speed ** 2) / (2 * dist_to_light)
        return max(required_acc, -params.b)

    def update_vehicle_state(self, vehicle: VehicleState, front_vehicle: Optional[VehicleState]):
        accPre = vehicle.acceleration
        params = vehicle.params if vehicle.params else self.default_params
        if self.red_light_remaining > 0:
            dist_to_light = self.intersection_pos - vehicle.distance
            stopping_dist_needed = vehicle.speed**2 / (2 * params.b)
            if front_vehicle:
                vehicle.acceleration = self.idm_acceleration(vehicle, front_vehicle)
            elif dist_to_light <= stopping_dist_needed + params.s0:
                vehicle.acceleration = self._get_stopping_acceleration_for_light(vehicle)
            else:
                vehicle.acceleration = self.idm_acceleration(vehicle, None)
            if vehicle.speed < 1:
                vehicle.waiting_time += self.time_step
        else:
            vehicle.acceleration = self.idm_acceleration(vehicle, front_vehicle)
        vehicle.acceleration = vehicle.acceleration + (1 - math.exp(-params.rtime / self.time_step)) * (accPre - vehicle.acceleration)
        vehicle.acceleration = max(-9.0, min(vehicle.acceleration, params.a))
        vehicle.speed += vehicle.acceleration * self.time_step
        vehicle.speed = max(0, vehicle.speed)
        if front_vehicle:
            max_dist = front_vehicle.distance - params.length - params.s0
            vehicle.distance = min(vehicle.distance + vehicle.speed * self.time_step, max_dist)
        else:
            vehicle.distance += vehicle.speed * self.time_step
        if not vehicle.has_passed and vehicle.distance >= self.intersection_pos:
            vehicle.has_passed = True

    def find_front_vehicle(self, current_vehicle_index: int) -> Optional[VehicleState]:
        if current_vehicle_index + 1 < len(self.vehicles):
            return self.vehicles[current_vehicle_index + 1]
        return None

    def simulate_step(self):
        self.total_time_simulated += self.time_step
        if self.red_light_remaining > 0:
            self.red_light_remaining = max(0, self.red_light_remaining - self.time_step)
       
        self.vehicles.sort(key=lambda v: self.intersection_pos - v.distance, reverse=True)
        
        for i in range(len(self.vehicles) - 1, -1, -1):
            vehicle = self.vehicles[i]
            front_vehicle = self.find_front_vehicle(i)
            self.update_vehicle_state(vehicle, front_vehicle)
        self._log_step()
        if self.visualization == 0:
            return
        output_dir = "simulation_frames"
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.set_xlim(0, self.intersection_pos + 100)
        ax.set_ylim(-1, 1)
        ax.set_title(f"时间: {self.total_time_simulated:.1f}s | 红灯剩余: {self.red_light_remaining:.1f}s", fontproperties="SimHei")
        ax.set_xlabel("位置 (m)", fontproperties="SimHei")
        ax.get_yaxis().set_visible(False)
        ax.axhline(0, color='gray', linestyle='-', linewidth=5)
        ax.axvline(x=self.intersection_pos, color='darkred', linestyle='--', linewidth=2, label='路口')
        light_color = 'red' if self.red_light_remaining > 0 else 'green'
        ax.plot(self.intersection_pos + 5, 0.5, 'o', markersize=20, color=light_color, markeredgecolor='black')
        for vehicle in self.vehicles:
            car_rear = vehicle.distance - (vehicle.params.length if vehicle.params else self.default_params.length)
            rect = plt.Rectangle((car_rear, -0.2), (vehicle.params.length if vehicle.params else self.default_params.length), 0.4, color='royalblue', ec='black')
            ax.add_patch(rect)
            label = f"ID:{vehicle.id}\n{vehicle.speed * 3.6:.1f} km/h"
            ax.text(vehicle.distance - (vehicle.params.length if vehicle.params else self.default_params.length) / 2, 0.3 + vehicle.id * 0.15, label,
                    ha='center', va='bottom', fontsize=7, fontproperties="SimHei")
        plt.legend(prop={"family": "SimHei"})
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"frame_{self._frame_count:05d}.jpg"), dpi=100)
        plt.close(fig)
        self._frame_count += 1

    def _log_step(self):
        step_info = {
            'time': self.total_time_simulated,
            'red_light_remaining': self.red_light_remaining,
            'vehicles': []
        }
        for vehicle in self.vehicles:
            vdict = vars(vehicle).copy()
            # VehicleParams不能直接序列化，转为dict
            if vdict['params'] is not None:
                vdict['params'] = vars(vdict['params']).copy()
            step_info['vehicles'].append(vdict)
        self.simulation_history.append(step_info)

    def run_simulation(self, max_duration: int = 100):
        print("\n" + "=" * 20 + " 开始交通模拟 " + "=" * 20)
        while self.total_time_simulated < max_duration:
            self.simulate_step()
            if all(v.distance > self.intersection_pos + 5 for v in self.vehicles):
                break
        print("\n" + "=" * 20 + " 模拟结束 " + "=" * 20)
        df = pd.DataFrame([{
            'time': step['time'],
            'red_light_remaining': step['red_light_remaining'],
            **{k: v for k, v in vehicle.items() if k != 'params'},
            **{f'param_{pk}': pv for pk, pv in (vehicle['params'] or {}).items()}
        } for step in self.simulation_history for vehicle in step['vehicles']])
        return df

# 示例参数
DEFAULT_PARAM = VehicleParams(v0=13, T=1.0, s0=2.0, a=2.0, b=2.0, delta=4.0, length=5.0, rtime=0.01)

if __name__ == "__main__":
    with open('output_pyGameInterfacd3sim.log', 'w', encoding='utf-8') as f:
        sys.stdout = f
        simulator = TrafficSimulator(DEFAULT_PARAM, time_step=0.1, intersection_pos=1000.0, num_types=4)
        simulator.add_vehicles([
            {'id': 1, 'distance': 950, 'speed': 0},
            {'id': 2, 'distance': 920, 'speed': 0},
            {'id': 3, 'distance': 850, 'speed': 0},
            {'id': 4, 'distance': 700, 'speed': 10},
            {'id': 5, 'distance': 650, 'speed': 15},
        ])
        # 单独为车辆1、2设置不同参数
        simulator.set_vehicle_params(1, VehicleParams(v0=15, T=1.1, s0=2.1, a=2.1, b=2.1, delta=4.0, length=5.0, rtime=0.02))
        simulator.set_vehicle_params(2, VehicleParams(v0=12, T=1.3, s0=2.3, a=1.7, b=2.3, delta=4.0, length=5.0, rtime=0.03))
        simulator.set_vehicle_params(5, VehicleParams(v0=10, T=1.8, s0=3.0, a=1.2, b=2.8, delta=4.0, length=5.0, rtime=0.01))
        simulator.set_red_light(20)
        df = simulator.run_simulation(max_duration=100)
        df.to_csv("simulation_history.csv", index=False, encoding='utf-8-sig', float_format='%.3f')
    sys.stdout = sys.__stdout__
    print("模拟完成，详细日志已写入 output.log 文件。")

