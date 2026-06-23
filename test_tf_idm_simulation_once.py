import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
from matplotlib.patches import Rectangle, Circle
import logging

# 日志配置
logging.basicConfig(
    filename='sim_log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
logging.getLogger().addHandler(console)

# 屏蔽matplotlib日志
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
plt.rcParams["axes.unicode_minus"] = False

# 导入仿真函数和常量
from tf_idm_simulation import tf_idm_simulation, get_idm_params
from modelsCollect9 import (
    LANE_POS_MAP, DEFAULT_DT, MIN_GAP, OFFSET_DISTANCE,
    get_car_pos_speed_cols, make_dir_safe
)

# ===================== 全局配置 =====================
GIF_SAVE_PATH = "./simulation_animation.gif"
PLOT_WIDTH = 15
PLOT_HEIGHT = 5
VEHICLE_WIDTH = 1
VEHICLE_HEIGHT = 1
FPS = 10
DPI = 300

NUM_TYPES = 6
DT = 0.1
MAX_STEPS = int(120 / DT)
NUM_VEH = 20
GO_FLAG = 0

# ===================== 数据筛选函数 =====================
def filter_sample(df):
    df['intersection_pos'] = df['lane'].map(LANE_POS_MAP)

    def count_queued(row):
        pos_cols, _ = get_car_pos_speed_cols(row.index)
        main_pos = row['main_car_position']
        count = 0
        for col in pos_cols:
            pos = row[col]
            if pos != -1 and not pd.isna(pos) and pos > main_pos:
                count += 1
        return count

    df['queued_vehicles'] = df.apply(count_queued, axis=1)
    df['redLightRemainingTime_sec'] = df['redLightRemainingTime'] / 30.0

    filter_cond = (
        (df['lane'] == 6)
        & (df['redLightRemainingTime_sec'] > 5.0)
        & (df['redLightRemainingTime_sec'] < 10.0)
        & (df['queued_vehicles'] > 2)
    )

    filtered_df = df[filter_cond].reset_index(drop=True)
    if len(filtered_df) == 0:
        raise ValueError("No qualified samples found: Lane 6, queue>2, red light>5s")

    sample = filtered_df.iloc[0:1].reset_index(drop=True)
    lane_id = sample['lane'].iloc[0]
    logging.info(f"Qualified sample found (Lane {lane_id}):")
    logging.info(f"  - Remaining red light: {sample['redLightRemainingTime_sec'].iloc[0]:.1f}s")
    logging.info(f"  - Queue vehicles: {sample['queued_vehicles'].iloc[0]}")
    logging.info(f"  - Ego vehicle pos: {sample['main_car_position'].iloc[0]:.1f}m")
    logging.info(f"  - Intersection pos: {sample['intersection_pos'].iloc[0]:.1f}m")
    return sample

# ===================== 仿真追踪函数（纯TF张量计算） =====================
def simulate_single_sample(sample):
    feature_cols = [
        c for c in sample.columns
        if ('car_position_' in c or 'car_speed_' in c or 'redLight' in c)
    ]
    base_cols = [
        'lane', 'intersection_pos', 'main_car_position',
        'main_car_speed'
    ]
    raw_cols_set = set(feature_cols)
    raw_cols_set.update(base_cols)
    raw_cols = sorted(list(raw_cols_set))

    X = sample[feature_cols].values.astype(np.float32)
    raw_data = sample[raw_cols].values.astype(np.float32)

    nn_output = np.ones((1, (NUM_TYPES + 1) * 6), dtype=np.float32) * 0.5

    pos_cols, speed_cols = get_car_pos_speed_cols(raw_cols)
    pos_idx = tf.constant([raw_cols.index(c) for c in pos_cols], dtype=tf.int32)
    speed_idx = tf.constant([raw_cols.index(c) for c in speed_cols], dtype=tf.int32)
    idx_main = tf.constant(raw_cols.index("main_car_position"), dtype=tf.int32)
    idx_inter = tf.constant(raw_cols.index("intersection_pos"), dtype=tf.int32)
    idx_red = tf.constant(raw_cols.index("redLightRemainingTime"), dtype=tf.int32)

    base_bounds = [
        (30/3.6, 75/3.6),
        (0.1, 2.0),
        (0.2, 1.0),
        (1.0, 6.0),
        (1.0, 9.0),
        (0.01, 1.0)
    ]
    param_bounds = np.stack([np.array(base_bounds, dtype=np.float32)] * NUM_TYPES, axis=0)

    sim_records = {}
    car_cross_time = {cid: None for cid in range(NUM_VEH)}

    car_positions = tf.gather(raw_data, pos_idx, axis=1)[0].numpy()
    car_speeds = tf.gather(raw_data, speed_idx, axis=1)[0].numpy()
    main_pos = raw_data[0, idx_main.numpy()]
    intersection_pos = raw_data[0, idx_inter.numpy()]

    main_car_id = np.argmin(np.abs(car_positions - main_pos))

    scaled_params0 = tf.reshape(nn_output, (1, NUM_TYPES + 1, 6))
    scaled_params = scaled_params0[:, :-1, :]
    scene_offset_full = scaled_params0[:, -1, :]

    bounds_tensor = tf.convert_to_tensor(param_bounds, dtype=tf.float32)
    low_2d = bounds_tensor[..., 0]
    high_2d = bounds_tensor[..., 1]
    target_shape = tf.shape(scaled_params)
    low = tf.broadcast_to(low_2d, target_shape)
    high = tf.broadcast_to(high_2d, target_shape)
    range_val = high - low + 1e-8
    real_params = low + scaled_params * range_val

    redlighttime_offset = scene_offset_full[:, 0]
    redlightpos2vanishpos_offset = scene_offset_full[:, 1]
    vehpos_offset = scene_offset_full[:, 2]
    redlightpos_offset = scene_offset_full[:, 3]
    vanishtime_offset = scene_offset_full[:, 4]
    distgap_offset = scene_offset_full[:, 5]

    zero_1d = tf.zeros_like(redlighttime_offset)
    redlighttime_offset = tf.cond(tf.equal(GO_FLAG, 1),
                                  lambda: (-1.0 + redlighttime_offset * 2.0) * 2.0,
                                  lambda: zero_1d)
    redlightpos2vanishpos_offset = tf.cond(tf.equal(GO_FLAG, 1),
                                           lambda: redlightpos2vanishpos_offset * 8.0,
                                           lambda: tf.zeros_like(redlightpos2vanishpos_offset))
    vehpos_offset = tf.cond(tf.equal(GO_FLAG, 1),
                            lambda: (-1.0 + vehpos_offset * 2.0) * 2.0,
                            lambda: tf.zeros_like(vehpos_offset))
    redlightpos_offset = tf.cond(tf.equal(GO_FLAG, 1),
                                 lambda: redlightpos_offset * 2.0,
                                 lambda: tf.zeros_like(redlightpos_offset))
    vanishtime_offset = tf.cond(tf.equal(GO_FLAG, 1),
                                lambda: (-1.0 + vanishtime_offset * 2.0) * 2.0,
                                lambda: tf.zeros_like(vanishtime_offset))
    distgap_offset = tf.cond(tf.equal(GO_FLAG, 1),
                             lambda: (-1.0 + distgap_offset * 2.0) * 2.0,
                             lambda: tf.zeros_like(distgap_offset))

    car_pos = tf.gather(raw_data, pos_idx, axis=1)[0].numpy()
    car_speed = tf.gather(raw_data, speed_idx, axis=1)[0].numpy()
    red_dur = raw_data[0, idx_red.numpy()] / 30.0

    mask_invalid = (car_pos == -1.0)
    rand_neg_pos = np.random.uniform(-5000.0, -100.0, size=car_pos.shape)
    car_pos = np.where(mask_invalid, rand_neg_pos, car_pos)
    vanished = mask_invalid.copy()

    car_pos += vehpos_offset.numpy()[0]
    intersection_pos += redlightpos_offset.numpy()[0]
    red_timer = red_dur + redlighttime_offset.numpy()[0]

    vtypes = np.tile(np.arange(NUM_VEH)[None, :] % NUM_TYPES, [1, 1])
    v0, T, s0, a_max, b, delta, length, rtime = get_idm_params(real_params, vtypes)
    v0 = v0[0]
    T = T[0]
    s0 = s0[0]
    a_max = a_max[0]
    b = b[0]
    delta = delta[0]
    length = length[0]
    rtime = rtime[0]

    # 转为TF张量全程运算
    pos = tf.constant(car_pos, dtype=tf.float32)
    vel = tf.constant(car_speed, dtype=tf.float32)
    current_red_timer = tf.constant(red_timer, dtype=tf.float32)
    vanished = tf.constant(vanished, dtype=tf.bool)
    intersection_pos_tf = tf.constant(intersection_pos, tf.float32)
    stop_gap = tf.constant(3.0, tf.float32)

    for step in range(MAX_STEPS):
        current_time = tf.constant(step * DT, tf.float32)
       

        if step not in sim_records:
            pos_np = pos.numpy()
            vel_np = vel.numpy()
            van_np = vanished.numpy()
            cars_info = {}
            for car_id in range(NUM_VEH):
                cars_info[car_id] = {
                    'pos': pos_np[car_id],
                    'speed': vel_np[car_id],
                    'vanished': van_np[car_id],
                    'car_cross_time': car_cross_time[car_id]
                }
            sim_records[step] = {
                "red_timer": round(current_red_timer.numpy(), 1),
                "cars": cars_info
            }

        # 记录过路口时间
        pos_np = pos.numpy()
        inter_np = intersection_pos_tf.numpy()
        for cid in range(NUM_VEH):
            if car_cross_time[cid] is None and pos_np[cid] > inter_np:
                car_cross_time[cid] = round(current_time.numpy(), 2)

        if car_cross_time[main_car_id] is not None:
            logging.info(f"Ego vehicle {main_car_id} passed intersection, simulation stopped at {step}step")
            break

        # 排序求gap
        idx_sort = tf.argsort(pos, direction="DESCENDING")
        pos_sorted = tf.gather(pos, idx_sort)
        vel_sorted = tf.gather(vel, idx_sort)
        inv_idx = tf.argsort(idx_sort)
        gap_raw = pos_sorted[:-1] - pos_sorted[1:] - length[:-1]
        gap_raw = tf.maximum(gap_raw, 0.1)
        gap_pad = tf.pad(gap_raw, paddings=[[1, 0]], constant_values=1000.0)
        gap = tf.gather(gap_pad, inv_idx)

        # IDM加速度纯TF计算
        v_opt = vel / (v0 + 1e-6)
        sqrt_vel = tf.sqrt(tf.maximum(vel, 1e-6))
        sqrt_ab = tf.sqrt(tf.maximum(a_max * b, 1e-6))
        s_opt = s0 + vel * T + vel * sqrt_vel / (2 * sqrt_ab)
        acc_idm = a_max * (1.0 - tf.pow(v_opt, delta) - tf.square(s_opt / (gap + 1e-6)))

        # 红灯制动逻辑张量实现
        dist_to_red = intersection_pos_tf - pos
        red_hold = (current_red_timer > 0) & (dist_to_red < 10.0)
        red_acc = tf.zeros_like(vel)

        # 向量化计算动态刹车减速度
        d_remain = dist_to_red - stop_gap
        mask_over = d_remain <= 0.1
        mask_safe = d_remain > 0.1
        d_safe = tf.maximum(d_remain, 0.01)
        dynamic_brake = -(tf.square(vel)) / (2 * d_safe)
        dynamic_brake = tf.clip_by_value(dynamic_brake, -b * 2.0, -0.1)
        red_acc = tf.where(mask_over, tf.constant(-9.0, tf.float32), dynamic_brake)
        red_acc = tf.where(~red_hold, tf.zeros_like(red_acc), red_acc)

        # 融合加速度
        acc = tf.where(red_hold, tf.minimum(acc_idm, red_acc), acc_idm)
        acc = tf.clip_by_value(acc, -9.0, a_max)

        # 更新速度、位置（纯TF，无循环赋值acc[car_id]=0）
        mask_pos_valid = pos > 0.0
        vel_update = vel + acc * DT
        vel_new = tf.clip_by_value(vel_update, 0.0, 50.0)
        pos_new = pos + vel_new * DT
        # 位置<=0车辆保持不动，速度置0
        vel_new = tf.where(mask_pos_valid, vel_new, tf.zeros_like(vel_new))
        pos_new = tf.where(mask_pos_valid, pos_new, pos)

        pos = pos_new
        vel = vel_new
        current_red_timer = tf.maximum(current_red_timer - DT, 0.0)

        # 日志打印仅临时转numpy
        acc_np = acc.numpy()
        pos_np = pos.numpy()
        vel_np = vel.numpy()
        logging.info(f"\n Step time={current_time.numpy():.2f}s")
        for vid in range(NUM_VEH):
            logging.info(f"vid{vid}--- pos={pos_np[vid]:.2f}, v={vel_np[vid]:.2f}, a={acc_np[vid]:.3f}")

    logging.info(f"Simulation finished, time range: 0~{max(sim_records.keys())}s")
    return sim_records, main_car_id, intersection_pos_tf.numpy(), car_cross_time

# ===================== 可视化函数 =====================
def create_simulation_gif(sim_records, main_car_id, intersection_pos):
    if os.path.exists(GIF_SAVE_PATH):
        os.remove(GIF_SAVE_PATH)
    frames = sorted(sim_records.keys())
    image_buffer = []
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT + 1.5), dpi=DPI/2)
    ax.set_xlim(intersection_pos - 100, intersection_pos + 100)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Single Lane 6")
    ax.set_title("IDM Traffic Simulation Lane 6")
    ax.axhline(y=1.5, color="gray", lw=2, alpha=0.6)
    ax.axvline(x=intersection_pos, color='red', linestyle='--', label='Stop Line')
    ax.legend(loc='upper right')

    vehicle_patches = {}
    bottom_info_texts = {}
    for car_id in range(NUM_VEH):
        color = 'red' if car_id == main_car_id else 'blue'
        patch = Rectangle((0, 0), VEHICLE_WIDTH, VEHICLE_HEIGHT, facecolor=color, alpha=0.7)
        vehicle_patches[car_id] = ax.add_patch(patch)
        t2 = ax.text(0, -2.0, "", fontsize=7, color="black", ha="center")
        bottom_info_texts[car_id] = t2

    time_text = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    red_timer_text = ax.text(0.5, 0.85, "", transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    for frame_t in frames:
        frame_data = sim_records[frame_t]
        red_t = frame_data["red_timer"]
        cars_data = frame_data["cars"]
        time_text.set_text(f"step: {frame_t}")
        if red_t > 0:
            red_timer_text.set_text(f"Red Light Left: {red_t:0.2f} s")
        else:
            red_timer_text.set_text("Green Light, Go")
        for car_id in range(NUM_VEH):
            car_data = cars_data[car_id]
            patch = vehicle_patches[car_id]
           
            t2 = bottom_info_texts[car_id]
            if car_data['pos'] < (intersection_pos - 100):
                patch.set_visible(False)
               
                t2.set_visible(False)
            else:
                y_pos = 1.0
                patch.set_xy((car_data['pos'] - VEHICLE_WIDTH/2, y_pos))
                patch.set_visible(True)
                info_str = f"ID{car_id}\n{car_data['pos']:.1f}m\n{car_data['speed']:.1f}m/s"
                t2.set_text(info_str)
                t2.set_x(car_data['pos'])
                t2.set_visible(True)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_buffer.append(img)
    plt.close(fig)
    imageio.mimsave(GIF_SAVE_PATH, image_buffer, fps=FPS)
    logging.info(f"GIF saved via imageio: {GIF_SAVE_PATH}")

# ===================== 主函数 =====================
def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    csv_path = "trainsamples_lane_5_6_7.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    df.rename(columns={'car_position': 'main_car_position', 'car_speed': 'main_car_speed'}, inplace=True)
    logging.info(f"Data loaded, total samples: {len(df)}")
    sample = filter_sample(df)
    sim_records, main_car_id, intersection_pos, car_cross_time = simulate_single_sample(sample)
    create_simulation_gif(sim_records, main_car_id, intersection_pos)
    logging.info("==== Simulation Summary ====")
    logging.info(f"Ego Vehicle ID: {main_car_id}")
    logging.info(f"Intersection Position: {intersection_pos:.1f} m")
    last_t = max(sim_records.keys())
    last_frame = sim_records[last_t]
    main_car_final = last_frame["cars"][main_car_id]
    logging.info(f"Ego vehicle passed intersection: {main_car_final['pos'] > intersection_pos + 5.0}")
    logging.info(f"Total simulation time: {last_t} s")
    logging.info("\n==== Vehicle Cross Intersection Time Record ====")
    for vid, cross_t in car_cross_time.items():
        if cross_t is not None:
            logging.info(f"Vehicle {vid} cross time: {cross_t} s")
        else:
            logging.info(f"Vehicle {vid}: Not pass intersection in simulation")
    logging.info("Simulation finished successfully")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    main()