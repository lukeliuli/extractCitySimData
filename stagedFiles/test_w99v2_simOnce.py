import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Rectangle
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'                 # 0=全部 1=关INFO 2=关INFO+WARNING 3=全关(含ERROR)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # 规避 protobuf GetPrototy
# 日志配置
logging.basicConfig(
    filename='w99_sim_log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
logging.getLogger().addHandler(console)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
plt.rcParams["axes.unicode_minus"] = False

# 导入被测试的参考仿真函数（用于交叉校验）
from tf_w99v2_simulation import tf_w99v2_simulation

# ===================== 全局配置 =====================
GIF_SAVE_PATH = "./w99_simulation_animation.gif"
PLOT_WIDTH = 15
PLOT_HEIGHT = 5
VEHICLE_WIDTH = 1
VEHICLE_HEIGHT = 1
FPS = 10
DPI = 300

NUM_TYPES = 4          # W99 车型数（0/1/2/3）
NUM_VEH = 20
DT = 0.1
MAX_STEPS = int(120 / DT)
GO_FLAG = 0            # 关闭 offset（=0 时所有 offset 置零）

# 与 model9w99RTlost.py 最后一份 BASE_BOUND_VEHICLE 一致（10 个 W99 参数）
BASE_BOUND_VEHICLE = [
    (1.0, 2.0),   # 0: CC0 停车间距
    (0.5, 2.0),   # 1: CC1 车头时距
    (0.1, 5),    # 2: CC2 SDX 安全距离附加值
    (0.5, 2.0),   # 3: CC3 following D 值速度差权重
    (-3.5, 0.0),  # 4: CC4 CLDV 负相对速度阈值
    (0.0, 3.5),   # 5: CC5 OPDV 正相对速度阈值
    (1, 3.0),     # 6: CC6 following P 值间距差权重
    (2, 5),       # 7: CC7 加速度波动
    (1.0, 3.0),   # 8: CC8 启动加速度
    (0.5, 1.5)    # 9: CC9 高速加速度
]
LANE_POS_MAP = {5: 53.05, 6: 53.13, 7: 53.30}

def get_car_pos_speed_cols(col_list):
    pos_cols = [c for c in col_list if c.startswith('car_position_')]
    speed_cols = [c for c in col_list if c.startswith('car_speed_')]
    return pos_cols, speed_cols

# ===================== 数据筛选函数 =====================
def filter_sample(df):
    df['intersection_pos'] = df['lane'].map(LANE_POS_MAP)
    pos_cols, _ = get_car_pos_speed_cols(df.columns)

    def count_queued(row):
        main_pos = row['main_car_position']
        return sum(1 for c in pos_cols
                   if row[c] != -1 and not pd.isna(row[c]) and row[c] > main_pos)

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
        raise ValueError("No qualified samples: Lane 6, queue>2, red light 5~10s")

    sample = filtered_df.iloc[0:1].reset_index(drop=True)
    logging.info(f"Qualified sample (Lane {sample['lane'].iloc[0]}):")
    logging.info(f"  - Remaining red: {sample['redLightRemainingTime_sec'].iloc[0]:.1f}s")
    logging.info(f"  - Queue vehicles: {sample['queued_vehicles'].iloc[0]}")
    logging.info(f"  - Ego pos: {sample['main_car_position'].iloc[0]:.1f}m")
    logging.info(f"  - Intersection pos: {sample['intersection_pos'].iloc[0]:.1f}m")
    return sample

# ===================== 参考实现交叉校验 =====================
def _run_reference(nn_output, raw_data, pos_idx, speed_idx, idx_main, idx_inter, idx_red):
    param_bounds = np.stack(
        [np.array(BASE_BOUND_VEHICLE, dtype=np.float32)] * NUM_TYPES, axis=0
    )  # (4, 10, 2)
    pred = tf_w99v2_simulation(
        tf.constant(nn_output, tf.float32),
        tf.constant(raw_data, tf.float32),
        param_bounds, NUM_TYPES,
        tf.constant(pos_idx, tf.int32),
        tf.constant(speed_idx, tf.int32),
        tf.constant(idx_main, tf.int32),
        tf.constant(idx_inter, tf.int32),
        tf.constant(idx_red, tf.int32),
        DT, GO_FLAG
    )
    return float(np.reshape(pred.numpy(), [-1])[0])

# ===================== 单样本逐步仿真（逐车记录中间细节） =====================
def simulate_single_sample(sample):
    sample_vanishTime = float(sample['time_to_vanish'].iloc[0]) / 30.0

    # 1. 构造 raw_cols（与 model9w99RTlost.py 完全一致）
    feature_cols = [f"car_position_{i}" for i in range(NUM_VEH)] \
                 + [f"car_speed_{i}" for i in range(NUM_VEH)]
    feature_cols += ['intersection_pos', 'lane', 'main_car_position',
                     'main_car_speed', 'queued_vehicles', 'redLightRemainingTime']
    raw_cols = feature_cols

    raw_data = sample[raw_cols].values.astype(np.float32)   # (1, n_raw_cols)

    pos_cols, speed_cols = get_car_pos_speed_cols(raw_cols)
    pos_idx = [raw_cols.index(c) for c in pos_cols]
    speed_idx = [raw_cols.index(c) for c in speed_cols]
    idx_main = raw_cols.index('main_car_position')
    idx_inter = raw_cols.index('intersection_pos')
    idx_red = raw_cols.index('redLightRemainingTime')

    # 2. mock 网络输出（可替换为真实模型输出）
    output_dim = (NUM_TYPES + 1) * 10
    nn_output = np.full((1, output_dim), 0.5, dtype=np.float32)

    # 3. 参数解码（与 tf_w99v2_simulation 一致）
    scaled_params0 = nn_output.reshape(1, NUM_TYPES + 1, 10)
    scaled_params = scaled_params0[:, :-1, :]                 # (1, 4, 10)
    bounds = np.array(BASE_BOUND_VEHICLE, dtype=np.float32)   # (10, 2)
    low, high = bounds[:, 0], bounds[:, 1]
    real_params = low + scaled_params * (high - low)          # (1, 4, 10)
    real_params_2d = real_params[0]                           # (4, 10)

    # 4. 数据提取（单样本）
    car_positions = raw_data[0, pos_idx].astype(np.float32)   # (20,)
    car_speeds = raw_data[0, speed_idx].astype(np.float32)
    main_pos = float(raw_data[0, idx_main])
    inter_pos = float(raw_data[0, idx_inter])
    red_dur_sec = float(raw_data[0, idx_red] / 30.0)

    # 5. 无效车(-1)处理
    mask_invalid = (car_positions == -1.0)
    rand_neg = np.random.uniform(-5000.0, -100.0, size=car_positions.shape)
    car_positions = np.where(mask_invalid, rand_neg, car_positions)
    vanished = mask_invalid.copy()

    # 6. 车型分配：按初始位置从大到小排名 -> 最前=车型0, 第二=车型1, 第三=车型2, 其后=车型3
    idx_sort_init = np.argsort(-car_positions)      # 降序：第一个=位置最大=最前车
    inv_init = np.argsort(idx_sort_init)
    rank_order = np.minimum(np.arange(NUM_VEH), NUM_TYPES - 1)  # [0,1,2,3,3,...]
    vtypes = rank_order[inv_init]                   # 对应原始车辆顺序

    cc0_s = real_params_2d[vtypes, 0].astype(np.float32)
    cc1_s = real_params_2d[vtypes, 1].astype(np.float32)
    cc2_s = real_params_2d[vtypes, 2].astype(np.float32)
    cc3_s = real_params_2d[vtypes, 3].astype(np.float32)
    cc4_s = real_params_2d[vtypes, 4].astype(np.float32)
    cc5_s = real_params_2d[vtypes, 5].astype(np.float32)
    cc6_s = real_params_2d[vtypes, 6].astype(np.float32)
    cc7_s = real_params_2d[vtypes, 7].astype(np.float32)
    cc8_s = real_params_2d[vtypes, 8].astype(np.float32)
    cc9_s = real_params_2d[vtypes, 9].astype(np.float32)
    
    # 7. 仿真状态初始化
    pos = car_positions.copy()
    vel = car_speeds.copy()
    time_counter = np.zeros_like(pos)
    vanished = mask_invalid.copy()
    red_timer = red_dur_sec

    sim_records = {}
    car_vanish_time = {cid: None for cid in range(NUM_VEH)}
    main_idx = int(np.argmin(np.abs(car_positions - main_pos)))

    MAX_BRAKE = -9.0
    v80 = 22.22

    for step in range(MAX_STEPS):
        # ---- 排序（从远到近 = 前车在前） ----
        idx_sort = np.argsort(-pos)
        pos_sorted = pos[idx_sort]
        vel_sorted = vel[idx_sort]
        inv_idx = np.argsort(idx_sort)

        # ---- gap / dv ----
        cc0_sorted = cc0_s[idx_sort]
        gap_raw = pos_sorted[:-1] - pos_sorted[1:] - cc0_sorted[:-1]
        gap_raw = np.maximum(gap_raw, 0.1)
        gap_pad = np.concatenate(([1000.0], gap_raw))
        gap = gap_pad[inv_idx]

        dv_raw = vel_sorted[:-1] - vel_sorted[1:]
        dv_pad = np.concatenate(([0.0], dv_raw))
        dv = dv_pad[inv_idx]
        dv_fb = dv

        # ---- 安全距离 / 自由流加速度 ----
        abx = cc0_s + cc1_s * vel
        sdx = cc0_s + cc1_s * vel + cc2_s
        sdx = np.maximum(sdx, 1e-3)

        v_ref = np.minimum(vel, v80)
        free_acc = cc8_s + (cc9_s - cc8_s) * (v_ref / v80)
        free_acc = np.maximum(free_acc, 0.0)

        CLDV = cc4_s
        OPDV = cc5_s

        # ---- 接近/减速 ----
        is_approaching = (dv_fb < CLDV) & (gap < sdx)
        closing = np.maximum(-dv, 0.0)
        denom = np.maximum(sdx - gap, 1e-3)
        decel_acc = -0.5 * closing * closing / denom
        decel_acc = np.clip(decel_acc, MAX_BRAKE, 0.0)

        # ---- 跟随 ----
        cc2_eff = np.maximum(cc2_s, 1e-3)
        cc7_mag = cc7_s
        cc6_mag = cc6_s
        is_following = (gap >= abx) & (gap <= sdx) & (dv > CLDV) & (dv < OPDV)
        follow_center = abx + cc2_eff * 0.5
        gap_err = (follow_center - gap) / follow_center
        opdv_safe = np.where(np.abs(OPDV) < 1e-6, 1e-6, OPDV)   # 防除零
        b_follow = cc3_s * dv_fb / opdv_safe - cc6_mag * gap_err
        b_follow = np.clip(b_follow, -cc7_mag, cc7_mag)

        # ---- 驶离/加速 ----
        is_departing = (dv_fb > OPDV) & (gap < sdx)
        accel_acc = b_follow

        # ---- 刹车 ----
        is_brake = gap < abx
        gap_err_brake = abx - gap
        abx_safe = np.maximum(abx, 1e-3)
        brake_acc = -cc6_mag * gap_err_brake / abx_safe

        # ---- 状态合成 ----
        acc_w99 = np.where(is_brake, brake_acc,
                  np.where(is_approaching, decel_acc,
                  np.where(is_departing, accel_acc,
                  np.where(is_following, b_follow, free_acc))))

        # ---- 红灯硬约束 ----
        dist_to_red = inter_pos - pos
        red_hold = (red_timer > 0) & (dist_to_red < 5.0)
        acc = np.where(red_hold, MAX_BRAKE, acc_w99)
        acc = np.clip(acc, MAX_BRAKE, 3.0)

        # ---- 记录每辆车中间状态 ----
        cars_info = {}
        for cid in range(NUM_VEH):
            if is_brake[cid]:
                state = 'brake'
            elif is_approaching[cid]:
                state = 'approach'
            elif is_departing[cid]:
                state = 'depart'
            elif is_following[cid]:
                state = 'follow'
            else:
                state = 'free'
            if red_hold[cid]:
                state = 'RED-' + state
            cars_info[cid] = {
                'pos': float(pos[cid]),
                'speed': float(vel[cid]),
                'acc_w99': float(acc_w99[cid]),
                'acc': float(acc[cid]),
                'gap': float(gap[cid]),
                'dv': float(dv[cid]),
                'abx': float(abx[cid]),
                'sdx': float(sdx[cid]),
                'dist_to_red': float(dist_to_red[cid]),
                'state': state,
                'vtype': int(vtypes[cid]),
                'vanished': bool(vanished[cid]),
                'vanish_time': car_vanish_time[cid],
            }
        sim_records[step] = {'red_timer': float(red_timer), 'cars': cars_info}

        # ---- 状态更新 ----
        mask_pos_valid = pos > 0.0
        vel_update = vel + acc * DT
        vel_new = np.clip(vel_update, 0.0, 50.0)
        vel_new = np.where(mask_pos_valid, vel_new, 0.0)
        pos_new = np.where(mask_pos_valid, pos + vel_new * DT, pos)

        # ---- 消失判定 ----
        step_sec = float(step) * DT
        new_vanish = (pos_new > inter_pos) & (~vanished)
        time_counter = np.where(new_vanish, step_sec, time_counter)
        for cid in range(NUM_VEH):
            if new_vanish[cid] and car_vanish_time[cid] is None:
                car_vanish_time[cid] = round(step_sec, 2)
        vanished = vanished | new_vanish
        red_timer = max(red_timer - DT, 0.0)

        pos = pos_new
        vel = vel_new

        if np.all(vanished):
            break

    manual_main_vanish_time = time_counter[main_idx]
    ref_main_vanish_time = _run_reference(
        nn_output, raw_data, pos_idx, speed_idx, idx_main, idx_inter, idx_red
    )

    logging.info(f"manual main_vanish_time = {manual_main_vanish_time:.2f}s  "
                 f"(car {main_idx}, vtype={vtypes[main_idx]})")
    logging.info(f"ref    main_vanish_time = {ref_main_vanish_time:.2f}s")
    logging.info(f"DataSamples   real_vanish_time = {sample_vanishTime:.2f}s")
    
    if abs(manual_main_vanish_time - ref_main_vanish_time) < 0.05:
        logging.info("[OK] 手写逐步仿真与 tf_w99v2_simulation 参考实现一致")
    else:
        logging.info("[WARN] 手写逐步仿真与参考实现不一致，请检查公式")

    return sim_records, main_idx, inter_pos, car_vanish_time

# ===================== 可视化函数 =====================
def create_simulation_gif(sim_records, main_car_id, intersection_pos):
    if os.path.exists(GIF_SAVE_PATH):
        os.remove(GIF_SAVE_PATH)
    frames = sorted(sim_records.keys())
    image_buffer = []
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT + 1.5), dpi=DPI / 2)
    ax.set_xlim(intersection_pos - 100, intersection_pos + 100)
    ax.set_ylim(-3, 4)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Single Lane 6")
    ax.set_title("Wiedemann99 Traffic Simulation Lane 6")
    ax.axhline(y=1.5, color="gray", lw=2, alpha=0.6)
    ax.axvline(x=intersection_pos, color='red', linestyle='--', label='Stop Line')
    ax.legend(loc='upper right')

    vehicle_patches = {}
    bottom_texts = {}
    for cid in range(NUM_VEH):
        color = 'red' if cid == main_car_id else 'blue'
        patch = Rectangle((0, 0), VEHICLE_WIDTH, VEHICLE_HEIGHT, facecolor=color, alpha=0.7)
        vehicle_patches[cid] = ax.add_patch(patch)
        text = ax.text(0, -2.0, "", fontsize=6, color="black", ha="center")
        bottom_texts[cid] = text

    time_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    red_text = ax.text(0.5, 0.88, "", transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    for frame_t in frames:
        fd = sim_records[frame_t]
        time_text.set_text(f"step: {frame_t} (t={frame_t * DT:.1f}s)")
        red_text.set_text(f"Red Left: {fd['red_timer']:.2f}s" if fd['red_timer'] > 0
                          else "Green Light, Go")
        for cid in range(NUM_VEH):
            cd = fd['cars'][cid]
            patch = vehicle_patches[cid]
            text = bottom_texts[cid]
            if cd['pos'] < intersection_pos - 100:
                patch.set_visible(False)
                text.set_visible(False)
            else:
                patch.set_xy((cd['pos'] - VEHICLE_WIDTH / 2, 1.0))
                patch.set_visible(True)
                info = (f"ID{cid}[T{cd['vtype']}] {cd['state']}\n"
                        f"{cd['pos']:.1f}m {cd['speed']:.1f}m/s a={cd['acc']:.2f}")
                text.set_text(info)
                text.set_x(cd['pos'])
                text.set_visible(True)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_buffer.append(img)
    plt.close(fig)
    imageio.mimsave(GIF_SAVE_PATH, image_buffer, fps=FPS)
    logging.info(f"GIF saved via imageio: {GIF_SAVE_PATH}")

# ===================== 主函数 =====================
def main():
    
    csv_path = "trainsamples_lane_5_6_7.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    df.rename(columns={'car_position': 'main_car_position',
                       'car_speed': 'main_car_speed'}, inplace=True)
    df['intersection_pos'] = df['lane'].map(LANE_POS_MAP)
    logging.info(f"Data loaded, total samples: {len(df)}")
    
    df1 = df
    if 0:
        for ii in range(len(df1)):
            row = df1.iloc[ii]
            inter_pos= float(row['intersection_pos'])
            row['main_car_position'] = inter_pos - row['main_car_position']
            for jj in range(20):
                colName = f"car_position_{jj}"
                pos = row[colName]
                if pos >0:
                    row[colName] = inter_pos - pos
            df1.iloc[ii] = row
    if 1:
        inter_pos = df1['intersection_pos']
        df1['main_car_position'] = inter_pos - df1['main_car_position']

        for jj in range(20):
            col_name = f"car_position_{jj}"
            pos = df1[col_name]
            # 只在 pos > 0 时用 inter_pos - pos，其余（-1/NaN）保持原值
            df1[col_name] = pos.where(pos <= 0, inter_pos - pos)


    sample = filter_sample(df1)
    sim_records, main_car_id, intersection_pos, car_vanish_time = simulate_single_sample(sample)
    create_simulation_gif(sim_records, main_car_id, intersection_pos)

    logging.info("==== Simulation Summary ====")
    logging.info(f"Ego Vehicle ID: {main_car_id}")
    logging.info(f"Intersection Position: {intersection_pos:.1f} m")
    last_t = max(sim_records.keys())
    last_frame = sim_records[last_t]
    main_car_final = last_frame["cars"][main_car_id]
    logging.info(f"Ego passed intersection: {main_car_final['pos'] > intersection_pos}")
    logging.info(f"Total simulation steps: {last_t} (t={last_t * DT:.1f}s)")
    logging.info("\n==== Vehicle Vanish Time Record ====")
    for vid, vt in car_vanish_time.items():
        if vt is not None:
            logging.info(f"Vehicle {vid} vanish time: {vt} s")
        else:
            logging.info(f"Vehicle {vid}: Not pass intersection in simulation")
    logging.info("Simulation finished successfully")

if __name__ == "__main__":
    print("conda activate py39_tf215_gpu ")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    main()