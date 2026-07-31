# 基于tf_idm_sumulation.py,将idm仿真函数改为fvdm仿真函数，保留核心逻辑，兼容被modelsCollect9.py训练调用逻辑
# 注意：纯tf函数，支持gpu训练;给出简单的测试代码;tensorflow版本2.15,python版本3.9;代码简洁逻辑明显，再考虑并行性能优化
import tensorflow as tf

# 简化参数获取函数（保持接口兼容，但只提取FVDM需要的参数）
@tf.function(reduce_retracing=True)
def get_fvdm_params(scaled_params, vtypes):
    """
    获取每个车辆的FVDM参数
    FVDM核心参数: v0(期望速度), alpha(灵敏度), lambda(相对速度灵敏度), length(车长)
    """
    # 注意：scaled_params 维度为 (batch, num_types, 6)
    # 我们重新定义这6个参数的含义以适配FVDM:
    # 0: v0 (期望速度)
    # 1: alpha (加速度灵敏度系数)
    # 2: lambda (前车相对速度灵敏度系数)
    # 3: length (车长，通常固定，这里为了兼容保留索引)
    # 4: rtime (反应时间，FVDM通常隐含在离散化中，这里保留占位)
    # 5: 预留
    
    v0 = tf.gather(scaled_params[:, :, 0], vtypes, axis=1, batch_dims=1)
    alpha = tf.gather(scaled_params[:, :, 1], vtypes, axis=1, batch_dims=1) # 灵敏度
    lam = tf.gather(scaled_params[:, :, 2], vtypes, axis=1, batch_dims=1)   # 相对速度系数
    length = tf.gather(scaled_params[:, :, 3], vtypes, axis=1, batch_dims=1) # 车长
    tanhP1 = tf.gather(scaled_params[:, :, 4], vtypes, axis=1, batch_dims=1) # 预留参数，FVDM中可用于最大加速度限制
    rtime = tf.gather(scaled_params[:, :, 5], vtypes, axis=1, batch_dims=1) # 反应时间
    # FVDM通常需要一个期望速度函数 V(s)，这里简化为线性或直接使用 v0 作为上限
    # 为了简化，我们假设 V(s) 的逻辑在加速度计算中体现，或者 v0 就是最大速度
    # 这里返回 v0 作为最大速度参考
    return v0, alpha, lam, length, tanhP1, rtime

# 核心仿真函数
def tf_fvdm_simulation(
    nn_output_batch, 
    raw_data_batch, 
    param_bounds, 
    num_types, 
    pos_idx, 
    speed_idx, 
    idx_main, 
    idx_inter, 
    idx_red, 
    dt, 
    go_flag
):
    """
    批量FVDM仿真函数（纯TF实现）
    参数说明与原版IDM保持一致，仅内部动力学逻辑变更
    """
    # ========== 1. 基础参数初始化 ==========
    batch_size = tf.shape(nn_output_batch)[0]
    max_steps = tf.cast(120.0 / dt, tf.int32)
    num_veh = 20 # 固定车辆数
    stop_gap = tf.constant(3.0, dtype=tf.float32)
    max_brake_acc = tf.constant(-9.0, dtype=tf.float32)
    amax = tf.constant(2.0, dtype=tf.float32)
    
    # ========== 2. 网络输出解码 ==========
    scaled_params0 = tf.reshape(nn_output_batch, (batch_size, num_types + 1, 6))
    scaled_params = scaled_params0[:, :-1, :] # 车辆类型参数
    scene_offset_full = scaled_params0[:, -1, :] # 场景偏移量
    
    # 参数反归一化
    bounds = tf.convert_to_tensor(param_bounds, dtype=tf.float32)
    low = bounds[..., 0]
    high = bounds[..., 1]
    real_params = low + scaled_params * (high - low)
    
    # 场景偏移量处理
    offsets = [scene_offset_full[:, i] for i in range(6)]
    offset_scales = [2.0, 8.0, 2.0, 2.0, 2.0, 2.0]
    offset_shifts = [-1.0, 0.0, -1.0, 0.0, -1.0, -1.0]
    
    processed_offsets = []
    for offset, scale, shift in zip(offsets, offset_scales, offset_shifts):
        if shift != 0:
            offset = (shift + offset * 2.0) * scale
        else:
            offset = offset * scale
        offset = tf.cond(tf.equal(go_flag, 1), lambda: offset, lambda: tf.zeros_like(offset))
        processed_offsets.append(offset)
        
    (redlighttime_offset, _, vehpos_offset, redlightpos_offset, vanishtime_offset, _) = processed_offsets
    #    (redlighttime_offset, redlightpos2vanishpos_offset, vehpos_offset,
    # redlightpos_offset, vanishtime_offset, distgap_offset) = processed_offsets

    # ========== 3. 原始数据提取 ==========
    car_positions = tf.gather(raw_data_batch, pos_idx, axis=1)
    car_speeds = tf.gather(raw_data_batch, speed_idx, axis=1)
    main_pos = raw_data_batch[:, idx_main]
    inter_pos = raw_data_batch[:, idx_inter]
    red_dur_sec = raw_data_batch[:, idx_red] / 30.0
    
    # ========== 4. 无效车辆处理 ==========
    mask_invalid = tf.equal(car_positions, -1.0)
    rand_neg_pos = tf.random.uniform(shape=tf.shape(car_positions), minval=-5000.0, maxval=-100.0, dtype=tf.float32)
    car_positions = tf.where(mask_invalid, rand_neg_pos, car_positions)
    init_vanished = mask_invalid
    
    # 应用偏移量
    car_positions += vehpos_offset[:, None]
    inter_pos += redlightpos_offset
    red_dur_sec += redlighttime_offset
    
    main_idx = tf.argmin(tf.abs(car_positions - main_pos[:, None]), axis=1)
    
    # ========== 5. 车辆类型初始化 + FVDM参数获取 ==========
    # 注意：这里vtypes逻辑保持不变，但get_fvdm_params内部只取需要的参数
    vtypes = tf.tile(tf.range(num_veh)[None, :] % num_types, [batch_size, 1])
    v0, alpha, lam, length,tanhP1, rtime = get_fvdm_params(real_params, vtypes)
    
    # FVDM特有参数初始化
    s_stopped = tf.constant(2.0, dtype=tf.float32) # 停车时的最小净空距 (jam spacing)

    # ========== 6. 仿真状态初始化 ==========
    pos = tf.identity(car_positions)
    vel = tf.identity(car_speeds)
    time_counter = tf.zeros_like(pos)
    vanished = tf.identity(init_vanished)
    red_timer = tf.identity(red_dur_sec)

    # ========== 7. 仿真循环体 ==========
    def sim_body(step, pos_in, vel_in, vanished_in, time_in, red_in):
        # 1. 排序 (从远到近)
        idx_sort = tf.argsort(pos_in, axis=1, direction='DESCENDING')
        pos_sorted = tf.gather(pos_in, idx_sort, batch_dims=1)
        vel_sorted = tf.gather(vel_in, idx_sort, batch_dims=1)
        inv_idx = tf.argsort(idx_sort, axis=1) # 恢复原序
        
        # 2. 计算间距 (gap) 和 相对速度 (dv)
        # gap = 前车位 - 后车位 - 车长
        gap_raw = pos_sorted[:, :-1] - pos_sorted[:, 1:] - tf.gather(length, idx_sort, batch_dims=1)[:, :-1]
        gap_raw = tf.maximum(gap_raw, 0.1) # 保护
        gap_pad = tf.pad(gap_raw, [[0, 0], [1, 0]], constant_values=1000.0) # 首车大间距
        gap = tf.gather(gap_pad, inv_idx, batch_dims=1) # 恢复原序
        
        # 相对速度 dv = 前车速度 - 本车速度
        dv_raw = vel_sorted[:, :-1] - vel_sorted[:, 1:]
        dv_pad = tf.pad(dv_raw, [[0, 0], [1, 0]], constant_values=0.0) # 首车相对速度0
        dv = tf.gather(dv_pad, inv_idx, batch_dims=1)
        
        # 3. FVDM 加速度计算
        # 期望速度函数 V(gap): 简单的线性或双曲正切函数，这里用简单的线性截断
        # 如果 gap < s_stopped, V=0; 否则 V 随 gap 增加趋向 v0
        # 简化版 V_des = v0 * tanh(gap - s_stopped) 或者简单的线性映射
        # 这里采用更平滑的 tanh 形式模拟驾驶员心理
        v_des = v0 * tf.tanh(tf.math.softplus(gap - s_stopped) / tanhP1)
        
        # FVDM公式: a = alpha * (V_des - v) + lambda * dv
        acc_fvdm = alpha * (v_des - vel_in) + lam * dv
        
        # 4. 红灯制动逻辑 (硬约束)
        dist_to_red = inter_pos[:, None] - pos_in
        # 如果红灯亮 且 距离停止线很近 (<5m)，强制刹车
        red_hold = (red_in[:, None] > 0) & (dist_to_red < 5.0)
        
        # 融合：取 IDM/FVDM加速度 和 红灯强制减速度 的最小值
        acc = tf.where(red_hold, max_brake_acc, acc_fvdm)
        acc = tf.clip_by_value(acc, max_brake_acc, amax) # 加速度限幅 (最大加速度通常由alpha决定或单独参数)

        # 5. 状态更新
        mask_pos_valid = pos_in > 0.0
        vel_update = vel_in + acc * dt
        vel_new = tf.clip_by_value(vel_update, 0.0, 50.0)
        
        # 无效车辆处理
        vel_new = tf.where(mask_pos_valid, vel_new, tf.zeros_like(vel_new))
        pos_new = tf.where(mask_pos_valid, pos_in + vel_new * dt, pos_in)
        
        # 6. 消失判定
        new_vanish = (pos_new > inter_pos[:, None] + 2.0) & ~vanished_in
        step_sec = tf.cast(step, tf.float32) * dt
        time_new = tf.where(new_vanish, step_sec, time_in)
        vanished_new = tf.logical_or(vanished_in, new_vanish)
        
        # 7. 红灯倒计时
        red_new_sec = tf.maximum(red_in - dt, 0.0)
        
        return step + 1, pos_new, vel_new, vanished_new, time_new, red_new_sec

    def sim_cond(step, pos_in, vel_in, vanished_in, time_in, red_in):
        has_vehicle_left = tf.logical_not(tf.reduce_all(vanished_in))
        return tf.logical_and(step < max_steps, has_vehicle_left)

    # ========== 8. 执行仿真 ==========
    _, pos, vel, vanished, time_counter, red_timer = tf.while_loop(
        cond=sim_cond,
        body=sim_body,
        loop_vars=[tf.constant(0), pos, vel, vanished, time_counter, red_timer],
        shape_invariants=[
            tf.TensorShape([]), pos.get_shape(), vel.get_shape(), 
            vanished.get_shape(), time_counter.get_shape(), red_timer.get_shape()
        ]
    )
    
    # ========== 9. 输出结果 ==========
    main_vanish_time = tf.gather(time_counter, main_idx[:, None], batch_dims=1)
    main_vanish_time = tf.squeeze(main_vanish_time, axis=1)
    main_vanish_time += vanishtime_offset
    
    return main_vanish_time