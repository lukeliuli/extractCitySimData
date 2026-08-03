import tensorflow as tf
@tf.function(reduce_retracing=True)
def get_w99_params(scaled_params, vtypes):
    """
    
    提取 Wiedemann 99 的 CC0-CC9 参数
    为了兼容之前的6参数结构，这里做映射：
    0: CC0 (停车间距 AX)
    1: CC1 (期望车头时距 BX)
    2: CC2 (跟车变量，影响SDX)
    3: CC3 SDX 的速度衰减系数
    4: CC4 (消极跟车阈值) 跟随阈值 / Following Threshold
    5: CC5 (积极跟车阈值) 接近阈值 / Approaching Threshold)
    6: CC6 速度衰减系数（Speed Decay Coefficient），用于控制 SDV/OPDV 随车速变化的速率
    7: CC7 (加速度波动)
    8: CC8 (静止启动加速度)
    9: CC9 (80km/h加速度)

        # --- 修正 CC4 和 CC5 ---
    (-0.35, 0.35), # 4: CC4 (跟随阈值) [m/s] - 原始文献通常在此范围
    (0.35, 0.65),  # 5: CC5 (接近阈值) [m/s] - 原始文献通常在此范围
    
    """
    # 假设 scaled_params 维度为 (batch, num_types, 10)
    # 这里为了兼容旧接口，假设传入的是10维，或者从其他维度映射
    # 为简化，假设 real_params 已经处理为 (batch, num_types, 10)
    cc0 = tf.gather(scaled_params[:, :, 0], vtypes, axis=1, batch_dims=1) # AX
    cc1 = tf.gather(scaled_params[:, :, 1], vtypes, axis=1, batch_dims=1) # BX
    cc2 = tf.gather(scaled_params[:, :, 2], vtypes, axis=1, batch_dims=1) # SDX变量
    cc3 = tf.gather(scaled_params[:, :, 3], vtypes, axis=1, batch_dims=1) # SDV阈值
    cc4 = tf.gather(scaled_params[:, :, 4], vtypes, axis=1, batch_dims=1) # OPDV负阈值
    cc5 = tf.gather(scaled_params[:, :, 5], vtypes, axis=1, batch_dims=1) # OPDV正阈值
    cc6 = tf.gather(scaled_params[:, :, 6], vtypes, axis=1, batch_dims=1) # 振动
    cc7 = tf.gather(scaled_params[:, :, 7], vtypes, axis=1, batch_dims=1) # 减速度波动
    cc8 = tf.gather(scaled_params[:, :, 8], vtypes, axis=1, batch_dims=1) # 启动加速度
    cc9 = tf.gather(scaled_params[:, :, 9], vtypes, axis=1, batch_dims=1) # 高速加速度
    
    return cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8, cc9

# ========== 2. 核心仿真函数 ==========
def tf_wiedemann99_simulation(
    nn_output_batch, raw_data_batch, param_bounds, num_types, 
    pos_idx, speed_idx, idx_main, idx_inter, idx_red, dt, go_flag
):
    batch_size = tf.shape(nn_output_batch)[0]
    max_steps = tf.cast(120.0 / dt, tf.int32)
    num_veh = 20
    max_brake_acc = tf.constant(-9.0, dtype=tf.float32)
    
    # 1. 参数解码与反归一化
    scaled_params0 = tf.reshape(nn_output_batch, (batch_size, num_types + 1, 10)) # 注意：W99需要10个参数
    scaled_params = scaled_params0[:, :-1, :]
    scene_offset_full = scaled_params0[:, -1, :]
    
    bounds = tf.convert_to_tensor(param_bounds, dtype=tf.float32)
    low, high = bounds[..., 0], bounds[..., 1]
    real_params = low + scaled_params * (high - low)

    safe_low = tf.constant([0.5, 0.5, 0.1, 1.0, -0.35, 0.0, 10.0, 0.1, 1.0, 0.5], dtype=tf.float32)
    safe_high = tf.constant([2.5, 2.5, 0.6, 8.0, 0.0, 0.35, 20.0, 0.5, 3.0, 1.5], dtype=tf.float32)
    real_params = tf.clip_by_value(real_params, safe_low, safe_high)

 # --- 修正版调试代码 START ---
    batch_size = tf.shape(real_params)[0]  # 动态获取batch大小
    
   
    # 正确提取每列参数（关键修复：real_params[:, i] 而非整个矩阵）
    tf.print("debug:Wiedemann99 参数 real_param[0,0,:]:\n", real_params[0,0,:], summarize=-1)  # 跟车时距
  
    # --- 修正版调试代码 END -
    # 偏移量处理 (与FVDM一致)
    offsets = [scene_offset_full[:, i] for i in range(6)]
    offset_scales = [2.0, 8.0, 2.0, 2.0, 2.0, 2.0]
    offset_shifts = [-1.0, 0.0, -1.0, 0.0, -1.0, -1.0]
    processed_offsets = []
    for offset, scale, shift in zip(offsets, offset_scales, offset_shifts):
        offset = (shift + offset * 2.0) * scale if shift != 0 else offset * scale
        offset = tf.cond(tf.equal(go_flag, 1), lambda: offset, lambda: tf.zeros_like(offset))
        processed_offsets.append(offset)
    (redlighttime_offset, _, _, redlightpos_offset, vanishtime_offset, _) = processed_offsets

    # 2. 数据提取与初始化
    car_positions = tf.gather(raw_data_batch, pos_idx, axis=1)
    car_speeds = tf.gather(raw_data_batch, speed_idx, axis=1)
    main_pos = raw_data_batch[:, idx_main]
    inter_pos = raw_data_batch[:, idx_inter]
    red_dur_sec = raw_data_batch[:, idx_red] / 30.0
    
    mask_invalid = tf.equal(car_positions, -1.0)
    rand_neg_pos = tf.random.uniform(shape=tf.shape(car_positions), minval=-5000.0, maxval=-100.0, dtype=tf.float32)
    car_positions = tf.where(mask_invalid, rand_neg_pos, car_positions)
    init_vanished = mask_invalid
    
    #car_positions += vehpos_offset[:, None]
    inter_pos += redlightpos_offset
    red_dur_sec += redlighttime_offset
    main_idx = tf.argmin(tf.abs(car_positions - main_pos[:, None]), axis=1)
    
    # 3. W99 参数获取
    vtypes = tf.tile(tf.range(num_veh)[None, :] % num_types, [batch_size, 1])
    cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8, cc9 = get_w99_params(real_params, vtypes)
    
    # 4. 仿真状态
    pos = tf.identity(car_positions)
    vel = tf.identity(car_speeds)
    time_counter = tf.zeros_like(pos)
    vanished = tf.identity(init_vanished)
    red_timer = tf.identity(red_dur_sec)

    # 5. 仿真循环体
    def sim_body(step, pos_in, vel_in, vanished_in, time_in, red_in):
        # 排序 (从远到近)
        idx_sort = tf.argsort(pos_in, axis=1, direction='DESCENDING')
        pos_sorted = tf.gather(pos_in, idx_sort, batch_dims=1)
        vel_sorted = tf.gather(vel_in, idx_sort, batch_dims=1)
        inv_idx = tf.argsort(idx_sort, axis=1)
        
        # 计算相对变量 (恢复原序)
        gap_raw = pos_sorted[:, :-1] - pos_sorted[:, 1:] - cc0[:, :-1] # AX作为车长/停车间距
        gap_raw = tf.maximum(gap_raw, 0.1)
        gap_pad = tf.pad(gap_raw, [[0, 0], [1, 0]], constant_values=1000.0)
        gap = tf.gather(gap_pad, inv_idx, batch_dims=1)
        
        dv_raw = vel_sorted[:, :-1] - vel_sorted[:, 1:]
        dv_pad = tf.pad(dv_raw, [[0, 0], [1, 0]], constant_values=0.0)
        dv = tf.gather(dv_pad, inv_idx, batch_dims=1)
        
        # 2. 动态计算 SDV 和 SDX
        #SDV 在这里代表的并不是“允许的最大相对速度”，而是触发减速的相对速度下限（或称“逼近感知阈值”）
        #它的取值范围在 cc4 到 cc5 之间
        sdv = cc4 + (cc5 - cc4) * tf.exp(-vel_in / tf.maximum(cc6, 1e-5))
        sdx = cc0 + vel_in * (cc1 + (cc2 - cc1) * tf.exp(-vel_in / tf.maximum(cc3, 1e-5)))
        sdx = tf.maximum(sdx, 1e-5)

        # A.0 检查是否触发减速条件: dv < SDV 且 gap < SDX
        is_approaching = tf.logical_and(dv < sdv, gap < sdx)
    
        # A.1计算减速加速度 (仅在触发条件下有效)
        #weight = tf.maximum(0.0, (sdx - gap) / tf.maximum(sdx, 1e-5))  # 车距权重项
        #raw_acc = -cc7 * (dv - sdv) * weight  # 原始公式
        decel_acc = -cc7* (sdv - dv) *tf.maximum(0.0, (sdx- gap)/sdx) # 简化的减速逻辑
        

  
        
        # B.0 自由流加速度 (基于CC8和CC9的线性插值)
        # 80 km/h = 22.22 m/s
        # 50 km/h = 13.89 m/s
        #a_free = CC8 * (1 - (vel_in / v0) ** CC9)  # 严格遵循Wiedemann 99原始定义
        max_v =13.89
        free_acc = cc8*(1- tf.pow(tf.minimum(vel_in, max_v) / max_v, cc9))
        free_acc = tf.maximum(free_acc, 0.0)
        
  
        # C.0  加速/驶离加速度 (基于OPDV)
        # 当 dv > OPDV 且 gap > SDX 时触发
        #自由行驶 → 接近：当 (实际距离 < SDX) AND (相对速度差 < SDV) 时触发
            # 3. 动态计算 OPDV
        opdv = cc8 + (cc9 - cc8) * tf.exp(-vel_in / tf.maximum(cc6, 1e-5))
        
        # 4. 检查是否触发加速条件: dv > OPDV 且 gap > SDX
        is_departing = tf.logical_and(dv > opdv, gap > sdx)
        
        # 5. 计算加速加速度 (仅在触发条件下有效)
        weight = tf.maximum(0.0, (gap - sdx) / tf.maximum(gap, 1e-5))  # 车距权重项 (分母为gap!)
        accel_acc = cc7 * (dv - opdv) * weight  # 原始公式
    
        
       
        
        # 优先级：减速 > 加速 > 自由流
        acc_w99 = tf.where(is_approaching, decel_acc, tf.where(is_departing, accel_acc, free_acc))
        
        # 4. 红灯硬约束
        dist_to_red = inter_pos[:, None] - pos_in
        red_hold = (red_in[:, None] > 0) & (dist_to_red < 5.0)
        acc = tf.where(red_hold, max_brake_acc, acc_w99)
        acc = tf.clip_by_value(acc, max_brake_acc, 3.0)
        
        # 5. 状态更新
        mask_pos_valid = pos_in > 0.0
        vel_new = tf.clip_by_value(vel_in + acc * dt, 0.0, 50.0)
        vel_new = tf.where(mask_pos_valid, vel_new, tf.zeros_like(vel_new))
        pos_new = tf.where(mask_pos_valid, pos_in + vel_new * dt, pos_in)
        
        # 6. 消失判定
        new_vanish = (pos_new > inter_pos[:, None] + 2.0) & ~vanished_in
        step_sec = tf.cast(step, tf.float32) * dt
        time_new = tf.where(new_vanish, step_sec, time_in)
        vanished_new = tf.logical_or(vanished_in, new_vanish)
        red_new_sec = tf.maximum(red_in - dt, 0.0)
        
        return step + 1, pos_new, vel_new, vanished_new, time_new, red_new_sec

    def sim_cond(step, pos_in, vel_in, vanished_in, time_in, red_in):
        has_vehicle_left = tf.logical_not(tf.reduce_all(vanished_in))
        return tf.logical_and(step < max_steps, has_vehicle_left)

    # 6. 执行循环
    _, pos, vel, vanished, time_counter, red_timer = tf.while_loop(
        cond=sim_cond, body=sim_body,
        loop_vars=[tf.constant(0), pos, vel, vanished, time_counter, red_timer],
        shape_invariants=[tf.TensorShape([]), pos.get_shape(), vel.get_shape(), 
                          vanished.get_shape(), time_counter.get_shape(), red_timer.get_shape()]
    )
    
    # 7. 输出
    main_vanish_time = tf.gather(time_counter, main_idx[:, None], batch_dims=1)
    #main_vanish_time = tf.squeeze(main_vanish_time, axis=1) + vanishtime_offset
    return main_vanish_time