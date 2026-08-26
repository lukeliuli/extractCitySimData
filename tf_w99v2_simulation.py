import tensorflow as tf
@tf.function(reduce_retracing=True)
def get_w99_params(scaled_params, vtypes):
    """
    
BASE_BOUND_VEHICLE = [
    (1.0, 2.0),   # 0: CC0 (停车间距) [m]
    (0.5, 2.0),   # 1: CC1 (车头时距) [s]
    (0.1, 30),   # 2: CC2  SDX=ABX+CC2,安全距离附加值
    (0.5, 2.0),   # 3: CC3  following状态下的D值,对应的速度差的权重
    (-0.35, 0.0), # 4: CC4 (负的相对速度阈值) [m/s]CLDV 为负
    (0.0, 0.35),  # 5: CC5 (正的相对速度阈值) [m/s] OPDV，为正
    (1,3.0), # 6: CC6 following状态下的P值，对应间距差的权重
    (2, 5),   # 7: CC7 (加速度波动) [m/s²]
    (1.0, 3.0),   # 8: CC8 (启动加速度) [m/s²]
    (0.5, 1.5)    # 9: CC9 (高速加速度) [m/s²]
]
    
    """
    # 假设 scaled_params 维度为 (batch, num_types, 10)
    # 这里为了兼容旧接口，假设传入的是10维，或者从其他维度映射
    # 为简化，假设 real_params 已经处理为 (batch, num_types, 10)
    cc0 = tf.gather(scaled_params[:, :, 0], vtypes, axis=1, batch_dims=1) # 
    cc1 = tf.gather(scaled_params[:, :, 1], vtypes, axis=1, batch_dims=1) # 
    cc2 = tf.gather(scaled_params[:, :, 2], vtypes, axis=1, batch_dims=1) # 
    cc3 = tf.gather(scaled_params[:, :, 3], vtypes, axis=1, batch_dims=1) # 
    cc4 = tf.gather(scaled_params[:, :, 4], vtypes, axis=1, batch_dims=1) # 
    cc5 = tf.gather(scaled_params[:, :, 5], vtypes, axis=1, batch_dims=1) # 
    cc6 = tf.gather(scaled_params[:, :, 6], vtypes, axis=1, batch_dims=1) #
    cc7 = tf.gather(scaled_params[:, :, 7], vtypes, axis=1, batch_dims=1) # 
    cc8 = tf.gather(scaled_params[:, :, 8], vtypes, axis=1, batch_dims=1) # 
    cc9 = tf.gather(scaled_params[:, :, 9], vtypes, axis=1, batch_dims=1) #
    
    return cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8, cc9

# ========== 2. 核心仿真函数 ==========
def tf_w99v2_simulation(
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

    safe_low = low
    safe_high = high
    real_params = tf.clip_by_value(real_params, safe_low, safe_high)

 # --- 修正版调试代码 START ---
    batch_size = tf.shape(real_params)[0]  # 动态获取batch大小
    
   
    # 正确提取每列参数（关键修复：real_params[:, i] 而非整个矩阵）
    #tf.print("---------- real_param[0,0,:]:", real_params[0,0,:], summarize=-1)  # 跟车时距

  
    # --- 修正版调试代码 END -
    # 偏移量处理 (与FVDM一致)
    offsets = [scene_offset_full[:, i] for i in range(10)]
    offset_scales = [2.0, 8.0, 2.0, 8.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    offset_shifts = [0.0, 0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    processed_offsets = []
    for offset, scale, shift in zip(offsets, offset_scales, offset_shifts):
        offset = (shift + offset * 2.0) * scale if shift != 0 else offset * scale
        offset = tf.cond(tf.equal(go_flag, 1), lambda: offset, lambda: tf.zeros_like(offset))
        processed_offsets.append(offset)
    (redlighttime_offset, _, _, redlightpos_offset, vanishtime_offset, _, _, _, _, _) = processed_offsets

    # 2. 数据提取与初始化
    car_positions = tf.gather(raw_data_batch, pos_idx, axis=1)
    car_speeds = tf.gather(raw_data_batch, speed_idx, axis=1)
    main_pos = raw_data_batch[:, idx_main]
    inter_pos = raw_data_batch[:, idx_inter]
    red_dur_sec = raw_data_batch[:, idx_red] / 30.0
    
    mask_invalid = tf.equal(car_positions, -1.0)
    rand_neg_pos = tf.random.uniform(shape=tf.shape(car_positions), minval=-5000.0, maxval=-100.0, dtype=tf.float32)
    car_positions = tf.where(mask_invalid, rand_neg_pos, car_positions)
    init_vanished = mask_invalid #位置为-1的，直接认定已经vanish对应位置为true
    
    #car_positions += vehpos_offset[:, None]
    inter_pos += redlightpos_offset
    red_dur_sec += redlighttime_offset
    main_idx = tf.argmin(tf.abs(car_positions - main_pos[:, None]), axis=1)
    
   
    
    
    # 4. 仿真状态
    pos = tf.identity(car_positions)
    vel = tf.identity(car_speeds)
    time_counter = tf.zeros_like(pos)
    vanished = tf.identity(init_vanished)
    red_timer = tf.identity(red_dur_sec)
    

    # 5. 仿真循环体
    def sim_body(step, pos_in, vel_in, vanished_in, time_in, red_in):
        # 排序 (从远到近)
        idx_sort = tf.argsort(pos_in, axis=1, direction='DESCENDING') #每一步都算，因为不存在变道，其实id_sort计算一次就OK了，
        pos_sorted = tf.gather(pos_in, idx_sort, batch_dims=1)
        vel_sorted = tf.gather(vel_in, idx_sort, batch_dims=1)
        inv_idx = tf.argsort(idx_sort, axis=1)

        #1.输入参数scaled_params，对应(batch, num_types, 10)，numtypes对应4种车型，其中分别放在0,1,2,3位置上
        #2.其中0车型的参数给到位置最大的车辆，1车型的参数排名第2的，2车型的参数给到排名第3的，3车型的参数给出到排名第4以及所有后面的车
        #。请根据idx_sort对车辆位置按从大到小的进行排列的位置，给出每辆车的车型，并基于inv_idx，一一对应 

        # 3. W99 参数获取 —— 车型按"初始位置从大到小"的排名分配
        # 排名0 → 车型0, 排名1 → 车型1, 排名2 → 车型2, 排名>=3 → 车型3
        #idx_sort_init = tf.argsort(car_positions, axis=1, direction='DESCENDING')
        #inv_idx_init  = tf.argsort(idx_sort_init, axis=1)
        idx_sort_init = idx_sort
        inv_idx_init = inv_idx

        rank_order = tf.minimum(tf.range(num_veh), num_types - 1)   # [0,1,2,3,3,...,3]
        rank_order = tf.tile(rank_order[None, :], [batch_size, 1])  # (batch, num_veh)

        # 原始第 k 辆车的车型 = rank_order[inv_idx_init[k]]
        vtypes = tf.gather(rank_order, inv_idx_init, batch_dims=1)

        cc0_s, cc1_s, cc2_s, cc3_s, cc4_s, cc5_s, cc6_s, cc7_s, cc8_s, cc9_s = get_w99_params(real_params, vtypes) #对应原始的车辆，没有排序


        
        # 计算相对变量 (恢复原序)
        cc0_sorted = tf.gather(cc0_s, idx_sort, batch_dims=1)
        gap_raw = pos_sorted[:, :-1] - pos_sorted[:, 1:] - cc0_sorted[:, :-1] # AX作为车长/停车间距
        gap_raw = tf.maximum(gap_raw, 0.1)
        gap_pad = tf.pad(gap_raw, [[0, 0], [1, 0]], constant_values=1000.0)
        gap = tf.gather(gap_pad, inv_idx, batch_dims=1)
        
        dv_raw = vel_sorted[:, :-1] - vel_sorted[:, 1:]
        dv_pad = tf.pad(dv_raw, [[0, 0], [1, 0]], constant_values=0.0)
        dv = tf.gather(dv_pad, inv_idx, batch_dims=1) 
        dv_fb = dv # dv_fb是front车减去后车的速度，论文中是后车减去前车

        


        # ---- 期望安全距离: dx_safe = CC0 + CC1*v (线性, Vissim 官方) ----
        abx = cc0_s + cc1_s * vel_in
        sdx = cc0_s + cc1_s * vel_in+ cc2_s #CC2必然大于0
        sdx = tf.maximum(sdx, 1e-3)

        # ---- 自由流加速度: 0 -> 80 km/h 线性插值 (Vissim 官方) ----
        v80 = 22.22
        v_ref = tf.minimum(vel_in, v80)
        free_acc = cc8_s + (cc9_s - cc8_s) * (v_ref / v80)
        free_acc = tf.maximum(free_acc, 0.0)

        CLDV = cc4_s#CLDV 注意论文dv和代码中dv的区别，CLDV为负值
        OPDV = cc5_s#OPDV 注意论文dv和代码中dv的区别，OPDV为正值

        # ---- 接近/减速: 后车快于前车 (dv<0) 且已进入安全距离内 ----
        #dv<0,等与前车速度小于后车，但是注意看论文dv>0等于后车快于前车，这里有坑
        
        is_approaching = tf.logical_and(dv_fb < CLDV, gap < sdx) #CLDV为负值，前车速度小于后车
        closing = tf.maximum(-dv, 0.0)                 # 接近速度差 [m/s]
        denom = tf.maximum(sdx - gap, 1e-3)            # 相对安全距离的侵入量 [m]
        #要让后车在走完 denom 这段距离时，相对速度恰好降到 0（即追上前车速度、不再逼近），所需的平均减速度。
        decel_acc = -0.5 * tf.square(closing) / denom  # 运动学制动 
        decel_acc = tf.clip_by_value(decel_acc, max_brake_acc, 0.0)




        #论文中夹在 OPDV,CLDV之间的区域，following
        cc2_eff = tf.maximum(cc2_s, 1e-3)
        cc7_mag = cc7_s #cc7一定大于0
        cc6_mag = cc6_s #cc6一定大于0
    
        is_following = tf.logical_and(
            tf.logical_and(gap >= abx, gap <= sdx),
            tf.logical_and(dv > CLDV, dv < OPDV) #注意论文dv和代码中dv的区别
           
        )
        follow_center = abx + cc2_eff * 0.5 #跟随区域,deltaX的中期
        gap_err = (follow_center - gap) / follow_center              # 带内偏差 [-1, 1]
        b_follow = cc3_s * dv_fb/OPDV -cc6_mag * gap_err # 速度差项 + 距离跟踪项，PD控制器  ,以及CC3没有用，直接用与PD
        b_follow = tf.clip_by_value(b_follow, -cc7_mag, cc7_mag)


        
        #---- 驶离/加速:  论文中OPDV= -CC5.  注意dv的坑。和论文不一样，直接用b_follow代替
        is_departing = tf.logical_and(dv_fb > OPDV, gap < sdx) #OPDV 为正值 
        #accel_acc = cc6_mag * gap_err   # 量纲一致、且速度受自由流插值约束
        #accel_acc= tf.clip_by_value(accel_acc, -cc7_mag, cc7_mag)
        accel_acc = b_follow


        is_brake = gap < abx
        gap_err_brake = abx - gap 
        brake_acc = -cc6_mag * gap_err_brake/abx


        # 优先级: 减速 > 加速 > 自由流 (跟车稳态自然落在 free_acc 上形成小振荡)
        #cc_w99 = tf.where(is_approaching, decel_acc,tf.where(is_departing, accel_acc, free_acc))
        # 优先级: 减速 > 加速 > 跟随 > 自由流
        acc_w99 = tf.where(is_brake,brake_acc,tf.where(
            is_approaching, decel_acc,
            tf.where(is_departing, accel_acc,
            tf.where(is_following, b_follow, free_acc))))
 
        
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
        new_vanish = (pos_new > inter_pos[:, None]) & ~vanished_in
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