import tensorflow as tf

# 简化IDM参数获取函数（保持兼容）
@tf.function(reduce_retracing=True)
def get_idm_params(scaled_params, vtypes):
    """获取每个车辆的IDM参数（批量处理）"""
    v0 = tf.gather(scaled_params[:, :, 0], vtypes, axis=1, batch_dims=1)
    T = tf.gather(scaled_params[:, :, 1], vtypes, axis=1, batch_dims=1)
    s0 = tf.gather(scaled_params[:, :, 2], vtypes, axis=1, batch_dims=1)
    a_max = tf.gather(scaled_params[:, :, 3], vtypes, axis=1, batch_dims=1)
    b = tf.gather(scaled_params[:, :, 4], vtypes, axis=1, batch_dims=1)
    delta = tf.broadcast_to(4.0, tf.shape(vtypes))  # 固定值
    length = tf.broadcast_to(4.0, tf.shape(vtypes)) # 车辆长度固定4米
    rtime = tf.gather(scaled_params[:, :, 5], vtypes, axis=1, batch_dims=1)
    return v0, T, s0, a_max, b, delta, length, rtime

# 核心仿真函数（批量处理，兼容训练调用）
def tf_idm_simulation(
    nn_output_batch, raw_data_batch, param_bounds, num_types,
    pos_idx, speed_idx, idx_main, idx_inter, idx_red, dt, go_flag
):
    """
    批量IDM仿真函数（纯TF实现，基于测试验证的正确逻辑）
    参数说明（与原函数一致，保证训练调用兼容）：
    - nn_output_batch: 网络输出 (batch_size, (num_types+1)*6)
    - raw_data_batch: 原始数据 (batch_size, feature_dim)
    - param_bounds: IDM参数边界 (num_types, 6, 2)
    - num_types: 车辆类型数
    - pos_idx/speed_idx: 车辆位置/速度列索引 (num_veh,)
    - idx_main/idx_inter/idx_red: 主车位置/路口位置/红灯时长 索引
    - dt: 仿真时间步长
    - go_flag: 是否放行（1/0）
    返回：
    - main_vanish_time: 主车通过路口的时间 (batch_size,)
    """
    # ========== 1. 基础参数初始化（简化版，保留核心逻辑） ==========
    batch_size = tf.shape(nn_output_batch)[0]
    max_steps = tf.cast(120.0 / dt, tf.int32)  # 最大仿真120秒
    num_veh = 20  # 固定车辆数（与测试代码一致）
    stop_gap = tf.constant(3.0, dtype=tf.float32)  # 停车安全距离
    max_brake_acc = tf.constant(-9.0, dtype=tf.float32)  # 最大刹车加速度

    # ========== 2. 网络输出解码（与测试逻辑对齐） ==========
    # 拆分车辆参数和场景偏移量
    scaled_params0 = tf.reshape(nn_output_batch, (batch_size, num_types + 1, 6))
    scaled_params = scaled_params0[:, :-1, :]  # 车辆类型参数
    scene_offset_full = scaled_params0[:, -1, :]  # 场景偏移量

    # 参数反归一化（从[0,1]映射到实际范围）
    bounds = tf.convert_to_tensor(param_bounds, dtype=tf.float32)
    low = bounds[..., 0]
    high = bounds[..., 1]
    real_params = low + scaled_params * (high - low)

    # 场景偏移量处理（根据go_flag开关）
    offsets = [
        scene_offset_full[:, 0],  # redlighttime_offset
        scene_offset_full[:, 1],  # redlightpos2vanishpos_offset
        scene_offset_full[:, 2],  # vehpos_offset
        scene_offset_full[:, 3],  # redlightpos_offset
        scene_offset_full[:, 4],  # vanishtime_offset
        scene_offset_full[:, 5]   # distgap_offset
    ]
    offset_scales = [2.0, 8.0, 2.0, 2.0, 2.0, 2.0]
    offset_shifts = [-1.0, 0.0, -1.0, 0.0, -1.0, -1.0]
    
    # 批量处理偏移量（简化条件判断）
    processed_offsets = []
    for i, (offset, scale, shift) in enumerate(zip(offsets, offset_scales, offset_shifts)):
        if shift != 0:
            offset = (shift + offset * 2.0) * scale
        else:
            offset = offset * scale
        # 放行开关控制：非放行时偏移量置0
        offset = tf.cond(tf.equal(go_flag, 1), lambda: offset, lambda: tf.zeros_like(offset))
        processed_offsets.append(offset)
    
    (redlighttime_offset, redlightpos2vanishpos_offset, vehpos_offset,
     redlightpos_offset, vanishtime_offset, distgap_offset) = processed_offsets

    # ========== 3. 原始数据提取（纯索引，无字符串操作） ==========
    car_positions = tf.gather(raw_data_batch, pos_idx, axis=1)  # (batch, num_veh)
    car_speeds = tf.gather(raw_data_batch, speed_idx, axis=1)    # (batch, num_veh)
    main_pos = raw_data_batch[:, idx_main]                       # (batch,)
    inter_pos = raw_data_batch[:, idx_inter]                     # (batch,)
    red_dur_sec = raw_data_batch[:, idx_red] / 30.0                  # (batch,) 红灯时长归一化

    # ========== 4. 无效车辆处理（与测试逻辑一致） ==========
    mask_invalid = tf.equal(car_positions, -1.0)  # 无效车辆掩码
    # 无效车辆填充随机负位置（避免干扰仿真）
    rand_neg_pos = tf.random.uniform(
        shape=tf.shape(car_positions), minval=-5000.0, maxval=-100.0, dtype=tf.float32
    )
    car_positions = tf.where(mask_invalid, rand_neg_pos, car_positions)
    init_vanished = mask_invalid  # 初始消失车辆掩码

    # 应用偏移量
    car_positions += vehpos_offset[:, None]
    inter_pos += redlightpos_offset
    red_dur_sec += redlighttime_offset

    # 找到主车索引（距离main_pos最近的车辆）
    main_idx = tf.argmin(tf.abs(car_positions - main_pos[:, None]), axis=1)  # (batch,)

    # ========== 5. 车辆类型初始化 + IDM参数获取 ==========
    vtypes = tf.tile(tf.range(num_veh)[None, :] % num_types, [batch_size, 1])  # (batch, num_veh)
    v0, T, s0, a_max, b, delta, length, rtime = get_idm_params(real_params, vtypes)

    # ========== 6. 仿真状态初始化 ==========
    pos = tf.identity(car_positions)          # 位置 (batch, num_veh)
    vel = tf.identity(car_speeds)             # 速度 (batch, num_veh)
    time_counter = tf.zeros_like(pos)         # 车辆通过时间计数器
    vanished = tf.identity(init_vanished)     # 消失掩码
    red_timer = tf.identity(red_dur_sec)          # 红灯剩余时间 (batch,)

    # ========== 7. 仿真循环体（核心逻辑，与测试代码对齐） ==========
    def sim_body(step, pos_in, vel_in, vanished_in, time_in, red_in):
        """单次仿真步长逻辑（纯TF，批量处理）"""
        # 步骤1: 车辆位置排序（从远到近）
        idx_sort = tf.argsort(pos_in, axis=1, direction='DESCENDING')
        pos_sorted = tf.gather(pos_in, idx_sort, batch_dims=1)
        vel_sorted = tf.gather(vel_in, idx_sort, batch_dims=1)
        inv_idx = tf.argsort(idx_sort, axis=1)  # 逆索引（恢复原顺序）

        # 步骤2: 计算车间距（IDM核心）
        gap_raw = pos_sorted[:, :-1] - pos_sorted[:, 1:] - length[:, :-1]
        gap_raw = tf.maximum(gap_raw, 0.1)  # 最小间距保护
        gap_pad = tf.pad(gap_raw, [[0, 0], [1, 0]], constant_values=100.0)  # 首车填充大间距
        gap = tf.gather(gap_pad, inv_idx, batch_dims=1)  # 恢复原车辆顺序

        # 步骤3: IDM基础加速度计算
        v_opt = vel_in / (v0 + 1e-6)  # 避免除0
        sqrt_vel = tf.sqrt(tf.maximum(vel_in, 1e-6))
        sqrt_ab = tf.sqrt(tf.maximum(a_max * b, 1e-6))
        s_opt = s0 + vel_in * T + vel_in * sqrt_vel / (2 * sqrt_ab)
        acc_idm = a_max * (1.0 - tf.pow(v_opt, delta) - tf.square(s_opt / (gap + 1e-6)))

        # 步骤4: 红灯制动逻辑（测试验证的动态刹车）
        dist_to_red = inter_pos[:, None] - pos_in  # 到停止线距离 (batch, num_veh)
        red_hold = (red_in[:, None] > 0) & (dist_to_red <5)
        
        # 动态刹车加速度计算,删除原因见测试代码。不管前车，慢慢停到红灯处,所以dynamic_brake肯定小于0
        #d_remain = dist_to_red - stop_gap  # 剩余制动距离
        #mask_over = d_remain <= 0.1        # 距离过近，紧急刹车
        #mask_safe = d_remain > 0.1         # 安全距离，动态计算刹车
        
        #d_safe = tf.maximum(d_remain, 0.01)  # 避免除0
        #dynamic_brake = -(tf.square(vel_in)) / (2 * d_safe)  # 动力学刹车公式
        #dynamic_brake = tf.clip_by_value(dynamic_brake, max_brake_acc, -0.1)
        

        # 步骤5: 融合加速度（取IDM和红灯刹车的最小值）
        acc = tf.where(red_hold, max_brake_acc, acc_idm) #max_brake_acc =-9.0
        acc = tf.clip_by_value(acc, max_brake_acc, a_max)  # 加速度裁剪

        # 步骤6: 速度/位置更新（过滤无效车辆+位置<=0的车辆）
        mask_pos_valid = pos_in > 0.0  # 有效位置掩码（排除负位置车辆）
        vel_update = vel_in + acc * dt
        vel_new = tf.clip_by_value(vel_update, 0.0, 50.0)  # 速度限制0~50m/s
        
        # 位置<=0的车辆：速度置0，位置保持
        vel_new = tf.where(mask_pos_valid, vel_new, tf.zeros_like(vel_new))
        pos_new = tf.where(mask_pos_valid, pos_in + vel_new * dt, pos_in)
        
        # 步骤7: 消失车辆更新（通过路口+1米判定为消失）
        new_vanish = (pos_new > inter_pos[:, None] + 2.0) & ~vanished_in
        step_sec = tf.cast(step, tf.float32) * dt  # 当前仿真时间
        time_new = tf.where(new_vanish, step_sec, time_in)  # 记录通过时间
        vanished_new = tf.logical_or(vanished_in, new_vanish)
        
        # 步骤8: 红灯时间更新
        red_new_sec = tf.maximum(red_in - dt, 0.0)

        return step + 1, pos_new, vel_new, vanished_new, time_new, red_new_sec

    # 仿真终止条件：步数未到上限 且 还有未消失车辆
    def sim_cond(step, pos_in, vel_in, vanished_in, time_in, red_in):
        has_vehicle_left = tf.logical_not(tf.reduce_all(vanished_in))
        return tf.logical_and(step < max_steps, has_vehicle_left)

    # ========== 8. 执行TF while循环（批量仿真） ==========
    _, pos, vel, vanished, time_counter, red_timer = tf.while_loop(
        cond=sim_cond,
        body=sim_body,
        loop_vars=[tf.constant(0), pos, vel, vanished, time_counter, red_timer],
        # 调试友好：设置变量形状不变（避免TF自动推断导致的性能问题）
        shape_invariants=[
            tf.TensorShape([]),
            pos.get_shape(),
            vel.get_shape(),
            vanished.get_shape(),
            time_counter.get_shape(),
            red_timer.get_shape()
        ]
    )

    # ========== 9. 提取主车通过时间 ==========
    main_vanish_time = tf.gather(time_counter, main_idx[:, None], batch_dims=1)
    main_vanish_time = tf.squeeze(main_vanish_time, axis=1)  # (batch,)
    main_vanish_time += vanishtime_offset  # 应用时间偏移

    return main_vanish_time