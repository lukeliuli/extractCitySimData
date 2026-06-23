import tensorflow as tf

# 明确声明 input_signature，避免不同输入形状导致的多次 retrace
@tf.function(
    input_signature=[
        tf.TensorSpec([None, None, 6], tf.float32),
        tf.TensorSpec([None, None], tf.int32),
    ],
    experimental_relax_shapes=True,
)
def get_idm_params(scaled_params, vtypes):
    v0 = tf.gather(scaled_params[:, :, 0], vtypes, axis=1, batch_dims=1)
    T = tf.gather(scaled_params[:, :, 1], vtypes, axis=1, batch_dims=1)
    s0 = tf.gather(scaled_params[:, :, 2], vtypes, axis=1, batch_dims=1)
    a_max = tf.gather(scaled_params[:, :, 3], vtypes, axis=1, batch_dims=1)
    b = tf.gather(scaled_params[:, :, 4], vtypes, axis=1, batch_dims=1)
    delta = tf.broadcast_to(4.0, tf.shape(vtypes))
    length = tf.broadcast_to(4.0, tf.shape(vtypes))
    rtime = tf.gather(scaled_params[:, :, 5], vtypes, axis=1, batch_dims=1)
    return v0, T, s0, a_max, b, delta, length, rtime

# 新函数签名：移除columns，增加5个预计算索引张量
@tf.function(
    input_signature=[
        tf.TensorSpec([None, None], tf.float32),        # nn_output_batch: [batch, out_dim]
        tf.TensorSpec([None, None], tf.float32),        # raw_data_batch: [batch, n_cols]
        tf.TensorSpec([None, 6, 2], tf.float32),        # param_bounds: [num_types, 6, 2]
        tf.TensorSpec([], tf.int32),                   # num_types: scalar
        tf.TensorSpec([None], tf.int32),               # pos_idx: [num_veh]
        tf.TensorSpec([None], tf.int32),               # speed_idx: [num_veh]
        tf.TensorSpec([], tf.int32),                   # idx_main: scalar
        tf.TensorSpec([], tf.int32),                   # idx_inter: scalar
        tf.TensorSpec([], tf.int32),                   # idx_red: scalar
        tf.TensorSpec([], tf.float32),                 # dt: scalar
        tf.TensorSpec([], tf.int32),                   # go_flag: scalar
    ],
    experimental_relax_shapes=True,
)
def tf_idm_simulation(
    nn_output_batch, raw_data_batch, param_bounds, num_types,
    pos_idx, speed_idx, idx_main, idx_inter, idx_red, dt, go_flag
):
    batch_size = tf.shape(nn_output_batch)[0]
    max_steps = tf.cast(120.0 / dt, tf.int32)
    num_veh = 20

    # 1. 网络输出解码（不变）
    scaled_params0 = tf.reshape(nn_output_batch, (batch_size, num_types + 1, 6))
    scaled_params = scaled_params0[:, :-1, :]
    scene_offset_full = scaled_params0[:, -1, :]
    low = tf.convert_to_tensor(param_bounds, dtype=tf.float32)[..., 0]
    high = tf.convert_to_tensor(param_bounds, dtype=tf.float32)[..., 1]
    real_params = low + scaled_params * (high - low)

    redlighttime_offset = scene_offset_full[:, 0]
    redlightpos2vanishpos_offset = scene_offset_full[:, 1]
    vehpos_offset = scene_offset_full[:, 2]
    redlightpos_offset = scene_offset_full[:, 3]
    vanishtime_offset = scene_offset_full[:, 4]
    distgap_offset = scene_offset_full[:, 5]

    zero_mask = tf.zeros_like(redlighttime_offset)
    redlighttime_offset = tf.cond(tf.equal(go_flag, 1),
                                  lambda: (-1.0 + redlighttime_offset * 2.0) * 2.0,
                                  lambda: zero_mask)
    redlightpos2vanishpos_offset = tf.cond(tf.equal(go_flag, 1),
                                           lambda: redlightpos2vanishpos_offset * 8.0,
                                           lambda: zero_mask)
    vehpos_offset = tf.cond(tf.equal(go_flag, 1),
                            lambda: (-1.0 + vehpos_offset * 2.0) * 2.0,
                            lambda: zero_mask)
    redlightpos_offset = tf.cond(tf.equal(go_flag, 1),
                                 lambda: redlightpos_offset * 2.0,
                                 lambda: zero_mask)
    vanishtime_offset = tf.cond(tf.equal(go_flag, 1),
                                lambda: (-1.0 + vanishtime_offset * 2.0) * 2.0,
                                lambda: zero_mask)
    distgap_offset = tf.cond(tf.equal(go_flag, 1),
                             lambda: (-1.0 + distgap_offset * 2.0) * 2.0,
                             lambda: zero_mask)

    # 2. 直接用预计算索引取数据，删除所有columns、tf.where、字符串匹配代码
    car_positions = tf.gather(raw_data_batch, pos_idx, axis=1)
    car_speeds = tf.gather(raw_data_batch, speed_idx, axis=1)
    main_pos = raw_data_batch[:, idx_main]
    inter_pos = raw_data_batch[:, idx_inter]
    red_dur = raw_data_batch[:, idx_red] / 30.0

    vtypes = tf.tile(tf.range(num_veh)[None, :] % num_types, [batch_size, 1])
    v0, T, s0, a_max, b, delta, length, rtime = get_idm_params(real_params, vtypes)

    # 3. 无效车辆填充逻辑（不变）
    mask_invalid = tf.equal(car_positions, -1.0)
    rand_neg_pos = tf.random.uniform(
        shape=tf.shape(car_positions),
        minval=-5000.0,
        maxval=-100.0,
        dtype=tf.float32
    )
    car_positions = tf.where(mask_invalid, rand_neg_pos, car_positions)
    init_vanished = mask_invalid

    car_positions += vehpos_offset[:, None]
    inter_pos += redlightpos_offset
    red_dur += redlighttime_offset
    main_idx = tf.argmin(tf.abs(car_positions - main_pos[:, None]), axis=1)

    # 仿真状态初始化
    pos = tf.identity(car_positions)
    vel = tf.identity(car_speeds)
    time_counter = tf.zeros((batch_size, num_veh), dtype=tf.float32)
    vanished = tf.identity(init_vanished)
    red_timer = tf.identity(red_dur)

    # TF while循环（不变）
    def sim_body(step, pos_in, vel_in, vanished_in, time_in, red_in):
        idx_sort = tf.argsort(pos_in, axis=1, direction='DESCENDING')
        pos_sorted = tf.gather(pos_in, idx_sort, batch_dims=1)
        vel_sorted = tf.gather(vel_in, idx_sort, batch_dims=1)
        inv_idx = tf.argsort(idx_sort, axis=1)
        gap_raw = pos_sorted[:, :-1] - pos_sorted[:, 1:] - length[:, :-1]
        gap_raw = tf.maximum(gap_raw, 0.1)
        gap_pad = tf.pad(gap_raw, [[0, 0], [1, 0]], constant_values=100.0)
        gap = tf.gather(gap_pad, inv_idx, batch_dims=1)
        v_opt = vel_in / (v0 + 1e-6)
        s_opt = s0 + vel_in * T + vel_in * tf.sqrt(tf.maximum(vel_in, 1e-6)) / (2 * tf.sqrt(tf.maximum(a_max * b, 1e-6)))
        acc = a_max * (1.0 - tf.pow(v_opt, delta) - tf.square(s_opt / (gap + 1e-6)))
        dist_to_red = inter_pos[:, None] - pos_in
        #red_hold = (red_timer[:, None] > 0) & (dist_to_red < 2.0) & (dist_to_red > -1.0)
        red_hold = (red_in[:, None] > 0) & (dist_to_red < 30.0) & (dist_to_red > -1.0) #按统计平均30到50米，开始收油减速

        red_brake_acc = -b * 2.0  # 红灯强制减速度

        # 核心修改：红灯时取两者最小值（更负=刹车更猛），不覆盖IDM的急刹需求
        acc = tf.where(red_hold, tf.minimum(acc, red_brake_acc), acc)
        acc = tf.clip_by_value(acc, -9, a_max)
        vel_new = tf.where(vanished_in, vel_in, tf.clip_by_value(vel_in + acc * dt, 0.0, 50.0))
        pos_new = tf.where(vanished_in, pos_in, pos_in + vel_new * dt)
        new_vanish = (pos_new > inter_pos[:, None] + 5.0) & ~vanished_in
        step_sec = tf.cast(step, tf.float32) * dt
        time_new = tf.where(new_vanish, step_sec, time_in)
        vanished_new = tf.logical_or(vanished_in, new_vanish)
        red_new = tf.maximum(red_in - dt, 0.0)
        return step + 1, pos_new, vel_new, vanished_new, time_new, red_new

    def sim_cond(step, pos_in, vel_in, vanished_in, time_in, red_in):
        has_vehicle_left = tf.logical_not(tf.reduce_all(vanished_in))
        return tf.logical_and(step < max_steps, has_vehicle_left)

    _, pos, vel, vanished, time_counter, red_timer = tf.while_loop(
        cond=sim_cond,
        body=sim_body,
        loop_vars=[tf.constant(0), pos, vel, vanished, time_counter, red_timer]
    )

    main_vanish_time = tf.gather(time_counter, main_idx[:, None], batch_dims=1)
    main_vanish_time = tf.squeeze(main_vanish_time, axis=1)
    main_vanish_time += vanishtime_offset
    return main_vanish_time