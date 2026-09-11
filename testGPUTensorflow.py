#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
testGPUTensorflow.py  v2  —— ResNet 深度 x batch size 训练性能基准

用途：测量不同 batch size、不同 ResNet 近似深度下的训练吞吐，
并输出便于诊断"2060 反而没比 1660Ti 快"之类反常识现象的探针数据：
   - 每卡型号 / 显存 / 理论 fp32 算力
   - 实际 samples/s 与理论算力归一化后的"计算效率"
   - GPU 利用率 / 温度 / 功耗（后台线程采样，区分降频 / 被占用 / 数据瓶颈）
   - 纯计算(缓存数据) vs 带数据pipeline 的步耗时，识别 CPU 数据瓶颈
   - fp16 混精度开关（2060 有 TensorCore，1660Ti 没有，二者对比即诊断核心）

用法示例：
  python testGPUTensorflow.py --gpu 0 --name 2060 --batch 32,64,128 --depth 8,18,34
  python testGPUTensorflow.py --gpu 0 --name 2060 --mixed --out res_2060.json
"""

import os
import sys
import time
import json
import argparse
import threading
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ===================== 参数 =====================
P = argparse.ArgumentParser(description="ResNet 训练性能基准（供 GPU 横向对比诊断）")
P.add_argument("--gpu", type=str, default="0", help="CUDA 设备号")
P.add_argument("--name", type=str, default="", help="本机标识，如 2060 / 1660TI，用于结果对比")
P.add_argument("--batch", type=str, default="32,64,128,256", help="逗号分隔的 batch size")
P.add_argument("--depth", type=str, default="8,18,34", help="逗号分隔的 ResNet 近似深度")
P.add_argument("--steps", type=int, default=50, help="每个配置稳定计时的步数")
P.add_argument("--warmup", type=int, default=10, help="预热（编译+显存分配）步数")
P.add_argument("--mixed", action="store_true", help="启用 fp16 混精度（2060 有 TensorCore，1660Ti 无）")
P.add_argument("--out", type=str, default="", help="结果 JSON 保存路径，便于两台机器合并对比")
args = P.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# ===================== 卡信息与理论算力 =====================
# 常见卡的 fp32 峰值算力参考（TFLOPS，NVIDIA spec）。用于归一化计算效率。
KNOWN_CARDS = {
    "RTX 2080 Ti": 13.45, "RTX 2080": 10.07, "RTX 2070": 7.47,
    "RTX 2060": 6.45,     "GTX 1660 Ti": 9.08, "GTX 1660": 6.32,
    "GTX 1650": 2.99,     "RTX 3080": 29.77,   "RTX 3070": 20.31,
    "RTX 3060": 12.74,    "GTX 1080 Ti": 11.34,
    "Titan RTX": 16.31,   "V100": 14.0,
}


def get_card_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", args.gpu],
            text=True, timeout=5)
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def get_theoretical_flops(card):
    for key, flops in KNOWN_CARDS.items():
        if key.lower() in card.lower():
            return flops
    return None


# ===================== 后台采样 nvidia-smi =====================
class GpuSampler(threading.Thread):
    """后台采集 GPU 利用率/温度/功耗/显存，训练中计算均值。"""
    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.rows = []
        self.stop_flag = threading.Event()

    def run(self):
        q = "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used"
        f = "--format=csv,noheader,nounits"
        while not self.stop_flag.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", q, f, "-i", self.gpu], text=True, timeout=5)
                vals = [float(x.strip()) for x in out.strip().split(",")]
                self.rows.append(vals)
            except Exception:
                pass
            time.sleep(0.5)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)

    def mean(self):
        if not self.rows:
            return {"util": 0.0, "temp": 0.0, "power": 0.0, "mem": 0.0}
        a = np.array(self.rows)
        return {"util": a[:, 0].mean(), "temp": a[:, 1].mean(),
                "power": a[:, 2].mean(), "mem": a[:, 3].mean()}


# ===================== mini ResNet 模型 =====================
def build_mini_resnet(depth):
    """近似指定深度的 ResNet：1 卷积 + depth 个残差块前后。"""
    inpt = layers.Input((224, 224, 3), name="input")
    x = layers.Conv2D(64, 7, 2, padding="same")(inpt)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, 2, padding="same")(x)

    def block(x, ch, stride):
        s = x
        x = layers.Conv2D(ch, 3, stride, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(ch, 3, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if stride != 1 or s.shape[-1] != ch:
            s = layers.Conv2D(ch, 1, stride)(s)
            s = layers.BatchNormalization()(s)
        x = layers.add([x, s])
        return layers.Activation("relu")(x)

    plan = [(64, 1)] * depth       # 同分辨率残差块
    x = block(x, 64, 1)
    for _ in range(depth):
        x = block(x, 64, 1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation="softmax")(x)
    return tf.keras.Model(inpt, x, name=f"ResNet~{2 * depth + 3}")


# ===================== 单步训练 =====================
def make_train_step(model, opt, loss_obj):
    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_obj(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return step


def make_data_source(batch, steps, mode):
    """mode='compute':缓存数据测纯GPU计算; mode='pipeline':生成器测数据+计算。"""
    y = tf.one_hot(tf.zeros([batch], tf.int32), 1000)
    if mode == "compute":
        x = tf.random.normal([batch, 224, 224, 3])
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch).repeat()
    else:
        ds = (tf.data.Dataset.from_tensor_slices(
            (tf.zeros([batch], tf.int32), tf.zeros([batch], tf.int32)))
            .map(lambda i, j: (tf.random.normal([224, 224, 3]), y[0]), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch).repeat())
    return ds.prefetch(tf.data.AUTOTUNE)


def bench_config(depth, batch, steps, warmup, ds, train_step, sampler=None):
    """对单一 (depth, batch) 计时，返回各诊断指标。"""
    times = []
    it = iter(ds)
    warm_x, warm_y = next(it)
    for _ in range(warmup):
        train_step(warm_x, warm_y)
    for _ in range(steps):
        x, y = next(it)
        t0 = time.perf_counter()
        train_step(x, y)
        times.append(time.perf_counter() - t0)
    return times


# ===================== 主流程 =====================
def main():
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if args.mixed else "float32")

    gpus = tf.config.list_physical_devices("GPU")
    print("=" * 72)
    print("TensorFlow:", tf.__version__, "| 本机标识:", args.name or "(未命名)")
    print("GPU 设备列表:", gpus)
    if not gpus:
        print("!! 未检测到 GPU，将实际在 CPU 运行（结果无对比意义）")
    card = get_card_name()
    theo = get_theoretical_flops(card)
    print(f"卡型号: {card} | 理论 fp32 峰值: {theo} TFLOPS" if theo else
          f"卡型号: {card} | 理论算力未知(可自行加入 KNOWN_CARDS)")
    print("混精度:", "mixed_float16(TensorCore)" if args.mixed else "float32")
    print("=" * 72)

    batches = [int(b) for b in args.batch.split(",")]
    depths = [int(d) for d in args.depth.split(",")]

    results = []

    for depth in depths:
        model = build_mini_resnet(depth)
        model.summary() if depth == depths[0] else None
        loss_obj = tf.keras.losses.CategoricalCrossentropy()
        opt = tf.keras.optimizers.Adam(1e-4)
        train_step = make_train_step(model, opt, loss_obj)

        for batch in batches:
            # 先测带数据 pipeline（真实训练情形）
            sampler = GpuSampler(args.gpu)
            sampler.start()
            ds = make_data_source(batch, args.steps, mode="pipeline")
            t_pipe = bench_config(depth, batch, args.steps, args.warmup, ds, train_step)
            sampler.stop()
            s_pipe = sampler.mean()

            # 再测纯计算（缓存数据，排除数据加载/CPU影响）
            ds2 = make_data_source(batch, args.steps, mode="compute")
            t_comp = bench_config(depth, batch, args.steps, args.warmup, ds2, train_step)

            step_ms_pipe = np.mean(t_pipe) * 1e3
            step_ms_comp = np.mean(t_comp) * 1e3
            samples_s = batch / (np.mean(t_pipe))
            samples_s_comp = batch / (np.mean(t_comp))

            row = {
                "name": args.name, "gpu": card, "depth": depth, "batch": batch,
                "step_ms_pipeline": round(step_ms_pipe, 3),
                "step_ms_compute_only": round(step_ms_comp, 3),
                "samples_per_sec": round(samples_s, 1),
                "samples_per_sec_compute_only": round(samples_s_comp, 1),
                "data_overhead_pct": round(max(0, (step_ms_pipe - step_ms_comp) / step_ms_comp * 100), 1),
                "gpu_util_pct": round(s_pipe["util"], 1),
                "gpu_temp_c": round(s_pipe["temp"], 1),
                "gpu_power_w": round(s_pipe["power"], 1),
                "gpu_mem_mb": round(s_pipe["mem"], 1),
                "compute_eff": None,
            }
            if theo:
                row["compute_eff"] = round(samples_s / (theo * 1e12) * 100.0, 4)  # 每算力样本量
            results.append(row)

    # ===================== 输出汇总表 =====================
    print("\n" + "=" * 100)
    print(f"{'config':<14}{'batch':>6}{'ms/step':>28}{'samples/s':>12}{'data_ohead':>10}"
          f"{'util%':>8}{'tempC':>7}{'pwrW':>8}{'eff%':>9}")
    print("-" * 100)
    for r in results:
        e = f"{r['compute_eff']:.4f}" if r["compute_eff"] is not None else "-"
        print(f"{r['gpu'][:14]:<14}{r['batch']:>6}"
              f"{str(r['step_ms_pipeline'])+'/'+str(r['step_ms_compute_only']):>28}"
              f"{r['samples_per_sec']:>12.1f}"
              f"{str(r['data_overhead_pct'])+'%':>10}{r['gpu_util_pct']:>8.0f}"
              f"{r['gpu_temp_c']:>7.0f}{r['gpu_power_w']:>8.0f}{e:>9}")
    print("=" * 100)

    # ===================== 诊断结论 =====================
    print("\n[诊断]")
    if theo:
        hi = max(results, key=lambda x: x["samples_per_sec"])
        print(f"  最高吞吐: {hi['gpu']} depth={hi['depth']} batch={hi['batch']} = {hi['samples_per_sec']} samples/s",
              f"(算法效率 {hi['compute_eff']}% of 理论?{theo}TFLOPS)")
    low_util = [r for r in results if r["gpu_util_pct"] < 70]
    high_temp = [r for r in results if r["gpu_temp_c"] > 80]
    high_overhead = [r for r in results if r["data_overhead_pct"] > 20]
    if low_util:
        print("  ! GPU 利用率偏低(<70%):", {r["gpu"]: r["batch"] for r in low_util},
              "-> 通常是 CPU 数据加载/单步开销瓶颈，或 batch 太小")
    if high_temp:
        print("  ! 温度偏高(>80C):", {r["gpu"]: int(r["gpu_temp_c"]) for r in high_temp},
              "-> 可能触发降频，2060 笔记本尤需关注功耗墙/散热")
    if high_overhead:
        print("  ! 数据加载开销大(>20%):", {r["gpu"]: r["batch"] for r in high_overhead},
              "-> 需检查 CPU->GPU 拷贝、num_parallel_calls、磁盘IO")
    if not low_util and not high_temp and not high_overhead:
        print("  未发现明显瓶颈指标。")
        if theo:
            print("  若出现'低端卡(1660Ti)比高端卡(2060)快'的观感，请对比两机的 compute_eff：")
            print("    - 纯 fp32 训练吞吐主要由 核心数x频率 决定，1660Ti 的 fp32 峰值(9.08 TFLOPS)")
            print("      甚至高于 2060(6.45 TFLOPS)，低端卡更快在算力层面完全合理。")
            print("    - 2060 的真正优势在 fp16 TensorCore：开 --mixed 后应明显反超。")
    if args.mixed:
        print("  [混精度已开] 若 2060 比 1660Ti 明显快 -> 说明 TensorCore(fp16) 生效，正是 2060 的本职优势。")
        print("  [混精度已开] 若二者差不多或 1660Ti 仍快 -> 训练不在张量运算瓶颈，需查数据/容量/占用。")

    # ===================== 保存/合并 =====================
    if args.out:
        holder = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                              os.path.basename(args.out))
        # 若已存在则可合并（同机多次 / 两台机器结果累加）
        if os.path.exists(holder):
            try:
                with open(holder, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if isinstance(old, list):
                    old.extend(results)
                    results = old
            except Exception:
                pass
        with open(holder, "w", encoding="utf-8") as f:
            json.dump(results, ensure_ascii=False, indent=2)
        print(f"\n结果已保存/合并: {holder}")


if __name__ == "__main__":
    main()