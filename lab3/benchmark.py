"""
实验3 复杂度与速度测评（1.4）
==============================
对 outputs/ 下每个训练好的模型，以及在参考模型上直接应用 ToMe（免训练）的若干变体，统计：
  - 参数量
  - FLOPs：torch.utils.flop_counter（只统计矩阵乘/卷积，1 MAC = 2 FLOPs），输入 1×3×32×32
  - 训练吞吐：单卡 batch=256、bf16 下一次 forward+backward+AdamW step 的 images/s
  - 推理吞吐：GPU batch=1024 bf16 images/s；GPU batch=1 延迟（ms）；CPU（8 线程）batch=64 fp32 images/s
  - ToMe 变体在均衡测试集上的分类表现（写入 outputs/tome_r*/metrics.json）
另外用 torchinfo 输出基线 ViT 的逐层结构表（outputs/vit_summary.txt）。

用法：python benchmark.py --ref <参考模型名> [--tome_r 4 6 8]
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from data import GPUData
from models import DynamicViT, load_model
from train import HERE, autocast, metrics_from, plot_confusion, predict, set_seed


def count_flops(model, device="cpu"):
    model = copy.deepcopy(model).to(device).eval().float()
    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


def bench(fn, sync, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def speed(model, device):
    res = {}
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    # 训练吞吐
    m = copy.deepcopy(model).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randn(256, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    def train_step():
        with autocast(device):
            out = m(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    res["train_img_s"] = 256 / bench(train_step, sync, iters=30)
    del m, opt

    m = copy.deepcopy(model).to(device).eval()
    with torch.no_grad():
        xb = torch.randn(1024, 3, 32, 32, device=device)
        x1 = torch.randn(1, 3, 32, 32, device=device)

        def inf_big():
            with autocast(device):
                m(xb)

        def inf_one():
            with autocast(device):
                m(x1)

        res["gpu_infer_img_s"] = 1024 / bench(inf_big, sync)
        res["gpu_latency_ms"] = bench(inf_one, sync, iters=200) * 1000

    mc = copy.deepcopy(model).cpu().eval().float()
    torch.set_num_threads(8)
    with torch.no_grad():
        xc = torch.randn(64, 3, 32, 32)
        res["cpu_infer_img_s"] = 64 / bench(lambda: mc(xc), lambda: None, warmup=3, iters=20)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="参考模型（ToMe 在其上免训练应用）")
    p.add_argument("--tome_r", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--out", default=os.path.join(HERE, "outputs"))
    args = p.parse_args()
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = GPUData(device, seed=42)
    with open(os.path.join(args.out, "train_counts.json"), "w") as f:
        json.dump(data.class_counts.tolist(), f)

    entries = []
    for name in sorted(os.listdir(args.out)):
        ck = os.path.join(args.out, name, "model.pth")
        if os.path.exists(ck) and not name.startswith("tome_"):
            entries.append((name, ck, 0))
    ref_ck = os.path.join(args.out, args.ref, "model.pth")
    entries += [(f"tome_r{r}", ref_ck, r) for r in args.tome_r]

    results = {}
    for name, ck, r in entries:
        model, meta = load_model(ck, device)
        if r:
            model.tome_r = r
            d = os.path.join(args.out, name)
            os.makedirs(d, exist_ok=True)
            tm = metrics_from(*predict(model, data, "test"))
            with open(os.path.join(d, "metrics.json"), "w") as f:
                json.dump(dict(name=name, imbalance=meta["args"]["imbalance"], light="tome", init=os.path.relpath(ck, HERE),
                               tome_r=r, params=sum(p.numel() for p in model.parameters()),
                               train_time_s=0.0, test=tm), f, indent=2)
            plot_confusion(tm["confusion"], os.path.join(d, "confusion.png"),
                           f"{name}  test acc={tm['acc']:.4f}")
        n_params = sum(p.numel() for p in model.parameters())
        flops = count_flops(model)
        sp = speed(model, device)
        results[name] = dict(params=n_params, flops=flops, **sp,
                             kind="dynamicvit" if isinstance(model, DynamicViT) else "vit", tome_r=r)
        print(f"{name:14s} params={n_params/1e6:6.3f}M  FLOPs={flops/1e9:6.3f}G  "
              f"train={sp['train_img_s']:8.0f} img/s  gpu_infer={sp['gpu_infer_img_s']:9.0f} img/s  "
              f"lat={sp['gpu_latency_ms']:.2f}ms  cpu={sp['cpu_infer_img_s']:7.0f} img/s")

    results["_env"] = dict(device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                           torch=torch.__version__, cpu_threads=8)
    with open(os.path.join(args.out, "benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)

    from torchinfo import summary
    base, _ = load_model(os.path.join(args.out, "baseline", "model.pth"), "cpu")
    s = summary(base, input_size=(1, 3, 32, 32), depth=4, verbose=0,
                col_names=("input_size", "output_size", "num_params", "mult_adds"))
    with open(os.path.join(args.out, "vit_summary.txt"), "w") as f:
        f.write(str(s))


if __name__ == "__main__":
    main()
