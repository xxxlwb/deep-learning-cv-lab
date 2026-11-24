"""
实验3 结果汇总：读取 outputs/ 下各模型的 metrics.json / history.json 与 benchmark.json，
生成对比表 outputs/results.md 和对比图 outputs/fig_*.png。

用法：python analyze.py --ref <参考模型名>
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import CLASSES
from train import HERE

IMB = ["baseline", "reweight", "resample", "logit_adjust"]
IMB_LABEL = {"baseline": "CE (baseline)", "reweight": "Re-weighting", "resample": "Re-sampling",
             "logit_adjust": "Logit adjustment"}


def load(out, name, f):
    p = os.path.join(out, name, f)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    a = ap.parse_args()
    out = a.out
    bm = json.load(open(os.path.join(out, "benchmark.json")))
    names = sorted(n for n in os.listdir(out) if os.path.exists(os.path.join(out, n, "metrics.json")))
    M = {n: load(out, n, "metrics.json") for n in names}
    H = {n: load(out, n, "history.json") for n in names}
    counts = np.array(json.load(open(os.path.join(out, "train_counts.json"))))

    lines = [f"# 实验3 结果汇总\n", f"设备：{bm['_env']['device']}，torch {bm['_env']['torch']}；"
             f"参考模型（轻量化的起点与蒸馏教师）：`{a.ref}`\n"]

    # ── 1.1 / 1.2 类别不均衡 ─────────────────────────────────────────────
    order = np.argsort(-counts)
    few = order[-3:][::-1]
    lines += ["## 基线与类别不均衡处理（均衡测试集）\n",
              "| 方法 | 测试准确率 | Macro-F1 | 最差类召回 | " + " | ".join(f"{CLASSES[i]}({counts[i]})" for i in few)
              + " | 训练时间(s) |", "|" + "---|" * (5 + len(few))]
    for n in IMB:
        if n not in M:
            continue
        t = M[n]["test"]
        r = np.array(t["per_class_recall"])
        lines.append(f"| {IMB_LABEL[n]} | {t['acc']*100:.2f}% | {t['macro_f1']*100:.2f} | "
                     f"{r.min()*100:.1f}% ({CLASSES[r.argmin()]}) | "
                     + " | ".join(f"{r[i]*100:.1f}%" for i in few) + f" | {M[n]['train_time_s']:.0f} |")
    lines.append("")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.8 / len([n for n in IMB if n in M])
    for k, n in enumerate([n for n in IMB if n in M]):
        r = np.array(M[n]["test"]["per_class_recall"])[order]
        ax.bar(np.arange(10) + k * w - 0.4 + w / 2, r, w, label=IMB_LABEL[n])
    ax.set_xticks(range(10), [f"{CLASSES[i]}\n(n={counts[i]})" for i in order])
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("test recall"); ax.legend(ncol=4, loc="lower left")
    ax.set_title("Per-class recall on balanced test set (classes sorted by training count)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_imbalance_recall.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n in IMB:
        if H.get(n):
            ax[0].plot([h["train_loss"] for h in H[n]], label=IMB_LABEL[n])
            ax[1].plot([h["val_balanced_acc"] for h in H[n]], label=IMB_LABEL[n])
    ax[0].set_title("training loss"); ax[1].set_title("validation balanced accuracy")
    for x in ax:
        x.set_xlabel("epoch"); x.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_imbalance_curves.png"), dpi=140); plt.close(fig)

    # ── 1.3 / 1.4 轻量化与复杂度 ──────────────────────────────────────────
    light = [a.ref, "dynamicvit", "prune"] + sorted(n for n in names if n.startswith("tome_"))
    light = [n for n in light if n in M and n in bm]
    ref_b, ref_acc = bm[a.ref], M[a.ref]["test"]["acc"]
    lines += ["## 轻量化方法与复杂度（参数量 / FLOPs / 速度）\n",
              "FLOPs：torch.utils.flop_counter，单张 32×32 输入，只统计矩阵乘与卷积（1 MAC = 2 FLOPs）。"
              "训练吞吐：batch 256、bf16；GPU 推理：batch 1024、bf16；CPU 推理：8 线程、batch 64、fp32。括号内为相对参考模型的倍数。\n",
              "| 模型 | 参数量 | FLOPs | 测试准确率 | Δ准确率 | 训练吞吐(img/s) | GPU推理(img/s) | GPU延迟 bs=1(ms) | CPU推理(img/s) | 额外训练时间(s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for n in light:
        b, t = bm[n], M[n]["test"]
        rel = lambda k: b[k] / ref_b[k]
        label = "参考 ViT" if n == a.ref else n
        lines.append(
            f"| {label} | {b['params']/1e6:.3f}M ({rel('params'):.2f}×) | {b['flops']/1e9:.3f}G ({rel('flops'):.2f}×) | "
            f"{t['acc']*100:.2f}% | {(t['acc']-ref_acc)*100:+.2f} | {b['train_img_s']:.0f} ({rel('train_img_s'):.2f}×) | "
            f"{b['gpu_infer_img_s']:.0f} ({rel('gpu_infer_img_s'):.2f}×) | {b['gpu_latency_ms']:.2f} | "
            f"{b['cpu_infer_img_s']:.0f} ({rel('cpu_infer_img_s'):.2f}×) | "
            f"{'—' if n == a.ref else int(M[n].get('train_time_s', 0))} |")
        rows.append((n, b, t["acc"]))
    others = [n for n in IMB if n in bm and n != a.ref]
    if others:
        lines += ["", "不均衡处理只改变损失/采样，不改变结构，其余模型的复杂度与参考模型相同："]
        for n in others:
            b = bm[n]
            lines.append(f"- {n}: {b['params']/1e6:.3f}M 参数, {b['flops']/1e9:.3f}G FLOPs, "
                         f"训练 {b['train_img_s']:.0f} img/s, GPU 推理 {b['gpu_infer_img_s']:.0f} img/s")
    if "prune" in M:
        lines += ["", f"剪枝：保留头数 {M['prune']['kept_heads']}，保留 MLP 宽度 {M['prune']['kept_mlp']}；"
                  f"剪枝后未微调的测试准确率 {M['prune']['test_acc_after_prune_before_ft']*100:.2f}%。"]
    lines.append("")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, b, acc in rows:
        ax[0].scatter(b["flops"] / 1e9, acc * 100, s=b["params"] / 1e4, alpha=0.7)
        ax[0].annotate(n, (b["flops"] / 1e9, acc * 100), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax[0].set_xlabel("GFLOPs / image"); ax[0].set_ylabel("test accuracy (%)")
    ax[0].set_title("Accuracy vs FLOPs (marker area ∝ params)")
    keys = [("train_img_s", "train"), ("gpu_infer_img_s", "GPU infer"), ("cpu_infer_img_s", "CPU infer")]
    w = 0.8 / len(keys)
    for k, (key, lab) in enumerate(keys):
        ax[1].bar(np.arange(len(rows)) + k * w - 0.4 + w / 2, [b[key] / ref_b[key] for _, b, _ in rows], w, label=lab)
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_xticks(range(len(rows)), [n for n, _, _ in rows], rotation=20)
    ax[1].set_ylabel("throughput relative to reference"); ax[1].legend(); ax[1].set_title("Speed-up")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_lightweight.png"), dpi=140); plt.close(fig)

    with open(os.path.join(out, "results.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
