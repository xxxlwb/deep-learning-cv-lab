"""
实验3 训练脚本：ViT 基线 / 类别不均衡处理 / 轻量化微调
==========================================================
用法（在 lab3/ 目录下）：
  # 1.1 基线 & 1.2 不均衡处理（从头训练）
  python train.py --name baseline     --imbalance none
  python train.py --name reweight     --imbalance reweight
  python train.py --name resample     --imbalance resample
  python train.py --name logit_adjust --imbalance logit_adjust
  # 1.3 轻量化（从参考模型初始化 + 蒸馏微调）
  python train.py --name dynamicvit --light dynamicvit --init outputs/<ref>/model.pth --imbalance <ref 的方法>
  python train.py --name prune      --light prune      --init outputs/<ref>/model.pth --imbalance <ref 的方法>

输出：outputs/<name>/{model.pth, history.json, metrics.json, curves.png, confusion.png}
"""

import argparse
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

from data import CLASSES, NUM_CLASSES, GPUData
from models import DynamicViT, ViT, load_model, prune_vit, save_model, taylor_importance

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--imbalance", default="none", choices=["none", "reweight", "resample", "logit_adjust"])
    p.add_argument("--light", default="none", choices=["none", "dynamicvit", "prune"])
    p.add_argument("--init", default=None, help="轻量化：参考模型 checkpoint（同时作为蒸馏教师）")
    p.add_argument("--epochs", type=int, default=None, help="默认：从头训练 200，轻量化微调 50")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=None, help="默认：从头训练 1e-3，轻量化微调 5e-4")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=1.0, help="logit adjustment 温度")
    p.add_argument("--keep_rate", type=float, default=0.7, help="DynamicViT 每级保留比例")
    p.add_argument("--prune_loc", type=int, nargs="+", default=[2, 4], help="DynamicViT 预测器插入位置")
    p.add_argument("--head_keep", type=float, default=0.5, help="剪枝：每层保留的注意力头比例")
    p.add_argument("--mlp_keep", type=float, default=0.5, help="剪枝：每层保留的 MLP 神经元比例")
    p.add_argument("--kd", type=float, default=0.5, help="蒸馏 KL 损失权重")
    p.add_argument("--out", default=os.path.join(HERE, "outputs"))
    a = p.parse_args()
    ft = a.light != "none"
    a.epochs = a.epochs or (50 if ft else 200)
    a.lr = a.lr or (5e-4 if ft else 1e-3)
    a.warmup = a.warmup if a.warmup is not None else (5 if ft else 10)
    if ft and not a.init:
        p.error("--light 需要 --init 指定参考模型")
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TaskLoss:
    """分类损失（含类别不均衡处理）。
    none         : 标准交叉熵
    reweight     : 类别权重 ∝ 1/n_c（归一化到均值 1）的加权交叉熵
    resample     : 损失不变，改为类别均衡采样（见 sample_weights）
    logit_adjust : 训练时 logits + τ·log π_c 再算交叉熵（Menon et al., ICLR 2021），推理时用原始 logits
    """

    def __init__(self, kind, counts, tau=1.0):
        self.kind = kind
        counts = counts.float()
        self.weight = None
        if kind == "reweight":
            w = 1.0 / counts
            self.weight = w / w.sum() * len(counts)
        self.log_prior = tau * torch.log(counts / counts.sum())

    def __call__(self, logits, y):
        logits = logits.float()
        if self.kind == "logit_adjust":
            logits = logits + self.log_prior
        return F.cross_entropy(logits, y, weight=self.weight)


@torch.no_grad()
def predict(model, data, split):
    model.eval()
    preds, labels = [], []
    for x, y in data.eval_batches(split):
        with autocast(x.device):
            preds.append(model(x).argmax(1))
        labels.append(y)
    return torch.cat(preds).cpu().numpy(), torch.cat(labels).cpu().numpy()


def metrics_from(pred, y):
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    recall = cm.diagonal() / cm.sum(1).clip(min=1)
    return dict(acc=float((pred == y).mean()), balanced_acc=float(recall.mean()),
                macro_f1=float(f1_score(y, pred, average="macro")),
                per_class_recall=recall.tolist(), confusion=cm.tolist())


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "pos_embed" in n or "cls_token" in n else decay).append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]


def plot_curves(hist, path, title):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_title("training loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train acc")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val acc")
    ax[1].plot(ep, [h["val_balanced_acc"] for h in hist], label="val balanced acc", ls="--")
    ax[1].set_xlabel("epoch"); ax[1].set_title("accuracy"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(cm, path, title):
    cm = np.array(cm, dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)

    data = GPUData(device, seed=42)
    counts = data.class_counts
    task_loss = TaskLoss(args.imbalance, counts.to(device), args.tau)
    sample_w = (1.0 / counts.float())[data.splits["train"][1]] if args.imbalance == "resample" else None

    teacher, extra = None, {}
    if args.light == "none":
        model = ViT().to(device)
    else:
        teacher, _ = load_model(args.init, device)
        teacher.eval().requires_grad_(False)
        base, _ = load_model(args.init, device)
        if args.light == "dynamicvit":
            model = DynamicViT(base, prune_loc=args.prune_loc, keep_rate=args.keep_rate).to(device)
        else:
            g = torch.Generator(device=device).manual_seed(args.seed)
            batches = [b for _, b in zip(range(20), data.train_batches(args.bs, g))]
            n_before = sum(p.numel() for p in base.parameters())
            h_imp, m_imp = taylor_importance(base, batches, task_loss)
            model = prune_vit(base, h_imp, m_imp, args.head_keep, args.mlp_keep)
            extra = dict(params_before_prune=n_before,
                         head_importance=[h.tolist() for h in h_imp],
                         kept_heads=model.cfg["layer_heads"], kept_mlp=model.cfg["layer_mlp"])
            # 剪枝后、微调前的精度
            pred, y = predict(model, data, "test")
            extra["test_acc_after_prune_before_ft"] = metrics_from(pred, y)["acc"]
            tqdm.write(f"pruned: params {n_before:,} -> {sum(p.numel() for p in model.parameters()):,}, "
                       f"test acc before finetune = {extra['test_acc_after_prune_before_ft']:.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = len(data.splits["train"][1]) // args.bs
    total_steps = steps_per_epoch * args.epochs
    warm_steps = steps_per_epoch * args.warmup

    def lr_at(step):
        if step < warm_steps:
            return args.lr * (step + 1) / warm_steps
        prog = (step - warm_steps) / max(1, total_steps - warm_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    print(f"[{args.name}] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else device.type}) "
          f"params={n_params:,} imbalance={args.imbalance} light={args.light} epochs={args.epochs} "
          f"bs={args.bs} lr={args.lr} seed={args.seed}")
    print(f"train per-class counts: {counts.tolist()}")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    hist, step = [], 0
    t_start = time.time()
    ebar = tqdm(range(1, args.epochs + 1), desc=args.name, unit="ep", dynamic_ncols=True)
    for epoch in ebar:
        model.train()
        t0 = time.time()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        bbar = tqdm(data.train_batches(args.bs, gen, sample_w), total=steps_per_epoch,
                    desc=f"epoch {epoch}", leave=False, dynamic_ncols=True, mininterval=1.0)
        for x, y in bbar:
            for g_ in opt.param_groups:
                g_["lr"] = lr_at(step)
            with autocast(device):
                if args.light == "dynamicvit":
                    logits, tokens, mask, decisions = model(x)
                else:
                    logits = model(x)
                loss = task_loss(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, t_tokens = teacher(x, return_tokens=True)
                    loss = loss + args.kd * F.kl_div(F.log_softmax(logits.float(), -1),
                                                     F.log_softmax(t_logits.float(), -1),
                                                     reduction="batchmean", log_target=True)
                    if args.light == "dynamicvit":
                        ratio = sum(((d.float().mean(1) - args.keep_rate ** (k + 1)) ** 2).mean()
                                    for k, d in enumerate(decisions)) / len(decisions)
                        m = mask.float()
                        tok = (((tokens.float() - t_tokens.float()) ** 2) * m).sum() / (m.sum() * tokens.shape[-1] + 1e-6)
                        loss = loss + 2.0 * ratio + 0.5 * tok
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tot_loss += loss.item() * len(y)
            tot_correct += (logits.argmax(1) == y).sum().item()
            tot_n += len(y)
            bbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_at(step):.1e}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0
        vm = metrics_from(*predict(model, data, "val"))
        h = dict(epoch=epoch, train_loss=tot_loss / tot_n, train_acc=tot_correct / tot_n,
                 val_acc=vm["acc"], val_balanced_acc=vm["balanced_acc"], epoch_time=ep_time)
        hist.append(h)
        ebar.set_postfix(loss=f"{h['train_loss']:.3f}", train=f"{h['train_acc']:.3f}",
                         val=f"{h['val_acc']:.3f}", s_ep=f"{ep_time:.1f}")
        tqdm.write(f"epoch {epoch:3d} | loss {h['train_loss']:.4f} | train acc {h['train_acc']:.4f} | "
                   f"val acc {h['val_acc']:.4f} | val bal acc {h['val_balanced_acc']:.4f} | {ep_time:.1f}s")
    train_time = time.time() - t_start

    tm = metrics_from(*predict(model, data, "test"))
    save_model(model, os.path.join(out_dir, "model.pth"), args=vars(args))
    summary = dict(name=args.name, imbalance=args.imbalance, light=args.light, init=args.init,
                   params=n_params, epochs=args.epochs, batch_size=args.bs, lr=args.lr, seed=args.seed,
                   device=torch.cuda.get_device_name(0) if device.type == "cuda" else device.type,
                   train_time_s=train_time, avg_epoch_time_s=float(np.mean([h["epoch_time"] for h in hist])),
                   test=tm, **extra)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    plot_curves(hist, os.path.join(out_dir, "curves.png"), args.name)
    plot_confusion(tm["confusion"], os.path.join(out_dir, "confusion.png"),
                   f"{args.name}  test acc={tm['acc']:.4f}")
    print(f"[{args.name}] done in {train_time:.1f}s | test acc {tm['acc']:.4f} | "
          f"balanced acc {tm['balanced_acc']:.4f} | macro F1 {tm['macro_f1']:.4f}")
    print("per-class recall: " + ", ".join(f"{c}={r:.3f}" for c, r in zip(CLASSES, tm["per_class_recall"])))


if __name__ == "__main__":
    main()
