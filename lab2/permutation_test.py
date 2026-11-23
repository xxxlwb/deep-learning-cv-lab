# 实验2 Task 2：结构先验检验
# 用固定像素置换，分别在原图和置换图上训练 MLP 与 CNN，
# 验证"CNN 的性能优势是否依赖局部空间结构"。

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 随机种子 ──────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── 固定置换（全程共用） ──────────────────────────────────────────────────────
perm = np.random.default_rng(0).permutation(784)

# ── 输出目录 ──────────────────────────────────────────────────────────────────
os.makedirs("outputs_task2", exist_ok=True)

# ── 设备 ──────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 置换 transform ────────────────────────────────────────────────────────────
class PermutePixels:
    def __init__(self, perm):
        self.perm = perm

    def __call__(self, x):  # x: (1,28,28) tensor
        return x.view(-1)[self.perm].view(1, 28, 28)


# ── 数据加载（原图 / 置换图） ─────────────────────────────────────────────────
def get_loaders(permuted: bool, batch_size: int = 256):
    base_tf = transforms.ToTensor()
    if permuted:
        tf = transforms.Compose([base_tf, PermutePixels(perm)])
    else:
        tf = base_tf

    full_train = datasets.MNIST("../data", train=True,  download=True, transform=tf)
    test_ds    = datasets.MNIST("../data", train=False, download=True, transform=tf)

    # 官方60k：前50k训练，后10k验证（与实验1一致）
    train_ds = Subset(full_train, range(50000))
    val_ds   = Subset(full_train, range(50000, 60000))

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


# ── 模型定义 ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── 单 epoch 训练 / 评估 ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


# ── 完整训练流程 ──────────────────────────────────────────────────────────────
def run_experiment(name: str, model: nn.Module, train_loader, val_loader,
                   test_loader, epochs: int = 30):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_accs, val_accs = [], []
    best_val_acc = 0.0
    ckpt_path = f"outputs_task2/{name.replace(' ', '_')}_best.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{name}] Epoch {epoch:3d}/{epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

    elapsed = time.time() - t0

    # 加载最佳 checkpoint 做测试
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion)

    print(f"[{name}] Test Acc: {test_acc:.4f}  Time: {elapsed:.1f}s\n")
    return train_accs, val_accs, test_acc, elapsed


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    epochs = 30

    configs = [
        ("MLP + 原图",   MLP(), False),
        ("MLP + 置换图", MLP(), True),
        ("CNN + 原图",   CNN(), False),
        ("CNN + 置换图", CNN(), True),
    ]

    results = {}
    for name, model, permuted in configs:
        print(f"\n{'='*60}")
        print(f"  Running: {name}  (params={count_params(model):,})")
        print(f"{'='*60}")
        train_loader, val_loader, test_loader = get_loaders(permuted)
        tr_accs, vl_accs, test_acc, elapsed = run_experiment(
            name, model, train_loader, val_loader, test_loader, epochs
        )
        results[name] = {
            "train_accs": tr_accs,
            "val_accs":   vl_accs,
            "test_acc":   test_acc,
            "elapsed":    elapsed,
            "params":     count_params(model),
        }

    # ── 绘图 1：训练 / 验证曲线（4 条线） ─────────────────────────────────────
    colors = {"MLP + 原图": "#1f77b4", "MLP + 置换图": "#aec7e8",
              "CNN + 原图": "#d62728", "CNN + 置换图": "#f7b6d2"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, r in results.items():
        xs = range(1, epochs + 1)
        axes[0].plot(xs, r["train_accs"], label=name, color=colors[name])
        axes[1].plot(xs, r["val_accs"],   label=name, color=colors[name])

    for ax, title in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Lab2 Task2 — MLP vs CNN on Original vs Permuted MNIST", fontsize=12)
    fig.tight_layout()
    curve_path = "outputs_task2/task2_curves.png"
    fig.savefig(curve_path, dpi=150)
    print(f"Saved: {curve_path}")

    # ── 绘图 2：测试准确率柱状图 ───────────────────────────────────────────────
    names = list(results.keys())
    test_accs = [results[n]["test_acc"] for n in names]
    bar_colors = [colors[n] for n in names]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.bar(names, test_accs, color=bar_colors, edgecolor="black", width=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Test Accuracy: MLP vs CNN × Original vs Permuted MNIST")
    for bar, acc in zip(bars, test_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{acc:.4f}", ha="center", va="bottom", fontsize=9)
    ax2.tick_params(axis="x", labelsize=9)
    fig2.tight_layout()
    bar_path = "outputs_task2/task2_test_acc_bar.png"
    fig2.savefig(bar_path, dpi=150)
    print(f"Saved: {bar_path}")

    # ── 控制台汇总表 ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'配置':<18} {'参数量':>10} {'Test Acc':>10} {'训练时间(s)':>12}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<18} {r['params']:>10,} {r['test_acc']:>10.4f} {r['elapsed']:>12.1f}")
    print("="*70)

    # ── 结论提示 ──────────────────────────────────────────────────────────────
    cnn_orig = results["CNN + 原图"]["test_acc"]
    cnn_perm = results["CNN + 置换图"]["test_acc"]
    mlp_orig = results["MLP + 原图"]["test_acc"]
    mlp_perm = results["MLP + 置换图"]["test_acc"]
    print(f"\n结论分析：")
    print(f"  CNN 原图 vs 置换图 下降：{cnn_orig - cnn_perm:+.4f}")
    print(f"  MLP 原图 vs 置换图 下降：{mlp_orig - mlp_perm:+.4f}")
    if cnn_orig - cnn_perm > mlp_orig - mlp_perm + 0.005:
        print("  → CNN 在置换图上性能下降显著大于 MLP，说明 CNN 的优势依赖局部空间结构。")
    else:
        print("  → CNN 与 MLP 对置换的鲁棒性相近，请进一步分析。")


if __name__ == "__main__":
    main()
