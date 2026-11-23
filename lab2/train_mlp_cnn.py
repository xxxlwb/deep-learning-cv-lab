"""
Lab 2 — Task 1: MLP vs CNN on Fashion-MNIST
=============================================
沿用实验1数据划分（50k train / 10k val / 10k test）和MLP基线，
在相同训练预算下比较：
  - MLP (单隐层 ReLU, 512 hidden)
  - CNN (2 组 Conv→ReLU→MaxPool + 全连接分类头)

输出：
  - 各层输出尺寸和参数量
  - 参数量对比、分类准确率、训练时间、混淆矩阵（错误类型）
"""

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────── reproducibility ────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
OUT_DIR     = "outputs_task1"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────── data ──────────────────────────────────────
def get_loaders(batch_size=256, data_dir="../data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=tf)
    test_set   = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(Subset(full_train, range(50_000)), batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(full_train, range(50_000, 60_000)), batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

# ───────────────────────────────── models ───────────────────────────────────
class MLP(nn.Module):
    """Baseline MLP from Lab 1 (single hidden layer, ReLU)."""
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )
    def forward(self, x): return self.net(x)


class CNN(nn.Module):
    """
    Two groups of Conv → ReLU → MaxPool.

    Group 1: Conv(1→32, 3×3, pad=1) → ReLU → MaxPool(2×2)
             Output: 32 × 14 × 14
    Group 2: Conv(32→64, 3×3, pad=1) → ReLU → MaxPool(2×2)
             Output: 64 × 7 × 7
    FC head:  64*7*7 → 256 → ReLU → 10
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # --- Group 1 ---
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 32×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 32×14×14
            # --- Group 2 ---
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 64×7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────── layer-by-layer summary ────────────────────────────
def print_layer_summary(model, input_shape=(1, 1, 28, 28)):
    """Print output shape and param count for each named layer."""
    print(f"\n{'Layer':<35} {'Output Shape':<25} {'Params':>10}")
    print("-" * 72)
    x = torch.zeros(*input_shape)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d,
                                nn.ReLU, nn.Flatten, nn.BatchNorm2d)):
            x = module(x)
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:<35} {str(tuple(x.shape)):<25} {params:>10,}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 72)
    print(f"{'Total trainable parameters':<35} {'':25} {total:>10,}\n")

# ───────────────────────────── training ─────────────────────────────────────
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        preds       = out.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, val_loader,
                epochs=20, lr=1e-3, name="model"):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc"]}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta       = train_epoch(model, train_loader, criterion, optimizer)
        vl, va, _, _ = evaluate(model, val_loader, criterion)
        for k, v in zip(history, [tl, vl, ta, va]):
            history[k].append(v)
        if ep % 5 == 0:
            print(f"[{name}] Ep {ep:>3}/{epochs} | train {ta:.4f} | val {va:.4f}")
    elapsed = time.time() - t0
    print(f"[{name}] Training time: {elapsed:.1f}s  | params: {count_params(model):,}")
    return history, elapsed

# ───────────────────────────────── plots ────────────────────────────────────
def plot_curves(histories, names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["steelblue", "tomato"]
    for h, n, c in zip(histories, names, colors):
        axes[0].plot(h["train_loss"], color=c, label=f"{n} train")
        axes[0].plot(h["val_loss"],   color=c, linestyle="--", label=f"{n} val")
        axes[1].plot(h["train_acc"],  color=c, label=f"{n} train")
        axes[1].plot(h["val_acc"],    color=c, linestyle="--", label=f"{n} val")
    for ax, title in zip(axes, ["Cross-Entropy Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    plt.suptitle("MLP vs CNN — Fashion-MNIST", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/curves_mlp_vs_cnn.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/curves_mlp_vs_cnn.png")

def plot_confusion(preds, labels, name, tag):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cm_{tag}.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/cm_{tag}.png")

def plot_error_comparison(preds_mlp, preds_cnn, labels, test_loader):
    """Show sample images where one model errs but the other is correct."""
    # Collect raw images from test loader
    all_imgs = []
    for x, _ in test_loader:
        all_imgs.append(x)
    all_imgs = torch.cat(all_imgs, dim=0)   # (10000,1,28,28)

    err_mlp  = set(np.where(preds_mlp != labels)[0])
    err_cnn  = set(np.where(preds_cnn != labels)[0])
    # CNN correct but MLP wrong
    mlp_only = sorted(err_mlp - err_cnn)[:8]
    # MLP correct but CNN wrong
    cnn_only = sorted(err_cnn - err_mlp)[:8]

    def _show_row(indices, row_title, ax_row):
        for col, idx in enumerate(indices[:8]):
            img = all_imgs[idx, 0].numpy()
            ax_row[col].imshow(img, cmap="gray")
            ax_row[col].set_title(
                f"True:{CLASS_NAMES[labels[idx]][:4]}\n"
                f"MLP:{CLASS_NAMES[preds_mlp[idx]][:4]}\n"
                f"CNN:{CLASS_NAMES[preds_cnn[idx]][:4]}",
                fontsize=5)
            ax_row[col].axis("off")
        while col < 7:
            col += 1; ax_row[col].axis("off")

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    _show_row(mlp_only, "CNN✓ MLP✗", axes[0])
    _show_row(cnn_only, "MLP✓ CNN✗", axes[1])
    axes[0][0].set_ylabel("CNN✓ MLP✗", fontsize=8)
    axes[1][0].set_ylabel("MLP✓ CNN✗", fontsize=8)
    plt.suptitle("Error Comparison (CNN correct vs MLP correct)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/error_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/error_comparison.png")

# ─────────────────────────────────── main ───────────────────────────────────
if __name__ == "__main__":
    EPOCHS = 20

    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_loaders()

    # ── CNN layer summary ──
    cnn_model = CNN()
    print("\n=== CNN Layer Summary ===")
    print_layer_summary(cnn_model)

    mlp_model = MLP()
    print("=== MLP Layer Summary ===")
    print_layer_summary(mlp_model, input_shape=(1, 784))

    # ── Train ──
    print("\n" + "="*50)
    print("Training MLP …")
    h_mlp, t_mlp = train_model(mlp_model, train_loader, val_loader, epochs=EPOCHS, name="MLP")

    print("\n" + "="*50)
    print("Training CNN …")
    h_cnn, t_cnn = train_model(cnn_model, train_loader, val_loader, epochs=EPOCHS, name="CNN")

    # ── Evaluate on test ──
    criterion = nn.CrossEntropyLoss()
    _, acc_mlp, preds_mlp, labels = evaluate(mlp_model, test_loader, criterion)
    _, acc_cnn, preds_cnn, _      = evaluate(cnn_model,  test_loader, criterion)

    # ── Summary table ──
    print("\n" + "="*55)
    print(f"{'Model':<18} {'Params':>10} {'TestAcc':>10} {'Time(s)':>10}")
    print("-"*55)
    print(f"{'MLP':<18} {count_params(mlp_model):>10,} {acc_mlp:>10.4f} {t_mlp:>10.1f}")
    print(f"{'CNN':<18} {count_params(cnn_model):>10,} {acc_cnn:>10.4f} {t_cnn:>10.1f}")
    print("="*55)

    # ── Per-class report ──
    print("\n--- MLP per-class ---")
    print(classification_report(labels, preds_mlp, target_names=CLASS_NAMES))
    print("\n--- CNN per-class ---")
    print(classification_report(labels, preds_cnn, target_names=CLASS_NAMES))

    # ── Save plots ──
    plot_curves([h_mlp, h_cnn], ["MLP", "CNN"])
    plot_confusion(preds_mlp, labels, "MLP", "mlp")
    plot_confusion(preds_cnn, labels, "CNN", "cnn")
    plot_error_comparison(preds_mlp, preds_cnn, labels, test_loader)

    # ── Save checkpoints ──
    torch.save(mlp_model.state_dict(), f"{OUT_DIR}/mlp.pth")
    torch.save(cnn_model.state_dict(), f"{OUT_DIR}/cnn.pth")
    print(f"\nCheckpoints saved to {OUT_DIR}/")
