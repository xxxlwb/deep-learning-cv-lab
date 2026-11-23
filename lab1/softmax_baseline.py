"""
Lab 1 — Task 2: Linear Softmax Baseline
========================================
在 Fashion-MNIST 上训练线性 Softmax 分类器。
记录：训练与验证曲线、测试准确率、混淆矩阵、逐类召回率、典型错误可视化。

数据划分：官方60k训练集固定切分为 50k训练 + 10k验证，官方10k测试集只用于评价。
随机种子：42
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
OUT_DIR = "outputs_lab1"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────── data ──────────────────────────────────────
def get_loaders(batch_size=256, data_dir="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=tf)
    test_set   = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(Subset(full_train, range(50_000)),
                              batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(full_train, range(50_000, 60_000)),
                              batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

# ──────────────────────────────── model ─────────────────────────────────────
class LinearSoftmax(nn.Module):
    """Single linear layer: 784 → 10."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# ──────────────────────────── training utils ────────────────────────────────
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
        out   = model(x); loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        preds       = out.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# ──────────────────────────────── plots ─────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="train"); axes[0].plot(epochs, val_losses, "--", label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.4)
    axes[1].plot(epochs, train_accs, label="train");  axes[1].plot(epochs, val_accs,  "--", label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.4)
    plt.suptitle(f"{name} — Fashion-MNIST", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/softmax_curves.png", dpi=150); plt.close()
    print(f"Saved {OUT_DIR}/softmax_curves.png")

def plot_confusion(preds, labels, name="LinearSoftmax"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/softmax_confusion.png", dpi=150); plt.close()
    print(f"Saved {OUT_DIR}/softmax_confusion.png")

def plot_typical_errors(model, test_loader, preds, labels, n=16):
    """Visualize typical misclassified samples."""
    all_imgs = torch.cat([x for x, _ in test_loader], dim=0)   # (10000,1,28,28)
    err_idx  = np.where(preds != labels)[0]
    # group errors by (true, pred) pair and pick one representative each
    pairs = {}
    for i in err_idx:
        key = (labels[i], preds[i])
        pairs.setdefault(key, []).append(i)
    # pick top-n most common error pairs
    top_pairs = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:n]

    ncols = 4; nrows = (len(top_pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.8))
    axes = np.array(axes).flatten()
    for ax_i, ((true_cls, pred_cls), indices) in enumerate(top_pairs):
        idx = indices[0]
        img = all_imgs[idx, 0].numpy()
        axes[ax_i].imshow(img, cmap="gray")
        axes[ax_i].set_title(
            f"True: {CLASS_NAMES[true_cls]}\nPred: {CLASS_NAMES[pred_cls]}\n({len(indices)} err)",
            fontsize=7)
        axes[ax_i].axis("off")
    for ax in axes[len(top_pairs):]:
        ax.axis("off")
    plt.suptitle("Typical Errors — Linear Softmax", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/softmax_errors.png", dpi=150); plt.close()
    print(f"Saved {OUT_DIR}/softmax_errors.png")

def plot_weights(model):
    """Visualise learnt weight matrix as class templates (784→10)."""
    W = model.fc.weight.detach().cpu().numpy()   # (10, 784)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        template = W[i].reshape(28, 28)
        vmax = max(abs(template.min()), abs(template.max()))
        ax.imshow(template, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(CLASS_NAMES[i], fontsize=8); ax.axis("off")
    plt.suptitle("Linear Softmax — Learnt Class Templates", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/softmax_weights.png", dpi=150); plt.close()
    print(f"Saved {OUT_DIR}/softmax_weights.png")

# ─────────────────────────────────── main ───────────────────────────────────
if __name__ == "__main__":
    EPOCHS = 30
    LR     = 0.1          # SGD with moderate lr works well for linear model
    WD     = 1e-4

    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_loaders()

    model     = LinearSoftmax().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # SGD + weight decay (L2 regularisation) is the standard for linear softmax
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        tl, ta       = train_epoch(model, train_loader, criterion, optimizer)
        vl, va, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);   val_accs.append(va)
        if ep % 5 == 0:
            print(f"Ep {ep:>3}/{EPOCHS} | train {ta:.4f} | val {va:.4f} | lr {scheduler.get_last_lr()[0]:.5f}")

    elapsed = time.time() - t0
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Training time : {elapsed:.1f}s")
    print(f"Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    print("\n--- Per-class Report ---")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # Plots
    plot_curves(train_losses, val_losses, train_accs, val_accs, "Linear Softmax")
    plot_confusion(preds, labels)
    plot_typical_errors(model, test_loader, preds, labels)
    plot_weights(model)

    torch.save(model.state_dict(), f"{OUT_DIR}/softmax.pth")
    print(f"\nCheckpoint saved to {OUT_DIR}/softmax.pth")
