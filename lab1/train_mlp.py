"""
Lab 1 — Task 4 & 5: Controlled Comparison + Training Diagnostics
=================================================================
Task 4 受控比较：
  - Model A: Linear Softmax     (784 → 10)
  - Model B: Two-layer Linear   (784 → 512 → 10, no activation)
  - Model C: MLP with ReLU      (784 → 512 → ReLU → 10)
  模型 B/C 使用相同隐藏层宽度 512，其余训练条件尽量一致。

Task 5 训练诊断：
  选择一个实际出现的训练问题（过拟合），先提出假设，
  再只改变一个因素（Dropout 正则化）进行验证对比。

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

# ───────────────────────────────── models ───────────────────────────────────
INPUT_DIM   = 784
HIDDEN_DIM  = 512
NUM_CLASSES = 10

class ModelA_LinearSoftmax(nn.Module):
    """Model A: Single linear layer (baseline)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(INPUT_DIM, NUM_CLASSES)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class ModelB_TwoLayerLinear(nn.Module):
    """Model B: Two linear layers, NO activation — still a linear classifier."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class ModelC_MLP(nn.Module):
    """Model C: Single hidden layer MLP with ReLU nonlinearity."""
    def __init__(self, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(HIDDEN_DIM, NUM_CLASSES))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# ──────────────────────────── training utils ────────────────────────────────
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x); loss = criterion(out, y)
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
        out  = model(x); loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        preds       = out.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, val_loader,
                epochs=30, lr=1e-3, name="model"):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc"]}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta       = train_epoch(model, train_loader, criterion, optimizer)
        vl, va, _, _ = evaluate(model, val_loader, criterion)
        for k, v in zip(history, [tl, vl, ta, va]):
            history[k].append(v)
        if ep % 5 == 0 or ep == 1:
            print(f"  [{name}] Ep {ep:>3}/{epochs} | train {ta:.4f} | val {va:.4f}")
    elapsed = time.time() - t0
    print(f"  [{name}] Done — {elapsed:.1f}s | params {count_params(model):,}")
    return history, elapsed

# ──────────────────────────────── plots ─────────────────────────────────────
COLORS = ["steelblue", "darkorange", "seagreen", "crimson"]

def plot_comparison_curves(histories, names, tag="comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for h, n, c in zip(histories, names, COLORS):
        axes[0].plot(h["train_loss"], color=c, label=f"{n} train")
        axes[0].plot(h["val_loss"],   color=c, linestyle="--", label=f"{n} val")
        axes[1].plot(h["train_acc"],  color=c, label=f"{n} train")
        axes[1].plot(h["val_acc"],    color=c, linestyle="--", label=f"{n} val")
    for ax, title in zip(axes, ["Cross-Entropy Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_title(title)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)
    plt.suptitle("Controlled Comparison — Fashion-MNIST", fontsize=12)
    plt.tight_layout()
    out = f"{OUT_DIR}/{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

def plot_confusion(preds, labels, name, tag):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    out = f"{OUT_DIR}/cm_{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

def plot_summary_bar(names, test_accs, times, params):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = range(len(names))

    axes[0].bar(x, test_accs, color=COLORS[:len(names)])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15, fontsize=8)
    axes[0].set_ylabel("Test Accuracy"); axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0.7, 1.0)
    for i, v in enumerate(test_accs):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    axes[1].bar(x, times, color=COLORS[:len(names)])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15, fontsize=8)
    axes[1].set_ylabel("Training Time (s)"); axes[1].set_title("Training Time")
    for i, v in enumerate(times):
        axes[1].text(i, v * 1.01, f"{v:.0f}s", ha="center", fontsize=8)

    axes[2].bar(x, [p / 1e3 for p in params], color=COLORS[:len(names)])
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=15, fontsize=8)
    axes[2].set_ylabel("Parameters (K)"); axes[2].set_title("Parameter Count")
    for i, v in enumerate(params):
        axes[2].text(i, v / 1e3 * 1.01, f"{v/1e3:.1f}K", ha="center", fontsize=8)

    plt.suptitle("Model Comparison Summary", fontsize=12)
    plt.tight_layout()
    out = f"{OUT_DIR}/summary_bar.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

def plot_diagnostic(h_no_drop, h_drop, label_no="MLP (no dropout)", label_yes="MLP + Dropout(0.3)"):
    """Task 5: overfitting diagnostic — compare train/val gap before & after fix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(h_no_drop["train_acc"]) + 1)
    for h, lbl, c, ls in [
        (h_no_drop, label_no,  "steelblue", "-"),
        (h_drop,    label_yes, "tomato",    "-"),
    ]:
        axes[0].plot(ep, h["train_loss"], color=c, linestyle=ls,  label=f"{lbl} train")
        axes[0].plot(ep, h["val_loss"],   color=c, linestyle="--", label=f"{lbl} val")
        axes[1].plot(ep, h["train_acc"],  color=c, linestyle=ls,  label=f"{lbl} train")
        axes[1].plot(ep, h["val_acc"],    color=c, linestyle="--", label=f"{lbl} val")
    for ax, title in zip(axes, ["Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.suptitle("Task 5: Training Diagnostic — Overfitting & Dropout Fix", fontsize=11)
    plt.tight_layout()
    out = f"{OUT_DIR}/diagnostic_dropout.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

# ─────────────────────────────────── main ───────────────────────────────────
if __name__ == "__main__":
    EPOCHS = 30
    LR     = 1e-3

    print(f"Device: {DEVICE}\n")
    train_loader, val_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    # ════════════════════════════════════════════
    #  Task 4 — Controlled Comparison
    # ════════════════════════════════════════════
    print("=" * 55)
    print("TASK 4: Controlled Comparison")
    print("=" * 55)

    models_cfg = [
        ("A_LinearSoftmax",  ModelA_LinearSoftmax()),
        ("B_TwoLayerLinear", ModelB_TwoLayerLinear()),
        ("C_MLP_ReLU",       ModelC_MLP(dropout=0.0)),
    ]

    histories, elapsed_times, test_accs, all_preds_list, labels_ref = [], [], [], [], None

    for name, model in models_cfg:
        print(f"\n--- {name} | params={count_params(model):,} ---")
        h, t = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, name=name)
        histories.append(h)
        elapsed_times.append(t)

        _, test_acc, preds, labels = evaluate(model, test_loader, criterion)
        test_accs.append(test_acc)
        all_preds_list.append(preds)
        labels_ref = labels

        print(f"  [{name}] Test Acc: {test_acc:.4f}")
        print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
        plot_confusion(preds, labels, name, name)
        torch.save(model.state_dict(), f"{OUT_DIR}/{name}.pth")

    # Summary plots
    names_short = ["A: Linear\nSoftmax", "B: 2-layer\nLinear", "C: MLP\nReLU"]
    plot_comparison_curves(histories, [n for n, _ in models_cfg])
    plot_summary_bar(names_short, test_accs, elapsed_times,
                     [count_params(m) for _, m in models_cfg])

    print("\n" + "=" * 55)
    print("TASK 4 SUMMARY")
    print("=" * 55)
    print(f"{'Model':<22} {'Params':>8} {'TestAcc':>8} {'Time(s)':>8}")
    print("-" * 55)
    for (name, model), acc, t in zip(models_cfg, test_accs, elapsed_times):
        print(f"{name:<22} {count_params(model):>8,} {acc:>8.4f} {t:>8.1f}")

    # ════════════════════════════════════════════
    #  Task 5 — Training Diagnostic
    # ════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("TASK 5: Training Diagnostic — Overfitting")
    print("=" * 55)
    print("""
Hypothesis:
  Model C (MLP ReLU) shows a train-val accuracy gap that grows
  with more epochs, indicating overfitting to the 50k training set.
  We hypothesize that adding Dropout(p=0.3) after the hidden layer
  will reduce this gap and improve validation/test accuracy.

Experiment:
  - Change ONLY: add Dropout(p=0.3) to Model C.
  - All other hyperparameters (lr, epochs, hidden dim, optimiser) unchanged.
""")

    # Already have h_no_drop from Model C above
    h_no_drop = histories[2]   # Model C without dropout

    # Retrain with dropout (re-seed for fair comparison)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model_drop = ModelC_MLP(dropout=0.3)
    print("Training MLP + Dropout(0.3) …")
    h_drop, t_drop = train_model(model_drop, train_loader, val_loader,
                                  epochs=EPOCHS, lr=LR, name="MLP+Dropout")
    _, test_acc_drop, preds_drop, _ = evaluate(model_drop, test_loader, criterion)
    print(f"\nMLP (no dropout) Test Acc : {test_accs[2]:.4f}")
    print(f"MLP + Dropout    Test Acc : {test_acc_drop:.4f}")

    # Compute train-val gap at final epoch
    gap_no   = h_no_drop["train_acc"][-1] - h_no_drop["val_acc"][-1]
    gap_drop = h_drop["train_acc"][-1]    - h_drop["val_acc"][-1]
    print(f"\nTrain-Val accuracy gap:")
    print(f"  No dropout : {gap_no:.4f}")
    print(f"  + Dropout  : {gap_drop:.4f}  (smaller gap = less overfitting)")

    plot_diagnostic(h_no_drop, h_drop)
    plot_confusion(preds_drop, labels_ref, "MLP+Dropout", "mlp_dropout")
    torch.save(model_drop.state_dict(), f"{OUT_DIR}/mlp_dropout.pth")

    print(f"\nAll outputs saved to {OUT_DIR}/")
