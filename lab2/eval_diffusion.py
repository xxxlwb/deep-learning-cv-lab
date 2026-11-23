"""
实验2 Task 4：生成结果评价
==========================
固定训练种子、采样种子和初始噪声，展示：
  1. 随机样本网格（16张）
  2. 加噪与去噪过程（单张图：t=0,50,100,200,300 加噪 + 完整去噪序列）
  3. 类别覆盖（10个类别各生成 N 张，KNN 分类判别类别）
  4. 训练集最近邻（L2 距离，每张生成图找最近的真实训练图）
  5. 失败样本（生成质量差的样本，不挑选成功样本）
  6. 采样时间记录

依赖：训练好的 `outputs_task3/ddpm_final.pth`
"""

import os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

# ── 路径设置：复用 diffusion_model.py 里的模型定义 ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── 超参数（需与训练时一致）────────────────────────────────────────────────
T          = 300
CHANNELS   = 64
SEED_TRAIN = 42        # 训练种子（固定，与 Task 3 一致）
SEED_SAMPLE = 0        # 采样种子（固定）
DATA_DIR   = "../data"
CKPT_PATH  = "outputs_task3/ddpm_final.pth"
OUT_DIR    = "outputs_task4"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ── 设备 ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(SEED_TRAIN)

# ── 噪声调度（必须与训练时完全相同）───────────────────────────────────────
betas      = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# ── 复制模型定义（避免循环 import argparse 副作用）──────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=128):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm1     = nn.GroupNorm(8, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.act       = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.conv_in  = nn.Conv2d(1, base_ch, 3, padding=1)
        self.down1    = ResBlock(base_ch,     base_ch*2, time_emb_dim)
        self.pool1    = nn.MaxPool2d(2)
        self.down2    = ResBlock(base_ch*2,   base_ch*4, time_emb_dim)
        self.pool2    = nn.MaxPool2d(2)
        self.mid      = ResBlock(base_ch*4,   base_ch*4, time_emb_dim)
        self.up1      = ResBlock(base_ch*8,          base_ch*2, time_emb_dim)
        self.up2      = ResBlock(base_ch*3,          base_ch,   time_emb_dim)
        self.conv_out = nn.Conv2d(base_ch, 1, 1)
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        s0  = self.conv_in(x)
        s1  = self.down1(s0, t_emb)
        s1p = self.pool1(s1)
        s2  = self.down2(s1p, t_emb)
        s2p = self.pool2(s2)
        h   = self.mid(s2p, t_emb)
        h   = F.interpolate(h, scale_factor=2, mode="nearest")
        h   = self.up1(torch.cat([h, s2], dim=1), t_emb)
        h   = F.interpolate(h, scale_factor=2, mode="nearest")
        h   = self.up2(torch.cat([h, s0], dim=1), t_emb)
        return self.conv_out(h)

# ── 采样函数 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def ddpm_sample(model, n=16, seed=SEED_SAMPLE, device=DEVICE, return_traj=False):
    """
    从纯噪声 T 步迭代去噪生成 n 张图。
    return_traj=True 时额外返回去噪轨迹（每隔 T//10 步保存一帧）。
    """
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=gen, device=device)

    traj = []
    save_steps = set(range(0, T, T // 10)) | {T - 1}

    for t_idx in range(T - 1, -1, -1):
        t_tensor   = torch.full((n,), t_idx, dtype=torch.long, device=device)
        pred_noise = model(x, t_tensor)

        beta_t  = betas[t_idx]
        alpha_t = alphas[t_idx]
        ab_t    = alpha_bars[t_idx]
        coef    = beta_t / (1 - ab_t).sqrt()
        mean    = (x - coef * pred_noise) / alpha_t.sqrt()

        if t_idx > 0:
            z = torch.randn_like(x)
            x = mean + beta_t.sqrt() * z
        else:
            x = mean

        if return_traj and t_idx in save_steps:
            traj.append((t_idx, x.clamp(-1, 1).cpu()))

    x = x.clamp(-1, 1)
    if return_traj:
        traj.reverse()   # 从低噪声到高噪声
        return x, traj
    return x

@torch.no_grad()
def q_sample_forward(x0, t_steps, noise=None):
    """前向加噪到指定步骤 t（返回 x_t）。"""
    results = []
    for t in t_steps:
        if noise is None:
            eps = torch.randn_like(x0)
        else:
            eps = noise
        ab  = alpha_bars[t]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        results.append(x_t.clamp(-1, 1).cpu())
    return results

# ── 图像工具 ────────────────────────────────────────────────────────────────
def to_np(t):
    """[-1,1] tensor → [0,1] numpy (H,W)"""
    return ((t.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)

def save_grid(imgs, path, title="", ncols=8):
    """imgs: list of (1,28,28) tensors in [-1,1]"""
    n     = len(imgs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.8))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(to_np(imgs[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

# ══════════════════════════════════════════════════════════════════════════════
# 加载模型
# ══════════════════════════════════════════════════════════════════════════════
print(f"Device: {DEVICE}")
print(f"Loading checkpoint: {CKPT_PATH}")
model = UNet(base_ch=CHANNELS).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ── 加载训练集（用于最近邻查找）──────────────────────────────────────────────
tf_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_ds   = datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=tf_norm)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)

print("Collecting training set features …")
train_imgs, train_labels = [], []
for x, y in train_loader:
    train_imgs.append(x)
    train_labels.append(y)
train_imgs   = torch.cat(train_imgs,   dim=0).numpy().reshape(len(train_ds), -1)   # (60k, 784)
train_labels = torch.cat(train_labels, dim=0).numpy()

# ══════════════════════════════════════════════════════════════════════════════
# 1. 随机样本网格（16 张，固定采样种子）
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Generating 16 random samples …")
t0 = time.time()
samples16 = ddpm_sample(model, n=16, seed=SEED_SAMPLE)
t_sample16 = time.time() - t0
print(f"    Sampling 16 images: {t_sample16:.1f}s")
save_grid([samples16[i] for i in range(16)],
          f"{OUT_DIR}/1_random_grid.png",
          title=f"Random Samples (seed={SEED_SAMPLE})", ncols=4)

# ══════════════════════════════════════════════════════════════════════════════
# 2. 加噪与去噪过程
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Forward noising process …")
# 取一张真实训练图
x0_real = torch.from_numpy(train_imgs[0].reshape(1, 1, 28, 28)).to(DEVICE)  # [-1,1] after norm

torch.manual_seed(SEED_SAMPLE)
noise_fixed = torch.randn_like(x0_real)

noise_steps = [0, 50, 100, 150, 200, 250, 299]
noised_imgs = q_sample_forward(x0_real, noise_steps, noise=noise_fixed)

# 去噪轨迹（单张图）
print("    Running reverse denoising trajectory …")
_, traj = ddpm_sample(model, n=1, seed=SEED_SAMPLE, return_traj=True)
denoise_imgs = [frame for _, frame in traj]      # T//10 帧，低噪→清晰

fig, axes = plt.subplots(2, max(len(noise_steps), len(denoise_imgs)),
                         figsize=(max(len(noise_steps), len(denoise_imgs)) * 2, 5))
for i, (step, img) in enumerate(zip(noise_steps, noised_imgs)):
    axes[0, i].imshow(to_np(img), cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"t={step}", fontsize=8); axes[0, i].axis("off")
for i, img in enumerate(denoise_imgs):
    axes[1, i].imshow(to_np(img), cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title(f"step {i}", fontsize=8); axes[1, i].axis("off")
for j in range(len(denoise_imgs), axes.shape[1]):
    axes[1, j].axis("off")
axes[0, 0].set_ylabel("Forward (noise ↑)", fontsize=8)
axes[1, 0].set_ylabel("Reverse (denoise →)", fontsize=8)
plt.suptitle("Noising & Denoising Process", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_noising_denoising.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/2_noising_denoising.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 类别覆盖（生成 100 张，KNN 分类，统计覆盖）
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Class coverage — generating 100 samples …")
t0 = time.time()
samples100 = ddpm_sample(model, n=100, seed=SEED_SAMPLE)
t_sample100 = time.time() - t0
print(f"    Sampling 100 images: {t_sample100:.1f}s")

# 用 KNN（在训练集上拟合）预测生成图的类别
print("    Fitting KNN on 10k training subset …")
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
knn.fit(train_imgs[:10000], train_labels[:10000])

gen_flat = samples100.cpu().numpy().reshape(100, -1)   # (100,784)  [-1,1]
# 训练集也是 [-1,1]，直接 L2
pred_labels = knn.predict(gen_flat)

counts = np.bincount(pred_labels, minlength=10)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(CLASS_NAMES, counts, color="steelblue")
ax.set_ylabel("Count (out of 100)")
ax.set_title("Class Coverage of 100 Generated Samples (KNN-predicted)")
ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=9)
for i, v in enumerate(counts):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_class_coverage.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/3_class_coverage.png")
print(f"    Covered classes: {(counts > 0).sum()}/10")
print(f"    Distribution: { {CLASS_NAMES[i]: int(counts[i]) for i in range(10)} }")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 训练集最近邻（每张生成图找 L2 最近的真实图）
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Nearest neighbors in training set …")
# 对比前 16 张生成图
gen16_flat = samples100.cpu().numpy()[:16].reshape(16, -1)
dists      = np.linalg.norm(train_imgs[:10000][None, :, :] - gen16_flat[:, None, :], axis=-1)  # (16, 10k)
nn_idx     = dists.argmin(axis=1)   # (16,)
nn_imgs    = train_imgs[nn_idx].reshape(16, 1, 28, 28)   # (16, 1, 28, 28) numpy

fig, axes = plt.subplots(2, 16, figsize=(32, 4))
for i in range(16):
    gen_img = to_np(torch.from_numpy(samples100[i].cpu().numpy()))
    nn_img  = ((nn_imgs[i, 0] + 1) / 2).clip(0, 1)
    axes[0, i].imshow(gen_img, cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[1, i].imshow(nn_img,  cmap="gray", vmin=0, vmax=1)
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Generated", fontsize=8)
axes[1, 0].set_ylabel("NN (train)", fontsize=8)
plt.suptitle("Generated vs Nearest Neighbor in Training Set", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_nearest_neighbor.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/4_nearest_neighbor.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 失败样本（质量差的生成图，不挑选成功样本）
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Failure samples (worst by pixel variance) …")
# 生成更多样本再从中挑质量差的（不挑质量好的）
t0 = time.time()
samples64 = ddpm_sample(model, n=64, seed=SEED_SAMPLE + 1)   # 不同 seed
t_64 = time.time() - t0

# 质量度量：像素方差低 = 生成了灰色噪声/模糊图 = 失败样本
flat64    = samples64.cpu().numpy().reshape(64, -1)
variances = flat64.var(axis=1)                 # (64,)
worst_idx = np.argsort(variances)[:16]         # 方差最小的 16 张
worst_imgs = [samples64[i] for i in worst_idx]
save_grid(worst_imgs,
          f"{OUT_DIR}/5_failure_samples.png",
          title="Failure Samples (lowest pixel variance, not cherry-picked)", ncols=4)

# ══════════════════════════════════════════════════════════════════════════════
# 6. 汇总采样时间
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("TASK 4 SUMMARY")
print("="*55)
print(f"Sampling 16  images  : {t_sample16:.2f}s  ({t_sample16/16*1000:.0f}ms/img)")
print(f"Sampling 100 images  : {t_sample100:.2f}s  ({t_sample100/100*1000:.0f}ms/img)")
print(f"Sampling 64  images  : {t_64:.2f}s  ({t_64/64*1000:.0f}ms/img)")
print(f"Class coverage       : {(counts>0).sum()}/10 classes covered")
print(f"Device               : {DEVICE}")
print("="*55)
print(f"\nAll outputs saved to {OUT_DIR}/")
