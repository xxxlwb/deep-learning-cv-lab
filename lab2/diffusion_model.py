"""
实验2 Task 3：扩散生成模型（DDPM）
针对 Fashion-MNIST 实现简化 DDPM，包含前向加噪、时间步条件去噪器、噪声预测训练与完整采样。
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ─── 超参数（可命令行覆盖）─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--T",          type=int,   default=300)
parser.add_argument("--epochs",     type=int,   default=30)
parser.add_argument("--batch_size", type=int,   default=256)
parser.add_argument("--lr",         type=float, default=2e-4)
parser.add_argument("--channels",   type=int,   default=64)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--data_dir",   type=str,   default="../data")
parser.add_argument("--out_dir",    type=str,   default="outputs_task3")
args = parser.parse_args()

T          = args.T
EPOCHS     = args.epochs
BATCH_SIZE = args.batch_size
LR         = args.lr
CHANNELS   = args.channels
SEED       = args.seed
DATA_DIR   = args.data_dir
OUT_DIR    = args.out_dir

# ─── 设备 ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 噪声调度 ──────────────────────────────────────────────────────────────
betas      = torch.linspace(1e-4, 0.02, T)                  # (T,)
alphas     = 1.0 - betas                                     # (T,)
alpha_bars = torch.cumprod(alphas, dim=0)                    # ᾱ_t, (T,)

# 预计算采样时需要的量，全部移到设备
betas      = betas.to(DEVICE)
alphas     = alphas.to(DEVICE)
alpha_bars = alpha_bars.to(DEVICE)


# ─── 时间嵌入 ──────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None, :]          # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb


# ─── ResBlock ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = 128):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm1  = nn.GroupNorm(8, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.act    = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        # 时间嵌入投影后加到空间特征
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ─── 简化 UNet ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, base_ch: int = 64, time_emb_dim: int = 128):
        super().__init__()
        # 时间嵌入
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Encoder
        self.conv_in = nn.Conv2d(1, base_ch, 3, padding=1)                 # (B,64,28,28)
        self.down1   = ResBlock(base_ch,       base_ch * 2, time_emb_dim)  # (B,128,28,28)
        self.pool1   = nn.MaxPool2d(2)                                      # (B,128,14,14)
        self.down2   = ResBlock(base_ch * 2,   base_ch * 4, time_emb_dim)  # (B,256,14,14)
        self.pool2   = nn.MaxPool2d(2)                                      # (B,256,7,7)

        # Bottleneck
        self.mid     = ResBlock(base_ch * 4,   base_ch * 4, time_emb_dim)  # (B,256,7,7)

        # Decoder
        # up1: cat(mid_out=base_ch*4, s2=base_ch*4) → base_ch*4 + base_ch*4 = base_ch*8
        self.up1     = ResBlock(base_ch * 8,            base_ch * 2, time_emb_dim)
        # up2: cat(up1_out=base_ch*2, s0=base_ch)   → base_ch*2 + base_ch   = base_ch*3
        self.up2     = ResBlock(base_ch * 3,            base_ch,     time_emb_dim)

        self.conv_out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, 28, 28)
        t : (B,) long
        returns: (B, 1, 28, 28) 预测噪声
        """
        t_emb = self.time_emb(t)                             # (B, 128)

        # Encoder
        s0 = self.conv_in(x)                                 # (B,64,28,28)
        s1 = self.down1(s0, t_emb)                           # (B,128,28,28)
        s1p = self.pool1(s1)                                 # (B,128,14,14)
        s2 = self.down2(s1p, t_emb)                          # (B,256,14,14)
        s2p = self.pool2(s2)                                 # (B,256,7,7)

        # Bottleneck
        h = self.mid(s2p, t_emb)                             # (B,256,7,7)

        # Decoder up1: 7→14, skip from down2 (s2)
        h = F.interpolate(h, scale_factor=2, mode="nearest") # (B,256,14,14)
        h = torch.cat([h, s2], dim=1)                        # (B,512,14,14)  256+256
        h = self.up1(h, t_emb)                               # (B,128,14,14)

        # Decoder up2: 14→28, skip from conv_in (s0)
        h = F.interpolate(h, scale_factor=2, mode="nearest") # (B,128,28,28)
        h = torch.cat([h, s0], dim=1)                        # (B,192,28,28)  128+64
        h = self.up2(h, t_emb)                               # (B,64,28,28)

        return self.conv_out(h)                              # (B,1,28,28)


# ─── 前向加噪（闭合形式）──────────────────────────────────────────────────
def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    x0 : (B,1,28,28)  原始图像 [-1,1]
    t  : (B,)  时间步索引 [0, T-1]
    noise : (B,1,28,28) ε ~ N(0,I)
    returns x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε
    """
    ab = alpha_bars[t][:, None, None, None]                  # (B,1,1,1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


# ─── DDPM 单步反向更新 ─────────────────────────────────────────────────────
@torch.no_grad()
def ddpm_step(x_t: torch.Tensor, t: int, pred_noise: torch.Tensor) -> torch.Tensor:
    """
    DDPM 单步去噪：
      x_{t-1} = 1/sqrt(α_t) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_pred) + σ_t * z
    """
    beta_t  = betas[t]
    alpha_t = alphas[t]
    ab_t    = alpha_bars[t]

    # 均值
    coef    = beta_t / (1 - ab_t).sqrt()
    mean    = (x_t - coef * pred_noise) / alpha_t.sqrt()

    if t > 0:
        # 方差：σ_t^2 = β_t（简化 DDPM，posterior variance = β_t）
        sigma_t = beta_t.sqrt()
        z       = torch.randn_like(x_t)
        return mean + sigma_t * z
    return mean


# ─── 完整采样（T 步迭代去噪）─────────────────────────────────────────────
@torch.no_grad()
def ddpm_sample(model: nn.Module, n: int = 16, seed: int = 0, device=DEVICE) -> torch.Tensor:
    """从纯噪声 x_T 经 T 步反向去噪生成 n 张图像。"""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x = torch.randn(n, 1, 28, 28, generator=gen, device=device)  # x_T ~ N(0,I)

    for t_idx in range(T - 1, -1, -1):
        t_tensor = torch.full((n,), t_idx, dtype=torch.long, device=device)
        pred_noise = model(x, t_tensor)
        x = ddpm_step(x, t_idx, pred_noise)

    return x.clamp(-1, 1)


# ─── 保存采样图像 ──────────────────────────────────────────────────────────
def save_samples(imgs: torch.Tensor, path: str):
    """imgs: (N,1,28,28) in [-1,1]"""
    imgs = (imgs + 1) / 2                                    # [0,1]
    n    = imgs.shape[0]
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ─── 主程序 ───────────────────────────────────────────────────────────────
def main():
    # ── 数据加载 ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    n_train = len(train_ds)

    # ── 记录基本信息 ──
    print(f"{'='*50}")
    print(f"实验2 Task 3 — 扩散生成模型（DDPM）")
    print(f"{'='*50}")
    print(f"训练样本数 : {n_train}")
    print(f"通道数     : {CHANNELS}")
    print(f"扩散步数 T : {T}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"学习率     : {LR}")
    print(f"设备       : {DEVICE}")
    print(f"输出目录   : {OUT_DIR}")
    print(f"{'='*50}")

    # ── 模型 / 优化器 ──
    model     = UNet(base_ch=CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量 : {n_params:,}")

    # ── 训练循环 ──
    loss_history = []
    t_start = time.time()

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"Ep {epoch:3d}/{EPOCHS}",
                         leave=False, unit="batch")
        for x0, _ in batch_bar:
            x0     = x0.to(DEVICE)
            B      = x0.size(0)
            t_samp = torch.randint(0, T, (B,), device=DEVICE)
            noise  = torch.randn_like(x0)
            x_t    = q_sample(x0, t_samp, noise)
            pred   = model(x_t, t_samp)
            loss   = criterion(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * B
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / n_train
        loss_history.append(avg_loss)
        elapsed  = time.time() - t_start
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", elapsed=f"{elapsed:.0f}s")

        # 每 10 epoch 保存 checkpoint
        if epoch % 10 == 0:
            ckpt_path = os.path.join(OUT_DIR, f"ddpm_ep{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            tqdm.write(f"  → checkpoint saved: {ckpt_path}")

    total_time = time.time() - t_start
    tqdm.write(f"\n训练完成！总训练时间：{total_time:.1f}s ({total_time/60:.1f} min)")

    # ── 保存最终模型 ──
    final_path = os.path.join(OUT_DIR, "ddpm_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存：{final_path}")

    # ── 保存训练 loss 曲线 ──
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("DDPM Training Loss (Fashion-MNIST)")
    plt.tight_layout()
    loss_fig_path = os.path.join(OUT_DIR, "train_loss.png")
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"训练 loss 曲线已保存：{loss_fig_path}")

    # ── 训练结束后采样 16 张 ──
    print("\n正在生成样本（16张，seed=0，T 步迭代去噪）…")
    samples = ddpm_sample(model, n=16, seed=0, device=DEVICE)
    sample_path = os.path.join(OUT_DIR, "samples_final.png")
    save_samples(samples, sample_path)
    print(f"生成样本已保存：{sample_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
