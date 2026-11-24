"""
实验3 数据：课程提供的 CIFAR-10 不均衡训练集 / 均衡测试集
================================================================
- 训练：data/cifar10/CIFAR10_unbalance/{0..9}/*.jpg（35,580 张，类别不均衡）
- 测试：data/cifar10/CIFAR10_balance/{0..9}/*.jpg  （4,870 张，每类 487 张）
- 从不均衡训练集中按类别分层留出 10% 作为验证集（种子 42），测试集只用于最终评价
- 首次运行把 JPG 解码成 uint8 数组缓存为 .npz，之后整体放到 GPU 上，
  数据增强（随机裁剪 + 水平翻转）也在 GPU 上批量完成
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cifar10")
TRAIN_DIR = "CIFAR10_unbalance"
TEST_DIR = "CIFAR10_balance"


def _load_folder(name, root=ROOT):
    cache = os.path.join(root, f"{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"]
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        folder = os.path.join(root, name, str(c))
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            xs.append(np.asarray(Image.open(os.path.join(folder, f)).convert("RGB")))
            ys.append(c)
    x, y = np.stack(xs), np.array(ys, dtype=np.int64)
    np.savez(cache, x=x, y=y)
    return x, y


def stratified_split(y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = int(round(len(idx) * val_ratio))
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


class GPUData:
    """把整个数据集放到设备上；返回 dict(train/val/test) -> (x uint8 NCHW, y)。"""

    def __init__(self, device, val_ratio=0.1, seed=42, root=ROOT):
        x_tr, y_tr = _load_folder(TRAIN_DIR, root)
        x_te, y_te = _load_folder(TEST_DIR, root)
        tr_idx, va_idx = stratified_split(y_tr, val_ratio, seed)

        def to_dev(x, y):
            return (torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device),
                    torch.from_numpy(y).to(device))

        self.splits = {
            "train": to_dev(x_tr[tr_idx], y_tr[tr_idx]),
            "val": to_dev(x_tr[va_idx], y_tr[va_idx]),
            "test": to_dev(x_te, y_te),
        }
        self.device = device
        self.class_counts = torch.bincount(self.splits["train"][1], minlength=NUM_CLASSES)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def normalize(self, x_uint8):
        return (x_uint8.float() / 255.0 - self.mean) / self.std

    def augment(self, x, gen):
        """随机裁剪（零填充 4 像素）+ 水平翻转，x 为已归一化的 float NCHW。"""
        B, C, H, W = x.shape
        dev = x.device
        flip = torch.rand(B, device=dev, generator=gen) < 0.5
        x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
        xp = F.pad(x, (4, 4, 4, 4))
        i = torch.randint(0, 9, (B,), device=dev, generator=gen)
        j = torch.randint(0, 9, (B,), device=dev, generator=gen)
        ar = torch.arange(H, device=dev)
        rows = (i[:, None] + ar)[:, None, :, None]
        cols = (j[:, None] + ar)[:, None, None, :]
        return xp[torch.arange(B, device=dev).view(B, 1, 1, 1),
                  torch.arange(C, device=dev).view(1, C, 1, 1), rows, cols]

    def train_batches(self, batch_size, gen, sample_weights=None):
        """一个 epoch 的训练 batch。sample_weights 不为 None 时按权重有放回采样（重采样）。"""
        x, y = self.splits["train"]
        n = len(y)
        if sample_weights is None:
            order = torch.randperm(n, device=self.device, generator=gen)
        else:
            order = torch.multinomial(sample_weights, n, replacement=True, generator=gen)
        for s in range(0, n - batch_size + 1, batch_size):  # drop_last
            idx = order[s:s + batch_size]
            yield self.augment(self.normalize(x[idx]), gen), y[idx]

    def eval_batches(self, split, batch_size=1024):
        x, y = self.splits[split]
        for s in range(0, len(y), batch_size):
            yield self.normalize(x[s:s + batch_size]), y[s:s + batch_size]


if __name__ == "__main__":
    d = GPUData("cpu")
    for k, (x, y) in d.splits.items():
        print(f"{k:5s} {tuple(x.shape)}  per-class: {torch.bincount(y, minlength=10).tolist()}")
    # 训练/测试是否有重复图片（逐像素完全相同）
    tr = {hash(a.tobytes()) for a in d.splits["train"][0].numpy()}
    tr |= {hash(a.tobytes()) for a in d.splits["val"][0].numpy()}
    dup = sum(hash(a.tobytes()) in tr for a in d.splits["test"][0].numpy())
    print(f"test images identical to a train/val image: {dup}")
