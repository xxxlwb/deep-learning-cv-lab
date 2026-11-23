# 实验1 任务3：仅使用 NumPy 实现单隐藏层 MLP，并进行梯度检验
#
# 网络结构：784 → 512 (ReLU) → 10 (Softmax)
# 损失函数：Cross-Entropy
# 验证方式：中心差分梯度检验
#
# 数据集：Fashion-MNIST，从官方源自动下载到 ../data，纯 numpy 读取，
#         不依赖 PyTorch / torchvision。

import gzip
import os
import struct
import urllib.request

import numpy as np


# ─────────────────────────────────────────────
# 0. Fashion-MNIST 数据加载（纯 numpy，无 torch 依赖）
# ─────────────────────────────────────────────

_FMNIST_URLS = {
    "train_images": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
}

# 备用镜像（官方源有时较慢）
_FMNIST_URLS_ALT = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
}


def _download(url: str, dest: str) -> None:
    """下载文件到 dest，若文件已存在则跳过。"""
    if os.path.exists(dest):
        return
    print(f"  下载 {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # 尝试备用链接
        alt_url = url.replace(
            "storage.googleapis.com/tensorflow/tf-keras-datasets",
            "fashion-mnist.s3-website.eu-central-1.amazonaws.com",
        )
        urllib.request.urlretrieve(alt_url, dest)


def _parse_images(path: str) -> np.ndarray:
    """解析 IDX3-ubyte（图像文件），返回 (N, 784) float64 数组，值域 [0, 1]。"""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x00000803, "图像文件 magic number 错误"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def _parse_labels(path: str) -> np.ndarray:
    """解析 IDX1-ubyte（标签文件），返回 (N,) int64 数组。"""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x00000801, "标签文件 magic number 错误"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_fashion_mnist(data_dir: str = "../data"):
    """下载并加载 Fashion-MNIST 训练集，返回 (X, y)。

    参数：
        data_dir : 数据存放目录（会自动创建）
    返回：
        X : (60000, 784) float64，像素值归一化到 [0, 1]
        y : (60000,)     int64，类别标签 0-9
    """
    fmnist_dir = os.path.join(data_dir, "FashionMNIST", "raw")
    os.makedirs(fmnist_dir, exist_ok=True)

    img_gz = os.path.join(fmnist_dir, "train-images-idx3-ubyte.gz")
    lbl_gz = os.path.join(fmnist_dir, "train-labels-idx1-ubyte.gz")

    _download(_FMNIST_URLS["train_images"], img_gz)
    _download(_FMNIST_URLS["train_labels"], lbl_gz)

    X = _parse_images(img_gz)
    y = _parse_labels(lbl_gz)
    return X, y


# ─────────────────────────────────────────────
# 1. 模型定义
# ─────────────────────────────────────────────

class NumpyMLP:
    """单隐藏层 MLP（Multi-Layer Perceptron）。

    结构：
        输入层  : 784 维（28×28 展平）
        隐藏层  : 512 维，激活函数 ReLU
        输出层  : 10 维，接 Softmax（在 loss 函数里计算）
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512,
                 output_dim: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)

        # He 初始化：方差 = 2 / fan_in，适合 ReLU 激活
        self.params = {
            "W1": rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros(hidden_dim),
            "W2": rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim),
            "b2": np.zeros(output_dim),
        }

    def forward(self, X: np.ndarray):
        """前向传播。

        参数：
            X : (N, 784) 输入特征矩阵

        返回：
            logits : (N, 10) 未经 Softmax 的原始得分
            cache  : 保存中间激活值，供 backward 使用
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        # 第一层线性变换
        z1 = X @ W1 + b1          # (N, 512)

        # ReLU 激活：负值置零
        a1 = np.maximum(0.0, z1)  # (N, 512)

        # 第二层线性变换（输出 logits）
        z2 = a1 @ W2 + b2         # (N, 10)

        cache = {"X": X, "z1": z1, "a1": a1}
        return z2, cache

    def backward(self, dlogits: np.ndarray, cache: dict) -> dict:
        """反向传播，计算各参数的梯度。

        参数：
            dlogits : (N, 10) 损失对 logits 的梯度（由 cross_entropy_loss 返回）
            cache   : forward 返回的中间激活值字典

        返回：
            grads : 包含 dW1, db1, dW2, db2 的字典
        """
        X  = cache["X"]           # (N, 784)
        z1 = cache["z1"]          # (N, 512)
        a1 = cache["a1"]          # (N, 512)
        W2 = self.params["W2"]    # (512, 10)
        N  = X.shape[0]

        # ── 输出层 ──────────────────────────────
        # dL/dW2 = a1^T · dlogits / N（除 N 因为 loss 已取了 batch 均值）
        dW2 = (a1.T @ dlogits) / N   # (512, 10)
        db2 = dlogits.mean(axis=0)   # (10,)

        # ── 隐藏层 ──────────────────────────────
        # 将梯度从 z2 反传到 a1
        da1 = dlogits @ W2.T         # (N, 512)

        # ReLU 的导数：z1 > 0 时为 1，否则为 0
        dz1 = da1 * (z1 > 0)        # (N, 512)

        # 继续反传到 W1, b1
        dW1 = (X.T @ dz1) / N       # (784, 512)
        db1 = dz1.mean(axis=0)      # (512,)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# ─────────────────────────────────────────────
# 2. 损失函数
# ─────────────────────────────────────────────

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray):
    """Softmax Cross-Entropy 损失。

    同时计算损失值和损失对 logits 的梯度，以提高数值稳定性。

    参数：
        logits : (N, C) 未经 Softmax 的原始得分
        labels : (N,)   整数类别标签，取值 0..C-1

    返回：
        loss    : 标量，batch 平均交叉熵
        dlogits : (N, C) dL/d(logits)，尚未除以 N（backward 里再除）
    """
    N = logits.shape[0]

    # 数值稳定的 Softmax：先减去每行最大值，防止指数爆炸
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z   = np.exp(shifted)
    probs   = exp_z / exp_z.sum(axis=1, keepdims=True)   # (N, C)

    # 交叉熵：-log(正确类别的概率)
    correct_log_probs = -np.log(probs[np.arange(N), labels] + 1e-15)
    loss = correct_log_probs.mean()

    # softmax-cross-entropy 对 logits 的梯度：
    #   dL/d(z_j) = p_j - 1(j == y)
    # 注意此处不除以 N，backward 里统一在 dW / db 处除
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1

    return loss, dlogits


# ─────────────────────────────────────────────
# 3. 梯度检验
# ─────────────────────────────────────────────

def gradient_check(model: NumpyMLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-5):
    """中心差分（center-difference）梯度检验。

    对每个参数 θ_i，用 (f(θ+ε) - f(θ-ε)) / (2ε) 近似数值梯度，
    与 backward 计算的解析梯度对比，输出最大相对误差。

    参数：
        model : NumpyMLP 实例
        X     : (N, 784) 输入
        y     : (N,)     标签
        eps   : 有限差分步长（默认 1e-5）
    """
    # 先跑一次完整的 forward + backward 得到解析梯度
    logits, cache = model.forward(X)
    _, dlogits    = cross_entropy_loss(logits, y)
    analytic_grads = model.backward(dlogits, cache)

    print("=" * 55)
    print(f"梯度检验（中心差分，eps={eps:.0e}）")
    print("=" * 55)

    all_pass = True

    for param_name in ["W1", "b1", "W2", "b2"]:
        param         = model.params[param_name]        # 参数数组的引用（in-place 修改）
        analytic_grad = analytic_grads["d" + param_name]

        # 只随机抽样部分元素，避免 W1(784×512) 全量检验耗时过长
        flat_size = param.size
        num_check = min(flat_size, 50)                  # 最多检验 50 个元素
        rng_check = np.random.default_rng(0)
        indices   = rng_check.choice(flat_size, size=num_check, replace=False)

        numerical_grad = np.zeros(num_check)

        for k, idx in enumerate(indices):
            multi_idx = np.unravel_index(idx, param.shape)

            # f(θ + ε)
            param[multi_idx] += eps
            logits_p, _ = model.forward(X)
            loss_p, _   = cross_entropy_loss(logits_p, y)

            # f(θ - ε)
            param[multi_idx] -= 2.0 * eps
            logits_m, _ = model.forward(X)
            loss_m, _   = cross_entropy_loss(logits_m, y)

            # 恢复原值
            param[multi_idx] += eps

            # 中心差分近似：(f(θ+ε) - f(θ-ε)) / (2ε)
            numerical_grad[k] = (loss_p - loss_m) / (2.0 * eps)

        # 取对应位置的解析梯度
        analytic_flat = analytic_grad.ravel()[indices]

        # 相对误差 = |数值梯度 - 解析梯度| / (|数值| + |解析| + ε_tiny)
        abs_diff = np.abs(numerical_grad - analytic_flat)
        denom    = np.abs(numerical_grad) + np.abs(analytic_flat) + 1e-8
        rel_errors  = abs_diff / denom
        max_rel_err = rel_errors.max()

        status = "✓ PASS" if max_rel_err < 1e-5 else "✗ FAIL"
        if max_rel_err >= 1e-5:
            all_pass = False

        print(f"  {param_name:4s}  最大相对误差: {max_rel_err:.2e}  {status}")

    print("=" * 55)
    print("总体结论:", "全部通过 ✓" if all_pass else "存在失败项 ✗")
    print("=" * 55)


# ─────────────────────────────────────────────
# 4. 主程序
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── 加载 Fashion-MNIST ─────────────────────
    print("加载 Fashion-MNIST ...")
    X_all, y_all = load_fashion_mnist(data_dir="../data")
    print(f"完整数据集: X={X_all.shape}, y={y_all.shape}")

    # 用随机种子 42 抽取 32 个样本作为固定 mini-batch
    BATCH_SIZE = 32
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_all), size=BATCH_SIZE, replace=False)
    X = X_all[idx]   # (32, 784)
    y = y_all[idx]   # (32,)

    print(f"Mini-batch shape: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y, minlength=10).tolist()}\n")

    # ── 实例化模型并做一次 forward / backward ──
    model = NumpyMLP(seed=42)

    logits, cache = model.forward(X)
    loss, dlogits = cross_entropy_loss(logits, y)
    grads = model.backward(dlogits, cache)

    print(f"初始 Loss: {loss:.4f}")
    for name, g in grads.items():
        print(f"  grad {name}: shape={g.shape}, ‖g‖={np.linalg.norm(g):.4f}")

    print()

    # ── 梯度检验 ───────────────────────────────
    gradient_check(model, X, y, eps=1e-5)
