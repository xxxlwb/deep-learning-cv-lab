# Deep Learning & Computer Vision Lab

Fashion-MNIST 分类与生成实验，覆盖实验1（MLP 基线）和实验2（CNN + 扩散模型）。

## 环境

Python 3.12，依赖由 [uv](https://docs.astral.sh/uv/) 管理。

```bash
# 安装 uv（已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在项目根目录一键创建虚拟环境并安装依赖
cd deep-learning-cv-lab
uv sync

# 所有脚本通过项目 venv 运行
.venv/bin/python lab1/softmax_baseline.py
# 或者激活后直接用 python
source .venv/bin/activate
```

## 数据

Fashion-MNIST 首次运行时自动下载到 `data/`（torchvision），无需手动准备。

**数据划分（实验1&2 完全一致）：**
- 训练集：官方60k 前 50,000 条
- 验证集：官方60k 后 10,000 条
- 测试集：官方 10,000 条，只用于最终评价

## 随机种子

| 用途 | 种子 |
|------|------|
| 所有 PyTorch / NumPy 实验 | `42` |
| Task 2 像素置换 | `np.random.default_rng(0)` |
| Task 3 扩散模型训练 | `42` |
| Task 4 采样 | `0` |

---

## 实验1

### Task 2：线性 Softmax 基线

```bash
cd lab1
../.venv/bin/python softmax_baseline.py
# 输出：outputs_lab1/softmax_*.png + softmax.pth
```

| 指标 | 值 |
|------|----|
| Test Accuracy | 83.56% |
| 参数量 | 7,850 |
| 训练时间 | ~171s |

### Task 3：NumPy 反向传播 + 梯度检验

```bash
../.venv/bin/python numpy_mlp.py
```

在固定 mini-batch(32) 上验证 backward 实现，中心差分梯度检验结果：

| 参数 | 最大相对误差 |
|------|------------|
| W1 | 7.81e-07 ✓ |
| b1 | 2.22e-08 ✓ |
| W2 | 1.87e-07 ✓ |
| b2 | 4.20e-09 ✓ |

### Task 4 & 5：受控比较 + 训练诊断

```bash
../.venv/bin/python train_mlp.py
# 输出：outputs_lab1/comparison.png + summary_bar.png + diagnostic_dropout.png
```

**Task 4 结果：**

| 模型 | 参数量 | Test Acc | 说明 |
|------|--------|---------|------|
| A: Linear Softmax | 7,850 | 84.47% | 单层线性 |
| B: 2-layer Linear | 407,050 | 83.99% | 无激活，仍是线性 |
| C: MLP + ReLU | 407,050 | **88.37%** | 非线性带来显著提升 |

**Task 5 结论：** MLP 存在过拟合（train-val gap = 0.084），加 Dropout(0.3) 后 gap 降至 0.047，test acc 提升至 89.13%。

---

## 实验2

### Task 1：MLP vs CNN

```bash
cd lab2
../.venv/bin/python train_mlp_cnn.py
# 输出：outputs_task1/
```

| 模型 | 参数量 | Test Acc | 训练时间 |
|------|--------|---------|---------|
| MLP | 407,050 | 89.33% | ~115s |
| CNN | 824,458 | **92.01%** | ~315s |

CNN 层输出尺寸：
```
Conv(1→32, 3×3)  : 32×28×28  (320 params)
MaxPool(2×2)      : 32×14×14
Conv(32→64, 3×3) : 64×14×14  (18,496 params)
MaxPool(2×2)      : 64×7×7
Linear(3136→256)  : 256       (803,072 params)
Linear(256→10)    : 10        (2,570 params)
```

### Task 2：结构先验检验（像素置换）

```bash
../.venv/bin/python permutation_test.py
# 输出：outputs_task2/
```

固定置换 `perm = np.random.default_rng(0).permutation(784)`，同时应用于训练/验证/测试。

| 配置 | 参数量 | Test Acc | 训练时间 |
|------|--------|---------|---------|
| MLP + 原图 | 407,050 | 98.29% | 44.5s |
| MLP + 置换图 | 407,050 | 98.13% | 53.4s | 
| CNN + 原图 | 824,458 | **99.23%** | 195.2s |
| CNN + 置换图 | 824,458 | 97.20% | 371.4s |

**结论：** CNN 原图→置换图下降 **2.03%**，MLP 仅下降 **0.16%**，验证了 CNN 的性能优势依赖局部空间结构，而 MLP 对像素顺序不敏感。

### Task 3：扩散生成模型（DDPM）

```bash
../.venv/bin/python diffusion_model.py
# 可选参数：--T 300 --epochs 30 --batch_size 256 --lr 2e-4
# 输出：outputs_task3/ddpm_ep10.pth, ddpm_ep20.pth, ddpm_ep30.pth, ddpm_final.pth
```

| 配置 | 值 |
|------|-----|
| 训练样本数 | 60,000 |
| 通道数（base_ch）| 64 |
| 扩散步数 T | 300 |
| Epochs | 30 |
| Batch size | 256 |
| Learning rate | 2e-4 |
| 设备 | CUDA（NVIDIA H200）|
| 训练时间 | **215s（3.6 min）** |
| 最终 avg_loss | **0.0701** |

### Task 4：生成结果评价

```bash
# 需先完成 Task 3 训练
../.venv/bin/python eval_diffusion.py
# 输出：outputs_task4/
```

| 指标 | 值 |
|------|-----|
| 类别覆盖 | **10/10 类别** |
| 采样速度（16张）| 0.80s（50ms/张）|
| 采样速度（100张）| 0.64s（6ms/张）|
| 设备 | CUDA（NVIDIA H200）|

类别分布（KNN预测，100张样本）：

| T-shirt | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |
|---------|---------|---------|-------|------|--------|-------|---------|-----|------------|
| 13 | 3 | 20 | 5 | 9 | 1 | 10 | 1 | 24 | 14 |

输出内容：
1. `1_random_grid.png` — 随机样本网格（seed=0，固定初始噪声）
2. `2_noising_denoising.png` — 加噪与去噪过程
3. `3_class_coverage.png` — 类别覆盖（KNN 分类）
4. `4_nearest_neighbor.png` — 训练集最近邻
5. `5_failure_samples.png` — 失败样本（不挑选成功样本）

## 文件结构

```
DeepLearning/
├── pyproject.toml          # uv 项目配置 & 依赖声明
├── uv.lock                 # 锁定依赖版本
├── .venv/                  # 项目专属虚拟环境（不入库）
├── data/                   # Fashion-MNIST 数据（不入库）
├── lab1/
│   ├── softmax_baseline.py # Task 2: 线性 Softmax
│   ├── numpy_mlp.py        # Task 3: NumPy 反向传播 + 梯度检验
│   └── train_mlp.py        # Task 4&5: 受控比较 + 训练诊断
└── lab2/
    ├── train_mlp_cnn.py    # Task 1: MLP vs CNN
    ├── permutation_test.py # Task 2: 结构先验检验
    ├── diffusion_model.py  # Task 3: DDPM 扩散模型
    └── eval_diffusion.py   # Task 4: 生成结果评价
```
