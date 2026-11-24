"""
实验3 模型：ViT 基线 + 两类轻量化
====================================
1. ViT（Dosovitskiy et al., 2021）：Patch Embedding → [CLS] + 位置编码 → L × Transformer Encoder → LayerNorm → 分类头
2. 轻量化方法
   a. DynamicViT（Rao et al., NeurIPS 2021）：轻量预测器逐级打分，动态丢弃不重要的 token
   b. ToMe（Bolya et al., ICLR 2023）：每层用二分软匹配把相似 token 合并，无需训练
   c. 结构化剪枝：按一阶 Taylor 重要性删除注意力头与 MLP 神经元，再用知识蒸馏微调

所有注意力都用显式矩阵乘法实现，便于 DynamicViT 的掩码注意力、ToMe 的比例注意力，
以及用 torch.utils.flop_counter 统一统计 FLOPs。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────── ViT ────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) >= self.p
        return x * keep / (1 - self.p)


def softmax_with_policy(attn, policy, eps=1e-6):
    """DynamicViT 训练期的掩码 softmax：被丢弃的 token 不参与注意力，但梯度仍可回传到 policy。
    attn: (B,H,N,N)；policy: (B,N,1)，1 = 保留。每个 token 始终可以关注自己。"""
    B, N, _ = policy.shape
    p = policy.reshape(B, 1, 1, N)
    eye = torch.eye(N, dtype=p.dtype, device=p.device).view(1, 1, N, N)
    p = p + (1.0 - p) * eye
    attn = attn.float()
    attn = (attn - attn.max(-1, keepdim=True)[0]).exp() * p
    return (attn + eps / N) / (attn.sum(-1, keepdim=True) + eps)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.qkv = nn.Linear(dim, 3 * inner)
        self.proj = nn.Linear(inner, dim)
        self.scale = head_dim ** -0.5

    def forward(self, x, policy=None, size=None):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:  # ToMe 比例注意力：合并了 s 个 token 的键权重加 log(s)
            attn = attn + size.log().view(B, 1, 1, N).to(attn.dtype)
        if policy is None:
            attn = attn.softmax(-1)
        else:
            attn = softmax_with_policy(attn, policy).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out), k.mean(1)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_hidden, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_hidden)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, policy=None, size=None, tome_r=0):
        a, metric = self.attn(self.norm1(x), policy, size)
        x = x + self.drop_path(a)
        if tome_r > 0:
            x, size = tome_merge(x, metric, size, tome_r)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


class ViT(nn.Module):
    def __init__(self, img_size=32, patch=4, dim=256, depth=6, heads=8, head_dim=32,
                 mlp_hidden=512, num_classes=10, drop_path=0.1,
                 layer_heads=None, layer_mlp=None):
        super().__init__()
        layer_heads = layer_heads or [heads] * depth
        layer_mlp = layer_mlp or [mlp_hidden] * depth
        self.cfg = dict(img_size=img_size, patch=patch, dim=dim, depth=depth, heads=heads,
                        head_dim=head_dim, mlp_hidden=mlp_hidden, num_classes=num_classes,
                        drop_path=drop_path, layer_heads=list(layer_heads), layer_mlp=list(layer_mlp))
        self.num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            Block(dim, layer_heads[i], head_dim, layer_mlp[i], dpr[i]) for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.tome_r = 0  # >0 时启用 ToMe（每层合并 r 个 token）
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def embed(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B, 64, dim)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1)
        return x + self.pos_embed                                    # (B, 65, dim)

    def forward(self, x, return_tokens=False):
        x = self.embed(x)
        size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype) if self.tome_r else None
        for blk in self.blocks:
            x, size = blk(x, size=size, tome_r=self.tome_r)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return (logits, x[:, 1:]) if return_tokens else logits


# ─────────────────────────────── ToMe ───────────────────────────────────────
def tome_merge(x, metric, size, r):
    """二分软匹配（Bipartite Soft Matching）：token 交替分成 A/B 两组，A 中与 B 最相似的 r 个
    token 合并进其最相似的 B token（按 size 加权平均）。[CLS] 永不被合并。"""
    B, N, C = x.shape
    r = min(r, (N - 1) // 2)
    if r <= 0:
        return x, size
    with torch.no_grad():
        m = metric / metric.norm(dim=-1, keepdim=True)
        a, b = m[:, ::2], m[:, 1::2]
        scores = a @ b.transpose(-1, -2)
        scores[:, 0, :] = -math.inf                                   # 保护 CLS
        node_max, node_idx = scores.max(-1)
        edge_idx = node_max.argsort(-1, descending=True)[..., None]
        unm_idx = edge_idx[:, r:].sort(dim=1)[0]                      # 未合并（保持顺序，CLS 在首位）
        src_idx = edge_idx[:, :r]
        dst_idx = node_idx[..., None].gather(1, src_idx)

    def merge(t, mode):
        src, dst = t[:, ::2], t[:, 1::2]
        c = t.shape[-1]
        unm = src.gather(1, unm_idx.expand(-1, -1, c))
        src = src.gather(1, src_idx.expand(-1, -1, c))
        dst = dst.scatter_reduce(1, dst_idx.expand(-1, -1, c), src, reduce=mode)
        return torch.cat([unm, dst], 1)

    x = merge(x * size, "sum")
    size = merge(size, "sum")
    return x / size, size


# ──────────────────────────── DynamicViT ────────────────────────────────────
class PredictorLG(nn.Module):
    """DynamicViT 的 token 重要性预测器：局部特征 + 全局（被保留 token 的平均）特征 → keep/drop 的对数概率。"""

    def __init__(self, dim):
        super().__init__()
        self.in_conv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.out_conv = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 2), nn.LogSoftmax(dim=-1))

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.shape
        local_x = x[..., :C // 2]
        global_x = (x[..., C // 2:] * policy).sum(1, keepdim=True) / policy.sum(1, keepdim=True)
        return self.out_conv(torch.cat([local_x, global_x.expand(B, N, C // 2)], -1))


class DynamicViT(nn.Module):
    """在 ViT 的若干层之前插入预测器，逐级保留 keep_rate^k 比例的 patch token。
    训练：Gumbel-Softmax 得到硬掩码 + 掩码注意力（token 数不变，可端到端训练）；
    推理：按预测分数 top-k 真正删除 token，从而减少计算量。"""

    def __init__(self, vit: ViT, prune_loc=(2, 4), keep_rate=0.7):
        super().__init__()
        self.vit = vit
        self.prune_loc = list(prune_loc)
        self.keep_rate = keep_rate
        self.predictors = nn.ModuleList(PredictorLG(vit.cfg["dim"]) for _ in self.prune_loc)
        self.dyn_cfg = dict(prune_loc=self.prune_loc, keep_rate=keep_rate)

    def forward(self, x, return_tokens=False):
        v = self.vit
        x = v.embed(x)
        B, N = x.shape[0], x.shape[1] - 1
        p = 0
        if self.training:
            prev = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
            policy, decisions = None, []
            for i, blk in enumerate(v.blocks):
                if i in self.prune_loc:
                    pred = self.predictors[p](x[:, 1:], prev).float()
                    hard = F.gumbel_softmax(pred, hard=True)[..., 0:1].to(x.dtype) * prev
                    decisions.append(hard.squeeze(-1))
                    policy = torch.cat([torch.ones(B, 1, 1, device=x.device, dtype=x.dtype), hard], 1)
                    prev, p = hard, p + 1
                x, _ = blk(x, policy=policy)
            x = v.norm(x)
            return v.head(x[:, 0]), x[:, 1:], prev, decisions
        for i, blk in enumerate(v.blocks):
            if i in self.prune_loc:
                ones = torch.ones(B, x.shape[1] - 1, 1, device=x.device, dtype=x.dtype)
                score = self.predictors[p](x[:, 1:], ones)[..., 0]
                n_keep = int(N * self.keep_rate ** (p + 1))
                keep = score.argsort(1, descending=True)[:, :n_keep]
                tokens = x[:, 1:].gather(1, keep[..., None].expand(-1, -1, x.shape[-1]))
                x = torch.cat([x[:, :1], tokens], 1)
                p += 1
            x, _ = blk(x)
        x = v.norm(x)
        logits = v.head(x[:, 0])
        return (logits, x[:, 1:]) if return_tokens else logits


# ─────────────────────────── 结构化剪枝 ──────────────────────────────────────
def taylor_importance(model: ViT, batches, loss_fn):
    """一阶 Taylor 重要性 |w·∂L/∂w|，按注意力头 / MLP 神经元分组求和。"""
    model.train()
    for blk in model.blocks:
        blk.drop_path.p = 0.0
    head_imp = [torch.zeros(b.attn.heads, device=next(model.parameters()).device) for b in model.blocks]
    mlp_imp = [torch.zeros(b.mlp.fc1.out_features, device=head_imp[0].device) for b in model.blocks]
    for x, y in batches:
        model.zero_grad()
        loss_fn(model(x), y).backward()
        for i, b in enumerate(model.blocks):
            H, D = b.attn.heads, b.attn.head_dim
            qkv = (b.attn.qkv.weight * b.attn.qkv.weight.grad).abs().sum(1) \
                + (b.attn.qkv.bias * b.attn.qkv.bias.grad).abs()
            proj = (b.attn.proj.weight * b.attn.proj.weight.grad).abs().sum(0)
            head_imp[i] += qkv.view(3, H, D).sum((0, 2)) + proj.view(H, D).sum(1)
            fc1 = (b.mlp.fc1.weight * b.mlp.fc1.weight.grad).abs().sum(1) \
                + (b.mlp.fc1.bias * b.mlp.fc1.bias.grad).abs()
            fc2 = (b.mlp.fc2.weight * b.mlp.fc2.weight.grad).abs().sum(0)
            mlp_imp[i] += fc1 + fc2
    model.zero_grad()
    return head_imp, mlp_imp


@torch.no_grad()
def prune_vit(model: ViT, head_imp, mlp_imp, head_keep=0.5, mlp_keep=0.5):
    """每层保留重要性最高的 head_keep 比例的头和 mlp_keep 比例的 MLP 神经元，返回一个更窄的新 ViT。"""
    cfg = dict(model.cfg)
    keep_h = [torch.topk(h, max(1, round(len(h) * head_keep))).indices.sort()[0] for h in head_imp]
    keep_m = [torch.topk(m, max(1, round(len(m) * mlp_keep))).indices.sort()[0] for m in mlp_imp]
    cfg["layer_heads"] = [len(k) for k in keep_h]
    cfg["layer_mlp"] = [len(k) for k in keep_m]
    new = ViT(**cfg).to(next(model.parameters()).device)
    sd = {k: v for k, v in model.state_dict().items()}
    D = cfg["head_dim"]
    for i, (old_b, new_b) in enumerate(zip(model.blocks, new.blocks)):
        H = old_b.attn.heads
        dims = (keep_h[i][:, None] * D + torch.arange(D, device=keep_h[i].device)).flatten()
        qkv_rows = torch.cat([s * H * D + dims for s in range(3)])
        new_b.attn.qkv.weight.copy_(old_b.attn.qkv.weight[qkv_rows])
        new_b.attn.qkv.bias.copy_(old_b.attn.qkv.bias[qkv_rows])
        new_b.attn.proj.weight.copy_(old_b.attn.proj.weight[:, dims])
        new_b.attn.proj.bias.copy_(old_b.attn.proj.bias)
        new_b.mlp.fc1.weight.copy_(old_b.mlp.fc1.weight[keep_m[i]])
        new_b.mlp.fc1.bias.copy_(old_b.mlp.fc1.bias[keep_m[i]])
        new_b.mlp.fc2.weight.copy_(old_b.mlp.fc2.weight[:, keep_m[i]])
        new_b.mlp.fc2.bias.copy_(old_b.mlp.fc2.bias)
        for n in ("norm1", "norm2"):
            getattr(new_b, n).load_state_dict(getattr(old_b, n).state_dict())
    for k in ("patch_embed.weight", "patch_embed.bias", "cls_token", "pos_embed",
              "norm.weight", "norm.bias", "head.weight", "head.bias"):
        new.state_dict()[k].copy_(sd[k])
    return new


# ─────────────────────────── 保存 / 加载 ─────────────────────────────────────
def save_model(model, path, **meta):
    if isinstance(model, DynamicViT):
        ck = dict(kind="dynamicvit", cfg=model.vit.cfg, dyn=model.dyn_cfg)
    else:
        ck = dict(kind="vit", cfg=model.cfg)
    ck.update(state_dict=model.state_dict(), **meta)
    torch.save(ck, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    vit = ViT(**ck["cfg"])
    model = DynamicViT(vit, **ck["dyn"]) if ck["kind"] == "dynamicvit" else vit
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck
