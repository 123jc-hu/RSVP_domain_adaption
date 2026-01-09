from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
from Models.eeg_models import model_dict
import math


# =========================
#    通道选择器（Concrete）
# =========================
class ConcreteMultiSelector(nn.Module):
    """
    K 个选择神经元，在 C 个通道上做 Gumbel-Softmax。
    - 训练期：
        per_sample=True → alpha + gumbel；False → 只用 alpha
        train_soft_only=True → 纯 soft；否则 ST (hard前向软反传)
    - 推理期：argmax → one-hot
    输入: x [B,1,C,T]
    返回: z [B,1,K,T], W [K,C] (或 hard), P [K,C]
    """
    def __init__(
        self,
        C: int,
        K: int,
        beta_start: float = 10.0,
        beta_end: float = 0.1,
        T_anneal: int = 150,
        per_sample: bool = True,  # 新增: 是否逐样本加 Gumbel 噪声
        train_soft_only: bool = False,    # ★ 新增：训练期是否纯soft（关闭ST）
    ):
        super().__init__()
        self.C, self.K = int(C), int(K)
        self.per_sample = bool(per_sample)
        self.train_soft_only = bool(train_soft_only)   # ★

        # α logits，小范围均匀初始化，避免一开始过硬
        self.alpha = nn.Parameter(torch.zeros(K, C))
        nn.init.uniform_(self.alpha, -1e-2, 1e-2)

        # 温度/阈值退火（指数曲线）
        self.beta_start, self.beta_end = float(beta_start), float(beta_end)
        self.T_anneal = max(1, int(T_anneal))
        self._t = 0  # 以 epoch 为单位的计数器

    # ---------- 调度 ----------
    def step_epoch(self, n: int = 1):
        """每训练完一个 epoch 调用一次，用于退火 β/τ。"""
        self._t += int(n)

    def _beta(self) -> float:
        """当前温度 β（指数退火），数值保护 ≥ 1e-6。"""
        t = min(self._t / self.T_anneal, 1.0)
        val = self.beta_start * (self.beta_end / self.beta_start) ** t
        return float(max(val, 1e-6))

    @torch.no_grad()
    def hard_indices(self):
        """返回当前每个选择神经元的 argmax 通道索引（K,）。"""
        return torch.argmax(self.alpha, dim=1).tolist()

    # ---------- 工具 ----------
    @staticmethod
    def _gumbel_like(x: torch.Tensor) -> torch.Tensor:
        """U~(0,1) → g = -log(-log(U))。"""
        u = torch.rand_like(x).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(u))

    @staticmethod
    def _row_softmax(logits: torch.Tensor, beta: float) -> torch.Tensor:
        """按行 softmax（每个 selection 神经元独立分布）。"""
        return torch.softmax(logits / beta, dim=1)

    def current_p_soft(self) -> torch.Tensor:
        """
        返回“当前温度下”的 P_soft（不带噪声，用于监控/正则）。
        形状: (K, C)
        """
        beta = self._beta()
        return self._row_softmax(self.alpha, beta)

    # ---------- 前向 ----------
    def forward(self, x: torch.Tensor, current_epoch: int = None):
        """
        x: [B, 1, C, T]
        """
        B, _, C, T = x.shape
        assert C == self.C, f"Selector expects C={self.C}, got C={C}"
        beta = self._beta()

        if self.training:
            # --- 是否加噪声：保持原per_sample语义 ---
            if self.per_sample:
                # 原有“整批共享一次Gumbel”的路径 (K,C)
                logits = self.alpha + self._gumbel_like(self.alpha)
            else:
                # 不加噪声：确定性 soft
                logits = self.alpha

            # —— 原有(K,C)路径保持不变 —— 
            P_soft = self._row_softmax(logits, beta)  # (K,C)

            if self.train_soft_only:
                W = P_soft                               # ★ 训练期纯soft
            else:
                with torch.no_grad():
                    idx = torch.argmax(logits, dim=1)
                    W_hard = torch.zeros_like(P_soft)
                    W_hard[torch.arange(self.K), idx] = 1.0
                W = P_soft + (W_hard - P_soft).detach()  # ST

            z = torch.einsum("kc,bct->bkt", W, x.squeeze(1)).unsqueeze(1)
            return z, W, P_soft

        # 推理期：保持你原实现（硬 one-hot）
        idx = torch.argmax(self.alpha, dim=1)
        W_hard = torch.zeros(self.K, self.C, device=x.device, dtype=x.dtype)
        W_hard[torch.arange(self.K), idx] = 1.0
        z = x[:, :, idx, :]
        return z, W_hard, W_hard

    def orth_regularizer(self, P_soft: torch.Tensor) -> torch.Tensor:
        """
        行间“近似正交”约束：P_soft P_soft^T 接近单位阵。
        """
        G = P_soft @ P_soft.t()  # (K,K)
        I = torch.eye(self.K, device=G.device, dtype=G.dtype)
        return ((G - I) ** 2).sum()

    @staticmethod
    def normalized_entropy(P_soft: torch.Tensor):
        """
        每行的归一化熵： H_k = - Σ_c p_kc log p_kc / log(C)
        返回：(H_k,) 以及其均值。
        """
        eps = 1e-12
        pk = P_soft.clamp_min(eps)
        Hk = -(pk * pk.log()).sum(dim=1) / math.log(pk.size(1))
        return Hk, Hk.mean()


# =========================
#   包装器：选择器 + 主干
# =========================
class ModelWithSelector(nn.Module):
    """
    将任意主干网络（要求以 K 通道初始化）前置一个 Concrete 通道选择器。
    """
    def __init__(
        self,
        backbone_cls,
        backbone_config: Dict[str, Any],
        C: int,
        K: int,
        beta_start: float = 10.0,
        beta_end: float = 0.1,
        T_anneal: int = 150,
        per_sample: bool = True,
    ):
        super().__init__()
        assert backbone_config["n_channels"] == K, "backbone 的 n_channels 必须等于 K"

        self.selector = ConcreteMultiSelector(
            C=C,
            K=K,
            beta_start=beta_start,
            beta_end=beta_end,
            T_anneal=T_anneal,
            per_sample=per_sample,
        )
        self.backbone = backbone_cls(backbone_config)

        # 选择后的小尺度自适应：避免分布漂移（初始化为 1.0）
        self.post_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, current_epoch: int = None, **kwargs):
        # 与模型 train()/eval 同步
        z, W, P = self.selector(x, current_epoch=current_epoch)
        z = z * self.post_scale
        out = self.backbone(z, **kwargs)
        return (*out, W) if isinstance(out, tuple) else (out, W)

    def step_epoch(self, n: int = 1):
        """外部每个 epoch 调用一次，用于退火 β/τ。"""
        self.selector.step_epoch(n)

    @torch.no_grad()
    def selected_indices(self):
        return self.selector.hard_indices()

    @torch.no_grad()
    def project_channel_attn(self, full_attn: torch.Tensor) -> torch.Tensor:
        """
        将完整通道注意力 [B,1,C,T] 投影到所选 K 通道（用于可视化/分析）。
        """
        assert full_attn.dim() == 4 and full_attn.size(2) == self.selector.C
        idx = torch.argmax(self.selector.alpha, dim=1)  # (K,)
        return full_attn[:, :, idx, :]


# =========================
#       模型工厂
# =========================
def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    统一工厂：
    - use_selector=False → 返回原模型 Model(config)
    - use_selector=True  → 先把主干改成 K 通道，然后前置选择器包装
    """
    BaseCls = model_dict()[config["model"]].Model
    use_selector = bool(config.get("use_selector", False))
    if not use_selector:
        return BaseCls(config)

    C = int(config["n_channels"])
    K = int(config.get("selector_K", max(1, C // 2)))
    beta_start = float(config.get("selector_beta_start", 10.0))
    beta_end = float(config.get("selector_beta_end", 0.1))
    T_anneal = int(config.get("selector_T", config.get("epochs", 100)))
    per_sample = bool(config.get("selector_per_sample", False))

    # 复制一份给主干：将 n_channels 改为 K
    backbone_cfg = dict(config)
    backbone_cfg["n_channels"] = K

    return ModelWithSelector(
        backbone_cls=BaseCls,
        backbone_config=backbone_cfg,
        C=C,
        K=K,
        beta_start=beta_start,
        beta_end=beta_end,
        T_anneal=T_anneal,
        per_sample=per_sample,
    )