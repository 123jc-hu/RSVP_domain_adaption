from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary
from Utils.config import load_config


# -------------------------
# LightweightConv1d
# -------------------------
class LightweightConv1d(nn.Module):
    """
    轻量时间卷积（按论文常见lightweight conv思路实现）
    - 支持 padding='same'（仅 stride=1；建议奇数核）
    - num_heads=1 时，输出通道数 = in_channels * depth_multiplier
    输入:  (B, C, T)
    输出:  (B, C*depth_multiplier, T)
    """
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1,
        depth_multiplier: int = 1,
        kernel_size: int = 1,
        stride: int = 1,
        padding: str | int = 0,
        bias: bool = True,
        weight_softmax: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.depth_multiplier = depth_multiplier
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = padding
        self.weight_softmax = weight_softmax

        # 每个 head 1 入通道，输出 depth_multiplier 个通道
        self.weight = nn.Parameter(
            torch.Tensor(num_heads * depth_multiplier, 1, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier)) if bias else None
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def _pad_same(self, T: int) -> int:
        # 仅支持 stride=1 的 same padding
        assert self.stride == 1, "padding='same' currently supports stride=1"
        # 建议奇数核（与原代码一致）
        return self.kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.size()
        H = self.num_heads
        assert C % H == 0, f"in_channels ({C}) must be divisible by num_heads ({H})"

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        # 按 head 切分通道
        # x: (B, C, T) -> (B*C/H, H, T)
        x = rearrange(x, "b (h c) t -> (b c) h t", h=H)

        # 处理 padding
        if isinstance(self.padding, str) and self.padding.lower() == "same":
            pad = self._pad_same(T)
        else:
            pad = int(self.padding)

        out = F.conv1d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=pad,
            groups=self.num_heads,  # depthwise over heads
        )
        # (B*C/H, H*depth_multiplier, T) -> (B, H*C/H * H*dm, T) == (B, C*dm, T)
        out = rearrange(out, "(b c) k t -> b (k c) t", b=B)
        return out


# -------------------------
# 方差池化与注意力
# -------------------------
class VarMaxPool1D(nn.Module):
    def __init__(self, T: int, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 动态夹紧 kernel，避免 kernel > T
        T = x.shape[-1]
        k = min(self.kernel_size, T)
        s = self.stride if isinstance(self.stride, int) else k  # 你原来就是 stride=k
        mean_of_squares = F.avg_pool1d(x**2, k, s, self.padding)
        square_of_mean = F.avg_pool1d(x,    k, s, self.padding) ** 2
        variance = mean_of_squares - square_of_mean
        out = F.avg_pool1d(variance, variance.shape[-1])  # 全局池化
        return out


class VarPool1D(nn.Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_of_squares = F.avg_pool1d(x ** 2, self.kernel_size, self.stride, self.padding)
        square_of_mean = F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        variance = mean_of_squares - square_of_mean
        return variance


class SSA(nn.Module):
    """
    Spatial-Spectral Attention（与你的实现等价，T参数不实际参与计算）
    """
    def __init__(self, T: int, num_channels: int, epsilon: float = 1e-5, mode: str = "var", after_relu: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu
        k_gp = max(2, int(round(T * 0.25)))  # 1000点→250；128点→32
        self.GP = VarMaxPool1D(T, k_gp)

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        if self.mode == "l2":
            embedding = (x.pow(2).sum(2, keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == "l1":
            _x = x if self.after_relu else torch.abs(x)
            embedding = _x.sum(2, keepdim=True)
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:  # "var"
            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)
        return x * gate, gate


class Mixer1D(nn.Module):
    """
    将通道按 kernel_sizes 数目等分 -> 各自走 VarPool1D -> Flatten -> 拼接
    输入:  (B, F2, T) ; 输出: (B, D)  (D 取决于T与各分支输出)
    """
    def __init__(self, dim: int, kernel_sizes: List[int] = [50, 100, 250]):
        super().__init__()
        assert dim % len(kernel_sizes) == 0, "F2 must be divisible by number of kernel sizes"
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d, L = x.shape
        splits = torch.split(x, d // self.L, dim=1)
        outs = []
        for i, xi in enumerate(splits):
            outs.append(self.var_layers[i](xi))
        y = torch.cat(outs, dim=1)
        return y


def ms_to_samples(ms, fs, *, odd=False, min_val=1):
    v = max(min_val, int(round(ms * fs / 1000.0)))
    if odd and v % 2 == 0:
        v += 1
    return v

# -------------------------
# Efficient_Encoder（与原版等价，输入 (B,C,T)）
# -------------------------
class Efficient_Encoder(nn.Module):
    def __init__(
        self,
        samples: int,
        chans: int,
        F1: int = 16,
        F2: int = 36,
        time_kernel1: int = 75,
        pool_kernels: List[int] = [50, 100, 250],
    ):
        super().__init__()
        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(chans * F1, F2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F2),
            nn.ELU(inplace=True),
        )

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.time_conv(x)      # -> (B, C*F1, T)
        x, _ = self.ssa(x)         # -> (B, C*F1, T)
        x = self.chanConv(x)       # -> (B, F2, T)
        feat = self.mixer(x)       # -> (B, D)
        return feat


# -------------------------
# 统一接口：Model
# -------------------------
class Model(nn.Module):
    """
    统一入口：
    - 从 config 读取：
        n_channels, fs, n_class
        可选超参：
        F1=9, F2=48, time_kernel1=75, pool_kernels=[50,100,200],
        samples / input_samples（若提供，将在 __init__ 中预热以初始化原型参数）
    - 输入: (B,1,C,T)
    - 输出: logits (B, n_class)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.n_class = int(config["n_class"])

        F1 = int(config.get("F1", 9))
        F2 = int(config.get("F2", 48))
        time_kernel1 = ms_to_samples(75, self.fs, odd=True, min_val=3)
        pool_kernels = [
            ms_to_samples(50,  self.fs, odd=False, min_val=2),
            ms_to_samples(100, self.fs, odd=False, min_val=2),
            ms_to_samples(200, self.fs, odd=False, min_val=2),
        ]

        self.encoder = Efficient_Encoder(
            samples=self.fs,   # T 形参给 SSA 用，不影响计算
            chans=self.n_channels,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )

        # 原型分类器（ISP/ICP）
        self.isp: Optional[nn.Parameter] = None
        self.icp: Optional[nn.Parameter] = None
        self._feature_dim: Optional[int] = None

    # 懒初始化原型参数
    def _init_prototypes(self, feat_dim: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        if self.isp is None or self.icp is None:
            self.isp = nn.Parameter(torch.randn(self.n_class, feat_dim, device=device), requires_grad=True)
            self.icp = nn.Parameter(torch.randn(self.n_class, feat_dim, device=device), requires_grad=True)
            nn.init.kaiming_normal_(self.isp)
            self._feature_dim = feat_dim

    def get_features(self):
        if hasattr(self, "_last_features") and self._last_features is not None:
            return self._last_features
        raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,C,T) -> (B,C,T)
        x = x.squeeze(1)
        feats = self.encoder(x)              # (B, D)
        # self._last_features = feats

        # 首次前向时若原型未初始化，则懒初始化
        if self.isp is None or self.icp is None:
            self._init_prototypes(feats.shape[-1], device=feats.device)

        # 约束 ISP 为行向量单位范数（与原实现一致）
        with torch.no_grad():
            self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)

        # logits = <features, ISP_class>
        logits = torch.einsum("bd,cd->bc", feats, self.isp)
        return logits


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)

    # 手动测试模型
    # x = torch.randn(1, 1, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状