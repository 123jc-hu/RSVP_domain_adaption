from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# =========================
# helpers
# =========================
def alloc_three(F_total: int, alpha: float):
    """
    按论文比例分配滤波器数：
      high:   αF
      medium: α(1-α)F
      low:    (1-α)^2 F
    并保证为整数且和为 F_total。
    在本文 α=0.75 时：
      F1=16  -> (12, 3, 1)
      F2=128 -> (96, 24, 8)
      F3=32  -> (24, 6, 2)
    """
    a = float(alpha)
    w = [a, a * (1 - a), (1 - a) ** 2]
    raw = [wi * F_total for wi in w]
    base = [int(math.floor(r)) for r in raw]
    s = sum(base)

    frac = [r - math.floor(r) for r in raw]
    order = sorted(range(3), key=lambda i: frac[i], reverse=True)

    k = 0
    while s < F_total:
        base[order[k % 3]] += 1
        s += 1
        k += 1
    while s > F_total:
        i = max(range(3), key=lambda j: base[j])
        base[i] -= 1
        s -= 1

    return base[0], base[1], base[2]


def same_pad_time(x: torch.Tensor, k_t: int) -> torch.Tensor:
    """只对时间维做 SAME padding（stride=1），支持偶数核。x: [B,C,H,T]"""
    if k_t <= 1:
        return x
    total = k_t - 1
    left = total // 2
    right = total - left
    return F.pad(x, (left, right, 0, 0))


def avgpool_time(x: torch.Tensor, k: int) -> torch.Tensor:
    """时间维 avgpool: kernel=(1,k), stride=(1,k)"""
    if k == 1:
        return x
    return F.avg_pool2d(x, kernel_size=(1, k), stride=(1, k))


class SpatialDropout2D(nn.Module):
    """Keras SpatialDropout2D 的近似：nn.Dropout2d（按通道整张 feature map 丢弃）"""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x)


# =========================
# blocks
# =========================
class ConvBNELUDrop(nn.Module):
    """
    普通 temporal conv: g(·; ω; κ) with κ=(1,k_t)
    后接 BN + ELU + Dropout
    dropout_type:
      - spatial=True  -> SpatialDropout2D
      - spatial=False -> nn.Dropout (元素级)
    """
    def __init__(self, in_ch: int, out_ch: int, k_t: int, p_drop: float = 0.5, spatial: bool = True):
        super().__init__()
        self.k_t = k_t
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, k_t), bias=False, padding=0)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU()
        self.drop = SpatialDropout2D(p_drop) if spatial else nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = same_pad_time(x, self.k_t)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DepthwiseSpatialConv(nn.Module):
    """
    depthwise spatial conv: d(·; ω; κ2=(C,1); τ=in_ch)
    用 groups=in_ch 实现 depthwise，允许 out_ch = in_ch * depth_multiplier。
    后接 BN + ELU + SpatialDropout2D
    """
    def __init__(self, in_ch: int, out_ch: int, C: int, p_drop: float = 0.5):
        super().__init__()
        assert out_ch % in_ch == 0, "out_ch must be multiple of in_ch for depthwise"
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(C, 1),
            groups=in_ch,
            bias=False,
            padding=0
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU()
        self.drop = SpatialDropout2D(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)      # [B,out,1,T]
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class SeparableConvBNELU(nn.Module):
    """
    separable conv: s(·; ω=F5; κ5=(1,k_t); τ=F5)
    = depthwise(1,k_t) + pointwise(1,1) -> out_ch=F5
    后接 BN + ELU
    """
    def __init__(self, ch: int, k_t: int):
        super().__init__()
        self.k_t = k_t
        self.dw = nn.Conv2d(ch, ch, kernel_size=(1, k_t), groups=ch, bias=False, padding=0)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = same_pad_time(x, self.k_t)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# =========================
# Model (MOCNN)
# =========================
class Model(nn.Module):
    """
    Paper-aligned MOCNN.

    config 需要字段:
      - "n_channels": C
      - "fs":         128
      - "seq_len":    T (建议等于 fs，即 128)
      - "n_class":    2

    可选字段:
      - "alpha":      默认 0.75
      - "dropout":    默认 0.5
      - 可覆盖结构参数: F1,F2,F3,F4,F5
    """

    def __init__(self, config: dict):
        super().__init__()

        self.C = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.T = int(config.get("seq_len", self.fs))
        self.n_class = int(config.get("n_class", 2))

        # paper hyper-params
        self.alpha = float(config.get("alpha", 0.75))
        self.p_drop = float(config.get("dropout", 0.5))

        # structural params (paper)
        self.F1 = int(config.get("F1", 16))
        self.F2 = int(config.get("F2", 128))
        self.F3 = int(config.get("F3", 32))
        self.F4 = int(config.get("F4", 2))
        self.F5 = int(config.get("F5", 6))

        # conv params (paper)
        self.k1 = 16  # κ1=(1,16)
        self.k3 = 8   # κ3=(1,8)
        self.k4 = 4   # κ4=(1,4)
        self.k5 = 4   # κ5=(1,4)

        # 论文默认是 0~1000ms, fs=128 => T=128；这些 pool 设计要求 T 可被 16 整除
        if self.T % 16 != 0:
            raise ValueError(f"MOCNN expects T divisible by 16 for pooling scheme, got T={self.T}")

        # splits
        F1h, F1m, F1l = alloc_three(self.F1, self.alpha)
        F2h, F2m, F2l = alloc_three(self.F2, self.alpha)
        F3h, F3m, F3l = alloc_three(self.F3, self.alpha)

        # sanity: LastOctConv concat channels should match F5
        if 3 * self.F4 != self.F5:
            raise ValueError(f"Paper uses F5=3*F4, got F4={self.F4}, F5={self.F5}")

        # -------------------------
        # FirstOctConv: p(·) + N1/N2/N3 (spatial dropout)
        # -------------------------
        self.N1 = ConvBNELUDrop(1, F1h, self.k1, p_drop=self.p_drop, spatial=True)
        self.N2 = ConvBNELUDrop(1, F1m, self.k1, p_drop=self.p_drop, spatial=True)
        self.N3 = ConvBNELUDrop(1, F1l, self.k1, p_drop=self.p_drop, spatial=True)

        # -------------------------
        # SpatialConv: D1/D2/D3, κ2=(C,1) (spatial dropout)
        # -------------------------
        self.D1 = DepthwiseSpatialConv(F1h, F2h, C=self.C, p_drop=self.p_drop)
        self.D2 = DepthwiseSpatialConv(F1m, F2m, C=self.C, p_drop=self.p_drop)
        self.D3 = DepthwiseSpatialConv(F1l, F2l, C=self.C, p_drop=self.p_drop)

        # -------------------------
        # CoreOctConv: 9 paths, κ3=(1,8)
        # BN/ELU/SpatialDropout at each destination (h/m/l)
        # -------------------------
        def conv(in_ch, out_ch):
            return nn.Conv2d(in_ch, out_ch, kernel_size=(1, self.k3), bias=False, padding=0)

        # h->*
        self.hh = conv(F2h, F3h)
        self.hm = conv(F2h, F3m)
        self.hl = conv(F2h, F3l)
        # m->*
        self.mh = conv(F2m, F3h)
        self.mm = conv(F2m, F3m)
        self.ml = conv(F2m, F3l)
        # l->*
        self.lh = conv(F2l, F3h)
        self.lm = conv(F2l, F3m)
        self.ll = conv(F2l, F3l)

        self.bn_h = nn.BatchNorm2d(F3h)
        self.bn_m = nn.BatchNorm2d(F3m)
        self.bn_l = nn.BatchNorm2d(F3l)
        self.act = nn.ELU()
        self.drop_h = SpatialDropout2D(self.p_drop)
        self.drop_m = SpatialDropout2D(self.p_drop)
        self.drop_l = SpatialDropout2D(self.p_drop)

        # -------------------------
        # LastOctConv: N4/N5/N6 -> pool align -> concat (spatial dropout)
        # κ4=(1,4), output channels = F4=2 each branch
        # -------------------------
        self.N4 = ConvBNELUDrop(F3h, self.F4, self.k4, p_drop=self.p_drop, spatial=True)
        self.N5 = ConvBNELUDrop(F3m, self.F4, self.k4, p_drop=self.p_drop, spatial=True)
        self.N6 = ConvBNELUDrop(F3l, self.F4, self.k4, p_drop=self.p_drop, spatial=True)

        # -------------------------
        # Classification: S1 separable conv κ5=(1,4)
        # dropout after S1 is NOT spatial (paper)
        # -------------------------
        self.S1 = SeparableConvBNELU(self.F5, self.k5)
        self.drop_after_S1 = nn.Dropout(self.p_drop)  # element-wise

        # FC: infer in_features by dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.C, self.T)
            flat = self._forward_to_flat(dummy)
            fc_in = flat.shape[-1]  # expected 48 when T=128
        self.fc = nn.Linear(fc_in, self.n_class)

    def _g(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        x = same_pad_time(x, self.k3)
        return conv(x)

    def _core_octconv(self, xh, xm, xl):
        """
        输入:
          xh: [B,F2h,1,T]
          xm: [B,F2m,1,T/2]
          xl: [B,F2l,1,T/4]
        输出:
          yh: [B,F3h,1,T/4]
          ym: [B,F3m,1,T/8]
          yl: [B,F3l,1,T/16]
        """
        # destination high: /4
        hh = self._g(avgpool_time(xh, 4), self.hh)
        mh = self._g(avgpool_time(xm, 2), self.mh)
        lh = self._g(xl, self.lh)
        yh = hh + mh + lh
        yh = self.drop_h(self.act(self.bn_h(yh)))

        # destination medium: /8
        hm = self._g(avgpool_time(xh, 8), self.hm)
        mm = self._g(avgpool_time(xm, 4), self.mm)
        lm = self._g(avgpool_time(xl, 2), self.lm)
        ym = hm + mm + lm
        ym = self.drop_m(self.act(self.bn_m(ym)))

        # destination low: /16
        hl = self._g(avgpool_time(xh, 16), self.hl)
        ml = self._g(avgpool_time(xm, 8), self.ml)
        ll = self._g(avgpool_time(xl, 4), self.ll)
        yl = hl + ml + ll
        yl = self.drop_l(self.act(self.bn_l(yl)))

        return yh, ym, yl

    def _forward_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,C,T]
        返回 flatten 后特征: [B, F5 * (T/16)]
        """
        # -------- FirstOctConv: decomposition --------
        xh1 = x                      # [B,1,C,T]
        xm1 = avgpool_time(x, 2)     # [B,1,C,T/2]
        xl1 = avgpool_time(x, 4)     # [B,1,C,T/4]

        xh2 = self.N1(xh1)           # [B,F1h,C,T]
        xm2 = self.N2(xm1)           # [B,F1m,C,T/2]
        xl2 = self.N3(xl1)           # [B,F1l,C,T/4]

        # -------- SpatialConv: κ2=(C,1), C->1 --------
        xh3 = self.D1(xh2)           # [B,F2h,1,T]
        xm3 = self.D2(xm2)           # [B,F2m,1,T/2]
        xl3 = self.D3(xl2)           # [B,F2l,1,T/4]

        # -------- CoreOctConv: κ3=(1,8) --------
        xh4, xm4, xl4 = self._core_octconv(xh3, xm3, xl3)  # [B,F3h,1,T/4], [B,F3m,1,T/8], [B,F3l,1,T/16]

        # -------- LastOctConv: κ4=(1,4), output F4 each --------
        yh = self.N4(xh4)            # [B,F4,1,T/4]
        ym = self.N5(xm4)            # [B,F4,1,T/8]
        yl = self.N6(xl4)            # [B,F4,1,T/16]

        # align time to T/16
        yh = avgpool_time(yh, 4)     # T/4  -> T/16
        ym = avgpool_time(ym, 2)     # T/8  -> T/16
        # yl already T/16

        x5 = torch.cat([yh, ym, yl], dim=1)  # [B,3*F4=F5,1,T/16]

        # -------- Classification: S1 κ5=(1,4) --------
        x5 = self.S1(x5)                  # [B,F5,1,T/16]
        x5 = self.drop_after_S1(x5)       # NOT spatial dropout (paper)

        flat = x5.flatten(start_dim=1)    # [B, F5*(T/16)]
        return flat

    def forward(self, x: torch.Tensor, train_stage: int = 2, **kwargs):
        """
        x: [B,1,C,T]
        返回:
          logits: [B,n_class]
          probs:  [B,n_class]
        """
        B, one, C, T = x.shape
        assert one == 1, f"Expected x shape [B,1,C,T], got second dim={one}"
        assert C == self.C, f"Expected C={self.C}, got {C}"
        assert T == self.T, f"Expected T={self.T}, got {T}"

        flat = self._forward_to_flat(x)
        logits = self.fc(flat)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


if __name__ == "__main__":
    args = {
        "n_channels": 62,
        "fs": 128,
        "seq_len": 128,
        "n_class": 2,
        # paper defaults (可不写)
        "alpha": 0.75,
        "dropout": 0.5,
        "F1": 16, "F2": 128, "F3": 32, "F4": 2, "F5": 6,
    }

    model = Model(args)
    input_shape = (1, 1, args["n_channels"], args["seq_len"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary(model, input_shape)

    # quick run
    # x = torch.randn(*input_shape).to(device)
    # logits, probs = model(x)
    # print(logits.shape, probs.shape)
