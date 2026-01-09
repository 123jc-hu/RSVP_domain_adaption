from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# -------------------------
# SAME padding (time axis) for Conv2d with kernel=(1, k)
# Supports even/odd k and keeps output length unchanged (stride=1, dilation=1)
# -------------------------
def same_pad_time_2d(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: [B, C, H, T]
    if k <= 1:
        return x
    # For SAME: left = floor((k-1)/2), right = ceil((k-1)/2)
    left = (k - 1) // 2
    right = (k - 1) - left
    return F.pad(x, (left, right, 0, 0))


class ConvBNELUDrop(nn.Module):
    """Conv2d(1,k) with true SAME padding + BN + ELU + Dropout."""
    def __init__(self, in_ch, out_ch, k_t, p_drop=0.5, spatial_drop=True, groups=1):
        super().__init__()
        self.k_t = int(k_t)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, self.k_t),
            groups=groups,
            bias=False,
            padding=0
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(inplace=True)
        self.drop = nn.Dropout2d(p_drop) if spatial_drop else nn.Dropout(p_drop)

    def forward(self, x):
        x = same_pad_time_2d(x, self.k_t)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MDCBank(nn.Module):
    """
    Multi-Scale Convolution Bank (paper calls it MDC bank).
    Input:  [B, in_ch, a, b]
    Output: [B, n*f,  a, b]  (concat along channel dim)
    where n = log2(b) - 1, kernel sizes: b/2, b/4, ..., 2
    """
    def __init__(self, in_channels: int, f: int, b: int, p_drop=0.5, use_bn_act_drop=True):
        super().__init__()
        b = int(b)
        n = int(math.log2(b)) - 1
        if 2 ** int(math.log2(b)) != b:
            raise ValueError(f"MDCBank expects b to be power-of-2, got b={b}")

        ks = [b // (2 ** (i + 1)) for i in range(n)]  # b/2, b/4, ..., 2
        self.branches = nn.ModuleList()

        for k in ks:
            if use_bn_act_drop:
                self.branches.append(
                    ConvBNELUDrop(
                        in_ch=in_channels,
                        out_ch=f,
                        k_t=k,
                        p_drop=p_drop,
                        spatial_drop=True,
                        groups=1
                    )
                )
            else:
                # minimal: conv only (still uses correct SAME padding)
                self.branches.append(nn.Conv2d(in_channels, f, kernel_size=(1, k), bias=False, padding=0))

        self.use_bn_act_drop = use_bn_act_drop
        self.ks = ks

    def forward(self, x):
        outs = []
        for layer, k in zip(self.branches, self.ks):
            if self.use_bn_act_drop:
                outs.append(layer(x))
            else:
                outs.append(layer(same_pad_time_2d(x, k)))
        return torch.cat(outs, dim=1)


class Model(nn.Module):
    """
    MDCNet (binary version): remove the 3-class attention module, keep feature extractor + 2-class head.

    config required:
      - n_channels: C
      - fs: 128
      - seq_len: T (建议 = fs = 128)
      - n_class: 2
    optional:
      - f: branch filters in MDC bank (paper uses f as symbol; set default 8)
      - dropout: default 0.5
      - use_bn_act_drop: default True
      - temp1_out, temp2_out: channels of two temporal convs (paper未在你贴的段落里给出，默认 64/128，可自行改)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.C = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.T = int(config.get("seq_len", self.fs))
        self.n_class = int(config.get("n_class", 2))

        self.f = int(config.get("f", 8))
        self.p_drop = float(config.get("dropout", 0.5))
        self.use_bn_act_drop = bool(config.get("use_bn_act_drop", True))

        # temporal conv channels (not specified in your quoted text; keep configurable)
        self.temp1_out = int(config.get("temp1_out", 64))
        self.temp2_out = int(config.get("temp2_out", 128))

        if self.T != self.fs:
            # 论文是 0-1000ms 且 downsample 到 128Hz => T=fs=128
            # 若你改了 T，也能跑，但 MDC bank 的 n=log2(b)-1 要求 b 是 2^k
            pass

        # -------- Feature Extractor --------
        # 1) MDC bank 1 on raw EEG: x in R^{1 x C x T}
        self.mdc1 = MDCBank(in_channels=1, f=self.f, b=self.T, p_drop=self.p_drop, use_bn_act_drop=self.use_bn_act_drop)
        n1 = int(math.log2(self.T)) - 1
        nf1 = n1 * self.f  # output channels of mdc1

        # 2) depthwise separable spatial conv with kernel (C,1): compress spatial dimension C->1
        # depthwise: groups=nf1
        self.spatial_dw = nn.Conv2d(nf1, nf1, kernel_size=(self.C, 1), groups=nf1, bias=False, padding=0)
        self.spatial_pw = nn.Conv2d(nf1, nf1, kernel_size=1, bias=False)

        self.spatial_bn = nn.BatchNorm2d(nf1) if self.use_bn_act_drop else nn.Identity()
        self.spatial_act = nn.ELU(inplace=True) if self.use_bn_act_drop else nn.Identity()
        self.spatial_drop = nn.Dropout2d(self.p_drop) if self.use_bn_act_drop else nn.Identity()

        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))  # T -> T/4

        # 3) MDC bank 2 on reduced time
        self.mdc2 = MDCBank(in_channels=nf1, f=self.f, b=self.T // 4, p_drop=self.p_drop, use_bn_act_drop=self.use_bn_act_drop)
        n2 = int(math.log2(self.T // 4)) - 1
        nf2 = n2 * self.f  # output channels of mdc2

        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # T/4 -> T/8

        # 4) temporal conv layers (1,8) then (1,4), each followed by avgpool(1,2)
        self.temp_conv1 = ConvBNELUDrop(
            in_ch=nf2, out_ch=self.temp1_out, k_t=8,
            p_drop=self.p_drop, spatial_drop=True, groups=1
        ) if self.use_bn_act_drop else nn.Conv2d(nf2, self.temp1_out, kernel_size=(1, 8), bias=False, padding=0)

        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # T/8 -> T/16

        self.temp_conv2 = ConvBNELUDrop(
            in_ch=self.temp1_out, out_ch=self.temp2_out, k_t=4,
            p_drop=self.p_drop, spatial_drop=True, groups=1
        ) if self.use_bn_act_drop else nn.Conv2d(self.temp1_out, self.temp2_out, kernel_size=(1, 4), bias=False, padding=0)

        self.pool4 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # T/16 -> T/32

        # -------- Classifier (binary) --------
        self.flatten = nn.Flatten()

        # infer fc_in
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.C, self.T)
            feat = self._forward_features(dummy)
            fc_in = feat.shape[1]

        self.classifier = nn.Linear(fc_in, self.n_class)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,C,T]
        out = self.mdc1(x)                 # [B,nf1,C,T]
        out = self.spatial_dw(out)         # [B,nf1,1,T]
        out = self.spatial_pw(out)         # [B,nf1,1,T]
        out = self.spatial_bn(out)
        out = self.spatial_act(out)
        out = self.spatial_drop(out)

        out = self.pool1(out)              # [B,nf1,1,T/4]
        out = self.mdc2(out)               # [B,nf2,1,T/4]
        out = self.pool2(out)              # [B,nf2,1,T/8]

        if self.use_bn_act_drop:
            out = self.temp_conv1(out)     # same length (T/8)
        else:
            out = self.temp_conv1(same_pad_time_2d(out, 8))
        out = self.pool3(out)              # [B,*,1,T/16]

        if self.use_bn_act_drop:
            out = self.temp_conv2(out)     # same length (T/16)
        else:
            out = self.temp_conv2(same_pad_time_2d(out, 4))
        out = self.pool4(out)              # [B,temp2_out,1,T/32]

        out = self.flatten(out)            # [B, temp2_out*(T/32)]
        return out

    def forward(self, x: torch.Tensor, **kwargs):
        # x: [B,1,C,T]
        feat = self._forward_features(x)
        logits = self.classifier(feat)     # [B,2]
        return logits, feat


if __name__ == "__main__":
    args = {"n_channels": 62, "fs": 128, "seq_len": 128, "n_class": 2, "f": 8, "dropout": 0.5}
    model = Model(args)
    input_shape = (1, 1, args["n_channels"], args["seq_len"])
    summary(model, input_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary(model, input_shape)
