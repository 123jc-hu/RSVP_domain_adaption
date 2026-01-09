from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from torch import nn
from einops.layers.torch import Rearrange
from Models.Ours_4 import LinearWithConstraint
from torchinfo import summary


class Model(nn.Module):
    """
    PyTorch 版 Encoder，对应原论文 Encoder 结构。

    约定输入:
        x: [B, 1, C, T]
           比如 B=1, C=62, T=128

    在 forward 内部：
        - 按时间维滑窗: win_len = fs//2 (=64), step = fs//8 (=16)
        - 得到形状 [B, n_win, C, win_len]，这里 n_win=5
        - 再送入 Block1 ~ Block4
    """

    def __init__(self, config: dict):
        super().__init__()

        # --------- 基本配置 ----------
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]                                           # 128
        self.input_T = config.get("input_T", self.fs)                    # 128

        # 滑窗参数：原文是 win = fs//2, stride = fs//8
        self.win_len = config.get("window_size", self.fs // 2)           # 64
        self.step = config.get("window_stride", self.fs // 8)            # 16

        # 根据输入长度算窗口数（比如 128 -> 5）
        self.n_windows = (self.input_T - self.win_len) // self.step + 1  # 5

        # 第一个 depthwise block 的输出通道数: n_win * depth_multiplier
        c1 = self.n_windows * 8  # 5 * 8 = 40

        # ---------------- Block1: 两个 depthwise 卷积 ----------------
        # 对应：
        #   DepthwiseConv2D((1, 32), depth_multiplier=8)
        #   BN + ELU + Dropout
        #   DepthwiseConv2D((1, 15), depth_multiplier=1)
        #   BN + ELU
        self.block1 = nn.Sequential(
            # 第一个 depthwise：沿 window 维做 depthwise conv
            nn.Conv2d(
                in_channels=self.n_windows,        # 5
                out_channels=c1,                   # 40 = 5*8
                kernel_size=(1, 32),
                groups=self.n_windows,             # depthwise over window 维
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            nn.ELU(),
            nn.Dropout(0.5),

            # 第二个 depthwise：kernel = (1, 15)
            nn.Conv2d(
                in_channels=c1,
                out_channels=c1,
                kernel_size=(1, 15),
                groups=c1,                         # depthwise
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            nn.ELU(),
        )

        # ---------------- Block2: 跨通道 depthwise + Pool + Reshape + SpatialDropout ----------------
        # 对应：
        #   DepthwiseConv2D((C, 1), depth_multiplier=1)
        #   BN + ELU
        #   AveragePooling2D((1, 8), strides=(1, 4))
        #   Reshape((1, ..., ...)) + SpatialDropout2D
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=c1,
                out_channels=c1,
                kernel_size=(self.n_channels, 1),  # (C, 1)：跨空间通道
                groups=c1,
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 2)),
            # [B, c1, 1, W] -> [B, W, 1, c1]，等价于 Keras 里的 Reshape((1, 48, 8))
            Rearrange("b c h w -> b w h c"),
            nn.Dropout2d(0.5),                    # SpatialDropout2D
        )

        # ---------------- Block3: SeparableConv2D + BN + ELU + AvgPool + Dropout ----------------
        # 对应：
        #   SeparableConv2D(16, (1, 16))
        #   BN + ELU
        #   AveragePooling2D((1, 12), strides=(1, 3))
        #   Dropout(0.5)
        #
        # SeparableConv2D = depthwise(1x16) + pointwise(1x1, 16 filters)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(1, 16),
                bias=False,
                groups=8
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 12),
                stride=(1, 3),
            ),
            nn.Dropout(0.5),
        )

        # ---------------- Block4: Flatten + Dense(64, max_norm=0.5) ----------------
        self.flatten = nn.Flatten()

        # 用 dummy 输入走一遍 conv，自动算出 FC 输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.input_T)
            feat = self._forward_conv(dummy)
            fc_in = feat.shape[1]

        self.fc_latent = LinearWithConstraint(fc_in, 64, max_norm=0.5, doWeightNorm=True)
        self.classifier = LinearWithConstraint(64, 2, max_norm=0.5, doWeightNorm=True)
    # ---------- 滑窗函数：在时间维上做窗口划分 ----------
    def slide_windows(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, C, T]
        返回:
            [B, n_win, C, win_len]，例如 [1, 5, 62, 64]
        """
        B, one, C, T = x.shape
        assert one == 1, f"expect channel=1, got {one}"
        win = self.win_len
        step = self.step

        # [B, 1, C, T] -> [B, C, T]
        x = x.squeeze(1)

        # 在最后一维 (时间) 上 unfold： [B, C, n_win, win_len]
        x_unf = x.unfold(dimension=-1, size=win, step=step)
        B, C, n_win, win_len = x_unf.shape
        assert n_win == self.n_windows, f"n_win={n_win}, but configured={self.n_windows}"

        # 调整成 [B, n_win, C, win_len]，让 window 维当作 conv 的“通道”
        x_win = x_unf.permute(0, 2, 1, 3).contiguous()
        return x_win

    # ---------- conv 主干，不含最后一层 FC ----------
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        # 先滑窗: [B, 1, C, T] -> [B, n_win, C, win_len]
        x = self.slide_windows(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor, train_stage: int = 2):
        """
        x: [B, 1, C, T]
        返回:
            encoder_out: [B, 64]
            feat:        [B, F]  (Flatten 后的特征)
        """
        feat = self._forward_conv(x)
        encoder_out = self.fc_latent(feat)
        classifier_out = self.classifier(encoder_out)

        return classifier_out, encoder_out


if __name__ == "__main__":
    # 简单自测
    args = {
        "n_channels": 62,
        "fs": 128,
        "input_T": 128,       # 输入时间长度
        "window_size": 64,
        "window_stride": 16,
    }
    model = Model(args)

    # x = torch.randn(1, 1, 62, 128)
    # y, feat = model(x)
    # print("encoder_out:", y.shape)  # [1, 64]
    # print("feat:", feat.shape)      # [1, F]
    input_shape = (1, 1, args["n_channels"], args["input_T"])  # 输入形状 (batch_size, n_channels, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary(model, input_shape)