from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from torch import nn
from einops.layers.torch import Rearrange, Reduce
from Models.EEGNet import calculate_outsize
import torch
from torchinfo import summary
from Utils.config import load_config
import torch.nn.functional as F
from Models.Ours_4 import LinearWithConstraint
from torch.autograd import Function


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_class = config["n_class"]
        k_small = 9
        d = max(1, (self.fs//4 - 1) // (k_small - 1))  # 近似匹配 RF
        pad = d * (k_small // 2)

        self.channel_tcb_block = nn.Sequential(
            Rearrange("b k c t -> b c k t"),
            # nn.Conv2d(
            #     self.n_channels,
            #     self.n_channels * 4,
            #     (1, self.fs // 4),
            #     stride=1,
            #     padding=(0, self.fs // 8),
            #     groups=self.n_channels,
            # ),
            nn.Conv2d(self.n_channels, self.n_channels*4, kernel_size=(1, k_small), stride=(1, 1),
                padding=(0, pad), dilation=(1, d), groups=self.n_channels, bias=False),
            nn.BatchNorm2d(self.n_channels * 4),
            nn.GELU(),
        )

        self.spatiotemporal_fusion_block = nn.Sequential(
            Rearrange("b (c h) k t -> b h c (k t)", h=4),
            nn.Conv2d(4, 16, kernel_size=(self.n_channels, 1)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3),
        )

        self.output_size = 16 * (self.fs // 8)

        self.primary_classifier = LinearWithConstraint(self.output_size, self.n_class, max_norm=1, doWeightNorm=True)

    def forward(self, x, train_stage=2):
        x = self.channel_tcb_block(x)
        x = self.spatiotemporal_fusion_block(x)

        feat = F.normalize(x.flatten(start_dim=1), p=2, dim=-1)
        logits = self.primary_classifier(x.flatten(start_dim=1))

        return logits, feat


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