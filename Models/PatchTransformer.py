import torch
import torch.nn as nn
from einops import rearrange

class Model(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.n_channels = args.n_channels
        self.n_timepoints = args.n_timepoints
        self.n_classes = args.n_classes
        self.dim = 32
        self.heads = 1
        self.patch_size = 16
        self.patch_embed = nn.Linear(self.patch_size * self.n_channels, self.dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.heads, dim_feedforward=64),
            num_layers=1
        )
        self.fc = nn.Linear(self.dim, self.n_classes)

    def forward(self, x):
        # 将时间序列分块
        x = rearrange(x, 'b c (t p) -> b t (c p)', p=self.patch_size)  # (batch, timepoints/patch_size, dim)
        x = self.patch_embed(x)  # (batch, timepoints/patch_size, dim)
        x = self.transformer(x)  # (batch, timepoints/patch_size, dim)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.fc(x)  # (batch, n_classes)
        return x