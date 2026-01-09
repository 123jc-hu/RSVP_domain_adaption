import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from Utils.config import load_config
from einops.layers.torch import Rearrange, Reduce


class Model(nn.Module):
    def __init__(self, configs: dict):
        super(Model, self).__init__()
        self.n_channels = configs["n_channels"]
        self.fs = configs["fs"]
        self.n_classes = configs["n_class"]
        self.Windows = 5
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.Windows, out_channels=self.Windows*8,
                      kernel_size=(1, 32), groups=self.Windows, bias=False),
            nn.BatchNorm2d(self.Windows*8),
            nn.ELU(),
        )
        # Block 1
        self.block1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=self.Windows*8, out_channels=self.Windows*8,
                      kernel_size=(1, 16), groups=self.Windows*8, bias=False),
            nn.BatchNorm2d(self.Windows*8),
            nn.ELU(),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.Windows*8, out_channels=self.Windows*8,
                      kernel_size=(self.n_channels, 1), groups=self.Windows*8, bias=False),
            nn.BatchNorm2d(self.Windows*8),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 2)),
            Rearrange("b c h w -> b w h c"),
            nn.Dropout2d(0.5),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 13), padding=0, bias=False, groups=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 3)),
            nn.Dropout(0.5)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 1 * 8, out_features=64),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            Rearrange("b (c h w) -> b c h w", c=16, h=1, w=8),  # reshape to (B, 16, 1, 8)
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 6), stride=(1, 2)),  # (B, 8, 1, 20)
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=(self.n_channels, 1)),  # (B, 8, 62, 20)
            nn.ELU(),
            Rearrange("b c h w -> b h (c w)"),  # reshape to (B, 62, 8*20)
            Rearrange("b h (c w) -> b c h w", c=5, w=32),  # reshape to (B, 5, 62, 32)
            nn.ConvTranspose2d(in_channels=5, out_channels=5, kernel_size=(1, 33), stride=(1, 1)),
            nn.ELU(),

        )

        # Classifier
        self.classifier = nn.Linear(in_features=64, out_features=self.n_classes)

    def forward(self, x):
        # x: (B, C, T) -> (B, 4, 62, 64)
        x = split_window(x).to(x.device)
        x_window = x  # (B, 5, 62, 64)
        x = self.first_conv(x)  # (B, 40, 1, 32)
        x = self.block1(x)  # (B, 40, 1, 16)
        x = self.block2(x)  # (B, 8, 1, 40)
        x = self.block3(x)  # (B, 16, 1, 8)
        x = self.block4(x)  # (B, 64)
        latent = x
        x_rec = self.decoder(latent)
        z = self.classifier(latent)

        return z, x_rec, x_window, latent


def split_window(X):
    X = X.squeeze(dim=1)  # (B, 1, C, T) -> (B, C, T)
    trial, C, T = X.shape
    X_sliding=torch.zeros((trial, 5, C, 64))
    for i in range(5):
        X_sliding[:, i, :, :] = X[:, :, i*16:i*16+64]  # 每个窗口的长度为64，步长为16
    return X_sliding


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 summary 打印模型信息
    summary(model, input_shape, device=str(device))

    # 手动测试模型
    # x = torch.randn(1, 1, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状
