import torch
import torch.nn as nn
from torchinfo import summary
from Utils.config import load_config


class Square(nn.Module):
    def forward(self, x):
        return torch.pow(x, 2)


class Logarithm(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(Logarithm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.epsilon))


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]

        self.feature_extract_block = nn.Sequential(
            nn.Conv2d(1, 40, (1, 12)),
            nn.Conv2d(40, 40, (self.n_channels, 1)),
            nn.BatchNorm2d(40),
            Square(),
            nn.AvgPool2d((1, 37), (1, 7)),
            Logarithm(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Conv2d(40, self.n_classes, (1, 12))

    def forward(self, x):
        x = self.feature_extract_block(x)
        feats = x.flatten(start_dim=1)
        x = self.classifier(x)
        x = x.squeeze()
        return x, feats


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
