import torch
import torch.nn as nn
from Models.EEGNet import calculate_outsize
from torchinfo import summary
from Utils.config import load_config


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]

        self.output_size = 16

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.output_size, kernel_size=(1, 7)),
            nn.Conv2d(self.output_size, self.output_size, kernel_size=(self.n_channels, 1)),
            nn.BatchNorm2d(self.output_size),
            nn.GELU(),
            nn.AvgPool2d((1, 17), (1, 3)),
            nn.Dropout(0.5),
        )
        self.BasicBlockOutputSize = calculate_outsize(self.feature_extractor, self.n_channels, self.fs)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.BasicBlockOutputSize, 512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.n_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.projection(x)
        feats = x
        x = self.classifier(x)
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
