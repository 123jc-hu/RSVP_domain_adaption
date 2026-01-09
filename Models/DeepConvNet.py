from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize
import torch
from torch import nn
from torchinfo import summary
from Utils.config import load_config


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]
        self.dropout_rate = 0.5

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        Block1 = nn.Sequential(
            Conv2dWithConstraint(
                1,
                25,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            Conv2dWithConstraint(
                25,
                25,
                (self.n_channels, 1),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(25, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )
        Block2 = nn.Sequential(
            Conv2dWithConstraint(
                25,
                50,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(50, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )
        Block3 = nn.Sequential(
            Conv2dWithConstraint(
                50,
                100,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(100, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )
        Block4 = nn.Sequential(
            Conv2dWithConstraint(
                100,
                200,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(200, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )

        return nn.Sequential(Block1, Block2, Block3, Block4)

    def classifier_block(self, input_size):
        module2 = nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(
                input_size,
                self.n_classes,
                max_norm=0.5,
                bias=False),
            # nn.Softmax(dim=1)
        )
        return module2

    def forward(self, x, train_stage: int = 2):
        x = self.BasicBlock(x)
        x = self.ClassifierBlock(x)
        return x


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 1, 62, 128)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 summary 打印模型信息
    summary(model, input_shape, device=str(device))

    # 手动测试模型
    # x = torch.randn(1, 10, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状
