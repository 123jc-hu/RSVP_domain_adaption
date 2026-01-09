"""来自论文《Convolutional neural network for multi-category rapid serial visual presentation BCI》
原文输入shape=(64, 64) 所以按照64Hz进行搭建 与原文保持一致"""

import torch
from torch import nn
import torch.nn.functional as F
from Models.EEGNet import calculate_outsize
from torchsummary import summary
from Utils.config import load_config


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_class = configs.n_class

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        return nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(self.n_channels, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Conv2d(96, 128, kernel_size=(1, 6), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Conv2d(128, 128, kernel_size=(1, 6), stride=(1, 1)),
            nn.ReLU(),
        )
    
    def classifier_block(self, input_size):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.n_class),
        )
    
    def forward(self, x):
        x = self.BasicBlock(x)
        x = self.ClassifierBlock(x)
        return x, None


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 62, 64)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)