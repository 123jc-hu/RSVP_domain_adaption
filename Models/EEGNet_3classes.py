import torch
from torch import nn
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_classes = configs.n_class
        self.kernel_length = self.fs // 2
        self.kernel_length2 = self.fs // 8
        self.dropout_rate = configs.dropout_rate

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        Block1 = nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
            nn.BatchNorm2d(self.F1),
            # DepthwiseConv2D =======================
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.n_channels, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
            ),
            # ========================================
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=self.dropout_rate)
        )

        Block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, self.kernel_length2),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length2 // 2),
                groups=self.F1 * self.D
            ),
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
            ),
            # ========================================
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate)
        )
        return nn.Sequential(Block1, Block2)

    def classifier_block(self, input_size):
        return nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(
                input_size,
                self.n_classes,
                bias=False,
                max_norm=0.25),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.BasicBlock(x)
        x = self.ClassifierBlock(x)
        return x, None
