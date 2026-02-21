import torch
from torch import nn

from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize


class _RearrangeBKCToBTCK(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 2, 1).contiguous()


class _RearrangeBTCKToBKCT(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 2, 1).contiguous()


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        block1 = nn.Sequential(
            Conv2dWithConstraint(
                1,
                8,
                (1, self.fs // 4),
                max_norm=0.5,
                stride=(1, self.fs // 32),
                bias=False,
            ),
            nn.BatchNorm2d(8),
        )

        block2 = nn.Sequential(
            _RearrangeBKCToBTCK(),
            Conv2dWithConstraint(
                25,
                25,
                (self.n_channels, 1),
                max_norm=0.5,
                stride=(1, 1),
                bias=False,
                groups=25,
            ),
            _RearrangeBTCKToBKCT(),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout2d(p=0.25),
        )

        block3 = nn.Sequential(
            Conv2dWithConstraint(8, 8, (1, 9), max_norm=0.5, stride=(1, 1), bias=False, groups=8),
            Conv2dWithConstraint(8, 16, 1, stride=(1, 1), bias=False, max_norm=0.5),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 17)),
            nn.Dropout2d(p=0.5),
        )

        return nn.Sequential(block1, block2, block3)

    def classifier_block(self, input_size: int):
        return nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(input_size, self.n_classes, bias=False, max_norm=0.1),
        )

    def forward(self, x, train_stage: int = 2):
        _ = train_stage
        x = self.BasicBlock(x)
        logits = self.ClassifierBlock(x)
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("PLNet", Model)
