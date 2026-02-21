import torch
import torch.nn as nn

from Models.EEGNet import calculate_outsize


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]

        self.Block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 4), dilation=(1, 4)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 8), dilation=(1, 8)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 16), dilation=(1, 16)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 32), dilation=(1, 32)),
            nn.BatchNorm2d(8),
            nn.ELU(),
        )

        self.Block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(self.n_channels, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.BasicBlockOutputSize = calculate_outsize(
            nn.Sequential(self.Block1, self.Block2), self.n_channels, self.fs
        )

        self.ClassifierBlock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.BasicBlockOutputSize, self.n_classes),
        )

    def forward(self, x, train_stage: int = 2):
        _ = train_stage
        x = self.Block1(x)
        x = self.Block2(x)
        logits = self.ClassifierBlock(x)
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("PPNN", Model)
