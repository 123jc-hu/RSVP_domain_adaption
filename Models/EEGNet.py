import torch
from torch import nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(input)


class DenseWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(input)


def calculate_outsize(model: nn.Module, channels: int, samples: int) -> int:
    """Calculate flattened output size for input (1, 1, channels, samples)."""
    data = torch.rand(1, 1, channels, samples)
    model.eval()
    out = model(data).shape
    return out.numel()


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16

        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]

        self.kernel_length = self.fs // 2
        self.kernel_length2 = self.fs // 8
        self.dropout_rate = 0.5

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        block1 = nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
            nn.BatchNorm2d(self.F1),
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.n_channels, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
            ),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=self.dropout_rate),
        )

        block2 = nn.Sequential(
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, self.kernel_length2),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length2 // 2),
                groups=self.F1 * self.D,
            ),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate),
        )
        return nn.Sequential(block1, block2)

    def classifier_block(self, input_size: int):
        return nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(input_size, self.n_classes, bias=False, max_norm=0.25),
        )

    def forward(self, x, train_stage: int = 2):
        _ = train_stage
        x = self.BasicBlock(x)
        x = self.ClassifierBlock(x)
        return x


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("EEGNet", Model)
