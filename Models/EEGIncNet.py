import torch
from torch import nn

from Models.EEGNet import DenseWithConstraint


class Model(nn.Module):
    """
    Hybrid backbone:
    - use EEGInception first multi-scale stage
    - after first concat, switch to EEGNet-style block2
    """

    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.n_classes = int(config["n_class"])

        self.dropout_rate = 0.25
        self.kernel_length_list1 = [int(self.fs // i) for i in (2, 4, 8)]
        self.kernel_length2 = int(self.fs // 8)

        self.Inception1_branch1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, self.kernel_length_list1[0]), stride=1, padding=(0, self.kernel_length_list1[0] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(8, 16, (self.n_channels, 1), stride=1, bias=False, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception1_branch2 = nn.Sequential(
            nn.Conv2d(1, 8, (1, self.kernel_length_list1[1]), stride=1, padding=(0, self.kernel_length_list1[1] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(8, 16, (self.n_channels, 1), stride=1, bias=False, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception1_branch3 = nn.Sequential(
            nn.Conv2d(1, 8, (1, self.kernel_length_list1[2]), stride=1, padding=(0, self.kernel_length_list1[2] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(8, 16, (self.n_channels, 1), stride=1, bias=False, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )

        self.post_concat_pool = nn.AvgPool2d((1, 4), stride=(1, 4))
        self.eegnet_block2 = nn.Sequential(
            nn.Conv2d(
                48,
                48,
                (1, self.kernel_length2),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length2 // 2),
                groups=48,
            ),
            nn.Conv2d(48, 16, (1, 1), stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate),
        )

        self.BasicBlockOutputSize = self._calculate_output_size()
        self.ClassifierBlock = DenseWithConstraint(self.BasicBlockOutputSize, self.n_classes, bias=False, max_norm=0.25)

    def _forward_basicblock(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Inception1_branch1(x)
        x2 = self.Inception1_branch2(x)
        x3 = self.Inception1_branch3(x)
        min_width = min(x1.size(-1), x2.size(-1), x3.size(-1))
        x1 = x1[..., :min_width]
        x2 = x2[..., :min_width]
        x3 = x3[..., :min_width]
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.post_concat_pool(x)
        x = self.eegnet_block2(x)
        return x

    def _calculate_output_size(self) -> int:
        with torch.no_grad():
            data = torch.rand(1, 1, self.n_channels, self.fs)
            out = self._forward_basicblock(data).shape
        return out.numel()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_basicblock(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x, train_stage: int = 2, return_features: bool = False):
        _ = train_stage
        features = self._forward_features(x)
        logits = self.ClassifierBlock(features)
        if return_features:
            return logits, {"features": features}
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("EEGIncNet", Model)
