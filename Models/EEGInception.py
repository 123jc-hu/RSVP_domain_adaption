import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]

        self.dropout_rate = 0.25
        self.kernel_length_list1 = [int(self.fs // i) for i in (2, 4, 8)]
        self.kernel_length_list2 = [k // 4 for k in self.kernel_length_list1]

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

        self.Inception2_branch1 = nn.Sequential(
            nn.Conv2d(48, 8, (1, self.kernel_length_list2[0]), stride=1, bias=False, padding=(0, self.kernel_length_list2[0] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception2_branch2 = nn.Sequential(
            nn.Conv2d(48, 8, (1, self.kernel_length_list2[1]), stride=1, bias=False, padding=(0, self.kernel_length_list2[1] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception2_branch3 = nn.Sequential(
            nn.Conv2d(48, 8, (1, self.kernel_length_list2[2]), stride=1, bias=False, padding=(0, self.kernel_length_list2[2] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )

        self.output_module = nn.Sequential(
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Conv2d(24, 12, (1, 8), stride=1, bias=False, padding=(0, 4)),
            nn.BatchNorm2d(12),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(12, 6, (1, 4), stride=1, bias=False, padding=(0, 2)),
            nn.BatchNorm2d(6),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=self.dropout_rate),
            nn.Flatten(),
            nn.Linear(24, self.n_classes),
        )

        self.AvgPool2d = nn.AvgPool2d((1, 4), stride=(1, 4))

    def forward(self, x, train_stage: int = 2):
        _ = train_stage
        x1 = self.Inception1_branch1(x)
        x2 = self.Inception1_branch2(x)
        x3 = self.Inception1_branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.AvgPool2d(x)

        x1 = self.Inception2_branch1(x)
        x2 = self.Inception2_branch2(x)
        x3 = self.Inception2_branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        logits = self.output_module(x)
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("EEGInception", Model)
