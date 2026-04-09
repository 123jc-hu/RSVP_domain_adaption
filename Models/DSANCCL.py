import torch
from torch import nn


class Model(nn.Module):
    """
    DSAN-CCL backbone from the provided paper description.

    Pipeline:
    temporal conv -> spatial conv -> BN/GELU -> avg pool -> dropout
    -> flatten -> projection(1024) -> classifier(512 -> n_class)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = int(config["n_channels"])
        self.seq_len = int(config["fs"])
        self.n_classes = int(config["n_class"])

        self.num_filters = int(config.get("dsan_num_filters", 64))
        self.temporal_kernel = int(config.get("dsan_temporal_kernel", 13))
        self.pool_kernel = int(config.get("dsan_pool_kernel", 35))
        self.pool_stride = int(config.get("dsan_pool_stride", 7))
        self.feature_dim = int(config.get("dsan_feature_dim", 1024))
        self.classifier_hidden_dim = int(config.get("dsan_classifier_hidden_dim", 512))
        self.dropout_rate = float(config.get("dsan_dropout", 0.5))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=(1, self.temporal_kernel), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(
                self.num_filters,
                self.num_filters,
                kernel_size=(self.n_channels, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(self.num_filters),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_kernel), stride=(1, self.pool_stride), padding=(0, 0)),
            nn.Dropout(p=self.dropout_rate),
        )

        projection_in_dim = self._infer_projection_in_dim()
        self.projection = nn.Linear(projection_in_dim, self.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.classifier_hidden_dim, self.n_classes),
        )

    def _infer_projection_in_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.seq_len)
            out = self.feature_extractor(dummy)
        if out.numel() <= 0:
            raise ValueError(
                "DSAN-CCL feature extractor produced empty output. "
                f"Check seq_len={self.seq_len}, temporal_kernel={self.temporal_kernel}, "
                f"pool_kernel={self.pool_kernel}, pool_stride={self.pool_stride}."
            )
        return int(out.reshape(1, -1).shape[1])

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.projection(x)

    def forward(self, x, train_stage: int = 2, return_features: bool = False):
        _ = train_stage
        features = self._forward_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, {"features": features}
        return logits
