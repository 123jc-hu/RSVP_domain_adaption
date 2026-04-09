import math

import torch
from torch import nn

from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize


class TangentVectorizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        cov_eps: float = 1.0e-4,
        cov_shrinkage_alpha: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_filters = int(n_filters)
        self.cov_eps = float(cov_eps)
        self.cov_shrinkage_alpha = float(max(0.0, min(1.0, cov_shrinkage_alpha)))

        self.channel_projector = None
        if self.in_channels != self.n_filters:
            self.channel_projector = nn.Conv2d(
                self.in_channels,
                self.n_filters,
                kernel_size=(1, 1),
                stride=1,
                bias=False,
            )

        tri = torch.triu_indices(self.n_filters, self.n_filters)
        scale = torch.ones(int(tri.shape[1]), dtype=torch.float32)
        off_diag = tri[0] != tri[1]
        scale[off_diag] = math.sqrt(2.0)
        self.register_buffer("tri_row", tri[0], persistent=False)
        self.register_buffer("tri_col", tri[1], persistent=False)
        self.register_buffer("tri_scale", scale, persistent=False)
        self.out_dim = int(self.n_filters * (self.n_filters + 1) // 2)

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if int(x.shape[2]) != 1:
                raise ValueError(f"Expected feature map height=1 before covariance pooling, got shape={tuple(x.shape)}")
            x = x.squeeze(2)
        if x.ndim != 3:
            raise ValueError(f"Expected feature map with 3 dims after squeeze, got shape={tuple(x.shape)}")

        x = x.float()
        x = x - x.mean(dim=-1, keepdim=True)
        denom = max(int(x.shape[-1]) - 1, 1)
        cov = torch.matmul(x, x.transpose(1, 2)) / float(denom)
        if self.cov_shrinkage_alpha > 0.0:
            mu = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / float(cov.shape[-1])
            eye = torch.eye(int(cov.shape[-1]), device=cov.device, dtype=cov.dtype).unsqueeze(0)
            cov = (1.0 - self.cov_shrinkage_alpha) * cov + self.cov_shrinkage_alpha * mu.unsqueeze(-1) * eye
        eye = torch.eye(int(cov.shape[-1]), device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = 0.5 * (cov + cov.transpose(1, 2)) + float(self.cov_eps) * eye
        return cov

    def _log_euclidean_vectorize(self, cov: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=float(self.cov_eps))
        log_cov = eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(1, 2)
        log_cov = 0.5 * (log_cov + log_cov.transpose(1, 2))
        vec = log_cov[:, self.tri_row, self.tri_col]
        return vec * self.tri_scale.to(device=vec.device, dtype=vec.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_projector is not None:
            x = self.channel_projector(x)
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            cov = self._covariance(x)
            return self._log_euclidean_vectorize(cov)


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16

        self.n_channels = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.n_classes = int(config["n_class"])

        self.kernel_length = self.fs // 2
        self.kernel_length2 = self.fs // 8
        self.dropout_rate = 0.5
        self.ts_cov_eps = float(config.get("eegnet_ts_cov_eps", 1.0e-4))
        self.ts_cov_shrinkage_alpha = float(config.get("eegnet_ts_cov_shrinkage_alpha", 0.0))
        self.ts_head_channels = int(config.get("eegnet_ts_head_channels", self.F2))

        self.block1 = nn.Sequential(
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

        self.block2_pre = nn.Sequential(
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
        )
        self.block2_activation = nn.ELU()
        self.flat_post = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate),
        )

        flat_out_size = calculate_outsize(
            nn.Sequential(self.block1, self.block2_pre, self.block2_activation, self.flat_post),
            self.n_channels,
            self.fs,
        )
        self.flat_classifier = DenseWithConstraint(flat_out_size, self.n_classes, bias=False, max_norm=0.25)
        self.ts_vectorizer = TangentVectorizer(
            in_channels=self.F2,
            n_filters=self.ts_head_channels,
            cov_eps=self.ts_cov_eps,
            cov_shrinkage_alpha=self.ts_cov_shrinkage_alpha,
        )

    def _shared_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2_pre(x)
        x = self.block2_activation(x)
        return x

    def _flat_features(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = self.flat_post(feature_map)
        return torch.flatten(x, start_dim=1)

    def forward(self, x, train_stage: int = 2, return_features: bool = False):
        _ = train_stage
        feature_map = self._shared_feature_map(x)
        flat_features = self._flat_features(feature_map)
        logits = self.flat_classifier(flat_features)
        ts_features = self.ts_vectorizer(feature_map)
        if return_features:
            return logits, {
                "features": flat_features,
                "flat_features": flat_features,
                "ts_features": ts_features,
                "feature_map": feature_map,
            }
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("EEGNetAuxTS", Model)
