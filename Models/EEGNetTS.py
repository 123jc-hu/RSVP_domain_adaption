import math

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


class TangentSpaceHead(nn.Module):
    """
    Covariance pooling + log-Euclidean tangent-space mapping.

    Input:
    - feature map of shape (B, C, 1, T) or (B, C, T)

    Output:
    - tangent vector with upper-triangular vectorization and sqrt(2) scaling
      on off-diagonal terms, preserving the Frobenius inner product.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        n_classes: int,
        cov_eps: float = 1.0e-4,
        cov_shrinkage_alpha: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_filters = int(n_filters)
        self.n_classes = int(n_classes)
        self.cov_eps = float(cov_eps)
        self.cov_shrinkage_alpha = float(max(0.0, min(1.0, cov_shrinkage_alpha)))

        if self.n_filters <= 0:
            raise ValueError("n_filters must be > 0 for TangentSpaceHead.")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0 for TangentSpaceHead.")

        self.channel_projector = None
        if self.in_channels != self.n_filters:
            # Only the tangent-space branch changes its working dimension.
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

        out_dim = int(self.n_filters * (self.n_filters + 1) // 2)
        self.out_dim = out_dim
        self.classifier = DenseWithConstraint(out_dim, self.n_classes, bias=False, max_norm=0.25)

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if int(x.shape[2]) != 1:
                raise ValueError(f"Expected feature map height=1 before covariance pooling, got shape={tuple(x.shape)}")
            x = x.squeeze(2)
        if x.ndim != 3:
            raise ValueError(f"Expected feature map with 3 dims after squeeze, got shape={tuple(x.shape)}")

        # Eigen decomposition is not implemented for fp16 on CUDA.
        # Keep the tangent-space branch in fp32 for numerical stability.
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

    def forward(self, x: torch.Tensor):
        if self.channel_projector is not None:
            x = self.channel_projector(x)
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            cov = self._covariance(x)
            ts_features = self._log_euclidean_vectorize(cov)
        logits = self.classifier(ts_features)
        return logits, ts_features


class Model(nn.Module):
    """
    EEGNet backbone with a tangent-space classification head.

    Design choice:
    - keep EEGNet's temporal + spatial + separable convolutional frontend
    - extract feature maps before the final aggressive temporal pooling
    - compute per-sample covariance over the feature-map time axis
    - classify and align in the same tangent-space vector
    """

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
        self.ts_feature_layer = str(config.get("eegnet_ts_feature_layer", "block2")).strip().lower()

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

        self.tangent_head = TangentSpaceHead(
            in_channels=(self.F1 * self.D if self.ts_feature_layer == "block1" else self.F2),
            n_filters=self.ts_head_channels,
            n_classes=self.n_classes,
            cov_eps=self.ts_cov_eps,
            cov_shrinkage_alpha=self.ts_cov_shrinkage_alpha,
        )

    def _forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        if self.ts_feature_layer == "block1":
            return x
        if self.ts_feature_layer == "block2":
            x = self.block2_pre(x)
            x = self.block2_activation(x)
            return x
        raise ValueError(f"Unsupported eegnet_ts_feature_layer={self.ts_feature_layer}. Use block1 or block2.")

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self._forward_feature_map(x)
        _, ts_features = self.tangent_head(feature_map)
        return ts_features

    def forward(self, x, train_stage: int = 2, return_features: bool = False):
        _ = train_stage
        feature_map = self._forward_feature_map(x)
        logits, ts_features = self.tangent_head(feature_map)
        if return_features:
            return logits, {"features": ts_features, "feature_map": feature_map}
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("EEGNetTS", Model)
