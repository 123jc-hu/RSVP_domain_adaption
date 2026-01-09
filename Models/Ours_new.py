from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops.layers.torch import Rearrange


class MaxNormLinear(nn.Linear):
    """Linear layer with max-norm constraint on each output unit (row-wise)."""
    def __init__(self, *args, max_norm: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # dim=0 -> constrain each row vector (each output neuron)
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class ZeroChannelGate(nn.Module):
    """
    Mask-aware channel gating (selector-friendly).
    - Computes per-channel energy (RMS) and predicts a conservative gain near 1.
    - Multiplies by a binary mask to explicitly suppress invalid/zero-augmented channels.
    - Optional renormalization keeps the average gain invariant across different K.
    Inputs:
        x:    [B, C, T]
        mask: [B, C] or [C] (1=valid electrode, 0=invalid/zero channel)
    Outputs:
        x_gated: [B, C, T]
        gate:    [B, C]
    """
    def __init__(self, hidden_dim: int = 8, alpha: float = 0.1, eps: float = 1e-6, renorm: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.renorm = bool(renorm)

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, C, T = x.shape

        if mask is None:
            mask = x.new_ones(B, C)
        else:
            # Accept mask shape [C] or [B,C]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            mask = mask.to(device=x.device, dtype=x.dtype)

        # Per-channel RMS energy (robust for zero channels)
        energy = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)  # [B,C,1]
        delta = self.mlp(energy)                                            # [B,C,1]

        # Conservative gain around 1: approx in (1-alpha, 1+alpha)
        gate = 1.0 + self.alpha * torch.tanh(delta)                         # [B,C,1]
        gate = gate.squeeze(-1) * mask                                      # [B,C]

        # Keep scale comparable across different numbers of valid channels
        if self.renorm:
            num_valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)        # [B,1]
            sum_gate  = gate.sum(dim=1, keepdim=True).clamp_min(self.eps)   # [B,1]
            gate = gate * (num_valid / sum_gate) * mask

        x_gated = x * gate.unsqueeze(-1)
        return x_gated, gate


class DepthwiseTemporalEncoder(nn.Module):
    """
    Component-wise temporal encoder.
    - Applies depthwise Conv1d over time (one temporal kernel per component).
    - Followed by BN + ELU, then AvgPool for temporal compression.
    Input:
        x: [B, D, T]   (D: #components)
    Output:
        tokens: [B, D, F]   (F: token dimension after temporal compression)
    """
    def __init__(self, n_components: int, k_temporal: int, pool_kernel: int = 8, pool_stride: int = 8, dropout: float = 0.25):
        super().__init__()
        padding = k_temporal // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=n_components,
                out_channels=n_components,
                kernel_size=k_temporal,
                stride=2,
                padding=padding,
                groups=n_components,
                bias=False,
            ),
            nn.BatchNorm1d(n_components),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LinearSpatialProjector(nn.Module):
    """
    Linear spatial projection from electrodes to components.
    - Learns a projection matrix W ∈ R^{D×C}.
    Input:
        x: [B, C, T]
    Output:
        y: [B, D, T]
    """
    def __init__(self, n_electrodes: int, n_components: int):
        super().__init__()
        self.n_electrodes = int(n_electrodes)
        self.n_components = int(n_components)

        self.weight = nn.Parameter(torch.empty(self.n_components, self.n_electrodes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == self.n_electrodes
        return torch.einsum("bct,dc->bdt", x, self.weight)


class GlobalComponentMixer(nn.Module):
    """
    Global component mixer (set-like aggregation).
    - tokens: [B, D, F] -> transpose -> [B, F, D]
    - depthwise Conv1d with kernel=D aggregates across all components
    - pointwise Conv1d keeps feature dimension F (your 4k-param design)
    Output:
        h: [B, F]
    """
    def __init__(self, n_components: int, token_dim: int, dropout: float = 0.25):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv1d(
                in_channels=token_dim,
                out_channels=token_dim,
                kernel_size=n_components,
                groups=token_dim,
                bias=False,
            ),
            nn.BatchNorm1d(token_dim),
            nn.ELU(inplace=True),
            nn.Conv1d(token_dim, token_dim, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.transpose(1, 2)   # [B,F,D]
        x = self.mixer(x)            # [B,F,1]
        return x.squeeze(-1)         # [B,F]


class Model(nn.Module):
    """
    SPaCE-style RSVP encoder (backbone only).
    Pipeline:
        (1) mask-aware channel gating
        (2) linear spatial projection: electrodes -> components
        (3) component-wise temporal encoding (depthwise)
        (4) global component mixing -> trial-level representation
        (5) max-norm linear classifier
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_electrodes = int(config["n_channels"])
        self.n_classes = int(config["n_class"])

        fs = int(config["fs"])
        k_temporal = int(config.get("k_temporal", fs // 2))

        # D: number of components
        self.n_components = int(config.get("n_components", self.n_electrodes // 2))

        # F: token dimension after temporal compression (keep your original rule)
        avp_dim = int(config.get("avp_dim", 8))
        token_dim = int(config.get("token_dim", fs // (2 * avp_dim)))

        self.z_gate = ZeroChannelGate(
            hidden_dim=int(config.get("zgate_hidden", 8)),
            alpha=float(config.get("zgate_alpha", 0.1)),
            renorm=bool(config.get("zgate_renorm", True)),
        )

        self.spatial_projector = LinearSpatialProjector(self.n_electrodes, self.n_components)
        self.temporal_encoder = DepthwiseTemporalEncoder(self.n_components, k_temporal, pool_kernel=8, pool_stride=8, dropout=0.25)
        self.component_mixer = GlobalComponentMixer(self.n_components, token_dim, dropout=0.25)

        self.classifier = MaxNormLinear(
            in_features=token_dim,
            out_features=self.n_classes,
            max_norm=float(config.get("max_norm", 0.1)),
        )

        self.last_gate = None

    def forward(self, x: torch.Tensor, train_stage: int = 2, mask: torch.Tensor = None, **kwargs):
        # x: [B,1,C,T]
        B, _, C, T = x.shape
        assert C == self.n_electrodes

        x = x.squeeze(1)  # [B,C,T]

        if mask is None:
            # 自动识别零通道：整条时间序列全 0 -> mask=0
            eps = 1e-6 if x.dtype in (torch.float32, torch.float64) else 1e-4
            mask = (x.abs().amax(dim=-1) > eps).to(x.dtype)  # [B,C]

        x_gated, gate = self.z_gate(x, mask=mask)  # [B,C,T], [B,C]
        self.last_gate = gate

        x_comp = self.spatial_projector(x_gated)  # [B,D,T]
        tokens = self.temporal_encoder(x_comp)  # [B,D,F]
        h = self.component_mixer(tokens)  # [B,F]

        feat = F.normalize(h, p=2, dim=-1)
        logits = self.classifier(h)
        return logits, feat


if __name__ == '__main__':
    args = {"n_channels": 62, "fs": 128, "n_class": 2, "seq_len": 128}
    model = Model(args)
    input_shape = (1, 1, args["n_channels"], args["seq_len"])  # 输入形状 (batch_size, n_channels, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary(model, input_shape)
    # x = torch.randn(1, 1, 62, 128).to(device)
    # logits, feat = model(x)
    # print(logits.shape, feat.shape)
