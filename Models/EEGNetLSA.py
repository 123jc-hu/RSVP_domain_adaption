import torch
from torch import nn

from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize


class LatentStyleAdapter2d(nn.Module):
    def __init__(self, num_features: int, *, eps: float = 1.0e-5, momentum: float = 0.1, init_gate: float = 0.5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)

        init_gate = float(min(max(init_gate, 0.0), 1.0))
        init_logit = torch.logit(torch.tensor(init_gate).clamp(1.0e-4, 1.0 - 1.0e-4))
        self.gate_logit = nn.Parameter(torch.full((self.num_features,), float(init_logit)))

        self.register_buffer("source_style_mean", torch.zeros(self.num_features))
        self.register_buffer("source_style_var", torch.ones(self.num_features))
        self.register_buffer("source_style_initialized", torch.tensor(False, dtype=torch.bool))
        self.apply_target_adapter: bool = False

    def set_context(self, *, apply_target_adapter: bool = False) -> None:
        self.apply_target_adapter = bool(apply_target_adapter)

    def _batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        return mean, var

    def _update_source_style(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        with torch.no_grad():
            if bool(self.source_style_initialized.item()):
                self.source_style_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                self.source_style_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
            else:
                self.source_style_mean.copy_(mean.detach())
                self.source_style_var.copy_(var.detach())
                self.source_style_initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = x
        mean, var = self._batch_stats(x.float())

        if not self.apply_target_adapter:
            self._update_source_style(mean, var)
            return pre, pre

        if not bool(self.source_style_initialized.item()):
            return pre, pre

        target_mean = mean.to(x.dtype).view(1, -1, 1, 1)
        target_std = torch.sqrt(var.to(x.dtype).view(1, -1, 1, 1) + self.eps)
        source_mean = self.source_style_mean.to(x.dtype).view(1, -1, 1, 1)
        source_std = torch.sqrt(self.source_style_var.to(x.dtype).view(1, -1, 1, 1) + self.eps)
        adapted = (x - target_mean) / target_std * source_std + source_mean
        gate = torch.sigmoid(self.gate_logit).to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        post = x + gate * (adapted - x)
        return pre, post


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16

        self.n_channels = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.n_classes = int(config["n_class"])
        self.style_momentum = float(config.get("lsa_style_momentum", 0.1))
        self.style_init_gate = float(config.get("lsa_init_gate", 0.5))
        self.dropout_rate = 0.5
        self.kernel_length = self.fs // 2
        self.kernel_length2 = self.fs // 8

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_length), stride=1, bias=False, padding=(0, self.kernel_length // 2)),
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
            nn.ELU(),
        )
        self.style_adapter = LatentStyleAdapter2d(
            self.F2,
            eps=1.0e-5,
            momentum=self.style_momentum,
            init_gate=self.style_init_gate,
        )
        self.block2_post = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate),
        )

        flat_out_size = calculate_outsize(
            nn.Sequential(self.block1, self.block2_pre, self.block2_post),
            self.n_channels,
            self.fs,
        )
        self.ClassifierBlock = DenseWithConstraint(flat_out_size, self.n_classes, bias=False, max_norm=0.25)

    def set_domain_context(self, domain_ids: torch.Tensor | None = None, *, use_target_stats: bool = False) -> None:
        _ = domain_ids
        self.style_adapter.set_context(apply_target_adapter=use_target_stats)

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.block1(x)
        pre_map = self.block2_pre(x)
        content_pre, styled_map = self.style_adapter(pre_map)
        post_map = self.block2_post(styled_map)
        features = torch.flatten(post_map, start_dim=1)
        return features, content_pre, styled_map

    def forward(self, x, train_stage: int = 2, return_features: bool = False):
        _ = train_stage
        features, content_pre, content_post = self._forward_features(x)
        logits = self.ClassifierBlock(features)
        if return_features:
            return logits, {
                "features": features,
                "lsa_content_pre": torch.flatten(content_pre, start_dim=1),
                "lsa_content_post": torch.flatten(content_post, start_dim=1),
            }
        return logits


if __name__ == "__main__":
    from Models.summary_utils import summarize_model

    summarize_model("EEGNetLSA", Model)
