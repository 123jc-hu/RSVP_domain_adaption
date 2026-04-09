import torch
from torch import nn

from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize


class GoldenSubjectDualStatisticsBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_source_domains: int,
        eps: float = 1.0e-5,
        momentum: float = 0.1,
        affine: bool = True,
        target_blend_alpha: float = 0.1,
        var_distance_weight: float = 1.0,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_source_domains = int(max(1, num_source_domains))
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.target_blend_alpha = float(min(max(target_blend_alpha, 0.0), 1.0))
        self.var_distance_weight = float(max(var_distance_weight, 0.0))

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("source_running_mean", torch.zeros(self.num_source_domains, self.num_features))
        self.register_buffer("source_running_var", torch.ones(self.num_source_domains, self.num_features))
        self.register_buffer("source_initialized", torch.zeros(self.num_source_domains, dtype=torch.bool))

        self.register_buffer("target_running_mean", torch.zeros(self.num_features))
        self.register_buffer("target_running_var", torch.ones(self.num_features))
        self.register_buffer("target_initialized", torch.tensor(False, dtype=torch.bool))

        self.current_domain_ids: torch.Tensor | None = None
        self.use_target_stats: bool = False

    def set_domain_context(self, domain_ids: torch.Tensor | None = None, *, use_target_stats: bool = False) -> None:
        self.current_domain_ids = domain_ids
        self.use_target_stats = bool(use_target_stats)

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            return x
        weight = self.weight.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        bias = self.bias.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return x * weight + bias

    def _normalize_with_stats(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        y = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return self._apply_affine(y)

    def _batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        return mean, var

    def _update_target_stats(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        with torch.no_grad():
            if bool(self.target_initialized.item()):
                self.target_running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                self.target_running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
            else:
                self.target_running_mean.copy_(mean.detach())
                self.target_running_var.copy_(var.detach())
                self.target_initialized.fill_(True)

    def _domain_index(self, domain_id: int) -> int | None:
        idx = int(domain_id) - 1
        if idx < 0 or idx >= self.num_source_domains:
            return None
        return idx

    def _aggregate_source_stats(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        mask = self.source_initialized
        if int(mask.sum().item()) <= 0:
            return None
        return self.source_running_mean[mask].mean(dim=0), self.source_running_var[mask].mean(dim=0)

    def _golden_source_prior(
        self,
        target_mean: torch.Tensor,
        target_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        mask = self.source_initialized
        if int(mask.sum().item()) <= 0:
            return None

        src_means = self.source_running_mean[mask]
        src_vars = self.source_running_var[mask]
        mean_dist = ((src_means - target_mean.unsqueeze(0)) ** 2).mean(dim=1)
        target_log_var = torch.log(target_var.unsqueeze(0) + self.eps)
        src_log_var = torch.log(src_vars + self.eps)
        var_dist = ((src_log_var - target_log_var) ** 2).mean(dim=1)
        dist = mean_dist + self.var_distance_weight * var_dist
        best_idx = int(torch.argmin(dist).item())
        return src_means[best_idx], src_vars[best_idx]

    def _blend_stats(
        self,
        target_mean: torch.Tensor,
        target_var: torch.Tensor,
        source_mean: torch.Tensor | None,
        source_var: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_mean is None or source_var is None:
            return target_mean, target_var
        alpha = self.target_blend_alpha
        mean = alpha * target_mean + (1.0 - alpha) * source_mean
        var = alpha * target_var + (1.0 - alpha) * source_var
        return mean, var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        domain_ids = self.current_domain_ids
        use_target_stats = bool(self.use_target_stats)

        if self.training:
            if use_target_stats:
                target_mean, target_var = self._batch_stats(x.float())
                self._update_target_stats(target_mean, target_var)
                source_prior = self._golden_source_prior(target_mean.detach(), target_var.detach())
                source_mean = source_prior[0] if source_prior is not None else None
                source_var = source_prior[1] if source_prior is not None else None
                mean, var = self._blend_stats(target_mean, target_var, source_mean, source_var)
                return self._normalize_with_stats(x, mean.to(x.dtype), var.to(x.dtype))

            if isinstance(domain_ids, torch.Tensor) and domain_ids.numel() == x.shape[0]:
                out = torch.empty_like(x)
                covered = torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device)
                for domain_id in torch.unique(domain_ids.detach()).tolist():
                    idx = self._domain_index(int(domain_id))
                    mask = domain_ids == int(domain_id)
                    if idx is None or int(mask.sum().item()) <= 0:
                        continue
                    x_g = x[mask]
                    mean, var = self._batch_stats(x_g.float())
                    with torch.no_grad():
                        if bool(self.source_initialized[idx].item()):
                            self.source_running_mean[idx].mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                            self.source_running_var[idx].mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
                        else:
                            self.source_running_mean[idx].copy_(mean.detach())
                            self.source_running_var[idx].copy_(var.detach())
                            self.source_initialized[idx] = True
                    out[mask] = self._normalize_with_stats(x_g, mean.to(x.dtype), var.to(x.dtype))
                    covered[mask] = True
                if bool(covered.all().item()):
                    return out

            mean, var = self._batch_stats(x.float())
            return self._normalize_with_stats(x, mean.to(x.dtype), var.to(x.dtype))

        if use_target_stats and bool(self.target_initialized.item()):
            source_prior = self._golden_source_prior(self.target_running_mean, self.target_running_var)
            source_mean = source_prior[0] if source_prior is not None else None
            source_var = source_prior[1] if source_prior is not None else None
            mean, var = self._blend_stats(
                self.target_running_mean,
                self.target_running_var,
                source_mean,
                source_var,
            )
            return self._normalize_with_stats(x, mean.to(x.dtype), var.to(x.dtype))

        if isinstance(domain_ids, torch.Tensor) and domain_ids.numel() == x.shape[0]:
            out = torch.empty_like(x)
            covered = torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device)
            for domain_id in torch.unique(domain_ids.detach()).tolist():
                idx = self._domain_index(int(domain_id))
                mask = domain_ids == int(domain_id)
                if idx is None or int(mask.sum().item()) <= 0 or (not bool(self.source_initialized[idx].item())):
                    continue
                out[mask] = self._normalize_with_stats(
                    x[mask],
                    self.source_running_mean[idx].to(x.dtype),
                    self.source_running_var[idx].to(x.dtype),
                )
                covered[mask] = True
            if bool(covered.all().item()):
                return out

        src_stats = self._aggregate_source_stats()
        if src_stats is not None:
            mean, var = src_stats
            return self._normalize_with_stats(x, mean.to(x.dtype), var.to(x.dtype))

        return self._apply_affine(x)


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16

        self.n_channels = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.n_classes = int(config["n_class"])
        self.num_source_domains = int(config.get("sub_num", 64))
        self.target_blend_alpha = float(config.get("gsldsa_target_blend_alpha", 0.1))
        self.var_distance_weight = float(config.get("gsldsa_var_distance_weight", 1.0))

        self.kernel_length = self.fs // 2
        self.kernel_length2 = self.fs // 8
        self.dropout_rate = 0.5

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

        self.block2_conv = nn.Sequential(
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
        )
        self.block2_gsldsa = GoldenSubjectDualStatisticsBatchNorm2d(
            self.F2,
            num_source_domains=self.num_source_domains,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            target_blend_alpha=self.target_blend_alpha,
            var_distance_weight=self.var_distance_weight,
        )
        self.block2_post = nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate),
        )

        flat_out_size = calculate_outsize(
            nn.Sequential(self.block1, self.block2_conv, nn.BatchNorm2d(self.F2), self.block2_post),
            self.n_channels,
            self.fs,
        )
        self.ClassifierBlock = DenseWithConstraint(flat_out_size, self.n_classes, bias=False, max_norm=0.25)

    def set_domain_context(self, domain_ids: torch.Tensor | None = None, *, use_target_stats: bool = False) -> None:
        self.block2_gsldsa.set_domain_context(domain_ids, use_target_stats=use_target_stats)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2_conv(x)
        x = self.block2_gsldsa(x)
        x = self.block2_post(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self._forward_features(x)
        logits = self.ClassifierBlock(features)
        if return_features:
            return logits, {"features": features}
        return logits


if __name__ == "__main__":
    from torchinfo import summarize

    config = {"fs": 250, "n_channels": 62, "n_class": 2, "sub_num": 64}
    Model = Model(config)
    summarize(Model, (1, 62, 250))
