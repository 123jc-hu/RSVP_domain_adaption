import torch
from torch import nn

from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize


class SimilarityWeightedLatentStyleAdapter2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_source_domains: int,
        *,
        eps: float = 1.0e-5,
        momentum: float = 0.1,
        target_blend_alpha: float = 0.1,
        similarity_tau: float = 1.0,
        var_distance_weight: float = 1.0,
        init_gate: float = 0.1,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_source_domains = int(max(1, num_source_domains))
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.target_blend_alpha = float(min(max(target_blend_alpha, 0.0), 1.0))
        self.similarity_tau = float(max(similarity_tau, 1.0e-6))
        self.var_distance_weight = float(max(var_distance_weight, 0.0))

        init_gate = float(min(max(init_gate, 0.0), 1.0))
        init_logit = torch.logit(torch.tensor(init_gate).clamp(1.0e-4, 1.0 - 1.0e-4))
        self.gate_logit = nn.Parameter(torch.full((self.num_features,), float(init_logit)))

        self.register_buffer("source_running_mean", torch.zeros(self.num_source_domains, self.num_features))
        self.register_buffer("source_running_var", torch.ones(self.num_source_domains, self.num_features))
        self.register_buffer("source_initialized", torch.zeros(self.num_source_domains, dtype=torch.bool))

        self.current_domain_ids: torch.Tensor | None = None
        self.apply_target_adapter: bool = False

    def set_context(self, domain_ids: torch.Tensor | None = None, *, apply_target_adapter: bool = False) -> None:
        self.current_domain_ids = domain_ids
        self.apply_target_adapter = bool(apply_target_adapter)

    def _batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        return mean, var

    def _domain_index(self, domain_id: int) -> int | None:
        idx = int(domain_id) - 1
        if idx < 0 or idx >= self.num_source_domains:
            return None
        return idx

    def _update_source_stats(self, x: torch.Tensor, domain_ids: torch.Tensor | None) -> None:
        if not isinstance(domain_ids, torch.Tensor) or domain_ids.numel() != x.shape[0]:
            mean, var = self._batch_stats(x.float())
            with torch.no_grad():
                if bool(self.source_initialized[0].item()):
                    self.source_running_mean[0].mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                    self.source_running_var[0].mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
                else:
                    self.source_running_mean[0].copy_(mean.detach())
                    self.source_running_var[0].copy_(var.detach())
                    self.source_initialized[0] = True
            return

        with torch.no_grad():
            for domain_id in torch.unique(domain_ids.detach()).tolist():
                idx = self._domain_index(int(domain_id))
                if idx is None:
                    continue
                mask = domain_ids == int(domain_id)
                if int(mask.sum().item()) <= 0:
                    continue
                mean, var = self._batch_stats(x[mask].float())
                if bool(self.source_initialized[idx].item()):
                    self.source_running_mean[idx].mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                    self.source_running_var[idx].mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
                else:
                    self.source_running_mean[idx].copy_(mean.detach())
                    self.source_running_var[idx].copy_(var.detach())
                    self.source_initialized[idx] = True

    def _weighted_source_prior(
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
        weights = torch.softmax(-dist / self.similarity_tau, dim=0)
        prior_mean = (weights.unsqueeze(1) * src_means).sum(dim=0)
        prior_var = (weights.unsqueeze(1) * src_vars).sum(dim=0)
        return prior_mean, prior_var

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = x
        if not self.apply_target_adapter:
            self._update_source_stats(x, self.current_domain_ids)
            return pre, pre

        target_mean, target_var = self._batch_stats(x.float())
        prior = self._weighted_source_prior(target_mean.detach(), target_var.detach())
        if prior is None:
            return pre, pre

        source_mean, source_var = prior
        alpha = self.target_blend_alpha
        style_mean = alpha * target_mean + (1.0 - alpha) * source_mean
        style_var = alpha * target_var + (1.0 - alpha) * source_var

        target_mean = target_mean.to(x.dtype).view(1, -1, 1, 1)
        target_std = torch.sqrt(target_var.to(x.dtype).view(1, -1, 1, 1) + self.eps)
        style_mean = style_mean.to(x.dtype).view(1, -1, 1, 1)
        style_std = torch.sqrt(style_var.to(x.dtype).view(1, -1, 1, 1) + self.eps)

        adapted = (x - target_mean) / target_std * style_std + style_mean
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
        self.num_source_domains = int(config.get("sub_num", 64))
        self.style_momentum = float(config.get("lsa_style_momentum", 0.1))
        self.style_init_gate = float(config.get("lsa_init_gate", 0.1))
        self.target_blend_alpha = float(config.get("lsa_target_blend_alpha", 0.1))
        self.similarity_tau = float(config.get("lsa_similarity_tau", 1.0))
        self.var_distance_weight = float(config.get("lsa_var_distance_weight", 1.0))
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
        self.style_adapter = SimilarityWeightedLatentStyleAdapter2d(
            self.F2,
            num_source_domains=self.num_source_domains,
            eps=1.0e-5,
            momentum=self.style_momentum,
            target_blend_alpha=self.target_blend_alpha,
            similarity_tau=self.similarity_tau,
            var_distance_weight=self.var_distance_weight,
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
        self.style_adapter.set_context(domain_ids=domain_ids, apply_target_adapter=use_target_stats)

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

    summarize_model("EEGNetLSAv2", Model)
