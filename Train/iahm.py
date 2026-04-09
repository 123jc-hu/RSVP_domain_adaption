import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _k_abs(curvature: float) -> float:
    k = float(curvature)
    if k >= 0.0:
        raise ValueError(f"curvature must be negative, got {curvature}")
    return abs(k)


def _to_tensor_counts(class_counts: Sequence[int], device: Optional[torch.device] = None) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    if counts.ndim != 1:
        raise ValueError(f"class_counts must be 1D, got shape={tuple(counts.shape)}")
    return counts


def lorentz_inner(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product: -z0*w0 + sum_{i>=1} zi*wi."""
    return -z[..., 0] * w[..., 0] + torch.sum(z[..., 1:] * w[..., 1:], dim=-1)


def project_to_lorentz(z: torch.Tensor, K: float = -1.0, eps: float = 1e-7) -> torch.Tensor:
    """
    Project ambient vectors to Lorentz manifold by fixing time coordinate:
    z0 = sqrt(1/|K| + ||z_spatial||^2).
    """
    kabs = _k_abs(K)
    spatial = z[..., 1:]
    sq = torch.sum(spatial * spatial, dim=-1, keepdim=True)
    time = torch.sqrt(torch.clamp((1.0 / kabs) + sq, min=eps))
    return torch.cat([time, spatial], dim=-1)


def project_to_tangent(p: torch.Tensor, v: torch.Tensor, K: float = -1.0) -> torch.Tensor:
    """
    Project ambient vector v to tangent space T_p:
    v_tan = v + |K| <p,v>_L p.
    """
    kabs = _k_abs(K)
    coeff = (kabs * lorentz_inner(p, v)).unsqueeze(-1)
    return v + coeff * p


def lorentz_norm(v: torch.Tensor, K: float = -1.0, eps: float = 1e-14) -> torch.Tensor:
    """
    Tangent vector norm under Lorentz metric.
    """
    _ = K
    inner = lorentz_inner(v, v)
    return torch.sqrt(torch.clamp(inner, min=eps))


def _acosh_stable(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # Keep exact acosh(1)=0 for identity-distance checks.
    x = torch.clamp(x, min=1.0)
    return torch.log(x + torch.sqrt(torch.clamp(x * x - 1.0, min=0.0)))


def lorentz_distance(z: torch.Tensor, w: torch.Tensor, K: float = -1.0, eps: float = 1e-7) -> torch.Tensor:
    """Geodesic distance on Lorentz manifold of curvature K<0."""
    kabs = _k_abs(K)
    inner = lorentz_inner(z, w)
    val = -kabs * inner
    return _acosh_stable(val, eps=eps) / math.sqrt(kabs)


def lorentz_origin(dim: int, K: float = -1.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Origin o=(1/sqrt(|K|),0,...,0) in ambient dimension dim+1."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    kabs = _k_abs(K)
    o = torch.zeros(dim + 1, dtype=torch.float32, device=device)
    o[0] = 1.0 / math.sqrt(kabs)
    return o


def lorentz_radial(z: torch.Tensor, K: float = -1.0, eps: float = 1e-7) -> torch.Tensor:
    """Radial distance from origin in Lorentz model."""
    kabs = _k_abs(K)
    val = math.sqrt(kabs) * z[..., 0]
    return _acosh_stable(val, eps=eps) / math.sqrt(kabs)


def lorentz_exp_map(p: torch.Tensor, v: torch.Tensor, K: float = -1.0, eps: float = 1e-7) -> torch.Tensor:
    """
    Exponential map at p on Lorentz manifold.
    """
    kabs = _k_abs(K)
    v_tan = project_to_tangent(p, v, K=K)
    v_norm = torch.clamp(lorentz_norm(v_tan, K=K, eps=eps), min=eps)
    scale = math.sqrt(kabs) * v_norm
    coeff_p = torch.cosh(scale)
    coeff_v = torch.sinh(scale) / (math.sqrt(kabs) * v_norm)
    out = coeff_p.unsqueeze(-1) * p + coeff_v.unsqueeze(-1) * v_tan
    return project_to_lorentz(out, K=K, eps=eps)


def lorentz_log_map(p: torch.Tensor, q: torch.Tensor, K: float = -1.0, eps: float = 1e-7) -> torch.Tensor:
    """
    Logarithmic map at p of q on Lorentz manifold.
    """
    kabs = _k_abs(K)
    inner_pq = lorentz_inner(p, q)
    alpha = torch.clamp(-kabs * inner_pq, min=1.0 + eps)
    dist = _acosh_stable(alpha, eps=eps) / math.sqrt(kabs)
    direction = q + (kabs * inner_pq).unsqueeze(-1) * p
    direction = project_to_tangent(p, direction, K=K)
    dir_norm = torch.clamp(lorentz_norm(direction, K=K, eps=eps), min=eps)
    v = (dist / dir_norm).unsqueeze(-1) * direction
    return project_to_tangent(p, v, K=K)


def compute_target_radii(
    class_counts: Sequence[int],
    r0: float = 1.0,
    gamma: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    counts = _to_tensor_counts(class_counts, device=device)
    n_max = torch.max(counts)
    return float(r0) + float(gamma) * torch.log(n_max / (counts + 1.0))


def compute_margins(
    class_counts: Sequence[int],
    m0: float = 1.0,
    alpha: float = 0.25,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    counts = _to_tensor_counts(class_counts, device=device)
    n_max = torch.max(counts)
    return float(m0) * torch.pow(n_max / (counts + 1.0), float(alpha))


class HyperbolicClassCentroids(nn.Module):
    """
    Lorentz class-centroid tracker with tangent-space EMA updates.
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int,
        curvature: float = -1.0,
        momentum: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.embed_dim = int(embed_dim)
        self.K = float(curvature)
        self.momentum = float(momentum)
        self.eps = float(eps)

        centers = torch.zeros(self.n_classes, self.embed_dim + 1, dtype=torch.float32)
        centers[:, 0] = 1.0 / math.sqrt(_k_abs(self.K))
        self.register_buffer("centers", centers)
        self.register_buffer("initialized", torch.zeros(self.n_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        if z.ndim != 2 or z.shape[1] != (self.embed_dim + 1):
            raise ValueError(
                f"Expected z shape [B,{self.embed_dim + 1}], got {tuple(z.shape)}"
            )
        y = y.long()
        for c in range(self.n_classes):
            mask = (y == c)
            if not torch.any(mask):
                continue
            z_c = z[mask]
            base = self.centers[c]
            if not bool(self.initialized[c]):
                # One-step Fréchet initialization from origin.
                v_mean = self._mean_tangent_vec(base, z_c)
                self.centers[c] = lorentz_exp_map(base, v_mean, K=self.K, eps=self.eps)
                self.initialized[c] = torch.tensor(True, device=self.initialized.device)
            else:
                v_mean = self._mean_tangent_vec(base, z_c)
                self.centers[c] = lorentz_exp_map(
                    base,
                    self.momentum * v_mean,
                    K=self.K,
                    eps=self.eps,
                )

    def get_centers(self) -> torch.Tensor:
        return self.centers

    def _mean_tangent_vec(self, base: torch.Tensor, z_batch: torch.Tensor) -> torch.Tensor:
        base_expand = base.unsqueeze(0).expand_as(z_batch)
        v = lorentz_log_map(base_expand, z_batch, K=self.K, eps=self.eps)
        return torch.mean(v, dim=0)


class RadialSeparationLoss(nn.Module):
    def __init__(self, curvature: float = -1.0, eps: float = 1e-7):
        super().__init__()
        self.K = float(curvature)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor, target_radii: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = lorentz_radial(z, K=self.K, eps=self.eps)
        target = target_radii[y.long()]
        return F.smooth_l1_loss(r, target)


class CompactnessLoss(nn.Module):
    def __init__(self, curvature: float = -1.0, eps: float = 1e-7):
        super().__init__()
        self.K = float(curvature)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor, y: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        same_center = centers[y.long()]
        dist = lorentz_distance(z, same_center, K=self.K, eps=self.eps)
        return torch.mean(dist * dist)


class MarginLoss(nn.Module):
    def __init__(self, n_classes: int, curvature: float = -1.0, eps: float = 1e-7):
        super().__init__()
        self.n_classes = int(n_classes)
        self.K = float(curvature)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor, y: torch.Tensor, centers: torch.Tensor, margins: torch.Tensor) -> torch.Tensor:
        y = y.long()
        required = margins[y]

        if self.n_classes == 2:
            opp = 1 - y
            opp_center = centers[opp]
            dist_opp = lorentz_distance(z, opp_center, K=self.K, eps=self.eps)
        else:
            # Multi-class fallback: nearest non-own center.
            b = z.shape[0]
            dist_all = []
            for c in range(self.n_classes):
                c_center = centers[c].unsqueeze(0).expand(b, -1)
                dist_all.append(lorentz_distance(z, c_center, K=self.K, eps=self.eps))
            dist_stack = torch.stack(dist_all, dim=1)  # [B, C]
            own_mask = F.one_hot(y, num_classes=self.n_classes).bool()
            dist_stack = dist_stack.masked_fill(own_mask, float("inf"))
            dist_opp, _ = torch.min(dist_stack, dim=1)

        return torch.mean(F.relu(required - dist_opp))


class IAHMLoss(nn.Module):
    """
    Imbalance-Aware Hyperbolic Margin loss:
      L = lambda_r * L_radial + lambda_c * L_compact + lambda_m * L_margin.
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int,
        curvature: float = -1.0,
        class_counts: Optional[Sequence[int]] = None,
        r0: float = 1.0,
        gamma: float = 1.0,
        m0: float = 1.0,
        margin_alpha: float = 0.25,
        lambda_r: float = 1.0,
        lambda_c: float = 0.5,
        lambda_m: float = 1.0,
        centroid_momentum: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.embed_dim = int(embed_dim)
        self.K = float(curvature)
        self.eps = float(eps)

        if class_counts is None:
            class_counts = [1 for _ in range(self.n_classes)]
        if len(class_counts) != self.n_classes:
            raise ValueError(
                f"class_counts length ({len(class_counts)}) != n_classes ({self.n_classes})"
            )

        target_r = compute_target_radii(class_counts, r0=r0, gamma=gamma)
        margins = compute_margins(class_counts, m0=m0, alpha=margin_alpha)
        self.register_buffer("target_radii", target_r)
        self.register_buffer("margins", margins)

        self.centroids = HyperbolicClassCentroids(
            n_classes=self.n_classes,
            embed_dim=self.embed_dim,
            curvature=self.K,
            momentum=float(centroid_momentum),
            eps=self.eps,
        )
        self.radial_loss = RadialSeparationLoss(curvature=self.K, eps=self.eps)
        self.compactness_loss = CompactnessLoss(curvature=self.K, eps=self.eps)
        self.margin_loss = MarginLoss(n_classes=self.n_classes, curvature=self.K, eps=self.eps)

        self.lambda_r = float(lambda_r)
        self.lambda_c = float(lambda_c)
        self.lambda_m = float(lambda_m)

    def forward(self, z: torch.Tensor, y: torch.Tensor, update_centroids: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        if z.ndim != 2 or z.shape[1] != (self.embed_dim + 1):
            raise ValueError(
                f"IAHM expects z shape [B,{self.embed_dim + 1}], got {tuple(z.shape)}"
            )
        y = y.long()
        if z.shape[0] != y.shape[0]:
            raise ValueError(f"Batch mismatch: z batch={z.shape[0]} vs y batch={y.shape[0]}")

        z = project_to_lorentz(z, K=self.K, eps=self.eps)
        if update_centroids:
            self.centroids.update(z.detach(), y)

        centers = self.centroids.get_centers()
        loss_radial = self.radial_loss(z, self.target_radii, y)
        loss_compact = self.compactness_loss(z, y, centers)
        loss_margin = self.margin_loss(z, y, centers, self.margins)

        loss = (
            self.lambda_r * loss_radial
            + self.lambda_c * loss_compact
            + self.lambda_m * loss_margin
        )
        details = {
            "radial": float(loss_radial.detach().item()),
            "compact": float(loss_compact.detach().item()),
            "margin": float(loss_margin.detach().item()),
            "target_radius_bg": float(self.target_radii[0].detach().item()) if self.n_classes >= 1 else float("nan"),
            "target_radius_p300": float(self.target_radii[1].detach().item()) if self.n_classes >= 2 else float("nan"),
            "margin_bg": float(self.margins[0].detach().item()) if self.n_classes >= 1 else float("nan"),
            "margin_p300": float(self.margins[1].detach().item()) if self.n_classes >= 2 else float("nan"),
        }
        return loss, details


def euclidean_to_lorentz(x: torch.Tensor, K: float = -1.0, eps: float = 1e-7) -> torch.Tensor:
    """
    Lift Euclidean features [B,d] to Lorentz points [B,d+1]:
      z0 = sqrt(1/|K| + ||x||^2), z_spatial = x.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [B,d], got {tuple(x.shape)}")
    kabs = _k_abs(K)
    sq = torch.sum(x * x, dim=-1, keepdim=True)
    z0 = torch.sqrt(torch.clamp((1.0 / kabs) + sq, min=eps))
    return torch.cat([z0, x], dim=-1)


class EuclideanClassCentroids(nn.Module):
    def __init__(self, n_classes: int, embed_dim: int, momentum: float = 0.1):
        super().__init__()
        self.n_classes = int(n_classes)
        self.embed_dim = int(embed_dim)
        self.momentum = float(momentum)
        self.register_buffer("centers", torch.zeros(self.n_classes, self.embed_dim, dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros(self.n_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        y = y.long()
        for c in range(self.n_classes):
            mask = (y == c)
            if not torch.any(mask):
                continue
            z_c = z[mask]
            mean_c = torch.mean(z_c, dim=0)
            if not bool(self.initialized[c]):
                self.centers[c] = mean_c
                self.initialized[c] = torch.tensor(True, device=self.initialized.device)
            else:
                self.centers[c] = (1.0 - self.momentum) * self.centers[c] + self.momentum * mean_c

    def get_centers(self) -> torch.Tensor:
        return self.centers


class EuclideanIAHMLoss(nn.Module):
    """
    Euclidean analogue of IAHM with the same radial / compact / margin structure.
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int,
        class_counts: Optional[Sequence[int]] = None,
        r0: float = 1.0,
        gamma: float = 1.0,
        m0: float = 1.0,
        margin_alpha: float = 0.25,
        lambda_r: float = 1.0,
        lambda_c: float = 0.5,
        lambda_m: float = 1.0,
        centroid_momentum: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.embed_dim = int(embed_dim)
        self.eps = float(eps)

        if class_counts is None:
            class_counts = [1 for _ in range(self.n_classes)]
        if len(class_counts) != self.n_classes:
            raise ValueError(
                f"class_counts length ({len(class_counts)}) != n_classes ({self.n_classes})"
            )

        target_r = compute_target_radii(class_counts, r0=r0, gamma=gamma)
        margins = compute_margins(class_counts, m0=m0, alpha=margin_alpha)
        self.register_buffer("target_radii", target_r)
        self.register_buffer("margins", margins)

        self.centroids = EuclideanClassCentroids(
            n_classes=self.n_classes,
            embed_dim=self.embed_dim,
            momentum=float(centroid_momentum),
        )
        self.lambda_r = float(lambda_r)
        self.lambda_c = float(lambda_c)
        self.lambda_m = float(lambda_m)

    def forward(self, z: torch.Tensor, y: torch.Tensor, update_centroids: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        if z.ndim != 2 or z.shape[1] != self.embed_dim:
            raise ValueError(
                f"Euclidean IAHM expects z shape [B,{self.embed_dim}], got {tuple(z.shape)}"
            )
        y = y.long()
        if z.shape[0] != y.shape[0]:
            raise ValueError(f"Batch mismatch: z batch={z.shape[0]} vs y batch={y.shape[0]}")

        if update_centroids:
            self.centroids.update(z.detach(), y)
        centers = self.centroids.get_centers()

        r = torch.linalg.norm(z, dim=-1)
        target = self.target_radii[y]
        loss_radial = F.smooth_l1_loss(r, target)

        same_center = centers[y]
        dist_same = torch.linalg.norm(z - same_center, dim=-1)
        loss_compact = torch.mean(dist_same * dist_same)

        if self.n_classes == 2:
            opp = 1 - y
            opp_center = centers[opp]
            dist_opp = torch.linalg.norm(z - opp_center, dim=-1)
        else:
            dist_all = []
            for c in range(self.n_classes):
                c_center = centers[c].unsqueeze(0).expand_as(z)
                dist_all.append(torch.linalg.norm(z - c_center, dim=-1))
            dist_stack = torch.stack(dist_all, dim=1)
            own_mask = F.one_hot(y, num_classes=self.n_classes).bool()
            dist_stack = dist_stack.masked_fill(own_mask, float("inf"))
            dist_opp, _ = torch.min(dist_stack, dim=1)

        required = self.margins[y]
        loss_margin = torch.mean(F.relu(required - dist_opp))

        loss = (
            self.lambda_r * loss_radial
            + self.lambda_c * loss_compact
            + self.lambda_m * loss_margin
        )
        details = {
            "radial": float(loss_radial.detach().item()),
            "compact": float(loss_compact.detach().item()),
            "margin": float(loss_margin.detach().item()),
            "target_radius_bg": float(self.target_radii[0].detach().item()) if self.n_classes >= 1 else float("nan"),
            "target_radius_p300": float(self.target_radii[1].detach().item()) if self.n_classes >= 2 else float("nan"),
            "margin_bg": float(self.margins[0].detach().item()) if self.n_classes >= 1 else float("nan"),
            "margin_p300": float(self.margins[1].detach().item()) if self.n_classes >= 2 else float("nan"),
        }
        return loss, details
