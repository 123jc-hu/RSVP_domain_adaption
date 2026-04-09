from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import torch

from pyriemann.utils.covariance import covariances
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_covariance

from Data.npz_io import ensure_npz_cache, ensure_subject_ea_cache
from numpy.lib.format import open_memmap


def _load_subject_xy(npz_path: Path, *, use_ea: bool, ea_cov_eps: float) -> tuple[np.memmap, np.memmap]:
    npz_path = Path(npz_path)
    if use_ea:
        cache_dir = ensure_subject_ea_cache(npz_path, x_key="x_data", cov_eps=float(ea_cov_eps))
        x = np.load(cache_dir / "x_data_ea.npy", mmap_mode="r")
    else:
        cache_dir = ensure_npz_cache(npz_path, ["x_data.npy", "y_data.npy"])
        x = np.load(cache_dir / "x_data.npy", mmap_mode="r")
    y = np.load(cache_dir / "y_data.npy", mmap_mode="r")
    return x, y


def _optimize_riemann(
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    n_filters: int,
    solver: str = "trust-constr",
    maxiter: int = 1000,
    tolerance: float = 1e-8,
    seed: int = 2026,
) -> np.ndarray:
    n_channels = int(p1.shape[0])
    if p1.shape != p2.shape:
        raise ValueError(f"RSF expects same-shape SPD matrices, got {p1.shape} vs {p2.shape}")
    if n_channels < int(n_filters):
        raise ValueError(f"RSF n_filters={n_filters} exceeds n_channels={n_channels}")

    rng = np.random.default_rng(int(seed))
    w0 = rng.standard_normal((n_channels, int(n_filters)))
    w0, _ = np.linalg.qr(w0)
    w0_flat = w0.ravel()

    def obj_func(w_flat: np.ndarray) -> float:
        w = w_flat.reshape(n_channels, -1)
        evals = eigh(w.T @ p1 @ w, w.T @ p2 @ w, eigvals_only=True)
        evals = np.clip(np.real(evals), a_min=1e-10, a_max=None)
        return -float(np.sum(np.log(evals) ** 2))

    options = {"maxiter": int(maxiter)}
    if str(solver) == "trust-constr":
        options["gtol"] = float(tolerance)
        options["verbose"] = 0

    result = minimize(
        obj_func,
        w0_flat,
        method=str(solver),
        options=options,
    )
    w_opt = result.x.reshape(n_channels, -1)
    d0 = -obj_func(w0_flat)
    d1 = -obj_func(result.x)
    return w0 if d0 > d1 else w_opt


def fit_plain_rsf(
    trials: np.ndarray,
    labels: np.ndarray,
    *,
    dim: int,
    cov_estimator: str = "lwf",
    solver: str = "trust-constr",
    maxiter: int = 1000,
    tolerance: float = 1e-8,
    seed: int = 2026,
) -> tuple[np.ndarray, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    unique_labels = np.unique(labels)
    if unique_labels.size != 2:
        raise ValueError(f"RSF currently supports binary labels, got {unique_labels.tolist()}")

    trial_cov = covariances(np.asarray(trials, dtype=np.float64), estimator=str(cov_estimator))
    cov_0 = mean_covariance(trial_cov[labels == unique_labels[0]], metric="riemann")
    cov_1 = mean_covariance(trial_cov[labels == unique_labels[1]], metric="riemann")
    w = _optimize_riemann(
        cov_0,
        cov_1,
        n_filters=int(dim),
        solver=str(solver),
        maxiter=int(maxiter),
        tolerance=float(tolerance),
        seed=int(seed),
    )
    score = float(distance(w.T @ cov_0 @ w, w.T @ cov_1 @ w, metric="riemann"))
    return np.asarray(w, dtype=np.float32), score


def _projected_riemann_distance_sq(w: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    return float(distance(w.T @ p1 @ w, w.T @ p2 @ w, metric="riemann", squared=True))


def fit_cc_domain_aware_rsf(
    *,
    global_class_covs: Dict[int, np.ndarray],
    subject_class_covs: Dict[str, Dict[int, np.ndarray]],
    dim: int,
    domain_lambda: float,
    solver: str = "trust-constr",
    maxiter: int = 1000,
    tolerance: float = 1e-8,
    seed: int = 2026,
) -> Tuple[np.ndarray, Dict[str, float]]:
    class_labels = sorted(int(k) for k in global_class_covs.keys())
    if len(class_labels) != 2:
        raise ValueError(f"CC-DA-RSF currently supports 2 classes, got {class_labels}")

    p0 = np.asarray(global_class_covs[class_labels[0]], dtype=np.float64)
    p1 = np.asarray(global_class_covs[class_labels[1]], dtype=np.float64)
    n_channels = int(p0.shape[0])
    n_filters = int(dim)
    if n_channels < n_filters:
        raise ValueError(f"CC-DA-RSF n_filters={n_filters} exceeds n_channels={n_channels}")

    penalty_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, class_map in subject_class_covs.items():
        for cls in class_labels:
            if cls not in class_map:
                continue
            penalty_pairs.append(
                (
                    np.asarray(class_map[cls], dtype=np.float64),
                    np.asarray(global_class_covs[cls], dtype=np.float64),
                )
            )

    # Initialize from the plain RSF solution, then refine with a lightweight
    # gradient-based objective under an orthonormal-column constraint.
    w0 = _optimize_riemann(
        p0,
        p1,
        n_filters=n_filters,
        solver="BFGS",
        maxiter=min(int(maxiter), 200),
        tolerance=float(tolerance),
        seed=int(seed),
    )

    device = torch.device("cpu")
    dtype = torch.float64
    eps = 1e-6
    p0_t = torch.as_tensor(p0, dtype=dtype, device=device)
    p1_t = torch.as_tensor(p1, dtype=dtype, device=device)
    penalty_pairs_t = [
        (
            torch.as_tensor(pi, dtype=dtype, device=device),
            torch.as_tensor(pg, dtype=dtype, device=device),
        )
        for pi, pg in penalty_pairs
    ]

    a = torch.nn.Parameter(torch.as_tensor(w0, dtype=dtype, device=device))
    optimizer = torch.optim.Adam([a], lr=5e-2)
    best_obj = -np.inf
    best_w = np.asarray(w0, dtype=np.float64)
    patience = 20
    stale_steps = 0

    def _orthonormal_w(mat: torch.Tensor) -> torch.Tensor:
        q, _ = torch.linalg.qr(mat, mode="reduced")
        return q[:, :n_filters]

    def _torch_projected_riemann_distance_sq(w_t: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        a_t = w_t.transpose(0, 1) @ c1 @ w_t
        b_t = w_t.transpose(0, 1) @ c2 @ w_t
        eye = torch.eye(a_t.shape[0], dtype=dtype, device=device)
        a_t = 0.5 * (a_t + a_t.transpose(0, 1)) + eps * eye
        b_t = 0.5 * (b_t + b_t.transpose(0, 1)) + eps * eye
        chol = torch.linalg.cholesky(b_t)
        left = torch.linalg.solve_triangular(chol, a_t, upper=False)
        middle = torch.linalg.solve_triangular(chol, left.transpose(0, 1), upper=False).transpose(0, 1)
        middle = 0.5 * (middle + middle.transpose(0, 1))
        eigvals = torch.linalg.eigvalsh(middle).clamp_min(1e-10)
        return torch.sum(torch.log(eigvals) ** 2)

    for _ in range(max(1, int(maxiter))):
        optimizer.zero_grad(set_to_none=True)
        w_t = _orthonormal_w(a)
        cls_term_t = _torch_projected_riemann_distance_sq(w_t, p0_t, p1_t)
        if penalty_pairs_t:
            penalty_t = torch.stack(
                [_torch_projected_riemann_distance_sq(w_t, pi_t, pg_t) for pi_t, pg_t in penalty_pairs_t]
            ).mean()
        else:
            penalty_t = torch.zeros((), dtype=dtype, device=device)
        objective_t = cls_term_t - float(domain_lambda) * penalty_t
        loss_t = -objective_t
        loss_t.backward()
        optimizer.step()

        objective_val = float(objective_t.detach().cpu().item())
        if objective_val > best_obj + float(tolerance):
            best_obj = objective_val
            best_w = _orthonormal_w(a).detach().cpu().numpy()
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= patience:
                break

    w_best = np.asarray(best_w, dtype=np.float64)
    cls_term = _projected_riemann_distance_sq(w_best, p0, p1)
    penalty = (
        float(np.mean([_projected_riemann_distance_sq(w_best, pi, pg) for pi, pg in penalty_pairs]))
        if penalty_pairs
        else 0.0
    )
    return np.asarray(w_best, dtype=np.float32), {
        "cls_distance_sq": float(cls_term),
        "domain_penalty_sq": float(penalty),
        "objective": float(cls_term - float(domain_lambda) * penalty),
        "n_penalty_pairs": int(len(penalty_pairs)),
    }


def apply_spatial_filter(w: np.ndarray, trials: np.ndarray) -> np.ndarray:
    return np.asarray(np.einsum("ij,kjl->kil", w.T, trials), dtype=np.float32)


def _rsf_cache_root(config: Dict[str, Any]) -> Path:
    root_cfg = config.get("rsf_cache_root", None)
    if root_cfg not in (None, "", "None", "none", "null", "NULL"):
        return Path(str(root_cfg))
    return Path(__file__).resolve().parents[1] / "Cache" / "rsf_preproc"


def _fit_indices_for_subject(
    labels_all: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    max_trials_per_class: Optional[int],
    balance_classes: bool,
    seed: int,
    subject_key: str,
) -> np.ndarray:
    idx = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx

    labels = np.asarray(labels_all[idx], dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed) + int(subject_key[3:]))
    class_values = np.unique(labels)
    class_to_indices: Dict[int, np.ndarray] = {}
    for cls in class_values:
        class_to_indices[int(cls)] = idx[labels == cls]

    if bool(balance_classes):
        if len(class_to_indices) < 2:
            return idx
        per_class_limit = min(arr.shape[0] for arr in class_to_indices.values())
        if max_trials_per_class not in (None, 0):
            per_class_limit = min(per_class_limit, int(max_trials_per_class))
        if per_class_limit <= 0:
            return np.empty(0, dtype=np.int64)
    else:
        per_class_limit = None

    if max_trials_per_class in (None, 0) and not bool(balance_classes):
        return idx

    max_trials = None if max_trials_per_class in (None, 0) else int(max_trials_per_class)
    parts: List[np.ndarray] = []
    for cls in class_values:
        cls = int(cls)
        cls_idx = class_to_indices[cls]
        limit = per_class_limit if per_class_limit is not None else max_trials
        if limit is None or cls_idx.size <= int(limit):
            parts.append(cls_idx)
            continue
        chosen = cls_idx[rng.permutation(cls_idx.shape[0])[: int(limit)]]
        parts.append(np.asarray(chosen, dtype=np.int64))
    out = np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.int64)
    if out.size > 0:
        out = out[rng.permutation(out.shape[0])]
    return np.asarray(out, dtype=np.int64)


def _hash_source_keys(source_keys: List[str]) -> str:
    joined = ",".join(sorted(str(k) for k in source_keys))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    mm = open_memmap(tmp_path, mode="w+", dtype=array.dtype, shape=array.shape)
    mm[...] = array
    del mm
    os.replace(tmp_path, path)


def ensure_rsf_fold_cache(
    *,
    config: Dict[str, Any],
    subject_file_map: Dict[str, str],
    held_out_key: str,
    source_subject_keys: List[str],
    source_train_indices_map: Dict[str, np.ndarray],
    source_val_indices_map: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    dim = int(config.get("rsf_dim", config.get("n_channels", 8)))
    use_ea = bool(config.get("ea_enable", False))
    ea_cov_eps = float(config.get("ea_cov_eps", 1e-6))
    cov_estimator = str(config.get("rsf_cov_estimator", "lwf")).strip().lower()
    solver = str(config.get("rsf_solver", "trust-constr")).strip()
    maxiter = int(config.get("rsf_maxiter", 1000))
    tolerance = float(config.get("rsf_tolerance", 1e-8))
    rsf_mode = str(config.get("rsf_mode", "plain")).strip().lower()
    domain_lambda = float(config.get("rsf_domain_lambda", 0.0))
    fit_cap = config.get("rsf_fit_max_trials_per_class_per_subject", None)
    fit_cap = None if fit_cap in (None, "", "None", "none", "null", "NULL") else int(fit_cap)
    fit_balance = bool(config.get("rsf_fit_balance_classes", False))
    seed = int(config.get("random_seed", 2026))

    dataset_name = str(config.get("dataset", "dataset")).strip()
    source_hash = _hash_source_keys(source_subject_keys)
    bg_tag = "bgds4" if bool(config.get("source_train_bg_downsample_enable", False)) else "nobgds"
    cache_dir = (
        _rsf_cache_root(config)
        / dataset_name
        / f"heldout_{held_out_key}"
        / f"{rsf_mode}_dim{dim}_ea{int(use_ea)}_{bg_tag}_fitcap{fit_cap if fit_cap is not None else 'all'}_bal{int(fit_balance)}_lam{str(domain_lambda).replace('.', 'p')}"
        / f"seed_{seed}_{source_hash}"
    )
    meta_path = cache_dir / "meta.json"

    expected_files: List[Path] = [
        cache_dir / "W.npy",
        cache_dir / f"{held_out_key}_target_x.npy",
        cache_dir / f"{held_out_key}_target_y.npy",
    ]
    for sub_key in source_subject_keys:
        expected_files.extend(
            [
                cache_dir / f"{sub_key}_train_x.npy",
                cache_dir / f"{sub_key}_train_y.npy",
                cache_dir / f"{sub_key}_val_x.npy",
                cache_dir / f"{sub_key}_val_y.npy",
            ]
        )

    if meta_path.exists() and all(path.exists() for path in expected_files):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {"cache_dir": cache_dir, "meta": meta}

    cache_dir.mkdir(parents=True, exist_ok=True)

    fit_trials_parts: List[np.ndarray] = []
    fit_labels_parts: List[np.ndarray] = []
    fit_subject_counts: Dict[str, int] = {}
    fit_subject_class_counts: Dict[str, Dict[int, int]] = {}
    fit_subject_class_covs: Dict[str, Dict[int, np.ndarray]] = {}

    for sub_key in source_subject_keys:
        x_mm, y_mm = _load_subject_xy(Path(subject_file_map[sub_key]), use_ea=use_ea, ea_cov_eps=ea_cov_eps)
        fit_idx = _fit_indices_for_subject(
            y_mm,
            np.asarray(source_train_indices_map[sub_key], dtype=np.int64),
            max_trials_per_class=fit_cap,
            balance_classes=fit_balance,
            seed=seed,
            subject_key=sub_key,
        )
        fit_subject_counts[sub_key] = int(fit_idx.shape[0])
        if fit_idx.size == 0:
            continue
        sub_trials = np.asarray(x_mm[fit_idx], dtype=np.float32)
        sub_labels = np.asarray(y_mm[fit_idx], dtype=np.int64)
        fit_trials_parts.append(sub_trials)
        fit_labels_parts.append(sub_labels)
        sub_counts: Dict[int, int] = {}
        sub_covs: Dict[int, np.ndarray] = {}
        trial_cov = covariances(np.asarray(sub_trials, dtype=np.float64), estimator=str(cov_estimator))
        for cls in np.unique(sub_labels):
            cls = int(cls)
            cls_mask = sub_labels == cls
            sub_counts[cls] = int(np.sum(cls_mask))
            if sub_counts[cls] > 0:
                sub_covs[cls] = mean_covariance(trial_cov[cls_mask], metric="riemann")
        fit_subject_class_counts[sub_key] = sub_counts
        fit_subject_class_covs[sub_key] = sub_covs

    if not fit_trials_parts:
        raise RuntimeError("RSF fit received no source training samples.")

    fit_trials = np.concatenate(fit_trials_parts, axis=0)
    fit_labels = np.concatenate(fit_labels_parts, axis=0)
    if rsf_mode == "plain":
        w, score = fit_plain_rsf(
            fit_trials,
            fit_labels,
            dim=dim,
            cov_estimator=cov_estimator,
            solver=solver,
            maxiter=maxiter,
            tolerance=tolerance,
            seed=seed,
        )
        fit_info: Dict[str, Any] = {"riemann_class_distance": float(score)}
    elif rsf_mode == "cc_domain_aware":
        all_cov = covariances(np.asarray(fit_trials, dtype=np.float64), estimator=str(cov_estimator))
        global_class_covs: Dict[int, np.ndarray] = {}
        for cls in np.unique(fit_labels):
            cls = int(cls)
            global_class_covs[cls] = mean_covariance(all_cov[fit_labels == cls], metric="riemann")
        w, fit_info = fit_cc_domain_aware_rsf(
            global_class_covs=global_class_covs,
            subject_class_covs=fit_subject_class_covs,
            dim=dim,
            domain_lambda=domain_lambda,
            solver=solver,
            maxiter=maxiter,
            tolerance=tolerance,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported rsf_mode={rsf_mode}")
    _save_array(cache_dir / "W.npy", np.asarray(w, dtype=np.float32))

    for sub_key in source_subject_keys:
        x_mm, y_mm = _load_subject_xy(Path(subject_file_map[sub_key]), use_ea=use_ea, ea_cov_eps=ea_cov_eps)
        train_idx = np.asarray(source_train_indices_map[sub_key], dtype=np.int64)
        val_idx = np.asarray(source_val_indices_map[sub_key], dtype=np.int64)
        train_x = apply_spatial_filter(w, np.asarray(x_mm[train_idx], dtype=np.float32))
        val_x = apply_spatial_filter(w, np.asarray(x_mm[val_idx], dtype=np.float32))
        _save_array(cache_dir / f"{sub_key}_train_x.npy", train_x)
        _save_array(cache_dir / f"{sub_key}_train_y.npy", np.asarray(y_mm[train_idx], dtype=np.int64))
        _save_array(cache_dir / f"{sub_key}_val_x.npy", val_x)
        _save_array(cache_dir / f"{sub_key}_val_y.npy", np.asarray(y_mm[val_idx], dtype=np.int64))

    target_x_mm, target_y_mm = _load_subject_xy(Path(subject_file_map[held_out_key]), use_ea=use_ea, ea_cov_eps=ea_cov_eps)
    target_idx = np.arange(int(target_y_mm.shape[0]), dtype=np.int64)
    target_x = apply_spatial_filter(w, np.asarray(target_x_mm[target_idx], dtype=np.float32))
    _save_array(cache_dir / f"{held_out_key}_target_x.npy", target_x)
    _save_array(cache_dir / f"{held_out_key}_target_y.npy", np.asarray(target_y_mm[target_idx], dtype=np.int64))

    meta = {
        "held_out_key": held_out_key,
        "rsf_mode": rsf_mode,
        "source_subject_keys": list(source_subject_keys),
        "dataset": dataset_name,
        "dim": dim,
        "ea_enable": use_ea,
        "ea_cov_eps": ea_cov_eps,
        "rsf_cov_estimator": cov_estimator,
        "rsf_solver": solver,
        "rsf_maxiter": maxiter,
        "rsf_tolerance": tolerance,
        "rsf_domain_lambda": float(domain_lambda),
        "rsf_fit_max_trials_per_class_per_subject": fit_cap,
        "rsf_fit_balance_classes": bool(fit_balance),
        "fit_trials_total": int(fit_trials.shape[0]),
        "fit_subject_counts": fit_subject_counts,
        "fit_subject_class_counts": fit_subject_class_counts,
    }
    meta.update(fit_info)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"cache_dir": cache_dir, "meta": meta}
