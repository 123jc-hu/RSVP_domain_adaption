from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib

import numpy as np

from Data.datamodule import NPZMemmapSubsetDataset


def _load_xy_memmap(subject_file: str):
    cache_dir = NPZMemmapSubsetDataset._ensure_npz_cache(
        Path(subject_file), ["x_data.npy", "y_data.npy"]
    )
    x = np.load(cache_dir / "x_data.npy", mmap_mode="r")
    y = np.load(cache_dir / "y_data.npy", mmap_mode="r")
    return x, y


def _stable_int_from_string(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def _project_spd(mat: np.ndarray, eps: float) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (mat + mat.T)


def _spd_log(mat: np.ndarray, eps: float) -> np.ndarray:
    mat = _project_spd(mat, eps)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T


def _spd_exp(mat: np.ndarray) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T


def _spd_sqrt(mat: np.ndarray, eps: float) -> np.ndarray:
    mat = _project_spd(mat, eps)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _spd_invsqrt(mat: np.ndarray, eps: float) -> np.ndarray:
    mat = _project_spd(mat, eps)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _prototype_stats(proto: np.ndarray, eps: float) -> Dict[str, float]:
    proto = _project_spd(proto, eps)
    eigvals, _ = np.linalg.eigh(proto)
    eigvals = np.clip(eigvals, eps, None)
    return {
        "trace": float(np.trace(proto)),
        "logdet": float(np.sum(np.log(eigvals))),
        "min_eig": float(np.min(eigvals)),
        "max_eig": float(np.max(eigvals)),
        "cond": float(np.max(eigvals) / np.min(eigvals)),
    }


def to_correlation(cov: np.ndarray, diag_eps: float = 1e-12, spd_eps: float = 1e-12) -> np.ndarray:
    """
    Convert SPD covariance to correlation-like SPD matrix.
    """
    cov = _project_spd(np.asarray(cov, dtype=np.float64), spd_eps)
    diag = np.diag(cov).astype(np.float64)
    diag = np.maximum(diag, float(diag_eps))
    inv_std = 1.0 / np.sqrt(diag)
    corr = (inv_std[:, None] * cov) * inv_std[None, :]
    corr = 0.5 * (corr + corr.T)
    corr = _project_spd(corr, spd_eps)
    # Keep exact unit diagonal for interpretability.
    np.fill_diagonal(corr, 1.0)
    return _project_spd(corr, spd_eps)


def _cov_sample(centered: np.ndarray) -> np.ndarray:
    # centered shape: [C, T]
    c, t = centered.shape
    denom = max(int(t - 1), 1)
    return (centered @ centered.T) / float(denom)


def _cov_lw(centered: np.ndarray) -> np.ndarray:
    """
    Lightweight Ledoit-Wolf style shrinkage estimator (self-contained).
    """
    c, t = centered.shape
    t_eff = max(int(t), 1)
    s = (centered @ centered.T) / float(t_eff)  # biased covariance
    mu = float(np.trace(s) / max(c, 1))
    target = mu * np.eye(c, dtype=np.float64)

    z = centered.T  # [T, C]
    x2 = z * z
    beta_hat = float(np.sum((x2.T @ x2) / float(t_eff) - (s * s)))
    delta_hat = float(np.sum((s - target) ** 2))
    if delta_hat <= 0.0:
        shrinkage = 0.0
    else:
        shrinkage = float(np.clip(beta_hat / delta_hat, 0.0, 1.0))
    return (1.0 - shrinkage) * s + shrinkage * target


def _cov_oas(centered: np.ndarray) -> np.ndarray:
    """
    Oracle Approximating Shrinkage (OAS) covariance estimator (self-contained).
    """
    c, t = centered.shape
    t_eff = max(int(t), 1)
    s = (centered @ centered.T) / float(t_eff)  # biased covariance
    mu = float(np.trace(s) / max(c, 1))
    target = mu * np.eye(c, dtype=np.float64)

    tr_s = float(np.trace(s))
    tr_s2 = float(np.trace(s @ s))
    p = float(max(c, 1))
    n = float(max(t_eff, 1))

    num = (1.0 - 2.0 / p) * tr_s2 + tr_s * tr_s
    den = (n + 1.0 - 2.0 / p) * (tr_s2 - (tr_s * tr_s) / p)
    if den <= 0.0:
        shrinkage = 1.0
    else:
        shrinkage = float(np.clip(num / den, 0.0, 1.0))
    return (1.0 - shrinkage) * s + shrinkage * target


def estimate_spd_cov(
    trial: np.ndarray,
    *,
    reg_eps: float = 1e-6,
    cov_estimator: str = "sample",
    cov_shrinkage: float = 0.0,
    use_correlation: bool = True,
    correlation_eps: float = 1e-12,
    input_layout: str = "channel_first",  # channel_first | time_first | auto
) -> np.ndarray:
    """
    Estimate one trial covariance with explicit [C, T] handling and SPD guarantee.
    """
    x = np.asarray(trial, dtype=np.float64)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Unsupported trial shape for covariance: {x.shape}")

    # Ensure [C, T] via explicit layout control.
    layout = str(input_layout).strip().lower()
    if layout in ("time_first", "t_c", "time_channel"):
        x = x.T
    elif layout in ("channel_first", "c_t", "channel_time"):
        pass
    elif layout in ("auto",):
        # Conservative auto: transpose only when first axis is much larger.
        if x.shape[0] >= 4 * x.shape[1]:
            x = x.T
    else:
        raise ValueError(f"Unknown input_layout: {input_layout}")

    x = x - x.mean(axis=1, keepdims=True)
    est = str(cov_estimator).strip().lower()
    if est in ("sample", "empirical"):
        cov = _cov_sample(x)
        if cov_shrinkage > 0.0:
            c = int(cov.shape[0])
            mu = float(np.trace(cov) / max(c, 1))
            cov = (1.0 - float(cov_shrinkage)) * cov + float(cov_shrinkage) * mu * np.eye(c, dtype=np.float64)
    elif est in ("lw", "ledoitwolf", "ledoit_wolf"):
        cov = _cov_lw(x)
    elif est in ("oas",):
        cov = _cov_oas(x)
    else:
        raise ValueError(f"Unknown cov_estimator: {cov_estimator}")

    cov = 0.5 * (cov + cov.T)
    cov = cov + float(reg_eps) * np.eye(cov.shape[0], dtype=np.float64)
    cov = _project_spd(cov, float(reg_eps))

    if bool(use_correlation):
        cov = to_correlation(
            cov,
            diag_eps=float(correlation_eps),
            spd_eps=float(reg_eps),
        )
        cov = cov + float(reg_eps) * np.eye(cov.shape[0], dtype=np.float64)
        cov = _project_spd(cov, float(reg_eps))
    return cov


def _log_euclidean_mean(covs: Sequence[np.ndarray], eps: float) -> np.ndarray:
    logs = [_spd_log(c, eps) for c in covs]
    mean_log = np.mean(logs, axis=0)
    return _project_spd(_spd_exp(mean_log), eps)


def _riemannian_mean_airm(
    covs: Sequence[np.ndarray],
    *,
    max_iter: int,
    tol: float,
    eps: float,
) -> np.ndarray:
    if len(covs) == 1:
        return _project_spd(covs[0], eps)

    g = _log_euclidean_mean(covs, eps)
    for _ in range(max_iter):
        g_sqrt = _spd_sqrt(g, eps)
        g_invsqrt = _spd_invsqrt(g, eps)

        delta = np.zeros_like(g)
        for c in covs:
            c_t = g_invsqrt @ c @ g_invsqrt
            delta += _spd_log(c_t, eps)
        delta /= float(len(covs))

        step_norm = float(np.linalg.norm(delta, ord="fro"))
        g = g_sqrt @ _spd_exp(delta) @ g_sqrt
        g = _project_spd(g, eps)
        if step_norm < tol:
            break
    return g


def _riemannian_distance_airm(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    a_invsqrt = _spd_invsqrt(a, eps)
    c = a_invsqrt @ b @ a_invsqrt
    c = _project_spd(c, eps)
    eigvals, _ = np.linalg.eigh(c)
    eigvals = np.clip(eigvals, eps, None)
    # eigvals is 1D, so this is vector L2 norm (canonical AIRM form).
    return float(np.linalg.norm(np.log(eigvals)))


def _distance_log_euclidean(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    return float(np.linalg.norm(_spd_log(a, eps) - _spd_log(b, eps), ord="fro"))


def _subsample_indices(
    indices: np.ndarray,
    *,
    max_trials: Optional[int],
    seed: int,
    salt: str,
) -> np.ndarray:
    if max_trials is None:
        return indices
    m = int(max_trials)
    if m <= 0 or indices.size <= m:
        return indices
    rng = np.random.default_rng(int(seed) + _stable_int_from_string(salt))
    return rng.choice(indices, size=m, replace=False).astype(np.int64)


def _target_amplitude_scores(trials: np.ndarray) -> np.ndarray:
    # peak-to-peak proxy across channels and time.
    tmax = np.max(trials, axis=(1, 2))
    tmin = np.min(trials, axis=(1, 2))
    return (tmax - tmin).astype(np.float64)


def _target_kurtosis_scores(trials: np.ndarray) -> np.ndarray:
    n = int(trials.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v = np.asarray(trials[i], dtype=np.float64).reshape(-1)
        v = v - np.mean(v)
        std = float(np.std(v))
        if std <= 1e-12:
            out[i] = 0.0
            continue
        z = v / std
        # Excess kurtosis proxy, large value means more spike-like.
        out[i] = float(np.mean(z ** 4) - 3.0)
    return out


def select_target_bg_subset(
    trials: np.ndarray,
    mode: str = "amplitude",
    ratio: float = 0.7,
) -> np.ndarray:
    """
    Select target background-like subset.
    Returns subset trials array (same dims except first axis).
    """
    idx = select_target_bg_subset_indices(trials=trials, mode=mode, ratio=ratio)
    return np.asarray(trials)[idx]


def select_target_bg_subset_indices(
    *,
    trials: np.ndarray,
    mode: str = "amplitude",
    ratio: float = 0.7,
    min_keep: int = 8,
) -> np.ndarray:
    n = int(trials.shape[0])
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= int(min_keep):
        return np.arange(n, dtype=np.int64)

    keep = int(round(float(np.clip(ratio, 0.05, 1.0)) * n))
    keep = max(int(min_keep), min(keep, n))
    mode_norm = str(mode).strip().lower()
    if mode_norm in ("all", "none"):
        return np.arange(n, dtype=np.int64)

    if mode_norm in ("amplitude", "energy"):
        scores = _target_amplitude_scores(trials)
    elif mode_norm in ("kurtosis", "kurt"):
        scores = _target_kurtosis_scores(trials)
    else:
        # safe fallback
        scores = _target_amplitude_scores(trials)

    # lower score -> more likely background
    order = np.argsort(scores)
    picked = np.sort(order[:keep].astype(np.int64))
    return picked


def _prototype_from_indices(
    *,
    subject_file: str,
    indices: np.ndarray,
    min_trials: int,
    seed: int,
    max_trials: Optional[int],
    sample_salt: str,
    cov_eps: float,
    cov_shrinkage: float,
    cov_estimator: str,
    use_correlation: bool,
    correlation_eps: float,
    input_layout: str,
    metric: str,
    mean_max_iter: int,
    mean_tol: float,
) -> Tuple[Optional[np.ndarray], int, int]:
    idx = np.asarray(indices, dtype=np.int64).ravel()
    total = int(idx.size)
    if total < int(min_trials):
        return None, total, 0

    idx_used = _subsample_indices(
        idx,
        max_trials=max_trials,
        seed=seed,
        salt=sample_salt,
    )
    used = int(idx_used.size)
    if used < int(min_trials):
        return None, total, used

    x, _ = _load_xy_memmap(subject_file)
    covs = [
        estimate_spd_cov(
            x[int(i)],
            reg_eps=float(cov_eps),
            cov_estimator=str(cov_estimator),
            cov_shrinkage=float(cov_shrinkage),
            use_correlation=bool(use_correlation),
            correlation_eps=float(correlation_eps),
            input_layout=str(input_layout),
        )
        for i in idx_used
    ]

    metric_norm = str(metric).strip().lower()
    if metric_norm in ("airm", "riemann", "riemannian"):
        proto = _riemannian_mean_airm(
            covs,
            max_iter=int(mean_max_iter),
            tol=float(mean_tol),
            eps=float(cov_eps),
        )
    else:
        proto = _log_euclidean_mean(covs, eps=float(cov_eps))
    return proto, total, used


def build_source_prototypes(
    *,
    subject_key: str,
    subject_file: str,
    positive_label: int,
    background_label: int,
    max_trials_per_class: int,
    min_trials_per_class: int,
    seed: int,
    cov_eps: float,
    cov_shrinkage: float,
    cov_estimator: str,
    use_correlation: bool,
    correlation_eps: float,
    input_layout: str,
    metric: str,
    mean_max_iter: int,
    mean_tol: float,
) -> Dict[str, Any]:
    _, y = _load_xy_memmap(subject_file)
    pos_idx = np.where(np.asarray(y) == int(positive_label))[0].astype(np.int64)
    bg_idx = np.where(np.asarray(y) == int(background_label))[0].astype(np.int64)

    pos_proto, pos_total, pos_used = _prototype_from_indices(
        subject_file=subject_file,
        indices=pos_idx,
        min_trials=int(min_trials_per_class),
        seed=int(seed),
        max_trials=int(max_trials_per_class) if int(max_trials_per_class) > 0 else None,
        sample_salt=f"{subject_file}|label:{int(positive_label)}",
        cov_eps=float(cov_eps),
        cov_shrinkage=float(cov_shrinkage),
        cov_estimator=str(cov_estimator),
        use_correlation=bool(use_correlation),
        correlation_eps=float(correlation_eps),
        input_layout=str(input_layout),
        metric=str(metric),
        mean_max_iter=int(mean_max_iter),
        mean_tol=float(mean_tol),
    )
    bg_proto, bg_total, bg_used = _prototype_from_indices(
        subject_file=subject_file,
        indices=bg_idx,
        min_trials=int(min_trials_per_class),
        seed=int(seed),
        max_trials=int(max_trials_per_class) if int(max_trials_per_class) > 0 else None,
        sample_salt=f"{subject_file}|label:{int(background_label)}",
        cov_eps=float(cov_eps),
        cov_shrinkage=float(cov_shrinkage),
        cov_estimator=str(cov_estimator),
        use_correlation=bool(use_correlation),
        correlation_eps=float(correlation_eps),
        input_layout=str(input_layout),
        metric=str(metric),
        mean_max_iter=int(mean_max_iter),
        mean_tol=float(mean_tol),
    )

    return {
        "subject": str(subject_key),
        "subject_file": str(subject_file),
        "p300_proto": pos_proto,
        "bg_proto": bg_proto,
        "positive_total_trials": int(pos_total),
        "positive_used_trials": int(pos_used),
        "background_total_trials": int(bg_total),
        "background_used_trials": int(bg_used),
    }


def build_prototypes(**kwargs):
    """
    Backward/PRD-friendly alias.
    """
    return build_source_prototypes(**kwargs)


def build_target_prototype(
    *,
    target_subject_file: str,
    background_label: int,
    min_trials_per_class: int,
    seed: int,
    cov_eps: float,
    cov_shrinkage: float,
    cov_estimator: str,
    use_correlation: bool,
    correlation_eps: float,
    input_layout: str,
    metric: str,
    mean_max_iter: int,
    mean_tol: float,
    target_use_all_trials: bool,
    target_max_trials: Optional[int],
    target_bg_mode: str,
    target_bg_ratio: float,
) -> Dict[str, Any]:
    x, y = _load_xy_memmap(target_subject_file)
    target_n = int(x.shape[0])
    bg_n = int(np.sum(np.asarray(y) == int(background_label)))

    if bool(target_use_all_trials):
        cand_idx = np.arange(target_n, dtype=np.int64)
        cand_mode = "all_trials_as_candidates"
    else:
        cand_idx = np.where(np.asarray(y) == int(background_label))[0].astype(np.int64)
        cand_mode = "background_label_only_candidates"

    if cand_idx.size == 0:
        return {
            "prototype": None,
            "target_total_trials": int(target_n),
            "background_total_trials": int(bg_n),
            "candidate_total_trials": 0,
            "candidate_used_trials": 0,
            "target_used_trials": 0,
            "target_proto_mode": cand_mode,
        }

    cand_idx = _subsample_indices(
        cand_idx,
        max_trials=int(target_max_trials) if target_max_trials not in (None, 0) else None,
        seed=int(seed),
        salt=f"{target_subject_file}|target_candidates",
    )
    candidate_trials = np.asarray(x[cand_idx], dtype=np.float64)
    local_pick = select_target_bg_subset_indices(
        trials=candidate_trials,
        mode=str(target_bg_mode),
        ratio=float(target_bg_ratio),
        min_keep=max(1, int(min_trials_per_class)),
    )
    picked_idx = cand_idx[local_pick] if local_pick.size > 0 else cand_idx

    proto, total, used = _prototype_from_indices(
        subject_file=target_subject_file,
        indices=picked_idx,
        min_trials=max(1, int(min_trials_per_class)),
        seed=int(seed),
        max_trials=None,
        sample_salt=f"{target_subject_file}|target_bg_proto",
        cov_eps=float(cov_eps),
        cov_shrinkage=float(cov_shrinkage),
        cov_estimator=str(cov_estimator),
        use_correlation=bool(use_correlation),
        correlation_eps=float(correlation_eps),
        input_layout=str(input_layout),
        metric=str(metric),
        mean_max_iter=int(mean_max_iter),
        mean_tol=float(mean_tol),
    )
    return {
        "prototype": proto,
        "target_total_trials": int(target_n),
        "background_total_trials": int(bg_n),
        "candidate_total_trials": int(cand_idx.size),
        "candidate_used_trials": int(picked_idx.size),
        "target_used_trials": int(used),
        "target_proto_mode": f"{cand_mode}|{str(target_bg_mode)}@{float(target_bg_ratio):.2f}",
    }


def score_subject(
    *,
    source_subject: str,
    source_pos_proto: np.ndarray,
    source_bg_proto: np.ndarray,
    target_bg_proto: np.ndarray,
    metric: str,
    cov_eps: float,
    score_mode: str,
    score_eps: float,
) -> Dict[str, float]:
    metric_norm = str(metric).strip().lower()
    if metric_norm in ("airm", "riemann", "riemannian"):
        distance_fn = _riemannian_distance_airm
    else:
        distance_fn = _distance_log_euclidean

    # Updated discriminability: subject-internal class distance.
    d_i = float(distance_fn(source_pos_proto, source_bg_proto, eps=float(cov_eps)))
    # Similarity: source-bg vs target-bg.
    s_i = float(distance_fn(source_bg_proto, target_bg_proto, eps=float(cov_eps)))

    mode = str(score_mode).strip().lower()
    eps_s = float(max(score_eps, 1e-12))
    if mode in ("rpcs", "pccs", "ratio", "d_over_s"):
        score = float(d_i / (s_i + eps_s))
    elif mode in ("similarity_only", "similarity", "sim_only", "1_over_s"):
        score = float(1.0 / (s_i + eps_s))
    elif mode in ("discrim_only", "discriminability_only", "d_only"):
        score = float(d_i)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    return {
        "subject": str(source_subject),
        "discriminability_distance": float(d_i),
        "similarity_distance": float(s_i),
        "score": float(score),
    }


def _apply_hard_constraints(
    rows: List[Dict[str, Any]],
    *,
    tau_d: Optional[float],
    tau_s: Optional[float],
    tau_d_percentile: float,
    tau_s_percentile: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not rows:
        return [], {
            "tau_d": float("nan"),
            "tau_s": float("nan"),
            "tau_d_source": "none",
            "tau_s_source": "none",
            "gate_fallback_used": False,
        }

    d_arr = np.asarray([float(r["discriminability_distance"]) for r in rows], dtype=np.float64)
    s_arr = np.asarray([float(r["similarity_distance"]) for r in rows], dtype=np.float64)

    if tau_d is None:
        tau_d_eff = float(np.percentile(d_arr, float(tau_d_percentile)))
        tau_d_src = f"percentile_{float(tau_d_percentile):.1f}"
    else:
        tau_d_eff = float(tau_d)
        tau_d_src = "fixed"

    if tau_s is None:
        tau_s_eff = float(np.percentile(s_arr, float(tau_s_percentile)))
        tau_s_src = f"percentile_{float(tau_s_percentile):.1f}"
    else:
        tau_s_eff = float(tau_s)
        tau_s_src = "fixed"

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        passed = (
            float(row["discriminability_distance"]) >= tau_d_eff
            and float(row["similarity_distance"]) <= tau_s_eff
        )
        row["hard_pass"] = int(passed)
        if passed:
            filtered.append(row)

    gate_fallback_used = False
    if not filtered:
        filtered = list(rows)
        gate_fallback_used = True

    filtered.sort(key=lambda r: float(r["score"]), reverse=True)
    meta = {
        "tau_d": float(tau_d_eff),
        "tau_s": float(tau_s_eff),
        "tau_d_source": tau_d_src,
        "tau_s_source": tau_s_src,
        "gate_fallback_used": bool(gate_fallback_used),
    }
    return filtered, meta


def select_topk(
    rows: List[Dict[str, Any]],
    *,
    k: Optional[int] = None,
    tau_d: Optional[float] = None,
    tau_s: Optional[float] = None,
    tau_d_percentile: float = 30.0,
    tau_s_percentile: float = 70.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ranked, meta = _apply_hard_constraints(
        rows,
        tau_d=tau_d,
        tau_s=tau_s,
        tau_d_percentile=tau_d_percentile,
        tau_s_percentile=tau_s_percentile,
    )
    if k is not None and int(k) > 0:
        ranked = ranked[: int(k)]
    return ranked, meta


def compute_pccs_source_scores(
    *,
    target_subject_file: str,
    source_subject_files: Dict[str, str],
    positive_label: int = 1,
    background_label: int = 0,
    max_trials_per_class: int = 128,
    min_trials_per_class: int = 4,
    seed: int = 2026,
    cov_eps: float = 1e-6,  # SPD regularization epsilon
    cov_shrinkage: float = 0.0,
    cov_estimator: str = "sample",  # sample | oas | lw
    use_correlation: bool = True,
    correlation_eps: float = 1e-12,
    input_layout: str = "channel_first",  # channel_first | time_first | auto
    mean_metric: str = "riemann",
    distance_metric: str = "riemann",
    mean_max_iter: int = 20,
    mean_tol: float = 1e-6,
    score_mode: str = "rpcs",
    score_eps: float = 1e-6,  # denominator stabilization epsilon
    target_use_all_trials: bool = True,
    target_max_trials: Optional[int] = None,
    target_bg_mode: str = "amplitude",
    target_bg_ratio: float = 0.7,
    tau_d: Optional[float] = None,
    tau_s: Optional[float] = None,
    tau_d_percentile: float = 30.0,
    tau_s_percentile: float = 70.0,
    return_details: bool = False,
    return_prototypes: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    R-PCS / PCCS source scoring for cross-subject RSVP.

    D_i = d(P_i,p300, P_i,bg)        (larger is better)
    S_i = d(P_i,bg,   P_t,bg)        (smaller is better)
    score = D_i / (S_i + score_eps)  (R-PCS default)
    """
    metric = str(mean_metric).strip().lower()
    dist_metric = str(distance_metric).strip().lower()
    # Enforce consistent metric for mean and distance.
    if metric != dist_metric:
        dist_metric = metric

    target_info = build_target_prototype(
        target_subject_file=target_subject_file,
        background_label=int(background_label),
        min_trials_per_class=int(min_trials_per_class),
        seed=int(seed),
        cov_eps=float(cov_eps),
        cov_shrinkage=float(cov_shrinkage),
        cov_estimator=str(cov_estimator),
        use_correlation=bool(use_correlation),
        correlation_eps=float(correlation_eps),
        input_layout=str(input_layout),
        metric=str(metric),
        mean_max_iter=int(mean_max_iter),
        mean_tol=float(mean_tol),
        target_use_all_trials=bool(target_use_all_trials),
        target_max_trials=target_max_trials,
        target_bg_mode=str(target_bg_mode),
        target_bg_ratio=float(target_bg_ratio),
    )
    target_bg_proto = target_info.get("prototype", None)
    if target_bg_proto is None:
        if return_details:
            return {}, {
                "target": {
                    "subject_file": str(target_subject_file),
                    "target_total_trials": int(target_info.get("target_total_trials", 0)),
                    "target_used_trials": int(target_info.get("target_used_trials", 0)),
                    "background_total_trials": int(target_info.get("background_total_trials", 0)),
                    "background_used_trials": int(target_info.get("candidate_used_trials", 0)),
                    "target_proto_mode": str(target_info.get("target_proto_mode", "")),
                },
                "ranking": [],
            }
        return {}

    rows_all: List[Dict[str, Any]] = []
    source_pos_map: Dict[str, np.ndarray] = {}
    source_bg_map: Dict[str, np.ndarray] = {}

    for sub_key, sub_file in source_subject_files.items():
        src = build_source_prototypes(
            subject_key=sub_key,
            subject_file=sub_file,
            positive_label=int(positive_label),
            background_label=int(background_label),
            max_trials_per_class=int(max_trials_per_class),
            min_trials_per_class=int(min_trials_per_class),
            seed=int(seed),
            cov_eps=float(cov_eps),
            cov_shrinkage=float(cov_shrinkage),
            cov_estimator=str(cov_estimator),
            use_correlation=bool(use_correlation),
            correlation_eps=float(correlation_eps),
            input_layout=str(input_layout),
            metric=str(metric),
            mean_max_iter=int(mean_max_iter),
            mean_tol=float(mean_tol),
        )
        src_pos_proto = src.get("p300_proto", None)
        src_bg_proto = src.get("bg_proto", None)
        if src_pos_proto is None or src_bg_proto is None:
            continue

        sc = score_subject(
            source_subject=str(sub_key),
            source_pos_proto=src_pos_proto,
            source_bg_proto=src_bg_proto,
            target_bg_proto=target_bg_proto,
            metric=str(dist_metric),
            cov_eps=float(cov_eps),
            score_mode=str(score_mode),
            score_eps=float(score_eps),
        )

        src_pos_stats = _prototype_stats(src_pos_proto, float(cov_eps))
        src_bg_stats = _prototype_stats(src_bg_proto, float(cov_eps))
        row = {
            "subject": str(sub_key),
            "score": float(sc["score"]),
            "distance": float(sc["discriminability_distance"]),  # legacy column alias
            "discriminability_distance": float(sc["discriminability_distance"]),
            "similarity_distance": float(sc["similarity_distance"]),
            "positive_total_trials": int(src["positive_total_trials"]),
            "positive_used_trials": int(src["positive_used_trials"]),
            "background_total_trials": int(src["background_total_trials"]),
            "background_used_trials": int(src["background_used_trials"]),
            "source_pos_proto_trace": src_pos_stats["trace"],
            "source_pos_proto_logdet": src_pos_stats["logdet"],
            "source_pos_proto_cond": src_pos_stats["cond"],
            "source_bg_proto_trace": src_bg_stats["trace"],
            "source_bg_proto_logdet": src_bg_stats["logdet"],
            "source_bg_proto_cond": src_bg_stats["cond"],
            "score_mode": str(score_mode),
            "hard_pass": 0,
        }
        rows_all.append(row)

        if return_prototypes:
            source_pos_map[str(sub_key)] = src_pos_proto
            source_bg_map[str(sub_key)] = src_bg_proto

    if not rows_all:
        if return_details:
            return {}, {
                "target": {
                    "subject_file": str(target_subject_file),
                    "target_total_trials": int(target_info.get("target_total_trials", 0)),
                    "target_used_trials": int(target_info.get("target_used_trials", 0)),
                    "background_total_trials": int(target_info.get("background_total_trials", 0)),
                    "background_used_trials": int(target_info.get("candidate_used_trials", 0)),
                    "target_proto_mode": str(target_info.get("target_proto_mode", "")),
                },
                "ranking": [],
            }
        return {}

    rows_ranked, gate_meta = select_topk(
        rows_all,
        k=None,
        tau_d=tau_d,
        tau_s=tau_s,
        tau_d_percentile=float(tau_d_percentile),
        tau_s_percentile=float(tau_s_percentile),
    )
    scores = {str(r["subject"]): float(r["score"]) for r in rows_ranked}

    if not return_details:
        return scores

    tgt_stats = _prototype_stats(target_bg_proto, float(cov_eps))
    details: Dict[str, Any] = {
        "target": {
            "subject_file": str(target_subject_file),
            "target_total_trials": int(target_info.get("target_total_trials", 0)),
            "target_used_trials": int(target_info.get("target_used_trials", 0)),
            "background_total_trials": int(target_info.get("background_total_trials", 0)),
            "background_used_trials": int(target_info.get("candidate_used_trials", 0)),
            "target_proto_mode": str(target_info.get("target_proto_mode", "")),
            "target_bg_proto_trace": tgt_stats["trace"],
            "target_bg_proto_logdet": tgt_stats["logdet"],
            "target_bg_proto_cond": tgt_stats["cond"],
        },
        "config": {
            "positive_label": int(positive_label),
            "background_label": int(background_label),
            "max_trials_per_class": int(max_trials_per_class),
            "min_trials_per_class": int(min_trials_per_class),
            "cov_eps": float(cov_eps),
            "cov_shrinkage": float(cov_shrinkage),
            "cov_estimator": str(cov_estimator),
            "use_correlation": bool(use_correlation),
            "correlation_eps": float(correlation_eps),
            "input_layout": str(input_layout),
            "metric": str(metric),
            "mean_max_iter": int(mean_max_iter),
            "mean_tol": float(mean_tol),
            "score_mode": str(score_mode),
            "score_eps": float(score_eps),
            "target_use_all_trials": bool(target_use_all_trials),
            "target_max_trials": None if target_max_trials is None else int(target_max_trials),
            "target_bg_mode": str(target_bg_mode),
            "target_bg_ratio": float(target_bg_ratio),
            "tau_d": gate_meta["tau_d"],
            "tau_s": gate_meta["tau_s"],
            "tau_d_source": gate_meta["tau_d_source"],
            "tau_s_source": gate_meta["tau_s_source"],
            "gate_fallback_used": bool(gate_meta["gate_fallback_used"]),
        },
        "ranking": rows_ranked,
        "ranking_all": rows_all,
    }
    if return_prototypes:
        details["prototypes"] = {
            "target_bg": target_bg_proto,
            "source_pos": source_pos_map,
            "source_bg": source_bg_map,
        }
    return scores, details


def compute_rpcs_source_scores(**kwargs):
    """
    Preferred R-PCS naming entry.
    Backward compatibility: delegates to compute_pccs_source_scores.
    """
    return compute_pccs_source_scores(**kwargs)
