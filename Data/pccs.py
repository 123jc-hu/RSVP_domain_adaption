from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import zlib

import numpy as np

from Data.datamodule import NPZMemmapSubsetDataset


_PROTOTYPE_CACHE: Dict[Tuple, Optional[np.ndarray]] = {}


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


def _estimate_spd_cov(trial: np.ndarray, eps: float = 1e-6, shrinkage: float = 0.0) -> np.ndarray:
    """
    Estimate one trial covariance and enforce SPD with epsilon regularization.
    """
    trial = np.asarray(trial, dtype=np.float64)
    if trial.ndim == 3 and trial.shape[0] == 1:
        trial = trial[0]
    if trial.ndim != 2:
        raise ValueError(f"Unsupported trial shape for covariance: {trial.shape}")
    if trial.shape[0] > trial.shape[1]:
        trial = trial.T

    trial = trial - trial.mean(axis=1, keepdims=True)
    t = max(int(trial.shape[1]), 1)
    cov = (trial @ trial.T) / t
    cov = 0.5 * (cov + cov.T)

    if shrinkage > 0.0:
        n_ch = int(cov.shape[0])
        mu = float(np.trace(cov) / max(n_ch, 1))
        cov = (1.0 - shrinkage) * cov + shrinkage * mu * np.eye(n_ch, dtype=np.float64)

    cov = cov + eps * np.eye(cov.shape[0], dtype=np.float64)
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
    return float(np.linalg.norm(np.log(eigvals), ord=2))


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
    mean_metric: str,
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
        _estimate_spd_cov(x[int(i)], eps=float(cov_eps), shrinkage=float(cov_shrinkage))
        for i in idx_used
    ]

    metric = str(mean_metric).strip().lower()
    if metric in ("airm", "riemann", "riemannian"):
        proto = _riemannian_mean_airm(
            covs,
            max_iter=int(mean_max_iter),
            tol=float(mean_tol),
            eps=float(cov_eps),
        )
    else:
        proto = _log_euclidean_mean(covs, eps=float(cov_eps))
    return proto, total, used


def _subject_class_prototype(
    *,
    subject_file: str,
    class_label: int,
    max_trials_per_class: int,
    min_trials_per_class: int,
    seed: int,
    cov_eps: float,
    cov_shrinkage: float,
    mean_metric: str,
    mean_max_iter: int,
    mean_tol: float,
) -> Tuple[Optional[np.ndarray], int, int]:
    cache_key = (
        str(subject_file),
        "class",
        int(class_label),
        int(max_trials_per_class),
        int(min_trials_per_class),
        int(seed),
        float(cov_eps),
        float(cov_shrinkage),
        str(mean_metric).lower(),
        int(mean_max_iter),
        float(mean_tol),
    )
    if cache_key in _PROTOTYPE_CACHE:
        proto = _PROTOTYPE_CACHE[cache_key]
        _, y = _load_xy_memmap(subject_file)
        idx = np.where(np.asarray(y) == int(class_label))[0].astype(np.int64)
        total = int(idx.size)
        used = min(total, int(max_trials_per_class)) if int(max_trials_per_class) > 0 else total
        return proto, total, used

    _, y = _load_xy_memmap(subject_file)
    idx = np.where(np.asarray(y) == int(class_label))[0].astype(np.int64)
    proto, total, used = _prototype_from_indices(
        subject_file=subject_file,
        indices=idx,
        min_trials=int(min_trials_per_class),
        seed=int(seed),
        max_trials=int(max_trials_per_class) if int(max_trials_per_class) > 0 else None,
        sample_salt=f"{subject_file}|label:{int(class_label)}",
        cov_eps=float(cov_eps),
        cov_shrinkage=float(cov_shrinkage),
        mean_metric=str(mean_metric),
        mean_max_iter=int(mean_max_iter),
        mean_tol=float(mean_tol),
    )
    _PROTOTYPE_CACHE[cache_key] = proto
    return proto, total, used


def _compute_score(
    *,
    discriminability: float,
    similarity: float,
    score_mode: str,
    score_eps: float,
) -> float:
    mode = str(score_mode).strip().lower()
    eps = float(max(score_eps, 1e-12))
    if mode in ("rpcs", "pccs", "ratio", "d_over_s"):
        return float(discriminability / (similarity + eps))
    if mode in ("similarity_only", "similarity", "sim_only", "1_over_s"):
        return float(1.0 / (similarity + eps))
    if mode in ("discrim_only", "discriminability_only", "d_only"):
        return float(discriminability)
    raise ValueError(f"Unknown score_mode: {score_mode}")


def compute_pccs_source_scores(
    *,
    target_subject_file: str,
    source_subject_files: Dict[str, str],
    positive_label: int = 1,
    background_label: int = 0,
    max_trials_per_class: int = 128,
    min_trials_per_class: int = 4,
    seed: int = 2026,
    cov_eps: float = 1e-6,
    cov_shrinkage: float = 0.0,
    mean_metric: str = "airm",
    distance_metric: str = "airm",
    mean_max_iter: int = 20,
    mean_tol: float = 1e-6,
    score_mode: str = "rpcs",
    score_eps: float = 1e-8,
    target_use_all_trials: bool = True,
    target_max_trials: Optional[int] = None,
    return_details: bool = False,
    return_prototypes: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    R-PCS/PCCS source scoring on SPD manifold.

    D_i = d(P_i,p300, P_t,bg)
    S_i = d(P_i,bg,   P_t,bg)

    score_i:
    - rpcs:           D_i / (S_i + eps)
    - similarity_only 1 / (S_i + eps)
    - discrim_only    D_i
    """
    target_x, target_y = _load_xy_memmap(target_subject_file)
    target_n = int(target_x.shape[0])
    target_bg_n = int(np.sum(np.asarray(target_y) == int(background_label)))

    if bool(target_use_all_trials):
        target_idx = np.arange(target_n, dtype=np.int64)
        target_mode_desc = "all_trials_as_background"
    else:
        target_idx = np.where(np.asarray(target_y) == int(background_label))[0].astype(np.int64)
        target_mode_desc = "background_label_only"

    target_proto, target_total, target_used = _prototype_from_indices(
        subject_file=target_subject_file,
        indices=target_idx,
        min_trials=max(1, int(min_trials_per_class)),
        seed=int(seed),
        max_trials=int(target_max_trials) if target_max_trials not in (None, 0) else None,
        sample_salt=f"{target_subject_file}|target_proto",
        cov_eps=float(cov_eps),
        cov_shrinkage=float(cov_shrinkage),
        mean_metric=str(mean_metric),
        mean_max_iter=int(mean_max_iter),
        mean_tol=float(mean_tol),
    )
    if target_proto is None:
        if return_details:
            return {}, {
                "target": {
                    "subject_file": str(target_subject_file),
                    "target_total_trials": int(target_total),
                    "target_used_trials": int(target_used),
                    "background_total_trials": int(target_bg_n),
                    "target_proto_mode": target_mode_desc,
                },
                "ranking": [],
            }
        return {}

    dist_mode = str(distance_metric).strip().lower()
    if dist_mode in ("airm", "riemann", "riemannian"):
        distance_fn = _riemannian_distance_airm
    else:
        distance_fn = _distance_log_euclidean

    scores: Dict[str, float] = {}
    ranking_rows = []
    source_pos_map: Dict[str, np.ndarray] = {}
    source_bg_map: Dict[str, np.ndarray] = {}
    for sub_key, sub_file in source_subject_files.items():
        src_pos_proto, src_pos_total, src_pos_used = _subject_class_prototype(
            subject_file=sub_file,
            class_label=int(positive_label),
            max_trials_per_class=int(max_trials_per_class),
            min_trials_per_class=int(min_trials_per_class),
            seed=int(seed),
            cov_eps=float(cov_eps),
            cov_shrinkage=float(cov_shrinkage),
            mean_metric=str(mean_metric),
            mean_max_iter=int(mean_max_iter),
            mean_tol=float(mean_tol),
        )
        src_bg_proto, src_bg_total, src_bg_used = _subject_class_prototype(
            subject_file=sub_file,
            class_label=int(background_label),
            max_trials_per_class=int(max_trials_per_class),
            min_trials_per_class=int(min_trials_per_class),
            seed=int(seed),
            cov_eps=float(cov_eps),
            cov_shrinkage=float(cov_shrinkage),
            mean_metric=str(mean_metric),
            mean_max_iter=int(mean_max_iter),
            mean_tol=float(mean_tol),
        )
        if src_pos_proto is None or src_bg_proto is None:
            continue

        d_i = float(distance_fn(src_pos_proto, target_proto, eps=float(cov_eps)))
        s_i = float(distance_fn(src_bg_proto, target_proto, eps=float(cov_eps)))
        score_i = _compute_score(
            discriminability=d_i,
            similarity=s_i,
            score_mode=str(score_mode),
            score_eps=float(score_eps),
        )
        scores[sub_key] = float(score_i)

        if return_prototypes:
            source_pos_map[sub_key] = src_pos_proto
            source_bg_map[sub_key] = src_bg_proto

        src_pos_stats = _prototype_stats(src_pos_proto, float(cov_eps))
        src_bg_stats = _prototype_stats(src_bg_proto, float(cov_eps))
        ranking_rows.append(
            {
                "subject": sub_key,
                "score": float(score_i),
                "distance": float(d_i),  # backward-compatible plotting column
                "discriminability_distance": float(d_i),
                "similarity_distance": float(s_i),
                "positive_total_trials": int(src_pos_total),
                "positive_used_trials": int(src_pos_used),
                "background_total_trials": int(src_bg_total),
                "background_used_trials": int(src_bg_used),
                "source_pos_proto_trace": src_pos_stats["trace"],
                "source_pos_proto_logdet": src_pos_stats["logdet"],
                "source_pos_proto_cond": src_pos_stats["cond"],
                "source_bg_proto_trace": src_bg_stats["trace"],
                "source_bg_proto_logdet": src_bg_stats["logdet"],
                "source_bg_proto_cond": src_bg_stats["cond"],
                "score_mode": str(score_mode),
            }
        )

    if not return_details:
        return scores

    ranking_rows.sort(key=lambda r: r["score"], reverse=True)
    tgt_stats = _prototype_stats(target_proto, float(cov_eps))
    details: Dict[str, Any] = {
        "target": {
            "subject_file": str(target_subject_file),
            "target_total_trials": int(target_total),
            "target_used_trials": int(target_used),
            "background_total_trials": int(target_bg_n),
            "background_used_trials": int(target_used if bool(target_use_all_trials) else target_used),
            "target_proto_mode": target_mode_desc,
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
            "mean_metric": str(mean_metric),
            "distance_metric": str(distance_metric),
            "mean_max_iter": int(mean_max_iter),
            "mean_tol": float(mean_tol),
            "score_mode": str(score_mode),
            "score_eps": float(score_eps),
            "target_use_all_trials": bool(target_use_all_trials),
            "target_max_trials": None if target_max_trials is None else int(target_max_trials),
        },
        "ranking": ranking_rows,
    }
    if return_prototypes:
        details["prototypes"] = {
            "target_bg": target_proto,
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
