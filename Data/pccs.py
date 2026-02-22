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


def _trial_covariance(trial: np.ndarray, eps: float = 1e-6, shrinkage: float = 0.0) -> np.ndarray:
    trial = np.asarray(trial, dtype=np.float64)
    if trial.ndim == 3 and trial.shape[0] == 1:
        trial = trial[0]
    if trial.ndim != 2:
        raise ValueError(f"Unsupported trial shape for covariance: {trial.shape}")
    # Expected shape is (C, T). If data is (T, C), transpose.
    if trial.shape[0] > trial.shape[1]:
        trial = trial.T

    trial = trial - trial.mean(axis=1, keepdims=True)

    t = max(int(trial.shape[1]), 1)
    cov = (trial @ trial.T) / t
    cov = 0.5 * (cov + cov.T)

    if shrinkage > 0.0:
        c = int(cov.shape[0])
        mu = float(np.trace(cov) / max(c, 1))
        cov = (1.0 - shrinkage) * cov + shrinkage * mu * np.eye(c, dtype=np.float64)

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
) -> Optional[np.ndarray]:
    cache_key = (
        str(subject_file),
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
        return _PROTOTYPE_CACHE[cache_key]

    x, y = _load_xy_memmap(subject_file)
    idx = np.where(np.asarray(y) == int(class_label))[0].astype(np.int64)

    if idx.size < int(min_trials_per_class):
        _PROTOTYPE_CACHE[cache_key] = None
        return None

    if max_trials_per_class > 0 and idx.size > int(max_trials_per_class):
        local_seed = int(seed) + _stable_int_from_string(str(subject_file)) + int(class_label) * 131
        rng = np.random.default_rng(local_seed)
        idx = rng.choice(idx, size=int(max_trials_per_class), replace=False)

    covs = [
        _trial_covariance(x[int(i)], eps=float(cov_eps), shrinkage=float(cov_shrinkage))
        for i in idx
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

    _PROTOTYPE_CACHE[cache_key] = proto
    return proto


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
    return_details: bool = False,
    return_prototypes: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Formal PCCS scoring for source subjects.

    For each source subject s:
      1) Build source P300 prototype on SPD manifold.
      2) Build target background prototype on SPD manifold.
      3) Score(s) = -d_R(source_p300, target_background)
         where d_R is AIRM (default) or log-Euclidean distance.

    Larger score means stronger morphological compatibility.
    """
    target_x, target_y = _load_xy_memmap(target_subject_file)
    target_bg_idx = np.where(np.asarray(target_y) == int(background_label))[0].astype(np.int64)
    target_total = int(target_bg_idx.size)
    target_used = int(min(target_total, max_trials_per_class)) if max_trials_per_class > 0 else target_total

    target_bg_proto = _subject_class_prototype(
        subject_file=target_subject_file,
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
    if target_bg_proto is None:
        if return_details:
            return {}, {
                "target": {
                    "subject_file": str(target_subject_file),
                    "background_total_trials": target_total,
                    "background_used_trials": 0,
                },
                "ranking": [],
            }
        return {}

    dist_mode = str(distance_metric).strip().lower()
    scores: Dict[str, float] = {}
    ranking_rows = []
    source_proto_map: Dict[str, np.ndarray] = {}
    for sub_key, sub_file in source_subject_files.items():
        src_x, src_y = _load_xy_memmap(sub_file)
        src_pos_idx = np.where(np.asarray(src_y) == int(positive_label))[0].astype(np.int64)
        src_total = int(src_pos_idx.size)
        src_used = int(min(src_total, max_trials_per_class)) if max_trials_per_class > 0 else src_total

        src_pos_proto = _subject_class_prototype(
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
        if src_pos_proto is None:
            continue

        if dist_mode in ("airm", "riemann", "riemannian"):
            dist = _riemannian_distance_airm(src_pos_proto, target_bg_proto, eps=float(cov_eps))
        else:
            dist = _distance_log_euclidean(src_pos_proto, target_bg_proto, eps=float(cov_eps))

        scores[sub_key] = -float(dist)
        if return_prototypes:
            source_proto_map[sub_key] = src_pos_proto
        src_stats = _prototype_stats(src_pos_proto, float(cov_eps))
        ranking_rows.append(
            {
                "subject": sub_key,
                "score": float(scores[sub_key]),
                "distance": float(dist),
                "positive_total_trials": src_total,
                "positive_used_trials": src_used,
                "source_proto_trace": src_stats["trace"],
                "source_proto_logdet": src_stats["logdet"],
                "source_proto_cond": src_stats["cond"],
            }
        )

    if not return_details:
        return scores

    ranking_rows.sort(key=lambda r: r["score"], reverse=True)
    tgt_stats = _prototype_stats(target_bg_proto, float(cov_eps))
    details: Dict[str, Any] = {
        "target": {
            "subject_file": str(target_subject_file),
            "background_total_trials": target_total,
            "background_used_trials": target_used,
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
        },
        "ranking": ranking_rows,
    }
    if return_prototypes:
        details["prototypes"] = {
            "target_bg": target_bg_proto,
            "source_pos": source_proto_map,
        }
    return scores, details
