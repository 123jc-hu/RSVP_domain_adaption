from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from Data.datamodule import NPZMemmapSubsetDataset


def _load_xy_memmap(subject_file: str):
    cache_dir = NPZMemmapSubsetDataset._ensure_npz_cache(
        Path(subject_file), ["x_data.npy", "y_data.npy"]
    )
    x = np.load(cache_dir / "x_data.npy", mmap_mode="r")
    y = np.load(cache_dir / "y_data.npy", mmap_mode="r")
    return x, y


def _trial_covariance(trial: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # trial shape: (C, T)
    trial = np.asarray(trial, dtype=np.float64)
    trial = trial - trial.mean(axis=1, keepdims=True)
    cov = (trial @ trial.T) / max(int(trial.shape[1]), 1)
    cov = 0.5 * (cov + cov.T)
    cov = cov + eps * np.eye(cov.shape[0], dtype=np.float64)
    return cov


def _mean_covariance(
    x: np.ndarray,
    indices: np.ndarray,
    *,
    max_trials: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if indices.size == 0:
        return None
    if indices.size > max_trials:
        indices = rng.choice(indices, size=max_trials, replace=False)
    covs = [_trial_covariance(x[int(i)]) for i in indices]
    return np.mean(covs, axis=0)


def _log_spd(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    log_diag = np.diag(np.log(eigvals))
    return eigvecs @ log_diag @ eigvecs.T


def _log_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    la = _log_spd(a)
    lb = _log_spd(b)
    return float(np.linalg.norm(la - lb, ord="fro"))


def compute_pccs_source_scores(
    *,
    target_subject_file: str,
    source_subject_files: Dict[str, str],
    positive_label: int = 1,
    background_label: int = 0,
    max_trials_per_class: int = 128,
    seed: int = 2024,
) -> Dict[str, float]:
    """
    Compute a PCCS-style compatibility score for each source subject.
    Current version is a lightweight scaffold:
    - source prototype: mean covariance of source positive class (P300)
    - target reference: mean covariance of target background class
    - score: negative log-Euclidean distance (larger is better)
    """
    rng = np.random.default_rng(int(seed))
    target_x, target_y = _load_xy_memmap(target_subject_file)
    target_bg_idx = np.where(np.asarray(target_y) == int(background_label))[0]
    target_bg_cov = _mean_covariance(
        target_x,
        target_bg_idx,
        max_trials=int(max_trials_per_class),
        rng=rng,
    )
    if target_bg_cov is None:
        # Fallback: if target has no background (unexpected), return empty scores.
        return {}

    scores: Dict[str, float] = {}
    for sub_key, sub_file in source_subject_files.items():
        src_x, src_y = _load_xy_memmap(sub_file)
        src_pos_idx = np.where(np.asarray(src_y) == int(positive_label))[0]
        src_pos_cov = _mean_covariance(
            src_x,
            src_pos_idx,
            max_trials=int(max_trials_per_class),
            rng=rng,
        )
        if src_pos_cov is None:
            continue
        dist = _log_euclidean_distance(src_pos_cov, target_bg_cov)
        scores[sub_key] = -dist

    return scores

