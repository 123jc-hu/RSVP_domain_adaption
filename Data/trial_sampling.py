from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from Data.npz_io import ensure_npz_cache
from Data.pccs import select_target_bg_subset_indices
from Data.source_selection import normalize_subject_key


SubjectLike = Union[str, int]


def load_subject_memmap(
    subject_file_map: Dict[str, str],
    subject_key: SubjectLike,
) -> Tuple[np.ndarray, np.ndarray]:
    skey = normalize_subject_key(subject_key)
    if skey not in subject_file_map:
        raise KeyError(f"Unknown subject key: {subject_key}")
    subject_file = subject_file_map[skey]
    cache_dir = ensure_npz_cache(Path(subject_file), ["x_data.npy", "y_data.npy"])
    x = np.load(cache_dir / "x_data.npy", mmap_mode="r")
    y = np.load(cache_dir / "y_data.npy", mmap_mode="r")
    return x, y


def get_subject_trials_by_label(
    *,
    subject_file_map: Dict[str, str],
    subject_key: SubjectLike,
    label: int,
    max_trials: Optional[int] = None,
    seed: int = 2026,
) -> np.ndarray:
    x, y = load_subject_memmap(subject_file_map, subject_key)
    idx = np.where(np.asarray(y) == int(label))[0].astype(np.int64)
    if max_trials is not None and int(max_trials) > 0 and idx.size > int(max_trials):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=int(max_trials), replace=False).astype(np.int64)
    return np.asarray(x[idx], dtype=np.float32)


def get_source_positive_trials(
    *,
    subject_file_map: Dict[str, str],
    source_subject_keys: List[str],
    positive_label: int = 1,
    max_trials_per_subject: Optional[int] = None,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for idx, sub in enumerate(source_subject_keys):
        data = get_subject_trials_by_label(
            subject_file_map=subject_file_map,
            subject_key=sub,
            label=int(positive_label),
            max_trials=max_trials_per_subject,
            seed=int(seed) + idx,
        )
        if data.ndim == 3 and int(data.shape[0]) > 0:
            out[str(sub)] = data
    return out


def get_target_background_trials(
    *,
    subject_file_map: Dict[str, str],
    test_subject_id: int,
    background_label: int = 0,
    target_use_all_trials: bool = True,
    target_max_trials: Optional[int] = None,
    target_bg_mode: str = "amplitude",
    target_bg_ratio: float = 0.7,
    target_bg_channel_indices: Optional[List[int]] = None,
    min_keep: int = 8,
    seed: int = 2026,
) -> np.ndarray:
    target_key = f"sub{int(test_subject_id)}"
    x, y = load_subject_memmap(subject_file_map, target_key)
    n = int(x.shape[0])

    if bool(target_use_all_trials):
        cand_idx = np.arange(n, dtype=np.int64)
    else:
        cand_idx = np.where(np.asarray(y) == int(background_label))[0].astype(np.int64)
    if cand_idx.size == 0:
        return np.empty((0,) + tuple(x.shape[1:]), dtype=np.float32)

    if target_max_trials is not None and int(target_max_trials) > 0 and cand_idx.size > int(target_max_trials):
        rng = np.random.default_rng(int(seed))
        cand_idx = rng.choice(cand_idx, size=int(target_max_trials), replace=False).astype(np.int64)

    candidate_trials = np.asarray(x[cand_idx], dtype=np.float64)
    local_pick = select_target_bg_subset_indices(
        trials=candidate_trials,
        mode=str(target_bg_mode),
        ratio=float(target_bg_ratio),
        min_keep=max(1, int(min_keep)),
        channel_indices=target_bg_channel_indices,
    )
    picked = cand_idx[local_pick] if int(local_pick.size) > 0 else cand_idx
    return np.asarray(x[picked], dtype=np.float32)
