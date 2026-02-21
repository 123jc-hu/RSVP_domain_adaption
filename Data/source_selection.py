import re
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


SubjectLike = Union[str, int]


def normalize_subject_key(subject: SubjectLike) -> str:
    if isinstance(subject, int):
        return f"sub{subject}"
    s = str(subject).strip()
    if re.fullmatch(r"\d+", s):
        return f"sub{int(s)}"
    m = re.fullmatch(r"sub(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"sub{int(m.group(1))}"
    raise ValueError(f"Invalid subject identifier: {subject}")


def normalize_subject_list(subjects: Optional[Sequence[SubjectLike]]) -> List[str]:
    if not subjects:
        return []
    return [normalize_subject_key(s) for s in subjects]


def normalize_score_map(score_map: Optional[Dict[SubjectLike, float]]) -> Dict[str, float]:
    if not score_map:
        return {}
    out: Dict[str, float] = {}
    for k, v in score_map.items():
        out[normalize_subject_key(k)] = float(v)
    return out


def _subject_sort_key(subject_key: str) -> int:
    return int(subject_key[3:])


def select_source_subjects(
    candidate_subjects: Sequence[str],
    held_out_subject: SubjectLike,
    *,
    mode: str = "all",
    seed: int = 2024,
    k: Optional[int] = None,
    manual_subjects: Optional[Sequence[SubjectLike]] = None,
    score_map: Optional[Dict[SubjectLike, float]] = None,
    min_score: Optional[float] = None,
) -> List[str]:
    held_out = normalize_subject_key(held_out_subject)
    normalized_candidates = [normalize_subject_key(s) for s in candidate_subjects]
    candidates = sorted({s for s in normalized_candidates if s != held_out}, key=_subject_sort_key)
    if not candidates:
        raise ValueError("No candidate source subjects available after excluding held-out subject.")

    mode_norm = str(mode).strip().lower()

    if mode_norm in ("all", "full"):
        selected = candidates
    elif mode_norm in ("random_k", "random", "rand_k"):
        if k is None or int(k) <= 0:
            raise ValueError("source_selection_k must be > 0 when source_selection_mode=random_k.")
        k_eff = min(int(k), len(candidates))
        held_out_id = int(normalize_subject_key(held_out)[3:])
        rng = np.random.default_rng(int(seed) + held_out_id)
        picked = rng.choice(len(candidates), size=k_eff, replace=False)
        selected = sorted([candidates[int(i)] for i in picked], key=_subject_sort_key)
    elif mode_norm in ("manual", "list", "fixed"):
        manual = normalize_subject_list(manual_subjects)
        selected = sorted([s for s in manual if s in set(candidates)], key=_subject_sort_key)
        if k is not None and int(k) > 0:
            selected = selected[: int(k)]
        if not selected:
            raise ValueError("source_selection_mode=manual but no valid source subjects were provided.")
    elif mode_norm in ("scores", "score_topk", "topk"):
        scores = normalize_score_map(score_map)
        scored = [(s, scores[s]) for s in candidates if s in scores]
        if min_score is not None:
            scored = [(s, v) for s, v in scored if float(v) >= float(min_score)]
        if not scored:
            raise ValueError("source_selection_mode=scores but no subject passed score constraints.")
        scored.sort(key=lambda x: (-float(x[1]), _subject_sort_key(x[0])))
        if k is not None and int(k) > 0:
            scored = scored[: int(k)]
        selected = [s for s, _ in scored]
    else:
        raise ValueError(f"Unknown source_selection_mode: {mode}")

    if not selected:
        raise ValueError("Source subject selection returned empty set.")
    return selected
