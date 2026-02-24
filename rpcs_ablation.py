import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Data.rpcs import compute_rpcs_source_scores


def _normalize_subject_key(subject: str) -> str:
    s = str(subject).strip()
    m = re.fullmatch(r"sub(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"sub{int(m.group(1))}"
    if re.fullmatch(r"\d+", s):
        return f"sub{int(s)}"
    raise ValueError(f"Invalid subject id: {subject}")


def _discover_subject_files(data_dir: Path) -> Dict[str, str]:
    subject_map: Dict[str, str] = {}
    for f in sorted(data_dir.glob("*.npz")):
        if "_10band" in f.name:
            continue
        m = re.search(r"sub(\d+)", f.name, flags=re.IGNORECASE)
        if not m:
            continue
        key = f"sub{int(m.group(1))}"
        subject_map[key] = str(f)
    if not subject_map:
        raise RuntimeError(f"No valid subject .npz files found in {data_dir}")
    return subject_map


def _filter_subjects(subject_map: Dict[str, str], subjects: Optional[Sequence[str]]) -> Dict[str, str]:
    if not subjects:
        return subject_map
    keep = {_normalize_subject_key(s) for s in subjects}
    out = {k: v for k, v in subject_map.items() if k in keep}
    if len(out) < 2:
        raise RuntimeError("Need at least 2 subjects for LOSO ablation.")
    return out


def _sorted_score_rows(score_map: Dict[str, float]) -> List[Tuple[str, float]]:
    rows = sorted(score_map.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return [(str(k), float(v)) for k, v in rows]


def _topk_subjects(score_map: Dict[str, float], k: int) -> List[str]:
    rows = _sorted_score_rows(score_map)
    k_eff = max(1, min(int(k), len(rows)))
    return [s for s, _ in rows[:k_eff]]


def _spearman_from_maps(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < 2:
        return float("nan")
    va = np.asarray([float(a[s]) for s in common], dtype=np.float64)
    vb = np.asarray([float(b[s]) for s in common], dtype=np.float64)
    ra = pd.Series(va).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(vb).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return float("nan")
    return float(len(sa & sb) / len(union))


def _compute_fold_scores(
    *,
    target_subject: str,
    subject_files: Dict[str, str],
    positive_label: int,
    background_label: int,
    score_mode: str,
    max_trials_per_class: Optional[int],
    target_bg_mode: str,
    target_bg_ratio: float,
    target_bg_channel_indices: Optional[Sequence[int]],
    random_seed: int,
):
    source_files = {k: v for k, v in subject_files.items() if k != target_subject}
    return compute_rpcs_source_scores(
        target_subject_file=subject_files[target_subject],
        source_subject_files=source_files,
        positive_label=int(positive_label),
        background_label=int(background_label),
        max_trials_per_class=max_trials_per_class,
        seed=int(random_seed),
        score_mode=str(score_mode),
        target_bg_mode=str(target_bg_mode),
        target_bg_ratio=float(target_bg_ratio),
        target_bg_channel_indices=target_bg_channel_indices,
        return_details=True,
    )


def run_target_filter_ablation(
    *,
    subject_files: Dict[str, str],
    output_dir: Path,
    positive_label: int,
    background_label: int,
    top_k: int,
    random_seed: int,
    target_bg_ratio: float,
    midline_channels: Optional[Sequence[int]],
) -> None:
    modes = ["all", "amplitude", "kurtosis", "amplitude_restricted"]
    rows_long: List[Dict[str, object]] = []
    pair_records: List[Dict[str, object]] = []

    for target_subject in sorted(subject_files.keys(), key=lambda s: int(s[3:])):
        fold_scores: Dict[str, Dict[str, float]] = {}
        fold_topk: Dict[str, List[str]] = {}

        for mode in modes:
            score_map, details = _compute_fold_scores(
                target_subject=target_subject,
                subject_files=subject_files,
                positive_label=positive_label,
                background_label=background_label,
                score_mode="rpcs",
                max_trials_per_class=128,
                target_bg_mode=mode,
                target_bg_ratio=target_bg_ratio,
                target_bg_channel_indices=(midline_channels if mode == "amplitude_restricted" else None),
                random_seed=random_seed,
            )

            ranking_all = list(details.get("ranking_all", []))
            if ranking_all:
                ranking_all = sorted(ranking_all, key=lambda r: (-float(r.get("score", np.nan)), str(r.get("subject"))))
                score_map_all = {str(r.get("subject")): float(r.get("score")) for r in ranking_all}
            else:
                score_map_all = dict(score_map)

            fold_scores[mode] = score_map_all
            fold_topk[mode] = _topk_subjects(score_map_all, top_k)

            rank_rows = _sorted_score_rows(score_map_all)
            topk_set = set(fold_topk[mode])
            for rank_idx, (source_subject, score_val) in enumerate(rank_rows, start=1):
                rows_long.append(
                    {
                        "fold": target_subject,
                        "mode": mode,
                        "source_subject": source_subject,
                        "rpcs_score": float(score_val),
                        "rank": int(rank_idx),
                        "in_topk": bool(source_subject in topk_set),
                    }
                )

        for mode_a, mode_b in itertools.combinations(modes, 2):
            rho = _spearman_from_maps(fold_scores[mode_a], fold_scores[mode_b])
            jac = _jaccard(fold_topk[mode_a], fold_topk[mode_b])
            pair_records.append(
                {
                    "fold": target_subject,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "spearman_rho": float(rho),
                    "jaccard_topk": float(jac),
                }
            )

    df_long = pd.DataFrame(rows_long)
    out_long = output_dir / "target_filter_ablation.csv"
    df_long.to_csv(out_long, index=False, encoding="utf-8-sig")

    df_pair = pd.DataFrame(pair_records)
    if not df_pair.empty:
        df_summary = (
            df_pair.groupby(["mode_a", "mode_b"], dropna=False)
            .agg(
                spearman_rho_mean=("spearman_rho", "mean"),
                spearman_rho_std=("spearman_rho", "std"),
                jaccard_topk_mean=("jaccard_topk", "mean"),
                jaccard_topk_std=("jaccard_topk", "std"),
            )
            .reset_index()
        )
    else:
        df_summary = pd.DataFrame(
            columns=[
                "mode_a",
                "mode_b",
                "spearman_rho_mean",
                "spearman_rho_std",
                "jaccard_topk_mean",
                "jaccard_topk_std",
            ]
        )
    out_summary = output_dir / "target_filter_summary.csv"
    df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")


def run_max_trials_ablation(
    *,
    subject_files: Dict[str, str],
    output_dir: Path,
    positive_label: int,
    background_label: int,
    top_k: int,
    random_seed: int,
    caps: Sequence[int],
) -> None:
    caps = [int(c) for c in caps]
    if 0 not in caps:
        caps = list(caps) + [0]
    caps = list(dict.fromkeys(caps))

    rows_long: List[Dict[str, object]] = []
    records_summary: List[Dict[str, object]] = []

    for target_subject in sorted(subject_files.keys(), key=lambda s: int(s[3:])):
        fold_scores: Dict[int, Dict[str, float]] = {}
        fold_topk: Dict[int, List[str]] = {}
        fold_time: Dict[int, float] = {}

        for cap in caps:
            t0 = time.perf_counter()
            score_map, details = _compute_fold_scores(
                target_subject=target_subject,
                subject_files=subject_files,
                positive_label=positive_label,
                background_label=background_label,
                score_mode="rpcs",
                max_trials_per_class=int(cap),
                target_bg_mode="amplitude",
                target_bg_ratio=0.7,
                target_bg_channel_indices=None,
                random_seed=random_seed,
            )
            elapsed = float(time.perf_counter() - t0)
            fold_time[cap] = elapsed

            ranking_all = list(details.get("ranking_all", []))
            if ranking_all:
                ranking_all = sorted(ranking_all, key=lambda r: (-float(r.get("score", np.nan)), str(r.get("subject"))))
                score_map_all = {str(r.get("subject")): float(r.get("score")) for r in ranking_all}
            else:
                score_map_all = dict(score_map)

            fold_scores[cap] = score_map_all
            fold_topk[cap] = _topk_subjects(score_map_all, top_k)

            rank_rows = _sorted_score_rows(score_map_all)
            for rank_idx, (source_subject, score_val) in enumerate(rank_rows, start=1):
                rows_long.append(
                    {
                        "fold": target_subject,
                        "max_trials": int(cap),
                        "source_subject": source_subject,
                        "rpcs_score": float(score_val),
                        "rank": int(rank_idx),
                        "time_seconds": float(elapsed),
                    }
                )

        ref_cap = 0
        ref_scores = fold_scores[ref_cap]
        ref_topk = fold_topk[ref_cap]
        for cap in caps:
            rho = _spearman_from_maps(fold_scores[cap], ref_scores)
            jac = _jaccard(fold_topk[cap], ref_topk)
            records_summary.append(
                {
                    "fold": target_subject,
                    "max_trials": int(cap),
                    "spearman_vs_all": float(rho),
                    "jaccard_vs_all": float(jac),
                    "time_per_fold": float(fold_time[cap]),
                }
            )

    df_long = pd.DataFrame(rows_long)
    out_long = output_dir / "max_trials_ablation.csv"
    df_long.to_csv(out_long, index=False, encoding="utf-8-sig")

    df_fold = pd.DataFrame(records_summary)
    if not df_fold.empty:
        df_summary = (
            df_fold.groupby("max_trials", dropna=False)
            .agg(
                spearman_vs_all_mean=("spearman_vs_all", "mean"),
                spearman_vs_all_std=("spearman_vs_all", "std"),
                jaccard_vs_all_mean=("jaccard_vs_all", "mean"),
                jaccard_vs_all_std=("jaccard_vs_all", "std"),
                time_per_fold_mean=("time_per_fold", "mean"),
                time_per_fold_std=("time_per_fold", "std"),
            )
            .reset_index()
        )
    else:
        df_summary = pd.DataFrame(
            columns=[
                "max_trials",
                "spearman_vs_all_mean",
                "spearman_vs_all_std",
                "jaccard_vs_all_mean",
                "jaccard_vs_all_std",
                "time_per_fold_mean",
                "time_per_fold_std",
            ]
        )
    out_summary = output_dir / "max_trials_summary.csv"
    df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")


def _parse_caps(text: str) -> List[int]:
    if not text:
        return [64, 128, 256, 512, 0]
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out or [64, 128, 256, 512, 0]


def parse_args():
    parser = argparse.ArgumentParser(description="R-PCS Ablation Runner")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing subject .npz files")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject ids, e.g. sub1 sub2 ...")
    parser.add_argument("--positive_label", type=int, default=1)
    parser.add_argument("--background_label", type=int, default=0)
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="both",
        choices=["target_filter", "max_trials", "both"],
    )
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--midline_channels", nargs="*", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--target_bg_ratio", type=float, default=0.7)
    parser.add_argument(
        "--max_trials_caps",
        type=str,
        default="64,128,256,512,0",
        help="Comma-separated caps. 0 means no cap.",
    )
    parser.add_argument("--random_seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_files = _discover_subject_files(data_dir)
    subject_files = _filter_subjects(subject_files, args.subjects)

    config_dump = {
        "data_dir": str(data_dir),
        "subjects": sorted(subject_files.keys(), key=lambda s: int(s[3:])),
        "positive_label": int(args.positive_label),
        "background_label": int(args.background_label),
        "ablation_type": args.ablation_type,
        "output_dir": str(output_dir),
        "midline_channels": [] if args.midline_channels is None else [int(i) for i in args.midline_channels],
        "top_k": int(args.top_k),
        "target_bg_ratio": float(args.target_bg_ratio),
        "max_trials_caps": _parse_caps(args.max_trials_caps),
        "random_seed": int(args.random_seed),
    }
    with open(output_dir / "ablation_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2, ensure_ascii=False)

    if args.ablation_type in ("target_filter", "both"):
        run_target_filter_ablation(
            subject_files=subject_files,
            output_dir=output_dir,
            positive_label=args.positive_label,
            background_label=args.background_label,
            top_k=args.top_k,
            random_seed=args.random_seed,
            target_bg_ratio=args.target_bg_ratio,
            midline_channels=args.midline_channels,
        )

    if args.ablation_type in ("max_trials", "both"):
        run_max_trials_ablation(
            subject_files=subject_files,
            output_dir=output_dir,
            positive_label=args.positive_label,
            background_label=args.background_label,
            top_k=args.top_k,
            random_seed=args.random_seed,
            caps=_parse_caps(args.max_trials_caps),
        )

    print(f"Ablation finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

