import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List

from main import main
from Utils.config import build_config, load_config


SUMMARY_FIELDS = [
    "trial_id",
    "description",
    "exp_tag",
    "result_path",
    "source_selection",
    "subject_batch_size",
    "subjects_per_batch",
    "effective_rpcs_top_k",
    "random_seed",
    "rpcs_score_mode",
    "rpcs_tau_d",
    "rpcs_tau_s",
    "rpcs_tau_d_percentile",
    "rpcs_tau_s_percentile",
    "rpcs_target_bg_mode",
    "rpcs_target_bg_ratio",
    "rpcs_max_trials_per_class",
    "rpcs_cov_estimator",
    "rpcs_use_correlation",
    "AUC",
    "BA",
    "F1",
    "TPR",
    "FPR",
]


def _apply_overrides(config: dict, overrides: dict) -> dict:
    merged = deepcopy(config)
    for key, value in (overrides or {}).items():
        merged[key] = value
    return merged


def _none_like(value: Any) -> bool:
    return value in (None, "", "None", "none", "null", "NULL")


def _compute_exp_dir(cfg: Dict[str, Any]) -> Path:
    root = Path("Experiments") / cfg["model"] / cfg["dataset"] / cfg["train_mode"]
    exp_tag = str(cfg.get("exp_tag", "") or "").strip()
    if exp_tag:
        root = root / exp_tag
    return root


def _read_subject_metrics(results_csv: Path, subject_id: int) -> Dict[str, str]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results file: {results_csv}")
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("SUB") == f"SUB{subject_id}":
                return row
    raise ValueError(f"Subject {subject_id} not found in {results_csv}")



def _write_summary(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})



def run(config_path: str):
    cfg_yaml = load_config(config_path)
    model_name = str(cfg_yaml.get("model", "")).strip()
    dataset_name = str(cfg_yaml.get("dataset", "")).strip() or None
    if not model_name:
        raise ValueError(f"No model configured in {config_path}")

    subject_id = int(cfg_yaml.get("target_subject_id", 2))
    base_overrides = dict(cfg_yaml.get("base_overrides", {}))
    trials = list(cfg_yaml.get("trials", []))
    if not trials:
        raise ValueError(f"No trials configured in {config_path}")

    summary_csv = cfg_yaml.get("summary_csv", None)
    summary_path = Path(summary_csv) if summary_csv else (
        Path("Experiments") / model_name / (dataset_name or "") / "cross-subject" / f"rpcs_tuning_sub{subject_id}_summary.csv"
    )

    summary_rows: List[Dict[str, Any]] = []

    for idx, trial in enumerate(trials, start=1):
        trial_id = str(trial.get("id", f"trial_{idx:02d}"))
        description = str(trial.get("description", "")).strip()
        trial_overrides = dict(trial.get("overrides", {}))

        cfg = build_config(model_name, dataset_name)
        cfg = _apply_overrides(cfg, base_overrides)
        cfg = _apply_overrides(cfg, trial_overrides)
        cfg.setdefault("held_out_start_id", subject_id)
        cfg.setdefault("held_out_end_id", subject_id)
        cfg.setdefault("source_selection", "R-PCS")
        cfg.setdefault("exp_tag", f"rpcs_tuning_sub{subject_id}/{trial_id}")

        main(cfg)

        exp_dir = _compute_exp_dir(cfg)
        result_path = exp_dir / "results.csv"
        row = _read_subject_metrics(result_path, subject_id)
        effective_top_k = cfg.get("rpcs_top_k", None)
        if _none_like(effective_top_k):
            effective_top_k = cfg.get("subjects_per_batch", None)

        summary_rows.append(
            {
                "trial_id": trial_id,
                "description": description,
                "exp_tag": cfg.get("exp_tag", ""),
                "result_path": str(result_path),
                "source_selection": cfg.get("source_selection", ""),
                "subject_batch_size": cfg.get("subject_batch_size", ""),
                "subjects_per_batch": cfg.get("subjects_per_batch", ""),
                "effective_rpcs_top_k": effective_top_k,
                "random_seed": cfg.get("random_seed", ""),
                "rpcs_score_mode": cfg.get("rpcs_score_mode", ""),
                "rpcs_tau_d": cfg.get("rpcs_tau_d", ""),
                "rpcs_tau_s": cfg.get("rpcs_tau_s", ""),
                "rpcs_tau_d_percentile": cfg.get("rpcs_tau_d_percentile", ""),
                "rpcs_tau_s_percentile": cfg.get("rpcs_tau_s_percentile", ""),
                "rpcs_target_bg_mode": cfg.get("rpcs_target_bg_mode", ""),
                "rpcs_target_bg_ratio": cfg.get("rpcs_target_bg_ratio", ""),
                "rpcs_max_trials_per_class": cfg.get("rpcs_max_trials_per_class", ""),
                "rpcs_cov_estimator": cfg.get("rpcs_cov_estimator", ""),
                "rpcs_use_correlation": cfg.get("rpcs_use_correlation", ""),
                "AUC": row.get("AUC", ""),
                "BA": row.get("BA", ""),
                "F1": row.get("F1", ""),
                "TPR": row.get("TPR", ""),
                "FPR": row.get("FPR", ""),
            }
        )
        _write_summary(summary_path, summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Configs/eeginception_rpcs_tuning_sub2.yaml",
        help="Path to an R-PCS tuning YAML file.",
    )
    args = parser.parse_args()
    run(args.config)
