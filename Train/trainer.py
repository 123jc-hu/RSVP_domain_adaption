from shutil import rmtree
import os
import re
import gc
import time as _t
import logging
from contextlib import contextmanager
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from Data.datamodule import EEGDataModuleCrossSubject, SubjectBatchDataLoader, DualStreamDataLoader
from Data.path_resolver import resolve_dataset_dir
from Data.pccs import build_source_prototypes, build_target_prototype, compute_pccs_source_scores
from Data.rpt_aug import RPTAugmentor
from Train.hyr_dpa_framework import HyRDPAScaffold
from Train.iahm import IAHMLoss, EuclideanIAHMLoss, euclidean_to_lorentz
from Utils.config import set_random_seed
from Utils.utils import load_from_checkpoint, EarlyStopping, SaveBestValBA
from Utils.metrics import calculate_metrics, cal_F1_score
from Models.model_registry import model_dict



class MemoryManager:
    """Centralized memory management for CUDA operations"""
    
    @staticmethod
    @contextmanager
    def cuda_memory_context():
        """Context manager for automatic GPU memory cleanup"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    @staticmethod
    def cleanup_tensors(*tensors):
        for t in tensors:
            if isinstance(t, torch.Tensor):
                del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OptimizedExperimentRunner:
    """Orchestrates LOSO training/evaluation across all subjects."""

    def __init__(self, config: Dict[str, Any], main_logger: logging.Logger):
        self.config = config
        self.log = main_logger
        self.minimal_log = bool(config.get("minimal_log", True))
        use_gpu = bool(config.get("use_gpu", True))
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.exp_dir = self._setup_exp_dir()
        self.datamodule = EEGDataModuleCrossSubject(self.config)
        self.hyr_dpa = HyRDPAScaffold(self.config, self.log)

        self._init_state_cache: Dict[tuple, Dict[str, torch.Tensor]] = {}
        self.runtime_records: List[Dict[str, Any]] = []
        self.source_selection_records: List[Dict[str, Any]] = []
        self.rpcs_ranking_records: List[Dict[str, Any]] = []
        self.rpcs_fold_records: List[Dict[str, Any]] = []

    def _log_detail(self, msg: str):
        if not self.minimal_log:
            self.log.info(msg)

    def _setup_exp_dir(self) -> Path:
        """Create `Experiments/<model>/<dataset>/<train_mode>[/<exp_tag>]` directory."""
        root = Path("Experiments") / self.config["model"] / self.config["dataset"] / self.config["train_mode"]
        exp_tag = str(self.config.get("exp_tag", "") or "").strip()
        if exp_tag:
            root = root / exp_tag
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _model_signature(self) -> tuple:
        """Return a hashable model signature used by init-state cache."""
        cfg = self.config
        return (
            cfg["model"],
            int(cfg.get("n_channels", -1)),
            int(cfg.get("n_class", -1)),
            int(cfg.get("fs", -1)),
        )

    def _build_backbone_model(self) -> nn.Module:
        """Build plain backbone model from `Models/model_registry.py` registry."""
        name = self.config["model"]
        registry = model_dict()
        if name not in registry:
            raise KeyError(f"Unsupported model: {name}. Available: {sorted(registry.keys())}")
        model_cls = registry[name].Model
        return model_cls(self.config)

    def _get_init_state(self) -> Dict[str, torch.Tensor]:
        """Build once and cache the initial state_dict on CPU."""
        sig = self._model_signature()
        if sig not in self._init_state_cache:
            model = self._build_backbone_model()
            self._init_state_cache[sig] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            del model
        return self._init_state_cache[sig]

    def _new_model(self) -> nn.Module:
        """Create a fresh model initialized from cached initial state."""
        model = self._build_backbone_model()
        init_state = self._get_init_state()
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return model.to(self.device)

    def _configure_training_components(self, model: nn.Module):
        """
        Build optimizer/scheduler/early-stop components for plain backbone training.
        """
        cfg = self.config

        lr = float(cfg["learning_rate"])
        eta_min_factor = float(cfg.get("cosine_eta_min_factor", 0.01))
        eta_min = float(cfg.get("cosine_eta_min", lr * eta_min_factor))
        t_max = int(cfg.get("cosine_t_max", cfg.get("epochs", 400)))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=eta_min,
        )

        stage_tag = cfg.get("train_mode", "train")
        early = EarlyStopping(
            patience=int(cfg.get("patience", 10)),
            verbose=False,
            delta=float(cfg.get("es_delta", 0.0)),
            path=f"best_model-{stage_tag}.pth",
        )
        save_ba = SaveBestValBA(
            verbose=False,
            delta=float(cfg.get("ba_delta", 0.0)),
            path=f"best_ba_model-{stage_tag}.pth",
        )
        return optimizer, scheduler, early, save_ba

    def run_experiment(self):
        """Run LOSO over all subjects in the selected dataset."""
        set_random_seed(
            int(self.config.get("random_seed", 2026)),
            deterministic=bool(self.config.get("deterministic_run", True)),
            benchmark=bool(self.config.get("cudnn_benchmark", False)),
            matmul_precision=str(self.config.get("matmul_precision", "highest")),
        )
        self.log.info(f"Experiment summary | {self._experiment_summary()}")

        self.runtime_records = []
        self.source_selection_records = []
        self.rpcs_ranking_records = []
        self.rpcs_fold_records = []

        dataset_dir = self._dataset_dir()

        def _subject_sort_key(filename: str):
            m = re.search(r"sub(\d+)", filename)
            return int(m.group(1)) if m else float("inf")

        subject_files = sorted(
            [f for f in os.listdir(dataset_dir) if f.endswith(".npz") and "_10band" not in f],
            key=_subject_sort_key,
        )
        if not subject_files:
            raise RuntimeError(f"No subject .npz files found in dataset directory: {dataset_dir}")

        fold_files = list(subject_files)
        start_id_raw = self.config.get("held_out_start_id", None)
        end_id_raw = self.config.get("held_out_end_id", None)
        start_id = None if start_id_raw in (None, "", "None", "none", "null", "NULL") else int(start_id_raw)
        end_id = None if end_id_raw in (None, "", "None", "none", "null", "NULL") else int(end_id_raw)
        if start_id is not None or end_id is not None:
            filtered = []
            for filename in fold_files:
                m = re.search(r"sub(\d+)", filename)
                if not m:
                    continue
                sid = int(m.group(1))
                if start_id is not None and sid < start_id:
                    continue
                if end_id is not None and sid > end_id:
                    continue
                filtered.append(filename)
            fold_files = filtered

        n_fold = self.config.get("n_fold", None)
        if n_fold not in (None, "", "None", "none", "null", "NULL"):
            n_fold = int(n_fold)
            if n_fold > 0:
                fold_files = fold_files[: min(n_fold, len(fold_files))]
                self._log_detail(
                    f"Quick-fold mode enabled | n_fold={n_fold} | running {len(fold_files)}/{len(subject_files)} held-out subjects"
                )

        subject_file_map = {}
        for filename in subject_files:
            m = re.search(r"sub(\d+)", filename)
            if not m:
                continue
            sid = int(m.group(1))
            subject_file_map[f"sub{sid}"] = str(dataset_dir / filename)

        self._get_init_state()

        metric_history = {k: [] for k in ["AUC", "BA", "F1", "TPR", "FPR"]}
        completed_subject_ids: List[int] = []
        resume_metrics = self._load_latest_subject_metrics_from_log()

        with MemoryManager.cuda_memory_context():
            for filename in fold_files:
                set_random_seed(
                    int(self.config.get("random_seed", 2026)),
                    deterministic=bool(self.config.get("deterministic_run", True)),
                    benchmark=bool(self.config.get("cudnn_benchmark", False)),
                    matmul_precision=str(self.config.get("matmul_precision", "highest")),
                )

                m = re.search(r"sub(\d+)", filename)
                if not m:
                    continue
                subject_id = int(m.group(1))
                if subject_id in resume_metrics:
                    metrics = resume_metrics[subject_id]
                    self._log_detail(f"Resume mode | skip finished subject {subject_id} from latest log block")
                    for k in metric_history:
                        metric_history[k].append(metrics[k])
                    completed_subject_ids.append(subject_id)
                    continue
                source_scores, manual_sources, forced_mode = self._fold_source_selection_inputs(
                    test_subject_id=subject_id,
                    subject_file_map=subject_file_map,
                )

                self.datamodule.prepare_data(
                    test_subject_id=subject_id,
                    subject_file_map=subject_file_map,
                    source_scores=source_scores,
                    manual_source_subjects=manual_sources,
                    forced_source_selection_mode=forced_mode,
                )

                metrics = self._run_subject(subject_id)
                self._log_subject(subject_id, metrics)
                for k in metric_history:
                    metric_history[k].append(metrics[k])
                completed_subject_ids.append(subject_id)

        self._save_results(metric_history, subject_ids=completed_subject_ids)
        self._save_source_selection()
        self._save_rpcs_artifacts()

        if self.runtime_records:
            import pandas as _pd
            rt_df = _pd.DataFrame(self.runtime_records)
            cols = [
                "model_p50_ms",
                "model_p95_ms",
                "model_jitter_ms",
                "e2e_p50_ms",
                "e2e_p95_ms",
                "e2e_jitter_ms",
                "throughput_samples_per_s",
                "peak_cuda_mem_mb",
                "deadline_miss_rate",
            ]
            cols = [c for c in cols if c in rt_df.columns]
            mean_row = rt_df[cols].mean(numeric_only=True).to_dict()
            std_row = rt_df[cols].std(numeric_only=True).to_dict()
            summary = {"subject": "AVG"}
            summary.update({k: float(mean_row[k]) for k in mean_row})
            summary_std = {"subject": "STD"}
            summary_std.update({k: float(std_row[k]) for k in std_row})
            rt_df_out = _pd.concat([rt_df, _pd.DataFrame([summary, summary_std])], ignore_index=True)
            out_csv = self.exp_dir / "runtime_dataset_summary.csv"
            rt_df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
            pretty = {k: f"{mean_row[k]:.3f}+/-{std_row[k]:.3f}" for k in cols if k in mean_row and k in std_row}
            self._log_detail(f"[RUNTIME-DATASET] {pretty}")

    def _experiment_summary(self) -> str:
        cfg = self.config
        parts = [
            f"model={cfg.get('model')}",
            f"dataset={cfg.get('dataset')}",
            f"exp_tag={cfg.get('exp_tag', 'default')}",
            f"held_out=sub{cfg.get('held_out_start_id', 'all')}~sub{cfg.get('held_out_end_id', 'all')}",
            f"seed={cfg.get('random_seed', 2026)}",
            f"source_selection={cfg.get('source_selection', 'All')}",
            f"batch_policy={cfg.get('all_source_batch_policy', 'all')}",
            f"subjects_per_batch={cfg.get('subjects_per_batch', 'all')}",
            f"ea_enable={bool(cfg.get('ea_enable', False))}",
            f"use_target_stream={bool(cfg.get('use_target_stream', False))}",
            f"alignment_loss={cfg.get('alignment_loss', 'none')}",
            f"lambda_align={float(cfg.get('lambda_align', 0.0))}",
            f"bgds4={bool(cfg.get('source_train_bg_downsample_enable', False))}",
        ]

        if bool(cfg.get("source_train_bg_downsample_enable", False)):
            parts.append(
                f"bg_to_pos_ratio={float(cfg.get('source_train_bg_downsample_bg_to_pos_ratio', 0.0))}"
            )

        if str(cfg.get("model", "")).strip() == "EEGNetDSBN":
            parts.extend(
                [
                    "dsbn_layer=block2_bn",
                    "dsbn_affine=shared",
                    "dsbn_target_stats=enabled",
                ]
            )

        if str(cfg.get("model", "")).strip() == "EEGNetLDSA":
            parts.extend(
                [
                    "ldsa_layer=block2_bn",
                    "ldsa_affine=shared",
                    "ldsa_target_stats=blended",
                    f"ldsa_target_blend_alpha={float(cfg.get('ldsa_target_blend_alpha', 0.7))}",
                ]
            )

        if str(cfg.get("model", "")).strip() == "EEGNetSWLDSA":
            parts.extend(
                [
                    "swldsa_layer=block2_bn",
                    "swldsa_affine=shared",
                    "swldsa_target_stats=similarity_weighted_blended",
                    f"swldsa_target_blend_alpha={float(cfg.get('swldsa_target_blend_alpha', 0.1))}",
                    f"swldsa_similarity_tau={float(cfg.get('swldsa_similarity_tau', 1.0))}",
                    f"swldsa_var_distance_weight={float(cfg.get('swldsa_var_distance_weight', 1.0))}",
                ]
            )
        return ", ".join(str(p) for p in parts)

    def _fold_source_selection_inputs(
        self,
        test_subject_id: int,
        subject_file_map: Dict[str, str],
    ) -> Tuple[Optional[Dict[str, float]], Optional[List[str]], Optional[str]]:
        """
        Hook for fold-specific source selection inputs.
        Return:
        - source_scores: {'sub2': score, ...}
        - manual_source_subjects: ['sub2', 'sub5', ...]
        - forced_mode: all | random_k | scores | manual
        """
        return self.hyr_dpa.build_source_selection_inputs(
            test_subject_id=test_subject_id,
            subject_file_map=subject_file_map,
        )

    def _cfg_pref(self, pref_key: str, legacy_key: str, default):
        if pref_key in self.config and self.config.get(pref_key) is not None:
            return self.config.get(pref_key)
        return self.config.get(legacy_key, default)

    def _build_rpt_augmentor_for_fold(self, subject_id: int) -> Optional[RPTAugmentor]:
        rpt_cfg = self.hyr_dpa.build_rpt_aug_config()
        if not bool(rpt_cfg.get("enable", False)):
            return None

        meta = dict(getattr(self.hyr_dpa, "last_source_selection_meta", {}) or {})
        selected_subjects = list(
            (self.datamodule.source_selection_info or {}).get("selected_subjects", [])
            or self.datamodule.source_subject_keys
        )
        if not selected_subjects:
            self.log.warning(f"RPT-Aug enabled but selected source list is empty for sub{subject_id}.")
            return None

        fold_stats = self._build_rpt_fold_stats(
            subject_id=subject_id,
            selected_subjects=selected_subjects,
            rpt_cfg=rpt_cfg,
            meta=meta,
        )
        if fold_stats is None:
            self.log.warning(f"RPT-Aug enabled but fold statistics could not be built for sub{subject_id}.")
            return None
        prototypes = dict(fold_stats.get("prototypes", {}) or {})
        score_map = dict(fold_stats.get("scores", {}) or {})
        ranking = list(fold_stats.get("ranking", []) or [])
        selected_subjects = list(fold_stats.get("selected_subjects", []) or selected_subjects)

        pos_label = int(self._cfg_pref("rpcs_positive_label", "pccs_positive_label", 1))
        bg_label = int(self._cfg_pref("rpcs_background_label", "pccs_background_label", 0))
        max_src_pos = self._cfg_pref("rpcs_max_trials_per_class", "pccs_max_trials_per_class", None)
        max_src_pos = None if max_src_pos in (None, "", "None", "none", "null", "NULL") else int(max_src_pos)
        min_trials = int(self._cfg_pref("rpcs_min_trials_per_class", "pccs_min_trials_per_class", 4))

        source_p300_trials = self.datamodule.get_source_positive_trials(
            positive_label=pos_label,
            subject_keys=selected_subjects,
            max_trials_per_subject=max_src_pos,
            seed=int(self.config.get("random_seed", 2026)),
        )
        if not source_p300_trials:
            self.log.warning(f"RPT-Aug enabled but no source positive trials available for sub{subject_id}.")
            return None

        target_bg_trials = self.datamodule.get_target_background_trials(
            background_label=bg_label,
            target_use_all_trials=bool(
                self._cfg_pref("rpcs_target_use_all_trials", "pccs_target_use_all_trials", True)
            ),
            target_max_trials=self._cfg_pref("rpcs_target_max_trials", "pccs_target_max_trials", None),
            target_bg_mode=str(self._cfg_pref("rpcs_target_bg_mode", "pccs_target_bg_mode", "amplitude")),
            target_bg_ratio=float(self._cfg_pref("rpcs_target_bg_ratio", "pccs_target_bg_ratio", 0.7)),
            target_bg_channel_indices=self._cfg_pref(
                "rpcs_target_bg_channel_indices",
                "pccs_target_bg_channel_indices",
                None,
            ),
            min_keep=max(1, min_trials),
            seed=int(self.config.get("random_seed", 2026)),
        )
        if int(target_bg_trials.shape[0]) <= 0:
            self.log.warning(f"RPT-Aug enabled but no target background trials available for sub{subject_id}.")
            return None

        rpcs_result = {
            "prototypes": prototypes,
            "ranking": ranking,
            "scores": score_map,
        }
        augmentor = RPTAugmentor(
            rpcs_result=rpcs_result,
            source_p300_trials=source_p300_trials,
            target_bg_trials=target_bg_trials,
            selected_subjects=selected_subjects,
            beta=float(rpt_cfg.get("beta", 1.0)),
            cov_eps=float(rpt_cfg.get("cov_eps", 1e-6)),
            use_correlation=bool(rpt_cfg.get("use_correlation", True)),
            correlation_eps=float(rpt_cfg.get("correlation_eps", 1e-12)),
            cov_estimator=str(rpt_cfg.get("cov_estimator", "sample")),
            cov_shrinkage=float(rpt_cfg.get("cov_shrinkage", 0.0)),
            input_layout=str(rpt_cfg.get("input_layout", "channel_first")),
            rpcs_weighted_sampling=bool(rpt_cfg.get("weighted_sampling", True)),
            clip_factor=float(rpt_cfg.get("clip_factor", 3.0)),
        )
        augmentor.prepare()
        self._log_detail(
            f"RPT-Aug ready for sub{subject_id} | selected_sources={len(selected_subjects)} "
            f"| source_pos_trials={sum(int(v.shape[0]) for v in source_p300_trials.values())} "
            f"| target_bg_trials={int(target_bg_trials.shape[0])}"
        )
        return augmentor

    def _build_rpt_fold_stats(
        self,
        *,
        subject_id: int,
        selected_subjects: List[str],
        rpt_cfg: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        details = dict(meta.get("details", {}) or {})
        prototypes = dict(details.get("prototypes", {}) or {})
        if prototypes:
            score_map, ranking = self._resolve_rpt_score_map(
                selected_subjects=selected_subjects,
                meta=meta,
                rpt_cfg=rpt_cfg,
            )
            return {
                "prototypes": prototypes,
                "scores": score_map,
                "ranking": ranking,
                "selected_subjects": list(selected_subjects),
            }

        if self.datamodule.subject_file_map is None:
            return None

        metric = str(self._cfg_pref("rpcs_mean_metric", "pccs_mean_metric", "riemann")).strip().lower()
        pos_label = int(self._cfg_pref("rpcs_positive_label", "pccs_positive_label", 1))
        bg_label = int(self._cfg_pref("rpcs_background_label", "pccs_background_label", 0))
        max_trials = self._cfg_pref("rpcs_max_trials_per_class", "pccs_max_trials_per_class", 128)
        min_trials = int(self._cfg_pref("rpcs_min_trials_per_class", "pccs_min_trials_per_class", 4))
        seed = int(self.config.get("random_seed", 2026))
        cov_eps = float(rpt_cfg.get("cov_eps", 1e-6))
        cov_estimator = str(rpt_cfg.get("cov_estimator", "sample"))
        cov_shrinkage = float(rpt_cfg.get("cov_shrinkage", 0.0))
        use_correlation = bool(rpt_cfg.get("use_correlation", True))
        correlation_eps = float(rpt_cfg.get("correlation_eps", 1e-12))
        input_layout = str(rpt_cfg.get("input_layout", "channel_first"))
        mean_max_iter = int(self._cfg_pref("rpcs_mean_max_iter", "pccs_mean_max_iter", 20))
        mean_tol = float(self._cfg_pref("rpcs_mean_tol", "pccs_mean_tol", 1e-6))

        source_pos_map: Dict[str, np.ndarray] = {}
        source_bg_map: Dict[str, np.ndarray] = {}
        valid_subjects: List[str] = []
        for sub in selected_subjects:
            subject_file = self.datamodule.subject_file_map.get(str(sub))
            if not subject_file:
                continue
            src = build_source_prototypes(
                subject_key=str(sub),
                subject_file=str(subject_file),
                positive_label=pos_label,
                background_label=bg_label,
                max_trials_per_class=max_trials,
                min_trials_per_class=min_trials,
                seed=seed,
                cov_eps=cov_eps,
                cov_shrinkage=cov_shrinkage,
                cov_estimator=cov_estimator,
                use_correlation=use_correlation,
                correlation_eps=correlation_eps,
                input_layout=input_layout,
                metric=metric,
                mean_max_iter=mean_max_iter,
                mean_tol=mean_tol,
            )
            pos_proto = src.get("p300_proto", None)
            bg_proto = src.get("bg_proto", None)
            if pos_proto is None or bg_proto is None:
                continue
            source_pos_map[str(sub)] = np.asarray(pos_proto, dtype=np.float64)
            source_bg_map[str(sub)] = np.asarray(bg_proto, dtype=np.float64)
            valid_subjects.append(str(sub))

        if not valid_subjects:
            return None

        target_file = self.datamodule.subject_file_map.get(f"sub{int(subject_id)}")
        if not target_file:
            return None
        target_info = build_target_prototype(
            target_subject_file=str(target_file),
            background_label=bg_label,
            min_trials_per_class=min_trials,
            seed=seed,
            cov_eps=cov_eps,
            cov_shrinkage=cov_shrinkage,
            cov_estimator=cov_estimator,
            use_correlation=use_correlation,
            correlation_eps=correlation_eps,
            input_layout=input_layout,
            metric=metric,
            mean_max_iter=mean_max_iter,
            mean_tol=mean_tol,
            target_use_all_trials=bool(
                self._cfg_pref("rpcs_target_use_all_trials", "pccs_target_use_all_trials", True)
            ),
            target_max_trials=self._cfg_pref("rpcs_target_max_trials", "pccs_target_max_trials", None),
            target_bg_mode=str(self._cfg_pref("rpcs_target_bg_mode", "pccs_target_bg_mode", "amplitude")),
            target_bg_ratio=float(self._cfg_pref("rpcs_target_bg_ratio", "pccs_target_bg_ratio", 0.7)),
            target_bg_channel_indices=self._cfg_pref(
                "rpcs_target_bg_channel_indices",
                "pccs_target_bg_channel_indices",
                None,
            ),
        )
        target_bg = target_info.get("prototype", None)
        if target_bg is None:
            return None

        score_map, ranking = self._resolve_rpt_score_map(
            selected_subjects=valid_subjects,
            meta=meta,
            rpt_cfg=rpt_cfg,
        )
        return {
            "prototypes": {
                "target_bg": np.asarray(target_bg, dtype=np.float64),
                "source_pos": source_pos_map,
                "source_bg": source_bg_map,
            },
            "scores": score_map,
            "ranking": ranking,
            "selected_subjects": valid_subjects,
        }

    def _resolve_rpt_score_map(
        self,
        *,
        selected_subjects: List[str],
        meta: Dict[str, Any],
        rpt_cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        mode = str(rpt_cfg.get("score_mode", "uniform")).strip().lower()
        subjects = [str(s) for s in selected_subjects]
        if mode == "uniform":
            score_map = {s: 1.0 for s in subjects}
            ranking = [{"subject": s, "score": 1.0} for s in subjects]
            return score_map, ranking

        ranking_raw = list(meta.get("ranking", []) or [])
        score_map: Dict[str, float] = {}
        ranking: List[Dict[str, Any]] = []
        for row in ranking_raw:
            sub = row.get("subject", None)
            if sub is None:
                continue
            sub_key = str(sub)
            if sub_key not in subjects:
                continue
            score = float(row.get("score", 1.0))
            if not np.isfinite(score):
                continue
            score_map[sub_key] = score
            ranking.append({"subject": sub_key, "score": score})
        if not score_map:
            details = dict(meta.get("details", {}) or {})
            raw_scores = details.get("scores", {}) or meta.get("scores", {}) or {}
            if isinstance(raw_scores, dict):
                for sub in subjects:
                    score = float(raw_scores.get(sub, 1.0))
                    if np.isfinite(score):
                        score_map[sub] = score
                        ranking.append({"subject": sub, "score": score})
        if len(score_map) != len(subjects):
            score_map = {s: float(score_map.get(s, 1.0)) for s in subjects}
            ranking = [{"subject": s, "score": score_map[s]} for s in subjects]
        ranking.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return score_map, ranking

    def _dataset_dir(self) -> Path:
        return resolve_dataset_dir(self.config)

    def _run_subject(self, subject_id: int) -> Dict[str, float]:
        """Train/evaluate one LOSO subject and return rounded metrics."""
        t0 = _t.perf_counter()
        model = self._new_model()
        metric_values = self._run_single_subject(model, subject_id)

        del model
        elapsed = _t.perf_counter() - t0
        self._log_detail(f"Subject {subject_id} completed in {elapsed:.2f}s")
        return {k: round(v, 4) for k, v in zip(["AUC", "BA", "F1", "TPR", "FPR"], metric_values)}

    def _run_single_subject(self, model: nn.Module, subject_id: int) -> Tuple[float, float, float, float, float]:
        """Single-stage train/eval for one LOSO target subject."""
        # Build split datasets first so class weights are fold-specific.
        self.datamodule.setup()
        self._record_source_selection(subject_id)
        self._configure_fold_target_anchor(model=model, subject_id=subject_id)

        class_weights = None
        if bool(self.config.get("class_weighted_ce", True)):
            class_weights = self.datamodule.source_class_weight_tensor(
                n_class=int(self.config.get("n_class", 2)),
                device=self.device,
            )
            self._log_detail(f"Subject {subject_id} source class weights: {class_weights.detach().cpu().tolist()}")

        rpt_augmentor = self._build_rpt_augmentor_for_fold(subject_id)
        iahm_cfg = self.hyr_dpa.build_iahm_config()
        source_class_counts = (
            self.datamodule.source_train_class_counts.tolist()
            if self.datamodule.source_train_class_counts is not None
            else None
        )

        optimizer, scheduler, early, save_ba = self._configure_training_components(model)

        trainer = OptimizedTrainer(
            config=self.config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            earlystopping=early,
            save_other_model=save_ba,
            device=self.device,
            logger=self.log,
            class_weights=class_weights,
        )
        trainer.configure_stage1_modules(
            rpt_augmentor=rpt_augmentor,
            iahm_cfg=iahm_cfg,
            source_class_counts=source_class_counts,
            positive_label=int(self._cfg_pref("rpcs_positive_label", "pccs_positive_label", 1)),
            background_label=int(self._cfg_pref("rpcs_background_label", "pccs_background_label", 0)),
        )

        ckpt_dir, log_dir = self._prepare_dirs(subject_id)
        if self.config.get("is_training", True):
            self._clean_subject_logs(log_dir, run_prefix="metrics")
        trainer.setup_logging(log_dir, ckpt_dir, stage=2, run_name=f"metrics_sub{subject_id}")

        if self.config.get("is_training", True):
            mode = str(self.config.get("training_mode", "End2End")).strip().lower()
            if mode == "decoupled":
                self._log_detail("Training mode: Decoupled | Stage 1 (feature alignment scaffold)")
                self.hyr_dpa.stage1_feature_alignment()
                self._train(trainer)
                self._log_detail("Training mode: Decoupled | Stage 2 (classifier rectification scaffold)")
                self.hyr_dpa.stage2_classifier_rectification()
            else:
                self._log_detail("Training mode: End2End")
                self._train(trainer)

        return self._eval(trainer, ckpt_dir)

    def _configure_fold_target_anchor(self, *, model: nn.Module, subject_id: int) -> None:
        setter = getattr(model, "set_target_anchor_domain_id", None)
        if not callable(setter):
            return

        anchor_subject = self._compute_discrim_anchor_subject(subject_id)
        anchor_id = None
        if isinstance(anchor_subject, str):
            m = re.fullmatch(r"sub(\d+)", anchor_subject.strip().lower())
            if m:
                anchor_id = int(m.group(1))
        setter(anchor_id)
        if anchor_id is not None:
            self._log_detail(
                f"Subject {subject_id} target anchor | mode=discriminability_top1 | anchor=sub{anchor_id}"
            )

    def _compute_discrim_anchor_subject(self, subject_id: int) -> Optional[str]:
        if self.datamodule.subject_file_map is None or self.datamodule.test_subject_id is None:
            return None

        held_out = f"sub{int(subject_id)}"
        target_file = self.datamodule.subject_file_map.get(held_out)
        if not target_file:
            return None

        source_keys = list(getattr(self.datamodule, "source_subject_keys", []) or [])
        if not source_keys:
            return None
        source_pool = {
            str(k): str(self.datamodule.subject_file_map[str(k)])
            for k in source_keys
            if str(k) in self.datamodule.subject_file_map
        }
        if not source_pool:
            return None

        pos_label = int(self._cfg_pref("rpcs_positive_label", "pccs_positive_label", 1))
        bg_label = int(self._cfg_pref("rpcs_background_label", "pccs_background_label", 0))
        max_trials = self._cfg_pref("rpcs_max_trials_per_class", "pccs_max_trials_per_class", 128)
        min_trials = int(self._cfg_pref("rpcs_min_trials_per_class", "pccs_min_trials_per_class", 4))
        seed = int(self.config.get("random_seed", 2026))
        mean_metric = str(self._cfg_pref("rpcs_mean_metric", "pccs_mean_metric", "riemann"))
        cov_eps = float(self._cfg_pref("rpcs_cov_eps", "pccs_cov_eps", 1e-6))
        cov_shrinkage = float(self._cfg_pref("rpcs_cov_shrinkage", "pccs_cov_shrinkage", 0.0))
        cov_estimator = str(self._cfg_pref("rpcs_cov_estimator", "pccs_cov_estimator", "sample"))
        use_correlation = bool(self._cfg_pref("rpcs_use_correlation", "pccs_use_correlation", True))
        correlation_eps = float(self._cfg_pref("rpcs_correlation_eps", "pccs_correlation_eps", 1e-12))
        input_layout = str(self._cfg_pref("rpcs_input_layout", "pccs_input_layout", "channel_first"))
        mean_max_iter = int(self._cfg_pref("rpcs_mean_max_iter", "pccs_mean_max_iter", 20))
        mean_tol = float(self._cfg_pref("rpcs_mean_tol", "pccs_mean_tol", 1e-6))
        score_eps = float(self._cfg_pref("rpcs_score_eps", "pccs_score_eps", 1e-6))

        _, details = compute_pccs_source_scores(
            target_subject_file=str(target_file),
            source_subject_files=source_pool,
            positive_label=pos_label,
            background_label=bg_label,
            max_trials_per_class=max_trials,
            min_trials_per_class=min_trials,
            seed=seed,
            cov_eps=cov_eps,
            cov_shrinkage=cov_shrinkage,
            cov_estimator=cov_estimator,
            use_correlation=use_correlation,
            correlation_eps=correlation_eps,
            input_layout=input_layout,
            mean_metric=mean_metric,
            distance_metric=mean_metric,
            mean_max_iter=mean_max_iter,
            mean_tol=mean_tol,
            score_mode="discrim_only",
            score_eps=score_eps,
            return_details=True,
            return_prototypes=False,
        )
        ranking = list((details or {}).get("ranking", []) or [])
        if not ranking:
            return None
        ranking.sort(key=lambda r: float(r.get("score", float("-inf"))), reverse=True)
        return str(ranking[0].get("subject"))

    def _train(self, trainer):
        """Build dataloaders and run fit()."""
        cfg = self.config
        train_unit_batch_size = int(cfg["subject_batch_size"])
        val_bs_raw = cfg.get("val_batch_size", None)
        if val_bs_raw in (None, "", "None", "none", "null", "NULL"):
            val_batch_size = int(train_unit_batch_size)
        else:
            val_batch_size = int(val_bs_raw)
        use_target_stream = bool(cfg.get("use_target_stream", False))
        need_target_stream = (
            bool(cfg.get("rpt_aug_enable", False))
            or float(cfg.get("lambda_align", 0.0)) > 0.0
            or float(cfg.get("lambda_class_align", 0.0)) > 0.0
            or float(cfg.get("lambda_ccl", 0.0)) > 0.0
            or float(cfg.get("lambda_prior", 0.0)) > 0.0
        )
        if need_target_stream and not use_target_stream:
            use_target_stream = True
            self._log_detail("Stage-1 alignment enabled -> force `use_target_stream=True` for training.")
        if use_target_stream:
            train_loader = self.datamodule.train_dataloader(target_batch_size=train_unit_batch_size)
        else:
            train_loader = self.datamodule.source_train_dataloader()
        val_loader = self.datamodule.val_dataloader(batch_size=val_batch_size)
        train_steps = len(train_loader) if hasattr(train_loader, "__len__") else None
        val_steps = len(val_loader) if hasattr(val_loader, "__len__") else None
        effective_train_batch_size = int(
            getattr(
                train_loader,
                "batch_size",
                int(cfg["subjects_per_batch"]) * int(cfg["subject_batch_size"]),
            )
        )
        self._log_detail(
            f"Training setup | use_target_stream={use_target_stream} | "
            f"train_samples={len(self.datamodule.train_dataset)} | "
            f"val_samples={len(self.datamodule.val_dataset)} | "
            f"train_steps={train_steps} | "
            f"val_steps={val_steps} | "
            f"effective_train_batch_size={effective_train_batch_size} | "
            f"val_batch_size={val_batch_size} | "
            f"subject_batch_policy={getattr(self.datamodule, 'train_subject_batch_policy', None)} | "
            f"subjects_per_batch={getattr(self.datamodule, 'train_subjects_per_batch', None)} | "
            f"subject_batch_size={getattr(self.datamodule, 'train_subject_batch_size', None)} | "
            f"step_ref_mode={getattr(self.datamodule, 'train_step_reference_mode', None)} | "
            f"step_ref_subject={getattr(self.datamodule, 'train_step_reference_subject', None)} | "
            f"step_ref_samples={getattr(self.datamodule, 'train_step_reference_samples', None)}"
        )
        trainer.configure_dynamic_feature_sampling(datamodule=self.datamodule)

        trainer.fit(train_loader, val_loader, stage=2, epochs=int(cfg["epochs"]))

        del self.datamodule.train_dataset, self.datamodule.val_dataset

        torch.save(trainer.model.state_dict(), trainer.checkpoint_dir / "last_model--2.pth")

    def _eval(self, trainer, ckpt_dir: Path) -> Tuple[float, float, float, float, float]:
        """Load best checkpoint and run evaluation."""
        best_path = ckpt_dir / "best_ba_model--2.pth"
        trainer.model.load_state_dict(load_from_checkpoint(Path.cwd() / best_path))

        test_loader = self.datamodule.test_dataloader(batch_size=int(self.config.get("test_batch_size", 1000)))
        trainer._eval_use_target_stats = True

        metrics = trainer._run_epoch(
            test_loader,
            training=False,
            loss_fn=trainer.loss_fn,
            profile_runtime=bool(self.config.get("log_runtime", True)),
        )
        trainer._eval_use_target_stats = False

        rt = getattr(trainer, "last_runtime_stats", None)
        try:
            subj = int(re.search(r"sub(\d+)", str(self.datamodule.test_path)).group(1)) \
                if hasattr(self.datamodule, "test_path") else None
        except Exception:
            subj = None

        if rt is not None:
            row = {"subject": subj}
            row.update(rt)
            self.runtime_records.append(row)

            self._log_detail("[RUNTIME] " + " | ".join(
                f"{k}: {rt[k]:.3f}" if isinstance(rt[k], (int, float)) else f"{k}: {rt[k]}"
                for k in [
                    "model_p50_ms",
                    "model_p95_ms",
                    "e2e_p50_ms",
                    "e2e_p95_ms",
                    "peak_cuda_mem_mb",
                    "throughput_samples_per_s",
                    "deadline_miss_rate",
                ] if k in rt
            ))

            try:
                import pandas as _pd
                df_rt = _pd.DataFrame([{"subject": subj, **rt}])
                out_csv = self.exp_dir / "runtime_metrics.csv"
                header = (not out_csv.exists())
                df_rt.to_csv(out_csv, mode="a", index=False, encoding="utf-8-sig", header=header)
            except Exception as _e:
                self.log.warning(f"Runtime metrics save failed: {_e}")
        del self.datamodule.test_dataset
        return metrics[1:]  # AUC, BA, F1, TPR, FPR

    def _prepare_dirs(self, subject_id: int) -> Tuple[Path, Path]:
        """Create per-subject checkpoint and log directories."""
        checkpoint_dir = self.exp_dir / "checkpoints" / f"sub{subject_id}"
        log_dir = self.exp_dir / "logs" / f"sub{subject_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir, log_dir
    
    def _clean_subject_logs(self, log_root: Path, *, run_prefix: str = "metrics"):
        """Delete previous TensorBoard runs under one subject log directory."""
        if not log_root.exists():
            return
        for child in log_root.iterdir():
            if child.is_dir() and child.name.startswith(run_prefix):
                rmtree(child, ignore_errors=True)

    def _log_subject(self, subject_id: int, metrics: Dict[str, float]):
        self.log.info(
            f"Subject {subject_id} | AUC: {metrics['AUC']:.4f} | BA: {metrics['BA']:.4f} "
            f"| F1: {metrics['F1']:.4f} | TPR: {metrics['TPR']:.4f} | FPR: {metrics['FPR']:.4f}"
        )

    def _load_latest_subject_metrics_from_log(self) -> Dict[int, Dict[str, float]]:
        if self.log is None or not getattr(self.log, "name", None):
            return {}
        log_path = Path(f"{self.log.name}.log")
        if not log_path.exists():
            return {}

        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return {}

        start_mark = f"Starting training for model: {self.config.get('model')}"
        start_idx = None
        for idx, line in enumerate(lines):
            if start_mark in line:
                start_idx = idx
        if start_idx is None:
            return {}

        metric_re = re.compile(
            r"Subject\s+(\d+)\s+\|\s+AUC:\s+([0-9.]+)\s+\|\s+BA:\s+([0-9.]+)\s+\|\s+F1:\s+([0-9.]+)\s+\|\s+TPR:\s+([0-9.]+)\s+\|\s+FPR:\s+([0-9.]+)"
        )
        parsed: Dict[int, Dict[str, float]] = {}
        for line in lines[start_idx + 1:]:
            m = metric_re.search(line)
            if not m:
                continue
            sid = int(m.group(1))
            parsed[sid] = {
                "AUC": float(m.group(2)),
                "BA": float(m.group(3)),
                "F1": float(m.group(4)),
                "TPR": float(m.group(5)),
                "FPR": float(m.group(6)),
            }
        return parsed

    def _save_results(self, results: Dict[str, List[float]], subject_ids: Optional[List[int]] = None):
        """Aggregate subject metrics, append AVG/STD, and save CSV."""
        df = pd.DataFrame(results)
        if subject_ids is None or len(subject_ids) != len(df):
            subject_ids = list(range(1, len(df) + 1))
        df.insert(0, "SUB", [f"SUB{i}" for i in subject_ids])

        metrics = ["AUC", "BA", "F1", "TPR", "FPR"]
        avg = df[metrics].mean().round(4)
        std = df[metrics].std().round(4)

        df = pd.concat([
            df,
            pd.Series({"SUB": "AVG", **avg}).to_frame().T,
            pd.Series({"SUB": "STD", **std}).to_frame().T,
        ], ignore_index=True)

        csv_path = self.exp_dir / "results.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")

        formatted = {k: f"{avg[k]:.4f}+/-{std[k]:.4f}" for k in metrics}
        self.log.info(f"Average metrics: {formatted}")
        self._log_detail(f"Results saved to {csv_path}")

    def _record_source_selection(self, subject_id: int):
        info = dict(getattr(self.datamodule, "source_selection_info", {}) or {})
        if not info:
            return
        selected = info.get("selected_subjects", [])
        selected_ids = [
            int(str(s)[3:]) for s in selected
            if isinstance(s, str) and str(s).startswith("sub") and str(s)[3:].isdigit()
        ]
        self._log_detail(
            f"Subject {subject_id} source selection | mode={info.get('mode')} | "
            f"selected={info.get('num_selected')}/{info.get('num_candidates')} | "
            f"subjects={selected}"
        )
        self.source_selection_records.append(
            {
                "subject": subject_id,
                "mode": info.get("mode"),
                "k": info.get("k"),
                "min_score": info.get("min_score"),
                "selection_seed": info.get("selection_seed"),
                "held_out_subject": f"sub{subject_id}",
                "num_candidates": info.get("num_candidates"),
                "num_selected": info.get("num_selected"),
                "selected_subjects": ",".join(selected),
                "selected_subject_ids": ",".join(str(v) for v in selected_ids),
            }
        )
        self._record_rpcs_details(subject_id, selected_subjects=selected)

    def _save_source_selection(self):
        if not self.source_selection_records:
            return
        df = pd.DataFrame(self.source_selection_records)
        out = self.exp_dir / "source_selection.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        self._log_detail(f"Source selection records saved to {out}")

        # Human-readable manifest for quick copy/reuse of LOSO source pools.
        manifest = self.exp_dir / "source_selection_manifest.txt"
        lines: List[str] = []
        lines.append(f"dataset={self.config.get('dataset')} | seed={self.config.get('random_seed')}")
        for row in self.source_selection_records:
            lines.append(
                f"{row.get('held_out_subject')} | mode={row.get('mode')} | "
                f"k={row.get('k')} | selected={row.get('selected_subjects')}"
            )
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._log_detail(f"Source selection manifest saved to {manifest}")

    def _record_rpcs_details(self, subject_id: int, selected_subjects: List[str]):
        meta = dict(getattr(self.hyr_dpa, "last_source_selection_meta", {}) or {})
        if str(meta.get("mode", "")).lower() != "scores":
            return
        details = dict(meta.get("details", {}) or {})
        ranking = list(details.get("ranking", []) or [])
        if not ranking:
            return

        selected_set = set(selected_subjects or [])
        target_info = dict(details.get("target", {}) or {})
        held_out = str(meta.get("held_out", f"sub{subject_id}"))
        prototypes = dict(details.get("prototypes", {}) or {})
        self._save_rpcs_fold_prototypes(
            subject_id=subject_id,
            held_out=held_out,
            ranking=ranking,
            selected_subjects=selected_subjects,
            prototypes=prototypes,
        )
        self._save_rpcs_fold_topk_plot(
            subject_id=subject_id,
            held_out=held_out,
            ranking=ranking,
            selected_subjects=selected_subjects,
        )

        for rank_idx, row in enumerate(ranking, start=1):
            d_val = float(
                row.get(
                    "discriminability_distance",
                    row.get("distance", float("nan")),
                )
            )
            s_val = float(row.get("similarity_distance", float("nan")))
            rec = {
                "target_subject": held_out,
                "target_id": int(subject_id),
                "source_subject": row.get("subject"),
                "rank": int(rank_idx),
                "score": float(row.get("score", float("nan"))),
                "distance": d_val,
                "discriminability_distance": d_val,
                "similarity_distance": s_val,
                "score_mode": row.get("score_mode", details.get("config", {}).get("score_mode", "rpcs")),
                "positive_total_trials": int(row.get("positive_total_trials", 0)),
                "positive_used_trials": int(row.get("positive_used_trials", 0)),
                "background_total_trials": int(row.get("background_total_trials", 0)),
                "background_used_trials": int(row.get("background_used_trials", 0)),
                "source_pos_proto_trace": float(
                    row.get("source_pos_proto_trace", row.get("source_proto_trace", float("nan")))
                ),
                "source_pos_proto_logdet": float(
                    row.get("source_pos_proto_logdet", row.get("source_proto_logdet", float("nan")))
                ),
                "source_pos_proto_cond": float(
                    row.get("source_pos_proto_cond", row.get("source_proto_cond", float("nan")))
                ),
                "source_bg_proto_trace": float(row.get("source_bg_proto_trace", float("nan"))),
                "source_bg_proto_logdet": float(row.get("source_bg_proto_logdet", float("nan"))),
                "source_bg_proto_cond": float(row.get("source_bg_proto_cond", float("nan"))),
                "is_selected": int(row.get("subject") in selected_set),
                "target_total_trials": int(target_info.get("target_total_trials", target_info.get("background_total_trials", 0))),
                "target_used_trials": int(target_info.get("target_used_trials", target_info.get("background_used_trials", 0))),
                "target_bg_total_trials": int(target_info.get("background_total_trials", 0)),
                "target_bg_used_trials": int(target_info.get("background_used_trials", 0)),
                "target_proto_mode": target_info.get("target_proto_mode", ""),
                "target_bg_proto_trace": float(target_info.get("target_bg_proto_trace", float("nan"))),
                "target_bg_proto_logdet": float(target_info.get("target_bg_proto_logdet", float("nan"))),
                "target_bg_proto_cond": float(target_info.get("target_bg_proto_cond", float("nan"))),
            }
            self.rpcs_ranking_records.append(rec)

        best_row = ranking[0]
        selected_rows = [r for r in ranking if r.get("subject") in selected_set]
        self.rpcs_fold_records.append(
            {
                "target_subject": held_out,
                "target_id": int(subject_id),
                "num_candidates": int(len(ranking)),
                "num_selected": int(len(selected_set)),
                "best_source_subject": best_row.get("subject"),
                "best_score": float(best_row.get("score", float("nan"))),
                "best_discriminability_distance": float(
                    best_row.get("discriminability_distance", best_row.get("distance", float("nan")))
                ),
                "best_similarity_distance": float(best_row.get("similarity_distance", float("nan"))),
                "selected_score_mean": float(np.mean([r.get("score", float("nan")) for r in selected_rows]))
                if selected_rows
                else float("nan"),
                "selected_discriminability_distance_mean": float(
                    np.mean(
                        [
                            r.get("discriminability_distance", r.get("distance", float("nan")))
                            for r in selected_rows
                        ]
                    )
                )
                if selected_rows
                else float("nan"),
                "selected_similarity_distance_mean": float(
                    np.mean([r.get("similarity_distance", float("nan")) for r in selected_rows])
                )
                if selected_rows
                else float("nan"),
                "target_bg_total_trials": int(target_info.get("background_total_trials", 0)),
                "target_bg_used_trials": int(target_info.get("background_used_trials", 0)),
            }
        )

    def _save_rpcs_fold_prototypes(
        self,
        *,
        subject_id: int,
        held_out: str,
        ranking: List[Dict[str, Any]],
        selected_subjects: List[str],
        prototypes: Dict[str, Any],
    ):
        if not prototypes:
            return
        target_bg = prototypes.get("target_bg", None)
        source_pos_map = dict(prototypes.get("source_pos", {}) or {})
        source_bg_map = dict(prototypes.get("source_bg", {}) or {})
        if target_bg is None or not source_pos_map:
            return

        rank_subjects = [
            str(r.get("subject"))
            for r in ranking
            if r.get("subject") in source_pos_map and r.get("subject") in source_bg_map
        ]
        if not rank_subjects:
            return
        all_pos_stack = np.stack([np.asarray(source_pos_map[s], dtype=np.float64) for s in rank_subjects], axis=0)
        all_bg_stack = np.stack([np.asarray(source_bg_map[s], dtype=np.float64) for s in rank_subjects], axis=0)
        selected_subjects = [s for s in (selected_subjects or []) if s in source_pos_map and s in source_bg_map]
        selected_pos_stack = (
            np.stack([np.asarray(source_pos_map[s], dtype=np.float64) for s in selected_subjects], axis=0)
            if selected_subjects
            else np.empty((0, *all_pos_stack.shape[1:]), dtype=np.float64)
        )
        selected_bg_stack = (
            np.stack([np.asarray(source_bg_map[s], dtype=np.float64) for s in selected_subjects], axis=0)
            if selected_subjects
            else np.empty((0, *all_bg_stack.shape[1:]), dtype=np.float64)
        )

        out_dir = self.exp_dir / "rpcs_prototypes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"{held_out}.npz"
        np.savez_compressed(
            out_npz,
            target_subject=np.array([held_out]),
            target_bg=np.asarray(target_bg, dtype=np.float64),
            source_subjects=np.array(rank_subjects),
            source_pos=all_pos_stack,
            source_bg=all_bg_stack,
            selected_subjects=np.array(selected_subjects),
            selected_source_pos=selected_pos_stack,
            selected_source_bg=selected_bg_stack,
        )

    def _save_rpcs_fold_topk_plot(
        self,
        *,
        subject_id: int,
        held_out: str,
        ranking: List[Dict[str, Any]],
        selected_subjects: List[str],
    ):
        if not ranking:
            return
        topk_cfg = self.config.get("rpcs_plot_top_k", self.config.get("pccs_plot_top_k", 10))
        topk = int(topk_cfg)
        topk = max(1, min(topk, len(ranking)))
        rows = ranking[:topk]
        labels = [str(r.get("subject")) for r in rows]
        values = [float(r.get("score", float("nan"))) for r in rows]
        selected_set = set(selected_subjects or [])
        colors = ["tab:orange" if lb in selected_set else "tab:blue" for lb in labels]

        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(9, 4))
            plt.bar(range(len(labels)), values, color=colors)
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            score_mode = str(
                rows[0].get(
                    "score_mode",
                    self.config.get("rpcs_score_mode", self.config.get("pccs_score_mode", "rpcs")),
                )
            )
            plt.ylabel(f"R-PCS score ({score_mode})")
            plt.title(f"R-PCS Top-{topk} for {held_out}")
            plt.tight_layout()
            out_dir = self.exp_dir / "rpcs_plots"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"rpcs_topk_{held_out}.png", dpi=150)
            plt.close(fig)
        except Exception as _e:
            self.log.warning(f"R-PCS top-k plot save failed for {held_out}: {_e}")

    def _save_rpcs_artifacts(self):
        if not self.rpcs_ranking_records:
            return

        df = pd.DataFrame(self.rpcs_ranking_records)
        out_long = self.exp_dir / "rpcs_ranking_long.csv"
        df.to_csv(out_long, index=False, encoding="utf-8-sig")

        df_selected = df[df["is_selected"] == 1].copy()
        out_selected = self.exp_dir / "rpcs_selected_topk.csv"
        df_selected.to_csv(out_selected, index=False, encoding="utf-8-sig")

        if self.rpcs_fold_records:
            df_fold = pd.DataFrame(self.rpcs_fold_records)
            out_fold = self.exp_dir / "rpcs_fold_summary.csv"
            df_fold.to_csv(out_fold, index=False, encoding="utf-8-sig")

        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(7, 4))
            plt.hist(df["discriminability_distance"].fillna(df["distance"]).dropna().values, bins=30)
            plt.title("R-PCS Discriminability Distance Distribution")
            plt.xlabel("Distance(P_source_p300, P_target_bg)")
            plt.ylabel("Count")
            plt.tight_layout()
            fig.savefig(self.exp_dir / "rpcs_distance_hist.png", dpi=150)
            plt.close(fig)

            if not df_selected.empty:
                fig = plt.figure(figsize=(7, 4))
                plt.hist(
                    df_selected["discriminability_distance"].fillna(df_selected["distance"]).dropna().values,
                    bins=20,
                )
                plt.title("R-PCS Selected Source Discriminability Distance")
                plt.xlabel("Distance(P_source_p300, P_target_bg)")
                plt.ylabel("Count")
                plt.tight_layout()
                fig.savefig(self.exp_dir / "rpcs_selected_distance_hist.png", dpi=150)
                plt.close(fig)

                if "similarity_distance" in df_selected.columns:
                    fig = plt.figure(figsize=(7, 4))
                    plt.hist(df_selected["similarity_distance"].dropna().values, bins=20)
                    plt.title("R-PCS Selected Source Similarity Distance")
                    plt.xlabel("Distance(P_source_bg, P_target_bg)")
                    plt.ylabel("Count")
                    plt.tight_layout()
                    fig.savefig(self.exp_dir / "rpcs_selected_similarity_hist.png", dpi=150)
                    plt.close(fig)

            # Fold x source score heatmap
            score_mat = df.pivot_table(
                index="target_subject",
                columns="source_subject",
                values="score",
                aggfunc="mean",
            )
            if not score_mat.empty:
                fig = plt.figure(figsize=(10, 6))
                arr = score_mat.values
                im = plt.imshow(arr, aspect="auto")
                plt.colorbar(im, fraction=0.03, pad=0.02, label="R-PCS score")
                plt.yticks(range(score_mat.shape[0]), score_mat.index.tolist())
                plt.xticks(range(score_mat.shape[1]), score_mat.columns.tolist(), rotation=90)
                plt.xlabel("Source subject")
                plt.ylabel("Target subject")
                plt.title("R-PCS Score Heatmap")
                plt.tight_layout()
                fig.savefig(self.exp_dir / "rpcs_score_heatmap.png", dpi=150)
                plt.close(fig)

            # Fold x source selected mask heatmap
            selected_mat = df.pivot_table(
                index="target_subject",
                columns="source_subject",
                values="is_selected",
                aggfunc="max",
                fill_value=0,
            )
            if not selected_mat.empty:
                fig = plt.figure(figsize=(10, 6))
                arr = selected_mat.values
                im = plt.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0)
                plt.colorbar(im, fraction=0.03, pad=0.02, label="Selected (1/0)")
                plt.yticks(range(selected_mat.shape[0]), selected_mat.index.tolist())
                plt.xticks(range(selected_mat.shape[1]), selected_mat.columns.tolist(), rotation=90)
                plt.xlabel("Source subject")
                plt.ylabel("Target subject")
                plt.title("R-PCS Selection Heatmap")
                plt.tight_layout()
                fig.savefig(self.exp_dir / "rpcs_selection_heatmap.png", dpi=150)
                plt.close(fig)
        except Exception as _e:
            self.log.warning(f"R-PCS visualization save failed: {_e}")

        self._log_detail(f"R-PCS ranking records saved to {out_long}")
        self._log_detail(f"R-PCS selected-source records saved to {out_selected}")


class OptimizedTrainer:
    """Training engine for one subject fold."""

    def __init__(self, config: Dict, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 earlystopping: EarlyStopping, save_other_model: SaveBestValBA,
                 device: torch.device,
                 class_weights: Optional[torch.Tensor] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.earlystopping = earlystopping
        self.save_other_model = save_other_model
        self.device = device
        self.logger = logger
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = class_weights.detach().to(self.device, dtype=torch.float32)

        self.loss_fn = self.loss_ce

        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(scaler_device, enabled=(self.device.type == "cuda"))
        self.tb_logger = None
        self.checkpoint_dir = None
        self.current_epoch = 0

        self.log_runtime: bool = bool(self.config.get("log_runtime", True))
        self.rt_deadline_ms: float = float(self.config.get("rt_deadline_ms", 200.0))
        self.last_runtime_stats: dict | None = None
        self.minimal_log: bool = bool(self.config.get("minimal_log", True))
        self.debug_domain_batch: bool = bool(self.config.get("debug_domain_batch", False))
        self.debug_domain_batch_max_steps: int = int(self.config.get("debug_domain_batch_max_steps", 3))
        self._debug_domain_steps_emitted: int = 0

        # Stage-1 optional modules (RPT-Aug + IAHM)
        self.stage1_rpt_augmentor: Optional[RPTAugmentor] = None
        self.stage1_iahm_cfg: Dict[str, Any] = {}
        self.stage1_source_class_counts: Optional[List[int]] = None
        self.stage1_positive_label: int = 1
        self.stage1_background_label: int = 0
        self.stage1_iahm_loss: Optional[IAHMLoss] = None
        self.stage1_global_step: int = 0

        self.lambda_align: float = float(self.config.get("lambda_align", 0.0))
        self.lambda_class_align: float = float(self.config.get("lambda_class_align", 0.0))
        self.lambda_ccl: float = float(self.config.get("lambda_ccl", 0.0))
        self.class_align_start_epoch: int = int(self.config.get("class_align_start_epoch", 30))
        self.class_align_conf_thresh: float = float(self.config.get("class_align_conf_thresh", 0.8))
        self.class_align_min_conf_samples: int = int(self.config.get("class_align_min_conf_samples", 8))
        self.class_align_use_soft_weights: bool = bool(self.config.get("class_align_use_soft_weights", True))
        self.ccl_start_epoch: int = int(self.config.get("ccl_start_epoch", 0))
        self.lambda_prior: float = float(self.config.get("lambda_prior", 0.0))
        self.prior_start_epoch: int = int(self.config.get("prior_start_epoch", 30))
        self.prior_min: float = float(self.config.get("prior_min", 0.03))
        self.prior_max: float = float(self.config.get("prior_max", 0.09))
        self.prior_loss_type: str = str(self.config.get("prior_loss_type", "l2")).strip().lower()
        self.alignment_loss_name: str = str(self.config.get("alignment_loss", "mmd")).strip().lower()
        self.lmmd_use_soft_target: bool = bool(self.config.get("lmmd_use_soft_target", True))
        self.lmmd_kernel_mul: float = float(self.config.get("lmmd_kernel_mul", 2.0))
        self.lmmd_kernel_num: int = int(self.config.get("lmmd_kernel_num", 5))
        self.lmmd_fix_sigma = self.config.get("lmmd_fix_sigma", None)
        self.uot_eps: float = float(self.config.get("uot_eps", 0.1))
        self.uot_tau_source: float = float(self.config.get("uot_tau_source", 1.0))
        self.uot_tau_target: float = float(self.config.get("uot_tau_target", 1.0))
        self.uot_max_iter: int = int(self.config.get("uot_max_iter", 30))
        self.dual_head_lambda_flat_ce: float = float(self.config.get("eegnet_dual_lambda_flat_ce", 0.0))
        self.dual_head_lambda_ts_ce: float = float(self.config.get("eegnet_dual_lambda_ts_ce", 0.0))
        self.aux_ts_lambda_align: float = float(self.config.get("eegnet_aux_ts_lambda_align", 0.0))
        self.lsa_content_lambda: float = float(self.config.get("lsa_content_lambda", 0.0))
        self.lsa_identity_lambda: float = float(self.config.get("lsa_identity_lambda", 0.0))
        self.prototype_enable: bool = bool(self.config.get("prototype_enable", False))
        self.prototype_lambda: float = float(self.config.get("prototype_lambda", 0.0))
        self.prototype_momentum: float = float(self.config.get("prototype_momentum", 0.9))
        self.prototype_positive_weight: float = float(self.config.get("prototype_positive_weight", 1.0))
        self.prototype_background_weight: float = float(self.config.get("prototype_background_weight", 1.0))
        self.prototype_positive_label: int = int(self.config.get("prototype_positive_label", 1))
        self.prototype_background_label: int = int(self.config.get("prototype_background_label", 0))
        self.prototype_separation_lambda: float = float(self.config.get("prototype_separation_lambda", 0.0))
        self.prototype_separation_margin: float = float(self.config.get("prototype_separation_margin", 1.0))
        self.prototype_vectors: Optional[torch.Tensor] = None
        self.prototype_initialized: Optional[torch.Tensor] = None
        self.posdist_enable: bool = bool(self.config.get("posdist_enable", False))
        self.posdist_lambda: float = float(self.config.get("posdist_lambda", 0.0))
        self.posdist_start_epoch: int = int(self.config.get("posdist_start_epoch", 0))
        self.posdist_var_weight: float = float(self.config.get("posdist_var_weight", 1.0))
        self.posdist_momentum: float = float(self.config.get("posdist_momentum", 0.9))
        self.posdist_positive_label: int = int(self.config.get("posdist_positive_label", 1))
        self.posdist_mean: Optional[torch.Tensor] = None
        self.posdist_var: Optional[torch.Tensor] = None
        self.posdist_initialized: bool = False
        self.rpt_synth_per_batch_cfg = self.config.get("rpt_aug_n_synth_per_batch", None)
        self.rpt_inject_to_ce: bool = bool(self.config.get("rpt_aug_inject_to_ce", True))
        self._alignment_warned = False
        self._eval_use_target_stats: bool = False
        self.dynamic_feature_sampling_cfg: Dict[str, Any] = {}
        self.dynamic_feature_sampling_datamodule: Optional[EEGDataModuleCrossSubject] = None

    def _log_info(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    @staticmethod
    def _print_console(msg: str) -> None:
        print(msg, flush=True)

    def _maybe_log_domain_debug(self, source_domain_id: Optional[torch.Tensor]) -> None:
        if not self.debug_domain_batch:
            return
        if source_domain_id is None or source_domain_id.numel() == 0:
            return
        if self._debug_domain_steps_emitted >= max(0, self.debug_domain_batch_max_steps):
            return

        domain_cpu = source_domain_id.detach().to("cpu")
        unique_ids, counts = torch.unique(domain_cpu, sorted=True, return_counts=True)
        domain_ids = [int(v) for v in unique_ids.tolist()]
        count_values = [int(v) for v in counts.tolist()]
        count_map = {int(k): int(v) for k, v in zip(domain_ids, count_values)}
        self._debug_domain_steps_emitted += 1
        self._log_info(
            f"[DOMAIN-DEBUG] epoch={self.current_epoch:03d} "
            f"step={self._debug_domain_steps_emitted:03d} "
            f"n_domains={len(domain_ids)} domains={domain_ids} counts={count_map}"
        )

    def setup_logging(self, log_dir: Path, checkpoint_dir: Path, stage: int, *, run_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name=run_name,
            version=None,
            default_hp_metric=False,
        )
        stage_suffix = f"-{stage}"
        self.earlystopping.path = checkpoint_dir / f"best_model-{stage_suffix}.pth"
        self.save_other_model.path = checkpoint_dir / f"best_ba_model-{stage_suffix}.pth"

    def configure_stage1_modules(
        self,
        *,
        rpt_augmentor: Optional[RPTAugmentor],
        iahm_cfg: Optional[Dict[str, Any]],
        source_class_counts: Optional[List[int]],
        positive_label: int = 1,
        background_label: int = 0,
    ) -> None:
        self.stage1_rpt_augmentor = rpt_augmentor
        self.stage1_iahm_cfg = dict(iahm_cfg or {})
        self.stage1_source_class_counts = (
            [int(v) for v in source_class_counts]
            if source_class_counts is not None
            else None
        )
        self.stage1_positive_label = int(positive_label)
        self.stage1_background_label = int(background_label)
        self.stage1_iahm_loss = None
        self.stage1_global_step = 0

    def configure_dynamic_feature_sampling(
        self,
        *,
        datamodule: EEGDataModuleCrossSubject,
    ) -> None:
        self.dynamic_feature_sampling_datamodule = datamodule
        self.dynamic_feature_sampling_cfg = {
            "enable": bool(self.config.get("dynamic_feature_sampling_enable", False)),
            "warmup_epochs": int(self.config.get("dynamic_feature_sampling_warmup_epochs", 10)),
            "refresh_every": int(self.config.get("dynamic_feature_sampling_refresh_every", 10)),
            "source_support_size": int(self.config.get("dynamic_feature_sampling_source_support_size", 128)),
            "target_support_size": int(self.config.get("dynamic_feature_sampling_target_support_size", 128)),
            "seed": int(self.config.get("dynamic_feature_sampling_seed", self.config.get("random_seed", 2026))),
            "metric": str(self.config.get("dynamic_feature_sampling_metric", "mmd")).strip().lower(),
            "temperature": float(self.config.get("dynamic_feature_sampling_temperature", 1.0)),
            "mix_alpha": float(self.config.get("dynamic_feature_sampling_mix_alpha", 0.2)),
            "l2_normalize": bool(self.config.get("dynamic_feature_sampling_l2_normalize", True)),
            "score_eps": float(self.config.get("dynamic_feature_sampling_score_eps", 1e-6)),
            "mmd_sigma": float(self.config.get("dynamic_feature_sampling_mmd_sigma", 1.0)),
            "batch_size": int(self.config.get("dynamic_feature_sampling_batch_size", 256)),
            "score_ema_enable": bool(self.config.get("dynamic_feature_sampling_score_ema_enable", False)),
            "score_ema_decay": float(self.config.get("dynamic_feature_sampling_score_ema_decay", 0.8)),
        }

    def _resolve_subject_batch_loader(self, train_loader):
        if isinstance(train_loader, DualStreamDataLoader):
            return train_loader.source_loader
        return train_loader

    def _dynamic_feature_sampling_active(self, train_loader) -> bool:
        cfg = dict(self.dynamic_feature_sampling_cfg or {})
        if not bool(cfg.get("enable", False)):
            return False
        if self.dynamic_feature_sampling_datamodule is None:
            return False
        source_loader = self._resolve_subject_batch_loader(train_loader)
        if not isinstance(source_loader, SubjectBatchDataLoader):
            return False
        info = dict(getattr(self.dynamic_feature_sampling_datamodule, "source_selection_info", {}) or {})
        mode = str(info.get("mode", "")).strip().lower()
        return mode == "all" and bool(getattr(source_loader, "random_subjects_each_step", False))

    def _extract_feature_tensor(self, dataset) -> torch.Tensor:
        cfg = dict(self.dynamic_feature_sampling_cfg or {})
        bs = max(1, int(cfg.get("batch_size", 256)))
        l2_normalize = bool(cfg.get("l2_normalize", True))
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        chunks: List[torch.Tensor] = []
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self.device, non_blocking=True)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                logits, extras = self._forward_unpack(x)
                feat = self._select_hyperbolic_input(logits, extras).detach().float()
                if l2_normalize:
                    feat = F.normalize(feat, p=2, dim=1)
                chunks.append(feat.cpu())
        if was_training:
            self.model.train()
        if not chunks:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def _extract_labeled_features_and_labels(self, dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = dict(self.dynamic_feature_sampling_cfg or {})
        bs = max(1, int(cfg.get("batch_size", 256)))
        l2_normalize = bool(cfg.get("l2_normalize", True))
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        feat_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    continue
                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].detach().to("cpu", dtype=torch.long)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                logits, extras = self._forward_unpack(x)
                feat = self._select_hyperbolic_input(logits, extras).detach().float()
                if l2_normalize:
                    feat = F.normalize(feat, p=2, dim=1)
                feat_chunks.append(feat.cpu())
                label_chunks.append(y)
        if was_training:
            self.model.train()
        if not feat_chunks:
            return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
        return torch.cat(feat_chunks, dim=0), torch.cat(label_chunks, dim=0)

    def _extract_target_support_features_and_probs(
        self,
        dataset,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = dict(self.dynamic_feature_sampling_cfg or {})
        bs = max(1, int(cfg.get("batch_size", 256)))
        l2_normalize = bool(cfg.get("l2_normalize", True))
        pos_label = int(self.config.get("rpcs_positive_label", 1))
        bg_label = int(self.config.get("rpcs_background_label", 0))
        use_confidence_weight = bool(cfg.get("use_confidence_weight", True))
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        feat_chunks: List[torch.Tensor] = []
        pos_chunks: List[torch.Tensor] = []
        bg_chunks: List[torch.Tensor] = []
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self.device, non_blocking=True)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                logits, extras = self._forward_unpack(x)
                feat = self._select_hyperbolic_input(logits, extras).detach().float()
                if l2_normalize:
                    feat = F.normalize(feat, p=2, dim=1)
                prob = torch.softmax(logits.detach().float(), dim=1)
                p_pos = prob[:, pos_label]
                p_bg = prob[:, bg_label]
                if use_confidence_weight:
                    conf = torch.abs(p_pos - p_bg)
                    p_pos = p_pos * conf
                    p_bg = p_bg * conf
                feat_chunks.append(feat.cpu())
                pos_chunks.append(p_pos.detach().cpu())
                bg_chunks.append(p_bg.detach().cpu())
        if was_training:
            self.model.train()
        if not feat_chunks:
            return (
                torch.empty((0, 0), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
        return (
            torch.cat(feat_chunks, dim=0),
            torch.cat(pos_chunks, dim=0),
            torch.cat(bg_chunks, dim=0),
        )

    def _pair_similarity_score(self, source_feat: torch.Tensor, target_feat: torch.Tensor) -> float:
        cfg = dict(self.dynamic_feature_sampling_cfg or {})
        metric = str(cfg.get("metric", "mmd")).strip().lower()
        score_eps = float(cfg.get("score_eps", 1e-6))
        if source_feat.numel() == 0 or target_feat.numel() == 0:
            return 0.0

        x = source_feat.to(self.device, dtype=torch.float32)
        y = target_feat.to(self.device, dtype=torch.float32)

        if metric == "mmd":
            sigma = float(cfg.get("mmd_sigma", 1.0))
            dist = float(self._mmd_rbf(x, y, sigma=sigma).detach().item())
            return 1.0 / max(dist + score_eps, score_eps)
        if metric == "mean_cosine":
            mx = x.mean(dim=0, keepdim=True)
            my = y.mean(dim=0, keepdim=True)
            sim = F.cosine_similarity(mx, my, dim=1)
            return float(torch.clamp(sim, min=0.0).detach().item()) + score_eps
        if metric == "mean_euclidean":
            mx = x.mean(dim=0)
            my = y.mean(dim=0)
            dist = float(torch.norm(mx - my, p=2).detach().item())
            return 1.0 / max(dist + score_eps, score_eps)
        raise ValueError(f"Unsupported dynamic_feature_sampling_metric: {metric}")

    def _soft_class_mmd_score(
        self,
        *,
        source_feat: torch.Tensor,
        source_labels: torch.Tensor,
        target_feat: torch.Tensor,
        target_pos_weights: torch.Tensor,
        target_bg_weights: torch.Tensor,
    ) -> float:
        cfg = dict(self.dynamic_feature_sampling_cfg or {})
        pos_label = int(self.config.get("rpcs_positive_label", 1))
        bg_label = int(self.config.get("rpcs_background_label", 0))
        pos_weight = float(cfg.get("pos_weight", 0.7))
        pos_weight = float(np.clip(pos_weight, 0.0, 1.0))
        bg_weight = 1.0 - pos_weight
        sigma = float(cfg.get("mmd_sigma", 1.0))
        score_eps = float(cfg.get("score_eps", 1e-6))
        min_class_samples = max(1, int(cfg.get("min_class_samples", 8)))

        if source_feat.numel() == 0 or target_feat.numel() == 0 or source_labels.numel() == 0:
            return 0.0

        src_feat = source_feat.to(self.device, dtype=torch.float32)
        src_labels = source_labels.to(self.device, dtype=torch.long)
        tgt_feat = target_feat.to(self.device, dtype=torch.float32)
        w_pos = target_pos_weights.to(self.device, dtype=torch.float32)
        w_bg = target_bg_weights.to(self.device, dtype=torch.float32)

        src_pos = src_feat[src_labels == pos_label]
        src_bg = src_feat[src_labels == bg_label]
        if int(src_pos.shape[0]) < min_class_samples or int(src_bg.shape[0]) < min_class_samples:
            return 0.0

        d_pos = self._mmd_rbf(src_pos, tgt_feat, y_weights=w_pos, sigma=sigma)
        d_bg = self._mmd_rbf(src_bg, tgt_feat, y_weights=w_bg, sigma=sigma)
        s_pos = 1.0 / max(float(d_pos.detach().item()) + score_eps, score_eps)
        s_bg = 1.0 / max(float(d_bg.detach().item()) + score_eps, score_eps)
        return pos_weight * s_pos + bg_weight * s_bg

    def _refresh_dynamic_feature_sampling_scores(self, train_loader) -> None:
        if not self._dynamic_feature_sampling_active(train_loader):
            return

        cfg = dict(self.dynamic_feature_sampling_cfg or {})
        source_loader = self._resolve_subject_batch_loader(train_loader)
        source_loader.subject_sampling_mix_alpha = float(np.clip(cfg.get("mix_alpha", 0.2), 0.0, 1.0))

        warmup_epochs = max(0, int(cfg.get("warmup_epochs", 10)))
        refresh_every = max(1, int(cfg.get("refresh_every", 10)))
        epoch = int(self.current_epoch)

        if epoch <= warmup_epochs:
            source_loader.subject_sampling_weights = {}
            if self.dynamic_feature_sampling_datamodule is not None:
                self.dynamic_feature_sampling_datamodule.clear_dynamic_source_scores()
            return

        should_refresh = (epoch == warmup_epochs + 1) or ((epoch - (warmup_epochs + 1)) % refresh_every == 0)
        if not should_refresh:
            return

        dm = self.dynamic_feature_sampling_datamodule
        assert dm is not None
        source_support_map = dm.get_dynamic_feature_source_support_datasets(
            max_samples_per_subject=int(cfg.get("source_support_size", 128)),
            seed=int(cfg.get("seed", self.config.get("random_seed", 2026))),
        )
        target_support = dm.get_dynamic_feature_target_support_dataset(
            max_samples=int(cfg.get("target_support_size", 128)),
            seed=int(cfg.get("seed", self.config.get("random_seed", 2026))),
        )
        metric = str(cfg.get("metric", "mmd")).strip().lower()
        if metric == "soft_class_mmd":
            target_feat, target_pos_w, target_bg_w = self._extract_target_support_features_and_probs(target_support)
        else:
            target_feat = self._extract_feature_tensor(target_support)
        raw_scores: Dict[str, float] = {}
        for skey, ds in source_support_map.items():
            if metric == "soft_class_mmd":
                source_feat, source_labels = self._extract_labeled_features_and_labels(ds)
                raw_scores[str(skey)] = float(
                    self._soft_class_mmd_score(
                        source_feat=source_feat,
                        source_labels=source_labels,
                        target_feat=target_feat,
                        target_pos_weights=target_pos_w,
                        target_bg_weights=target_bg_w,
                    )
                )
            else:
                source_feat = self._extract_feature_tensor(ds)
                raw_scores[str(skey)] = float(self._pair_similarity_score(source_feat, target_feat))

        score_eps = float(cfg.get("score_eps", 1e-6))
        temp = max(float(cfg.get("temperature", 1.0)), 1e-6)
        score_keys = list(raw_scores.keys())
        score_arr = np.asarray([max(float(raw_scores[k]), score_eps) for k in score_keys], dtype=np.float64)
        logits = np.log(score_arr + score_eps) / temp
        logits = logits - float(np.max(logits))
        weight_arr = np.exp(logits)
        if not np.isfinite(weight_arr).all() or float(np.sum(weight_arr)) <= 0.0:
            weight_arr = np.ones_like(weight_arr)
        score_map = {k: float(v) for k, v in zip(score_keys, weight_arr.tolist())}

        ema_enable = bool(cfg.get("score_ema_enable", False))
        ema_decay = float(np.clip(cfg.get("score_ema_decay", 0.8), 0.0, 0.999))
        prev_score_map = dict(getattr(dm, "_dynamic_source_scores", {}) or {})
        if ema_enable and prev_score_map:
            ema_map: Dict[str, float] = {}
            for k in score_keys:
                prev_v = float(prev_score_map.get(k, 0.0))
                cur_v = float(score_map.get(k, 0.0))
                ema_map[k] = ema_decay * prev_v + (1.0 - ema_decay) * cur_v
            score_map = ema_map

        source_loader.subject_sampling_weights = dict(score_map)
        dm.set_dynamic_source_scores(score_map)
        top5 = sorted(((k, raw_scores[k]) for k in raw_scores.keys()), key=lambda kv: kv[1], reverse=True)[:5]
        preview = ", ".join([f"{k}:{v:.4f}" for k, v in top5])
        self._log_info(
            f"[DYN-SAMPLING] epoch={epoch:03d} metric={cfg.get('metric', 'mmd')} "
            f"updated {len(score_map)} source scores"
            f"{' | ema=true' if ema_enable else ''} | top-5 raw: {preview}"
        )

    def fit(self, train_loader, val_loader, stage: int, epochs: int):
        patience = int(self.config.get("patience", 20))
        early_stop_start_epoch = int(self.config.get("early_stop_start_epoch", 0))
        ba_delta = float(self.config.get("ba_delta", 0.0))
        best_ba = float("-inf")
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self._refresh_dynamic_feature_sampling_scores(train_loader)
            with MemoryManager.cuda_memory_context():
                train_metrics = self._run_epoch(
                    train_loader,
                    training=True,
                    loss_fn=self.loss_fn,
                    profile_runtime=False,
                )
                train_loss_avg = train_metrics[0][0][0]
                train_ba = train_metrics[2]

                val_metrics = self._run_epoch(
                    val_loader,
                    training=False,
                    loss_fn=self.loss_fn,
                    profile_runtime=False,
                )
                val_loss_avg = val_metrics[0][0][0]
                val_ba = val_metrics[2]

                if self.minimal_log:
                    self._display_progress_console(epoch, epochs, train_loss_avg, val_loss_avg, val_ba)
                else:
                    self._display_progress(epoch, epochs, train_loss_avg, val_loss_avg, val_ba)
                self._log_training_metrics(train_metrics, mode="train", stage=stage)
                self._log_training_metrics(val_metrics, mode="val", stage=stage)
                if self.tb_logger is not None and train_ba is not None and val_ba is not None:
                    ba_gap = float(train_ba) - float(val_ba)
                    self.tb_logger.experiment.add_scalar(f"val/stage{stage}/BA_gap_train_minus_val", ba_gap, self.current_epoch)

                if val_ba is not None:
                    val_ba_f = float(val_ba)
                    self.save_other_model(val_ba_f, self.model)

                    if val_ba_f > (best_ba + ba_delta):
                        best_ba = val_ba_f
                        no_improve_epochs = 0
                    elif epoch > early_stop_start_epoch:
                        no_improve_epochs += 1
                elif epoch > early_stop_start_epoch:
                    no_improve_epochs += 1

                if epoch > early_stop_start_epoch and no_improve_epochs >= patience:
                    if self.minimal_log:
                        self._print_console(
                            f"Early stopping triggered on BA | epoch={epoch} | "
                            f"start_epoch={early_stop_start_epoch} | patience={patience} | best_BA={best_ba:.4f}"
                        )
                    else:
                        self._log_info(
                            f"Early stopping triggered on BA | epoch={epoch} | "
                            f"start_epoch={early_stop_start_epoch} | patience={patience} | best_BA={best_ba:.4f}"
                        )
                    break

                self.scheduler.step()

                if hasattr(self.model, "step_epoch"):
                    self.model.step_epoch(1)


    def _stage1_has_aux(self) -> bool:
        has_iahm = bool(self.stage1_iahm_cfg.get("enable", False))
        has_align = float(self.lambda_align) > 0.0
        has_class_align = float(self.lambda_class_align) > 0.0
        has_ccl = float(self.lambda_ccl) > 0.0
        has_prior = float(self.lambda_prior) > 0.0
        has_lsa_content = float(self.lsa_content_lambda) > 0.0
        has_lsa_identity = float(self.lsa_identity_lambda) > 0.0
        return has_iahm or has_align or has_class_align or has_ccl or has_prior or has_lsa_content or has_lsa_identity

    def _zero_tensor(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def _current_class_align_lambda(self) -> float:
        lam = float(self.lambda_class_align)
        if lam <= 0.0:
            return 0.0
        epoch = int(self.current_epoch)
        start = max(0, int(self.class_align_start_epoch))
        if epoch < start:
            return 0.0
        return lam

    def _current_ccl_lambda(self) -> float:
        lam = float(self.lambda_ccl)
        if lam <= 0.0:
            return 0.0
        epoch = int(self.current_epoch)
        start = max(0, int(self.ccl_start_epoch))
        if epoch < start:
            return 0.0
        return lam

    def _current_prior_lambda(self) -> float:
        lam = float(self.lambda_prior)
        if lam <= 0.0:
            return 0.0
        epoch = int(self.current_epoch)
        start = max(0, int(self.prior_start_epoch))
        if epoch < start:
            return 0.0
        return lam

    def _prior_interval_loss(self, logits_t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if logits_t is None or logits_t.numel() == 0:
            return self._zero_tensor(), float("nan")
        pos_label = int(self.stage1_positive_label)
        probs_t = torch.softmax(logits_t.float(), dim=1)
        mean_pos_prob = torch.mean(probs_t[:, pos_label])
        prior_min = float(min(self.prior_min, self.prior_max))
        prior_max = float(max(self.prior_min, self.prior_max))
        deficit = torch.relu(torch.as_tensor(prior_min, dtype=mean_pos_prob.dtype, device=mean_pos_prob.device) - mean_pos_prob)
        excess = torch.relu(mean_pos_prob - torch.as_tensor(prior_max, dtype=mean_pos_prob.dtype, device=mean_pos_prob.device))
        if self.prior_loss_type == "l1":
            loss = deficit + excess
        else:
            loss = deficit.pow(2) + excess.pow(2)
        return loss, float(mean_pos_prob.detach().item())

    def _ensure_iahm_loss(self, embed_dim: int) -> Optional[IAHMLoss]:
        cfg = dict(self.stage1_iahm_cfg or {})
        if not bool(cfg.get("enable", False)):
            return None
        if self.stage1_iahm_loss is not None:
            return self.stage1_iahm_loss

        n_classes = int(self.config.get("n_class", 2))
        class_counts = self.stage1_source_class_counts
        if class_counts is None or len(class_counts) < n_classes:
            class_counts = [1 for _ in range(n_classes)]
        else:
            class_counts = list(class_counts[:n_classes])

        space = str(cfg.get("space", "hyperbolic")).strip().lower()
        init_kwargs = dict(
            n_classes=n_classes,
            embed_dim=int(embed_dim),
            class_counts=class_counts,
            r0=float(cfg.get("r0", 1.0)),
            gamma=float(cfg.get("gamma", 1.0)),
            m0=float(cfg.get("m0", 1.0)),
            margin_alpha=float(cfg.get("margin_alpha", 0.25)),
            lambda_r=float(cfg.get("lambda_r", 1.0)),
            lambda_c=float(cfg.get("lambda_c", 0.5)),
            lambda_m=float(cfg.get("lambda_m", 1.0)),
            centroid_momentum=float(cfg.get("centroid_momentum", 0.1)),
        )
        if space == "hyperbolic":
            init_kwargs["curvature"] = float(cfg.get("curvature", -1.0))
            self.stage1_iahm_loss = IAHMLoss(**init_kwargs).to(self.device)
        else:
            self.stage1_iahm_loss = EuclideanIAHMLoss(**init_kwargs).to(self.device)
        self._log_info(
            "IAHM initialized | "
            f"embed_dim={embed_dim} | "
            f"class_counts={class_counts} | "
            f"space={space} | "
            f"curvature={cfg.get('curvature', -1.0)}"
        )
        return self.stage1_iahm_loss

    def _mmd_rbf(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_weights: Optional[torch.Tensor] = None,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        if x is None or y is None or x.numel() == 0 or y.numel() == 0:
            return self._zero_tensor()
        if x.ndim != 2 or y.ndim != 2:
            return self._zero_tensor()

        def _kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a2 = torch.sum(a * a, dim=1, keepdim=True)
            b2 = torch.sum(b * b, dim=1, keepdim=True).T
            dist2 = torch.clamp(a2 + b2 - 2.0 * (a @ b.T), min=0.0)
            denom = 2.0 * max(float(sigma) ** 2, 1e-12)
            return torch.exp(-dist2 / denom)

        k_xx = _kernel(x, x)
        term_xx = torch.mean(k_xx)
        k_xy = _kernel(x, y)
        k_yy = _kernel(y, y)

        if y_weights is None:
            term_xy = torch.mean(k_xy)
            term_yy = torch.mean(k_yy)
        else:
            w = y_weights.reshape(-1).to(y.device, dtype=y.dtype)
            w = torch.clamp(w, min=0.0)
            ws = torch.sum(w)
            if float(ws.detach().item()) <= 0.0:
                w = torch.ones_like(w)
                ws = torch.sum(w)
            w = w / ws
            term_xy = torch.mean(torch.sum(k_xy * w.unsqueeze(0), dim=1))
            term_yy = torch.sum(k_yy * (w.unsqueeze(1) * w.unsqueeze(0)))
        return term_xx + term_yy - 2.0 * term_xy

    def _coral_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if x is None or y is None or x.numel() == 0 or y.numel() == 0:
            return self._zero_tensor()
        if x.ndim != 2 or y.ndim != 2:
            return self._zero_tensor()
        if x.shape[1] != y.shape[1]:
            return self._zero_tensor()

        def _cov(a: torch.Tensor) -> torch.Tensor:
            n = int(a.shape[0])
            d = int(a.shape[1])
            if n <= 1:
                return torch.eye(d, device=a.device, dtype=a.dtype)
            mean = torch.mean(a, dim=0, keepdim=True)
            centered = a - mean
            cov = (centered.T @ centered) / float(max(n - 1, 1))
            cov = cov + float(eps) * torch.eye(d, device=a.device, dtype=a.dtype)
            return cov

        cov_x = _cov(x)
        cov_y = _cov(y)
        d = int(x.shape[1])
        return torch.sum((cov_x - cov_y) ** 2) / float(4 * d * d)

    def _uot_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        eps: Optional[float] = None,
        tau_source: Optional[float] = None,
        tau_target: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> torch.Tensor:
        if x is None or y is None or x.numel() == 0 or y.numel() == 0:
            return self._zero_tensor()
        if x.ndim != 2 or y.ndim != 2 or int(x.shape[1]) != int(y.shape[1]):
            return self._zero_tensor()

        x = x.float()
        y = y.float()
        n = int(x.shape[0])
        m = int(y.shape[0])
        if n == 0 or m == 0:
            return self._zero_tensor()

        # Keep the transport geometry bounded; without this, batch feature
        # scales can make the unbalanced Sinkhorn updates blow up early.
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        eps_val = float(self.uot_eps if eps is None else eps)
        tau_s = float(self.uot_tau_source if tau_source is None else tau_source)
        tau_t = float(self.uot_tau_target if tau_target is None else tau_target)
        n_iter = int(self.uot_max_iter if max_iter is None else max_iter)
        eps_val = max(eps_val, 1.0e-6)
        tau_s = max(tau_s, 1.0e-6)
        tau_t = max(tau_t, 1.0e-6)
        n_iter = max(n_iter, 1)
        tiny = 1.0e-8

        x_norm = torch.sum(x * x, dim=1, keepdim=True)
        y_norm = torch.sum(y * y, dim=1, keepdim=True)
        cost = torch.clamp(x_norm + y_norm.T - 2.0 * (x @ y.T), min=0.0)
        cost = cost / float(max(int(x.shape[1]), 1))

        a = torch.full((n,), 1.0 / float(n), device=x.device, dtype=x.dtype)
        b = torch.full((m,), 1.0 / float(m), device=y.device, dtype=y.dtype)

        kernel = torch.exp(-cost / eps_val).clamp_min(tiny)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        p_s = tau_s / (tau_s + eps_val)
        p_t = tau_t / (tau_t + eps_val)

        for _ in range(n_iter):
            kv = torch.clamp(kernel @ v, min=tiny)
            ratio_u = torch.clamp(a / kv, min=tiny, max=1.0e6)
            u = torch.pow(ratio_u, p_s).clamp(min=tiny, max=1.0e6)
            ktu = torch.clamp(kernel.T @ u, min=tiny)
            ratio_v = torch.clamp(b / ktu, min=tiny, max=1.0e6)
            v = torch.pow(ratio_v, p_t).clamp(min=tiny, max=1.0e6)

        gamma = ((u.unsqueeze(1) * kernel) * v.unsqueeze(0)).clamp(min=tiny, max=1.0e6)
        row_mass = torch.clamp(gamma.sum(dim=1), min=tiny)
        col_mass = torch.clamp(gamma.sum(dim=0), min=tiny)

        transport_cost = torch.sum(gamma * cost)
        kl_row = torch.sum(row_mass * (torch.log(row_mass) - torch.log(a)) - row_mass + a)
        kl_col = torch.sum(col_mass * (torch.log(col_mass) - torch.log(b)) - col_mass + b)
        return transport_cost + tau_s * kl_row + tau_t * kl_col

    def _pairwise_similarity_preservation_loss(
        self,
        feat_before: Optional[torch.Tensor],
        feat_after: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            feat_before is None
            or feat_after is None
            or feat_before.numel() == 0
            or feat_after.numel() == 0
            or feat_before.ndim != 2
            or feat_after.ndim != 2
            or feat_before.shape != feat_after.shape
            or int(feat_before.shape[0]) <= 1
        ):
            return self._zero_tensor()

        fb = F.normalize(feat_before.float(), dim=1)
        fa = F.normalize(feat_after.float(), dim=1)
        sim_before = fb @ fb.T
        sim_after = fa @ fa.T
        return F.mse_loss(sim_after, sim_before.detach())

    def _feature_identity_loss(
        self,
        feat_before: Optional[torch.Tensor],
        feat_after: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            feat_before is None
            or feat_after is None
            or feat_before.numel() == 0
            or feat_after.numel() == 0
            or feat_before.shape != feat_after.shape
        ):
            return self._zero_tensor()
        return F.mse_loss(feat_after.float(), feat_before.detach().float())

    def _lmmd_loss(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
        source_labels: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        if (
            source_feat is None
            or target_feat is None
            or source_labels is None
            or target_probs is None
            or source_feat.numel() == 0
            or target_feat.numel() == 0
            or source_labels.numel() == 0
            or target_probs.numel() == 0
        ):
            return self._zero_tensor()
        if source_feat.ndim != 2 or target_feat.ndim != 2 or target_probs.ndim != 2:
            return self._zero_tensor()
        if source_feat.shape[1] != target_feat.shape[1]:
            return self._zero_tensor()

        n_classes = int(self.config.get("n_class", target_probs.shape[1]))
        if int(target_probs.shape[1]) != n_classes:
            return self._zero_tensor()

        total = torch.cat([source_feat, target_feat], dim=0)
        total_norm = torch.sum(total * total, dim=1, keepdim=True)
        dist2 = torch.clamp(total_norm + total_norm.T - 2.0 * (total @ total.T), min=0.0)

        fix_sigma = self.lmmd_fix_sigma
        if fix_sigma in (None, "", "None", "none", "null", "NULL"):
            n_total = int(total.shape[0])
            denom = max(n_total * n_total - n_total, 1)
            bandwidth = torch.sum(dist2.detach()) / float(denom)
        else:
            bandwidth = torch.as_tensor(float(fix_sigma), device=total.device, dtype=total.dtype)
        bandwidth = torch.clamp(bandwidth, min=1.0e-6)
        bandwidth = bandwidth / float(self.lmmd_kernel_mul ** (self.lmmd_kernel_num // 2))

        kernels = torch.zeros_like(dist2)
        for i in range(max(1, int(self.lmmd_kernel_num))):
            bw = torch.clamp(
                bandwidth * float(self.lmmd_kernel_mul ** i),
                min=1.0e-6,
            )
            kernels = kernels + torch.exp(-dist2 / bw)

        ns = int(source_feat.shape[0])
        k_ss = kernels[:ns, :ns]
        k_tt = kernels[ns:, ns:]
        k_st = kernels[:ns, ns:]

        source_onehot = F.one_hot(source_labels.long(), num_classes=n_classes).float()
        source_weights = source_onehot / torch.clamp(source_onehot.sum(dim=0, keepdim=True), min=1.0)

        if bool(self.lmmd_use_soft_target):
            target_weights = target_probs.float()
        else:
            target_hard = torch.argmax(target_probs, dim=1)
            target_weights = F.one_hot(target_hard.long(), num_classes=n_classes).float()
        target_weights = target_weights / torch.clamp(target_weights.sum(dim=0, keepdim=True), min=1.0)

        loss = self._zero_tensor()
        for k in range(n_classes):
            ws = source_weights[:, k]
            wt = target_weights[:, k]
            if float(ws.sum().detach().item()) <= 0.0 or float(wt.sum().detach().item()) <= 0.0:
                continue
            w_ss = ws.unsqueeze(1) * ws.unsqueeze(0)
            w_tt = wt.unsqueeze(1) * wt.unsqueeze(0)
            w_st = ws.unsqueeze(1) * wt.unsqueeze(0)
            loss = loss + torch.sum(w_ss * k_ss) + torch.sum(w_tt * k_tt) - 2.0 * torch.sum(w_st * k_st)
        return loss / float(max(1, n_classes))

    def _ccl_loss(
        self,
        target_probs: torch.Tensor,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:
        if target_probs is None or target_probs.numel() == 0 or target_probs.ndim != 2:
            return self._zero_tensor()
        batch_size = int(target_probs.shape[0])
        n_classes = int(target_probs.shape[1])
        if batch_size <= 0 or n_classes <= 0:
            return self._zero_tensor()

        probs = torch.clamp(target_probs.float(), min=float(eps), max=1.0)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        weights = 1.0 + torch.exp(-entropy)
        weights = float(batch_size) * weights / torch.clamp(torch.sum(weights), min=float(eps))
        weighted_probs = probs * weights.unsqueeze(1)
        ccm = probs.T @ weighted_probs
        ccm = ccm / torch.clamp(ccm.sum(dim=1, keepdim=True), min=float(eps))
        return (torch.sum(ccm) - torch.trace(ccm)) / float(max(1, n_classes))

    def _sample_rpt_batch(
        self,
        *,
        source_labels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.stage1_rpt_augmentor is None:
            return None, None, None

        n_cfg = self.rpt_synth_per_batch_cfg
        if n_cfg in (None, "", "None", "none", "null", "NULL"):
            n_pos = int((source_labels == int(self.stage1_positive_label)).sum().detach().item())
            n_synth = max(1, n_pos)
        else:
            n_synth = max(1, int(n_cfg))

        synth = self.stage1_rpt_augmentor.sample(
            n=int(n_synth),
            seed=int(self.current_epoch * 100000 + self.stage1_global_step),
        )
        x_rpt_np = np.asarray(synth.get("trials", np.empty((0,))), dtype=np.float32)
        if x_rpt_np.ndim != 3 or int(x_rpt_np.shape[0]) <= 0:
            return None, None, None

        x_rpt = torch.from_numpy(x_rpt_np).to(self.device)
        if len(x_rpt.shape) == 3:
            x_rpt = x_rpt.unsqueeze(1)
        if not torch.isfinite(x_rpt).all():
            return None, None, None
        y_np = np.asarray(
            synth.get(
                "labels",
                np.full((int(x_rpt_np.shape[0]),), int(self.stage1_positive_label), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        y_rpt = torch.from_numpy(y_np).to(self.device, dtype=torch.long)
        w_np = np.asarray(synth.get("weights", np.ones((x_rpt_np.shape[0],))), dtype=np.float32)
        w_rpt = torch.from_numpy(w_np).to(self.device)
        return x_rpt, y_rpt, w_rpt

    def _stage1_aux_losses(
        self,
        *,
        source_embed: torch.Tensor,
        source_labels: torch.Tensor,
        source_domain_id: Optional[torch.Tensor],
        target_x: Optional[torch.Tensor],
        training: bool,
        synth_embed: Optional[torch.Tensor] = None,
        synth_labels: Optional[torch.Tensor] = None,
        synth_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """
        Optional Stage-1 auxiliary losses:
        - class-conditional alignment (MMD) using source/target and RPT-Aug synth positives
        - IAHM on Lorentz embeddings (source + optional synth positives)
        """
        if not self._stage1_has_aux():
            return self._zero_tensor(), OrderedDict()

        aux_items = OrderedDict()
        iahm_cfg = dict(self.stage1_iahm_cfg or {})
        curv = float(iahm_cfg.get("curvature", -1.0))
        space = str(iahm_cfg.get("space", "hyperbolic")).strip().lower()
        normalize_mode = str(iahm_cfg.get("input_normalize", "none")).strip().lower()

        def _prep_iahm_input(x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            if normalize_mode == "l2":
                x = F.normalize(x, p=2, dim=1)
            if space == "hyperbolic":
                return euclidean_to_lorentz(x, K=curv)
            return x

        z_src = _prep_iahm_input(source_embed)

        # Target embeddings for negative/background alignment branch.
        z_tgt = None
        target_logits = None
        target_probs = None
        target_content_pre = None
        target_content_post = None
        if target_x is not None:
            if len(target_x.shape) == 3:
                target_x = target_x.unsqueeze(1)
            self._set_model_domain_context(domain_ids=None, use_target_stats=True)
            target_logits, target_extras = self._forward_unpack(target_x)
            target_embed = self._select_hyperbolic_input(target_logits, target_extras)
            z_tgt = _prep_iahm_input(target_embed)
            target_probs = torch.softmax(target_logits.float(), dim=1)
            if isinstance(target_extras, dict):
                target_content_pre = target_extras.get("lsa_content_pre", None)
                target_content_post = target_extras.get("lsa_content_post", None)

        z_rpt = _prep_iahm_input(synth_embed) if synth_embed is not None and synth_embed.numel() > 0 else None
        w_rpt = synth_weights

        # Alignment loss
        loss_align = self._zero_tensor()
        if float(self.lambda_align) > 0.0:
            if self.alignment_loss_name not in ("mmd", "lmmd", "global_mmd", "per_subject_mmd", "global_coral", "per_subject_coral", "uot"):
                if not self._alignment_warned:
                    self._log_info(
                        f"Unsupported alignment_loss={self.alignment_loss_name}, fallback to mmd."
                    )
                    self._alignment_warned = True
            if self.alignment_loss_name == "lmmd":
                if (
                    z_tgt is not None
                    and z_tgt.numel() > 0
                    and z_src.numel() > 0
                    and target_probs is not None
                    and target_probs.numel() > 0
                ):
                    loss_align = self._lmmd_loss(
                        source_feat=z_src,
                        target_feat=z_tgt,
                        source_labels=source_labels,
                        target_probs=target_probs,
                    )
                aux_items["align_lmmd"] = float(loss_align.detach().item())
                aux_items["align"] = float(loss_align.detach().item())
            elif self.alignment_loss_name == "global_mmd":
                if z_tgt is not None and z_tgt.numel() > 0 and z_src.numel() > 0:
                    loss_align = self._mmd_rbf(z_src, z_tgt, y_weights=None)
                aux_items["align_global"] = float(loss_align.detach().item())
                aux_items["align"] = float(loss_align.detach().item())
            elif self.alignment_loss_name == "global_coral":
                if z_tgt is not None and z_tgt.numel() > 0 and z_src.numel() > 0:
                    loss_align = self._coral_loss(z_src, z_tgt)
                aux_items["align_global_coral"] = float(loss_align.detach().item())
                aux_items["align"] = float(loss_align.detach().item())
            elif self.alignment_loss_name == "uot":
                if z_tgt is not None and z_tgt.numel() > 0 and z_src.numel() > 0:
                    loss_align = self._uot_loss(z_src, z_tgt)
                aux_items["align_uot"] = float(loss_align.detach().item())
                aux_items["align"] = float(loss_align.detach().item())
            elif self.alignment_loss_name == "per_subject_mmd":
                if (
                    z_tgt is not None
                    and z_tgt.numel() > 0
                    and z_src.numel() > 0
                    and source_domain_id is not None
                    and source_domain_id.numel() == z_src.shape[0]
                ):
                    unique_ids = torch.unique(source_domain_id.detach())
                    losses = []
                    for domain_id in unique_ids:
                        mask = (source_domain_id == domain_id)
                        z_src_k = z_src[mask]
                        if z_src_k.numel() == 0:
                            continue
                        losses.append(self._mmd_rbf(z_src_k, z_tgt, y_weights=None))
                    if losses:
                        loss_align = torch.stack(losses).mean()
                        aux_items["align_n_domains"] = float(len(losses))
                    else:
                        loss_align = self._zero_tensor()
                elif z_tgt is not None and z_tgt.numel() > 0 and z_src.numel() > 0:
                    # Fallback when domain ids are unavailable.
                    loss_align = self._mmd_rbf(z_src, z_tgt, y_weights=None)
                    aux_items["align_fallback_global"] = 1.0
                aux_items["align_per_subject"] = float(loss_align.detach().item())
                aux_items["align"] = float(loss_align.detach().item())
            elif self.alignment_loss_name == "per_subject_coral":
                if (
                    z_tgt is not None
                    and z_tgt.numel() > 0
                    and z_src.numel() > 0
                    and source_domain_id is not None
                    and source_domain_id.numel() == z_src.shape[0]
                ):
                    unique_ids = torch.unique(source_domain_id.detach())
                    losses = []
                    for domain_id in unique_ids:
                        mask = (source_domain_id == domain_id)
                        z_src_k = z_src[mask]
                        if z_src_k.numel() == 0:
                            continue
                        losses.append(self._coral_loss(z_src_k, z_tgt))
                    if losses:
                        loss_align = torch.stack(losses).mean()
                        aux_items["align_n_domains"] = float(len(losses))
                    else:
                        loss_align = self._zero_tensor()
                elif z_tgt is not None and z_tgt.numel() > 0 and z_src.numel() > 0:
                    loss_align = self._coral_loss(z_src, z_tgt)
                    aux_items["align_fallback_global"] = 1.0
                aux_items["align_per_subject_coral"] = float(loss_align.detach().item())
                aux_items["align"] = float(loss_align.detach().item())
            else:
                pos_mask = (source_labels == int(self.stage1_positive_label))
                neg_mask = (source_labels == int(self.stage1_background_label))
                z_s_pos = z_src[pos_mask]
                z_s_neg = z_src[neg_mask]

                loss_align_pos = self._zero_tensor()
                if z_rpt is not None and z_rpt.numel() > 0 and z_s_pos.numel() > 0:
                    loss_align_pos = self._mmd_rbf(z_s_pos, z_rpt, y_weights=w_rpt)

                loss_align_neg = self._zero_tensor()
                if z_tgt is not None and z_tgt.numel() > 0 and z_s_neg.numel() > 0:
                    loss_align_neg = self._mmd_rbf(z_s_neg, z_tgt, y_weights=None)

                loss_align = loss_align_pos + loss_align_neg
                aux_items["align_pos"] = float(loss_align_pos.detach().item())
                aux_items["align_neg"] = float(loss_align_neg.detach().item())
                aux_items["align"] = float(loss_align.detach().item())

        # IAHM loss (source + synthetic positive)
        loss_iahm = self._zero_tensor()
        iahm_embed_dim = int(source_embed.shape[1])
        iahm_fn = self._ensure_iahm_loss(embed_dim=iahm_embed_dim)
        if iahm_fn is not None:
            if z_rpt is not None and z_rpt.numel() > 0 and synth_labels is not None:
                z_all = torch.cat([z_src, z_rpt], dim=0)
                y_all = torch.cat([source_labels.long(), synth_labels.long()], dim=0)
            else:
                z_all = z_src
                y_all = source_labels.long()
            loss_iahm, iahm_items = iahm_fn(z_all, y_all, update_centroids=bool(training))
            aux_items["iahm"] = float(loss_iahm.detach().item())
            for k, v in iahm_items.items():
                aux_items[f"iahm_{k}"] = float(v)

        loss_class_align = self._zero_tensor()
        lambda_class_align_eff = self._current_class_align_lambda()
        if (
            lambda_class_align_eff > 0.0
            and target_logits is not None
            and target_logits.numel() > 0
            and z_tgt is not None
            and z_tgt.numel() > 0
            and z_src.numel() > 0
        ):
            probs_t = torch.softmax(target_logits.detach().float(), dim=1)
            conf_t, hard_t = torch.max(probs_t, dim=1)
            conf_mask = conf_t > float(self.class_align_conf_thresh)
            n_conf = int(conf_mask.sum().detach().item())
            aux_items["class_align_n_conf"] = float(n_conf)
            aux_items["lambda_class_align_eff"] = float(lambda_class_align_eff)
            if n_conf >= max(1, int(self.class_align_min_conf_samples)):
                z_tgt_conf = z_tgt[conf_mask]
                probs_conf = probs_t[conf_mask]

                pos_label = int(self.stage1_positive_label)
                bg_label = int(self.stage1_background_label)
                pos_mask = (source_labels == pos_label)
                neg_mask = (source_labels == bg_label)
                z_s_pos = z_src[pos_mask]
                z_s_neg = z_src[neg_mask]

                if self.class_align_use_soft_weights:
                    w_pos = probs_conf[:, pos_label]
                    w_bg = probs_conf[:, bg_label]
                else:
                    hard_conf = hard_t[conf_mask]
                    w_pos = (hard_conf == pos_label).float()
                    w_bg = (hard_conf == bg_label).float()

                loss_cc_pos = self._zero_tensor()
                if z_s_pos.numel() > 0 and float(w_pos.sum().detach().item()) > 0.0:
                    loss_cc_pos = self._mmd_rbf(z_s_pos, z_tgt_conf, y_weights=w_pos)

                loss_cc_neg = self._zero_tensor()
                if z_s_neg.numel() > 0 and float(w_bg.sum().detach().item()) > 0.0:
                    loss_cc_neg = self._mmd_rbf(z_s_neg, z_tgt_conf, y_weights=w_bg)

                loss_class_align = loss_cc_pos + loss_cc_neg
                aux_items["class_align_pos"] = float(loss_cc_pos.detach().item())
                aux_items["class_align_neg"] = float(loss_cc_neg.detach().item())
                aux_items["class_align"] = float(loss_class_align.detach().item())

        loss_prior = self._zero_tensor()
        lambda_prior_eff = self._current_prior_lambda()
        if lambda_prior_eff > 0.0 and target_logits is not None and target_logits.numel() > 0:
            loss_prior, mean_pos_prob = self._prior_interval_loss(target_logits)
            aux_items["lambda_prior_eff"] = float(lambda_prior_eff)
            aux_items["prior_mean_pos_prob"] = float(mean_pos_prob)
            aux_items["prior"] = float(loss_prior.detach().item())

        loss_ccl = self._zero_tensor()
        lambda_ccl_eff = self._current_ccl_lambda()
        if lambda_ccl_eff > 0.0 and target_probs is not None and target_probs.numel() > 0:
            loss_ccl = self._ccl_loss(target_probs)
            aux_items["lambda_ccl_eff"] = float(lambda_ccl_eff)
            aux_items["ccl"] = float(loss_ccl.detach().item())

        loss_lsa_content = self._zero_tensor()
        if float(self.lsa_content_lambda) > 0.0 and target_content_pre is not None and target_content_post is not None:
            loss_lsa_content = self._pairwise_similarity_preservation_loss(
                target_content_pre,
                target_content_post,
            )
            aux_items["lsa_content"] = float(loss_lsa_content.detach().item())
            aux_items["lambda_lsa_content"] = float(self.lsa_content_lambda)

        loss_lsa_identity = self._zero_tensor()
        if float(self.lsa_identity_lambda) > 0.0 and target_content_pre is not None and target_content_post is not None:
            loss_lsa_identity = self._feature_identity_loss(
                target_content_pre,
                target_content_post,
            )
            aux_items["lsa_identity"] = float(loss_lsa_identity.detach().item())
            aux_items["lambda_lsa_identity"] = float(self.lsa_identity_lambda)

        lambda_iahm = float((self.stage1_iahm_cfg or {}).get("lambda_total", 1.0))
        aux_total = (
            float(self.lambda_align) * loss_align
            + lambda_iahm * loss_iahm
            + float(lambda_class_align_eff) * loss_class_align
            + float(lambda_ccl_eff) * loss_ccl
            + float(lambda_prior_eff) * loss_prior
            + float(self.lsa_content_lambda) * loss_lsa_content
            + float(self.lsa_identity_lambda) * loss_lsa_identity
        )
        aux_items["aux_total"] = float(aux_total.detach().item())
        return aux_total, aux_items

    # ---------- epoch core ----------
    def _run_epoch(self, dataloader, *, training: bool, loss_fn: Callable, profile_runtime: bool = False):
        """
        Run one training/eval epoch and return:
          [
            (loss_avgs_list, loss_names_list),
            AUC, BA, F1, TPR, FPR
          ]
        """
        self.model.train() if training else self.model.eval()
        collect_runtime = bool((not training) and profile_runtime)
        pre_times_ms, model_times_ms, e2e_times_ms = [], [], []

        if collect_runtime and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        loss_names: List[str] = []
        loss_sums: List[float] = []
        n_batches = 0
        all_probs, all_preds, all_targets = [], [], []
        maybe_no_grad = torch.enable_grad if training else torch.no_grad
        sample_count = 0

        with maybe_no_grad():
            for batch in dataloader:
                self.stage1_global_step += 1
                n_batches += 1
                t0 = _t.perf_counter()

                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Batch must be (x, y) or (x, y, target_x).")
                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)
                target_x = None
                source_domain_id = None
                if len(batch) == 3 and batch[2] is not None:
                    candidate = batch[2]
                    if isinstance(candidate, torch.Tensor) and candidate.ndim == 1 and candidate.shape[0] == y.shape[0]:
                        source_domain_id = candidate.to(self.device, non_blocking=True)
                    else:
                        target_x = candidate.to(self.device, non_blocking=True)
                elif len(batch) >= 4:
                    if batch[2] is not None:
                        target_x = batch[2].to(self.device, non_blocking=True)
                    if batch[3] is not None:
                        source_domain_id = batch[3].to(self.device, non_blocking=True)
                if training:
                    self._maybe_log_domain_debug(source_domain_id)
                sample_count += x.size(0)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)

                if collect_runtime and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = _t.perf_counter()

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    x_rpt = y_rpt = w_rpt = None
                    if training and self.rpt_inject_to_ce:
                        x_rpt, y_rpt, w_rpt = self._sample_rpt_batch(source_labels=y)

                    eval_use_target = (not training) and (source_domain_id is None) and bool(
                        getattr(self, "_eval_use_target_stats", False)
                    )
                    self._set_model_domain_context(
                        domain_ids=source_domain_id,
                        use_target_stats=eval_use_target,
                    )
                    logits, extras = self._forward_unpack(x)
                    logits_ce = logits
                    y_ce = y
                    synth_embed = None
                    if x_rpt is not None and y_rpt is not None:
                        self._set_model_domain_context(domain_ids=None, use_target_stats=False)
                        synth_logits, synth_extras = self._forward_unpack(x_rpt)
                        logits_ce = torch.cat([logits, synth_logits], dim=0)
                        y_ce = torch.cat([y, y_rpt], dim=0)
                        synth_embed = self._select_hyperbolic_input(synth_logits, synth_extras)

                    loss_total, items = loss_fn(logits_ce, y_ce, extras)
                    if training and self._stage1_has_aux():
                        source_embed = self._select_hyperbolic_input(logits, extras)
                        aux_loss, aux_items = self._stage1_aux_losses(
                            source_embed=source_embed,
                            source_labels=y,
                            source_domain_id=source_domain_id,
                            target_x=target_x,
                            training=training,
                            synth_embed=synth_embed,
                            synth_labels=y_rpt,
                            synth_weights=w_rpt,
                        )
                        loss_total = loss_total + aux_loss
                        merged = OrderedDict(items)
                        for k, v in aux_items.items():
                            merged[k] = float(v)
                        items = merged
                    if (
                        training
                        and target_x is not None
                        and float(self.aux_ts_lambda_align) > 0.0
                        and isinstance(extras, dict)
                    ):
                        src_ts = extras.get("ts_features", None)
                        if isinstance(src_ts, torch.Tensor) and src_ts.numel() > 0:
                            target_x_aux = target_x.unsqueeze(1) if len(target_x.shape) == 3 else target_x
                            self._set_model_domain_context(domain_ids=None, use_target_stats=True)
                            _, target_extras_aux = self._forward_unpack(target_x_aux)
                            if isinstance(target_extras_aux, dict):
                                tgt_ts = target_extras_aux.get("ts_features", None)
                                if isinstance(tgt_ts, torch.Tensor) and tgt_ts.numel() > 0:
                                    aux_ts_align = self._mmd_rbf(src_ts.float(), tgt_ts.float(), y_weights=None)
                                    loss_total = loss_total + float(self.aux_ts_lambda_align) * aux_ts_align
                                    merged = OrderedDict(items)
                                    merged["aux_ts_align_global"] = float(aux_ts_align.detach().item())
                                    items = merged
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(1)

                if collect_runtime and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = _t.perf_counter()

                if collect_runtime:
                    pre_times_ms.append((t1 - t0) * 1000.0)
                    model_times_ms.append((t2 - t1) * 1000.0)
                    e2e_times_ms.append((t2 - t0) * 1000.0)

                if training:
                    self.scaler.scale(loss_total).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                if not loss_names:
                    loss_names = ["Total"] + list(items.keys())
                    loss_sums = [0.0 for _ in loss_names]

                loss_sums[0] += float(loss_total.detach().item())
                for i, k in enumerate(items.keys(), start=1):
                    loss_sums[i] += float(items[k])

                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_targets.append(y.detach().cpu())
                MemoryManager.cleanup_tensors(x, y, logits, source_domain_id, x_rpt, y_rpt)

        probabilities = torch.cat(all_probs) if len(all_probs) else torch.empty(0)
        predictions = torch.cat(all_preds) if len(all_preds) else torch.empty(0, dtype=torch.long)
        targets = torch.cat(all_targets) if len(all_targets) else torch.empty(0, dtype=torch.long)

        if len(all_probs):
            ba, _, tpr, fpr, auc = calculate_metrics(targets, predictions, y_prob=probabilities)
            f1 = cal_F1_score(targets, predictions)
            AUC = auc.numpy().round(4) if auc is not None else None
            BA = ba.numpy().round(4)
            F1 = f1.numpy().round(4)
            TPR = tpr.numpy().round(4)
            FPR = fpr.numpy().round(4)
        else:
            AUC = BA = F1 = TPR = FPR = None

        denom = max(n_batches, 1)
        loss_avgs = [s / denom for s in loss_sums] if loss_sums else [0.0]
        metrics = [(loss_avgs, loss_names), AUC, BA, F1, TPR, FPR]

        if len(all_probs):
            MemoryManager.cleanup_tensors(probabilities, predictions, targets)

        if collect_runtime and len(e2e_times_ms) > 0:
            def _p50(a):
                b = sorted(a)
                return b[len(b) // 2]

            def _p95(a):
                b = sorted(a)
                return b[int(len(b) * 0.95)]

            def _jitter(a):
                return _p95(a) - _p50(a)

            p50_model = _p50(model_times_ms)
            p95_model = _p95(model_times_ms)
            jit_model = _jitter(model_times_ms)
            p50_e2e = _p50(e2e_times_ms)
            p95_e2e = _p95(e2e_times_ms)
            jit_e2e = _jitter(e2e_times_ms)

            total_samples = int(sample_count)
            total_time_s = sum(e2e_times_ms) / 1000.0
            throughput = (total_samples / total_time_s) if total_time_s > 1e-9 else float("nan")

            peak_mb = float("nan")
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated(device=self.device) / 1e6

            ddl = float(self.rt_deadline_ms)
            miss_rate = sum(1 for t in e2e_times_ms if t > ddl) / len(e2e_times_ms)

            self.last_runtime_stats = {
                "preproc_p50_ms": _p50(pre_times_ms),
                "preproc_p95_ms": _p95(pre_times_ms),
                "model_p50_ms": p50_model,
                "model_p95_ms": p95_model,
                "model_jitter_ms": jit_model,
                "e2e_p50_ms": p50_e2e,
                "e2e_p95_ms": p95_e2e,
                "e2e_jitter_ms": jit_e2e,
                "throughput_samples_per_s": throughput,
                "peak_cuda_mem_mb": peak_mb,
                "deadline_miss_rate": miss_rate,
                "deadline_ms": ddl,
                "num_samples": total_samples,
                "num_batches": len(e2e_times_ms),
            }
        else:
            self.last_runtime_stats = None
        return metrics

    def _cross_entropy(self, logits, y):
        if self.class_weights is None:
            return F.cross_entropy(logits, y)
        return F.cross_entropy(logits, y, weight=self.class_weights)

    def _set_model_domain_context(
        self,
        *,
        domain_ids: Optional[torch.Tensor] = None,
        use_target_stats: bool = False,
    ) -> None:
        setter = getattr(self.model, "set_domain_context", None)
        if callable(setter):
            setter(domain_ids=domain_ids, use_target_stats=use_target_stats)

    def _ensure_prototypes(self, feature_dim: int) -> None:
        if (
            self.prototype_vectors is not None
            and self.prototype_initialized is not None
            and int(self.prototype_vectors.shape[1]) == int(feature_dim)
        ):
            return
        n_class = int(self.config.get("n_class", 2))
        self.prototype_vectors = torch.zeros(
            (n_class, int(feature_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        self.prototype_initialized = torch.zeros((n_class,), dtype=torch.bool, device=self.device)

    def _prototype_class_weight(self, class_id: int) -> float:
        if int(class_id) == int(self.prototype_positive_label):
            return float(self.prototype_positive_weight)
        if int(class_id) == int(self.prototype_background_label):
            return float(self.prototype_background_weight)
        return 1.0

    def _prototype_regularization(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, OrderedDict]:
        if (
            (not self.prototype_enable)
            or features is None
            or labels is None
            or features.numel() == 0
            or labels.numel() == 0
            or (
                float(self.prototype_lambda) <= 0.0
                and float(self.prototype_separation_lambda) <= 0.0
            )
        ):
            return self._zero_tensor(), OrderedDict()

        features = features.float()
        labels = labels.long()
        self._ensure_prototypes(int(features.shape[1]))
        assert self.prototype_vectors is not None
        assert self.prototype_initialized is not None

        pull_acc = self._zero_tensor()
        pull_weight_sum = 0.0
        batch_means: Dict[int, torch.Tensor] = {}

        for cls in torch.unique(labels).tolist():
            cls = int(cls)
            mask = labels == cls
            if int(mask.sum().item()) <= 0:
                continue
            cls_feats = features[mask]
            batch_mean = cls_feats.mean(dim=0).detach()
            batch_means[cls] = batch_mean

            if float(self.prototype_lambda) > 0.0:
                if bool(self.prototype_initialized[cls].item()):
                    proto = self.prototype_vectors[cls].detach()
                else:
                    proto = batch_mean
                cls_pull = torch.mean(torch.sum((cls_feats - proto.unsqueeze(0)) ** 2, dim=1))
                cls_weight = float(self._prototype_class_weight(cls))
                pull_acc = pull_acc + cls_weight * cls_pull
                pull_weight_sum += cls_weight

        pull_loss = self._zero_tensor()
        if pull_weight_sum > 0.0:
            pull_loss = pull_acc / float(pull_weight_sum)

        sep_loss = self._zero_tensor()
        if float(self.prototype_separation_lambda) > 0.0:
            pos_cls = int(self.prototype_positive_label)
            bg_cls = int(self.prototype_background_label)
            pos_proto = batch_means.get(pos_cls, None)
            bg_proto = batch_means.get(bg_cls, None)
            if pos_proto is None and bool(self.prototype_initialized[pos_cls].item()):
                pos_proto = self.prototype_vectors[pos_cls].detach()
            if bg_proto is None and bool(self.prototype_initialized[bg_cls].item()):
                bg_proto = self.prototype_vectors[bg_cls].detach()
            if pos_proto is not None and bg_proto is not None:
                proto_dist = torch.norm(pos_proto - bg_proto, p=2)
                sep_loss = torch.relu(float(self.prototype_separation_margin) - proto_dist) ** 2

        with torch.no_grad():
            mu = float(self.prototype_momentum)
            for cls, batch_mean in batch_means.items():
                if bool(self.prototype_initialized[cls].item()):
                    self.prototype_vectors[cls].mul_(mu).add_((1.0 - mu) * batch_mean)
                else:
                    self.prototype_vectors[cls].copy_(batch_mean)
                    self.prototype_initialized[cls] = True

        total = float(self.prototype_lambda) * pull_loss + float(self.prototype_separation_lambda) * sep_loss
        items = OrderedDict()
        if float(self.prototype_lambda) > 0.0:
            items["proto_pull"] = float(pull_loss.detach().item())
        if float(self.prototype_separation_lambda) > 0.0:
            items["proto_sep"] = float(sep_loss.detach().item())
        return total, items

    def _positive_distribution_regularization(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        if (
            (not self.posdist_enable)
            or float(self.posdist_lambda) <= 0.0
            or features is None
            or labels is None
            or features.numel() == 0
            or labels.numel() == 0
            or int(self.current_epoch) < max(0, int(self.posdist_start_epoch))
        ):
            return self._zero_tensor(), OrderedDict()

        mask = labels.long() == int(self.posdist_positive_label)
        if int(mask.sum().item()) <= 1:
            return self._zero_tensor(), OrderedDict()

        pos_feat = features[mask].float()
        batch_mean = pos_feat.mean(dim=0)
        batch_var = torch.var(pos_feat, dim=0, unbiased=False)

        if (
            self.posdist_mean is None
            or self.posdist_var is None
            or int(self.posdist_mean.numel()) != int(batch_mean.numel())
        ):
            self.posdist_mean = torch.zeros_like(batch_mean)
            self.posdist_var = torch.zeros_like(batch_var)
            self.posdist_initialized = False

        if self.posdist_initialized:
            ref_mean = self.posdist_mean.detach()
            ref_var = self.posdist_var.detach()
            mean_loss = torch.mean((batch_mean - ref_mean) ** 2)
            var_loss = torch.mean((batch_var - ref_var) ** 2)
        else:
            mean_loss = self._zero_tensor()
            var_loss = self._zero_tensor()

        with torch.no_grad():
            mu = float(self.posdist_momentum)
            if self.posdist_initialized:
                self.posdist_mean.mul_(mu).add_((1.0 - mu) * batch_mean.detach())
                self.posdist_var.mul_(mu).add_((1.0 - mu) * batch_var.detach())
            else:
                self.posdist_mean.copy_(batch_mean.detach())
                self.posdist_var.copy_(batch_var.detach())
                self.posdist_initialized = True

        total = float(self.posdist_lambda) * (mean_loss + float(self.posdist_var_weight) * var_loss)
        items = OrderedDict(
            [
                ("posdist_mean", float(mean_loss.detach().item())),
                ("posdist_var", float(var_loss.detach().item())),
            ]
        )
        return total, items

    def loss_ce(self, logits, y, extras):
        """Pure CE"""
        ce = self._cross_entropy(logits, y)
        total = ce
        items = OrderedDict([("ce", float(ce.detach().item()))])

        if isinstance(extras, dict):
            flat_logits = extras.get("flat_logits", None)
            ts_logits = extras.get("ts_logits", None)

            if isinstance(flat_logits, torch.Tensor) and self.dual_head_lambda_flat_ce > 0.0:
                ce_flat = self._cross_entropy(flat_logits, y)
                total = total + self.dual_head_lambda_flat_ce * ce_flat
                items["ce_flat"] = float(ce_flat.detach().item())

            if isinstance(ts_logits, torch.Tensor) and self.dual_head_lambda_ts_ce > 0.0:
                ce_ts = self._cross_entropy(ts_logits, y)
                total = total + self.dual_head_lambda_ts_ce * ce_ts
                items["ce_ts"] = float(ce_ts.detach().item())

            feat = extras.get("features", None)
            if (
                isinstance(feat, torch.Tensor)
                and self.prototype_enable
                and self.model.training
            ):
                proto_loss, proto_items = self._prototype_regularization(feat, y)
                total = total + proto_loss
                for k, v in proto_items.items():
                    items[k] = float(v)

            if (
                isinstance(feat, torch.Tensor)
                and self.posdist_enable
                and self.model.training
            ):
                posdist_loss, posdist_items = self._positive_distribution_regularization(feat, y)
                total = total + posdist_loss
                for k, v in posdist_items.items():
                    items[k] = float(v)

        return total, items

    def _select_hyperbolic_input(self, logits: torch.Tensor, extras: Any) -> torch.Tensor:
        """
        Prefer penultimate features if model exposes them; fallback to logits.
        """
        if extras is None:
            return logits

        # Case 1: extras is dict.
        if isinstance(extras, dict):
            feat = extras.get("features", None)
            if isinstance(feat, torch.Tensor):
                return feat
            return logits

        # Case 2: extras packed in tuple/list (legacy unpack path).
        if isinstance(extras, (tuple, list)) and len(extras) > 0:
            # Single dict payload: ({'features': ...},)
            if len(extras) == 1 and isinstance(extras[0], dict):
                feat = extras[0].get("features", None)
                if isinstance(feat, torch.Tensor):
                    return feat
            # First tensor payload fallback.
            if isinstance(extras[0], torch.Tensor):
                return extras[0]
        return logits

    def _forward_unpack(self, x):
        out = None
        # Prefer requesting features from backbone if supported.
        tried = []
        for call in (
            lambda: self.model(x, return_features=True),
            lambda: self.model(x),
            lambda: self.model(x, 2, True),
            lambda: self.model(x, 2),
        ):
            try:
                out = call()
                break
            except TypeError as e:
                tried.append(str(e))
                continue
        if out is None:
            # Re-raise with compact context.
            raise TypeError(f"Model forward call failed after fallbacks: {tried[:2]} ...")
        if isinstance(out, tuple):
            logits = out[0]
            if len(out) == 2 and isinstance(out[1], dict):
                extras = out[1]
            else:
                extras = out[1:]
        else:
            logits = out
            extras = None
        return logits, extras

    def _display_progress(self, epoch: int, epochs: int, train_loss, val_loss: float, val_ba: float):
        self._log_info(
            f"Epoch {epoch:03d}/{epochs:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BA: {val_ba:.4f}"
        )

    def _display_progress_console(self, epoch: int, epochs: int, train_loss, val_loss: float, val_ba: float):
        self._print_console(
            f"Epoch {epoch:03d}/{epochs:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BA: {val_ba:.4f}"
        )

    def _log_training_metrics(self, metrics, mode, stage: int):
        if self.tb_logger is None:
            return
        assert mode in ["train", "val"]
        prefix = f"{mode}/stage{stage}"

        loss_avgs, loss_names = metrics[0]
        for name, value in zip(loss_names, loss_avgs):
            if value is not None:
                self.tb_logger.experiment.add_scalar(f"{prefix}/Loss/{name}", value, self.current_epoch)

        auc, ba, f1, tpr, fpr = metrics[1:]
        for name, value in zip(["AUC", "BA", "F1", "TPR", "FPR"], [auc, ba, f1, tpr, fpr]):
            if value is not None:
                self.tb_logger.experiment.add_scalar(f"{prefix}/{name}", value, self.current_epoch)
