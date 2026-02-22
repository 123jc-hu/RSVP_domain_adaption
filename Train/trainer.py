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
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from lightning.pytorch.loggers import TensorBoardLogger

from Data.datamodule import EEGDataModuleCrossSubject
from Train.hyr_dpa_framework import HyRDPAScaffold
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

    def _setup_exp_dir(self) -> Path:
        """Create `Experiments/<model>/<dataset>/<train_mode>` directory."""
        root = Path("Experiments") / self.config["model"] / self.config["dataset"] / self.config["train_mode"]
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
        set_random_seed(int(self.config.get("random_seed", 2026)))
        self.log.info(f"HyR-DPA switches | {self.hyr_dpa.describe()}")

        self.runtime_records = []
        self.source_selection_records = []
        self.rpcs_ranking_records = []
        self.rpcs_fold_records = []

        dataset_dir = self._dataset_dir()

        subject_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".npz") and "_10band" not in f])
        subject_file_map = {}
        for filename in subject_files:
            m = re.search(r"sub(\d+)", filename)
            if not m:
                continue
            sid = int(m.group(1))
            subject_file_map[f"sub{sid}"] = str(dataset_dir / filename)

        self._get_init_state()

        metric_history = {k: [] for k in ["AUC", "BA", "F1", "TPR", "FPR"]}

        with MemoryManager.cuda_memory_context():
            for filename in subject_files:
                set_random_seed(int(self.config.get("random_seed", 2026)))

                m = re.search(r"sub(\d+)", filename)
                if not m:
                    continue
                subject_id = int(m.group(1))
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

        self._save_results(metric_history)
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
            self.log.info(f"[RUNTIME-DATASET] {pretty}")

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

    def _dataset_dir(self) -> Path:
        """Resolve dataset directory path for current dataset and sampling rate."""
        cfg = self.config
        root = Path(
            cfg.get(
                "dataset_root",
                os.environ.get("RSVP_DATASET_ROOT", "E:/learning_projects/few-shot-learning_RSVP/Dataset"),
            )
        )
        if "_" in cfg["dataset"]:
            dataset, task = cfg["dataset"].split("_")
            return root / dataset / f"Standard_{cfg['fs']}Hz" / f"task{task}"
        return root / cfg["dataset"] / f"Standard_{cfg['fs']}Hz"

    def _run_subject(self, subject_id: int) -> Dict[str, float]:
        """Train/evaluate one LOSO subject and return rounded metrics."""
        t0 = _t.perf_counter()
        model = self._new_model()
        metric_values = self._run_single_subject(model, subject_id)

        del model
        elapsed = _t.perf_counter() - t0
        self.log.info(f"Subject {subject_id} completed in {elapsed:.2f}s")
        return {k: round(v, 4) for k, v in zip(["AUC", "BA", "F1", "TPR", "FPR"], metric_values)}

    def _run_single_subject(self, model: nn.Module, subject_id: int) -> Tuple[float, float, float, float, float]:
        """Single-stage train/eval for one LOSO target subject."""
        # Build split datasets first so class weights are fold-specific.
        self.datamodule.setup()
        self._record_source_selection(subject_id)

        class_weights = None
        if bool(self.config.get("class_weighted_ce", True)):
            class_weights = self.datamodule.source_class_weight_tensor(
                n_class=int(self.config.get("n_class", 2)),
                device=self.device,
            )
            self.log.info(f"Subject {subject_id} source class weights: {class_weights.detach().cpu().tolist()}")

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

        ckpt_dir, log_dir = self._prepare_dirs(subject_id)
        if self.config.get("is_training", True):
            self._clean_subject_logs(log_dir, run_prefix="metrics")
        trainer.setup_logging(log_dir, ckpt_dir, stage=2, run_name=f"metrics_sub{subject_id}")

        if self.config.get("is_training", True):
            mode = str(self.config.get("training_mode", "End2End")).strip().lower()
            if mode == "decoupled":
                self.log.info("Training mode: Decoupled | Stage 1 (feature alignment scaffold)")
                self.hyr_dpa.stage1_feature_alignment()
                self._train(trainer)
                self.log.info("Training mode: Decoupled | Stage 2 (classifier rectification scaffold)")
                self.hyr_dpa.stage2_classifier_rectification()
            else:
                self.log.info("Training mode: End2End")
                self._train(trainer)

        return self._eval(trainer, ckpt_dir)

    def _train(self, trainer):
        """Build dataloaders and run fit()."""
        cfg = self.config
        batch_size = int(cfg["batch_size"])
        use_target_stream = bool(cfg.get("use_target_stream", False))
        if use_target_stream:
            train_loader = self.datamodule.train_dataloader(batch_size=batch_size)
        else:
            train_loader = self.datamodule.source_train_dataloader(batch_size=batch_size)
        val_loader = self.datamodule.val_dataloader(batch_size=batch_size)
        self.log.info(
            f"Training setup | use_target_stream={use_target_stream} | "
            f"train_samples={len(self.datamodule.train_dataset)} | val_samples={len(self.datamodule.val_dataset)}"
        )

        trainer.fit(train_loader, val_loader, stage=2, epochs=int(cfg["epochs"]))

        del self.datamodule.train_dataset, self.datamodule.val_dataset

        torch.save(trainer.model.state_dict(), trainer.checkpoint_dir / "last_model--2.pth")

    def _eval(self, trainer, ckpt_dir: Path) -> Tuple[float, float, float, float, float]:
        """Load best checkpoint and run evaluation."""
        best_path = ckpt_dir / "best_ba_model--2.pth"
        trainer.model.load_state_dict(load_from_checkpoint(Path.cwd() / best_path))

        test_loader = self.datamodule.test_dataloader(batch_size=int(self.config.get("test_batch_size", 1000)))

        metrics = trainer._run_epoch(test_loader, training=False, loss_fn=trainer.loss_fn)

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

            self.log.info("[RUNTIME] " + " | ".join(
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

    def _save_results(self, results: Dict[str, List[float]]):
        """Aggregate subject metrics, append AVG/STD, and save CSV."""
        df = pd.DataFrame(results)
        df.insert(0, "SUB", [f"SUB{i+1}" for i in range(len(df))])

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
        self.log.info(f"Results saved to {csv_path}")

    def _record_source_selection(self, subject_id: int):
        info = dict(getattr(self.datamodule, "source_selection_info", {}) or {})
        if not info:
            return
        selected = info.get("selected_subjects", [])
        self.log.info(
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
                "num_candidates": info.get("num_candidates"),
                "num_selected": info.get("num_selected"),
                "selected_subjects": ",".join(selected),
            }
        )
        self._record_rpcs_details(subject_id, selected_subjects=selected)

    def _save_source_selection(self):
        if not self.source_selection_records:
            return
        df = pd.DataFrame(self.source_selection_records)
        out = self.exp_dir / "source_selection.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        self.log.info(f"Source selection records saved to {out}")

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

        self.log.info(f"R-PCS ranking records saved to {out_long}")
        self.log.info(f"R-PCS selected-source records saved to {out_selected}")


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

        self.scaler = GradScaler()
        self.tb_logger = None
        self.checkpoint_dir = None
        self.current_epoch = 0

        self.log_runtime: bool = bool(self.config.get("log_runtime", True))
        self.rt_deadline_ms: float = float(self.config.get("rt_deadline_ms", 200.0))
        self.last_runtime_stats: dict | None = None

    def _log_info(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

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

    def fit(self, train_loader, val_loader, stage: int, epochs: int):
        patience = int(self.config.get("patience", 20))
        early_stop_start_epoch = int(self.config.get("early_stop_start_epoch", 0))
        ba_delta = float(self.config.get("ba_delta", 0.0))
        best_ba = float("-inf")
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            with MemoryManager.cuda_memory_context():
                train_metrics = self._run_epoch(train_loader, training=True, loss_fn=self.loss_fn)
                train_loss_avg = train_metrics[0][0][0]
                train_ba = train_metrics[2]

                val_metrics = self._run_epoch(val_loader, training=False, loss_fn=self.loss_fn)
                val_loss_avg = val_metrics[0][0][0]
                val_ba = val_metrics[2]

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
                    self._log_info(
                        f"Early stopping triggered on BA | epoch={epoch} | "
                        f"start_epoch={early_stop_start_epoch} | patience={patience} | best_BA={best_ba:.4f}"
                    )
                    break

                self.scheduler.step()

                if hasattr(self.model, "step_epoch"):
                    self.model.step_epoch(1)


    # ---------- epoch core ----------
    def _run_epoch(self, dataloader, *, training: bool, loss_fn: Callable):
        """
        Run one training/eval epoch and return:
          [
            (loss_avgs_list, loss_names_list),
            AUC, BA, F1, TPR, FPR
          ]
        """
        self.model.train() if training else self.model.eval()
        collect_runtime = (self.log_runtime and (not training))
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
                n_batches += 1
                t0 = _t.perf_counter()

                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)
                sample_count += x.size(0)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = _t.perf_counter()

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    logits, extras = self._forward_unpack(x)
                    loss_total, items = loss_fn(logits, y, extras)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(1)

                if torch.cuda.is_available():
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
                MemoryManager.cleanup_tensors(x, y, logits)

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

    def loss_ce(self, logits, y, extras):
        """Pure CE"""
        ce = self._cross_entropy(logits, y)
        return ce, OrderedDict([("ce", float(ce.detach().item()))])

    def _forward_unpack(self, x):
        try:
            out = self.model(x)
        except TypeError:
            out = self.model(x, 2)
        if isinstance(out, tuple):
            logits = out[0]
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
