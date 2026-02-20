from shutil import rmtree
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Any, Optional
from collections import OrderedDict
from lightning.pytorch.loggers import TensorBoardLogger
import logging
import gc
from contextlib import contextmanager
import time as _t
from pathlib import Path
import os
import re
from Data.datamodule import EEGDataModuleCrossSubject
from Utils.config import set_random_seed
from Utils.utils import load_from_checkpoint, EarlyStopping, SaveBestValBA
from Utils.metrics import calculate_metrics, cal_F1_score
# from Models.eeg_models import model_dict
# from Models.embedded_selectors import build_model
from Models.adaptive_os_selector import build_model
from Models.regularized_gumbel import build_model_dup
from Utils.Loss import *  # Import all loss functions


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
    """
     Experiment Runner?
    -  build_model(config) ?
    -  subject/fold 
    - ?(is_training=True/False) + 
    - ?10band / normal normal 
    - ?OptimizedTrainer epoch  + ?loss?
    """

    def __init__(self, config: Dict[str, Any], main_logger: logging.Logger):
        self.config = config
        self.log = main_logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_dir = self._setup_exp_dir()
        self.datamodule = EEGDataModuleCrossSubject(self.config)

        # 
        self._init_state_cache: Dict[tuple, Dict[str, torch.Tensor]] = {}

        self.selection_records = []  # /{subject, K, uniq, reg, dataset}

        self.runtime_records: List[Dict[str, Any]] = []  #  runtime 

    # ----------------  ----------------
    def _setup_exp_dir(self) -> Path:
        """xperiments/<model>/<dataset>/<train_mode>"""
        root = Path("Experiments") / self.config["model"] / self.config["dataset"] / self.config["train_mode"]
        root.mkdir(parents=True, exist_ok=True)
        return root

    # ----------------  dict key?----------------
    def _model_signature(self) -> tuple:
        """
        ?
         tuple?dict ?key?
        """
        cfg = self.config
        return (
            cfg["model"],
            bool(cfg.get("use_selector", False)),
            int(cfg.get("selector_K", -1)),
            float(cfg.get("selector_beta_start", 5.0)),
            float(cfg.get("selector_beta_end", 0.1)),
            float(cfg.get("selector_T", cfg.get("epochs", 100))),
            bool(cfg.get("selector_per_sample", True)),
            int(cfg.get("n_channels", -1)),
            int(cfg.get("n_class", -1)),
            int(cfg.get("fs", -1)),
        )

    # ---------------- ?----------------
    def _get_init_state(self) -> Dict[str, torch.Tensor]:
        """
        ?state_dictPU ?
        ?
        """
        sig = self._model_signature()
        if sig not in self._init_state_cache:
            model = build_model(self.config) if self.config["train_algorithm"] != "reg" else build_model_dup(self.config)
            self._init_state_cache[sig] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            del model
        return self._init_state_cache[sig]

    def _new_model(self) -> nn.Module:
        """Create a fresh model initialized from cached initial state."""
        model = build_model(self.config) if self.config["train_algorithm"] != "reg" else build_model_dup(self.config)
        init_state = self._get_init_state()
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return model.to(self.device)

    # ---------------- //?----------------
    def _configure_training_components(self, model: nn.Module):
        """
        ?backbone ?selector ?param group?
        - backbone: lr=learning_rate, weight_decay=weight_decay
        - selector: lr=selector_lr?lr? weight_decay=selector_weight_decay?wd?
        """
        cfg = self.config

        # param groups: selector vs backbone
        sel_params, bb_params = [], []
        for n, p in model.named_parameters():
            (sel_params if n.startswith("selector.") else bb_params).append(p)

        param_groups = []
        if bb_params:
            param_groups.append({
                "params": bb_params,
                "lr": float(cfg["learning_rate"]),
                "weight_decay": float(cfg.get("weight_decay", 0.0)),
            })
        if sel_params:
            param_groups.append({
                "params": sel_params,
                "lr": float(cfg.get("selector_lr", cfg["learning_rate"])),
                "weight_decay": float(cfg.get("selector_weight_decay", cfg.get("weight_decay", 0.0))),
                "name": "selector",
            })

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(cfg.get("lr_patience", 5))
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

    # ---------------- ?subject ?----------------
    def run_experiment(self):
        set_random_seed(int(self.config.get("random_seed", 2024)))

        # 
        self.runtime_records = []
        self.selection_records = []

        base_dir = self._dataset_base()
        indices_path = base_dir / f"balanced_indices_for_{self.config['train_mode']}.pkl"

        #  & ?sub->path OSO 
        files = sorted([f for f in os.listdir(base_dir) if f.endswith(".npz") and "_10band" not in f])
        subject_file_map = {}
        for fname in files:
            m = re.search(r"sub(\d+)", fname)
            if not m:
                continue
            sid = int(m.group(1))
            subject_file_map[f"sub{sid}"] = str(base_dir / fname)

        # 
        _ = self._get_init_state()

        results = {k: [] for k in ["AUC", "BA", "F1", "TPR", "TNR"]}

        with MemoryManager.cuda_memory_context():
            for fname in files:
                set_random_seed(int(self.config.get("random_seed", 2024)))

                m = re.search(r"sub(\d+)", fname)
                if not m:
                    continue
                subject_id = int(m.group(1))

                # Cross-subject LOSO only.
                self.datamodule.prepare_data(
                    test_subject_id=subject_id,
                    subject_file_map=subject_file_map,
                    indices_path=indices_path,
                    mode="normal",
                    train_range=self.config.get("train_range", "train"),  # "train" or "all"
                )

                #  +  self.datamodule.setup(...)?
                metrics = self._run_subject(subject_id)
                self._log_subject(subject_id, metrics)
                for k in results:
                    results[k].append(metrics[k])

        # =====  =====
        if self.selection_records:
            ratios = [rec["uniq"] / rec["K"] for rec in self.selection_records if rec["K"] > 0]
            mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
            std_ratio = float(np.std(ratios)) if ratios else float("nan")
            n_subjects = len(ratios)
            sum_uniq = int(sum(rec["uniq"] for rec in self.selection_records))
            unique_Ks = sorted({rec["K"] for rec in self.selection_records})
            if len(unique_Ks) == 1:
                K0 = unique_Ks[0]
                total_ratio = sum_uniq / (K0 * n_subjects) if (K0 > 0 and n_subjects > 0) else float("nan")
                self.log.info(f"[RUN-UNIQ] subjects={n_subjects} K={K0} "
                      f"mean(uniq/K)={mean_ratio:.4f}{std_ratio:.4f} "
                      f"sum(uniq)={sum_uniq} total_ratio={total_ratio:.4f}")
            else:
                self.log.info(f"[RUN-UNIQ] subjects={n_subjects} Ks={unique_Ks} "
                      f"mean(uniq/K)={mean_ratio:.4f}{std_ratio:.4f} sum(uniq)={sum_uniq}")

        self._save_results(results)

        # ===== ?=====
        if self.runtime_records:
            import pandas as _pd
            rt_df = _pd.DataFrame(self.runtime_records)
            cols = [
                "p50(ms) / Model p50",
                "p95(ms) / Model p95",
                "(ms) / Model jitter",
                "50(ms) / E2E p50",
                "95(ms) / E2E p95",
                "?ms) / E2E jitter",
                "(samples/s) / Throughput",
                "?MB) / Peak CUDA Mem",
                "Deadline?/ Miss Rate",
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
            pretty = {k: f"{mean_row[k]:.3f}{std_row[k]:.3f}" for k in cols if k in mean_row and k in std_row}
            print("[RUNTIME-DATASET] ", pretty)

    # ----------------  ----------------
    def _dataset_base(self) -> Path:
        """
        E:/.../<dataset>/Standard_<fs>Hz/[task?]
        ?10band ?
        """
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

    # ----------------  subject ?----------------
    def _run_subject(self, subject_id: int) -> Dict[str, float]:
        """
        ?subject?
        - is_training?
        - ?OptimizedTrainer ?_run_epoch(training=False, loss_fn)?
        """
        start = _t.perf_counter()

        #  subject ?
        model = self._new_model()
        vals = self._run_single_subject(model, subject_id)

        del model
        elapsed = _t.perf_counter() - start
        print(f"Subject {subject_id} completed in {elapsed:.2f}s")

        # 
        return {k: round(v, 4) for k, v in zip(["AUC", "BA", "F1", "TPR", "TNR"], vals)}

    def _run_single_subject(self, model: nn.Module, subject_id: int) -> Tuple[float, float, float, float, float]:
        """Single-stage train/eval for one LOSO target subject."""
        stage = 2

        # Build split datasets first so class weights are fold-specific.
        self.datamodule.setup(stage=2, is_split_domains=True)

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
            algorithm=self.config.get("train_algorithm", "ce"),
            class_weights=class_weights,
        )

        ckpt_dir, log_dir = self._prepare_dirs(subject_id)
        if self.config.get("is_training", True):
            self._clean_subject_logs(log_dir, run_prefix="metrics")
        trainer.setup_logging(log_dir, ckpt_dir, stage=2, run_name=f"metrics_sub{subject_id}")

        if self.config.get("is_training", True):
            self._train(trainer)

        return self._eval(trainer, ckpt_dir)

    def _train(self, trainer):
        """Build dataloaders and run fit()."""
        cfg = self.config

        batch_size = int(cfg["batch_size"])

        # domain-aware sampler??
        # sampler = None
        # if cfg["train_mode"] == "cross-subject" or stage == 2:
        # labels_np, doms_np = self.act_labels_domains(self.datamodule.train_dataset)
        # sampler = BalancedRandomDomainSampler(
        #     data_source=self.datamodule.train_dataset,
        #     batch_size=batch_size,
        #     n_domains_per_batch=int(self.config.get("domains_per_batch", 32)),
        #     labels=labels_np,
        #     domain_labels=doms_np
        # )

        use_target_stream = bool(cfg.get("use_target_stream", False))
        if use_target_stream:
            train_loader = self.datamodule.train_dataloader(batch_size=batch_size)
        else:
            train_loader = self.datamodule.source_train_dataloader(batch_size=batch_size)
        val_loader   = self.datamodule.val_dataloader(batch_size=batch_size)
        # print(f"  Training samples: {len(ds)} val samples: {len(self.datamodule.val_dataset)}")

        # train_loader = make_zero_channel_dataloader(train_loader, shuffle=True, add_zero_channel=False, random_seed=2025)
        # val_loader   = make_zero_channel_dataloader(val_loader,   shuffle=False, add_zero_channel=False, random_seed=2025)

        # 
        # self.mask_dict = generate_mask_lists(n_channels=62, Ms=(5, 10, 15, 20), n_lists=10, random_seed=2025)
        # self.mask_channels = self.mask_dict[self.config["masked_num_channels"]][self.config["mask_index"]]
        # self.log.info(f"Channel Perturbation Masks ({self.config['masked_num_channels']} channels): {self.mask_channels}")
        # train_loader = make_corrupted_dataloader(
        #     dataloader=train_loader,  #  DataLoader
        #     mask_channels=self.mask_channels,
        #     mode="noise",  # or "noise"
        #     shuffle=True,  # ?True None( False)
        #     noise_std=1.0,  # ?z-score?.0 
        #     n_channels=62,  # ?
        #     noise_seed=2025,  # 
        # )
        # val_loader = make_corrupted_dataloader(
        #     dataloader=val_loader,
        #     mask_channels=self.mask_channels,
        #     mode="noise",
        #     shuffle=False,
        #     noise_std=1.0,
        #     n_channels=62,
        #     noise_seed=2025,
        # )

        # ?
        # self.channel_selector = CorrelationBasedChannelSelector(dataloader=train_loader, K=self.config.get("selector_K", 32), device="cuda").fit()
        # self.channel_selector = SparseEAChannelSelector(
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     fs=self.config["fs"],
        #     target_K=self.config.get("selector_K", 32),
        #     population_size=60,
        #     generations=40,
        #     win_ms=100,
        # ).fit()
        # bands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 40)]
        # self.channel_selector = FrequencyWeightedCSPChannelSelector(
        #     train_loader=train_loader,
        #     fs=self.config["fs"],
        #     K=self.config.get("selector_K", 32),
        #     bands=bands,
        #     aggregate="mean",   # "mean"/"max"/"median"
        #     fir_taps=129,
        #     use_filter=True,    #  False 
        # ).fit()
        # self.channel_selector = ABMOHSChannelSelector(
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     fs=self.config["fs"],
        #     target_K=self.config.get("selector_K", 32),  # Kmax
        #     harmony_size=50,
        #     iterations=100,
        #     n_runs=10,
        #     cv_folds=10,  # ?
        #     fitness_mode="val",  # ?"val"
        #     precompute_cov=True,
        # ).fit()
        # print("Selected channels:", self.channel_selector.selected_channels_)
        # train_loader = self.channel_selector.transform(train_loader, batch_size=train_loader.batch_size, shuffle=True)
        # val_loader   = self.channel_selector.transform(val_loader,   batch_size=val_loader.batch_size,   shuffle=False)

        # ?
        trainer.fit(train_loader, val_loader, stage=2, epochs=int(cfg["epochs"]))

        # 
        del self.datamodule.train_dataset, self.datamodule.val_dataset

        # ?
        torch.save(trainer.model.state_dict(), trainer.checkpoint_dir / "last_model--2.pth")

    def _eval(self, trainer, ckpt_dir: Path) -> Tuple[float, float, float, float, float]:
        """Load best checkpoint and run evaluation with the active loss recipe."""
        best_path = ckpt_dir / "best_model--2.pth"
        trainer.model.load_state_dict(load_from_checkpoint(Path.cwd() / best_path))

        # 
        if hasattr(trainer.model, "selector"):
            indices = trainer.model.selector.hard_indices()  # List[int], =K
            K = int(self.config.get("selector_K", len(indices)))
            uniq = len(set(indices))
            reg_used = float(self.config.get("selector_lambda", 0.0)) > 0.0  # ?

            # ?datamodule  ID?
            subj = self.datamodule.subject_id

            self.selection_records.append({
                "subject": subj,
                "K": K,
                "uniq": uniq,
                "reg": reg_used,
                "dataset": self.config.get("dataset", "NA"),
            })
            print(f"[SEL-STAT] subject={subj} K={K} uniq={uniq} reg={reg_used}")
            print(f"  indices: {indices}")
            self.log.info(f"subject={subj} selected channels: {indices}")

        # 
        test_loader = self.datamodule.test_dataloader(batch_size=int(self.config.get("test_batch_size", 1000)))
        # test_loader = make_zero_channel_dataloader(test_loader, shuffle=False, add_zero_channel=False, random_seed=2025)


        # 
        # test_loader = make_corrupted_dataloader(
        #     dataloader=test_loader,  #  DataLoader
        #     mask_channels=self.mask_channels,
        #     mode="noise",  # or "noise"
        #     shuffle=False,  # ?
        #     noise_std=1.0,  # ?z-score?.0 
        #     n_channels=62,  # ?
        #     noise_seed=2025,  # 
        # )

        # ?
        # test_loader = self.channel_selector.transform(test_loader, batch_size=test_loader.batch_size, shuffle=False)

        metrics = trainer._run_epoch(test_loader, training=False, loss_fn=trainer.loss_fn)

        #  runner 
        rt = getattr(trainer, "last_runtime_stats", None)
        # ?subject 
        try:
            subj = int(re.search(r"sub(\d+)", str(self.datamodule.test_path)).group(1)) \
                if hasattr(self.datamodule, "test_path") else None
        except Exception:
            subj = None

        if rt is not None:
            #  runner
            row = {"subject": subj}; row.update(rt)
            self.runtime_records.append(row)

            # ?
            print("[RUNTIME] ", " | ".join(
                f"{k}: {rt[k]:.3f}" if isinstance(rt[k], (int, float)) else f"{k}: {rt[k]}"
                for k in [
                    "p50(ms) / Model p50",
                    "p95(ms) / Model p95",
                    "50(ms) / E2E p50",
                    "95(ms) / E2E p95",
                    "?MB) / Peak CUDA Mem",
                    "(samples/s) / Throughput",
                    "Deadline?/ Miss Rate",
                ] if k in rt
            ))

            # ?CSV
            try:
                import pandas as _pd
                df_rt = _pd.DataFrame([{"subject": subj, **rt}])
                out_csv = self.exp_dir / "runtime_metrics.csv"
                header = (not out_csv.exists())
                df_rt.to_csv(out_csv, mode="a", index=False, encoding="utf-8-sig", header=header)
            except Exception as _e:
                print(f"[WARN] runtime metrics save failed: {_e}")
        del self.datamodule.test_dataset
        return metrics[1:]  # AUC, BA, F1, TPR, TNR

    # ---------------- // ----------------
    def _prepare_dirs(self, subject_id: int) -> Tuple[Path, Path]:
        """ subject ?checkpoint ?log """
        base = self.exp_dir / "checkpoints" / f"sub{subject_id}"
        logs = self.exp_dir / "logs" / f"sub{subject_id}"
        base.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        return base, logs

    def _maybe_clean_dirs(self, *paths: Path):
        # 
        for p in paths:
            if p.exists() and any(p.iterdir()):
                continue  # ?
            p.mkdir(parents=True, exist_ok=True)
    
    def _clean_subject_logs(self, log_root: Path, *, run_prefix: str = "metrics"):
        """
        ?subject ?TensorBoard ?
        - ?log_root  run_prefix ?metricsetrics_sub1etrics_stage2 
        - 
        """
        if not log_root.exists():
            return
        for child in log_root.iterdir():
            if child.is_dir() and child.name.startswith(run_prefix):
                rmtree(child, ignore_errors=True)

    def _log_subject(self, subject_id: int, metrics: Dict[str, float]):
        self.log.info(
            f"Subject {subject_id} | AUC: {metrics['AUC']:.4f} | BA: {metrics['BA']:.4f} "
            f"| F1: {metrics['F1']:.4f} | TPR: {metrics['TPR']:.4f} | TNR: {metrics['TNR']:.4f}"
        )

    def _save_results(self, results: Dict[str, List[float]]):
        """Aggregate subject metrics, append AVG/STD, and save CSV."""
        df = pd.DataFrame(results)
        df.insert(0, "SUB", [f"SUB{i+1}" for i in range(len(df))])

        metrics = ["AUC", "BA", "F1", "TPR", "TNR"]
        avg = df[metrics].mean().round(4)
        std = df[metrics].std().round(4)

        # 
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
    
    def _extract_labels_domains(ds):
        #  ConcatDataset(WithDomainLabel) ?
        if hasattr(ds, "datasets"):  # Concat
            ys, doms = [], []
            for dom_id, part in enumerate(ds.datasets):
                # ?y_data  x 
                y = np.asarray(part.y_data[part.indices], dtype=np.int64)
                ys.append(y)
                doms.append(np.full(y.shape, dom_id, dtype=np.int64))
            return np.concatenate(ys), np.concatenate(doms)
        else:
            y = np.asarray(ds.y_data[ds.indices], dtype=np.int64)
            return y, np.zeros_like(y)


class OptimizedTrainer:
    """
    Optimized trainer with unified epoch core and Lightning-style loss dispatch.
    - loss_fn must return: (loss_total: Tensor, loss_items: OrderedDict[str, float-like])
        * loss_items ?"ce"
    """

    def __init__(self, config: Dict, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 earlystopping: EarlyStopping, save_other_model: SaveBestValBA,
                 device: torch.device, algorithm: str = "ce",
                 class_weights: Optional[torch.Tensor] = None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.earlystopping = earlystopping
        self.save_other_model = save_other_model
        self.device = device
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = class_weights.detach().to(self.device, dtype=torch.float32)

        # ---- loss routing ----
        self.loss_fn = self._resolve_loss_fn(algorithm)

        # ---- infra ----
        self.scaler = GradScaler()
        self.tb_logger = None
        self.checkpoint_dir = None
        self.current_epoch = 0

        # ?runtime perf config & state ? ?
        self.log_runtime: bool = bool(self.config.get("log_runtime", True))
        self.rt_deadline_ms: float = float(self.config.get("rt_deadline_ms", 200.0))  # ?
        self.last_runtime_stats: dict | None = None  #  _run_epoch ?

    # ---------- public ----------
    def setup_logging(self, log_dir: Path, checkpoint_dir: Path, stage: int, *, run_name: str):
        self.checkpoint_dir = checkpoint_dir
        #  subject  run
        self.tb_logger = TensorBoardLogger(
            save_dir=log_dir,          # e.g. Experiments/.../logs/sub11
            name=run_name,             # e.g. f"metrics_stage{stage}"
            version=None,              #  version
            default_hp_metric=False,
        )
        stage_suffix = f"-{stage}"
        self.earlystopping.path = checkpoint_dir / f"best_model-{stage_suffix}.pth"
        self.save_other_model.path = checkpoint_dir / f"best_ba_model-{stage_suffix}.pth"

    def fit(self, train_loader, val_loader, stage: int, epochs: int):
        # -------- 2)  + Legacy ?-------
        base_external_beta = bool(self.config.get("selector_beta_external", False))

        mode = str(self.config.get("train_algorithm", "")).lower()        # NEW
        is_dup = (mode == "dup")                                          # NEW
        #  algorithm == reg dup ?
        use_external_beta = (self.config["train_algorithm"] == "reg") or base_external_beta

        beta_s = float(self.config.get("selector_beta_start", 10.0))
        beta_e = float(self.config.get("selector_beta_end",   0.1))
        T      = int(self.config.get("selector_T", epochs))

        tau_s  = float(self.config.get("selector_tau_start", 3.0))
        tau_e  = float(self.config.get("selector_tau_end",   1.1))

        # ---- dup  ----
        H_thresh         = float(self.config.get("selector_entropy_thresh", 0.70))  # CHANGED: ?0.70 ?
        entropy_patience = int(self.config.get("selector_entropy_patience", 3))
        #  fit 
        self._es_armed = False                          # CHANGED:  reg
        self._h_below_count = 0                         # CHANGED

        def _exp_anneal(start: float, end: float, t: int, T: int) -> float:
            frac = min(max(t / max(T, 1), 0.0), 1.0)
            return float(start) * ((float(end) / float(start)) ** frac)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # -------- 3) --------
            if use_external_beta and hasattr(self.model, "selector"):
                if hasattr(self.model.selector, "set_temperature"):
                    beta_t = _exp_anneal(beta_s, beta_e, epoch, T)
                    self.model.selector.set_temperature(beta_t)
                # tau  reg 
                if (self.config["train_algorithm"] == "reg") and hasattr(self.model.selector, "set_thresh"):
                    tau_t = _exp_anneal(tau_s, tau_e, epoch, T)
                    self.model.selector.set_thresh(tau_t)

            with MemoryManager.cuda_memory_context():
                # ---- train ----
                train_metrics = self._run_epoch(train_loader, training=True, loss_fn=self.loss_fn, epoch=epoch)
                train_loss_avg = train_metrics[0][0][0]  # total is first

                # ---- val (same loss recipe, eval mode) ----
                val_metrics = self._run_epoch(val_loader, training=False, loss_fn=self.loss_fn, epoch=epoch)
                val_loss_avg = val_metrics[0][0][0]
                val_ba = val_metrics[2]

                self._display_progress(epoch, epochs, train_loss_avg, val_loss_avg, val_ba)

                self._log_training_metrics(train_metrics, mode="train", stage=stage)
                self._log_training_metrics(val_metrics,   mode="val",   stage=stage)

                if (self.current_epoch % 5 == 0) and hasattr(self.model, "selector"):
                    d = self._selector_diag()
                    if d is not None:
                        tau_show = (tau_t if (use_external_beta and (self.config["train_algorithm"] == "reg") and hasattr(self.model.selector, "set_thresh")) else None)
                        extra = (f" tau={tau_show:.3f}" if tau_show is not None else "")
                        print(
                            f"[SEL] ep={self.current_epoch} "
                            f"beta={d['beta']:.3f}{extra} H_mean={d['H_mean']:.3f} Top1={d['Top1_mean']:.3f} "
                            f"Kuniq={d['K_unique']} | ||alpha||={d['alpha_norm']:.2f} "
                            f"||grad||={d['alpha_grad_norm'] if d['alpha_grad_norm'] is not None else 'NA'} "
                            f"indices={d['indices_head']}"
                        )
                    with torch.no_grad():
                        P = self.model.selector.current_p_soft()      # (K,C)
                        col_sum = P.sum(dim=0)                        # (C,)
                        tau = (self.model.selector._tau() if hasattr(self.model.selector, "_tau")
                            else float(self.config.get("selector_tau_end", 1.1)))
                        violating = (col_sum > tau).sum().item()
                        margin = float(col_sum.max().item() - tau)
                        dup_rate = 1.0 - (P.argmax(dim=1).unique().numel() / P.size(0))
                        print(f"[dup] tau={tau:.3f} max={col_sum.max():.3f} mean={col_sum.mean():.3f} "
                            f"viol={violating} margin={margin:+.3f} dup_rate={dup_rate:.3f}")

                # ---------- NEW: dup  ??----------
                if is_dup and hasattr(self.model, "selector"):                 # CHANGED: reg ?dup
                    d2 = self._selector_diag()
                    Hm = d2.get("H_mean") if d2 is not None else None
                    if Hm is not None:
                        if Hm <= H_thresh:
                            self._h_below_count += 1
                        else:
                            self._h_below_count = 0
                        if (not self._es_armed) and (self._h_below_count >= entropy_patience):
                            self._es_armed = True
                            print(f"[EarlyStop] ARMED at epoch {epoch}: H_mean={Hm:.4f} ?{H_thresh} "
                                f"(for {self._h_below_count} consecutive epochs)")
                            # 
                            if hasattr(self.earlystopping, "reset") and callable(self.earlystopping.reset):
                                self.earlystopping.reset()
                            else:
                                if hasattr(self.earlystopping, "counter"):
                                    self.earlystopping.counter = 0
                                if hasattr(self.earlystopping, "early_stop"):
                                    self.earlystopping.early_stop = False
                                if hasattr(self.earlystopping, "best_score"):
                                    self.earlystopping.best_score = None
                                if hasattr(self.earlystopping, "val_loss_min"):
                                    self.earlystopping.val_loss_min = float("inf")

                # early stop / save / sched
                if (not is_dup) or (self._es_armed is True):                   # CHANGED: reg ?dup
                    self.earlystopping(val_loss_avg, self.model)
                    self.save_other_model(val_ba, self.model)
                    if self.earlystopping.early_stop:
                        print("Early stopping triggered\n")
                        break
                else:
                    # dup ?
                    self.save_other_model(val_ba, self.model)

                self.scheduler.step(val_loss_avg)

                # ?****poch
                if hasattr(self.model, "step_epoch"):
                    if not use_external_beta:
                        self.model.step_epoch(1)
                    else:
                        pass


    # ---------- epoch core ----------
    def _run_epoch(self, dataloader, *, training: bool, loss_fn: Callable, epoch: int = None):
        """
        training=True: .train() + backward + step
        training=False: .eval() + no_grad
        :
          [
            (loss_avgs_list, loss_names_list),  # e.g., ([total, ce, func2, func3], ["Total", "ce", "Func2", "Func3"])
            AUC, BA, F1, TPR, TNR
          ]
        """
        self.model.train() if training else self.model.eval()

        # ?
        collect_runtime = (self.log_runtime and (not training))

        # ?
        pre_times_ms, model_times_ms, e2e_times_ms = [], [], []

        #  CUDA ?
        if collect_runtime and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        # ?loss ?
        loss_names: List[str] = []       # ["Total", "ce", "Func2", ...]
        loss_sums:  List[float] = []     # ?
        n_batches = 0

        all_probs, all_preds, all_targets = [], [], []
        maybe_no_grad = torch.enable_grad if training else torch.no_grad

        sample_count = 0  #  epoch ?
        with maybe_no_grad():
            for batch in dataloader:
                n_batches += 1

                # ? H2D copy?E2E ?I/O
                t0 = _t.perf_counter()  # E2E ?

                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)
                sample_count += x.size(0)  # ?batch 

                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                
                #  ? t1
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = _t.perf_counter()

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    logits, extras = self._forward_unpack(x, current_epoch=epoch)
                    loss_total, items = loss_fn(logits, y, extras)
                    # items: OrderedDict[str, float-like]?"ce"

                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(1)
                
                #  t2
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t2 = _t.perf_counter()

                # ?
                if collect_runtime:
                    pre_ms   = (t1 - t0) * 1000.0
                    model_ms = (t2 - t1) * 1000.0
                    e2e_ms   = (t2 - t0) * 1000.0
                    pre_times_ms.append(pre_ms)
                    model_times_ms.append(model_ms)
                    e2e_times_ms.append(e2e_ms)

                if training:
                    self.scaler.scale(loss_total).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # ??loss otal ?
                if not loss_names:
                    loss_names = ["Total"] + list(items.keys())
                    loss_sums  = [0.0 for _ in loss_names]

                # total
                loss_sums[0] += float(loss_total.detach().item())
                # sub-items
                for i, k in enumerate(items.keys(), start=1):
                    loss_sums[i] += float(items[k])

                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_targets.append(y.detach().cpu())

                MemoryManager.cleanup_tensors(x, y, logits)

        # metrics
        probabilities = torch.cat(all_probs) if len(all_probs) else torch.empty(0)
        predictions  = torch.cat(all_preds)  if len(all_preds)  else torch.empty(0, dtype=torch.long)
        targets      = torch.cat(all_targets) if len(all_targets) else torch.empty(0, dtype=torch.long)

        if len(all_probs):
            ba, _, tpr, tnr, auc = calculate_metrics(targets, predictions, y_prob=probabilities)
            f1 = cal_F1_score(targets, predictions)
            AUC = auc.numpy().round(4) if auc is not None else None
            BA  = ba.numpy().round(4)
            F1  = f1.numpy().round(4)
            TPR = tpr.numpy().round(4)
            TNR = tnr.numpy().round(4)
        else:
            AUC = BA = F1 = TPR = TNR = None

        # 
        denom = max(n_batches, 1)
        loss_avgs = [s / denom for s in loss_sums] if loss_sums else [0.0]
        metrics = [
            (loss_avgs, loss_names),  # <- ?
            AUC, BA, F1, TPR, TNR
        ]

        if len(all_probs):
            MemoryManager.cleanup_tensors(probabilities, predictions, targets)
        
        # ========== ?==========
        if collect_runtime and len(e2e_times_ms) > 0:
            def _p50(a): 
                b = sorted(a); 
                return b[len(b)//2]
            def _p95(a): 
                b = sorted(a); 
                return b[int(len(b)*0.95)]
            def _jitter(a): 
                return _p95(a) - _p50(a)

            p50_model = _p50(model_times_ms); p95_model = _p95(model_times_ms); jit_model = _jitter(model_times_ms)
            p50_e2e   = _p50(e2e_times_ms);   p95_e2e   = _p95(e2e_times_ms);   jit_e2e   = _jitter(e2e_times_ms)

            #  / 2E?
            total_samples = int(sample_count)
            total_time_s = sum(e2e_times_ms) / 1000.0
            throughput = (total_samples / total_time_s) if total_time_s > 1e-9 else float('nan')

            # MB?
            peak_mb = float('nan')
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated(device=self.device) / 1e6

            # Deadline Miss Rate
            ddl = float(self.rt_deadline_ms)
            miss_rate = sum(1 for t in e2e_times_ms if t > ddl) / len(e2e_times_ms)

            # ?epoch 
            self.last_runtime_stats = {
                "50(ms) / Preproc p50": _p50(pre_times_ms),
                "95(ms) / Preproc p95": _p95(pre_times_ms),
                "p50(ms) / Model p50": p50_model,
                "p95(ms) / Model p95": p95_model,
                "(ms) / Model jitter": jit_model,
                "50(ms) / E2E p50": p50_e2e,
                "95(ms) / E2E p95": p95_e2e,
                "?ms) / E2E jitter": jit_e2e,
                "(samples/s) / Throughput": throughput,
                "?MB) / Peak CUDA Mem": peak_mb,
                "Deadline?/ Miss Rate": miss_rate,
                "Deadline?ms) / Deadline": ddl,
                "?/ #Samples": total_samples,
                "?/ #Batches": len(e2e_times_ms),
            }
        else:
            # 
            self.last_runtime_stats = None
        # ========== ??=========
        return metrics

    # ---------- loss functions ----------
    def _cross_entropy(self, logits, y):
        if self.class_weights is None:
            return F.cross_entropy(logits, y)
        return F.cross_entropy(logits, y, weight=self.class_weights)

    def loss_ce(self, logits, y, extras):
        """Pure CE"""
        ce = self._cross_entropy(logits, y)
        return ce, OrderedDict([("ce", float(ce.detach().item()))])

    def loss_gumbel(self, logits, y, extras):
        assert hasattr(self.model, "selector"), "loss_gumbel requires `model.selector`."
        ce = self._cross_entropy(logits, y)

        beta   = self._selector_beta()
        P_soft = self._selector_psoft(beta)  # (K,C),  detach

        if self.current_epoch < self.model.selector.warmup_epochs:
            orth_t = self.model.selector.orth_regularizer(P_soft)
        else:
            orth_t = self.model.selector.orth_regularizer_new(P_soft)

        orth_coef = float(self.config.get("selector_orth", 0.0))

        total = ce + orth_coef * orth_t

        # if self.current_epoch <= self.model.selector.warmup_epochs:
        #     entropy = -torch.sum(P_soft * torch.log(P_soft + 1e-8), dim=1)
        #     compactness_loss = entropy.mean()
        #     total += 1 * compactness_loss  # weak bonus during warmup

        items = OrderedDict([
            ("ce",   float(ce.detach().item())),
            ("orth", float(orth_t.detach().item())),
        ])
        return total, items
    
    def differentiable_sparsity_loss(self, P_soft, target_sparsity=0.7):
        """Differentiable sparsity regularization."""
        # ?
        # P_soft: (K, C+1)
        real_channel_probs = P_soft[:, :-1].sum(dim=0)  # ?
        expected_real_count = real_channel_probs.sum()   # 
        
        # 
        current_sparsity = 1 - (expected_real_count / P_soft.size(0))  # K
        
        # ?
        sparsity_loss = (current_sparsity - target_sparsity) ** 2
        
        return sparsity_loss
    
    def loss_dup(self, logits, y, extras):
        """
         Duplicate-penalty q.5
        L(P) = sum_c ReLU( sum_k P[k,c] - tau )
        total = CE + lambda * L(P)

        ?L2 orth_regularizer?
        """
        assert hasattr(self.model, "selector"), "loss_dup requires `model.selector`."

        # 
        ce = self._cross_entropy(logits, y)

        #  P_soft (K, C) ?
        beta   = self._selector_beta()
        P_soft = self._selector_psoft(beta)  # (K, C)

        # ? selector ? config?1.1
        sel = self.model.selector
        if hasattr(sel, "_tau"):
            tau = float(sel._tau())  # ?Dup ?_tau()
        elif hasattr(sel, "layer") and hasattr(sel.layer, "thresh"):
            tau = float(sel.layer.thresh)     # Legacy SelectionLayer ?
        else:
            tau = float(self.config.get("selector_tau_end", 1.1))

        #  L(P)im=0 ReLU(-) ?
        col_sum = P_soft.sum(dim=0)          # (C,)
        dup_t   = F.relu(col_sum - tau).sum()

        #  ?orth ?
        dup_coef = float(self.config.get("selector_dup", 0.1))  # ?YAML ?selector_dup: 0.1

        total = ce + dup_coef * dup_t

        items = OrderedDict([
            ("ce",  float(ce.detach().item())),
            ("dup", float(dup_t.detach().item())),
            ("tau", float(tau)),
        ])
        return total, items

    # ---------- helpers ----------
    def _resolve_loss_fn(self, algorithm: str) -> Callable:
        mode = (algorithm or "ce").lower()
        if mode == "ce":
            return self.loss_ce
        if mode in ("gumbel", "gumbel_selector", "ce_selreg"):
            if hasattr(self.model, "selector"):
                return self.loss_gumbel
            else:
                print("[WARN] model has no selector, fallback to 'ce'.")
                return self.loss_ce
        if mode in ("dup", "duplicate", "dup_penalty", "ce_dup"):
            if hasattr(self.model, "selector"):
                return self.loss_dup
            else:
                print("[WARN] model has no selector, fallback to 'ce'.")
                return self.loss_ce
        print(f"[WARN] Unknown loss_algorithm '{algorithm}', fallback to 'ce'.")
        return self.loss_ce
    
    def _selector_diag(self):
        m = getattr(self.model, "selector", None)
        if m is None:
            return None

        with torch.no_grad():
            P = m.current_p_soft()                 # (K,C) ?
            Hk, Hm = m.normalized_entropy(P)       # (K,), scalar
            top1 = P.max(dim=1).values.mean()      #  Top-1 
            idx  = torch.argmax(m.alpha, dim=1)    # (K,)
            k_unique = idx.unique().numel()
            alpha_norm = m.alpha.norm(p=2).item()

        # ?
        grad_norm = None
        if m.alpha.grad is not None:
            grad_norm = m.alpha.grad.norm(p=2).item()

        return {
            "beta": getattr(m, "_beta")() if hasattr(m, "_beta") else None,
            "H_mean": float(Hm),
            "Top1_mean": float(top1),
            "K_unique": int(k_unique),
            "alpha_norm": float(alpha_norm),
            "alpha_grad_norm": (float(grad_norm) if grad_norm is not None else None),
            "indices_head": idx[:8].tolist(),
        }

    def _selector_beta(self) -> float:
        return float(self.model.selector._beta()) if hasattr(self.model, "selector") and hasattr(self.model.selector, "_beta") else 1.0

    def _selector_psoft(self, beta: float) -> torch.Tensor:
        return torch.softmax(self.model.selector.alpha / max(beta, 1e-6), dim=1)

    def _forward_unpack(self, x, current_epoch: int = None):
        out = self.model(x, current_epoch)
        if isinstance(out, tuple):
            logits = out[0]
            extras = out[1:]
        else:
            logits = out
            extras = None
        return logits, extras

    def _display_progress(self, epoch: int, epochs: int, train_loss, val_loss: float, val_ba: float):
        print(f"Epoch {epoch:03d}/{epochs:3d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val BA: {val_ba:.4f}")

    def _log_training_metrics(self, metrics, mode, stage: int):
        if self.tb_logger is None:
            return
        assert mode in ["train", "val"]
        prefix = f"{mode}/stage{stage}"

        # losses
        loss_avgs, loss_names = metrics[0]
        # e.g., names=["Total","ce","Func2","Func3"]
        for name, value in zip(loss_names, loss_avgs):
            if value is not None:
                self.tb_logger.experiment.add_scalar(f"{prefix}/Loss/{name}", value, self.current_epoch)

        # val-only scalar metrics
        if mode == "val":
            AUC, BA, F1, TPR, TNR = metrics[1:]
            val_names = ["AUC", "BA", "F1", "TPR", "TNR"]
            for name, value in zip(val_names, [AUC, BA, F1, TPR, TNR]):
                if value is not None:
                    self.tb_logger.experiment.add_scalar(f"{prefix}/{name}", value, self.current_epoch)


