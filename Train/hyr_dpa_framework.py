from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Data.pccs import compute_pccs_source_scores


@dataclass
class HyRDPASettings:
    source_selection: str = "All"      # PCCS | All | Random
    embedding_space: str = "Euclidean" # Hyperbolic | Euclidean
    psi_method: str = "None"           # Tangent_Space | Time_Domain | None
    training_mode: str = "End2End"     # Decoupled | End2End
    logit_adjustment: bool = False
    pseudo_refinement: bool = False

    @classmethod
    def from_config(cls, cfg: Dict) -> "HyRDPASettings":
        return cls(
            source_selection=str(cfg.get("source_selection", "All")),
            embedding_space=str(cfg.get("embedding_space", "Euclidean")),
            psi_method=str(cfg.get("psi_method", "None")),
            training_mode=str(cfg.get("training_mode", "End2End")),
            logit_adjustment=bool(cfg.get("logit_adjustment", False)),
            pseudo_refinement=bool(cfg.get("pseudo_refinement", False)),
        )


class HyRDPAScaffold:
    """
    Config-driven scaffold for HyR-DPA stages.
    This class intentionally provides minimal runnable hooks (not full algorithm).
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.log = logger
        self.settings = HyRDPASettings.from_config(config)
        self.last_source_selection_meta: Dict = {}

    def describe(self) -> str:
        s = self.settings
        return (
            f"source_selection={s.source_selection}, embedding_space={s.embedding_space}, "
            f"psi_method={s.psi_method}, training_mode={s.training_mode}, "
            f"logit_adjustment={s.logit_adjustment}, pseudo_refinement={s.pseudo_refinement}"
        )

    def build_source_selection_inputs(
        self,
        *,
        test_subject_id: int,
        subject_file_map: Dict[str, str],
    ) -> Tuple[Optional[Dict[str, float]], Optional[List[str]], Optional[str]]:
        """
        Return fold-level source-selection inputs for datamodule.
        Returns: (source_scores, manual_source_subjects, forced_mode)
        """
        held_out = f"sub{int(test_subject_id)}"
        source_pool = {k: v for k, v in subject_file_map.items() if k != held_out}
        self.last_source_selection_meta = {
            "held_out": held_out,
            "strategy": str(self.settings.source_selection),
            "mode": None,
            "ranking": [],
        }

        mode = str(self.settings.source_selection).strip().lower()
        if mode == "all":
            self.last_source_selection_meta["mode"] = "all"
            return None, None, "all"
        if mode == "random":
            self.last_source_selection_meta["mode"] = "random_k"
            return None, None, "random_k"
        if mode == "pccs":
            scores, details = compute_pccs_source_scores(
                target_subject_file=subject_file_map[held_out],
                source_subject_files=source_pool,
                positive_label=int(self.config.get("pccs_positive_label", 1)),
                background_label=int(self.config.get("pccs_background_label", 0)),
                max_trials_per_class=int(self.config.get("pccs_max_trials_per_class", 128)),
                min_trials_per_class=int(self.config.get("pccs_min_trials_per_class", 4)),
                seed=int(self.config.get("random_seed", 2026)),
                cov_eps=float(self.config.get("pccs_cov_eps", 1e-6)),
                cov_shrinkage=float(self.config.get("pccs_cov_shrinkage", 0.0)),
                mean_metric=str(self.config.get("pccs_mean_metric", "airm")),
                distance_metric=str(self.config.get("pccs_distance_metric", "airm")),
                mean_max_iter=int(self.config.get("pccs_mean_max_iter", 20)),
                mean_tol=float(self.config.get("pccs_mean_tol", 1e-6)),
                return_details=True,
                return_prototypes=True,
            )
            if not scores:
                # Safe fallback for folds where scaffold score can't be computed.
                self.last_source_selection_meta["mode"] = "all"
                return None, None, "all"
            rank = details.get("ranking", [])
            preview = ", ".join([f"{r['subject']}:{r['score']:.4f}" for r in rank[:5]])
            if self.log is not None:
                self.log.info(f"PCCS ranking for sub{test_subject_id} (top-5): {preview}")
            self.last_source_selection_meta = {
                "held_out": held_out,
                "strategy": "PCCS",
                "mode": "scores",
                "details": details,
                "ranking": rank,
            }
            return scores, None, "scores"
        self.last_source_selection_meta["mode"] = None
        return None, None, None

    def stage1_feature_alignment(self) -> None:
        """
        Stage-1 placeholder:
        - source supervised loss
        - optional PSI for target-stylized positives
        - optional alignment in selected embedding space
        """
        return

    def stage2_classifier_rectification(self) -> None:
        """
        Stage-2 placeholder for decoupled mode:
        - optional logit adjustment
        - optional pseudo-label consistency refinement
        """
        return
