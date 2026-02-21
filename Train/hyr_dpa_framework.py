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

        mode = str(self.settings.source_selection).strip().lower()
        if mode == "all":
            return None, None, "all"
        if mode == "random":
            return None, None, "random_k"
        if mode == "pccs":
            scores = compute_pccs_source_scores(
                target_subject_file=subject_file_map[held_out],
                source_subject_files=source_pool,
                positive_label=int(self.config.get("pccs_positive_label", 1)),
                background_label=int(self.config.get("pccs_background_label", 0)),
                max_trials_per_class=int(self.config.get("pccs_max_trials_per_class", 128)),
                seed=int(self.config.get("random_seed", 2024)) + int(test_subject_id),
            )
            if not scores:
                # Safe fallback for folds where scaffold score can't be computed.
                return None, None, "all"
            return scores, None, "scores"
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

