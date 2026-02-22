from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Data.rpcs import compute_rpcs_source_scores


@dataclass
class HyRDPASettings:
    source_selection: str = "All"      # All | Random | R-PCS | SimilarityOnly | DiscrimOnly
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

        mode = (
            str(self.settings.source_selection)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
        )
        if mode == "all":
            self.last_source_selection_meta["mode"] = "all"
            return None, None, "all"
        if mode == "random":
            self.last_source_selection_meta["mode"] = "random_k"
            return None, None, "random_k"
        if mode in ("pccs", "rpcs", "similarityonly", "discrimonly"):
            def cfg(pref_key: str, legacy_key: str, default):
                if pref_key in self.config and self.config.get(pref_key) is not None:
                    return self.config.get(pref_key)
                return self.config.get(legacy_key, default)

            if mode == "similarityonly":
                score_mode = "similarity_only"
            elif mode == "discrimonly":
                score_mode = "discrim_only"
            else:
                score_mode = str(cfg("rpcs_score_mode", "pccs_score_mode", "rpcs"))
            mean_metric = str(cfg("rpcs_mean_metric", "pccs_mean_metric", "riemann")).strip().lower()
            dist_metric = str(cfg("rpcs_distance_metric", "pccs_distance_metric", mean_metric)).strip().lower()
            if mean_metric != dist_metric and self.log is not None:
                self.log.warning(
                    f"R-PCS metric mismatch requested ({mean_metric} vs {dist_metric}); "
                    f"forcing both to {mean_metric} for consistency."
                )
            dist_metric = mean_metric
            scores, details = compute_rpcs_source_scores(
                target_subject_file=subject_file_map[held_out],
                source_subject_files=source_pool,
                positive_label=int(cfg("rpcs_positive_label", "pccs_positive_label", 1)),
                background_label=int(cfg("rpcs_background_label", "pccs_background_label", 0)),
                max_trials_per_class=int(cfg("rpcs_max_trials_per_class", "pccs_max_trials_per_class", 128)),
                min_trials_per_class=int(cfg("rpcs_min_trials_per_class", "pccs_min_trials_per_class", 4)),
                seed=int(self.config.get("random_seed", 2026)),
                cov_eps=float(cfg("rpcs_cov_eps", "pccs_cov_eps", 1e-6)),
                cov_shrinkage=float(cfg("rpcs_cov_shrinkage", "pccs_cov_shrinkage", 0.0)),
                cov_estimator=str(cfg("rpcs_cov_estimator", "pccs_cov_estimator", "sample")),
                use_correlation=bool(cfg("rpcs_use_correlation", "pccs_use_correlation", True)),
                correlation_eps=float(cfg("rpcs_correlation_eps", "pccs_correlation_eps", 1e-12)),
                input_layout=str(cfg("rpcs_input_layout", "pccs_input_layout", "channel_first")),
                mean_metric=mean_metric,
                distance_metric=dist_metric,
                mean_max_iter=int(cfg("rpcs_mean_max_iter", "pccs_mean_max_iter", 20)),
                mean_tol=float(cfg("rpcs_mean_tol", "pccs_mean_tol", 1e-6)),
                score_mode=str(score_mode),
                score_eps=float(cfg("rpcs_score_eps", "pccs_score_eps", 1e-6)),
                target_use_all_trials=bool(
                    cfg("rpcs_target_use_all_trials", "pccs_target_use_all_trials", True)
                ),
                target_max_trials=cfg("rpcs_target_max_trials", "pccs_target_max_trials", None),
                target_bg_mode=str(cfg("rpcs_target_bg_mode", "pccs_target_bg_mode", "amplitude")),
                target_bg_ratio=float(cfg("rpcs_target_bg_ratio", "pccs_target_bg_ratio", 0.7)),
                tau_d=cfg("rpcs_tau_d", "pccs_tau_d", None),
                tau_s=cfg("rpcs_tau_s", "pccs_tau_s", None),
                tau_d_percentile=float(cfg("rpcs_tau_d_percentile", "pccs_tau_d_percentile", 30.0)),
                tau_s_percentile=float(cfg("rpcs_tau_s_percentile", "pccs_tau_s_percentile", 70.0)),
                return_details=True,
                return_prototypes=True,
            )
            if not scores:
                # Safe fallback for folds where scaffold score can't be computed.
                self.last_source_selection_meta["mode"] = "all"
                return None, None, "all"
            rank = details.get("ranking", [])
            preview = ", ".join(
                [
                    f"{r['subject']}:score={r['score']:.4f},D={r.get('discriminability_distance', float('nan')):.4f},S={r.get('similarity_distance', float('nan')):.4f}"
                    for r in rank[:5]
                ]
            )
            if self.log is not None:
                self.log.info(f"R-PCS ranking for sub{test_subject_id} (top-5): {preview}")
            self.last_source_selection_meta = {
                "held_out": held_out,
                "strategy": str(self.settings.source_selection),
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
