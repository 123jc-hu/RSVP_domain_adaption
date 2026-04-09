from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from Data.rpcs import compute_rpcs_source_scores
try:
    from Data.simple_subject_scores import (
        compute_corr_all_source_scores,
        compute_jsd_all_source_scores,
        compute_mmd_all_source_scores,
    )
except ImportError:
    compute_corr_all_source_scores = None
    compute_jsd_all_source_scores = None
    compute_mmd_all_source_scores = None


def _parse_optional_int(value, default: Optional[int] = None) -> Optional[int]:
    if value in (None, "", "None", "none", "null", "NULL"):
        return default
    return int(value)


@dataclass
class HyRDPASettings:
    source_selection: str = "All"      # All | Random | R-PCS | SimilarityOnly | DiscrimOnly | CorrAll | MMDAll | JSDAll
    embedding_space: str = "Euclidean" # Hyperbolic | Euclidean
    psi_method: str = "None"           # Tangent_Space | Time_Domain | None
    training_mode: str = "End2End"     # Decoupled | End2End
    logit_adjustment: bool = False
    pseudo_refinement: bool = False
    iahm_enable: bool = False
    rpt_aug_enable: bool = False
    rpt_aug_beta: float = 1.0

    @classmethod
    def from_config(cls, cfg: Dict) -> "HyRDPASettings":
        return cls(
            source_selection=str(cfg.get("source_selection", "All")),
            embedding_space=str(cfg.get("embedding_space", "Euclidean")),
            psi_method=str(cfg.get("psi_method", "None")),
            training_mode=str(cfg.get("training_mode", "End2End")),
            logit_adjustment=bool(cfg.get("logit_adjustment", False)),
            pseudo_refinement=bool(cfg.get("pseudo_refinement", False)),
            iahm_enable=bool(cfg.get("iahm_enable", False)),
            rpt_aug_enable=bool(cfg.get("rpt_aug_enable", False)),
            rpt_aug_beta=float(cfg.get("rpt_aug_beta", 1.0)),
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
            f"logit_adjustment={s.logit_adjustment}, pseudo_refinement={s.pseudo_refinement}, "
            f"iahm_enable={s.iahm_enable}, "
            f"rpt_aug_enable={s.rpt_aug_enable}, rpt_aug_beta={s.rpt_aug_beta}"
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

        def _compute_score_based_inputs(score_mode_name: str):
            score_mode_name = str(score_mode_name).strip().lower().replace("-", "").replace("_", "")
            if score_mode_name in ("corrall",):
                if compute_corr_all_source_scores is None:
                    raise ImportError("Data.simple_subject_scores.compute_corr_all_source_scores is unavailable.")
                scores, details = compute_corr_all_source_scores(
                    target_subject_file=subject_file_map[held_out],
                    source_subject_files=source_pool,
                    max_trials=_parse_optional_int(self.config.get("simple_score_max_trials", 128)),
                    seed=int(self.config.get("random_seed", 2026)),
                    return_details=True,
                )
                label = "Corr-All"
            elif score_mode_name in ("mmdall",):
                if compute_mmd_all_source_scores is None:
                    raise ImportError("Data.simple_subject_scores.compute_mmd_all_source_scores is unavailable.")
                sigma2_val = self.config.get("mmd_sigma2", None)
                sigma2 = None if sigma2_val in (None, "", "None", "none", "null") else float(sigma2_val)
                scores, details = compute_mmd_all_source_scores(
                    target_subject_file=subject_file_map[held_out],
                    source_subject_files=source_pool,
                    max_trials=_parse_optional_int(self.config.get("simple_score_max_trials", 128)),
                    seed=int(self.config.get("random_seed", 2026)),
                    sigma2=sigma2,
                    score_eps=float(self.config.get("mmd_score_eps", 1e-6)),
                    return_details=True,
                )
                label = "MMD-All"
            elif score_mode_name in ("jsdall",):
                if compute_jsd_all_source_scores is None:
                    raise ImportError("Data.simple_subject_scores.compute_jsd_all_source_scores is unavailable.")
                value_range_cfg = self.config.get("jsd_value_range", None)
                value_range = None
                if isinstance(value_range_cfg, (list, tuple)) and len(value_range_cfg) == 2:
                    value_range = (float(value_range_cfg[0]), float(value_range_cfg[1]))
                scores, details = compute_jsd_all_source_scores(
                    target_subject_file=subject_file_map[held_out],
                    source_subject_files=source_pool,
                    max_trials=_parse_optional_int(self.config.get("simple_score_max_trials", 128)),
                    seed=int(self.config.get("random_seed", 2026)),
                    num_bins=int(self.config.get("jsd_num_bins", 128)),
                    value_range=value_range,
                    score_eps=float(self.config.get("jsd_score_eps", 1e-6)),
                    return_details=True,
                )
                label = "JSD-All"
            else:
                def cfg(pref_key: str, legacy_key: str, default):
                    if pref_key in self.config and self.config.get(pref_key) is not None:
                        return self.config.get(pref_key)
                    return self.config.get(legacy_key, default)

                if score_mode_name in ("similarityonly",):
                    score_mode = "similarity_only"
                elif score_mode_name in ("discrimonly",):
                    score_mode = "discrim_only"
                elif score_mode_name in ("rpcs", "pccs"):
                    score_mode = str(cfg("rpcs_score_mode", "pccs_score_mode", "rpcs"))
                else:
                    raise ValueError(f"Unknown score source for all-source weighted sampling: {score_mode_name}")
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
                    target_use_all_trials=bool(cfg("rpcs_target_use_all_trials", "pccs_target_use_all_trials", True)),
                    target_max_trials=cfg("rpcs_target_max_trials", "pccs_target_max_trials", None),
                    target_bg_mode=str(cfg("rpcs_target_bg_mode", "pccs_target_bg_mode", "amplitude")),
                    target_bg_ratio=float(cfg("rpcs_target_bg_ratio", "pccs_target_bg_ratio", 0.7)),
                    target_bg_channel_indices=cfg("rpcs_target_bg_channel_indices", "pccs_target_bg_channel_indices", None),
                    tau_d=cfg("rpcs_tau_d", "pccs_tau_d", None),
                    tau_s=cfg("rpcs_tau_s", "pccs_tau_s", None),
                    tau_d_percentile=float(cfg("rpcs_tau_d_percentile", "pccs_tau_d_percentile", 30.0)),
                    tau_s_percentile=float(cfg("rpcs_tau_s_percentile", "pccs_tau_s_percentile", 70.0)),
                    return_details=True,
                    return_prototypes=True,
                )
                label = "R-PCS"
            return scores, details, label

        mode = (
            str(self.settings.source_selection)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
        )
        if mode == "all":
            sampling_mode = str(self.config.get("all_source_sampling_mode", "uniform")).strip().lower()
            if sampling_mode == "score_weighted":
                score_source = str(self.config.get("all_source_score_source", "mmd_all"))
                scores, details, label = _compute_score_based_inputs(score_source)
                rank = list(details.get("ranking", []) or [])
                preview = ", ".join(
                    [
                        f"{r['subject']}:score={r['score']:.4f},S={r.get('similarity_distance', float('nan')):.4f}"
                        for r in rank[:5]
                    ]
                )
                if self.log is not None:
                    self.log.info(f"{label} ranking for sub{test_subject_id} (top-5): {preview}")
                self.last_source_selection_meta = {
                    "held_out": held_out,
                    "strategy": str(self.settings.source_selection),
                    "mode": "all",
                    "details": details,
                    "ranking": rank,
                    "sampling_mode": "score_weighted",
                    "score_source": str(score_source),
                }
                return scores, None, "all"
            self.last_source_selection_meta["mode"] = "all"
            return None, None, "all"
        if mode == "random":
            self.last_source_selection_meta["mode"] = "random_k"
            return None, None, "random_k"
        if mode in ("corrall", "mmdall", "jsdall"):
            scores, details, label = _compute_score_based_inputs(mode)
            if not scores:
                self.last_source_selection_meta["mode"] = "all"
                return None, None, "all"
            rank = details.get("ranking", [])
            preview = ", ".join(
                [
                    f"{r['subject']}:score={r['score']:.4f},S={r.get('similarity_distance', float('nan')):.4f}"
                    for r in rank[:5]
                ]
            )
            if self.log is not None:
                self.log.info(f"{label} ranking for sub{test_subject_id} (top-5): {preview}")
            self.last_source_selection_meta = {
                "held_out": held_out,
                "strategy": str(self.settings.source_selection),
                "mode": "scores",
                "details": details,
                "ranking": rank,
            }
            return scores, None, "scores"
        if mode in ("pccs", "rpcs", "similarityonly", "discrimonly"):
            scores, details, label = _compute_score_based_inputs(mode)
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

    def build_rpt_aug_config(self) -> Dict[str, Any]:
        """
        Return normalized RPT-Aug config for fold-level integration.
        """
        cfg = self.config
        return {
            "enable": bool(cfg.get("rpt_aug_enable", False)),
            "beta": float(cfg.get("rpt_aug_beta", 1.0)),
            "default_n_synth": int(cfg.get("rpt_aug_default_n_synth", 100)),
            "weighted_sampling": bool(cfg.get("rpt_aug_weighted_sampling", True)),
            "score_mode": str(cfg.get("rpt_aug_score_mode", "uniform")),
            "inject_to_ce": bool(cfg.get("rpt_aug_inject_to_ce", True)),
            "clip_factor": float(cfg.get("rpt_aug_clip_factor", 3.0)),
            "cov_eps": float(cfg.get("rpt_aug_cov_eps", cfg.get("rpcs_cov_eps", 1e-6))),
            "cov_estimator": str(cfg.get("rpt_aug_cov_estimator", cfg.get("rpcs_cov_estimator", "sample"))),
            "cov_shrinkage": float(cfg.get("rpt_aug_cov_shrinkage", cfg.get("rpcs_cov_shrinkage", 0.0))),
            "use_correlation": bool(cfg.get("rpt_aug_use_correlation", cfg.get("rpcs_use_correlation", True))),
            "correlation_eps": float(cfg.get("rpt_aug_correlation_eps", cfg.get("rpcs_correlation_eps", 1e-12))),
            "input_layout": str(cfg.get("rpt_aug_input_layout", cfg.get("rpcs_input_layout", "channel_first"))),
        }

    def build_iahm_config(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "enable": bool(cfg.get("iahm_enable", False)),
            "space": str(cfg.get("iahm_space", "hyperbolic")),
            "input_normalize": str(cfg.get("iahm_input_normalize", "none")),
            "curvature": float(cfg.get("iahm_curvature", -1.0)),
            "r0": float(cfg.get("iahm_r0", 1.0)),
            "gamma": float(cfg.get("iahm_gamma", 1.0)),
            "m0": float(cfg.get("iahm_m0", 1.0)),
            "margin_alpha": float(cfg.get("iahm_margin_alpha", 0.25)),
            "lambda_r": float(cfg.get("iahm_lambda_r", 1.0)),
            "lambda_c": float(cfg.get("iahm_lambda_c", 0.5)),
            "lambda_m": float(cfg.get("iahm_lambda_m", 1.0)),
            "lambda_total": float(cfg.get("iahm_lambda_total", 1.0)),
            "centroid_momentum": float(cfg.get("iahm_centroid_momentum", 0.1)),
        }
