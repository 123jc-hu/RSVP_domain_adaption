import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _is_none_like(value: Any) -> bool:
    return value in (None, "", "None", "none", "null", "NULL")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_random_seed(
    seed: int = 2026,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
    matmul_precision: str = "highest",
) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = bool(benchmark)
    cudnn.deterministic = bool(deterministic)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))


class Logger:
    """Simple logger factory for both file and console outputs."""

    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(f"{self.logger.name}.log", mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger


def _cast_if_present(config: Dict[str, Any], fields, cast_fn) -> None:
    for field in fields:
        if field in config and not _is_none_like(config[field]):
            config[field] = cast_fn(config[field])


def _cast_optional(config: Dict[str, Any], fields, cast_fn) -> None:
    for field in fields:
        if field in config:
            config[field] = None if _is_none_like(config[field]) else cast_fn(config[field])


def _normalize_int_list(value: Any):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if _is_none_like(s):
            return []
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [int(x.strip()) for x in inner.split(",") if x.strip()]
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(value)]


def build_config(model_name: str, dataset_name: str = None) -> Dict[str, Any]:
    config_path = os.path.join("Configs", "config.yaml")
    full_config = load_config(config_path)

    if dataset_name is None:
        dataset_name = full_config["dataset"]

    shared_config = {k: v for k, v in full_config.items() if k not in ("models", "datasets")}
    model_specific = full_config.get("models", {}).get(model_name, {})
    dataset_specific = full_config.get("datasets", {}).get(dataset_name, {})

    merged_config = {**shared_config, **model_specific, **dataset_specific}
    merged_config["model"] = model_name
    merged_config["dataset"] = dataset_name
    merged_config["train_mode"] = "cross-subject"

    n_fold_val = merged_config.get("n_fold", None)
    merged_config["n_fold"] = None if _is_none_like(n_fold_val) else int(n_fold_val)
    if "seed" not in merged_config:
        merged_config["seed"] = merged_config.get("random_seed", 2026)

    int_fields = [
        "sub_num",
        "n_channels",
        "fs",
        "n_class",
        "subject_batch_size",
        "dynamic_feature_sampling_warmup_epochs",
        "dynamic_feature_sampling_refresh_every",
        "dynamic_feature_sampling_source_support_size",
        "dynamic_feature_sampling_target_support_size",
        "dynamic_feature_sampling_seed",
        "dynamic_feature_sampling_batch_size",
        "dynamic_feature_sampling_min_class_samples",
        "val_batch_size",
        "epochs",
        "patience",
        "early_stop_start_epoch",
        "class_align_start_epoch",
        "class_align_min_conf_samples",
        "ccl_start_epoch",
        "prior_start_epoch",
        "cosine_t_max",
        "rpt_aug_default_n_synth",
        "source_train_bg_downsample_seed",
        "source_train_bg_downsample_positive_label",
        "source_train_bg_downsample_background_label",
        "lmmd_kernel_num",
        "dsan_num_filters",
        "dsan_temporal_kernel",
        "dsan_pool_kernel",
        "dsan_pool_stride",
        "dsan_feature_dim",
        "dsan_classifier_hidden_dim",
        "eegnet_ts_head_channels",
        "prototype_positive_label",
        "prototype_background_label",
        "posdist_positive_label",
        "posdist_start_epoch",
        "rsf_dim",
        "rsf_maxiter",
        "uot_max_iter",
    ]
    _cast_if_present(merged_config, int_fields, int)

    float_fields = [
        "learning_rate",
        "weight_decay",
        "lambda_align",
        "lambda_class_align",
        "lambda_ccl",
        "lambda_prior",
        "class_align_conf_thresh",
        "prior_min",
        "prior_max",
        "lsa_content_lambda",
        "lsa_identity_lambda",
        "lsa_style_momentum",
        "lsa_init_gate",
        "lsa_target_blend_alpha",
        "lsa_similarity_tau",
        "lsa_var_distance_weight",
        "gsldsa_target_blend_alpha",
        "gsldsa_var_distance_weight",
        "dgldsa_target_blend_alpha",
        "cosine_eta_min_factor",
        "cosine_eta_min",
        "iahm_curvature",
        "iahm_r0",
        "iahm_gamma",
        "iahm_m0",
        "iahm_margin_alpha",
        "iahm_lambda_r",
        "iahm_lambda_c",
        "iahm_lambda_m",
        "iahm_lambda_total",
        "iahm_centroid_momentum",
        "rpt_aug_beta",
        "rpt_aug_clip_factor",
        "rpt_aug_cov_eps",
        "rpt_aug_cov_shrinkage",
        "rpt_aug_correlation_eps",
        "ea_cov_eps",
        "rpcs_mean_tol",
        "rpcs_cov_eps",
        "rpcs_cov_shrinkage",
        "rpcs_correlation_eps",
        "rpcs_score_eps",
        "rpcs_target_bg_ratio",
        "rpcs_tau_d",
        "rpcs_tau_s",
        "rpcs_tau_d_percentile",
        "rpcs_tau_s_percentile",
        "pccs_mean_tol",
        "pccs_cov_eps",
        "pccs_cov_shrinkage",
        "pccs_correlation_eps",
        "pccs_score_eps",
        "pccs_target_bg_ratio",
        "pccs_tau_d",
        "pccs_tau_s",
        "pccs_tau_d_percentile",
        "pccs_tau_s_percentile",
        "source_train_bg_downsample_bg_to_pos_ratio",
        "dynamic_feature_sampling_temperature",
        "dynamic_feature_sampling_mix_alpha",
        "dynamic_feature_sampling_score_eps",
        "dynamic_feature_sampling_mmd_sigma",
        "dynamic_feature_sampling_pos_weight",
        "dynamic_feature_sampling_score_ema_decay",
        "lmmd_kernel_mul",
        "dsan_dropout",
        "eegnet_ts_cov_eps",
        "eegnet_ts_cov_shrinkage_alpha",
        "eegnet_dual_fusion_flat_weight",
        "eegnet_dual_lambda_flat_ce",
        "eegnet_dual_lambda_ts_ce",
        "eegnet_aux_ts_lambda_align",
        "prototype_lambda",
        "prototype_momentum",
        "prototype_positive_weight",
        "prototype_background_weight",
        "prototype_separation_lambda",
        "prototype_separation_margin",
        "posdist_lambda",
        "posdist_var_weight",
        "posdist_momentum",
        "rsf_tolerance",
        "rsf_domain_lambda",
        "uot_eps",
        "uot_tau_source",
        "uot_tau_target",
    ]
    _cast_if_present(merged_config, float_fields, float)

    optional_int_fields = [
        "subjects_per_batch",
        "debug_domain_batch_max_steps",
        "rpt_aug_n_synth_per_batch",
        "source_selection_k",
        "rpcs_top_k",
        "rpcs_target_max_trials",
        "pccs_top_k",
        "pccs_target_max_trials",
        "rpcs_positive_label",
        "rpcs_background_label",
        "rpcs_max_trials_per_class",
        "rpcs_min_trials_per_class",
        "rpcs_mean_max_iter",
        "rpcs_plot_top_k",
        "pccs_positive_label",
        "pccs_background_label",
        "pccs_max_trials_per_class",
        "pccs_min_trials_per_class",
        "pccs_mean_max_iter",
        "pccs_plot_top_k",
        "rsf_fit_max_trials_per_class_per_subject",
        "rsf_fit_balance_classes",
    ]
    _cast_optional(merged_config, optional_int_fields, int)

    optional_float_fields = ["source_selection_min_score", "lmmd_fix_sigma"]
    _cast_optional(merged_config, optional_float_fields, float)

    list_int_fields = [
        "rpcs_target_bg_channel_indices",
        "pccs_target_bg_channel_indices",
    ]
    for field in list_int_fields:
        if field in merged_config:
            merged_config[field] = _normalize_int_list(merged_config[field])

    bool_fields = [
        "is_training",
        "use_gpu",
        "log_runtime",
        "minimal_log",
        "debug_domain_batch",
        "deterministic_run",
        "cudnn_benchmark",
        "class_weighted_ce",
        "use_target_stream",
        "iahm_enable",
        "rpt_aug_enable",
        "rpt_aug_inject_to_ce",
        "rpt_aug_weighted_sampling",
        "rpt_aug_use_correlation",
        "logit_adjustment",
        "pseudo_refinement",
        "rpcs_target_use_all_trials",
        "rpcs_use_correlation",
        "pccs_target_use_all_trials",
        "pccs_use_correlation",
        "source_train_bg_downsample_enable",
        "ea_enable",
        "dynamic_feature_sampling_enable",
        "dynamic_feature_sampling_l2_normalize",
        "dynamic_feature_sampling_use_confidence_weight",
        "dynamic_feature_sampling_score_ema_enable",
        "class_align_use_soft_weights",
        "lmmd_use_soft_target",
        "prototype_enable",
        "posdist_enable",
        "rsf_enable",
    ]
    for field in bool_fields:
        if field in merged_config:
            merged_config[field] = _to_bool(merged_config[field])

    str_fields = [
        "iahm_space",
        "iahm_input_normalize",
        "rpt_aug_score_mode",
        "dynamic_feature_sampling_metric",
        "prior_loss_type",
        "eegnet_ts_feature_layer",
        "eegnet_dual_output_mode",
        "rsf_cov_estimator",
        "rsf_solver",
        "rsf_mode",
    ]
    for field in str_fields:
        if field in merged_config and merged_config[field] is not None:
            merged_config[field] = str(merged_config[field]).strip()

    return merged_config
