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


def set_random_seed(seed: int = 2026) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


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
        "batch_size",
        "epochs",
        "patience",
        "early_stop_start_epoch",
        "cosine_t_max",
    ]
    _cast_if_present(merged_config, int_fields, int)

    float_fields = [
        "learning_rate",
        "weight_decay",
        "cosine_eta_min_factor",
        "cosine_eta_min",
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
    ]
    _cast_if_present(merged_config, float_fields, float)

    optional_int_fields = [
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
    ]
    _cast_optional(merged_config, optional_int_fields, int)

    optional_float_fields = ["source_selection_min_score"]
    _cast_optional(merged_config, optional_float_fields, float)

    bool_fields = [
        "is_training",
        "use_gpu",
        "data_mix",
        "log_runtime",
        "class_weighted_ce",
        "use_target_stream",
        "logit_adjustment",
        "pseudo_refinement",
        "rpcs_target_use_all_trials",
        "rpcs_use_correlation",
        "pccs_target_use_all_trials",
        "pccs_use_correlation",
    ]
    for field in bool_fields:
        if field in merged_config:
            merged_config[field] = _to_bool(merged_config[field])

    return merged_config

