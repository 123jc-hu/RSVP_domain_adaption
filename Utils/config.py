import torch
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from argparse import Namespace
import logging


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "on"):
            return True
        if v in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)


def _is_none_like(value):
    return value in (None, "", "None", "none", "null", "NULL")


def load_config(config_path):
    """从 yaml 文件中加载配置"""
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # return Namespace(**config)
    return config

def set_random_seed(seed=2026):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Logger:
    """日志记录器类"""
    def __init__(self, model_name):
        self.logger = logging.getLogger(model_name)
        self.logger.setLevel(logging.INFO)
        self.setup_handlers()

    def setup_handlers(self):
        """设置日志处理器"""
        # ★ 关键：清理旧的 handlers，避免重复打印
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(f'{self.logger.name}.log', mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """获取日志记录器"""
        return self.logger


def build_config(model_name, dataset_name=None):
    config_path = os.path.join("Configs", "config.yaml")
    full_config = load_config(config_path)

    if dataset_name is None:
        dataset_name = full_config["dataset"]

    # 通用配置（除去 models / datasets）
    shared_config = {k: v for k, v in full_config.items() if k not in ("models", "datasets")}

    # 模型/数据集特定配置
    model_specific   = full_config.get("models", {}).get(model_name, {})
    dataset_specific = full_config.get("datasets", {}).get(dataset_name, {})

    # 合并（后者覆盖前者）
    merged_config = {**shared_config, **model_specific, **dataset_specific}
    merged_config["model"]   = model_name
    merged_config["dataset"] = dataset_name
    # Domain adaptation experiments are cross-subject only.
    merged_config["train_mode"] = "cross-subject"

    # --- n_fold 兼容解析：None / 'None' / '' 都视为 None，其他转 int
    n_fold_val = merged_config.get("n_fold", None)
    if n_fold_val in (None, "None", "", "null", "NULL"):
        merged_config["n_fold"] = None
    else:
        merged_config["n_fold"] = int(n_fold_val)

    # --- seed 命名统一（你的 runner 用的是 'seed'，而 YAML 用的是 'random_seed'）
    if "seed" not in merged_config:
        merged_config["seed"] = merged_config.get("random_seed", 2026)

    # （可选）确保关键类型是正确的
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
    for f in int_fields:
        if f in merged_config and not _is_none_like(merged_config[f]):
            merged_config[f] = int(merged_config[f])

    float_fields = [
        "learning_rate",
        "weight_decay",
        "cosine_eta_min_factor",
        "cosine_eta_min",
        "rpcs_mean_tol",
        "rpcs_cov_eps",
        "rpcs_cov_shrinkage",
        "rpcs_score_eps",
        "pccs_mean_tol",
        "pccs_cov_eps",
        "pccs_cov_shrinkage",
        "pccs_score_eps",
    ]
    for f in float_fields:
        if f in merged_config and not _is_none_like(merged_config[f]):
            merged_config[f] = float(merged_config[f])

    optional_int_fields = [
        "source_selection_k",
        "rpcs_top_k",
        "rpcs_target_max_trials",
        "pccs_top_k",
        "pccs_target_max_trials",
    ]
    optional_int_fields.extend(
        [
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
    )
    for f in optional_int_fields:
        if f in merged_config:
            merged_config[f] = None if _is_none_like(merged_config[f]) else int(merged_config[f])

    optional_float_fields = ["source_selection_min_score"]
    for f in optional_float_fields:
        if f in merged_config:
            merged_config[f] = None if _is_none_like(merged_config[f]) else float(merged_config[f])

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
        "pccs_target_use_all_trials",
    ]
    for f in bool_fields:
        if f in merged_config:
            merged_config[f] = _to_bool(merged_config[f])
    
    return merged_config
