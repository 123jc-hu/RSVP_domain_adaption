import torch
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from argparse import Namespace
import logging

def load_config(config_path):
    """从 yaml 文件中加载配置"""
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # return Namespace(**config)
    return config

def set_random_seed(seed=2024):
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

    # --- n_fold 兼容解析：None / 'None' / '' 都视为 None，其他转 int
    n_fold_val = merged_config.get("n_fold", None)
    if n_fold_val in (None, "None", "", "null", "NULL"):
        merged_config["n_fold"] = None
    else:
        merged_config["n_fold"] = int(n_fold_val)

    # --- seed 命名统一（你的 runner 用的是 'seed'，而 YAML 用的是 'random_seed'）
    if "seed" not in merged_config:
        merged_config["seed"] = merged_config.get("random_seed", 2024)

    # （可选）确保关键类型是正确的
    int_fields = ["sub_num", "n_channels", "fs", "n_class", "batch_size", "epochs"]
    for f in int_fields:
        if f in merged_config:
            merged_config[f] = int(merged_config[f])

    float_fields = ["learning_rate", "weight_decay", "selector_beta_start", "selector_beta_end", "selector_orth"]
    for f in float_fields:
        if f in merged_config:
            merged_config[f] = float(merged_config[f])

    bool_fields = ["is_training", "use_gpu", "data_mix", "log_runtime", "use_selector", "selector_per_sample"]
    for f in bool_fields:
        if f in merged_config:
            merged_config[f] = bool(merged_config[f])
    
    if merged_config["model"] == "LMDANet":
        if merged_config['train_mode'] == 'single-subject':
            merged_config["lmda_avgpool"] = merged_config["fs"] // 10
        elif merged_config['train_mode'] == 'cross-subject':
            merged_config["lmda_avgpool"] = 1

    return merged_config
