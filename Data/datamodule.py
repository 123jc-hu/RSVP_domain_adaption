import os
import pickle
import re
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset, Sampler
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from typing import Any, Optional, Dict, Tuple, List, Union
import random
from sklearn.model_selection import StratifiedShuffleSplit
import zipfile

# channel_list = [7, 8, 9, 10, 11, 15, 16, 17, 19, 20, 21, 25, 27, 29, 35, 37, 41, 43, 45, 47, 49, 58, 59, 60]
# channel_list = [43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61]
# channel_list = [27, 39, 43, 45, 47, 53, 56, 58, 59, 60]
# channel_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 21, 22, 25, 32, 35, 50, 41, 44, 47, 48, 49, 50, 53, 54, 57, 58, 59, 60, 61]
# channel_list = [0, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 36, 37, 39, 41, 43, 45, 47, 49, 52, 54, 58, 59, 60] # Hoffmann-32
# channel_list = [9, 17, 19, 25, 27, 29, 35, 37, 41, 43, 45, 47, 49, 58, 59, 60]  # Hoffmann-16


class EEGDataModuleSingleSubject:
    """Memory-optimized EEG data module for single subject experiments"""

    # ========= 新增：只负责“加 0 通道”的包装 Dataset =========
    class _ZeroChannelDataset(Dataset):
        """
        包装一个已有 dataset：
        - 输入: base_ds，__getitem__ 返回 (x, y, yd, ...) 之类
        - 输出: (x_new, y, yd, ...)，其中 x_new 比 x 多一行全 0 通道
        """

        def __init__(self, base_ds: Dataset):
            self.base_ds = base_ds

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            item = self.base_ds[idx]

            if not isinstance(item, (tuple, list)):
                raise TypeError(
                    f"期望 dataset[idx] 返回 tuple/list，实际得到 {type(item)}，"
                    "当前实现假定第一个元素是 x。"
                )

            x, *others = item  # x, y, yd, ...

            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"_ZeroChannelDataset 要求 x 为 torch.Tensor，实际为 {type(x)}"
                )

            # ---- 在通道维加一行 0：支持 (C, T) 或 (1, C, T) ----
            if x.dim() == 2:
                # x: (C, T)
                C, T = x.shape
                zero_chan = x.new_zeros(1, T)  # (1, T)
                x_new = torch.cat([x, zero_chan], dim=0)  # -> (C+1, T)

            elif x.dim() == 3:
                # 兼容 (1, C, T)
                _, C, T = x.shape
                zero_chan = x.new_zeros(1, 1, T)  # (1, 1, T)
                x_new = torch.cat([x, zero_chan], dim=1)  # -> (1, C+1, T)

            else:
                raise ValueError(
                    f"期望单个样本 x 形状为 (C, T) 或 (1, C, T)，但得到 {x.shape}"
                )

            return (x_new, *others)

    # ========= 下面是你原来的代码，保持不动 =========
    def __init__(self, config: Dict[str, Any]):
        self.fs = config["fs"]
        self.seed = getattr(config, "random_seed", 2024)
        self.dataset = config["dataset"]

        # Cache configuration
        self.dataset_config = {
            "THU": {"train": slice(0, 8000), "test": slice(8000, 16000), "group_num": 8000, "block_num": 4000},
            "CAS": {"train": slice(0, 4200), "test": slice(4200, 8400), "group_num": 4200, "block_num": 1400},
            "GIST": {"train": slice(0, 480), "test": slice(480, 600), "group_num": 480, "block_num": 240},
            "TCTR_1": {"train": slice(0, 2400), "test": slice(2400, 3000), "group_num": 2400, "block_num": 1200},
            "TCTR_2": {"train": slice(0, 2400), "test": slice(2400, 3000), "group_num": 2400, "block_num": 1200},
            "TCTR_A": {"train": slice(0, 3200), "test": slice(3200, 4000), "group_num": 3200, "block_num": 1600},
            "TCTR_B": {"train": slice(0, 3200), "test": slice(3200, 4000), "group_num": 3200, "block_num": 1600},
        }

        # State variables
        self.subject_id = None
        self.selected_indices = None
        self.subject_file = None
        self.mode = None

    def prepare_data(self, test_subject_id: int, data_path: str, indices_path: str, mode: str = "normal"):
        """Optimized data preparation with caching"""
        self.subject_id = test_subject_id
        self.mode = mode

        with open(indices_path, "rb") as f:
            all_indices = pickle.load(f)
            key = f"sub{self.subject_id}"
            if key not in all_indices:
                raise ValueError(f"Missing index info for subject: {key}")
            self.selected_indices = {key: all_indices[key]}

        self.subject_file = data_path

    def setup(self, stage: int = 1, is_split_domains: bool = False):
        cfg = self.dataset_config[self.dataset]
        x_key, y_key = "x_data", "y_data"
        dataset_class = LazyEEGDataset if self.mode == "10band" else InMemoryEEGDataset

        if stage == 1:
            # Use full training range
            idx = np.arange(cfg["train"].start, cfg["train"].stop)

            if is_split_domains:
                # Apply domain splitting logic to stage 1
                group_num = cfg["group_num"]
                block_num = cfg["block_num"]
                num_domain = group_num // block_num

                # Ensure we only use data that can form complete blocks
                total_domain_data = num_domain * block_num
                idx = idx[idx < total_domain_data]

                # Split into domains
                domain_indices_list = np.array_split(idx, num_domain)
                train_datasets = []
                val_datasets = []

                for domain_indices in domain_indices_list:
                    train_idx, val_idx = self._split_indices(domain_indices)
                    train_datasets.append(dataset_class(self.subject_file, x_key, y_key, train_idx))
                    val_datasets.append(dataset_class(self.subject_file, x_key, y_key, val_idx))

                self.train_dataset = ConcatDatasetWithDomainLabel(train_datasets)
                self.val_dataset = ConcatDatasetWithDomainLabel(val_datasets)
            else:
                # Original stage 1 logic without domain splitting
                train_indices, val_indices = self._split_indices(idx)
                self.train_dataset = dataset_class(self.subject_file, x_key, y_key, train_indices)
                self.val_dataset = dataset_class(self.subject_file, x_key, y_key, val_indices)


        elif stage == 2:
            # 用选择的 index 对 cfg["train"] 做子集索引
            idx = self.selected_indices[f"sub{self.subject_id}"]
            group_num = cfg["group_num"]
            block_num = cfg["block_num"]

            num_domain = group_num // block_num
            total_domain_data = num_domain * block_num
            idx = idx[idx < total_domain_data]  # 忽略无法构成完整 block 的数据

            if is_split_domains:
                # 自动划分多个 domain，并分别划分 train/val
                domain_indices_list = np.array_split(idx, num_domain)
                train_datasets = []
                val_datasets = []

                for domain_indices in domain_indices_list:
                    train_idx, val_idx = self._split_indices(domain_indices)
                    train_datasets.append(dataset_class(self.subject_file, x_key, y_key, train_idx))
                    val_datasets.append(dataset_class(self.subject_file, x_key, y_key, val_idx))

                self.train_dataset = ConcatDatasetWithDomainLabel(train_datasets)
                self.val_dataset = ConcatDatasetWithDomainLabel(val_datasets)
            else:
                train_indices, val_indices = self._split_indices(idx)
                self.train_dataset = dataset_class(self.subject_file, x_key, y_key, train_indices)
                self.val_dataset = dataset_class(self.subject_file, x_key, y_key, val_indices)

        # 测试集
        self.test_dataset = dataset_class(self.subject_file, x_key, y_key, cfg["test"])

    def _split_indices(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.load(self.subject_file, mmap_mode='r')["y_data"][indices]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
        train_idx, val_idx = next(sss.split(indices, labels))
        return indices[train_idx], indices[val_idx]

    # ========= 关键改动：三个 dataloader 接口 =========

    def train_dataloader(
            self,
            batch_size: int = 1,
            sampler=None,
            add_zero_channel: bool = False,
    ) -> DataLoader:
        """
        add_zero_channel=False:
            完全等价于你原来的实现，保证历史结果不变
        add_zero_channel=True:
            仅在 dataset 外包一层 _ZeroChannelDataset，采样顺序不变
        """
        dataset = self.train_dataset
        if add_zero_channel:
            dataset = self._ZeroChannelDataset(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,  # Avoid multiprocessing overhead for small datasets
            drop_last=(sampler is None)
        )

    def val_dataloader(
            self,
            batch_size: int = 1,
            add_zero_channel: bool = False,
    ) -> DataLoader:
        dataset = self.val_dataset
        if add_zero_channel:
            dataset = self._ZeroChannelDataset(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def test_dataloader(
            self,
            batch_size: int = 1000,
            add_zero_channel: bool = False,
    ) -> DataLoader:
        dataset = self.test_dataset
        if add_zero_channel:
            dataset = self._ZeroChannelDataset(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )


class EEGDataModuleCrossSubject:
    """
    Memory-optimized EEG data module for cross-subject (LOSO) experiments.

    Usage pattern:
        dm = EEGDataModuleCrossSubject(config)
        dm.prepare_data(
            test_subject_id=heldout_id,
            subject_file_map={"sub1": "...npz", "sub2": "...npz", ...},
            indices_path=None,                 # 可选：stage-2 的已选索引（dict: subX->np.ndarray）
            mode="normal",                     # "normal" 或 "10band"
            train_range="train"                # "train" 或 "all"（train+test）
        )
        dm.setup(stage=1)  # 或 stage=2（若你要在第二阶段使用 selected_indices）
        train_loader = dm.train_dataloader(batch_size=..., sampler=..., add_zero_channel=...)
        val_loader   = dm.val_dataloader(batch_size=..., add_zero_channel=...)
        test_loader  = dm.test_dataloader(batch_size=..., add_zero_channel=...)
    """

    # ========= NEW: 只负责“加 0 通道”的包装 Dataset =========
    class _ZeroChannelDataset(Dataset):
        """
        包装一个已有 dataset：
        - 输入: base_ds，__getitem__ 返回 (x, y, yd, ...) 之类
        - 输出: (x_new, y, yd, ...)，其中 x_new 比 x 多一行全 0 通道
        """

        def __init__(self, base_ds: Dataset):
            self.base_ds = base_ds

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            item = self.base_ds[idx]

            if not isinstance(item, (tuple, list)):
                raise TypeError(
                    f"期望 dataset[idx] 返回 tuple/list，实际得到 {type(item)}，"
                    "当前实现假定第一个元素是 x。"
                )

            x, *others = item  # x, y, yd, ...

            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"_ZeroChannelDataset 要求 x 为 torch.Tensor，实际为 {type(x)}"
                )

            # ---- 在通道维加一行 0：支持 (C, T) 或 (1, C, T) ----
            if x.dim() == 2:
                # x: (C, T)
                C, T = x.shape
                zero_chan = x.new_zeros(1, T)             # (1, T)
                x_new = torch.cat([x, zero_chan], dim=0)  # -> (C+1, T)

            elif x.dim() == 3:
                # 兼容 (1, C, T)
                _, C, T = x.shape
                zero_chan = x.new_zeros(1, 1, T)          # (1, 1, T)
                x_new = torch.cat([x, zero_chan], dim=1)  # -> (1, C+1, T)

            else:
                raise ValueError(
                    f"期望单个样本 x 形状为 (C, T) 或 (1, C, T)，但得到 {x.shape}"
                )

            return (x_new, *others)

    def __init__(self, config: Dict[str, Any]):
        self.fs = config["fs"]
        self.seed = getattr(config, "random_seed", 2024)
        self.dataset = config["dataset"]

        # 与单被试版本同风格的数据段配置（不需要 group_num / block_num）
        self.dataset_config = {
            "THU":   {"train": slice(0, 8000),  "test": slice(8000, 16000)},
            "CAS":   {"train": slice(0, 4200),  "test": slice(4200, 8400)},
            "GIST":  {"train": slice(0, 480),   "test": slice(480, 600)},
            "TCTR_1":{"train": slice(0, 2400),  "test": slice(2400, 3000)},
            "TCTR_2":{"train": slice(0, 2400),  "test": slice(2400, 3000)},
            "TCTR_A":{"train": slice(0, 3200),  "test": slice(3200, 4000)},
            "TCTR_B":{"train": slice(0, 3200),  "test": slice(3200, 4000)},
        }

        # 状态变量
        self.test_subject_id: Optional[int] = None
        self.subject_file_map: Optional[Dict[str, str]] = None
        self.selected_indices: Optional[Dict[str, np.ndarray]] = None  # 可选：stage-2 使用
        self.mode: str = "normal"
        self.train_range: str = "train"  # "train" or "all"

        # 数据集句柄
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(
            self,
            test_subject_id: int,
            subject_file_map: Dict[str, str],
            indices_path: Optional[str] = None,
            mode: str = "normal",
            train_range: str = "train",
    ):
        """
        Args:
            test_subject_id: 留作测试的被试编号（整数）
            subject_file_map: dict, 形如 {"sub1": "/path/sub1.npz", "sub2": "...", ...}
            indices_path: 可选 pkl 路径，内容为 {"subX": np.ndarray([...]), ...}
            mode: "normal" 使用 InMemoryEEGDataset；"10band" 使用 LazyEEGDataset
            train_range: "train" 仅用配置的 train 段；"all" 使用 train+test（合并两段）
        """
        self.subject_id = test_subject_id
        # —— 强校验：cross-subject 必须传 dict；防止误传单路径 ——
        if not isinstance(subject_file_map, dict):
            raise TypeError(
                "EEGDataModuleCrossSubject.prepare_data 需要 'subject_file_map' 为字典："
                "{'sub1': '/path/sub1.npz', 'sub2': '/path/sub2.npz', ...}。"
                "如果你在 single-subject 模式，请使用 EEGDataModuleSingleSubject，并传 data_path。"
            )

        # 规范化 key 与 path，并检查 key 格式
        norm_map: Dict[str, str] = {}
        for k, v in subject_file_map.items():
            kk = f"sub{k}" if isinstance(k, int) else str(k)
            if not re.match(r"^sub\d+$", kk):
                raise ValueError(f"subject_file_map 的 key 必须形如 'sub<数字>'，收到：{kk}")
            norm_map[kk] = str(v)
        self.subject_file_map = norm_map

        heldout_key = f"sub{int(test_subject_id)}"
        if heldout_key not in self.subject_file_map:
            raise KeyError(f"找不到被试 {heldout_key} 的数据路径，请确认 subject_file_map 是否完整。")

        self.test_subject_id = int(test_subject_id)
        self.mode = mode
        self.train_range = train_range

        # 加载（可选）selected_indices：要求为 dict[str -> np.ndarray]
        self.selected_indices = None
        if indices_path is not None and os.path.isfile(indices_path):
            with open(indices_path, "rb") as f:
                idx_obj = pickle.load(f)
            if not isinstance(idx_obj, dict):
                raise TypeError("indices_path 文件内容必须是 dict：{'subX': np.ndarray([...]), ...}")
            # 只保留存在于 subject_file_map 的键，并做 dtype 规范化
            self.selected_indices = {
                (f"sub{k}" if isinstance(k, int) else str(k)): np.asarray(v, dtype=np.int64)
                for k, v in idx_obj.items()
                if (f"sub{k}" if isinstance(k, int) else str(k)) in self.subject_file_map
            }

        # 便于日志/调试：保存测试被试的路径
        self.test_path = self.subject_file_map[heldout_key]

    def setup(self, stage: int = 1, is_split_domains: bool = False):
        assert self.subject_file_map is not None, "Call prepare_data() first."

        # 便于外部读取/日志
        self.test_subject_id_attr = int(self.test_subject_id)
        self.test_path = self.subject_file_map[f"sub{self.test_subject_id}"]

        cfg = self.dataset_config[self.dataset]
        x_key, y_key = "x_data", "y_data"
        dataset_class = NPZMemmapSubsetDataset

        heldout_key = f"sub{self.test_subject_id}"

        # === 训练用的被试列表：排除被测被试，并做“自然顺序”排序，保证可复现 ===
        train_sub_keys = [k for k in self.subject_file_map.keys() if k != heldout_key]
        train_sub_keys.sort(key=lambda s: int(s[3:]))  # 'sub12' -> 12
        self.train_subject_keys = train_sub_keys  # 可选：供外部日志使用

        # === 测试集：固定用 held-out 的 test 段，确保与训练无重叠 ===
        test_sl = cfg["test"]
        test_idx = self._safe_range(heldout_key, test_sl.start, test_sl.stop)
        self.test_dataset = dataset_class(self.subject_file_map[heldout_key], x_key, y_key, test_idx)

        # === 为每个训练被试构造其 train/val 数据集（各被试内部 80/20 分层），最后跨被试拼接 ===
        train_datasets_per_sub: List[Any] = []
        val_datasets_per_sub: List[Any] = []

        for dom_id, sub_key in enumerate(train_sub_keys):
            # 根据 train_range 构建基础索引
            if self.train_range == "train":
                tr_sl = cfg["train"]
                base_idx = self._safe_range(sub_key, tr_sl.start, tr_sl.stop)
            elif self.train_range == "all":
                tr_sl = cfg["train"]; te_sl = cfg["test"]
                idx_train = self._safe_range(sub_key, tr_sl.start, tr_sl.stop)
                idx_test  = self._safe_range(sub_key, te_sl.start, te_sl.stop)
                base_idx = np.unique(np.concatenate([idx_train, idx_test], axis=0))  # 去重防边界重叠
            else:
                raise ValueError(f"Unknown train_range: {self.train_range}")

            # stage==2 且提供了选中索引：与基础段求交集（允许某些被试没提供条目）
            if stage == 2 and self.selected_indices is not None:
                sel = self.selected_indices.get(sub_key, None)
                if sel is not None and len(sel) > 0:
                    base_idx = np.intersect1d(base_idx, sel, assume_unique=False)

            # 该被试内部分层 80/20
            sub_train_idx, sub_val_idx = self._split_indices_for_subject(
                subject_file=self.subject_file_map[sub_key],
                indices=base_idx
            )

            ds_tr = dataset_class(self.subject_file_map[sub_key], x_key, y_key, sub_train_idx)
            ds_va = dataset_class(self.subject_file_map[sub_key], x_key, y_key, sub_val_idx)

            train_datasets_per_sub.append(ds_tr)
            val_datasets_per_sub.append(ds_va)

        # === 跨被试拼接 ===
        if is_split_domains:
            # 显式检查类是否已导入，避免 NameError
            try:
                ConcatDatasetWithDomainLabel
            except NameError as e:
                raise ImportError(
                    "is_split_domains=True，但未导入 ConcatDatasetWithDomainLabel。请先 from ... import 该类。"
                ) from e

            self.train_dataset = ConcatDatasetWithDomainLabel(train_datasets_per_sub)
            self.val_dataset = ConcatDatasetWithDomainLabel(val_datasets_per_sub)
        else:
            self.train_dataset = ConcatDataset(train_datasets_per_sub)
            self.val_dataset = ConcatDataset(val_datasets_per_sub)

    # ----------- helpers -----------

    def _y_len(self, sub_key: str) -> int:
        return int(np.load(self.subject_file_map[sub_key], mmap_mode='r')["y_data"].shape[0])

    def _safe_range(self, sub_key: str, start: int, stop: int) -> np.ndarray:
        n = self._y_len(sub_key)
        s = 0 if start is None else int(start)
        e = n if stop  is None else int(min(stop, n))
        if s >= e:
            return np.empty(0, dtype=np.int64)
        return np.arange(s, e, dtype=np.int64)

    def _build_base_indices_for_subject(self, sub_key: str, cfg: Dict[str, slice]) -> np.ndarray:
        """根据 train_range 选择该被试用于训练/验证的基础索引"""
        if self.train_range == "train":
            idx = np.arange(cfg["train"].start, cfg["train"].stop)
        elif self.train_range == "all":
            # 合并 train + test 段
            idx_train = np.arange(cfg["train"].start, cfg["train"].stop)
            idx_test = np.arange(cfg["test"].start, cfg["test"].stop)
            idx = np.concatenate([idx_train, idx_test], axis=0)
        else:
            raise ValueError(f"Unknown train_range: {self.train_range}")
        return idx

    def _split_indices_for_subject(self, subject_file: str, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对单个被试的 indices 按标签做分层 80/20 划分"""
        labels = np.load(subject_file, mmap_mode='r')["y_data"][indices]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
        train_idx, val_idx = next(sss.split(indices, labels))
        return indices[train_idx], indices[val_idx]

    # ----------- dataloaders -----------

    def train_dataloader(
        self,
        batch_size: int = 1,
        sampler=None,
        add_zero_channel: bool = False,   # NEW
    ) -> DataLoader:
        """
        add_zero_channel=False:
            完全等价于原始行为，顺序和历史结果保持一致
        add_zero_channel=True:
            仅在 dataset 外包一层 _ZeroChannelDataset，采样顺序不变，只是 x 多一行 0 通道
        """
        dataset = self.train_dataset
        if add_zero_channel:
            dataset = self._ZeroChannelDataset(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            drop_last=True
        )

    def val_dataloader(
        self,
        batch_size: int = 1,
        add_zero_channel: bool = False,   # NEW
    ) -> DataLoader:
        dataset = self.val_dataset
        if add_zero_channel:
            dataset = self._ZeroChannelDataset(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def test_dataloader(
        self,
        batch_size: int = 1000,
        add_zero_channel: bool = False,   # NEW
    ) -> DataLoader:
        dataset = self.test_dataset
        if add_zero_channel:
            dataset = self._ZeroChannelDataset(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    
class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_to_domain_id = {}
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        return img, target, domain_id


class RandomDomainSampler(Sampler):
    """
    Randomly sample N domains, then randomly select K samples in each domain to form a mini-batch of size N x K.

    Args:
        data_source (ConcatDataset): Dataset that contains data from multiple domains.
        batch_size (int): Total number of samples in a mini-batch (N x K).
        n_domains_per_batch (int): Number of domains to include in each mini-batch (N).
    """
    def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch: int):
        super().__init__()
        # Number of domains in the dataset
        self.n_domains_in_dataset = len(data_source.cumulative_sizes)
        n_domains_per_batch = min(n_domains_per_batch, self.n_domains_in_dataset)
        # Number of domains to include in each mini-batch
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch, \
            "Number of domains in dataset must be >= n_domains_per_batch"

        # Store indices for each domain
        self.indices_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            self.indices_per_domain.append(list(range(start, end)))
            start = end

        # Ensure batch_size is divisible by n_domains_per_batch
        assert batch_size % n_domains_per_batch == 0, \
            "batch_size must be divisible by n_domains_per_batch"
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.batch_size_per_domain_per_class = self.batch_size_per_domain // 2
        self.length = self._calculate_total_samples()

    def _calculate_total_samples(self):
        return sum(len(indices) for indices in self.indices_per_domain)

    def __iter__(self):
        # Create a copy of the indices for each domain to avoid modifying the original
        indices_per_domain = [domain_indices.copy() for domain_indices in self.indices_per_domain]
        batch_indices = []

        # Select domains for the current mini-batch
        if self.n_domains_per_batch == self.n_domains_in_dataset:
            # If n_domains_per_batch equals total domains, use all domains in order
            selected_domains = list(range(self.n_domains_in_dataset))
        else:
            # Otherwise, randomly select domains
            selected_domains = random.sample(range(self.n_domains_in_dataset), self.n_domains_per_batch)

        while True:
            # Select samples from each selected domain
            for domain in selected_domains:
                domain_indices = indices_per_domain[domain]
                class_len = len(domain_indices) // 2
                if len(domain_indices) < self.batch_size_per_domain:
                    # If not enough samples, sample with replacement
                    selected_indices_0 = np.random.choice(domain_indices[:class_len], self.batch_size_per_domain_per_class, replace=True)
                    selected_indices_1 = np.random.choice(domain_indices[class_len:], self.batch_size_per_domain_per_class, replace=True)
                else:
                    selected_indices_0 = random.sample(domain_indices[:class_len], self.batch_size_per_domain_per_class)
                    selected_indices_1 = random.sample(domain_indices[class_len:], self.batch_size_per_domain_per_class)
                selected_indices = selected_indices_0 + selected_indices_1
                batch_indices.extend(selected_indices)

                # Remove selected indices to avoid duplicates
                for idx in selected_indices:
                    if idx in indices_per_domain[domain]:
                        indices_per_domain[domain].remove(idx)

                # # Stop if any domain runs out of samples
                # if len(indices_per_domain[domain]) < self.batch_size_per_domain:
                #     return iter(batch_indices)
            can_continue = all(
                len(indices_per_domain[domain]) >= self.batch_size_per_domain
                for domain in selected_domains
            )
            if not can_continue:
                return iter(batch_indices)

    def __len__(self):
        return self.length


class BalancedRandomDomainSampler(Sampler):
    """
    Alternative implementation with cleaner sampling logic.
    """
    def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch: int, 
                 labels, domain_labels):
        super().__init__()

        self.n_domains_in_dataset = len(data_source.cumulative_sizes)
        self.n_domains_per_batch = min(n_domains_per_batch, self.n_domains_in_dataset)
        self.batch_size = batch_size
        
        assert batch_size % self.n_domains_per_batch == 0
        assert (batch_size // self.n_domains_per_batch) % 2 == 0
        
        self.samples_per_domain = batch_size // self.n_domains_per_batch
        self.samples_per_domain_per_class = self.samples_per_domain // 2
        
        # Organize indices by (domain, class)
        self.domain_class_indices = self._organize_indices(data_source, labels, domain_labels)
        
    def _organize_indices(self, data_source, labels, domain_labels):
        domain_class_indices = {}
        
        start_idx = 0
        for domain_idx, end_idx in enumerate(data_source.cumulative_sizes):
            domain_class_indices[domain_idx] = {0: [], 1: []}
            
            for idx in range(start_idx, end_idx):
                class_label = int(labels[idx])
                domain_class_indices[domain_idx][class_label].append(idx)
            
            start_idx = end_idx
            
        return domain_class_indices
    
    def __iter__(self):
        # Create working copy
        working_indices = {}
        for domain in self.domain_class_indices:
            working_indices[domain] = {
                0: self.domain_class_indices[domain][0].copy(),
                1: self.domain_class_indices[domain][1].copy()
            }
        
        while True:
            # Select domains for this batch
            available_domains = []
            for domain in range(self.n_domains_in_dataset):
                if (len(working_indices[domain][0]) >= self.samples_per_domain_per_class and
                    len(working_indices[domain][1]) >= self.samples_per_domain_per_class):
                    available_domains.append(domain)
            
            if len(available_domains) < self.n_domains_per_batch:
                break
                
            selected_domains = random.sample(available_domains, self.n_domains_per_batch)
            
            batch_indices = []
            for domain in selected_domains:
                # Sample from each class in this domain
                for class_label in [0, 1]:
                    selected = random.sample(
                        working_indices[domain][class_label], 
                        self.samples_per_domain_per_class
                    )
                    batch_indices.extend(selected)
                    
                    # Remove selected indices
                    for idx in selected:
                        working_indices[domain][class_label].remove(idx)
            
            # Shuffle the batch to mix domains and classes
            random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        total = 0
        for domain in self.domain_class_indices:
            for class_label in [0, 1]:
                total += len(self.domain_class_indices[domain][class_label])
        return total
    
    
class LazyEEGDataset(Dataset):
    def __init__(self, file_path: Union[str, Path], x_key: str, y_key: str, indices: Optional[Union[np.ndarray, slice]] = None):
        self.file_path = str(file_path)
        self.x_key = x_key
        self.y_key = y_key

        # 只加载一次，并保存 mmap 引用
        data = np.load(self.file_path, mmap_mode="r")
        self.x_data = data[self.x_key]  # mmap对象
        self.y_data = data[self.y_key]

        # 统一 indices
        if indices is None:
            self.indices = np.arange(len(self.y_data))
        elif isinstance(indices, slice):
            self.indices = np.arange(len(self.y_data))[indices]
        else:
            self.indices = np.array(indices)

        self.dataset_len = len(self.indices)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.x_data[real_idx]  # 不会加载整个文件，只访问当前样本
        y = self.y_data[real_idx]
        return torch.from_numpy(x).float(), torch.tensor(y).long()


class InMemoryEEGDataset(Dataset):
    def __init__(self, file_path: Union[str, Path], x_key: str, y_key: str, indices: Optional[Union[np.ndarray, slice]] = None):
        data = np.load(file_path)
        # self.x_data = data[x_key][:, channel_list, :]  # 只选定通道
        self.x_data = data[x_key]
        self.y_data = data[y_key]

        if indices is None:
            self.indices = np.arange(len(self.y_data))
        elif isinstance(indices, slice):
            self.indices = np.arange(len(self.y_data))[indices]
        else:
            self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.x_data[real_idx]
        y = self.y_data[real_idx]
        return torch.from_numpy(x).float(), torch.tensor(y).long()
    

class NPZMemmapSubsetDataset(Dataset):
    """
    安全内存占用的数据集：
    - 从 .npz 内将 <x_key>.npy / <y_key>.npy 流式解压到磁盘缓存（.__cache__）
    - 通过 np.load(..., mmap_mode='r') 内存映射访问，不把整块数组放入 RAM
    - 仅使用给定 indices 进行取样
    """
    def __init__(self, file_path: Union[str, Path], x_key: str, y_key: str,
                 indices: Optional[Union[np.ndarray, slice]] = None):
        self.file_path = Path(file_path)
        self.x_key = x_key
        self.y_key = y_key

        # 1) 确保缓存存在，并把 .npz 内的 <key>.npy 成员“流式解压”出来
        cache_dir = self._ensure_npz_cache(self.file_path, [f"{x_key}.npy", f"{y_key}.npy"])

        # 2) mmap 读取缓存的 .npy（不会占用大块内存）
        self.x_data = np.load(cache_dir / f"{x_key}.npy", mmap_mode='r')
        self.y_data = np.load(cache_dir / f"{y_key}.npy", mmap_mode='r')

        # 3) 索引处理与越界过滤
        n = int(self.y_data.shape[0])
        if indices is None:
            self.indices = np.arange(n, dtype=np.int64)
        elif isinstance(indices, slice):
            self.indices = np.arange(n, dtype=np.int64)[indices]  # 自动截断
        else:
            idx = np.asarray(indices, dtype=np.int64).ravel()
            self.indices = idx[(idx >= 0) & (idx < n)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.x_data[real_idx]
        y = self.y_data[real_idx]
        return torch.from_numpy(x).float(), torch.tensor(int(y)).long()

    @staticmethod
    def _ensure_npz_cache(npz_path: Path, member_names: list) -> Path:
        """
        把 .npz 内的指定成员（<key>.npy）流式解压到磁盘缓存目录：
        - 不会把整个数组读入内存
        - 只要缓存存在就复用
        """
        cache_dir = npz_path.with_suffix("")  # e.g., /.../sub1
        cache_dir = cache_dir.parent / (cache_dir.name + ".__cache__")
        cache_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(npz_path, 'r') as zf:
            namelist = set(zf.namelist())
            for member in member_names:
                out_path = cache_dir / member
                if out_path.exists():
                    continue
                # .npz 内部成员名通常就是 "<key>.npy"
                if member not in namelist:
                    raise KeyError(f"{npz_path} 内缺少成员 {member}，请确认保存时键名是否为该 key")
                # 流式解压（分块 16MB）
                with zf.open(member, 'r') as src, open(out_path, 'wb') as dst:
                    while True:
                        chunk = src.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
        return cache_dir


def make_zero_channel_dataloader(
    dataloader,
    add_zero_channel: bool = True,
    shuffle: bool = False,
    random_seed: int = 2025,
):
    """
    输入:
        dataloader: 原始 DataLoader，batch 输出形如:
                    x: (B, C, T)
                    其他: (B, ...)，例如 y, yd 等
        add_zero_channel: True 时在通道维新增一个全零通道；False 时保持 x 不变
        shuffle:          新 dataloader 是否打乱（一般你这里 False）
        random_seed:      DataLoader 的随机种子（保证对比实验顺序一致）

    输出:
        new_loader: 一个新的 DataLoader，其中:
            - 若 add_zero_channel=True:
                x: (B, C+1, T)
            - 若 add_zero_channel=False:
                x: (B, C, T) 原样
            - 其他字段 (y, yd, ...) 完全保持不变
    """

    base_dataset = dataloader.dataset

    class ZeroChannelDataset(Dataset):
        def __init__(self, base_ds, add_zero_channel: bool):
            self.base_ds = base_ds
            self.add_zero_channel = add_zero_channel

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            # 原始返回可以是 (x, y) / (x, y, yd) / (x, y, yd, ...)
            item = self.base_ds[idx]

            if not isinstance(item, (tuple, list)):
                raise TypeError(
                    f"期望 dataset[idx] 返回 tuple/list，实际得到 {type(item)}，"
                    "当前实现约定第一个元素为 x。"
                )

            x, *others = item

            # 不加通道的情况：直接原样返回，保证顺序一致，只是包装了一层
            if not self.add_zero_channel:
                return (x, *others)

            # -------- 下面是加通道的情况 --------
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"add_zero_channel=True 时要求 x 为 torch.Tensor，实际为 {type(x)}"
                )

            if x.dim() == 2:
                # x: (C, T)
                C, T = x.shape
                zero_chan = x.new_zeros(1, T)             # (1, T)
                x_new = torch.cat([x, zero_chan], dim=0)  # -> (C+1, T)

            elif x.dim() == 3:
                # 兼容 (1, C, T) 的情况
                _, C, T = x.shape
                zero_chan = x.new_zeros(1, 1, T)          # (1, 1, T)
                x_new = torch.cat([x, zero_chan], dim=1)  # -> (1, C+1, T)
            else:
                raise ValueError(
                    f'期望单个样本 x 形状为 (C, T) 或 (1, C, T)，但得到 {x.shape}'
                )

            return (x_new, *others)

    wrapped_dataset = ZeroChannelDataset(base_dataset, add_zero_channel)

    # 保证可复现（即使 shuffle=False 也统一设置）
    g = torch.Generator()
    g.manual_seed(random_seed)

    new_loader = DataLoader(
        wrapped_dataset,
        batch_size=dataloader.batch_size,
        shuffle=shuffle,
        num_workers=dataloader.num_workers,
        pin_memory=getattr(dataloader, "pin_memory", False),
        drop_last=dataloader.drop_last,
        collate_fn=dataloader.collate_fn,
        worker_init_fn=getattr(dataloader, "worker_init_fn", None),
        generator=g,
    )

    return new_loader