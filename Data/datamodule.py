import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from Data.source_selection import (
    normalize_score_map,
    normalize_subject_key,
    normalize_subject_list,
    select_source_subjects,
)

class NPZMemmapSubsetDataset(Dataset):
    """
    Memory-safe subset reader for .npz:
    - stream-extract x_data.npy / y_data.npy to disk cache once
    - read by numpy memmap
    - only expose requested indices
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        x_key: str = "x_data",
        y_key: str = "y_data",
        indices: Optional[Union[np.ndarray, slice]] = None,
    ):
        self.file_path = Path(file_path)
        cache_dir = self._ensure_npz_cache(self.file_path, [f"{x_key}.npy", f"{y_key}.npy"])
        self.x_data = np.load(cache_dir / f"{x_key}.npy", mmap_mode="r")
        self.y_data = np.load(cache_dir / f"{y_key}.npy", mmap_mode="r")

        n = int(self.y_data.shape[0])
        if indices is None:
            self.indices = np.arange(n, dtype=np.int64)
        elif isinstance(indices, slice):
            self.indices = np.arange(n, dtype=np.int64)[indices]
        else:
            idx = np.asarray(indices, dtype=np.int64).ravel()
            self.indices = idx[(idx >= 0) & (idx < n)]

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = self.x_data[real_idx]
        y = int(self.y_data[real_idx])
        return torch.from_numpy(x).float(), torch.tensor(y).long()

    @staticmethod
    def _ensure_npz_cache(npz_path: Path, members: List[str]) -> Path:
        cache_dir = npz_path.with_suffix("")
        cache_dir = cache_dir.parent / (cache_dir.name + ".__cache__")
        cache_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(npz_path, "r") as zf:
            names = set(zf.namelist())
            for member in members:
                out_path = cache_dir / member
                if out_path.exists():
                    continue
                if member not in names:
                    raise KeyError(f"{npz_path} missing member {member}")
                with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                    while True:
                        chunk = src.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
        return cache_dir


class UnlabeledViewDataset(Dataset):
    """View a labeled dataset as unlabeled by returning x only."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if not isinstance(item, (tuple, list)) or len(item) == 0:
            raise ValueError("Expected base dataset item as (x, y, ...).")
        return item[0]


class DualStreamDataLoader:
    """
    Yield domain-adaptation batches:
    (source_x, source_y, target_x)
    """

    def __init__(self, source_loader: DataLoader, target_loader: DataLoader, cycle_target: bool = True):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.cycle_target = cycle_target

        # Keep common DataLoader attributes for compatibility.
        self.batch_size = source_loader.batch_size
        self.dataset = source_loader.dataset
        self.drop_last = source_loader.drop_last
        self.num_workers = source_loader.num_workers

    def __len__(self):
        return len(self.source_loader)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        target_iter = iter(self.target_loader)
        for src_batch in self.source_loader:
            if not isinstance(src_batch, (tuple, list)) or len(src_batch) < 2:
                raise ValueError("Source batch must be (source_x, source_y, ...).")
            source_x, source_y = src_batch[0], src_batch[1]

            try:
                tgt_batch = next(target_iter)
            except StopIteration:
                if not self.cycle_target:
                    break
                target_iter = iter(self.target_loader)
                tgt_batch = next(target_iter)

            target_x = tgt_batch[0] if isinstance(tgt_batch, (tuple, list)) else tgt_batch
            yield source_x, source_y, target_x


class EEGDataModuleCrossSubject:
    """
    LOSO datamodule for cross-subject domain adaptation.

    Fold i:
    - target domain: subject i (all trials; unlabeled for adaptation, labeled for test)
    - source domain: all other subjects
    - source split: 80% train / 20% val (stratified per subject)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.seed = int(config.get("random_seed", 2024))
        self.source_val_ratio = float(config.get("source_val_ratio", 0.2))

        self.subject_file_map: Optional[Dict[str, str]] = None
        self.test_subject_id: Optional[int] = None
        self.test_path: Optional[str] = None
        self.subject_id: Optional[int] = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.target_unlabeled_dataset = None
        self.source_subject_keys: List[str] = []
        self.source_candidate_keys: List[str] = []
        self.source_train_class_counts: Optional[np.ndarray] = None
        self.source_selection_info: Dict[str, Any] = {}

        self._label_cache: Dict[str, np.ndarray] = {}
        self._external_source_scores: Dict[str, float] = {}
        self._external_manual_sources: List[str] = []
        self._forced_source_selection_mode: Optional[str] = None

    def prepare_data(
        self,
        test_subject_id: int,
        subject_file_map: Dict[str, str],
        source_scores: Optional[Dict[Union[str, int], float]] = None,
        manual_source_subjects: Optional[List[Union[str, int]]] = None,
        forced_source_selection_mode: Optional[str] = None,
    ):
        if not isinstance(subject_file_map, dict):
            raise TypeError("subject_file_map must be dict: {'sub1': '/path/sub1.npz', ...}")

        normalized = {}
        for key, path in subject_file_map.items():
            skey = normalize_subject_key(key)
            normalized[skey] = str(path)

        self.subject_file_map = normalized
        self.test_subject_id = int(test_subject_id)
        self.subject_id = self.test_subject_id
        held_out = f"sub{self.test_subject_id}"
        if held_out not in self.subject_file_map:
            raise KeyError(f"Missing held-out subject: {held_out}")
        self.test_path = self.subject_file_map[held_out]

        self._external_source_scores = normalize_score_map(source_scores)
        self._external_manual_sources = normalize_subject_list(manual_source_subjects)
        self._forced_source_selection_mode = (
            str(forced_source_selection_mode).strip().lower()
            if forced_source_selection_mode is not None
            else None
        )

    def setup(self):
        assert self.subject_file_map is not None, "Call prepare_data() first."
        assert self.test_subject_id is not None, "test_subject_id is not set."

        held_out = f"sub{self.test_subject_id}"
        self.source_candidate_keys = sorted(
            [k for k in self.subject_file_map.keys() if k != held_out],
            key=lambda x: int(x[3:]),
        )
        self.source_subject_keys = self._select_source_subjects(held_out, self.source_candidate_keys)
        if not self.source_subject_keys:
            raise RuntimeError("Selected source subject list is empty.")

        source_train_parts: List[Dataset] = []
        source_val_parts: List[Dataset] = []
        n_class = int(self.config.get("n_class", 2))
        class_counts = np.zeros(n_class, dtype=np.int64)
        for sub_key in self.source_subject_keys:
            n = self._subject_len(sub_key)
            all_idx = np.arange(n, dtype=np.int64)
            tr_idx, va_idx = self._split_indices_for_subject(
                self.subject_file_map[sub_key],
                all_idx,
                test_size=self.source_val_ratio,
            )
            source_train_parts.append(NPZMemmapSubsetDataset(self.subject_file_map[sub_key], indices=tr_idx))
            source_val_parts.append(NPZMemmapSubsetDataset(self.subject_file_map[sub_key], indices=va_idx))
            labels = self._load_subject_labels(self.subject_file_map[sub_key])[tr_idx]
            class_counts += np.bincount(labels.astype(np.int64), minlength=n_class)

        self.train_dataset = ConcatDataset(source_train_parts)
        self.val_dataset = ConcatDataset(source_val_parts)
        self.source_train_class_counts = class_counts

        target_n = self._subject_len(held_out)
        target_idx = np.arange(target_n, dtype=np.int64)
        target_labeled = NPZMemmapSubsetDataset(self.subject_file_map[held_out], indices=target_idx)
        self.target_unlabeled_dataset = UnlabeledViewDataset(target_labeled)
        self.test_dataset = target_labeled

    def train_dataloader(
        self,
        batch_size: int = 32,
        sampler=None,
        target_batch_size: Optional[int] = None,
        drop_last: bool = True,
    ) -> DualStreamDataLoader:
        assert self.train_dataset is not None, "Call setup() first."
        assert self.target_unlabeled_dataset is not None, "Call setup() first."

        source_loader = self.source_train_dataloader(
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
        )
        tb = int(target_batch_size) if target_batch_size is not None else int(batch_size)
        target_loader = DataLoader(
            self.target_unlabeled_dataset,
            batch_size=tb,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        return DualStreamDataLoader(source_loader, target_loader, cycle_target=True)

    def source_train_dataloader(
        self,
        batch_size: int = 32,
        sampler=None,
        drop_last: bool = True,
    ) -> DataLoader:
        """Source-only supervised loader for baseline training."""
        assert self.train_dataset is not None, "Call setup() first."
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            drop_last=drop_last,
        )

    def val_dataloader(self, batch_size: int = 64) -> DataLoader:
        assert self.val_dataset is not None, "Call setup() first."
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self, batch_size: int = 1000) -> DataLoader:
        assert self.test_dataset is not None, "Call setup() first."
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def target_adaptation_dataloader(self, batch_size: int = 64) -> DataLoader:
        assert self.target_unlabeled_dataset is not None, "Call setup() first."
        return DataLoader(
            self.target_unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    def _select_source_subjects(self, held_out_key: str, candidate_keys: List[str]) -> List[str]:
        mode = self._forced_source_selection_mode
        if not mode:
            mode = str(self.config.get("source_selection_mode", "")).strip().lower()
        if not mode:
            strategy = str(self.config.get("source_selection", "All")).strip().lower()
            strategy_map = {"all": "all", "random": "random_k", "pccs": "scores"}
            mode = strategy_map.get(strategy, "all")

        k_val = self.config.get("source_selection_k", None)
        min_score_val = self.config.get("source_selection_min_score", None)
        k = None if k_val in (None, "", "None", "none", "null", "NULL") else int(k_val)
        min_score = None if min_score_val in (None, "", "None", "none", "null", "NULL") else float(min_score_val)

        manual_sources = self._external_manual_sources
        if not manual_sources:
            manual_sources = normalize_subject_list(self.config.get("source_selection_subjects", []))

        source_scores = self._external_source_scores
        if not source_scores:
            cfg_scores = self.config.get("source_selection_scores", {})
            source_scores = normalize_score_map(cfg_scores if isinstance(cfg_scores, dict) else {})

        selected = select_source_subjects(
            candidate_subjects=candidate_keys,
            held_out_subject=held_out_key,
            mode=mode,
            seed=self.seed,
            k=k,
            manual_subjects=manual_sources,
            score_map=source_scores,
            min_score=min_score,
        )

        self.source_selection_info = {
            "held_out": held_out_key,
            "mode": mode,
            "k": k,
            "min_score": min_score,
            "num_candidates": len(candidate_keys),
            "num_selected": len(selected),
            "selected_subjects": list(selected),
        }
        return selected

    def _subject_len(self, sub_key: str) -> int:
        y = self._load_subject_labels(self.subject_file_map[sub_key])
        return int(y.shape[0])

    def _split_indices_for_subject(
        self,
        subject_file: str,
        indices: np.ndarray,
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size <= 1:
            return indices, np.empty(0, dtype=np.int64)

        labels = self._load_subject_labels(subject_file)[indices]
        labels = np.asarray(labels).reshape(-1)
        rng = np.random.default_rng(self.seed)

        # Class-wise split inside each subject:
        # for every label c, randomly move ~20% samples of c to validation.
        tr_parts: List[np.ndarray] = []
        va_parts: List[np.ndarray] = []
        for c in np.unique(labels):
            cls_idx = indices[labels == c]
            n_cls = int(cls_idx.shape[0])
            if n_cls == 0:
                continue
            perm = rng.permutation(n_cls)
            cls_idx = cls_idx[perm]

            if n_cls == 1:
                # Keep singleton class sample in train to avoid empty train class.
                n_val = 0
            else:
                n_val = int(round(n_cls * test_size))
                n_val = max(1, min(n_val, n_cls - 1))

            va_parts.append(cls_idx[:n_val])
            tr_parts.append(cls_idx[n_val:])

        train_idx = np.concatenate(tr_parts, axis=0) if tr_parts else np.empty(0, dtype=np.int64)
        val_idx = np.concatenate(va_parts, axis=0) if va_parts else np.empty(0, dtype=np.int64)

        # Shuffle final train/val indices to mix classes.
        if train_idx.size > 0:
            train_idx = train_idx[rng.permutation(train_idx.shape[0])]
        if val_idx.size > 0:
            val_idx = val_idx[rng.permutation(val_idx.shape[0])]
        return train_idx, val_idx

    def source_class_weights(self, n_class: Optional[int] = None) -> np.ndarray:
        if self.source_train_class_counts is None:
            raise RuntimeError("Call setup() before requesting source class weights.")
        if n_class is None:
            n_class = int(self.config.get("n_class", len(self.source_train_class_counts)))
        counts = self.source_train_class_counts.astype(np.float64)
        if counts.shape[0] < n_class:
            counts = np.pad(counts, (0, n_class - counts.shape[0]), mode="constant")
        counts = counts[:n_class]
        total = float(np.sum(counts))
        weights = np.zeros(n_class, dtype=np.float32)
        nonzero = counts > 0
        if total > 0 and np.any(nonzero):
            weights[nonzero] = (total / float(n_class)) / counts[nonzero]
            mean_w = float(np.mean(weights[nonzero]))
            if mean_w > 0:
                weights[nonzero] = weights[nonzero] / mean_w
        return weights

    def source_class_weight_tensor(
        self, n_class: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        w = self.source_class_weights(n_class=n_class)
        t = torch.tensor(w, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    def _load_subject_labels(self, subject_file: str) -> np.ndarray:
        subject_file = str(subject_file)
        if subject_file not in self._label_cache:
            cache_dir = NPZMemmapSubsetDataset._ensure_npz_cache(
                Path(subject_file), ["y_data.npy"]
            )
            self._label_cache[subject_file] = np.load(cache_dir / "y_data.npy", mmap_mode="r")
        return self._label_cache[subject_file]
