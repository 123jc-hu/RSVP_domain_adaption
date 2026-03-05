from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from Data.npz_io import ensure_npz_cache
from Data.source_selection import (
    SelectionManager,
    normalize_score_map,
    normalize_subject_key,
    normalize_subject_list,
)
from Data.trial_sampling import (
    get_source_positive_trials as sample_source_positive_trials,
    get_subject_trials_by_label as sample_subject_trials_by_label,
    get_target_background_trials as sample_target_background_trials,
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
        cache_dir = ensure_npz_cache(self.file_path, [f"{x_key}.npy", f"{y_key}.npy"])
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
        # Copy to writable ndarray before torch conversion to avoid memmap readonly warning.
        x = np.asarray(self.x_data[real_idx], dtype=np.float32).copy()
        y = int(self.y_data[real_idx])
        return torch.from_numpy(x), torch.tensor(y).long()

    @staticmethod
    def _ensure_npz_cache(npz_path: Path, members: List[str]) -> Path:
        # Backward-compatible wrapper.
        return ensure_npz_cache(npz_path, members)


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


class SubjectBatchDataLoader:
    """
    Subject-aware batch loader.

    One step yields:
    - choose subject keys (fixed K keys OR random K keys each step)
    - sample `per_subject_batch_size` trials from each subject
    - concatenate into global batch of size `subjects_per_batch * per_subject_batch_size`
    """

    def __init__(
        self,
        *,
        subject_datasets: Dict[str, NPZMemmapSubsetDataset],
        subject_keys_for_sampling: List[str],
        subjects_per_batch: int,
        per_subject_batch_size: int,
        steps_per_epoch: int,
        random_subjects_each_step: bool,
        seed: int = 2026,
        pin_memory: bool = False,
    ):
        if not subject_datasets:
            raise ValueError("subject_datasets must be non-empty.")
        if subjects_per_batch <= 0:
            raise ValueError("subjects_per_batch must be > 0.")
        if per_subject_batch_size <= 0:
            raise ValueError("per_subject_batch_size must be > 0.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0.")
        if not subject_keys_for_sampling:
            raise ValueError("subject_keys_for_sampling must be non-empty.")

        self.subject_datasets = dict(subject_datasets)
        self.subject_keys_for_sampling = list(subject_keys_for_sampling)
        self.subjects_per_batch = int(subjects_per_batch)
        self.per_subject_batch_size = int(per_subject_batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.random_subjects_each_step = bool(random_subjects_each_step)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)

        if self.random_subjects_each_step and self.subjects_per_batch > len(self.subject_keys_for_sampling):
            raise ValueError(
                f"subjects_per_batch={self.subjects_per_batch} is larger than sampling pool={len(self.subject_keys_for_sampling)}."
            )
        if (not self.random_subjects_each_step) and self.subjects_per_batch > len(self.subject_keys_for_sampling):
            raise ValueError(
                f"subjects_per_batch={self.subjects_per_batch} requires at least that many fixed subject keys."
            )

        self._epoch = 0

        # Keep DataLoader-like attributes for compatibility.
        self.batch_size = self.subjects_per_batch * self.per_subject_batch_size
        self.dataset = subject_datasets
        self.drop_last = True
        self.num_workers = 0

    def __len__(self):
        return self.steps_per_epoch

    def _pick_subjects(self, rng: np.random.Generator) -> List[str]:
        if self.random_subjects_each_step:
            idx = rng.choice(
                len(self.subject_keys_for_sampling),
                size=self.subjects_per_batch,
                replace=False,
            )
            return [self.subject_keys_for_sampling[int(i)] for i in idx]
        return list(self.subject_keys_for_sampling[: self.subjects_per_batch])

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(self.steps_per_epoch):
            chosen_subjects = self._pick_subjects(rng)
            x_parts: List[torch.Tensor] = []
            y_parts: List[torch.Tensor] = []

            for skey in chosen_subjects:
                ds = self.subject_datasets[skey]
                n = len(ds)
                if n <= 0:
                    continue

                local_idx = rng.integers(0, n, size=self.per_subject_batch_size, endpoint=False)
                real_idx = ds.indices[local_idx]

                x_np = np.asarray(ds.x_data[real_idx], dtype=np.float32)
                y_np = np.asarray(ds.y_data[real_idx], dtype=np.int64)

                x_parts.append(torch.from_numpy(x_np))
                y_parts.append(torch.from_numpy(y_np))

            if not x_parts:
                continue

            x_batch = torch.cat(x_parts, dim=0)
            y_batch = torch.cat(y_parts, dim=0).long()

            perm_np = rng.permutation(int(y_batch.shape[0])).astype(np.int64, copy=False)
            perm = torch.from_numpy(perm_np)
            x_batch = x_batch.index_select(0, perm)
            y_batch = y_batch.index_select(0, perm)

            if self.pin_memory and torch.cuda.is_available():
                x_batch = x_batch.pin_memory()
                y_batch = y_batch.pin_memory()

            yield x_batch, y_batch


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
        self.seed = int(config.get("random_seed", 2026))
        self.source_val_ratio = float(config.get("source_val_ratio", 0.2))
        self.num_workers = int(config.get("num_workers", 0))
        self.pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
        self.persistent_workers = bool(config.get("persistent_workers", self.num_workers > 0))
        pf = config.get("prefetch_factor", None)
        self.prefetch_factor = None if pf in (None, "", "None", "none", "null", "NULL") else int(pf)

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
        self.source_train_dataset_map: Dict[str, NPZMemmapSubsetDataset] = {}
        self.source_candidate_train_counts: Dict[str, int] = {}
        self.train_steps_per_epoch: Optional[int] = None
        self.train_subject_batch_size: Optional[int] = None
        self.train_subjects_per_batch: Optional[int] = None
        self.train_subject_batch_policy: Optional[str] = None
        self.train_subject_sampling_keys: List[str] = []

        self._label_cache: Dict[str, np.ndarray] = {}
        self._external_source_scores: Dict[str, float] = {}
        self._external_manual_sources: List[str] = []
        self._forced_source_selection_mode: Optional[str] = None
        self.selection_manager = SelectionManager(seed=self.seed)

    def _loader_common_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

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
        self.source_train_dataset_map = {}
        self.source_candidate_train_counts = {}
        n_class = int(self.config.get("n_class", 2))
        class_counts = np.zeros(n_class, dtype=np.int64)
        selected_set = set(self.source_subject_keys)
        for sub_key in self.source_candidate_keys:
            n = self._subject_len(sub_key)
            all_idx = np.arange(n, dtype=np.int64)
            tr_idx, va_idx = self._split_indices_for_subject(
                self.subject_file_map[sub_key],
                all_idx,
                test_size=self.source_val_ratio,
            )
            self.source_candidate_train_counts[sub_key] = int(tr_idx.shape[0])

            if sub_key in selected_set:
                train_ds = NPZMemmapSubsetDataset(self.subject_file_map[sub_key], indices=tr_idx)
                val_ds = NPZMemmapSubsetDataset(self.subject_file_map[sub_key], indices=va_idx)
                self.source_train_dataset_map[sub_key] = train_ds
                source_train_parts.append(train_ds)
                source_val_parts.append(val_ds)
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
            drop_last=False,
            **self._loader_common_kwargs(),
        )
        return DualStreamDataLoader(source_loader, target_loader, cycle_target=True)

    def source_train_dataloader(
        self,
        batch_size: int = 32,
        sampler=None,
        drop_last: bool = True,
    ):
        """Source-only supervised loader for baseline training."""
        assert self.train_dataset is not None, "Call setup() first."
        assert self.source_train_dataset_map, "Call setup() first."

        if bool(self.config.get("subject_batching", False)) and sampler is None:
            return self._subject_batch_train_loader(fallback_batch_size=batch_size)

        self.train_steps_per_epoch = None
        self.train_subject_batch_size = int(batch_size)
        self.train_subjects_per_batch = 1
        self.train_subject_batch_policy = "flat_concat"
        self.train_subject_sampling_keys = list(self.source_subject_keys)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=drop_last,
            **self._loader_common_kwargs(),
        )

    def _subject_batch_train_loader(self, fallback_batch_size: int):
        subjects_per_batch_raw = self.config.get("subjects_per_batch", None)
        if subjects_per_batch_raw in (None, "", "None", "none", "null", "NULL"):
            subjects_per_batch = len(self.source_subject_keys)
        else:
            subjects_per_batch = int(subjects_per_batch_raw)
        if subjects_per_batch <= 0:
            raise ValueError("subjects_per_batch must be > 0.")

        subject_batch_size_raw = self.config.get("subject_batch_size", None)
        if subject_batch_size_raw in (None, "", "None", "none", "null", "NULL"):
            per_subject_batch_size = int(fallback_batch_size)
        else:
            per_subject_batch_size = int(subject_batch_size_raw)
        if per_subject_batch_size <= 0:
            raise ValueError("subject_batch_size must be > 0.")

        all_source_mode = str(self.source_selection_info.get("mode", "")).strip().lower() == "all"
        all_source_policy = str(self.config.get("all_source_batch_policy", "random_k")).strip().lower()

        if all_source_mode and all_source_policy == "random_k":
            policy = "random_k_each_step"
            random_subjects_each_step = True
            subject_keys_for_sampling = list(self.source_subject_keys)
        else:
            policy = "fixed_k_subjects"
            random_subjects_each_step = False
            if len(self.source_subject_keys) < subjects_per_batch:
                raise ValueError(
                    f"Selected source subjects={len(self.source_subject_keys)} < subjects_per_batch={subjects_per_batch}."
                )
            subject_keys_for_sampling = list(self.source_subject_keys[:subjects_per_batch])

        ref_mode = str(self.config.get("epoch_step_reference", "all_candidates")).strip().lower()
        override_steps_raw = self.config.get("epoch_steps_override", None)
        if override_steps_raw not in (None, "", "None", "none", "null", "NULL"):
            steps_per_epoch = int(override_steps_raw)
        else:
            global_batch = subjects_per_batch * per_subject_batch_size
            if global_batch <= 0:
                raise ValueError("Global batch size must be > 0.")

            if ref_mode in ("all_candidates", "all", "candidates"):
                total_ref_samples = int(sum(self.source_candidate_train_counts.values()))
            elif ref_mode in ("selected_sources", "selected", "pool"):
                total_ref_samples = int(
                    sum(len(self.source_train_dataset_map[k]) for k in subject_keys_for_sampling)
                )
            else:
                raise ValueError(f"Unknown epoch_step_reference: {ref_mode}")

            steps_per_epoch = int(np.ceil(float(total_ref_samples) / float(global_batch)))
        steps_per_epoch = max(1, int(steps_per_epoch))

        self.train_steps_per_epoch = steps_per_epoch
        self.train_subject_batch_size = per_subject_batch_size
        self.train_subjects_per_batch = subjects_per_batch
        self.train_subject_batch_policy = policy
        self.train_subject_sampling_keys = list(subject_keys_for_sampling)

        return SubjectBatchDataLoader(
            subject_datasets=self.source_train_dataset_map,
            subject_keys_for_sampling=subject_keys_for_sampling,
            subjects_per_batch=subjects_per_batch,
            per_subject_batch_size=per_subject_batch_size,
            steps_per_epoch=steps_per_epoch,
            random_subjects_each_step=random_subjects_each_step,
            seed=self.seed + int(self.test_subject_id or 0) * 1000,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, batch_size: int = 64) -> DataLoader:
        assert self.val_dataset is not None, "Call setup() first."
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **self._loader_common_kwargs(),
        )

    def test_dataloader(self, batch_size: int = 1000) -> DataLoader:
        assert self.test_dataset is not None, "Call setup() first."
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **self._loader_common_kwargs(),
        )

    def target_adaptation_dataloader(self, batch_size: int = 64) -> DataLoader:
        assert self.target_unlabeled_dataset is not None, "Call setup() first."
        return DataLoader(
            self.target_unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            **self._loader_common_kwargs(),
        )

    def _select_source_subjects(self, held_out_key: str, candidate_keys: List[str]) -> List[str]:
        mode = self._forced_source_selection_mode
        if not mode:
            mode = str(self.config.get("source_selection_mode", "")).strip().lower()
        if not mode:
            strategy = (
                str(self.config.get("source_selection", "All"))
                .strip()
                .lower()
                .replace("-", "")
                .replace("_", "")
            )
            strategy_map = {
                "all": "all",
                "random": "random_k",
                "pccs": "scores",
                "rpcs": "scores",
                "similarityonly": "scores",
                "discrimonly": "scores",
            }
            mode = strategy_map.get(strategy, "all")

        k_val = self.config.get("source_selection_k", None)
        if k_val in (None, "", "None", "none", "null", "NULL"):
            k_val = self.config.get("rpcs_top_k", None)
        if k_val in (None, "", "None", "none", "null", "NULL"):
            k_val = self.config.get("pccs_top_k", None)
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

        selected = self.selection_manager.select(
            candidate_subjects=candidate_keys,
            held_out_subject=held_out_key,
            policy=mode,
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

    def get_subject_trials_by_label(
        self,
        *,
        subject_key: str,
        label: int,
        max_trials: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return trials of one subject for one label as ndarray [N, C, T].
        """
        if self.subject_file_map is None:
            raise RuntimeError("Call prepare_data() first.")
        use_seed = self.seed if seed is None else int(seed)
        return sample_subject_trials_by_label(
            subject_file_map=self.subject_file_map,
            subject_key=subject_key,
            label=int(label),
            max_trials=max_trials,
            seed=use_seed,
        )

    def get_source_positive_trials(
        self,
        *,
        positive_label: int = 1,
        subject_keys: Optional[List[str]] = None,
        max_trials_per_subject: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return source-domain positive trials per subject:
        {sub_key: [N_i, C, T]}.
        """
        if self.subject_file_map is None:
            raise RuntimeError("Call prepare_data() first.")
        keys = list(subject_keys) if subject_keys is not None else list(self.source_subject_keys)
        use_seed = self.seed if seed is None else int(seed)
        return sample_source_positive_trials(
            subject_file_map=self.subject_file_map,
            source_subject_keys=keys,
            positive_label=int(positive_label),
            max_trials_per_subject=max_trials_per_subject,
            seed=use_seed,
        )

    def get_target_background_trials(
        self,
        *,
        background_label: int = 0,
        target_use_all_trials: bool = True,
        target_max_trials: Optional[int] = None,
        target_bg_mode: str = "amplitude",
        target_bg_ratio: float = 0.7,
        target_bg_channel_indices: Optional[List[int]] = None,
        min_keep: int = 8,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return target-domain background-like trial subset [N, C, T].
        """
        if self.test_subject_id is None or self.subject_file_map is None:
            raise RuntimeError("Call prepare_data() first.")
        use_seed = self.seed if seed is None else int(seed)
        return sample_target_background_trials(
            subject_file_map=self.subject_file_map,
            test_subject_id=int(self.test_subject_id),
            background_label=int(background_label),
            target_use_all_trials=bool(target_use_all_trials),
            target_max_trials=target_max_trials,
            target_bg_mode=str(target_bg_mode),
            target_bg_ratio=float(target_bg_ratio),
            target_bg_channel_indices=target_bg_channel_indices,
            min_keep=max(1, int(min_keep)),
            seed=use_seed,
        )

    def _load_subject_labels(self, subject_file: str) -> np.ndarray:
        subject_file = str(subject_file)
        if subject_file not in self._label_cache:
            cache_dir = ensure_npz_cache(Path(subject_file), ["y_data.npy"])
            self._label_cache[subject_file] = np.load(cache_dir / "y_data.npy", mmap_mode="r")
        return self._label_cache[subject_file]
