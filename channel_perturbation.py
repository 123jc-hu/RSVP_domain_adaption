# -*- coding: utf-8 -*-
from typing import List, Dict, Sequence, Optional, Tuple
import random
import torch
from torch.utils.data import Dataset, DataLoader

class ChannelCorruptionWrapper(Dataset):
    """
    Wrap an existing EEG dataset and corrupt a fixed set of channels for every sample.

    Args:
        base_dataset: the original Dataset returning (x, y) per __getitem__.
                      x is expected to be a torch.Tensor with shape (C, T) or (1, C, T) or (C, 1, T).
        mask_channels: list of channel indices to corrupt (0-based).
        mode: 'zero' (hard failure) or 'noise' (soft failure).
        noise_std: std of Gaussian noise if mode=='noise'. Since inputs are z-scored, 1.0 is a good default.
        n_channels: total number of EEG channels (helps disambiguate channel dimension). If None, auto-infer.
        noise_seed: base seed to make noise reproducible across runs/epochs.
    """
    def __init__(
        self,
        base_dataset: Dataset,
        mask_channels: Sequence[int],
        mode: str = "zero",
        noise_std: float = 1.0,
        n_channels: Optional[int] = None,
        noise_seed: int = 0,
    ):
        super().__init__()
        assert mode in ("zero", "noise"), "mode must be 'zero' or 'noise'"
        self.ds = base_dataset
        self.mask = sorted(set(int(i) for i in mask_channels))
        self.mode = mode
        self.noise_std = float(noise_std)
        self.n_channels = n_channels
        self.noise_seed = int(noise_seed)

    def __len__(self):
        return len(self.ds)

    def _infer_channel_dim(self, x: torch.Tensor) -> int:
        """
        Infer which dimension is the channel axis.
        Priority:
          1) If n_channels provided, pick the dim whose size == n_channels.
          2) Heuristics for common EEG shapes: (C, T) -> 0; (1, C, T) -> 1; (C, 1, T) -> 0.
        """
        if self.n_channels is not None:
            for d in range(x.ndim):
                if x.shape[d] == self.n_channels:
                    return d

        if x.ndim == 2:
            # (C, T)
            return 0
        elif x.ndim == 3:
            # Try (1, C, T)
            if x.shape[1] != 1:
                return 1
            # Else likely (C, 1, T)
            return 0
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}; please provide n_channels.")

    def _apply_zero(self, x: torch.Tensor, ch_dim: int) -> torch.Tensor:
        x = x.clone()
        # Build slice to zero specific channels along ch_dim
        index = [slice(None)] * x.ndim
        index[ch_dim] = torch.tensor(self.mask, dtype=torch.long)
        x[tuple(index)] = 0.0
        return x

    def _apply_noise(self, x: torch.Tensor, ch_dim: int, idx: int) -> torch.Tensor:
        x = x.clone()
        # Prepare index for masked channels
        index = [slice(None)] * x.ndim
        index[ch_dim] = torch.tensor(self.mask, dtype=torch.long)

        # Generate reproducible noise per-sample using (noise_seed + idx)
        gen = torch.Generator(device=x.device)
        gen.manual_seed(self.noise_seed + int(idx))

        # Shape of the selected channel slice
        sel = x[tuple(index)]
        noise = torch.randn(sel.shape, generator=gen, device=sel.device, dtype=sel.dtype) * self.noise_std
        x[tuple(index)] = sel + noise
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.ds[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise RuntimeError("Base dataset must return (x, y).")

        x, y = sample[0], sample[1]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        ch_dim = self._infer_channel_dim(x)

        if self.mode == "zero":
            x = self._apply_zero(x, ch_dim)
        else:  # 'noise'
            x = self._apply_noise(x, ch_dim, idx)

        return x, y


def make_corrupted_dataloader(
    dataloader: DataLoader,
    mask_channels: Sequence[int],
    mode: str = "zero",
    shuffle: Optional[bool] = None,
    noise_std: float = 1.0,
    n_channels: Optional[int] = None,
    noise_seed: int = 0,
) -> DataLoader:
    """
    Create a new DataLoader whose dataset has the given channels corrupted.

    Args:
        dataloader: existing DataLoader (will be used to copy batch_size/num_workers/...).
        mask_channels: list of channel indices to corrupt.
        mode: 'zero' or 'noise'.
        shuffle: override shuffle behavior (True/False). If None, defaults to False unless the original
                 loader clearly used random sampling (not always detectable).
        noise_std: std of Gaussian noise when mode=='noise' (default 1.0 for z-scored inputs).
        n_channels: total number of channels to help locate channel dimension (optional).
        noise_seed: reproducibility seed for noise.

    Returns:
        A new DataLoader instance wrapping a ChannelCorruptionWrapper dataset.
    """
    # Wrap base dataset
    wrapped_ds = ChannelCorruptionWrapper(
        base_dataset=dataloader.dataset,
        mask_channels=mask_channels,
        mode=mode,
        noise_std=noise_std,
        n_channels=n_channels,
        noise_seed=noise_seed,
    )

    # Build kwargs for the new DataLoader, copying what we safely can.
    kwargs = dict(
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        pin_memory=getattr(dataloader, "pin_memory", False),
        drop_last=getattr(dataloader, "drop_last", False),
        collate_fn=dataloader.collate_fn,
        persistent_workers=getattr(dataloader, "persistent_workers", False),
    )

    prefetch_factor = getattr(dataloader, "prefetch_factor", None)
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor

    # Decide shuffle behavior
    if shuffle is None:
        # Default to False; if you rely on a custom sampler (e.g., WeightedRandomSampler),
        # please rebuild it outside and pass shuffle=False here.
        shuffle = False
    kwargs["shuffle"] = bool(shuffle)

    # IMPORTANT: we DO NOT copy the original sampler here because it may be bound to the old dataset.
    # If you use a custom sampler for class balancing, rebuild it for `wrapped_ds` and pass it
    # to DataLoader yourself instead of using `shuffle`.
    new_loader = DataLoader(dataset=wrapped_ds, **kwargs)
    return new_loader


def generate_mask_lists(
    n_channels: int = 62,
    Ms: Sequence[int] = (5, 10, 15, 20),
    n_lists: int = 10,
    random_seed: int = 1234,
) -> Dict[int, List[List[int]]]:
    """
    Generate corruption lists for multiple masking scales.

    For each M in Ms, returns `n_lists` different lists of unique channel indices (0-based).
    Lists are sampled uniformly over [0, n_channels), without replacement within a list.
    Overlap across lists is allowed (recommended for Monte-Carlo averaging).

    Args:
        n_channels: total number of EEG channels (default 62 for THU).
        Ms: iterable of masking sizes, e.g., (5,10,15,20).
        n_lists: number of lists per M (default 10).
        random_seed: global seed for reproducibility.

    Returns:
        A dict: { M: [list1, list2, ..., list_n_lists], ... } where each list is sorted ascending.
    """
    assert all(1 <= m <= n_channels for m in Ms), "Each M must be in [1, n_channels]."
    rng = random.Random(int(random_seed))

    results: Dict[int, List[List[int]]] = {}
    for M in Ms:
        group: List[List[int]] = []
        for _ in range(n_lists):
            # Sample M distinct channels without replacement
            lst = rng.sample(range(n_channels), M)
            lst.sort()
            group.append(lst)
        results[M] = group
    return results


if __name__ == "__main__":
    # 1) 生成坏道列表
    mask_dict = generate_mask_lists(n_channels=62, Ms=(5, 10, 15, 20), n_lists=10, random_seed=2025)
    # 取 M=10 的第 1 组坏道
    mask_channels = mask_dict[10][0]  # e.g., [1, 7, 12, 19, 23, 28, 33, 41, 44, 58]

    # 2) 基于已有 dataloader 构造“置零坏道”的新 DataLoader（保持 batch/num_workers 等）
    corrupted_loader_zero = make_corrupted_dataloader(
        dataloader=train_loader,  # 你现有的 DataLoader
        mask_channels=mask_channels,
        mode="zero",  # or "noise"
        shuffle=False,  # 或 True，或 None(默认 False)
        noise_std=1.0,  # 你的数据已 z-score，1.0 合理
        n_channels=62,  # 指定通道数，便于定位通道轴
        noise_seed=2025,  # 复现实验
    )

    # 3) 若要噪声模式（软失效）
    corrupted_loader_noise = make_corrupted_dataloader(
        dataloader=train_loader,
        mask_channels=mask_channels,
        mode="noise",
        noise_std=1.0,
        n_channels=62,
        noise_seed=2025,
    )
