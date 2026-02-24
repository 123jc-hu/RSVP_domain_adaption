import zipfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def ensure_npz_cache(npz_path: Path, members: Iterable[str]) -> Path:
    """
    Extract selected members from .npz into a side cache folder for memmap access.
    """
    npz_path = Path(npz_path)
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


def load_npz_xy_memmap(npz_path: Path, x_key: str = "x_data", y_key: str = "y_data") -> Tuple[np.memmap, np.memmap]:
    cache_dir = ensure_npz_cache(npz_path, [f"{x_key}.npy", f"{y_key}.npy"])
    x = np.load(cache_dir / f"{x_key}.npy", mmap_mode="r")
    y = np.load(cache_dir / f"{y_key}.npy", mmap_mode="r")
    return x, y

