import hashlib
import os
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _cache_dir_for(npz_path: Path) -> Path:
    """
    Resolve cache directory for one npz file.

    If RSVP_NPZ_CACHE_ROOT is set, keep cache under that directory (recommended
    when source dataset is on NTFS and cache can be placed on ext4/tmpfs).
    """
    cache_root = os.environ.get("RSVP_NPZ_CACHE_ROOT", "").strip()
    if cache_root:
        src = str(npz_path.resolve())
        digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
        return Path(cache_root) / f"{npz_path.stem}-{digest}.__cache__"

    cache_dir = npz_path.with_suffix("")
    return cache_dir.parent / (cache_dir.name + ".__cache__")


def ensure_npz_cache(npz_path: Path, members: Iterable[str]) -> Path:
    """
    Extract selected members from .npz into a side cache folder for memmap access.
    """
    npz_path = Path(npz_path)
    cache_dir = _cache_dir_for(npz_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    members = list(members)

    # Fast path: avoid opening zip file when all cache members already exist.
    if all((cache_dir / member).exists() for member in members):
        return cache_dir

    with zipfile.ZipFile(npz_path, "r") as zf:
        names = set(zf.namelist())
        for member in members:
            out_path = cache_dir / member
            if out_path.exists():
                continue
            if member not in names:
                raise KeyError(f"{npz_path} missing member {member}")
            tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
            with zf.open(member, "r") as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.replace(tmp_path, out_path)
    return cache_dir


def load_npz_xy_memmap(npz_path: Path, x_key: str = "x_data", y_key: str = "y_data") -> Tuple[np.memmap, np.memmap]:
    cache_dir = ensure_npz_cache(npz_path, [f"{x_key}.npy", f"{y_key}.npy"])
    x = np.load(cache_dir / f"{x_key}.npy", mmap_mode="r")
    y = np.load(cache_dir / f"{y_key}.npy", mmap_mode="r")
    return x, y
