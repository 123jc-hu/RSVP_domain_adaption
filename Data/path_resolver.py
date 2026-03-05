import os
from pathlib import Path
from typing import Any, Dict


def resolve_dataset_dir(config: Dict[str, Any]) -> Path:
    """
    Resolve dataset directory path for current dataset and sampling rate.
    """
    root = Path(
        config.get(
            "dataset_root",
            os.environ.get("RSVP_DATASET_ROOT", "E:/learning_projects/few-shot-learning_RSVP/Dataset"),
        )
    )

    dataset_name = str(config["dataset"])
    fs = int(config["fs"])
    if "_" in dataset_name:
        dataset, task = dataset_name.split("_", 1)
        return root / dataset / f"Standard_{fs}Hz" / f"task{task}"
    return root / dataset_name / f"Standard_{fs}Hz"
