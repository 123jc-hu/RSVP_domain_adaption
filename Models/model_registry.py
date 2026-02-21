from importlib import import_module
from typing import Any, Dict


class _LazyBackbone:
    """Lazy module proxy with .Model interface."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    @property
    def Model(self):
        return import_module(f"Models.{self.module_name}").Model


def model_dict() -> Dict[str, Any]:
    """Supported backbone models (lazy-loaded)."""
    return {
        "DeepConvNet": _LazyBackbone("DeepConvNet"),
        "EEGNet": _LazyBackbone("EEGNet"),
        "PLNet": _LazyBackbone("PLNet"),
        "EEGInception": _LazyBackbone("EEGInception"),
        "PPNN": _LazyBackbone("PPNN"),
    }
