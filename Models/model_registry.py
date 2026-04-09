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
        "DSANCCL": _LazyBackbone("DSANCCL"),
        "EEGConformer": _LazyBackbone("EEGConformer"),
        "EEGIncNet": _LazyBackbone("EEGIncNet"),
        "EEGNetAuxTS": _LazyBackbone("EEGNetAuxTS"),
        "EEGNetDSBN": _LazyBackbone("EEGNetDSBN"),
        "EEGNetLDSA": _LazyBackbone("EEGNetLDSA"),
        "EEGNetLSA": _LazyBackbone("EEGNetLSA"),
        "EEGNetLSAv2": _LazyBackbone("EEGNetLSAv2"),
        "EEGNetDualHead": _LazyBackbone("EEGNetDualHead"),
        "EEGNetDGLDSA": _LazyBackbone("EEGNetDGLDSA"),
        "EEGNetGSLDSA": _LazyBackbone("EEGNetGSLDSA"),
        "EEGNet": _LazyBackbone("EEGNet"),
        "EEGNetSWLDSA": _LazyBackbone("EEGNetSWLDSA"),
        "EEGNetTS": _LazyBackbone("EEGNetTS"),
        "PLNet": _LazyBackbone("PLNet"),
        "EEGInception": _LazyBackbone("EEGInception"),
        "PPNN": _LazyBackbone("PPNN"),
    }
