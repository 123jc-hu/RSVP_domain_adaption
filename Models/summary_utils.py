from typing import Optional, Type

import torch
from torch import nn

from Utils.config import build_config


def summarize_model(
    model_name: str,
    model_cls: Type[nn.Module],
    dataset_name: Optional[str] = None,
    batch_size: int = 1,
) -> None:
    """
    Print model architecture summary using torchinfo.
    Falls back to parameter counts if torchinfo is unavailable.
    """
    config = build_config(model_name, dataset_name)
    model = model_cls(config)

    n_channels = int(config["n_channels"])
    seq_len = int(config["fs"])
    input_size = (batch_size, 1, n_channels, seq_len)
    use_gpu = bool(config.get("use_gpu", True))
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    print(f"\n=== {model_name} | dataset={config['dataset']} | device={device} ===")
    print(f"input_size={input_size}")

    try:
        from torchinfo import summary

        summary(
            model,
            input_size=input_size,
            device=str(device),
            col_names=("input_size", "output_size", "num_params", "kernel_size"),
            depth=4,
            verbose=1,
        )
    except Exception as exc:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"torchinfo unavailable ({exc}).")
        print(f"total_params={total:,} | trainable_params={trainable:,}")

