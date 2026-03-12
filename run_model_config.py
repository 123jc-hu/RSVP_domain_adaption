import argparse
from copy import deepcopy

from main import main
from Utils.config import build_config, load_config


def _apply_overrides(config: dict, overrides: dict) -> dict:
    merged = deepcopy(config)
    for key, value in (overrides or {}).items():
        merged[key] = value
    return merged


def run(config_path: str):
    cfg_yaml = load_config(config_path)
    model_name = str(cfg_yaml.get("model", "")).strip()
    if not model_name:
        raise ValueError(f"No model configured in {config_path}")

    dataset_name = cfg_yaml.get("dataset", None)
    cfg = build_config(model_name, dataset_name)
    cfg = _apply_overrides(cfg, dict(cfg_yaml.get("overrides", {})))
    main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Configs/eegincnet_sub1_20.yaml",
        help="Path to a single-model YAML file.",
    )
    args = parser.parse_args()
    run(args.config)
