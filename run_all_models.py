from main import main
from Utils.config import build_config


DEFAULT_MODELS = ["DeepConvNet", "EEGNet", "EEGInception", "PLNet", "PPNN"]
DEFAULT_DATASETS = ["THU", "CAS", "GIST", "TCTR_1", "TCTR_2", "TCTR_A", "TCTR_B"]


def run_all_models(models=None, datasets=None):
    model_names = list(models) if models is not None else list(DEFAULT_MODELS)
    dataset_names = list(datasets) if datasets is not None else list(DEFAULT_DATASETS)

    for dataset_name in dataset_names:
        for model_name in model_names:
            config = build_config(model_name, dataset_name)
            main(config)


if __name__ == "__main__":
    run_all_models()

