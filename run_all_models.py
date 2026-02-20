from Utils.config import build_config
from main import main


def run_all_models():
    model_list = ["DeepConvNet", "EEGNet", "EEGInception", "PLNet", "PPNN"]
    dataset_list = ["THU", "CAS", "GIST", "TCTR_1", "TCTR_2", "TCTR_A", "TCTR_B"]

    for dataset_name in dataset_list:
        for model_name in model_list:
            config = build_config(model_name, dataset_name)
            main(config)


if __name__ == "__main__":
    run_all_models()
