from Utils.config import build_config
from main import main


def run_all_models():
    model_list = [
        # "DeepConvNet",
        # "EEGNet", 
        # "EEGInception", 
        # "PLNet", 
        # "PPNN",
        # "lightweight_erp", 
        # "LMDANet", 
        # "LENet", "1DCNN_BiLSTM",
        # "Ours_light_without_K",
        # "Ours_light",
        # "Ours_light_reg",
        # "adaptive_ours_light",
        # "Ours_new",
        "Ours_ZOS",
        # "Correlation_new",
        # "Frequency_new",
        # "SparseEA_new",
        # "ABMOHS",
        # "TSCAE",
        # "MOCNN",
        # "GSS", 
    ]

    dataset_list = [
        # "THU", 
        "TCTR_1", 
        # "TCTR_2", "TCTR_A", "TCTR_B", 
        # "CAS",
        # "GIST"
    ]
    # for k_num in [1] + list(range(2, 63, 2)):  # 2,4,...,62
    for k_num in [16, 24, 32, 48]:  
    # for k_num in [4, 8, 12, 16, 24]:  
        # if k_num < 8:
        #     continue
        for dataset_name in dataset_list:
            # print(f"\n================ Running Dataset: {dataset_name} ================\n")
            for model_name in model_list:
                # print(f"\n================ Running model: {model_name} ================\n")
                config = build_config(model_name, dataset_name)
                config["selector_K"] = k_num  # 动态修改 K
                # config["n_channels"] = k_num
                main(config)


if __name__ == "__main__":
    run_all_models()