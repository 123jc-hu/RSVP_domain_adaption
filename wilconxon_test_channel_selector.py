from scipy.stats import wilcoxon
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
from wilconxon_test import load_scores

if __name__ == '__main__':
    # 初始化数据集信息
    dataset_dict = {
        # 'THU': 64,
        # 'TCTR_1': 15,
        # 'TCTR_2': 15,
        # 'TCTR_A': 15,
        # 'TCTR_B': 15,
        # 'CAS': 14,
        'GIST': 55,
    }
    dataset_list = list(dataset_dict.keys())

    target_model = 'Ours_ZOS'  # 作为对比基准的模型名称
    model_list = [
        "Correlation_new",
        "Frequency_new",
        "SparseEA_new",
        "ABMOHS",
        "GSS",
    ]
    train_mode = [
        'single-subject',
        # 'cross-subject'
    ]
    K = 24

    for mode in train_mode:
        result_dir = Path('Results')
        result_dir.mkdir(exist_ok=True)

        # 创建txt文件路径（包含mode信息）
        result_txt_path = result_dir / f'wilcoxon_{mode}_K={K}_result.txt'

        # 清空旧结果
        with open(result_txt_path, 'w') as f:
            pass

        # 遍历每个数据集
        for dataset in dataset_list:
            # 为每个数据集创建独立的Excel文件路径
            result_excel_path = result_dir / f'wilcoxon_{mode}_{dataset}_K={K}_result.xlsx'

            result_data = []  # 为每个数据集初始化结果列表

            with tqdm(total=len(model_list), desc=f'Dataset: {dataset}', ncols=100) as pbar:
                for model in model_list:
                    target_path = Path(f'Experiments/{target_model}/{dataset}/{mode}/Results.csv')
                    model_path = Path(f'Experiments/{model}/{dataset}/{mode}/Results.csv')

                    auc1, ba1, f11 = load_scores(target_path)
                    auc2, ba2, f12 = load_scores(model_path)

                    if not (auc1 and auc2):
                        pbar.write(f"Skipping {model} on {dataset} due to missing data.")
                        pbar.update(1)
                        continue

                    try:
                        auc_stat, auc_p = wilcoxon(auc1, auc2, alternative='two-sided')
                        ba_stat, ba_p = wilcoxon(ba1, ba2, alternative='two-sided')
                        f1_stat, f1_p = wilcoxon(f11, f12, alternative='two-sided')

                        log_line = (
                            f'{target_model} vs {model} on {dataset}: '
                            f'AUC stat={auc_stat:.4f}, p={auc_p:.4e} | '
                            f'BA stat={ba_stat:.4f}, p={ba_p:.4e} | '
                            f'F1 stat={f1_stat:.4f}, p={f1_p:.4e}'
                        )
                        print(log_line)

                        with open(result_txt_path, 'a') as f:
                            f.write(log_line + '\n')

                        result_data.append([
                            dataset, target_model, model,
                            auc_stat, auc_p,
                            ba_stat, ba_p,
                            f1_stat, f1_p
                        ])

                    except Exception as e:
                        print(f"Wilcoxon test failed for {model} on {dataset}: {e}")

                    pbar.update(1)

            # 每个数据集结束后保存对应的Excel文件
            if result_data:  # 只有在有数据的情况下才保存
                df = pd.DataFrame(result_data, columns=[
                    'Dataset', 'Target Model', 'Compared Model',
                    'AUC Stat', 'AUC p-value',
                    'BA Stat', 'BA p-value',
                    'F1 Stat', 'F1 p-value'
                ])
                df.to_excel(result_excel_path, index=False)
                print(f"\n[✔] {dataset} Results saved to: {result_excel_path}")