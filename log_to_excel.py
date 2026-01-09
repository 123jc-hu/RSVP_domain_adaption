import re

import pandas as pd

from pathlib import Path


# 输入日志字符串

log_str = r"""
2025-12-18 14:22:02,949 - INFO - Subject 1 | AUC: 0.9509 | BA: 0.9062 | F1: 0.8656 | TPR: 1.0000 | TNR: 0.8125
2025-12-18 14:22:17,447 - INFO - Subject 10 | AUC: 0.9308 | BA: 0.9241 | F1: 0.8890 | TPR: 1.0000 | TNR: 0.8482
2025-12-18 14:22:34,652 - INFO - Subject 11 | AUC: 0.7545 | BA: 0.6830 | F1: 0.8698 | TPR: 0.5000 | TNR: 0.8661
2025-12-18 14:22:47,393 - INFO - Subject 12 | AUC: 0.8203 | BA: 0.7232 | F1: 0.7750 | TPR: 0.7500 | TNR: 0.6964
2025-12-18 14:23:04,598 - INFO - Subject 13 | AUC: 0.9900 | BA: 0.9777 | F1: 0.9628 | TPR: 1.0000 | TNR: 0.9554
2025-12-18 14:23:16,231 - INFO - Subject 14 | AUC: 0.8996 | BA: 0.8527 | F1: 0.8702 | TPR: 0.8750 | TNR: 0.8304
2025-12-18 14:23:28,804 - INFO - Subject 15 | AUC: 0.8772 | BA: 0.8214 | F1: 0.8295 | TPR: 0.8750 | TNR: 0.7679
2025-12-18 14:23:43,112 - INFO - Subject 16 | AUC: 0.9933 | BA: 0.9420 | F1: 0.9126 | TPR: 1.0000 | TNR: 0.8839
2025-12-18 14:23:51,884 - INFO - Subject 17 | AUC: 0.9163 | BA: 0.8929 | F1: 0.8480 | TPR: 1.0000 | TNR: 0.7857
2025-12-18 14:24:03,775 - INFO - Subject 18 | AUC: 0.9721 | BA: 0.8705 | F1: 0.8933 | TPR: 0.8750 | TNR: 0.8661
2025-12-18 14:24:13,390 - INFO - Subject 19 | AUC: 0.9615 | BA: 0.8036 | F1: 0.8057 | TPR: 0.8750 | TNR: 0.7321
2025-12-18 14:24:15,813 - INFO - Subject 2 | AUC: 0.7656 | BA: 0.7902 | F1: 0.8629 | TPR: 0.7500 | TNR: 0.8304
2025-12-18 14:24:29,922 - INFO - Subject 20 | AUC: 0.9141 | BA: 0.8929 | F1: 0.9229 | TPR: 0.8750 | TNR: 0.9107
2025-12-18 14:24:46,472 - INFO - Subject 21 | AUC: 0.9766 | BA: 0.8393 | F1: 0.9269 | TPR: 0.7500 | TNR: 0.9286
2025-12-18 14:24:48,288 - INFO - Subject 22 | AUC: 0.4749 | BA: 0.5000 | F1: 0.9011 | TPR: 0.0000 | TNR: 1.0000
2025-12-18 14:25:05,351 - INFO - Subject 23 | AUC: 0.9587 | BA: 0.8348 | F1: 0.9208 | TPR: 0.7500 | TNR: 0.9196
2025-12-18 14:25:22,365 - INFO - Subject 24 | AUC: 0.8917 | BA: 0.8036 | F1: 0.8057 | TPR: 0.8750 | TNR: 0.7321
2025-12-18 14:25:35,463 - INFO - Subject 25 | AUC: 0.9766 | BA: 0.9509 | F1: 0.9246 | TPR: 1.0000 | TNR: 0.9018
2025-12-18 14:25:40,410 - INFO - Subject 26 | AUC: 0.7879 | BA: 0.7232 | F1: 0.7750 | TPR: 0.7500 | TNR: 0.6964
2025-12-18 14:25:56,092 - INFO - Subject 27 | AUC: 0.9219 | BA: 0.7991 | F1: 0.7997 | TPR: 0.8750 | TNR: 0.7232
2025-12-18 14:26:12,923 - INFO - Subject 28 | AUC: 0.9665 | BA: 0.8750 | F1: 0.8992 | TPR: 0.8750 | TNR: 0.8750
2025-12-18 14:26:27,102 - INFO - Subject 29 | AUC: 1.0000 | BA: 0.9643 | F1: 0.9432 | TPR: 1.0000 | TNR: 0.9286
2025-12-18 14:26:39,262 - INFO - Subject 3 | AUC: 0.8449 | BA: 0.7857 | F1: 0.7814 | TPR: 0.8750 | TNR: 0.6964
2025-12-18 14:26:48,779 - INFO - Subject 30 | AUC: 0.9475 | BA: 0.8929 | F1: 0.9229 | TPR: 0.8750 | TNR: 0.9107
2025-12-18 14:26:56,918 - INFO - Subject 31 | AUC: 0.5268 | BA: 0.5402 | F1: 0.7615 | TPR: 0.3750 | TNR: 0.7054
2025-12-18 14:27:07,193 - INFO - Subject 32 | AUC: 0.8783 | BA: 0.8036 | F1: 0.8057 | TPR: 0.8750 | TNR: 0.7321
2025-12-18 14:27:19,631 - INFO - Subject 33 | AUC: 0.8739 | BA: 0.8080 | F1: 0.8117 | TPR: 0.8750 | TNR: 0.7411
2025-12-18 14:27:27,637 - INFO - Subject 34 | AUC: 0.7634 | BA: 0.7232 | F1: 0.6898 | TPR: 0.8750 | TNR: 0.5714
2025-12-18 14:27:41,853 - INFO - Subject 35 | AUC: 0.9989 | BA: 0.9598 | F1: 0.9369 | TPR: 1.0000 | TNR: 0.9196
2025-12-18 14:27:54,729 - INFO - Subject 36 | AUC: 0.8103 | BA: 0.8214 | F1: 0.8295 | TPR: 0.8750 | TNR: 0.7679
2025-12-18 14:28:11,755 - INFO - Subject 37 | AUC: 0.9397 | BA: 0.9152 | F1: 0.8773 | TPR: 1.0000 | TNR: 0.8304
2025-12-18 14:28:23,548 - INFO - Subject 38 | AUC: 0.9040 | BA: 0.7902 | F1: 0.7024 | TPR: 1.0000 | TNR: 0.5804
2025-12-18 14:28:30,577 - INFO - Subject 39 | AUC: 0.6306 | BA: 0.6295 | F1: 0.6338 | TPR: 0.7500 | TNR: 0.5089
2025-12-18 14:28:43,090 - INFO - Subject 4 | AUC: 0.9174 | BA: 0.8705 | F1: 0.8182 | TPR: 1.0000 | TNR: 0.7411
2025-12-18 14:28:52,728 - INFO - Subject 40 | AUC: 0.7243 | BA: 0.6384 | F1: 0.7376 | TPR: 0.6250 | TNR: 0.6518
2025-12-18 14:29:07,015 - INFO - Subject 41 | AUC: 0.9063 | BA: 0.8571 | F1: 0.8759 | TPR: 0.8750 | TNR: 0.8393
2025-12-18 14:29:16,182 - INFO - Subject 42 | AUC: 0.8147 | BA: 0.7589 | F1: 0.8227 | TPR: 0.7500 | TNR: 0.7679
2025-12-18 14:29:25,689 - INFO - Subject 43 | AUC: 0.9185 | BA: 0.7812 | F1: 0.8515 | TPR: 0.7500 | TNR: 0.8125
2025-12-18 14:29:39,308 - INFO - Subject 44 | AUC: 0.8013 | BA: 0.7991 | F1: 0.7997 | TPR: 0.8750 | TNR: 0.7232
2025-12-18 14:29:44,807 - INFO - Subject 45 | AUC: 0.6032 | BA: 0.5491 | F1: 0.6928 | TPR: 0.5000 | TNR: 0.5982
2025-12-18 14:30:01,987 - INFO - Subject 46 | AUC: 0.9609 | BA: 0.9464 | F1: 0.9186 | TPR: 1.0000 | TNR: 0.8929
2025-12-18 14:30:18,277 - INFO - Subject 47 | AUC: 0.9542 | BA: 0.9107 | F1: 0.8715 | TPR: 1.0000 | TNR: 0.8214
2025-12-18 14:30:30,163 - INFO - Subject 48 | AUC: 0.9442 | BA: 0.8170 | F1: 0.8972 | TPR: 0.7500 | TNR: 0.8839
2025-12-18 14:30:45,952 - INFO - Subject 49 | AUC: 0.7600 | BA: 0.6920 | F1: 0.8100 | TPR: 0.6250 | TNR: 0.7589
2025-12-18 14:30:59,343 - INFO - Subject 5 | AUC: 0.9922 | BA: 0.9420 | F1: 0.9126 | TPR: 1.0000 | TNR: 0.8839
2025-12-18 14:31:13,613 - INFO - Subject 50 | AUC: 0.8348 | BA: 0.7634 | F1: 0.8285 | TPR: 0.7500 | TNR: 0.7768
2025-12-18 14:31:23,108 - INFO - Subject 51 | AUC: 0.6272 | BA: 0.5893 | F1: 0.6644 | TPR: 0.6250 | TNR: 0.5536
2025-12-18 14:31:35,744 - INFO - Subject 52 | AUC: 0.8471 | BA: 0.8214 | F1: 0.8295 | TPR: 0.8750 | TNR: 0.7679
2025-12-18 14:31:52,843 - INFO - Subject 53 | AUC: 0.9743 | BA: 0.9196 | F1: 0.8832 | TPR: 1.0000 | TNR: 0.8393
2025-12-18 14:32:06,890 - INFO - Subject 54 | AUC: 0.9877 | BA: 0.9509 | F1: 0.9246 | TPR: 1.0000 | TNR: 0.9018
2025-12-18 14:32:21,162 - INFO - Subject 55 | AUC: 0.8996 | BA: 0.8080 | F1: 0.8857 | TPR: 0.7500 | TNR: 0.8661
2025-12-18 14:32:34,031 - INFO - Subject 6 | AUC: 0.9844 | BA: 0.9375 | F1: 0.9067 | TPR: 1.0000 | TNR: 0.8750
2025-12-18 14:32:49,014 - INFO - Subject 7 | AUC: 0.9219 | BA: 0.7768 | F1: 0.8458 | TPR: 0.7500 | TNR: 0.8036
2025-12-18 14:33:01,061 - INFO - Subject 8 | AUC: 0.8270 | BA: 0.7277 | F1: 0.8554 | TPR: 0.6250 | TNR: 0.8304
2025-12-18 14:33:18,295 - INFO - Subject 9 | AUC: 0.9353 | BA: 0.8259 | F1: 0.8353 | TPR: 0.8750 | TNR: 0.7768
2025-12-18 14:33:18,368 - INFO - Average metrics: {'AUC': '0.8719+/-0.1213', 'BA': '0.8131+/-0.1133', 'F1': '0.8441+/-0.0744', 'TPR': '0.8295+/-0.1869', 'TNR': '0.7968+/-0.1075'}
2025-12-18 14:33:18,368 - INFO - Results saved to Experiments\SparseEA_new\GIST\single-subject\results.csv
2025-12-18 14:33:18,372 - INFO - Finished training and testing!
2025-12-18 14:33:18,372 - INFO - Selected K: 24
2025-12-18 14:33:18,372 - INFO - Elapsed time: 11.37 minutes 
"""


# 1. 正则匹配每个 Subject 的指标（忽略日志里的 Subject 编号）

pattern = re.compile(

    r"Subject\s+\d+\s*\|\s*AUC:\s*([\d.]+)\s*\|\s*BA:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)\s*\|\s*TPR:\s*([\d.]+)\s*\|\s*TNR:\s*([\d.]+)"

)


matches = pattern.findall(log_str)


# 2. 解析成字典列表，并生成连续的 SUB 编号

data = []

for i, (auc, ba, f1, tpr, tnr) in enumerate(matches, start=1):

    data.append({

        "SUB": f"SUB{i}",

        "AUC": float(auc),

        "BA": float(ba),

        "F1": float(f1),

        "TPR": float(tpr),

        "TNR": float(tnr)

    })


# 3. 生成 DataFrame

df = pd.DataFrame(data)


# 4. 计算平均值和标准差

metrics = ["AUC", "BA", "F1", "TPR", "TNR"]

avg = df[metrics].mean().round(4)

std = df[metrics].std().round(4)


# 5. 追加 AVG 和 STD 行

df = pd.concat([

    df,

    pd.Series({"SUB": "AVG", **avg}).to_frame().T,

    pd.Series({"SUB": "STD", **std}).to_frame().T

], ignore_index=True)


# 6. 获取保存路径（倒数第三行）

save_path_match = re.search(r"Results saved to\s+(.+?\.(?:csv|xlsx))", log_str)

if not save_path_match:

    raise ValueError("未找到保存路径")

save_path = Path(save_path_match.group(1))


# 确保目录存在

save_path.parent.mkdir(parents=True, exist_ok=True)


# 7. 根据后缀保存 Excel / CSV

if save_path.suffix.lower() == ".xlsx":

    df.to_excel(save_path, index=False, float_format="%.4f")

else:

    df.to_csv(save_path, index=False, float_format="%.4f")


# 8. 打印均值 ± 方差

formatted = {k: f"{avg[k]:.4f}+/-{std[k]:.4f}" for k in metrics}

print(f"Average metrics: {formatted}")

print(f"Results saved to {save_path}")
