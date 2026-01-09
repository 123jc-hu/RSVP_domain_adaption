import numpy as np
from pathlib import Path
import os


def convert_npz_to_npy(npz_path):
    """将npz文件转换为npy格式"""
    with np.load(npz_path) as data:
        subject_name = Path(npz_path).stem
        np.save(npz_path.parent / f"{subject_name}_x.npy", data['x_data'])
        np.save(npz_path.parent / f"{subject_name}_y.npy", data['y_data'])


def delete_npy_files(npy_paths):
    """删除指定的npy文件"""
    for npy_path in npy_paths:
        try:
            os.remove(npy_path)
            print(f"已删除文件: {npy_path}")
        except FileNotFoundError:
            print(f"文件未找到: {npy_path}")
        except Exception as e:
            print(f"删除文件时发生错误: {e}")


if __name__ == '__main__':
    # 遍历所有npz文件执行转换
    data_dir = Path("../Dataset/THU/Standard_128Hz")
    # for npz_file in data_dir.glob("*.npz"):
    #     convert_npz_to_npy(npz_file)
    for npz_file in data_dir.glob("*.npy"):
        delete_npy_files([npz_file])
