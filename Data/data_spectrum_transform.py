from pathlib import Path
import numpy as np
from scipy.signal import cheby2, sosfiltfilt
import os


def bandpass_filterbank_numpy(x_np, sfreq=128):
    """
    更快版本：使用向量化 sosfiltfilt，无需线程池
    """
    assert x_np.ndim == 3
    bands = [
        (0.3, 4), (4, 8), (8, 12), (12, 16), (16, 20),
        (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)
    ]
    batch, C, T = x_np.shape
    filtered = []

    for low, high in bands:
        nyq = 0.5 * sfreq
        sos = cheby2(N=4, rs=20, Wn=[low/nyq, high/nyq], btype='bandpass', output='sos')

        band_filtered = np.zeros((batch, C, T), dtype=x_np.dtype)
        for c in range(C):
            band_filtered[:, c, :] = sosfiltfilt(sos, x_np[:, c, :], axis=-1)
        filtered.append(band_filtered)

    return np.stack(filtered, axis=1)  # (batch, 10, C, T)


if __name__ == '__main__':
    dataset_name = 'TCTR'
    num_sub = 15
    base_dir = Path.cwd().parent
    data_dir = base_dir / "Dataset" / dataset_name / "Standard_128Hz"
    task_list = ['task2', 'taskA', 'taskB'] if dataset_name == 'TCTR' else [None]
    for sub_folder in task_list:
        if sub_folder:
            data_dir_specific = data_dir / sub_folder
        else:
            data_dir_specific = data_dir
        all_files = [f for f in os.listdir(data_dir_specific) if f.endswith(".npz")]
        npz_files = [f for f in all_files if "_10band" not in f]
        for file_name in npz_files:
            subject_file = data_dir_specific / file_name
            sub_name = subject_file.stem
            output_file = data_dir_specific / f"{sub_name}_10band.npz"

            print(f"Processing {subject_file.name}...")

            with np.load(subject_file, mmap_mode='r') as data:
                x_origin = data['x_data']  # (batch, C, T)
                y = data['y_data']

            # Apply bandpass filtering
            x_filtered = bandpass_filterbank_numpy(x_origin, sfreq=128)

            # Save result
            np.savez_compressed(output_file, x_data=x_filtered, y_data=y)

            print(f"Saved filtered data to {output_file.name}")
