import matplotlib.pyplot as plt
import numpy as np

# ============================
# 1. 全局设置
# ============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 12

ks = [8, 16, 24, 32, 48]

methods_config = {
    'CSS': {'color': '#1f77b4', 'fmt': '--s', 'label': 'CSS'},
    'SBW': {'color': '#ff7f0e', 'fmt': '--^', 'label': 'SBW'},
    'SparseEA': {'color': '#2ca02c', 'fmt': '--v', 'label': 'SparseEA'},
    'ABMOHS': {'color': '#9467bd', 'fmt': '--x', 'label': 'ABMOHS'},
    'GSS': {'color': '#7f7f7f', 'fmt': ':D', 'label': 'GSS'},
    'ZOS-Selector': {'color': '#d62728', 'fmt': '-o', 'label': 'ZOS (Ours)'}
}


# ============================
# 2. 数据模拟 (请在此处填入真实数据)
# ============================
# 为了演示，我用随机数生成了数据结构。请用你的真实数据替换这里！

def get_thu_data():
    # 这里填入你之前给我的 THU 真实数据
    return {
        'AUC': {
            'CSS': {'mean': [0.876, 0.900, 0.913, 0.922, 0.927], 'std': [0.082, 0.072, 0.067, 0.064, 0.061],
                    'sig': [2, 2, 2, 2, 0]},
            'SBW': {'mean': [0.864, 0.909, 0.919, 0.927, 0.929], 'std': [0.087, 0.060, 0.056, 0.058, 0.058],
                    'sig': [2, 2, 2, 0, 0]},
            'SparseEA': {'mean': [0.881, 0.902, 0.912, 0.922, 0.922], 'std': [0.082, 0.071, 0.069, 0.057, 0.060],
                         'sig': [2, 2, 2, 1, 2]},
            'ABMOHS': {'mean': [0.865, 0.889, 0.889, 0.897, 0.897], 'std': [0.087, 0.075, 0.080, 0.078, 0.073],
                       'sig': [2, 2, 2, 2, 2]},
            'GSS': {'mean': [0.826, 0.853, 0.850, 0.838, 0.857], 'std': [0.095, 0.083, 0.080, 0.094, 0.085],
                    'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.894, 0.918, 0.924, 0.929, 0.930], 'std': [0.072, 0.054, 0.055, 0.051, 0.057],
                             'sig': [0, 0, 0, 0, 0]},
        },
        'Balanced Accuracy': {  # 真实数据
            'CSS': {'mean': [0.809, 0.836, 0.851, 0.859, 0.864], 'std': [0.082, 0.079, 0.073, 0.072, 0.073],
                    'sig': [2, 2, 2, 2, 0]},
            'SBW': {'mean': [0.795, 0.840, 0.854, 0.864, 0.869], 'std': [0.085, 0.071, 0.069, 0.068, 0.070],
                    'sig': [2, 2, 2, 0, 0]},
            'SparseEA': {'mean': [0.813, 0.835, 0.847, 0.858, 0.859], 'std': [0.083, 0.077, 0.078, 0.067, 0.073],
                         'sig': [2, 2, 2, 1, 1]},
            'ABMOHS': {'mean': [0.798, 0.821, 0.824, 0.830, 0.831], 'std': [0.086, 0.080, 0.087, 0.086, 0.079],
                       'sig': [2, 2, 2, 2, 2]},
            'GSS': {'mean': [0.736, 0.763, 0.737, 0.740, 0.718], 'std': [0.087, 0.072, 0.083, 0.072, 0.084],
                    'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.825, 0.852, 0.862, 0.868, 0.868], 'std': [0.078, 0.066, 0.066, 0.063, 0.069],
                             'sig': [0, 0, 0, 0, 0]},
        },
        'F1-score': {  # 真实数据
            'CSS': {'mean': [0.881, 0.906, 0.916, 0.924, 0.929], 'std': [0.068, 0.051, 0.040, 0.037, 0.036],
                    'sig': [2, 2, 2, 0, 0]},
            'SBW': {'mean': [0.879, 0.912, 0.919, 0.927, 0.930], 'std': [0.057, 0.038, 0.037, 0.034, 0.037],
                    'sig': [2, 2, 2, 0, 0]},
            'SparseEA': {'mean': [0.890, 0.907, 0.915, 0.922, 0.923], 'std': [0.046, 0.047, 0.043, 0.036, 0.042],
                         'sig': [2, 2, 2, 0, 2]},
            'ABMOHS': {'mean': [0.882, 0.899, 0.906, 0.913, 0.907], 'std': [0.052, 0.049, 0.044, 0.041, 0.050],
                       'sig': [2, 2, 2, 2, 2]},
            'GSS': {'mean': [0.829, 0.834, 0.716, 0.784, 0.663], 'std': [0.061, 0.046, 0.149, 0.061, 0.168],
                    'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.899, 0.917, 0.928, 0.927, 0.931], 'std': [0.043, 0.039, 0.034, 0.034, 0.036],
                             'sig': [0, 0, 0, 0, 0]},
        }
    }

# ============================
# 3. 绘图A: THU 详细对比 (1行3列)
# ============================
def plot_thu_detailed(thu_data):
    metrics = ['AUC', 'Balanced Accuracy', 'F1-score']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))  # 稍微压扁一点

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = thu_data[metric]

        for method_name, style in methods_config.items():
            if method_name not in metric_data: continue

            m_data = metric_data[method_name]
            means = np.array(m_data['mean'])
            stds = np.array(m_data['std'])
            sigs = m_data['sig']

            lw = 2.5 if method_name == 'ZOS-Selector' else 1.5
            alpha_line = 1.0 if method_name == 'ZOS-Selector' else 0.9
            zorder = 10 if method_name == 'ZOS-Selector' else 1

            # 画线
            ax.plot(ks, means, style['fmt'], color=style['color'],
                    linewidth=lw, markersize=6, alpha=alpha_line,
                    label=style['label'], zorder=zorder)

            # # [修改点] 画阴影：还原真实 Std，但透明度极低，且只给 ZOS 画
            # if method_name == 'ZOS-Selector':
            #     ax.fill_between(ks, means - stds, means + stds,
            #                     color=style['color'], alpha=0.1, linewidth=0, zorder=zorder - 1)

            # [修改点] 画显著性：在点上方标注，颜色与线一致
            for i, s_val in enumerate(sigs):
                if s_val > 0:
                    marker_txt = '*' if s_val == 1 else '**'
                    # 偏移量需要根据数据范围微调，这里设为 Range 的 2% 左右
                    y_range = max(means) - min(means)
                    offset = y_range * 0.05 if y_range > 0 else 0.01
                    ax.text(ks[i], means[i] + offset, marker_txt,
                            ha='center', va='bottom', fontsize=12,
                            color=style['color'], fontweight='bold', zorder=zorder)

        ax.set_title(metric, fontsize=14, weight='bold', pad=10)
        ax.set_xlabel('Selected Channels ($K$)', fontsize=12)
        ax.set_xticks(ks)
        ax.grid(True, linestyle='--', alpha=0.4)

        # 只有第一列显示 Y 轴标签
        if idx == 0:
            ax.set_ylabel('Performance Score', fontsize=12)

    # [修改点] 图例位置优化
    handles, labels = axes[0].get_legend_handles_labels()
    # 排序：ZOS 第一
    zos_label = methods_config['ZOS-Selector']['label']
    if zos_label in labels:
        zos_idx = labels.index(zos_label)
        order = [zos_idx] + [i for i in range(len(labels)) if i != zos_idx]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.02),  # [修改点] 贴近 X 轴
               ncol=6, frameon=False, fontsize=12)
    # 留出底部空间给图例
    plt.subplots_adjust(bottom=0.22, left=0.06, right=0.98, wspace=0.25)
    plt.savefig("Fig_Select_THU_Detail.pdf", dpi=300)
    plt.show()




# ============================
# 5. 执行
# ============================
# 画图 1: THU 详细
thu_data = get_thu_data()
plot_thu_detailed(thu_data)
