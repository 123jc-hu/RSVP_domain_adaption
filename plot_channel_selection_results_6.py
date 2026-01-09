import matplotlib.pyplot as plt
import numpy as np

# ============================
# 1. 全局设置 (Arial Font & Style)
# ============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 12

# 定义不同数据集的 X 轴 (K值)
ks_standard = [8, 16, 24, 32, 48]
ks_gist = [4, 8, 12, 16, 24]

methods_config = {
    'CSS': {'color': '#1f77b4', 'fmt': '--s', 'label': 'CSS'},
    'SBW': {'color': '#ff7f0e', 'fmt': '--^', 'label': 'SBW'},
    'SparseEA': {'color': '#2ca02c', 'fmt': '--v', 'label': 'SparseEA'},
    'ABMOHS': {'color': '#9467bd', 'fmt': '--x', 'label': 'ABMOHS'},
    'GSS': {'color': '#7f7f7f', 'fmt': ':D', 'label': 'GSS'},
    'ZOS-Selector': {'color': '#d62728', 'fmt': '-o', 'label': 'ZOS (Ours)'}
}


# ============================
# 2. 数据录入 (根据你的表格提取)
# ============================
# 结构: 'mean': [...], 'sig': [2, 1, 0...] (2=**, 1=*, 0=无)

def get_all_ba_data():
    data = {}

    # --- TCTR_1 (New!) ---
    data['TCTR_1'] = {
        'ks': ks_standard,
        'methods': {
            'CSS': {'mean': [0.840, 0.871, 0.900, 0.915, 0.921], 'sig': [2, 2, 2, 2, 0]},
            'SBW': {'mean': [0.867, 0.895, 0.909, 0.920, 0.930], 'sig': [1, 1, 1, 0, 0]},
            'SparseEA': {'mean': [0.886, 0.897, 0.908, 0.923, 0.922], 'sig': [0, 1, 1, 0, 1]},
            'ABMOHS': {'mean': [0.855, 0.868, 0.866, 0.879, 0.867], 'sig': [2, 2, 2, 2, 2]},
            'GSS': {'mean': [0.796, 0.830, 0.816, 0.835, 0.836], 'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.886, 0.920, 0.932, 0.933, 0.939], 'sig': [0, 0, 0, 0, 0]}
        }
    }

    # --- TCTR_2 (New!) ---
    data['TCTR_2'] = {
        'ks': ks_standard,
        'methods': {
            'CSS': {'mean': [0.837, 0.854, 0.899, 0.909, 0.926], 'sig': [1, 2, 2, 1, 1]},
            'SBW': {'mean': [0.859, 0.897, 0.918, 0.942, 0.931], 'sig': [0, 2, 1, 0, 0]},
            'SparseEA': {'mean': [0.865, 0.900, 0.899, 0.931, 0.931], 'sig': [0, 1, 1, 0, 0]},
            'ABMOHS': {'mean': [0.849, 0.867, 0.867, 0.865, 0.879], 'sig': [0, 2, 2, 2, 2]},
            'GSS': {'mean': [0.809, 0.815, 0.809, 0.819, 0.826], 'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.862, 0.930, 0.923, 0.932, 0.940], 'sig': [0, 0, 0, 0, 0]}
        }
    }

    # --- TCTR_A (New!) ---
    data['TCTR_A'] = {
        'ks': ks_standard,
        'methods': {
            'CSS': {'mean': [0.799, 0.828, 0.846, 0.864, 0.886], 'sig': [2, 2, 0, 1, 0]},
            'SBW': {'mean': [0.804, 0.837, 0.853, 0.873, 0.879], 'sig': [2, 2, 0, 0, 0]},
            'SparseEA': {'mean': [0.817, 0.841, 0.862, 0.876, 0.879], 'sig': [1, 1, 0, 0, 0]},
            'ABMOHS': {'mean': [0.788, 0.821, 0.800, 0.822, 0.828], 'sig': [2, 2, 2, 2, 2]},
            'GSS': {'mean': [0.771, 0.776, 0.768, 0.775, 0.773], 'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.832, 0.860, 0.860, 0.883, 0.880], 'sig': [0, 0, 0, 0, 0]}
        }
    }

    # --- TCTR_B (Previous) ---
    data['TCTR_B'] = {
        'ks': ks_standard,
        'methods': {
            'CSS': {'mean': [0.771, 0.806, 0.829, 0.837, 0.852], 'sig': [2, 0, 0, 0, 0]},
            'SBW': {'mean': [0.764, 0.799, 0.821, 0.846, 0.856], 'sig': [1, 1, 0, 0, 0]},
            'SparseEA': {'mean': [0.773, 0.806, 0.818, 0.832, 0.832], 'sig': [1, 0, 0, 0, 0]},
            'ABMOHS': {'mean': [0.744, 0.764, 0.774, 0.774, 0.777], 'sig': [2, 2, 0, 2, 2]},
            'GSS': {'mean': [0.768, 0.749, 0.745, 0.753, 0.751], 'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.792, 0.825, 0.833, 0.845, 0.844], 'sig': [0, 0, 0, 0, 0]}
        }
    }

    # --- CAS (Previous) ---
    data['CAS'] = {
        'ks': ks_standard,
        'methods': {
            'CSS': {'mean': [0.768, 0.779, 0.785, 0.796, 0.803], 'sig': [1, 0, 0, 0, 0]},
            'SBW': {'mean': [0.743, 0.777, 0.784, 0.775, 0.795], 'sig': [2, 0, 0, 0, 0]},
            'SparseEA': {'mean': [0.793, 0.795, 0.795, 0.791, 0.815], 'sig': [0, 0, 0, 0, 0]},
            'ABMOHS': {'mean': [0.748, 0.778, 0.776, 0.777, 0.777], 'sig': [1, 0, 0, 0, 1]},
            'GSS': {'mean': [0.729, 0.758, 0.764, 0.746, 0.759], 'sig': [2, 0, 0, 1, 1]},
            'ZOS-Selector': {'mean': [0.804, 0.782, 0.800, 0.803, 0.803], 'sig': [0, 0, 0, 0, 0]}
        }
    }

    # --- GIST (Previous, Special K) ---
    data['GIST'] = {
        'ks': ks_gist,
        'methods': {
            'CSS': {'mean': [0.714, 0.768, 0.801, 0.843, 0.849], 'sig': [0, 0, 0, 0, 0]},
            'SBW': {'mean': [0.703, 0.785, 0.794, 0.834, 0.850], 'sig': [0, 0, 0, 1, 0]},
            'SparseEA': {'mean': [0.700, 0.741, 0.751, 0.802, 0.813], 'sig': [0, 0, 2, 2, 1]},
            'ABMOHS': {'mean': [0.661, 0.702, 0.743, 0.734, 0.755], 'sig': [2, 2, 2, 2, 2]},
            'GSS': {'mean': [0.578, 0.520, 0.594, 0.570, 0.517], 'sig': [2, 2, 2, 2, 2]},
            'ZOS-Selector': {'mean': [0.722, 0.762, 0.814, 0.843, 0.849], 'sig': [0, 0, 0, 0, 0]}
        }
    }

    return data


# ============================
# 3. 绘图函数 (2行3列 - 自动汇总)
# ============================
def plot_ba_summary(all_data):
    # 手动指定绘图顺序，符合逻辑
    dataset_order = ['TCTR_1', 'TCTR_2', 'TCTR_A', 'TCTR_B', 'CAS', 'GIST']

    fig, axes = plt.subplots(2, 3, figsize=(16, 6.5))  # 16x8.5 是比较标准的尺寸
    axes = axes.flatten()

    for idx, ds_name in enumerate(dataset_order):
        ax = axes[idx]

        # 获取该数据集数据
        ds_info = all_data.get(ds_name)
        if not ds_info: continue

        current_ks = ds_info['ks']
        methods_data = ds_info['methods']

        for method_name, style in methods_config.items():
            if method_name not in methods_data: continue

            m_data = methods_data[method_name]
            means = np.array(m_data['mean'])
            sigs = m_data['sig']

            # ZOS 加粗，突出显示
            lw = 2.5 if method_name == 'ZOS-Selector' else 1.5
            zorder = 10 if method_name == 'ZOS-Selector' else 1

            # 绘制线条 (无阴影，保持清爽)
            ax.plot(current_ks, means, style['fmt'], color=style['color'],
                    linewidth=lw, markersize=6, alpha=0.9,
                    label=style['label'] if idx == 0 else "",  # 只在第一个子图生成标签
                    zorder=zorder)

            # 绘制显著性标记 (*)
            for i, s_val in enumerate(sigs):
                if s_val > 0:
                    marker_txt = '*' if s_val == 1 else '**'
                    # 计算偏移量 (y轴范围的 2% 左右)
                    # 简单处理：向上偏移 0.005
                    offset = 0.005
                    ax.text(current_ks[i], means[i] + offset, marker_txt,
                            ha='center', va='bottom', fontsize=12,
                            color=style['color'], fontweight='bold', zorder=zorder)

        # 设置标题 (去下划线)
        display_name = ds_name.replace('_', ' ')  # 如 TCTR_1 -> TCTR 1
        ax.set_title(display_name, fontsize=14, weight='bold')
        ax.set_xticks(current_ks)
        ax.grid(True, linestyle='--', alpha=0.4)

        # 坐标轴标签控制
        # 只有第二行 (idx 3,4,5) 显示 X Label
        if idx >= 3:
            ax.set_xlabel('Selected Channels ($K$)', fontsize=12)
        # 只有第一列 (idx 0,3) 显示 Y Label
        if idx % 3 == 0:
            ax.set_ylabel('Balanced Accuracy', fontsize=12)

    # 4. 生成全局图例
    # 提取第一个子图的 handles/labels
    handles, labels = axes[0].get_legend_handles_labels()

    # 重新排序：把 ZOS 放到第一个
    zos_label = methods_config['ZOS-Selector']['label']
    if zos_label in labels:
        zos_idx = labels.index(zos_label)
        # 将 ZOS 移到列表最前
        order = [zos_idx] + [i for i in range(len(labels)) if i != zos_idx]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

    # 将图例放在底部中央
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.02),
               ncol=6, frameon=False, fontsize=13)

    # 5. 布局调整
    # bottom=0.15 留出底部给图例, wspace/hspace 调整子图间距
    plt.subplots_adjust(bottom=0.15, left=0.06, right=0.98, top=0.92, hspace=0.25, wspace=0.2)

    plt.savefig("Fig_Select_Others_Summary.pdf", dpi=300)
    plt.show()


# ============================
# 5. 运行
# ============================
all_ba_data = get_all_ba_data()
plot_ba_summary(all_ba_data)