# === Pareto scatter (Params & FLOPs) — final TNNLS-ready ===
import math
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "grid.alpha": 0.45,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
})

# name, params, flops, BA_single, BA_cross
models = [
    ("EEGNet",          2224,   8.48, 0.8178, 0.8608),
    ("PLNet",           2102,   0.83, 0.8613, 0.8476),
    ("PPNN",           12930,  14.61, 0.8424, 0.8592),
    ("LMDA-Net",        2752,  16.42, 0.8593, 0.8657),
    ("1DCNN-BiLSTM",   19354,   3.40, 0.7913, 0.8506),
    ("LENet",           4586,  17.42, 0.7347, 0.7929),
    ("Lightweight-ERP", 2507,   1.78, 0.8224, 0.8569),
    ("DDLM-Extractor",  4333,   0.42, 0.8634, 0.8653),
]

def pareto_front(data, x_func, y_idx):
    data_sorted = sorted(data, key=lambda m: x_func(m))
    front, best = [], -1.0
    for m in data_sorted:
        y = m[y_idx]
        if y > best:
            front.append(m); best = y
    return front

def plot_one(ax, x_mode, y_idx, y_label_suffix):
    blue = "#1d3557"
    red  = "#D62728"
    grey = "#8a8a8a"

    if x_mode == "log_params":
        xfun = lambda m: math.log10(m[1])
        xlabel = r'$\log_{10}(\mathrm{Params})$'
    else:
        xfun = lambda m: m[2]
        xlabel = "FLOPs (Millions)"

    # Pareto
    front = pareto_front(models, xfun, y_idx)

    texts = []
    for name, p, fl, ba_s, ba_c in models:
        y = ba_s if y_idx == 3 else ba_c
        x = xfun((name, p, fl, ba_s, ba_c))
        if name == "DDLM-Extractor":
            ax.scatter(x, y, marker='*', s=110, color=red,
                       edgecolor='k', linewidths=0.7, zorder=3)
        else:
            ax.scatter(x, y, marker='o', s=38, edgecolor='k', linewidths=0.5, color=blue, zorder=2)
        # 文本先统一加上，后面再 adjust
        texts.append(ax.text(x, y, name, fontsize=8, ha='left', va='bottom'))

    # Pareto 虚线
    px = [xfun(m) for m in front]
    py = [m[y_idx] for m in front]
    ax.plot(px, py, '--', lw=0.9, color=grey, zorder=1)

    # 自动避让（限制只在图内移动）
    adjust_text(
        texts, ax=ax, arrowprops=dict(arrowstyle='-', lw=0.3, color='gray'),
        only_move={'points':'xy', 'text':'xy'},  # 允许双向微调
        lim=200
    )

    # 轴与网格
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Balanced Accuracy (BA, {y_label_suffix})')
    ax.set_ylim(0.72, 0.89)      # 你当前的范围
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.margins(x=0.03, y=0.02)

# 画两幅图
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
plot_one(axes[0], "log_params", 3, "Within-Subject")
plot_one(axes[1], "flops",      4, "Cross-Subject")

# 轻量图例（手工构造，不会在轴上多画点）
legend_handles = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor="#D62728",
           markeredgecolor='k', markersize=10, linewidth=0, label="DDLM-Extractor"),
    Line2D([0], [0], linestyle='--', color="#8a8a8a", label="Pareto frontier")
]
axes[1].legend(handles=legend_handles, loc='lower right', fontsize=8, handletextpad=0.6)

fig.tight_layout(w_pad=1.2)
plt.savefig("Fig_Pareto_TNNLS_final.pdf", bbox_inches='tight')
plt.show()
