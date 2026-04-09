# RSVP_domain_adaption

Cross-subject RSVP-EEG baseline and domain-adaptation codebase.

## Current Mainline

当前主线先固定为 `EEGNet`，只保留 3 个主实验：

1. `Exp 0`: `All + bgds4 + CE`
2. `Exp 1`: `Exp 0 + EA`
3. `Exp 2`: `Exp 1 + global MMD`

当前不作为主线继续推进的分支包括：
- weighted sampling
- per-subject MMD / CORAL
- class-wise LMMD
- entropy minimization
- prior constraint
- IAHM
- RPT-Aug
- proxy validation

这些分支已经做过探索，但目前都没有稳定超过 `Exp 2`。

## Current Best Setting

当前 `EEGNet` 主线最优版本是：

- `EA + global MMD`
- 配置文件：
  - [eegnet_all_sub1_20_k32_bgds4_ea_globalmmd.yaml](/home/cdd/RSVP/domain_adaption/Configs/eegnet_all_sub1_20_k32_bgds4_ea_globalmmd.yaml)
- 结果文件：
  - [results.csv](/home/cdd/RSVP/domain_adaption/Experiments/EEGNet/THU/cross-subject/eegnet_all_sub1_20_k32_bgds4_ea_globalmmd/results.csv)

20-sub 平均结果：
- `AUC = 0.8752`
- `BA = 0.8013`
- `F1 = 0.9154`
- `TPR = 0.7337`
- `FPR = 0.1311`

## Mainline Experiments

### Exp 0: All + bgds4 + CE

目的：
- 建立当前快速开发基线。

设置：
- backbone: `EEGNet`
- source policy: `All`
- source-train background downsampling: `1:4` (`bgds4`)
- loss: source-only class-weighted CE
- target stream: 关闭
- DA loss: 无
- preprocessing: 无

配置：
- [eegnet_all_sub1_20_k32_bgds4.yaml](/home/cdd/RSVP/domain_adaption/Configs/eegnet_all_sub1_20_k32_bgds4.yaml)

结果：
- [results.csv](/home/cdd/RSVP/domain_adaption/Experiments/EEGNet/THU/cross-subject/eegnet_all_sub1_20_k32_bgds4/results.csv)
- 平均：
  - `AUC = 0.8712`
  - `BA = 0.7892`
  - `F1 = 0.9197`
  - `TPR = 0.7007`
  - `FPR = 0.1224`

### Exp 1: + EA

目的：
- 验证数据级二阶统计对齐是否有稳定增益。

设置：
- 在 `Exp 0` 基础上新增 `EA`
- 其余保持不变

EA 含义：
- 每个 subject 独立计算自身全部 trial 的平均协方差
- 用该协方差的逆平方根对白化该 subject 的所有 trial
- source 和 target 都做
- 不共享 target 协方差，不用 target 去变换 source

实现位置：
- [npz_io.py](/home/cdd/RSVP/domain_adaption/Data/npz_io.py)

配置：
- [eegnet_all_sub1_20_k32_bgds4_ea.yaml](/home/cdd/RSVP/domain_adaption/Configs/eegnet_all_sub1_20_k32_bgds4_ea.yaml)

结果：
- [results.csv](/home/cdd/RSVP/domain_adaption/Experiments/EEGNet/THU/cross-subject/eegnet_all_sub1_20_k32_bgds4_ea/results.csv)
- 平均：
  - `AUC = 0.8758`
  - `BA = 0.7988`
  - `F1 = 0.9200`
  - `TPR = 0.7203`
  - `FPR = 0.1227`

结论：
- `EA` 有明确正收益。
- 相比 `Exp 0`，主要提升来自 `TPR` 上升，而 `FPR` 基本不变。

### Exp 2: + global MMD

目的：
- 在 `EA` 基础上验证显式域对齐是否进一步有增益。

设置：
- 在 `Exp 1` 基础上开启 `target stream`
- loss:
  - `CE(source)`
  - `+ lambda_align * global_MMD(source_features, target_features)`
- 当前主线用的是 `global MMD`，不是 per-subject MMD，也不是 CORAL

实现位置：
- [trainer.py](/home/cdd/RSVP/domain_adaption/Train/trainer.py)

配置：
- [eegnet_all_sub1_20_k32_bgds4_ea_globalmmd.yaml](/home/cdd/RSVP/domain_adaption/Configs/eegnet_all_sub1_20_k32_bgds4_ea_globalmmd.yaml)

结果：
- [results.csv](/home/cdd/RSVP/domain_adaption/Experiments/EEGNet/THU/cross-subject/eegnet_all_sub1_20_k32_bgds4_ea_globalmmd/results.csv)
- 平均：
  - `AUC = 0.8752`
  - `BA = 0.8013`
  - `F1 = 0.9154`
  - `TPR = 0.7337`
  - `FPR = 0.1311`

结论：
- `Exp 2` 是当前 `EEGNet` 主线最优版本。
- 相比 `Exp 1`，`TPR` 继续提高，但 `FPR` 也会略升。
- 对主指标 `BA` 来说，这一步仍然是净增益。

## Why The Mainline Stops At Exp 2

原因不是后面的分支完全无效，而是它们目前都没有稳定超过 `Exp 2`。

已验证但暂不继续推进的方向：
- `weighted sampling`
  - 在 `EA + global MMD` 上没有额外净收益
- `per-subject MMD`
  - 与 `global MMD` 基本持平但略弱
- `global CORAL`
  - 与 `global MMD` 接近，但略弱
- `per-subject CORAL`
  - 未显示出优于 global 版本的趋势
- `plain entropy`
  - 会把模型往更保守方向推，`BA` 不增反降
- `prior constraint`
  - 小权重几乎不起作用；大权重会明显压低 `TPR`
- `class-wise LMMD`
  - 在 `global MMD` 之上没有继续带来增益

因此当前阶段更务实的做法是：
- 先把主线固定为 `Exp 0 / 1 / 2`
- 后续如果迁移到更强 backbone（如 `EEGInception`），优先复验这条已站住的主线

## Shared Training Setting

当前 `EEGNet` 主线默认共享设置：
- dataset: `THU`
- protocol: LOSO cross-subject
- held-out subjects: `sub1~20`
- `subject_batch_size = 64`
- `subjects_per_batch = 32`
- `bgds4`
- `early_stop_start_epoch = 30`
- `patience = 10`
- `learning_rate = 0.001`
- `random_seed = 2026`

## Project Structure

- [Configs/config.yaml](/home/cdd/RSVP/domain_adaption/Configs/config.yaml): global config + dataset/model defaults
- [Data/datamodule.py](/home/cdd/RSVP/domain_adaption/Data/datamodule.py): LOSO split, per-subject train/val split, loaders
- [Data/npz_io.py](/home/cdd/RSVP/domain_adaption/Data/npz_io.py): memmap loading and EA cache generation
- `Models/`: backbone models
- [Train/trainer.py](/home/cdd/RSVP/domain_adaption/Train/trainer.py): training loop, aux losses, evaluation
- `Utils/`: config builder, metrics, misc training helpers
- [main.py](/home/cdd/RSVP/domain_adaption/main.py): single experiment entry

## Run

运行单个配置：

```bash
cd /home/cdd/RSVP/domain_adaption
python run_model_config.py --config Configs/eegnet_all_sub1_20_k32_bgds4_ea_globalmmd.yaml
```

如果只跑主线三组：

```bash
python run_model_config.py --config Configs/eegnet_all_sub1_20_k32_bgds4.yaml
python run_model_config.py --config Configs/eegnet_all_sub1_20_k32_bgds4_ea.yaml
python run_model_config.py --config Configs/eegnet_all_sub1_20_k32_bgds4_ea_globalmmd.yaml
```
