# RSVP_domain_adaption

Cross-subject RSVP-EEG baseline and domain-adaptation codebase.

## Current Scope

- Cross-subject only (`LOSO`): hold out one subject as target, others as source.
- Supported backbones: `DeepConvNet`, `EEGNet`, `PLNet`, `EEGInception`, `PPNN`.
- Baseline training: source-supervised CE with class weights from source-train split.
- Optional DA data stream: dual-stream loader can provide unlabeled `target_x` when needed.

## Project Structure

- `Configs/config.yaml`: global config + dataset/model settings.
- `Data/datamodule.py`: LOSO split, stratified per-subject train/val split, dataloaders.
- `Models/`: backbone models + selector wrappers.
- `Train/trainer.py`: experiment loop, training/evaluation, checkpoint/results writing.
- `Utils/`: config builder, losses, metrics, misc training utilities.
- `main.py`: single experiment entry.
- `run_all_models.py`: run all kept backbones on all configured datasets.

## Data Pipeline (Baseline)

1. Build LOSO fold: `sub_i` as target test domain; remaining subjects as source.
2. For each source subject, split trials into train/val by class (80/20).
3. Merge source-train and source-val across subjects.
4. Train with source labeled data only (`use_target_stream: false`).
5. Validate on source-val and test on held-out target subject.

## Key Config Flags

In `Configs/config.yaml`:

- `dataset_root`: base path to preprocessed `.npz` dataset.
- `class_weighted_ce`: enable class-weighted CE (recommended for imbalance).
- `use_target_stream`:
  - `false`: source-only baseline training.
  - `true`: use dual-stream loader `(source_x, source_y, target_x)` for DA methods.

You can also set dataset root via env var `RSVP_DATASET_ROOT`.

## Run

```bash
python main.py
```

or

```bash
python run_all_models.py
```
