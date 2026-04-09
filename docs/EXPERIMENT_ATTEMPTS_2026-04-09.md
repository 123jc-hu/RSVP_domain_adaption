# RSVP Domain Adaptation / Generalization Attempt Log

Date: 2026-04-09

This note summarizes the main experimental branches explored so far, what was kept, and what was ruled out. It is meant to be a compact reference before code/log cleanup.

## Current strong baselines

### 1. EEGNet + EA + global MMD
- Status: stable, strong baseline
- Typical setting:
  - `source_selection=All`
  - `random_k`
  - `bgds4`
  - held-out `sub1~20`
- Role:
  - primary baseline for later methods

### 2. EEGNetDSBN + EA + global MMD
- Status: stronger engineering baseline than plain EEGNet
- Main observation:
  - improves `AUC / BA / F1`
  - typically lowers `FPR`
- Limitation:
  - method novelty is weak
  - tied to BN placement

### 3. EEGNetLDSA / EEGNetSWLDSA
- Status: effective latent-statistics refinements over DSBN
- LDSA:
  - target stats blended with source prior
- SWLDSA:
  - source prior built by target-source similarity weighting
- Main observation:
  - `SWLDSA` > `LDSA` > `DSBN` on quick comparisons
- Limitation:
  - still feels like a BN/statistics trick, not a clean model-agnostic framework

## Attempted branches and conclusions

### A. Tangent-space / manifold branch
- Files involved:
  - `EEGNetTS`
  - auxiliary TS / dual-head variants
- Main observations:
  - tangent-space branch contains information
  - often lowers `FPR`
  - but did not stably beat the main EEGNet baseline
- Variants tried:
  - `TS@block1`
  - `TS@block2`
  - pre-activation vs post-activation
  - head channels `8 / 16 / 32`
  - shrinkage covariance
  - dual-head fusion
  - auxiliary TS regularization
- Conclusion:
  - useful diagnostic line
  - not worth continuing as main method in current form

### B. Prototype / center / distribution branches
- Variants tried:
  - symmetric point prototype
  - asymmetric point prototype
  - positive-only distribution prototype
- Main observations:
  - point prototype methods clearly hurt, mainly by raising `FPR`
  - positive-only distribution version was stable but gains were negligible
- Conclusion:
  - not worth further investment for the current RSVP setting

### C. Latent style adapter / feature-space style transfer
- Variants tried:
  - minimal LSA
  - similarity-weighted style prior
  - content preservation
  - identity restraint
- Main observations:
  - idea is interesting and likely more method-like
  - two minimal versions still underperformed LDSA/SWLDSA
- Conclusion:
  - keep as future innovation direction
  - current implementations not strong enough

### D. RSF / domain-aware RSF
- Reference:
  - RSF paper is subject-dependent, not cross-subject LOSO
- Variants tried:
  - plain `EA -> RSF -> EEGNet + global MMD`
  - balanced-support RSF diagnostic
  - class-conditional domain-aware RSF
- Main observations:
  - plain RSF performs poorly in cross-subject RSVP
  - class imbalance affects it, but is not the main failure source
  - domain-aware RSF was more reasonable than plain RSF, but still far below baseline
- Conclusion:
  - RSF branch currently not strong enough to justify further focus

### E. UOT replacing global MMD
- Variant:
  - `EA -> EEGNet -> UOT`
- Main observations:
  - training can be stabilized
  - best current point: `eps=0.5`, `lambda_align=0.05`
  - performance is very close to global MMD, but not clearly better
- Conclusion:
  - feasible, but not a strong win
  - not worth deep hyperparameter investment for now

### F. Golden-subject anchor ideas
- Variants tried:
  - hard top-1 by similarity (`GSLDSA`)
- Main observations:
  - hard top-1 anchor is more brittle than soft mixture
  - compared with `SWLDSA`, it raises `TPR` a bit but also raises `FPR` noticeably
- Conclusion:
  - soft similarity-weighted source mixture is better than hard top-1 by similarity

### G. Top-1 by discriminability anchor
- Status:
  - smoke version wired up
  - not promoted to main experiment
- Main observation so far:
  - setup overhead is heavy
  - no evidence yet that it is promising enough to replace current anchors
- Conclusion:
  - deprioritized for now

## Current methodological tension

The strongest-performing line so far is:
- `EA + EEGNet + global MMD + DSBN/LDSA/SWLDSA`

But this line has limited method novelty because it is:
- tied to BN-based implementation details
- somewhat architecture-specific

This motivates a shift toward a cleaner framework-like direction.

## Recommended next research direction

The most promising next direction is:
- a model-agnostic framework
- preferably not tied to one BN layer
- ideally compatible with multiple backbones

Two candidate directions emerged:

### 1. Domain Generalization (preferred for framework novelty)
- episodic multi-subject training
- subject-invariant representation learning
- no target-domain participation during adaptation

### 2. Source-only learnable alignment matrix
- inspired by AEA, but adapted to the current source-accessible setting
- learn a reusable alignment matrix from source domains
- use it before any backbone at test time

## Practical takeaway

Keep as strong baselines:
- `EEGNet + EA + global MMD`
- `EEGNetDSBN + EA + global MMD`
- `EEGNetLDSA`
- `EEGNetSWLDSA`

Treat as archived/abandoned branches:
- tangent-space family
- prototype family
- RSF family
- current latent-style-adapter family
- UOT as a main replacement
- hard golden-subject anchor
