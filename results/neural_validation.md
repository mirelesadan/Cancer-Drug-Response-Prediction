# MLP training and validation

Completed 2026-09-21. These are results on the **frozen primary training and validation partitions only**. No test predictions or metrics, Uprosertib sensitivity analysis, or new architecture were run. Exact settings are in `configs/neural_mlp.json`; [individual runs](neural_runs.csv), [three-seed summaries](neural_summary.csv), [drug-level validation metrics](neural_validation_by_drug.csv), and the [manifest](neural_validation_manifest.json) are tracked compact results. Six best checkpoints, per-epoch learning curves, and row-level validation predictions are saved locally under Git-ignored `results/checkpoints/` and `results/tmp/`.

## Environment and fixed method

The dedicated Python 3.12.8 environment gained **PyTorch 2.11.0+cu128** from the [official Windows CUDA 12.8 wheel index](https://pytorch.org/get-started/previous-versions/). It passed `pip check` and a CUDA tensor operation on the **NVIDIA GeForce RTX 4070 Laptop GPU** (8.0 GiB, driver 610.88). The complete package set is in `configs/environment-neural-lock.txt`; the direct install recipe is in `configs/environment-neural.txt`. No torchvision or torchaudio package was needed.

The model concatenates the existing **128 standardized expression PCA scores** and **2,048 binary Morgan fingerprint bits** for each response row. It has hidden widths **256 → 128**, ReLU and dropout **0.1** after each hidden activation, then one unrestricted linear output. It minimizes MSE on the unchanged supplied `Y` scale with AdamW, weight decay **1e-4**, and batch size **256**. Only training response rows update weights. Index-based feature assembly builds each minibatch from one saved profile per cell line and one fingerprint per drug; it does not replicate a full feature matrix for every response row.

The only compared learning rates were **1e-3** and **3e-4**, each with seeds **17, 29, 43**. Seeds set initialization and minibatch order; they never alter the saved cell-line assignment. Each run had a 100-epoch cap and stopped after 10 epochs without a strictly lower validation RMSE. The best checkpoint was saved, restored, and scored in evaluation mode with dropout disabled. The selection rule was lowest **mean best-checkpoint validation RMSE across all three seeds**; all three checkpoints at the selected rate are retained. Deterministic PyTorch algorithms were enabled with cuDNN benchmarking off, TF32 off, no data-loading worker subprocesses, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. These settings do not imply byte-identical reproduction across GPU hardware or software builds.

## Six runs

| Learning rate | Seed | Best / last epoch | Train RMSE | Validation RMSE | Validation MAE | Median within-drug Spearman | Runtime (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 17 | 1 / 11 | 1.1527 | 1.3646 | 1.0365 | 0.5465 | 29.8 |
| 1e-3 | 29 | 1 / 11 | 1.1669 | 1.3331 | 1.0113 | 0.5323 | 34.2 |
| 1e-3 | 43 | 1 / 11 | 1.1629 | 1.3555 | 1.0322 | 0.5363 | 32.1 |
| **3e-4** | **17** | **3 / 13** | **1.0510** | **1.3278** | **1.0009** | **0.5479** | **36.7** |
| **3e-4** | **29** | **4 / 14** | **1.0000** | **1.3393** | **1.0149** | **0.5436** | **41.0** |
| **3e-4** | **43** | **4 / 14** | **0.9899** | **1.3278** | **1.0107** | **0.5193** | **46.3** |

All runs stopped for patience; **none reached the epoch cap**. Each run's validation metrics are computed on **13,661 response rows** and training metrics on **63,946 rows** in evaluation mode. Each run had **136/136 defined** within-drug Spearman estimates under the existing rule: at least three validation rows, nonconstant target, and nonconstant predictions. Drug-level counts, errors, correlations, and status are saved in the linked table.

| Learning rate | Mean validation RMSE ± sample SD | Mean validation MAE ± sample SD | Mean of run-level median Spearman ± sample SD | Best epochs | Three-run runtime |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 1.3511 ± 0.0162 | 1.0266 ± 0.0135 | 0.5384 ± 0.0073 | 1, 1, 1 | 96.0 s |
| **3e-4 selected** | **1.3316 ± 0.0066** | **1.0088 ± 0.0072** | **0.5369 ± 0.0154** | **3, 4, 4** | **124.0 s** |

The full invocation, including input checks, smoke check, training, checkpointing, tables, and hashes, took **232.3 seconds**. The SD values describe **training-seed variability on one fixed cell-line split**. They are not confidence intervals and do not measure uncertainty from sampling different cell lines; cell-line-clustered uncertainty belongs to final evaluation.

## Comparison and interpretation

| Validation model | Response-weighted RMSE | Response-weighted MAE | Equal-drug mean RMSE | Median within-drug Spearman |
| --- | ---: | ---: | ---: | ---: |
| Per-drug training mean | 1.5283 | 1.1651 | 1.4936 | undefined: constant within-drug predictions |
| Pooled Ridge, α=10 | 1.3489 | 1.0161 | 1.3068 | 0.5226 |
| **Selected MLP, mean over three seeds** | **1.3316** | **1.0088** | **1.3306** | **0.5369** |
| Per-drug Ridge, α=10 | **1.2367** | **0.9352** | **1.2394** | **0.5725** |

The selected MLP improves response-weighted validation RMSE over pooled Ridge by about **0.0173**, but its **equal-drug mean RMSE is worse** (1.3306 versus 1.3068). Its advantage therefore depends on how drugs are weighted and should not be interpreted as a uniform drug-level gain. The MLP does **not** outperform per-drug Ridge: its mean validation RMSE is higher by about **0.0949**. Per-drug Ridge also leads on equal-drug RMSE and median within-drug Spearman. The MLP can model nonlinear expression–drug interactions, but this fixed architecture and training schedule did not yield the strongest validation result.

Training RMSE at the selected checkpoints averaged **1.0136**, versus validation RMSE **1.3316**. Learning curves show validation error rising while training error continues to fall after the best epoch, particularly at 1e-3. Early stopping limited this overfitting; the best epochs occurred very early (1–4). No extra rates, patience settings, or architectures were tried. These observations are conditional on the validation split; final test performance remains unknown.

## Verification and limits

The 32-row CUDA smoke check passed for feature order, device placement, finite one-step loss, output/target shape, and eval-mode inference. Eight focused tests passed for preprocessing, Ridge, MLP architecture, and feature assembly. The training script checked frozen input hashes, Uprosertib exclusion, split and feature–label alignment, finite batch losses and eval metrics, and checkpoint restoration. `src/check_neural.py` independently replayed **all six** validation prediction columns after checkpoint reload and checked row IDs, labels, per-run metrics, selected seeds, and artifact hashes. GPU runs need not be byte-identical elsewhere. BI-2536 still has no test rows in the frozen split. No conclusions here concern patient treatment or unseen drugs.
