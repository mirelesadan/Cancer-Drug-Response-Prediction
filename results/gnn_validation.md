# Exploratory molecular GNN: training and validation

Completed 2026-09-23 on the unchanged primary data and seed-17 cell-line assignment. This is an **optional architecture extension**, separate from the original validation-selected per-drug Ridge study. The GNN uses no test response for training or early stopping. Exact settings are in the [extension configuration](../configs/gnn_extension.json); [individual runs](gnn_validation_runs.csv), [drug-level metrics](gnn_validation_by_drug.csv), [aggregate learning curves](gnn_learning_curves.csv), [manifest](gnn_validation_manifest.json), and [independent validation replay](gnn_replay_check.json) retain the compact evidence. Checkpoints and row-level predictions remain local and Git-ignored.

## Fixed method

RDKit converts each frozen drug SMILES to one molecular graph, with **155 atom features**, **12 bond features**, and a directed edge in each direction per bond. A two-layer **GINEConv** encoder (64 hidden channels) with global mean pooling replaces the fixed Morgan fingerprint. Its 64-dimensional drug embedding is concatenated with the unchanged **128 standardized expression PCA scores** for each response, followed by a 256→128 ReLU head with dropout 0.1 and an unrestricted scalar output. `Y` stays on the provided scale. The primary Uprosertib exclusion, one profile per cell line, cell-line splits, and training-only expression preprocessing are unchanged.

The predeclared training setup uses MSE, AdamW, learning rate **3e-4**, weight decay **1e-4**, batch size **2,048**, seeds **17/29/43**, at most 100 epochs, and patience 10 on validation RMSE. Only training responses update weights. Each run restores its best validation checkpoint; all three seeds are retained. This was one fixed GNN specification, not an architecture or learning-rate search. PyTorch Geometric **2.8.0.post1** ran with PyTorch **2.11.0+cu128** on the NVIDIA GeForce RTX 4070 Laptop GPU; see the [environment recipe](../configs/environment-gnn.txt) and [installed lock](../configs/environment-gnn-lock.txt). GPU graph aggregation is not asserted byte-identical, so deterministic algorithms were disabled; seeds, cuDNN settings, TF32 setting, and the graph digest are recorded in the manifest.

## Results

Metrics below use **63,946 training** and **13,661 validation** primary response rows. Train and validation scores are calculated in evaluation mode. The equal-drug mean RMSE gives each of the 136 drugs one vote; response-weighted RMSE gives each row one vote. Spearman is computed within drugs, with at least three rows and nonconstant targets and predictions.

| Seed | Best / last epoch | Train RMSE | Validation RMSE | Validation MAE | Equal-drug mean RMSE | Median within-drug Spearman | Runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 46 / 56 | 1.4871 | 1.6288 | 1.2643 | 1.6182 | 0.4832 | 27.6 s |
| 29 | 42 / 52 | 1.5279 | 1.6957 | 1.3231 | 1.6741 | 0.4967 | 20.1 s |
| 43 | 61 / 71 | 1.3735 | 1.6422 | 1.2907 | 1.6711 | 0.4839 | 25.5 s |
| **Mean ± sample SD** | — | **1.4628 ± 0.0800** | **1.6556 ± 0.0354** | **1.2927 ± 0.0295** | **1.6545 ± 0.0314** | **0.4879 ± 0.0076** | **73.2 s total** |

All three runs stopped for patience and none hit the epoch cap. All **136/136 eligible** validation drugs had defined within-drug Spearman estimates for each GNN seed. The sample SD describes variation from initialization and training order on this one split, not uncertainty over sampled cell lines.

| Model | Validation response-weighted RMSE | Validation MAE | Equal-drug mean RMSE | Median within-drug Spearman |
| --- | ---: | ---: | ---: | ---: |
| Per-drug training mean | 1.5283 | 1.1651 | 1.4936 | undefined: constant predictions |
| Pooled Ridge | 1.3489 | 1.0161 | 1.3068 | 0.5226 |
| Per-drug Ridge | **1.2367** | **0.9352** | **1.2394** | **0.5725** |
| MLP, three-seed metric mean | 1.3316 | 1.0088 | 1.3306 | 0.5369 |
| **GNN, three-seed metric mean** | **1.6556** | **1.2927** | **1.6545** | **0.4879** |

This fixed GNN performed worse than all three classical references and the earlier MLP on validation RMSE. Its mean train-to-validation RMSE gap is about **0.193**; substantial drug-level offsets and within-drug errors both contribute to the validation loss. Neither these results nor the single architecture establish that graph representations in general are inferior to fingerprints. No additional GNN tuning was done after seeing these metrics.

## Verification and boundary

A CUDA smoke check verified row alignment, finite gradients, output/target shape, and evaluation-mode inference. Frozen input hashes, graph construction order, training-drug coverage, finite losses, and best-checkpoint restoration were checked. The independent replay reproduced all **13,661 validation predictions per seed** within `1e-5` maximum absolute difference (observed maximum `1.43e-6`), appropriate to the enabled CUDA graph operations. The original held-out test results were already public before this GNN architecture was chosen; the subsequent [held-out comparison](gnn_final.md) is therefore labeled **exploratory**, even though the GNN checkpoints and extension procedure were frozen before computing its test predictions.
