# Locked sensitivity analysis and held-out evaluation

Completed 2026-09-21. The [final protocol](../configs/final_evaluation.json) was written and its 16 input/model SHA-256 hashes verified **before reading test outcomes**. The validation-selected model was per-drug Ridge. The primary response table, seed-17 cell-line split, transforms, three primary model fits, and three selected-rate MLP checkpoints remained unchanged. Neither training-plus-validation refitting, extra tuning, seed selection by test result, nor ensembling was performed. The separate sensitivity fits use the pre-existing mean-aggregated table. Source code: [sensitivity](../src/run_sensitivity.py), [evaluation](../src/evaluate_final.py), and [independent replay](../src/check_final.py).

## Primary test results

All values below use the **same 13,622 non-Uprosertib response rows from 120 held-out cell lines**. There are 135 drugs with test observations; BI-2536 has zero and receives no test metric. RMSE and MAE give each response row equal weight. Equal-drug mean RMSE averages the 135 drug-level RMSEs, one vote per drug. Within-drug Spearman is the median of defined drug-level correlations; eligibility requires at least three test responses, nonconstant targets, and nonconstant predictions. Correlations that fail these rules are missing, never zero. Full precision, drug counts, and statuses are in the [comparison table](final_test_comparison.csv) and [drug-level table](final_test_by_drug.csv).

| Locked model | Test RMSE | Test MAE | Equal-drug mean RMSE | Median Spearman; defined/eligible |
| --- | ---: | ---: | ---: | ---: |
| Per-drug training mean | 1.5427 | 1.1785 | 1.4729 | undefined; 0/124 |
| Pooled Ridge, α=10 | 1.3539 | 1.0186 | 1.3063 | 0.5350; 124/124 |
| **Per-drug Ridge, α=10** | **1.2628** | **0.9536** | **1.2345** | **0.5626; 124/124** |
| MLP, seed 17 | 1.3294 | 0.9965 | 1.3063 | 0.5377; 124/124 |
| MLP, seed 29 | 1.3375 | 1.0088 | 1.3199 | 0.5355; 124/124 |
| MLP, seed 43 | 1.3372 | 1.0117 | 1.3237 | 0.5217; 124/124 |
| MLP metric mean ± sample SD | 1.3347 ± 0.0046 | 1.0056 ± 0.0080 | 1.3166 ± 0.0091 | 0.5316 ± 0.0087; 124/124 |

The MLP summary is the arithmetic mean and sample SD of **three separate seed-specific metrics**. It is not a prediction ensemble. Eleven tested drugs have fewer than three responses, making their correlations undefined; the per-drug mean's predictions are constant within every drug. Pooled response performance is therefore supplemented by within-drug behavior and equal-drug error. Per-drug Ridge wins on all four summaries in this split. The small MLP did not add value beyond per-drug Ridge, even though each MLP seed improves upon the per-drug training mean.

## Paired uncertainty on the primary test set

We drew **2,000** bootstrap samples with seed **2026**, sampling all **120 whole cell-line clusters** with replacement for each draw. Every selected cluster contributes all its drug-response rows, repeated according to its draw multiplicity. The same draws were applied to each model. The intervals are empirical 2.5th–97.5th percentiles; the paired difference is `other RMSE − per-drug Ridge RMSE`, so a positive value favors per-drug Ridge. [Full intervals](final_bootstrap_intervals.csv) and ignored per-draw values are saved. The MLP summary is re-averaged from the three seed-specific RMSE/MAE values **inside each draw**; no predictions were averaged.

| Model | RMSE 95% interval | MAE 95% interval | Paired RMSE difference 95% interval |
| --- | ---: | ---: | ---: |
| Per-drug mean | [1.4542, 1.6335] | [1.1040, 1.2555] | [0.2041, 0.3600] |
| Pooled Ridge | [1.2899, 1.4209] | [0.9709, 1.0697] | [0.0673, 0.1154] |
| Per-drug Ridge | [1.1969, 1.3297] | [0.9045, 1.0046] | 0 by definition |
| MLP, three-seed metric mean | [1.2555, 1.4142] | [0.9462, 1.0648] | [0.0398, 0.1022] |

All paired RMSE-difference intervals are above zero. These intervals capture variation from resampling cell lines **conditional on these fitted models and this particular split**. They are distinct from the MLP training-seed SD and do not cover retraining uncertainty, a different held-out split, other drugs, or patients. The absolute intervals can overlap even when the paired difference interval does not, because paired draws retain shared cell-line difficulty.

## Uprosertib sensitivity

The primary analysis excludes all **1,474** Uprosertib source responses because conflicting values may reflect distinct compound or experiment entries that the TDC snapshot does not identify. The separate sensitivity table retains **800** Uprosertib drug–cell pairs after verifying identical molecular and expression representations within each pair and taking an arithmetic mean of `Y` (560 train, 121 validation, 119 test). It retains every source-row ID, measurement count, and response range. **Averaging is an analytical assumption, not an established data correction.** The primary exclusion is unchanged.

Sensitivity baselines were fitted only on sensitivity training responses using α=10; sensitivity MLPs started afresh at the already selected 3e-4 rate, seeds 17/29/43, for their corresponding primary best-checkpoint epochs **3/4/4**. No sensitivity validation choice was made. The same saved cell-line assignments and feature transforms were used. Common non-Uprosertib validation pairs (13,661) and test pairs (13,622) were aligned by source-row identity and have identical labels. Per-drug mean and per-drug Ridge common-pair predictions are unchanged within 1e-8 in both partitions, as expected because their fits are independent by drug. Pooled Ridge and MLP fits can change when Uprosertib is included.

| Model | Primary common test RMSE | Sensitivity common test RMSE | Sensitivity Uprosertib test RMSE (119 pairs) |
| --- | ---: | ---: | ---: |
| Per-drug mean | 1.5427 | 1.5427 | 1.5456 |
| Pooled Ridge | 1.3539 | 1.3542 | 1.3811 |
| Per-drug Ridge | 1.2628 | 1.2628 | 1.3895 |
| MLP, mean of three seed metrics | 1.3347 | 1.3423 | 1.3779 |

On identical common validation pairs, the three-seed MLP mean RMSE is 1.3316 primary versus 1.3359 sensitivity. The common test MLP mean is slightly worse under sensitivity (1.3423 versus 1.3347); individual seeds move in different directions. The added Uprosertib results are shown separately and cannot be pooled with the primary test metric for a fair comparison. They do not establish which original Uprosertib measurement is correct. [Validation sensitivity table](sensitivity_validation_comparison.csv) and [test sensitivity table](final_sensitivity_comparison.csv) provide each seed and supporting metrics.

## Drug-level errors and limits

The [drug-level figure](figures/drug_level_errors.png) plots mean MLP seed-specific drug RMSE against per-drug Ridge on identical test pairs. For drugs with at least 20 test rows, the MLP has larger mean drug RMSE on Trametinib (1.771 vs 1.480; 116 rows), Sapitinib (1.674 vs 1.414; 120 rows), and Dasatinib (2.043 vs 1.829; 115 rows). It performs better on MG-132 (0.622 vs 0.704; 116 rows). These are descriptive examples; sparse drugs need particular caution. Two large per-drug Ridge residuals are Gemcitabine–HH (`Y` 5.591, prediction −3.162; absolute error 8.754) and AZD5582–SK-OV-3 (`Y` −4.507, prediction 3.343; absolute error 7.851). Such errors warrant source-data review and independent replication before biological interpretation. The [test-performance figure](figures/test_performance.png) shows cluster-bootstrap uncertainty; the [learning curves](figures/mlp_learning_curves.png) show training RMSE continuing downward after validation RMSE bottoms out at epoch 3 or 4, consistent with early overfitting in this MLP.

This study has **one frozen cell-line split**, not repeated external validation. It targets known drugs and experimental cell lines only. It uses the dataset-provided `Y` unchanged: local loader code passes it through; upstream Sanger describes `LN_IC50` as a natural logarithm, and TDC documents the underlying concentration in µM, while the exact snapshot-to-upstream row mapping and logged concentration reference remain unresolved ([data audit](data_audit.md)). The 128-component expression PCA retains 65.303% of standardized training variance, so discarded components may contain predictive signal. Duplicate gene symbols remain separate positional features. The Uprosertib mean depends on an unverified interchangeability assumption. No result estimates clinical treatment effectiveness or response to an unseen drug.

## Reproduction and verification

Install the [final environment recipe](../configs/environment-final.txt) in the dedicated Python 3.12 environment; the [full tested package lock](../configs/environment-final-lock.txt) records the actual installed versions. These commands assume Milestones 2–5 have regenerated the ignored raw data, processed arrays, and primary checkpoints. The frozen protocol's artifact hashes guard against changed inputs. Reproducing sensitivity fits on another GPU or software build may not yield byte-identical MLP checkpoints; the method and seeds remain fixed.

```powershell
& ".venv\Scripts\python.exe" "src\run_sensitivity.py"
& ".venv\Scripts\python.exe" "src\evaluate_final.py"
& ".venv\Scripts\python.exe" "src\check_final.py"
& ".venv\Scripts\python.exe" "tests\test_final_evaluation.py"
& ".venv\Scripts\python.exe" -m pip check
```

The [evaluation manifest](final_evaluation_manifest.json) records hashes and runtime. [Replay results](final_replay_check.json) verify row identity, all 12 reloaded model prediction columns, metrics, bootstrap cluster multiplicities, and saved hashes. Row-level predictions, fitted checkpoints, and per-draw bootstrap arrays stay in Git-ignored locations.
