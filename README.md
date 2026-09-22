# Cancer drug-response prediction for unseen cell lines

## Research question

How accurately can gene expression and molecular features predict drug response in **previously unseen cancer cell lines**, for **drugs represented in training**? This is an experimental cell-line study. Its results do not establish patient treatment effectiveness or generalization to unseen drugs.

## Data and evaluation

The study uses the [TDC GDSC2 DrugRes dataset](https://tdcommons.ai/multi_pred_tasks/drugres/): gene-expression vectors for cell lines, drug SMILES, and the dataset-provided response `Y`. The raw TDC snapshot has 92,703 rows. The primary analysis excludes all 1,474 Uprosertib rows because conflicting drug–cell responses cannot be matched to distinct upstream compound or experiment IDs. It analyzes **91,229 response rows, 805 cell lines, and 136 drugs**. The separate [sensitivity analysis](results/final_evaluation.md#uprosertib-sensitivity) evaluates an explicitly assumed arithmetic mean for Uprosertib pairs.

One seed-17 assignment splits unique cell-line identities into **564 training, 121 validation, and 120 test cell lines** (approximately 70/15/15). Every response from one cell line stays in one partition; every evaluated drug appears in training. The split was fixed before fitting, and validation selected hyperparameters and MLP checkpoints. No model was refit on training plus validation. The test set contains 13,622 non-Uprosertib responses from 120 held-out cell lines. BI-2536 has no test observations under this fixed split.

Expression is stored once per cell line and represented by 128 PCA scores, then standardized; variance filtering, scaling, PCA, and score scaling were fitted on **unique training cell lines only**. A 2,048-bit radius-2 Morgan fingerprint is stored once per drug. The target remains on the provided `Y` scale, without another logarithm. See the [data audit](results/data_audit.md) and [preprocessing report](results/preprocessing_report.md) for schema, checks, hashes, and handling choices.

## Held-out results

All rows in this table refer to the **same primary test pairs**. RMSE and MAE weight each response equally. The MLP entry averages three separately trained seed-specific **metrics**, not their predictions.

| Model | Test RMSE | Test MAE |
| --- | ---: | ---: |
| Per-drug training mean | 1.5427 | 1.1785 |
| Pooled Ridge | 1.3539 | 1.0186 |
| **Per-drug Ridge** | **1.2628** | **0.9536** |
| MLP, three-seed metric mean ± sample SD | 1.3347 ± 0.0046 | 1.0056 ± 0.0080 |

Validation-selected **per-drug Ridge reduced test RMSE by 18.1% versus the per-drug training-mean baseline**: `(1.542654 - 1.262829) / 1.542654`. The fixed 256→128 PyTorch MLP improved on the mean baseline but did **not** improve on per-drug Ridge. Pooled Ridge adds a common expression effect and drug fingerprint effect; separate per-drug Ridge fits allow drug-specific expression coefficients. [Full test metrics](results/final_test_comparison.csv), [drug-level results](results/final_test_by_drug.csv), and the [final report](results/final_evaluation.md) include equal-drug RMSE, within-drug Spearman with undefined cases, sample counts, and limitations.

![Held-out test RMSE with 95% paired cell-line bootstrap intervals](results/figures/test_performance.png)

The figure shows uncertainty from **2,000 paired whole-cell-line bootstrap draws**, conditional on the fitted models and this split. It does not measure variation from retraining or new drugs. The [paired intervals](results/final_bootstrap_intervals.csv), [drug-level error figure](results/figures/drug_level_errors.png), and [MLP learning curves](results/figures/mlp_learning_curves.png) provide complementary views.

## Data access and terms

No raw or processed dataset, row-level prediction, or fitted checkpoint is distributed here. `src/retrieve_gdsc2.py` downloads the two **TDC-identified Harvard Dataverse files** and verifies their MD5 hashes. The [data retrieval notes](data/README.md) and [audit](results/data_audit.md#environment-and-retrieval) record the file IDs, source URLs, retrieval date, SHA-256 hashes, and PyTDC loader compatibility issue. These files identify the TDC snapshot but do not resolve its exact upstream GDSC release or row-level mapping.

TDC lists [CC BY-NC-ND 2.5 for GDSC2](https://tdcommons.ai/multi_pred_tasks/drugres/); the upstream [Sanger data-use policy](https://depmap.sanger.ac.uk/documentation/data-usage-policy/) also applies. Retrieve the data from the source and review those terms for your use. This repository does **not** assert permission to redistribute the dataset or derived row-level records.

## Reproduction and checks

Run from the repository root with **Python 3.12**. Create `.venv` using your preferred Python manager; the commands below use the Windows virtual-environment executable and relative paths. The tested NVIDIA setup is captured in [configs/environment-final.txt](configs/environment-final.txt) and its [full lock](configs/environment-final-lock.txt). That recipe pins a CUDA-enabled PyTorch wheel; for a CPU or other accelerator setup, select the matching wheel from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) while keeping the other pinned dependencies. The research scripts also support CPU execution.

```powershell
& ".venv\Scripts\python.exe" --version
& ".venv\Scripts\python.exe" -m pip install -r "configs\environment-final.txt"
& ".venv\Scripts\python.exe" "src\retrieve_gdsc2.py"
& ".venv\Scripts\python.exe" "src\audit_gdsc2.py"
& ".venv\Scripts\python.exe" "src\prepare_gdsc2.py"
& ".venv\Scripts\python.exe" "src\check_preprocessing.py"
& ".venv\Scripts\python.exe" -m unittest discover -s "tests" -v
& ".venv\Scripts\python.exe" -m pip check
```

The version check should report Python 3.12. On macOS/Linux, replace the executable with `.venv/bin/python` and use equivalent shell syntax. Model scripts are [`run_baselines.py`](src/run_baselines.py), [`run_neural.py`](src/run_neural.py) (`--smoke-only` is available), [`run_sensitivity.py`](src/run_sensitivity.py), and [`evaluate_final.py`](src/evaluate_final.py), with matching `check_*.py` replay checks in `src/`. The [baseline](results/baseline_validation.md), [MLP](results/neural_validation.md), and [final](results/final_evaluation.md#reproduction-and-verification) reports state their order and settings.

**Reproduction boundary:** `configs/final_evaluation.json` is a locked record of the original run and checks exact input, manifest, and checkpoint hashes. A newly trained MLP or regenerated manifest may have different bytes, so the final sensitivity/evaluation scripts can intentionally reject a fresh run even when its method is the same. The published aggregate results document the original locked evaluation; exact replay requires its locally retained, Git-ignored artifacts. Do not treat the frozen lock as a portable promise of byte-identical GPU retraining.

## Limitations

The study uses one cell-line split and only drugs seen during training. The response is the provided `Y` scale: TDC calls it log-normalized IC50, and Sanger describes upstream `LN_IC50` as a natural logarithm, but the exact snapshot-to-source mapping and concentration reference remain unresolved. The 128-component PCA retains **65.303%** of standardized training-expression variance, so discarded dimensions could contain predictive signal. Uprosertib's conflicting entries lack source identifiers; excluding them defines the primary analysis, while averaging them in the sensitivity table is an analytical assumption. These cell-line results are not clinical predictions.

## Repository contents

| Path | Contents |
| --- | --- |
| [`src/`](src/) | Retrieval, audit, preparation, modeling, evaluation, and replay code |
| [`tests/`](tests/) | Focused alignment, split, preprocessing, metric, and inference checks |
| [`configs/`](configs/) | Fixed configurations and environment specifications |
| [`results/`](results/) | Compact reports, aggregate tables, manifests, and figures |
| [`data/README.md`](data/README.md) | Retrieval and provenance guidance; data files are excluded |

Downloaded data, generated arrays, frozen splits, checkpoints, row-level predictions, and local application materials are intentionally outside the public files.
