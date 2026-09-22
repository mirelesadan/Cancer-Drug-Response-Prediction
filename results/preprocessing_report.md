# GDSC2 preparation and frozen cell-line split

Prepared 2026-09-21 with `configs/preprocessing.json` and `src/prepare_gdsc2.py`. This is a data-preparation report; no predictive model was fitted and no test performance was computed. The tracked [manifest](preprocessing_manifest.json) records raw and artifact SHA-256 hashes, versions, shapes, counts, and training variance retained. Raw and processed row-level data remain local under Git-ignored `data/`.

## Response handling and provenance

The [TDC DrugRes file](https://tdcommons.ai/multi_pred_tasks/drugres/) exposes drug name `ID1`, cell-line name `ID2`, SMILES `X1`, expression `X2`, and supplied response `Y`; it exposes no original drug or experiment identifier. [Sanger's GDSC drug-sensitivity documentation](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) defines upstream `DRUG_ID`, `NLME_RESULT_ID`, and `NLME_CURVE_ID` fields and says GDSC2 contains multiple compound entries due to internal tracking. None of those identifiers are in this TDC snapshot, so no row-level crosswalk can distinguish Uprosertib entries. Its conflicting pairs cannot be assumed interchangeable biological or technical replicates. The original pickle is unchanged.

- **Primary table:** exclude **all 1,474 Uprosertib source rows** (800 drug–cell pairs), leaving **91,229 rows**, 805 cell lines, and 136 drugs. All other pairs occur once. The table keeps each original zero-based `source_row_id` and the untransformed `Y` bit-for-bit.
- **Sensitivity table:** retain Uprosertib and represent each drug–cell pair once: **92,029 pairs**, 805 cell lines, 137 drugs. Uprosertib contributes 800 pairs: 674 pairs with two measurements (1,348 source rows) and 126 singleton pairs. Before aggregation, the script verified that every pair has identical SMILES and expression representations by value; it also verified a single representation per drug and cell line globally. For each pair it keeps all source-row IDs, measurement count, minimum, maximum, range, and **arithmetic-mean `Y`**. Averaging the conflicting Uprosertib measurements is an analytical assumption for sensitivity analysis, not an established correction or a claim that the source entries are equivalent. Singleton `Y` values are unchanged.
- Source-row IDs are stable zero-based positions in the checksum-identified pickle. `source_row_mapping.csv` maps every source row to its sensitivity pair and, when retained, its primary row. No other records were removed.
- Both tables keep values on the provided `Y` scale. No logarithm, rescaling, or other target transformation was applied. The remaining uncertainty about the exact upstream concentration reference and TDC processing is documented in [the audit](data_audit.md).

## Split and coverage

The sole split was generated from the **sorted 805 unique cell-line IDs** using NumPy `default_rng(17).permutation` and rounded 70%/15% counts. Explicit assignments are saved in `cell_line_assignments.csv`; response rows reference the same assignment in both tables. The split was not selected using response values or changed to improve coverage.

| Partition | Cell lines | Primary rows | Sensitivity pairs | Uprosertib sensitivity pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 564 | 63,946 | 64,506 | 560 |
| Validation | 121 | 13,661 | 13,782 | 121 |
| Test | 120 | 13,622 | 13,741 | 119 |

Cell-line sets are disjoint and every source row maps to exactly one assignment. All **136 primary drugs** and all **137 sensitivity drugs** have training observations. The minimum training count for a primary drug is 33. All primary drugs have validation observations (minimum 5). **BI-2536 has no test observations** in this fixed split; it can be fitted and validated but will have no drug-specific test estimate. Eleven other drugs have only 1–4 test rows, so their test correlations may be undefined or unstable. These limitations are recorded without reseeding.

## Stored features and train-only fitting

- A sorted 805 × 17,737 `float64` matrix stores **one raw expression profile per cell line**. The feature catalog pairs each original gene symbol with its input position and stable `expr_00000`-style feature ID. All **317 repeated gene-symbol positions** remain separate; no averaging, merging, or column overwriting occurs.
- Zero-variance filtering removed **0 of 17,737** columns. `StandardScaler` and seeded randomized PCA with **128 components** were fitted on **564 unique training cell lines only**, with one profile per cell line. The saved 805 × 128 transformed matrix applies those fitted objects to all cells. The components account for **65.303% of total standardized training-expression variance**. The remaining approximately 34.697% of training variance is omitted; unsupervised variance retention does not prove preservation of response-predictive signal.
- A sorted drug catalog stores one SMILES per drug. RDKit **2023.09.6** generated one 2,048-bit Morgan fingerprint per drug (radius 2, binary bits, chirality and bond types included, ring membership included, no count simulation or redundant environments), yielding a 137 × 2,048 `uint8` matrix. Config and code fix the exact options and drug ordering.
- Response rows reference `cell_index` and `drug_index`; the large feature arrays are never repeated per measurement. The fitted selector, scaler, PCA, training-ID list, and feature-ID ordering are in the ignored `expression_transform.joblib`. Save/load this local artifact only from a trusted run.

## Verification and later comparison

`python -m unittest discover -s tests -v` passed four synthetic checks for deterministic, disjoint ID splits; duplicate-symbol preservation; train-only fitting despite a held-out outlier; and deterministic Morgan bits. `src/check_preprocessing.py` passed saved-artifact checks for hashes, every source-label mapping, pair means and ranges, response-to-feature indexes, split assignments and training-drug coverage, finite outputs, transform replay, fingerprint replay, and training-only scaler/PCA statistics. A second full preparation run produced the same manifest and artifact hashes.

Later primary and sensitivity analyses will use **these exact cell-line assignments**. Compare performance on **common non-Uprosertib evaluation pairs separately from Uprosertib pairs**, so any change caused by retaining Uprosertib is visible. Do not use this report or table creation as a test-set performance assessment.
