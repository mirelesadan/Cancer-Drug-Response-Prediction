# GDSC2 data audit

Audited 2026-09-21. No records were removed or transformed. Reproduce with `& ".venv\Scripts\python.exe" "src\audit_gdsc2.py"` from the repository root; detailed aggregate results are in [data_audit_stats.json](data_audit_stats.json).

## Environment and retrieval

- Windows 11, Python 3.12.8 in project-local `.venv`; approximately 63.7 GiB RAM (26.8 GiB available at inventory), 49.1 GiB free disk at inventory and 32.6 GiB at final check. The new environment and raw files occupy about 214 and 111 MiB, respectively; the larger free-space change is outside this project. NVIDIA RTX 4070 Laptop GPU, 8,188 MiB VRAM; the audit ran on CPU. `pip check` reported no broken requirements. Direct and full environment specifications are in `configs/`. Python 3.12 was selected after checking [PyTorch's current Windows Python range](https://docs.pytorch.org/get-started/locally/) and the [published Windows RDKit wheel](https://pypi.org/project/rdkit/2023.9.6/). PyTorch was not installed for this audit.
- PyTDC 1.1.15 names GDSC2 in its `tdc/metadata.py` as Dataverse file **4165727** (`pkl`) and the gene-symbol table as **5255026** (`tab`). Its `DrugRes` import failed because `tdc.multi_pred.__init__` imports a single-cell module requiring `tiledbsoma`. Installing that unrelated stack was unnecessary. The retrieval script used the same Dataverse file IDs and URLs as PyTDC's `download_wrapper`; no replacement dataset was used.
- GDSC2: [Dataverse file 4165727](https://dataverse.harvard.edu/api/access/datafile/4165727), 116,571,213 bytes; SHA-256 `0d09b9d69cc714fd2b4a31e2b1f602c09ed82c282533683c5fc734ceb226f9e5`; MD5 `217ccb2c49dc43485924f8678eaf7e34`.
- Gene symbols: [Dataverse file 5255026](https://dataverse.harvard.edu/api/access/datafile/5255026), 149,384 bytes; SHA-256 `5712acc83f8be3b03345cbd56f5ae3289582ef08caf6991b92eca1728eb73296`; MD5 `a352ceb1a4586042324c43b357c90cf3`.
- Retrieval date: 2026-09-21 UTC. The Dataverse response gave `Last-Modified` 2026-02-15, which is a file-serving timestamp, **not** a verified GDSC or TDC dataset release date. No dataset version or upstream processing script was exposed by the TDC file or accessible metadata endpoint. The exact file checksum identifies the snapshot used here.
- [TDC lists CC BY-NC-ND 2.5 for GDSC2](https://tdcommons.ai/multi_pred_tasks/drugres/). [Sanger's data-use policy](https://depmap.sanger.ac.uk/documentation/data-usage-policy/) also governs upstream GDSC data. Keep raw and derived row-level data local; publish retrieval instructions and compact aggregate findings only. Confirm any broader redistribution rights before sharing data.

## Actual schema and alignment

| Field | Observed content |
| --- | --- |
| `ID1` | Drug name (string identifier), 137 unique |
| `ID2` | Cell-line name (string identifier), 805 unique |
| `X1` | Drug SMILES, 137 unique; one SMILES per drug name |
| `X2` | NumPy `float64` expression vector, length 17,737; 805 unique objects, one per cell line |
| `Y` | Numeric response (`float64`) |

There are **92,703 response rows** and **92,029 unique drug–cell pairs**. All 137 drug names map to one SMILES each; no SMILES or RDKit-canonical structure is shared across different drug names. All 805 cell-line names map to one expression object each, and no profile is shared between cell lines. Every response row has its drug structure and cell-line profile in the same record, so there are **no unmatched records within the TDC file**. Upstream Sanger model and drug IDs are absent, preventing an independent crosswalk to the original screening records.

The gene-symbol table has 17,737 rows, matching every expression-vector length. It contains **317 repeated gene-symbol entries**. Feature positions must remain distinct until the original probe/gene mapping is resolved; do not merge columns merely because their symbols match.

## Quality and distributions

- Missing values: zero in `ID1`, `ID2`, `X1`, `X2`, and `Y`. Nonfinite `Y`: zero. Nonfinite expression entries across the 805 unique vectors: zero. Invalid SMILES by RDKit: zero.
- Response `Y`: median **3.107**, interquartile range **1.393–4.573**, 1st–99th percentile **−5.539–7.161**, full range **−8.609–10.036**. These values describe the provided target and were not used to infer its transformation.
- Measurements per cell line: median **122**, range **13–137**. Measurements per drug: median **748**, range **43–1,474**. BI-2536 has the fewest rows (43); Uprosertib has the most (1,474). Unique cell lines per drug range from **43 to 804**.
- Exactly **674 drug–cell pairs repeat once**, creating 674 extra rows. **All are Uprosertib**, and all 674 pairs have different `Y` values; no exact duplicate drug–cell–response rows occur. The within-pair absolute response difference has median **0.697** and maximum **5.713**. The TDC file provides no separate screening ID, replicate label, or source drug ID to resolve these conflicts. [Sanger notes](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) that GDSC2 can have multiple entries for a compound because of internal tracking, but the cause of these particular Uprosertib pairs is not verified.
- The pickle references one expression vector per cell line across repeated response rows. Preserve the efficient representation in later processing as an 805 × 17,737 matrix plus a response table with cell-line keys, rather than copying every vector into every observation. Compute learned expression transforms on unique training cell-line profiles.

## Target meaning and limits of verification

**Verified in the loader:** PyTDC 1.1.15 `interaction_dataset_load` reads the pickle, selects `Y`, drops only null targets, and passes the values through to `DrugRes` without a label transform when no data configuration is supplied. The local file has no null targets, so its `Y` column is the intended dataset-provided response. No additional logarithm is warranted.

**Documented upstream:** [TDC calls GDSC2 `Y` “log normalized IC50”](https://tdcommons.ai/multi_pred_tasks/drugres/). [Sanger defines `LN_IC50` as the natural log of fitted IC50](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) and describes normalization of *viability readouts* to plate controls before curve fitting. Sanger lists drug-screen concentrations in micromolar; the [TDC dataset paper](https://arxiv.org/pdf/2102.09548) also lists µM for GDSC2. This supports interpreting larger `Y` as requiring a larger drug concentration for 50% inhibition, hence lower sensitivity in this assay.

**Still unresolved for this particular TDC snapshot:** The source file does not label `Y` as `LN_IC50`, identify the exact upstream release, or document how the curated rows were matched to expression and SMILES. A row-level crosscheck against a version-matched Sanger fitted-response file has not been established, so the precise convention for the logarithm's concentration reference (for example, `ln(IC50 / 1 µM)`) and any TDC-side normalization before serialization cannot be independently certified. Do not label `Y` as a per-drug z-score or convert it back to absolute µM without that confirmation. Report model errors in the **provided `Y` scale**.

## Split feasibility and proposed decisions

In 100 independent diagnostic 70/15/15 partitions of the 805 unique cell-line names (seed 20260921), all 137 drugs had at least one measured cell line in **each** partition. This is a feasibility check only; no model split was chosen or saved, and target values were not consulted. In Milestone 3, save one final grouped split, assert disjoint cell lines and training-drug coverage, and report any rare-drug validation/test counts.

Proposed cleaning policy for review before modeling: retain every original row in the raw file; keep the 317 duplicate gene-symbol positions distinct; retain Uprosertib's two responses with pair IDs and their discrepancy in the audit trail. Decide explicitly whether the primary modeling table will keep both observations with pair-aware weighting, aggregate each Uprosertib pair, or exclude that drug. Compare at least one sensitivity analysis so this choice does not silently determine the result. No exclusion or aggregation has been applied in this milestone.
