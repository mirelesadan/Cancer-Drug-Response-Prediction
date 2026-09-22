# Data access and provenance

The source is [TDC DrugRes / GDSC2](https://tdcommons.ai/multi_pred_tasks/drugres/). PyTDC 1.1.15 maps `gdsc2` to Harvard Dataverse file 4165727 and `gdsc_gene_symbols` to file 5255026. The current PyTDC package imports unrelated single-cell dependencies when accessing `DrugRes`, so `src/retrieve_gdsc2.py` retrieves these exact TDC-identified files directly from Dataverse. This changes the access path, not the selected dataset.

From the repository root, run `& ".venv\Scripts\python.exe" "src\retrieve_gdsc2.py"`. The script checks the known MD5 values and writes full SHA-256/MD5 checksums and retrieval metadata to ignored `data/raw/provenance.json`. It does not overwrite an existing file. Run `& ".venv\Scripts\python.exe" "src\audit_gdsc2.py"` to regenerate the compact audit statistics. Only load the pickle after verifying its source and checksum; pickle is not safe for untrusted input.

See [the audit report](../results/data_audit.md) for the source, terms, actual schema, findings, and remaining provenance questions. TDC lists [CC BY-NC-ND 2.5](https://tdcommons.ai/multi_pred_tasks/drugres/) for GDSC2, and [Sanger's upstream data-use policy](https://depmap.sanger.ac.uk/documentation/data-usage-policy/) applies. Downloaded data and derived row-level files are excluded from Git. This repository does not assert permission to redistribute them.
