"""Small deterministic GDSC-shaped input for fresh-run integration checks.

These values are invented. They are never used as study results or a substitute
for TDC GDSC2, and the fixture is generated only in a temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


N_CELLS = 190
N_EXPRESSION_FEATURES = 160
PRIMARY_RESPONSE_ROWS = 349
UPROSERTIB_SOURCE_ROWS = 16


def write_synthetic_gdsc2(input_dir: Path) -> pd.DataFrame:
    """Write raw files with repeated profiles and conflicting Uprosertib Y.

    Three ordinary drugs have intentionally unequal cell coverage so response
    repetitions cannot accidentally stand in for unique-cell scaler fitting.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    expression = rng.normal(size=(N_CELLS, N_EXPRESSION_FEATURES))
    expression[:, -1] = 4.0  # one training-constant feature to be filtered
    cell_ids = [f"SYN_C{index:03d}" for index in range(N_CELLS)]
    structures = {
        "DrugA": "CCO",
        "DrugB": "CCN",
        "DrugC": "c1ccccc1O",
        "Uprosertib": "CC(=O)N",  # synthetic placeholder, not its real structure
    }
    offsets = {"DrugA": 1.0, "DrugB": 2.0, "DrugC": 3.0, "Uprosertib": 2.5}
    records: list[tuple[str, str, str, np.ndarray, float]] = []
    for index, cell_id in enumerate(cell_ids):
        profile = expression[index]
        drugs = ["DrugA"]
        if index % 2 == 0:
            drugs.append("DrugB")
        if index % 3 == 0:
            drugs.append("DrugC")
        if index < 12:
            drugs.append("Uprosertib")
        for drug in drugs:
            signal = 0.35 * profile[0] - 0.2 * profile[1] + 0.1 * profile[2]
            y = offsets[drug] + signal + float(rng.normal(scale=0.03))
            records.append((drug, cell_id, structures[drug], profile.copy(), y))
            if drug == "Uprosertib" and index < 4:
                records.append((drug, cell_id, structures[drug], profile.copy(), y + 0.8))
    raw = pd.DataFrame(records, columns=["ID1", "ID2", "X1", "X2", "Y"])
    if len(raw) != PRIMARY_RESPONSE_ROWS + UPROSERTIB_SOURCE_ROWS:
        raise AssertionError("Synthetic row recipe changed")
    symbols = [f"GENE_{position:03d}" for position in range(N_EXPRESSION_FEATURES)]
    symbols[11] = symbols[10]  # preserve repeated gene names by position
    pd.DataFrame({"gene_symbol": symbols}).to_csv(input_dir / "gdsc_gene_symbols.tab", sep="\t", index=False)
    raw.to_pickle(input_dir / "gdsc2.pkl")
    return raw
