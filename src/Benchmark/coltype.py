from __future__ import annotations

import numpy as np
import pandas as pd


def is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def infer_col(cells: list) -> dict:
    cells = [str(c).strip() for c in cells if str(c).strip() and str(c).lower() != "nan"]
    n = len(cells)
    if n == 0:
        return {"numeric_fraction": 0.0, "cardinality_ratio": 0.0,
                "avg_char_len": 0.0, "col_type": "empty", "n_sampled": 0}
    n_num = sum(1 for c in cells if is_float(c))
    num_frac = n_num / n
    card_ratio = len(set(cells)) / n
    char_len = float(np.mean([len(c) for c in cells]))
    if num_frac >= 0.8 and card_ratio >= 0.5:
        ct = "id"
    elif num_frac >= 0.8:
        ct = "numeric"
    elif card_ratio < 0.2 and char_len < 30:
        ct = "categorical"
    else:
        ct = "text"
    return {"numeric_fraction": num_frac, "cardinality_ratio": card_ratio,
            "avg_char_len": char_len, "col_type": ct, "n_sampled": n}


def classify_columns(df: pd.DataFrame) -> dict[str, str]:
    return {c: infer_col(df[c].dropna().tolist())["col_type"] for c in df.columns}
