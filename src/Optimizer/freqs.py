from pathlib import Path

import pandas as pd


def build_freqs_dict(db, index_table="blend_index", *, out_path) -> Path:
    """Write the lake's token frequencies (tokenized -> count) to a CSV."""
    out_path = Path(out_path)
    print(f"computing token frequencies over {index_table} ...", flush=True)
    rows = db.execute_and_fetchall(
        "SELECT tokenized, COUNT(*) AS frequency FROM AllTables GROUP BY tokenized"
    )
    df = pd.DataFrame(rows, columns=["tokenized", "frequency"])
    df.to_csv(out_path, index=False)
    return out_path
