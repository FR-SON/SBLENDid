def compute_features(columns, db):
    rows = [tuple(row) for row in zip(*columns)]
    freqs = db.get_token_frequencies(set().union(*columns))
    prod = 1
    for col in columns:
        prod *= sum(freqs[t] for t in set(db.clean_value_collection(col)) if t in freqs)
    return [len(set(rows)), prod ** (1 / len(columns)), len(columns)]
