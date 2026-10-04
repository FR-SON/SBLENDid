def overlay(*, dropped_ids, added_ids, relevant, i2b) -> dict:
    """Count optimizer drops/additions that are in GT (symmetric lower bound)."""
    drop_b = {i2b[i] for i in dropped_ids if i in i2b}
    add_b = {i2b[i] for i in added_ids if i in i2b}
    return {"real_losses": len(drop_b & relevant), "real_gains": len(add_b & relevant)}
