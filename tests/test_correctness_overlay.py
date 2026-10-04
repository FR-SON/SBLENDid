from src.Benchmark.correctness.overlay import overlay


def test_overlay_counts_gt_members():
    i2b = {1: "a.csv", 2: "b.csv", 3: "c.csv", 4: "d.csv"}
    relevant = {"a.csv", "c.csv"}
    res = overlay(dropped_ids=[1, 4], added_ids=[3], relevant=relevant, i2b=i2b)
    assert res == {"real_losses": 1, "real_gains": 1}
