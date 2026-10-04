import pytest
from src.Benchmark.prepare import guard_outputs

def test_guard_blocks_existing(tmp_path):
    p = tmp_path / "query.csv"
    p.write_text("x")
    with pytest.raises(FileExistsError):
        guard_outputs([p], force=False)
    guard_outputs([p], force=True)
