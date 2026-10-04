import csv
import subprocess
from pathlib import Path

from scripts.build_overlap_registry import (
    norm_value, norm_name, resolve_col, is_numeric_column, pair_stats, union_best,
    norm_distinct, make_loader, discover_gt,
    FIELDS, score_row, process_file, GtFile,
)


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _loader():
    return make_loader(0, 0, 128)


def test_norm_value_strips_quotes_casefolds_and_nulls():
    assert norm_value("'Hello'") == "hello"
    assert norm_value('"WORLD"') == "world"
    assert norm_value("  NaN ") == ""
    assert norm_value("none") == ""
    assert norm_value("") == ""


def test_norm_name_strips_bom_and_quotes():
    assert norm_name("﻿Id") == "id"
    assert norm_name('"Country"') == "country"


def test_resolve_col_exact_then_normalized():
    cols = ["﻿Id", "Country"]
    assert resolve_col(cols, "﻿Id") == "﻿Id"
    assert resolve_col(cols, "id") == "﻿Id"
    assert resolve_col(cols, "missing") is None


def test_is_numeric_column():
    assert is_numeric_column(["12", "34", "560"]) is True
    assert is_numeric_column(["ab", "cd", "ef"]) is False
    assert is_numeric_column([]) is False


def test_pair_stats_math():
    nq, nc, ninter, cq, cx, jac = pair_stats(["a", "b", "c"], ["b", "c", "d"])
    assert (nq, nc, ninter) == (3, 3, 2)
    assert cq == 2 / 3 and cx == 2 / 3
    assert jac == 2 / 4


def test_pair_stats_empty():
    assert pair_stats([], ["a"]) == (0, 1, 0, 0.0, 0.0, 0.0)


def test_union_best_picks_symmetric_max():
    q = {"qa": ["x", "y"], "qb": ["1", "2", "3", "4"]}
    c = {"ca": ["x", "y", "z", "w"], "cb": ["9"]}
    qcol, ccol, st = union_best(q, c)
    assert (qcol, ccol) == ("qa", "ca")
    assert max(st[3], st[4]) == 1.0


def test_norm_distinct_dedups_and_normalizes(tmp_path):
    p = tmp_path / "t.csv"
    _write_csv(p, ["a", "b"], [["X", "1"], ["x", "1"], ["", "2"]])
    cols = norm_distinct(p, row_cap=0, max_distinct=0)
    assert cols["a"] == ["x"]
    assert cols["b"] == ["1", "2"]


def test_norm_distinct_missing_file_returns_empty(tmp_path):
    assert norm_distinct(tmp_path / "nope.csv", 0, 0) == {}


def test_norm_distinct_max_distinct_caps(tmp_path):
    p = tmp_path / "t.csv"
    _write_csv(p, ["a"], [["1"], ["2"], ["3"]])
    assert norm_distinct(p, row_cap=0, max_distinct=2)["a"] == ["1", "2"]


def test_discover_gt_classifies_and_ignores_overlap(tmp_path):
    ds = tmp_path / "ds1"
    gt = ds / "groundtruth"
    _write_csv(gt / "j_ground_truth.csv",
               ["query_table", "candidate_table", "query_column", "candidate_column"], [])
    _write_csv(gt / "u_ground_truth.csv", ["query_table", "candidate_table"], [])
    _write_csv(gt / "j_ground_truth.overlap.csv", ["dataset", "gt_file"], [])
    found = {g.path.name: g.kind for g in discover_gt(tmp_path)}
    assert found == {"j_ground_truth.csv": "join", "u_ground_truth.csv": "union"}


def test_score_row_join_ok(tmp_path):
    csvs = tmp_path / "csvs"
    _write_csv(csvs / "q.csv", ["k"], [["a"], ["b"], ["c"]])
    _write_csv(csvs / "c.csv", ["k"], [["b"], ["c"], ["d"]])
    row = {"query_table": "q.csv", "candidate_table": "c.csv",
           "query_column": "k", "candidate_column": "k"}
    rec = score_row("join", "ds", "gt.csv", 0, row, csvs, _loader())
    assert rec["status"] == "ok"
    assert rec["n_inter"] == 2 and rec["n_query"] == 3
    assert rec["cont_q2x"] == "0.666667"
    assert rec["numeric_q"] == "false"


def test_score_row_missing_table(tmp_path):
    csvs = tmp_path / "csvs"
    _write_csv(csvs / "q.csv", ["k"], [["a"]])
    row = {"query_table": "q.csv", "candidate_table": "gone.csv",
           "query_column": "k", "candidate_column": "k"}
    rec = score_row("join", "ds", "gt.csv", 0, row, csvs, _loader())
    assert rec["status"] == "missing_table"
    assert rec["n_inter"] == ""


def test_score_row_missing_column(tmp_path):
    csvs = tmp_path / "csvs"
    _write_csv(csvs / "q.csv", ["k"], [["a"]])
    _write_csv(csvs / "c.csv", ["k"], [["a"]])
    row = {"query_table": "q.csv", "candidate_table": "c.csv",
           "query_column": "nope", "candidate_column": "k"}
    rec = score_row("join", "ds", "gt.csv", 0, row, csvs, _loader())
    assert rec["status"] == "missing_column"


def test_score_row_union_argmax(tmp_path):
    csvs = tmp_path / "csvs"
    _write_csv(csvs / "q.csv", ["qa", "qb"], [["x", "1"], ["y", "2"]])
    _write_csv(csvs / "c.csv", ["ca", "cb"], [["x", "9"], ["y", "8"]])
    row = {"query_table": "q.csv", "candidate_table": "c.csv"}
    rec = score_row("union", "ds", "gt.csv", 0, row, csvs, _loader())
    assert rec["status"] == "ok"
    assert (rec["query_column"], rec["candidate_column"]) == ("qa", "ca")
    assert rec["n_inter"] == 2


def test_process_file_writes_sibling(tmp_path):
    ds = tmp_path / "ds1"
    csvs = ds / "csvs"
    gt_dir = ds / "groundtruth"
    _write_csv(csvs / "q.csv", ["k"], [["a"], ["b"]])
    _write_csv(csvs / "c.csv", ["k"], [["a"], ["z"]])
    _write_csv(gt_dir / "g_ground_truth.csv",
               ["query_table", "candidate_table", "query_column", "candidate_column"],
               [["q.csv", "c.csv", "k", "k"]])
    gt = GtFile(gt_dir / "g_ground_truth.csv", "ds1", "join", csvs)
    out = process_file(gt, _loader())
    assert out.name == "g_ground_truth.overlap.csv"
    written = list(csv.DictReader(out.open(newline="")))
    assert list(written[0].keys()) == FIELDS
    assert written[0]["n_inter"] == "1"


def test_cli_end_to_end(tmp_path):
    ds = tmp_path / "ds1"
    csvs = ds / "csvs"
    gt_dir = ds / "groundtruth"
    _write_csv(csvs / "q.csv", ["k"], [["a"], ["b"], ["c"]])
    _write_csv(csvs / "c.csv", ["k"], [["b"], ["c"], ["d"]])
    _write_csv(gt_dir / "g_ground_truth.csv",
               ["query_table", "candidate_table", "query_column", "candidate_column"],
               [["q.csv", "c.csv", "k", "k"]])
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_overlap_registry.py"
    r = subprocess.run(
        ["uv", "run", "python", str(script), "--datasets-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    out = gt_dir / "g_ground_truth.overlap.csv"
    assert out.exists()
    rec = next(csv.DictReader(out.open(newline="")))
    assert rec["n_inter"] == "2" and rec["status"] == "ok"
