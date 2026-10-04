import json

import pytest

from src.Benchmark.runspec import (
    Field, backend_slug, bench_results_root, open_run, render_slug,
    runtime_info, sanitize)


def test_sanitize_decimal_point_becomes_p():
    assert sanitize(0.01) == "0p01"
    assert sanitize("0.5") == "0p5"


def test_sanitize_other_punctuation():
    assert sanitize("opendata-split-11") == "opendata_split_11"
    assert sanitize("liftus:default") == "liftus_default"
    assert sanitize("a b/c") == "a_b_c"


def test_render_bare_and_prefixed():
    fields = (Field("task"), Field("k", prefix="k"))
    assert render_slug(fields, {"task": "union", "k": 10}) == "union-k10"


def test_render_omits_default_and_none():
    fields = (Field("repeat", prefix="r", default=1), Field("limit", prefix="lim"),
              Field("task"))
    assert render_slug(fields, {"repeat": 1, "limit": None, "task": "join"}) == "join"
    assert render_slug(fields, {"repeat": 5, "limit": 20, "task": "join"}) == "r5-lim20-join"


def test_render_list_joins_with_underscore_and_caps():
    fields = (Field("seekers"),)
    assert render_slug(fields, {"seekers": ["sc", "sho"]}) == "sc_sho"
    assert render_slug(fields, {"seekers": ["sc", "sho", "liftus"]}) == "sc_sho_liftus"
    assert render_slug(
        fields, {"seekers": ["sc", "sho", "liftus", "snoopy", "deepjoin"]}
    ) == "sc_sho_liftus+2"


def test_render_list_default_and_empty_drop_out():
    fields = (Field("efs", prefix="ef", default=[16, 32]), Field("variants"))
    assert render_slug(fields, {"efs": [16, 32], "variants": []}) == ""
    assert render_slug(fields, {"efs": [64], "variants": []}) == "ef64"


def test_render_float_list_stays_unambiguous():
    fields = (Field("selectivities", prefix="sel"),)
    assert render_slug(fields, {"selectivities": [0.01, 0.1]}) == "sel0p01_0p1"


def test_render_bool_is_a_flag():
    fields = (Field("pair_hits", prefix="ph"), Field("io"))
    assert render_slug(fields, {"pair_hits": True, "io": False}) == "ph"
    assert render_slug(fields, {"pair_hits": False, "io": True}) == "io"


def test_render_empty_field_set():
    assert render_slug((), {}) == ""


def _fields():
    return (Field("task"), Field("seeker"), Field("k", prefix="k"))


def _values():
    return {"task": "union", "seeker": "sho", "k": 10}


def test_open_run_path_shape(tmp_path):
    run = open_run("run", lake="toy", fields=_fields(), values=_values(),
                   config={"dbms": "duckdb"}, root=tmp_path)
    assert run.path.parent == bench_results_root("toy", tmp_path) / "run"
    assert run.path.name.endswith("-union-sho-k10")
    assert run.path.name.split("-")[0].endswith("Z")


def test_open_run_tag_sits_between_stamp_and_slug(tmp_path):
    run = open_run("run", lake="toy", fields=_fields(), values=_values(),
                   tag="ablation 3", root=tmp_path)
    assert run.path.name.endswith("-ablation_3-union-sho-k10")


def test_open_run_out_overrides_derived_path(tmp_path):
    dest = tmp_path / "somewhere" / "else"
    run = open_run("run", lake="toy", fields=_fields(), values=_values(),
                   out=dest, root=tmp_path)
    assert run.path == dest and dest.is_dir()


def test_open_run_without_lake_or_out_raises(tmp_path):
    with pytest.raises(ValueError):
        open_run("regime-recall", lake=None, fields=(), values={}, root=tmp_path)


def _manifest(run):
    return json.loads((run.path / "manifest.json").read_text())


def test_manifest_is_written_before_any_work(tmp_path):
    run = open_run("run", lake="toy", fields=_fields(), values=_values(),
                   config={"dbms": "duckdb"}, root=tmp_path)
    m = _manifest(run)
    assert m["status"] == "running"
    assert m["cmd"] == "run" and m["lake"] == "toy"
    assert m["args"] == {"task": "union", "seeker": "sho", "k": 10}


def test_manifest_ok_on_clean_exit(tmp_path):
    with open_run("run", lake="toy", fields=_fields(), values=_values(),
                  root=tmp_path) as run:
        pass
    m = _manifest(run)
    assert m["status"] == "ok" and m["error"] is None
    assert isinstance(m["wall_ms"], float)


def test_manifest_failed_records_the_error_and_reraises(tmp_path):
    with pytest.raises(RuntimeError):
        with open_run("run", lake="toy", fields=_fields(), values=_values(),
                      root=tmp_path) as run:
            raise RuntimeError("index missing")
    m = _manifest(run)
    assert m["status"] == "failed"
    assert "index missing" in m["error"]


def test_manifest_config_upgrades_via_set_config(tmp_path):
    with open_run("tiebreak-bench", lake="toy", fields=(), values={},
                  config={"dbms": "duckdb", "vector_backend": "faiss"},
                  root=tmp_path) as run:
        run.set_config(layout="two_table", profile="duckdb-faiss-two_table")
    cfg = _manifest(run)["config"]
    assert cfg["layout"] == "two_table"
    assert cfg["profile"] == "duckdb-faiss-two_table"


def test_write_table_emits_csv_and_json(tmp_path):
    rows = [{"query": "a.csv", "p@10": 0.5, "runtime_ms": 12.3}]
    summary = {"evaluated": 1, "mean_p@10": 0.5}
    with open_run("run", lake="toy", fields=_fields(), values=_values(),
                  root=tmp_path) as run:
        csv_path = run.write_table(rows, summary)
    assert csv_path.name == "results.csv"
    txt = csv_path.read_text()
    assert "query" in txt and "a.csv" in txt
    j = json.loads(run.file("results.json").read_text())
    assert j["summary"]["evaluated"] == 1
    assert j["args"]["seeker"] == "sho"


def test_write_table_handles_no_rows(tmp_path):
    with open_run("run", lake="toy", fields=(), values={}, root=tmp_path) as run:
        csv_path = run.write_table([], {"evaluated": 0})
    assert csv_path.read_text().strip() == "query"


class _StubDB:
    dbms = "postgres"
    vector_backend = "pgvector"
    layout = "two_table"

    def optimizer_profile_str(self):
        return "postgres-pgvector-two_table"


def test_runtime_info_prefers_the_open_handler(tmp_path):
    rt = runtime_info("toy", db=_StubDB())
    assert rt == {"dbms": "postgres", "vector_backend": "pgvector",
                  "layout": "two_table", "profile": "postgres-pgvector-two_table"}
    assert backend_slug(rt) == "postgres_pgvector"


def test_runtime_info_without_a_handler_reads_the_active_config(monkeypatch, tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text("[Dataset]\nname = toy\n\n[Database]\ndbms = duckdb\n\n"
                   "[Semantic]\nvector_backend = faiss\n")
    monkeypatch.setenv("BLEND_CONFIG", str(ini))
    rt = runtime_info("toy")
    assert rt["dbms"] == "duckdb" and rt["vector_backend"] == "faiss"
    assert rt["layout"] is None and rt["profile"] is None
    assert backend_slug(rt) == "duckdb_faiss"


def test_slug_reflects_overrides_not_the_ini(monkeypatch, tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text("[Dataset]\nname = toy\n\n[Database]\ndbms = duckdb\n\n"
                   "[Semantic]\nvector_backend = faiss\nfaiss_k_coarse = 500\n")
    monkeypatch.setenv("BLEND_CONFIG", str(ini))
    effective_k_coarse = 1000
    fields = (Field("k_coarse", prefix="kc", default=500), Field("backend"))
    values = {"k_coarse": effective_k_coarse,
              "backend": backend_slug(runtime_info("toy"))}
    assert render_slug(fields, values) == "kc1000-duckdb_faiss"
