import pandas as pd
import pytest


from src.Optimizer import semantic_cost_fit as fit


def _hnsw_csv(path, ef=64, kc=500):
    pd.DataFrame([
        {"seeker": "SU", "efSearch": ef, "k_coarse": kc, "table_id": "t",
         "col_name": "", "n_cols": 2, "faiss_ms": float("nan"),
         "total_ms": 8.0, "overhead_ms": float("nan")},
        {"seeker": "SU", "efSearch": ef, "k_coarse": kc, "table_id": "u",
         "col_name": "", "n_cols": 4, "faiss_ms": float("nan"),
         "total_ms": 16.0, "overhead_ms": float("nan")},
    ]).to_csv(path, index=False)


def _exact_csv(path):
    pd.DataFrame([
        {"seeker": "SU", "table_id": "t", "col_name": "", "n_query_cols": 1,
         "m_tables": 1, "n_gids": 1000, "exact_ms": 2.0},
        {"seeker": "SU", "table_id": "u", "col_name": "", "n_query_cols": 1,
         "m_tables": 1, "n_gids": 5000, "exact_ms": 6.0},
    ]).to_csv(path, index=False)


def _filtered_csv(path):
    pd.DataFrame([
        {"seeker": "SU", "table_id": "t", "col_name": "", "n_query_cols": 1,
         "m_tables": 1, "n_gids": 1000, "exact_ms": 3.0},
        {"seeker": "SU", "table_id": "u", "col_name": "", "n_query_cols": 1,
         "m_tables": 1, "n_gids": 5000, "exact_ms": 7.0},
    ]).to_csv(path, index=False)


def test_hnsw_per_col_ms_is_median_of_total_over_ncols(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_csv(p)
    assert fit.hnsw_per_col_ms(p, 64, 500) == 4.0


def test_fit_exact_curve_recovers_beta_alpha(tmp_path):
    p = tmp_path / "exact_curve_SU.csv"
    _exact_csv(p)
    beta, alpha = fit.fit_exact_curve(p)
    assert round(beta, 6) == 1.0
    assert round(alpha, 6) == 0.001


def test_derive_flip_analytic():
    assert fit.derive_flip(4.0, 1.0, 0.001, n_vectors=10000) == 3000
    assert fit.derive_flip(4.0, 1.0, 0.001, n_vectors=1000) is None
    assert fit.derive_flip(4.0, 1.0, 0.0, n_vectors=10000) is None
    assert fit.derive_flip(2.0, 5.0, 0.001, n_vectors=10000) is None


def _lake_meta():
    return {"SU": {"n_vectors": 6322, "avg_cols_per_table": 10.8}}


def test_build_model_assembles_op_entry(tmp_path):
    _hnsw_csv(tmp_path / "hnsw_surface_SU.csv")
    _filtered_csv(tmp_path / "filtered_curve_SU.csv")
    model = fit.build_model(
        tmp_path, ops=["SU"], ef=64, kc=500,
        lake_meta=_lake_meta(), backend="pgvector",
    )
    su = model["SU"]
    assert su["hnsw_per_col_ms"] == 4.0
    assert round(su["exact_beta_ms"], 6) == 2.0
    assert round(su["exact_alpha_ms_per_col"], 6) == 0.001
    assert su["n_vectors"] == 6322 and su["backend"] == "pgvector"
    assert su["flip_a_cols"] == 2000


def test_build_model_faiss_stores_no_flip(tmp_path):
    _hnsw_csv(tmp_path / "hnsw_surface_SU.csv")
    _exact_csv(tmp_path / "exact_curve_SU.csv")
    model = fit.build_model(
        tmp_path, ops=["SU"], ef=64, kc=500,
        lake_meta=_lake_meta(), backend="faiss",
    )
    assert model["SU"]["flip_a_cols"] is None
    assert round(model["SU"]["exact_beta_ms"], 6) == 1.0


def test_pgvector_fit_uses_filtered_over_stale_exact(tmp_path):
    _hnsw_csv(tmp_path / "hnsw_surface_SU.csv")
    _exact_csv(tmp_path / "exact_curve_SU.csv")
    _filtered_csv(tmp_path / "filtered_curve_SU.csv")
    model = fit.build_model(tmp_path, ops=["SU"], ef=64, kc=500,
                            lake_meta=_lake_meta(), backend="pgvector")
    assert round(model["SU"]["exact_beta_ms"], 6) == 2.0


def test_faiss_fit_uses_exact_over_stale_filtered(tmp_path):
    _hnsw_csv(tmp_path / "hnsw_surface_SU.csv")
    _exact_csv(tmp_path / "exact_curve_SU.csv")
    _filtered_csv(tmp_path / "filtered_curve_SU.csv")
    model = fit.build_model(tmp_path, ops=["SU"], ef=64, kc=500,
                            lake_meta=_lake_meta(), backend="faiss")
    assert round(model["SU"]["exact_beta_ms"], 6) == 1.0


def _hnsw_grid_csv(path, op, *, a, b, c, d, ncols, kcs, ef=64):
    rows = []
    for n in ncols:
        for kc in kcs:
            rows.append({"seeker": op, "efSearch": ef, "k_coarse": kc,
                         "table_id": f"t{n}", "col_name": "", "n_cols": n,
                         "faiss_ms": float("nan"),
                         "total_ms": a + b * n + c * kc + d * n * kc,
                         "overhead_ms": float("nan")})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_fit_unfiltered_su_recovers_bilinear(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_grid_csv(p, "SU", a=1.0, b=0.5, c=0.002, d=0.0001,
                   ncols=[1, 2, 4, 8], kcs=[100, 500, 1000])
    u = fit.fit_unfiltered(p, "SU", ef=64)
    assert u["form"] == "su_bilinear"
    assert round(u["coef"]["a"], 3) == 1.0
    assert round(u["coef"]["b_ncols"], 3) == 0.5
    assert round(u["coef"]["c_kc"], 4) == 0.002
    assert round(u["coef"]["d"], 5) == 0.0001
    assert u["k_coarse_grid"] == [100, 500, 1000]
    assert u["ef_search"] == 64
    assert u["r2"] == 1.0 and u["mae"] == 0.0
    assert u["n_train"] + u["n_test"] == 12 and u["n_test"] >= 1


def test_fit_unfiltered_sj_is_linear_in_kc(tmp_path):
    p = tmp_path / "hnsw_surface_SJ.csv"
    rows = []
    for rep in range(2):
        for kc in [100, 500, 1000]:
            rows.append({"seeker": "SJ", "efSearch": 64, "k_coarse": kc,
                         "table_id": f"c{rep}", "col_name": "x", "n_cols": 1,
                         "faiss_ms": float("nan"), "total_ms": 2.0 + 0.003 * kc,
                         "overhead_ms": float("nan")})
    pd.DataFrame(rows).to_csv(p, index=False)
    u = fit.fit_unfiltered(p, "SJ", ef=64)
    assert u["form"] == "sj_linear_kc"
    assert round(u["coef"]["a"], 3) == 2.0
    assert round(u["coef"]["c_kc"], 4) == 0.003
    assert "b_ncols" not in u["coef"] and "d" not in u["coef"]


def test_fit_unfiltered_too_few_rows_skips_validation(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_csv(p)
    u = fit.fit_unfiltered(p, "SU", ef=64)
    assert u["n_test"] == 0 and u["r2"] is None and u["mae"] is None
    assert set(u["coef"]) == {"a", "b_ncols", "c_kc", "d"}


def test_dominant_plan_label_picks_mode_at_ef(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    pd.DataFrame([
        {"seeker": "SU", "efSearch": 64, "k_coarse": 500, "n_cols": 2,
         "total_ms": 8.0, "plan_label": "seqscan"},
        {"seeker": "SU", "efSearch": 64, "k_coarse": 800, "n_cols": 2,
         "total_ms": 9.0, "plan_label": "seqscan"},
        {"seeker": "SU", "efSearch": 64, "k_coarse": 1000, "n_cols": 2,
         "total_ms": 9.0, "plan_label": "hnsw"},
        {"seeker": "SU", "efSearch": 16, "k_coarse": 500, "n_cols": 2,
         "total_ms": 8.0, "plan_label": "hnsw"},
    ]).to_csv(p, index=False)
    assert fit.dominant_plan_label(p, 64) == "seqscan"


def test_dominant_plan_label_absent_column_is_none(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_csv(p)
    assert fit.dominant_plan_label(p, 64) is None


def test_format_fit_summary_reports_validation_per_op(tmp_path):
    model = {
        "SU": {"k_coarse": 500, "ef_search": 64, "plan_label": "seqscan",
               "unfiltered": {"form": "su_bilinear", "r2": 0.987, "mae": 1.23,
                              "n_train": 600, "n_test": 200,
                              "k_coarse_grid": [50, 100, 500]}},
        "SJ": {"k_coarse": 500, "ef_search": 64, "plan_label": None,
               "unfiltered": {"form": "sj_linear_kc", "r2": None, "mae": None,
                              "n_train": 6, "n_test": 0, "k_coarse_grid": [500]}},
    }
    s = fit.format_fit_summary(model)
    assert "SU" in s and "0.987" in s and "1.23" in s and "seqscan" in s
    assert "SJ" in s and "n/a" in s


def test_build_model_includes_unfiltered_and_plan_label(tmp_path):
    _hnsw_csv(tmp_path / "hnsw_surface_SU.csv")
    _filtered_csv(tmp_path / "filtered_curve_SU.csv")
    model = fit.build_model(tmp_path, ops=["SU"], ef=64, kc=500,
                            lake_meta=_lake_meta(), backend="pgvector")
    su = model["SU"]
    assert su["unfiltered"]["form"] == "su_bilinear"
    assert "plan_label" in su and su["plan_label"] is None
    assert su["hnsw_per_col_ms"] == 4.0


def _hnsw_diag_csv(path, op, *, a, b, c, d, ncols, kcs, off_diagonal_ms=999.0):
    rows = []
    for ef in kcs:
        for kc in kcs:
            for n in ncols:
                on_diag = ef == kc
                rows.append({"seeker": op, "efSearch": ef, "k_coarse": kc,
                             "table_id": f"t{n}", "col_name": "", "n_cols": n,
                             "faiss_ms": float("nan"),
                             "total_ms": (a + b * n + c * kc + d * n * kc
                                          if on_diag else off_diagonal_ms),
                             "overhead_ms": float("nan"),
                             "plan_label": "hnsw" if on_diag else "seqscan"})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_fit_unfiltered_diagonal_reads_only_ef_equals_kc(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_diag_csv(p, "SU", a=1.0, b=0.5, c=0.002, d=0.0001,
                   ncols=[1, 2, 4, 8], kcs=[30, 60, 110, 210])
    u = fit.fit_unfiltered(p, "SU", ef=30, diagonal=True)
    assert u["depth_rule"] == "diagonal"
    assert u["k_coarse_grid"] == [30, 60, 110, 210]
    assert round(u["coef"]["a"], 3) == 1.0
    assert round(u["coef"]["b_ncols"], 3) == 0.5
    assert round(u["coef"]["c_kc"], 4) == 0.002
    assert round(u["coef"]["d"], 5) == 0.0001
    assert u["r2"] == 1.0 and u["mae"] == 0.0


def test_fit_unfiltered_fixed_ef_sees_one_kc_on_a_diagonal_sweep(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_diag_csv(p, "SU", a=1.0, b=0.5, c=0.002, d=0.0001,
                   ncols=[1, 2, 4, 8], kcs=[30, 60, 110, 210])
    diag = pd.read_csv(p)
    diag = diag[diag.efSearch == diag.k_coarse]
    diag.to_csv(p, index=False)
    u = fit.fit_unfiltered(p, "SU", ef=30, diagonal=False)
    assert u["depth_rule"] == "fixed_ef"
    assert u["k_coarse_grid"] == [30]


def test_fit_unfiltered_diagonal_needs_two_depths(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_grid_csv(p, "SU", a=1.0, b=0.5, c=0.002, d=0.0001,
                   ncols=[1, 2], kcs=[64], ef=64)
    with pytest.raises(ValueError, match="diagonal fit needs >=2 distinct k_coarse"):
        fit.fit_unfiltered(p, "SU", ef=64, diagonal=True)


def test_fit_unfiltered_diagonal_empty_names_the_diagonal(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_grid_csv(p, "SU", a=1.0, b=0.5, c=0.002, d=0.0001,
                   ncols=[1, 2], kcs=[100, 500], ef=64)
    with pytest.raises(ValueError, match="on the diagonal"):
        fit.fit_unfiltered(p, "SU", ef=64, diagonal=True)


def test_dominant_plan_label_follows_the_row_selection(tmp_path):
    p = tmp_path / "hnsw_surface_SU.csv"
    _hnsw_diag_csv(p, "SU", a=1.0, b=0.5, c=0.002, d=0.0001,
                   ncols=[1, 2], kcs=[30, 60, 110])
    assert fit.dominant_plan_label(p, 30, diagonal=True) == "hnsw"
    assert fit.dominant_plan_label(p, 30, diagonal=False) == "seqscan"


def test_build_model_threads_diagonal(tmp_path):
    _hnsw_diag_csv(tmp_path / "hnsw_surface_SU.csv", "SU",
                   a=1.0, b=0.5, c=0.002, d=0.0001,
                   ncols=[1, 2, 4], kcs=[30, 60, 110])
    _exact_csv(tmp_path / "exact_curve_SU.csv")
    model = fit.build_model(tmp_path, ops=["SU"], ef=30, kc=30,
                            lake_meta=_lake_meta(), backend="faiss", diagonal=True)
    su = model["SU"]
    assert su["unfiltered"]["depth_rule"] == "diagonal"
    assert su["unfiltered"]["k_coarse_grid"] == [30, 60, 110]
    assert su["ef_search"] == 30 and su["k_coarse"] == 30
    assert su["plan_label"] == "hnsw"


def test_format_fit_summary_shows_the_depth_rule(tmp_path):
    model = {"SU": {"k_coarse": 30, "ef_search": 30, "plan_label": "hnsw",
                    "unfiltered": {"form": "su_bilinear", "r2": 0.99, "mae": 0.5,
                                   "n_train": 9, "n_test": 3, "depth_rule": "diagonal",
                                   "k_coarse_grid": [30, 60, 110]}}}
    s = fit.format_fit_summary(model)
    assert "rule=diagonal" in s and "ef=kc" in s


def _surface(path, *, repeats, noise, seed=0):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(repeats):
        for kc in (30, 60, 110, 210):
            for i, n in enumerate((4, 8, 16, 32)):
                for q in range(12):
                    mu = 0.5 + 0.02 * n + 0.001 * kc + 0.0004 * n * kc
                    ms = mu * (1 + rng.normal(0, noise)) if noise else mu
                    rows.append({"seeker": "SU", "repeat": rep, "efSearch": 64,
                                 "k_coarse": kc, "table_id": f"t{i}_{q}",
                                 "col_name": "", "n_cols": n,
                                 "faiss_ms": 0.0, "total_ms": max(ms, 1e-6),
                                 "overhead_ms": 0.0, "plan_label": "hnsw"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_ceiling_is_none_without_repeats(tmp_path):
    from src.Optimizer.semantic_cost_fit import fit_unfiltered
    m = fit_unfiltered(_surface(tmp_path / "h.csv", repeats=1, noise=0.2), "SU", 64)
    assert m["ceiling_r2"] is None and m["r2_of_ceiling"] is None
    assert m["n_replicated_points"] == 0


def test_ceiling_tracks_injected_noise(tmp_path):
    from src.Optimizer.semantic_cost_fit import fit_unfiltered
    lo = fit_unfiltered(_surface(tmp_path / "lo.csv", repeats=3, noise=0.05), "SU", 64)
    hi = fit_unfiltered(_surface(tmp_path / "hi.csv", repeats=3, noise=0.35), "SU", 64)
    assert lo["ceiling_r2"] > hi["ceiling_r2"]
    assert hi["n_replicated_points"] == 192
    assert 0.75 < hi["r2_of_ceiling"] < 1.15


def test_noise_free_data_still_recovered_exactly(tmp_path):
    from src.Optimizer.semantic_cost_fit import fit_unfiltered
    m = fit_unfiltered(_surface(tmp_path / "c.csv", repeats=2, noise=0.0), "SU", 64)
    assert m["r2"] == 1.0 and m["ceiling_r2"] == 1.0
    assert abs(m["coef"]["b_ncols"] - 0.02) < 1e-6


def test_mdape_survives_a_near_zero_target(tmp_path):
    import pandas as pd
    from src.Optimizer.semantic_cost_fit import fit_unfiltered
    path = _surface(tmp_path / "z.csv", repeats=3, noise=0.05)
    d = pd.read_csv(path)
    d.loc[0, "total_ms"] = 1e-9
    d.to_csv(path, index=False)
    m = fit_unfiltered(path, "SU", 64)
    assert m["mdape"] < 1.0
