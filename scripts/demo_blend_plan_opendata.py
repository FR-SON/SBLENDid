"""Demo: chained Blend Plans mixing SU + SJ + token-based seekers on opendata-split-11."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

# torch + faiss-cpu libomp conflict on macOS; must precede any torch/faiss import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

_T = TypeVar("_T")

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

import duckdb

from src.Operators import Seekers, Combiners
from src.Plan import Plan
from src.Semantic.config import SemanticConfig
from src.index_routing import physical_table


def _demo_layout(con) -> str:
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    return "two_table" if {"blend_index_token", "blend_index_tableid"} <= names else "single"

_K = 20
_K_CHEAP_HNSW = 200
_PRINT_TOP = 10
_SQL_EXCERPT = 400


def _norm(v) -> str:
    """Mirror df_to_index's per-cell normalisation so seeker inputs match the lake."""
    s = (str(v).lower()
         .replace("\\", "").replace("'", "").replace('"', "")
         .replace("\t", "").replace("\n", "").replace("\r", "")
         .strip()[:200])
    return "" if s in ("nan", "none") else s


def _norm_drop_empty(values) -> list[str]:
    out = [_norm(v) for v in values]
    return [v for v in out if v]


def _timed(fn: Callable[[], _T]) -> tuple[_T, float]:
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000.0


def _print_row(name: str, ids: list[int], ms: float | None = None) -> None:
    body = ", ".join(str(i) for i in ids[:_PRINT_TOP]) if ids else "(empty)"
    suffix = "" if len(ids) <= _PRINT_TOP else f"  …(+{len(ids) - _PRINT_TOP})"
    ms_str = f"  [{ms:7.1f}ms]" if ms is not None else ""
    print(f"  {name:32s}{ms_str} -> {body}{suffix}")


def _ids_to_pushdown(ids) -> str:
    """Build Intersection's `additionals` fragment for a set of ints; empty input yields ''."""
    ids = list(ids)
    if not ids:
        return ""
    return f" AND TableId IN ({','.join(str(i) for i in ids)}) "


def _load_basename_map(cfg: SemanticConfig) -> tuple[set[str], dict[str, int], dict[int, str]]:
    df = pd.read_parquet(cfg.blend_basenames_path)
    bn_to_int = dict(zip(df["basename"].astype(str), df["table_int_id"].astype(int)))
    int_to_bn = {v: k for k, v in bn_to_int.items()}
    return set(bn_to_int), bn_to_int, int_to_bn


def _pick_query_tables(cfg: SemanticConfig) -> tuple[list[dict], set[str], dict[str, int], dict[int, str]]:
    in_split, bn_to_int, int_to_bn = _load_basename_map(cfg)
    base = cfg.dataset.dir()
    uq = pd.read_csv(base / "query" / "opendata_union_query.csv")
    jq = pd.read_csv(base / "query" / "opendata_join_query.csv")
    ugt = pd.read_csv(base / "groundtruth" / "opendata_union_ground_truth.csv")
    jgt = pd.read_csv(base / "groundtruth" / "opendata_join_ground_truth.csv")

    ugt_in = ugt[ugt["query_table"].isin(in_split) & ugt["candidate_table"].isin(in_split)]
    jgt_in = jgt[jgt["query_table"].isin(in_split) & jgt["candidate_table"].isin(in_split)]
    uq_tables = set(uq[uq["query_table"].isin(in_split)]["query_table"])
    jq_tables = set(jq[jq["query_table"].isin(in_split)]["query_table"])
    both = uq_tables & jq_tables

    ranked = []
    for t in both:
        u_set = set(ugt_in.loc[ugt_in["query_table"] == t, "candidate_table"])
        j_rows = jgt_in[jgt_in["query_table"] == t]
        if not u_set or j_rows.empty:
            continue
        col_best = j_rows.groupby("query_column")["candidate_table"].nunique().idxmax()
        j_set = set(j_rows.loc[j_rows["query_column"] == col_best, "candidate_table"])
        ranked.append((t, col_best, u_set, j_set))
    if not ranked:
        raise SystemExit(
            f"no candidate query table in {cfg.dataset.name}: "
            "union_query ∩ join_query ∩ in-split with non-empty GT on both is empty"
        )
    ranked.sort(key=lambda r: len(r[2]) + len(r[3]), reverse=True)
    picks = [
        {
            "T": t,
            "q_sj_col": col,
            "union_gt": u_set,
            "join_gt": j_set,
            "union_gt_count": len(u_set),
            "join_gt_count": len(j_set),
            "in_split": in_split,
            "bn_to_int": bn_to_int,
            "int_to_bn": int_to_bn,
        }
        for (t, col, u_set, j_set) in ranked
    ]
    return picks, in_split, bn_to_int, int_to_bn


def _load_query_df(cfg: SemanticConfig, basename: str) -> pd.DataFrame:
    path = cfg.dataset.dir() / "csvs" / basename
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.attrs["table_id"] = basename
    return df


def _string_density(series: pd.Series) -> float:
    s = series.astype(str)
    non_empty = s.str.strip().ne("")
    not_numeric = pd.to_numeric(s, errors="coerce").isna()
    return float((non_empty & not_numeric).mean())


def _pick_columns(df: pd.DataFrame) -> dict:
    n = max(len(df), 1)
    distincts = {c: df[c].nunique() for c in df.columns}
    numeric_ratio = {
        c: float(pd.to_numeric(df[c], errors="coerce").notna().mean())
        for c in df.columns
    }
    density = {c: _string_density(df[c]) for c in df.columns}
    non_id = [c for c in df.columns if distincts[c] / n <= 0.5]
    if not non_id:
        non_id = list(df.columns)

    sc_col = max(non_id, key=lambda c: distincts[c])
    c_src_candidates = [c for c in df.columns if distincts[c] >= 4]
    if not c_src_candidates:
        c_src_candidates = [c for c in df.columns if distincts[c] >= 2]
    c_src_col = min(c_src_candidates, key=lambda c: distincts[c]) if c_src_candidates else df.columns[0]
    tgt_candidates = {c: r for c, r in numeric_ratio.items() if c != c_src_col}
    c_tgt_col = max(tgt_candidates, key=tgt_candidates.get) if tgt_candidates else c_src_col

    mc_cols = sorted(non_id, key=lambda c: density[c], reverse=True)[:3]
    return {
        "sc_col": sc_col,
        "c_src_col": c_src_col,
        "c_tgt_col": c_tgt_col,
        "mc_cols": mc_cols,
        "distincts": distincts,
        "numeric_ratio": numeric_ratio,
        "density": density,
    }


def _pick_kw_broad_tokens(
    cfg: SemanticConfig,
    df: pd.DataFrame,
    sc_col: str,
    max_tokens: int = 4,
    probe_cap: int = 500,
) -> tuple[list[str], dict[str, int]]:
    candidates = _norm_drop_empty(df[sc_col].tolist())
    seen, ordered = set(), []
    for t in candidates:
        if t not in seen:
            seen.add(t); ordered.append(t)
        if len(ordered) >= probe_cap:
            break
    if not ordered:
        return [], {}
    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        con.register("probe", pd.DataFrame({"tok": ordered}))
        sql = ("SELECT tokenized, COUNT(DISTINCT tableid) AS tc "
               "FROM AllTables WHERE tokenized IN (SELECT tok FROM probe) "
               "GROUP BY tokenized ORDER BY tc DESC")
        sql = sql.replace("AllTables", physical_table("blend_index", _demo_layout(con), sql))
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    counts = {tok: int(tc) for tok, tc in rows}
    broad = [tok for tok, _ in rows[:max_tokens]]
    return broad, counts


def _score(branch_ids: list[int], gt_basenames: set[str],
           int_to_bn: dict[int, str], k: int) -> dict:
    bn_hits = {int_to_bn[i] for i in branch_ids if i in int_to_bn}
    true_hits = bn_hits & gt_basenames
    n_ret = len(set(branch_ids))
    return {
        "prec_at_k": len(true_hits) / k if k else 0.0,
        "prec_at_n": len(true_hits) / n_ret if n_ret else 0.0,
        "recall":    len(true_hits) / len(gt_basenames) if gt_basenames else 0.0,
        "hits":      len(true_hits),
        "n_ret":     n_ret,
        "n_gt":      len(gt_basenames),
    }


def _print_gt_table(rows: list[tuple[str, str, list[int]]],
                    gt: dict[str, set[str]], int_to_bn: dict[int, str]) -> None:
    print(f"  {'branch':<20}{'k':>3}  {'n_ret':>5}  {'prec@k':>7}  "
          f"{'prec@n':>7}  {'rec@k':>7}  found")
    print(f"  {'-'*20}{'-'*3}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*22}")
    for name, task, ids in rows:
        s = _score(ids, gt[task], int_to_bn, _K)
        print(f"  {name:<20}{_K:>3}  {s['n_ret']:>5}  "
              f"{s['prec_at_k']:>7.3f}  {s['prec_at_n']:>7.3f}  "
              f"{s['recall']:>7.3f}  "
              f"{s['hits']:>3} / {s['n_gt']} in-split {task}-GT")


def _pick_kw_rare_tokens(
    cfg: SemanticConfig,
    df: pd.DataFrame,
    sc_col: str,
    max_tokens: int = 4,
    rare_threshold: int = 5,
    probe_cap: int = 2000,
) -> tuple[list[str], dict[str, int]]:
    seen, ordered = set(), []
    for col in df.columns:
        for raw in df[col]:
            t = _norm(raw)
            if t and t not in seen:
                seen.add(t); ordered.append(t)
                if len(ordered) >= probe_cap:
                    break
        if len(ordered) >= probe_cap:
            break
    if not ordered:
        return [], {}
    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        con.register("probe", pd.DataFrame({"tok": ordered}))
        sql = ("SELECT tokenized, COUNT(DISTINCT tableid) AS tc "
               "FROM AllTables WHERE tokenized IN (SELECT tok FROM probe) "
               "GROUP BY tokenized ORDER BY tc ASC")
        sql = sql.replace("AllTables", physical_table("blend_index", _demo_layout(con), sql))
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    counts = {tok: int(tc) for tok, tc in rows}
    rare = [tok for tok, tc in rows if tc <= rare_threshold][:max_tokens]
    if not rare and rows:
        rare = [rows[0][0]]
    return rare, counts


def _build_cheap_seekers(df: pd.DataFrame, cols: dict, kw_rare: list[str]) -> dict:
    sc = Seekers.SC(_norm_drop_empty(df[cols["sc_col"]]), k=_K_CHEAP_HNSW)
    c = Seekers.C(
        source_values=[_norm(v) for v in df[cols["c_src_col"]]],
        target_values=pd.to_numeric(df[cols["c_tgt_col"]], errors="coerce").tolist(),
        k=_K_CHEAP_HNSW,
    )
    kw = Seekers.Keyword(kw_rare, k=_K)
    return {"sc": sc, "c": c, "kw_rare": kw}


class _AutoTuneError(ValueError):
    """Auto-tune rejected this T — caller may retry with the next candidate."""


def _auto_tune_threshold(cheap: dict) -> dict:
    sc_raw, sc_ms = _timed(cheap["sc"].run)
    c_raw, c_ms = _timed(cheap["c"].run)
    kw_raw, kw_ms = _timed(cheap["kw_rare"].run)
    sc_ids = set(sc_raw)
    c_ids = set(c_raw)
    kw_ids = set(kw_raw)
    small_fs = len(kw_ids)
    large_fs = len(sc_ids & c_ids)
    if small_fs == 0:
        raise _AutoTuneError(
            f"kw_rare filter is empty ({small_fs}); rare-token probe found "
            "nothing matching the lake."
        )
    if large_fs == 0:
        raise _AutoTuneError(
            f"sc ∩ c filter is empty ({large_fs}); upstream chain doesn't "
            "overlap. T may be isolated from the rest of the lake."
        )
    if small_fs >= large_fs - 4:
        raise _AutoTuneError(
            f"filter sizes don't bracket cleanly: small_fs={small_fs}, "
            f"large_fs={large_fs}. Need small_fs ≤ large_fs - 5."
        )
    T_thr = (small_fs + large_fs) // 2
    return {
        "T_thr": T_thr,
        "small_fs": small_fs,
        "large_fs": large_fs,
        "sc_ids": sc_ids,
        "c_ids": c_ids,
        "kw_ids": kw_ids,
        "sc_raw": sc_raw,
        "c_raw": c_raw,
        "kw_raw": kw_raw,
        "sc_ms": sc_ms,
        "c_ms": c_ms,
        "kw_ms": kw_ms,
    }


def _setup(cfg: SemanticConfig) -> dict:
    print(f"dataset: {cfg.dataset.name}\n")
    picks, _, _, _ = _pick_query_tables(cfg)
    print(f"=== 1. query selection ===")
    print(f"  ranked candidates: {len(picks)} (will fall through on auto-tune reject)")

    pick = df = cols = kw_rare = cheap = tune = None
    for rank, candidate in enumerate(picks):
        print(f"\n  --- candidate #{rank+1}: {candidate['T']} ---")
        print(f"    q_sj_col   = {candidate['q_sj_col']!r}")
        print(f"    in-split union-GT: {candidate['union_gt_count']}   "
              f"join-GT: {candidate['join_gt_count']}")
        cand_df = _load_query_df(cfg, candidate["T"])
        print(f"    T loaded   : {len(cand_df)} rows × {cand_df.shape[1]} cols")
        cand_cols = _pick_columns(cand_df)
        print(f"    sc_col     = {cand_cols['sc_col']!r}    (distinct={cand_cols['distincts'][cand_cols['sc_col']]})")
        print(f"    c_src_col  = {cand_cols['c_src_col']!r}  (distinct={cand_cols['distincts'][cand_cols['c_src_col']]})")
        print(f"    c_tgt_col  = {cand_cols['c_tgt_col']!r}  (numeric ratio={cand_cols['numeric_ratio'][cand_cols['c_tgt_col']]:.2f})")
        print(f"    mc_cols    = {cand_cols['mc_cols']}")
        cand_kw, _ = _pick_kw_rare_tokens(cfg, cand_df, cand_cols["sc_col"])
        print(f"    kw_rare    = {cand_kw}")
        cand_cheap = _build_cheap_seekers(cand_df, cand_cols, cand_kw)
        try:
            cand_tune = _auto_tune_threshold(cand_cheap)
        except _AutoTuneError as e:
            print(f"    REJECTED: {e}  → next candidate")
            continue
        print(f"    ACCEPTED. threshold auto-tune:")
        print(f"      |kw_rare|       = {cand_tune['small_fs']}")
        print(f"      |sc ∩ c|        = {cand_tune['large_fs']}")
        print(f"      T_thr           = {cand_tune['T_thr']}   "
              f"(brackets [{cand_tune['small_fs']}, {cand_tune['large_fs']}])")
        pick, df, cols, kw_rare, cheap, tune = (
            candidate, cand_df, cand_cols, cand_kw, cand_cheap, cand_tune
        )
        break

    if pick is None:
        raise SystemExit(
            f"no candidate T in {cfg.dataset.name} passed auto-tune. "
            "Loosen bracket margin in _auto_tune_threshold or widen "
            "c_src/sc selection in _pick_columns."
        )

    return {
        "cfg": cfg, "pick": pick, "df": df, "cols": cols,
        "kw_rare": kw_rare, "tune": tune, "cheap": cheap,
    }


def plan_a(ctx: dict) -> None:
    """Plan A: Counter(A_union, B_union) with SU/SJ ids pushed down into MC."""
    print(f"\n############ PLAN A — chained semantic + MC pushdown ############\n")
    pick, df, cols, tune, cheap = (
        ctx["pick"], ctx["df"], ctx["cols"], ctx["tune"], ctx["cheap"]
    )
    su = Seekers.SU(df, k=_K, exact_threshold=tune["T_thr"])
    sj = Seekers.SJ(
        df[[pick["q_sj_col"]]], k=_K, exact_threshold=tune["T_thr"],
        query_table_id=pick["T"],
    )
    mc = Seekers.MC(df[cols["mc_cols"]].head(500).map(_norm), k=_K)

    plan = Plan()
    plan.add("kw_rare", cheap["kw_rare"])
    plan.add("sc",      cheap["sc"])
    plan.add("c",       cheap["c"])
    plan.add("su",      su)
    plan.add("sj",      sj)
    plan.add("mc",      mc)
    plan.add("A_hnsw_SU",  Combiners.Intersection(k=_K), inputs=["sc", "c", "su", "mc"])
    plan.add("A_hnsw_SJ",  Combiners.Intersection(k=_K), inputs=["sc", "c", "sj", "mc"])
    plan.add("B_exact_SU", Combiners.Intersection(k=_K), inputs=["kw_rare", "su"])
    plan.add("B_exact_SJ", Combiners.Intersection(k=_K), inputs=["kw_rare", "sj"])
    plan.add("A_union",    Combiners.Union(k=_K),
             inputs=["A_hnsw_SU", "A_hnsw_SJ"])
    plan.add("B_union",    Combiners.Union(k=_K),
             inputs=["B_exact_SU", "B_exact_SJ"])
    plan.add("terminal",   Combiners.Counter(k=_K),
             inputs=["A_union", "B_union"])

    print(f"  plan built: {len(plan._operators)} operators, "
          f"terminal = {next(iter(plan._terminal_candidates))!r}")
    print(f"  seeker costs: " + ", ".join(
        f"{n}={plan._operators[n].cost()}" for n in ("kw_rare","sc","c","su","sj","mc")
    ))

    print(f"\n=== 2. seekers standalone (.run(), no pushdown) ===")
    cached = {
        "kw_rare": (tune["kw_raw"], tune["kw_ms"]),
        "sc":      (tune["sc_raw"], tune["sc_ms"]),
        "c":       (tune["c_raw"],  tune["c_ms"]),
    }
    for name in ("kw_rare", "sc", "c", "su", "sj", "mc"):
        if name in cached:
            ids, ms = cached[name]
        else:
            ids, ms = _timed(plan._operators[name].run)
        _print_row(name, ids, ms)

    sc_ids = tune["sc_ids"]
    c_ids = tune["c_ids"]
    kw_ids = tune["kw_ids"]
    T_thr = tune["T_thr"]

    branch_results: dict[str, list[int]] = {}

    def _run_hnsw_branch(name: str, semantic_name: str) -> list[int]:
        print(f"\n  --- {name} = Intersection(sc, c, {semantic_name}, mc) ---", flush=True)
        upstream = sc_ids & c_ids
        print(f"    sc ∩ c:              |F|={len(upstream)}   → passed to {semantic_name}")
        mode = "EXACT" if len(upstream) <= T_thr else "HNSW"
        print(f"    {semantic_name} receives:        |F|={len(upstream)} "
              f"{'≤' if mode == 'EXACT' else '>'} T_thr={T_thr}  → {mode} path", flush=True)
        sem_op = plan._operators[semantic_name]
        sem_ids, sem_ms = _timed(lambda: sem_op.run(_ids_to_pushdown(upstream)))
        _print_row(f"    {semantic_name} result", sem_ids, sem_ms)
        mc_filter = upstream & set(sem_ids)
        mc_pushdown = _ids_to_pushdown(mc_filter)
        print(f"    mc receives:         |F|={len(mc_filter)} ids in pushdown")
        print(f"    mc pushdown clause:  {mc_pushdown.strip()[:_SQL_EXCERPT]}"
              f"{'…' if len(mc_pushdown) > _SQL_EXCERPT else ''}", flush=True)
        wired_ids, wired_ms = _timed(plan._operators[name].run)
        _print_row(f"    plan-wired {name}", wired_ids, wired_ms)
        branch_results[name] = wired_ids
        return wired_ids

    def _run_exact_branch(name: str, semantic_name: str) -> list[int]:
        print(f"\n  --- {name} = Intersection(kw_rare, {semantic_name}) ---", flush=True)
        upstream = kw_ids
        mode = "EXACT" if len(upstream) <= T_thr else "HNSW"
        print(f"    kw_rare result:      |F|={len(upstream)}   → passed to {semantic_name}")
        print(f"    {semantic_name} receives:        |F|={len(upstream)} "
              f"{'≤' if mode == 'EXACT' else '>'} T_thr={T_thr}  → {mode} path", flush=True)
        sem_op = plan._operators[semantic_name]
        sem_ids, sem_ms = _timed(lambda: sem_op.run(_ids_to_pushdown(upstream)))
        _print_row(f"    {semantic_name} result", sem_ids, sem_ms)
        wired_ids, wired_ms = _timed(plan._operators[name].run)
        _print_row(f"    plan-wired {name}", wired_ids, wired_ms)
        branch_results[name] = wired_ids
        return wired_ids

    print(f"\n=== 3. A_hnsw branches (semantic in HNSW mode) ===", flush=True)
    _run_hnsw_branch("A_hnsw_SU", "su")
    _run_hnsw_branch("A_hnsw_SJ", "sj")

    print(f"\n=== 4. B_exact branches (semantic in exact mode) ===", flush=True)
    _run_exact_branch("B_exact_SU", "su")
    _run_exact_branch("B_exact_SJ", "sj")

    for _name, _ids in branch_results.items():
        _op = plan._operators[_name]
        _op._cached_sql = plan.DB.table_ids_to_sql(_ids) if _ids else "SELECT TableId FROM AllTables WHERE 1=0"
        _op.create_sql_query = lambda db, additionals="", _self=_op: _self._cached_sql

    print(f"\n=== 5. Unions + terminal Counter ===")
    a_union_ids, a_union_ms = _timed(plan._operators["A_union"].run)
    _print_row("A_union", a_union_ids, a_union_ms)
    b_union_ids, b_union_ms = _timed(plan._operators["B_union"].run)
    _print_row("B_union", b_union_ids, b_union_ms)
    terminal_ids, terminal_ms = _timed(plan.run)
    _print_row("terminal (plan.run())", terminal_ids, terminal_ms)

    print(f"\n=== 6. GT comparison (k={_K}, filtered to in-split GT) ===")
    gt = {"union": pick["union_gt"], "join": pick["join_gt"]}
    _print_gt_table(
        [
            ("A_hnsw_SU",  "union", plan._operators["A_hnsw_SU"].run()),
            ("B_exact_SU", "union", plan._operators["B_exact_SU"].run()),
            ("A_hnsw_SJ",  "join",  plan._operators["A_hnsw_SJ"].run()),
            ("B_exact_SJ", "join",  plan._operators["B_exact_SJ"].run()),
        ],
        gt,
        pick["int_to_bn"],
    )


def plan_b(ctx: dict) -> None:
    """Plan B: HNSW-vs-exact dispatch for both semantic seekers, Counter terminal, no MC."""
    print(f"\n############ PLAN B — dispatch knob isolation (no MC) ############\n")
    pick, df, tune, cheap = (
        ctx["pick"], ctx["df"], ctx["tune"], ctx["cheap"]
    )
    su = Seekers.SU(df, k=_K, exact_threshold=tune["T_thr"])
    sj = Seekers.SJ(
        df[[pick["q_sj_col"]]], k=_K, exact_threshold=tune["T_thr"],
        query_table_id=pick["T"],
    )

    plan = Plan()
    plan.add("kw_rare", cheap["kw_rare"])
    plan.add("sc",      cheap["sc"])
    plan.add("c",       cheap["c"])
    plan.add("su",      su)
    plan.add("sj",      sj)
    plan.add("hnsw_SU",  Combiners.Intersection(k=_K), inputs=["sc", "c", "su"])
    plan.add("hnsw_SJ",  Combiners.Intersection(k=_K), inputs=["sc", "c", "sj"])
    plan.add("exact_SU", Combiners.Intersection(k=_K), inputs=["kw_rare", "su"])
    plan.add("exact_SJ", Combiners.Intersection(k=_K), inputs=["kw_rare", "sj"])
    plan.add("terminal", Combiners.Counter(k=_K),
             inputs=["hnsw_SU", "hnsw_SJ", "exact_SU", "exact_SJ"])
    print(f"  plan built: {len(plan._operators)} operators, "
          f"terminal = Counter(hnsw_SU, hnsw_SJ, exact_SU, exact_SJ)")

    T_thr = tune["T_thr"]
    upstream_hnsw = tune["sc_ids"] & tune["c_ids"]
    upstream_exact = tune["kw_ids"]
    print(f"  upstream sizes: hnsw |F|={len(upstream_hnsw)} (>{T_thr}=HNSW), "
          f"exact |F|={len(upstream_exact)} (≤{T_thr}=EXACT)")

    print(f"\n=== branches ===")
    branch_ids: dict[str, list[int]] = {}
    for name, sem_name in (("hnsw_SU","su"),("hnsw_SJ","sj"),
                           ("exact_SU","su"),("exact_SJ","sj")):
        ids, ms = _timed(plan._operators[name].run)
        branch_ids[name] = ids
        _print_row(f"  {name}", ids, ms)

    for _n, _ids in branch_ids.items():
        _op = plan._operators[_n]
        _op._cached_sql = plan.DB.table_ids_to_sql(_ids) if _ids else "SELECT TableId FROM AllTables WHERE 1=0"
        _op.create_sql_query = lambda db, additionals="", _self=_op: _self._cached_sql

    terminal_ids, terminal_ms = _timed(plan.run)
    _print_row("  terminal (Counter)", terminal_ids, terminal_ms)

    print(f"\n=== HNSW vs EXACT diff (same seeker, different mode) ===")
    for sem in ("SU", "SJ"):
        h, e = set(branch_ids[f"hnsw_{sem}"]), set(branch_ids[f"exact_{sem}"])
        print(f"  {sem}:  hnsw={len(h)}  exact={len(e)}  ∩={len(h & e)}  "
              f"hnsw-only={sorted(h - e) or '∅'}  exact-only={sorted(e - h) or '∅'}")

    print(f"\n=== GT (k={_K}, in-split filtered) ===")
    gt = {"union": pick["union_gt"], "join": pick["join_gt"]}
    _print_gt_table(
        [
            ("hnsw_SU",  "union", branch_ids["hnsw_SU"]),
            ("exact_SU", "union", branch_ids["exact_SU"]),
            ("hnsw_SJ",  "join",  branch_ids["hnsw_SJ"]),
            ("exact_SJ", "join",  branch_ids["exact_SJ"]),
        ],
        gt,
        pick["int_to_bn"],
    )


def plan_c(ctx: dict) -> None:
    """Plan C: ensemble vote Counter(kw_broad, sc, su, sj) without pushdown."""
    print(f"\n############ PLAN C — ensemble vote (no pushdown) ############\n")
    cfg, pick, df, cols = ctx["cfg"], ctx["pick"], ctx["df"], ctx["cols"]
    kw_broad, broad_counts = _pick_kw_broad_tokens(cfg, df, cols["sc_col"])
    print(f"  kw_broad tokens (most common in T[{cols['sc_col']!r}]):")
    for t in kw_broad:
        print(f"    {t!r}: {broad_counts.get(t, 0)} distinct tables")
    if not kw_broad:
        raise SystemExit("kw_broad probe returned no tokens; T's sc_col is empty?")

    kw = Seekers.Keyword(kw_broad, k=_K)
    sc = Seekers.SC(_norm_drop_empty(df[cols["sc_col"]]), k=_K)
    su = Seekers.SU(df, k=_K)
    sj = Seekers.SJ(
        df[[pick["q_sj_col"]]], k=_K, query_table_id=pick["T"],
    )

    plan = Plan()
    plan.add("kw_broad", kw)
    plan.add("sc",       sc)
    plan.add("su",       su)
    plan.add("sj",       sj)
    plan.add("terminal", Combiners.Counter(k=_K),
             inputs=["kw_broad", "sc", "su", "sj"])
    print(f"  plan built: {len(plan._operators)} operators, "
          f"terminal = Counter(kw_broad, sc, su, sj)")

    print(f"\n=== seekers standalone ===")
    standalone: dict[str, list[int]] = {}
    for name in ("kw_broad", "sc", "su", "sj"):
        ids, ms = _timed(plan._operators[name].run)
        standalone[name] = ids
        _print_row(f"  {name}", ids, ms)

    terminal_ids, terminal_ms = _timed(plan.run)
    _print_row("\n  terminal (Counter)", terminal_ids, terminal_ms)

    seeker_sets = {n: set(ids) for n, ids in standalone.items()}
    vote_rows = []
    for tid in terminal_ids:
        votes = [n for n, s in seeker_sets.items() if tid in s]
        vote_rows.append((tid, len(votes), votes))
    print(f"\n=== vote distribution (terminal top-{len(terminal_ids)}) ===")
    for tid, n_votes, voters in vote_rows:
        print(f"  TableId {tid:>4}  votes={n_votes}  by={voters}")

    print(f"\n=== GT (k={_K}, in-split filtered) ===")
    gt = {"union": pick["union_gt"], "join": pick["join_gt"]}
    _print_gt_table(
        [
            ("kw_broad",    "union", standalone["kw_broad"]),
            ("sc",          "union", standalone["sc"]),
            ("su",          "union", standalone["su"]),
            ("sj",          "join",  standalone["sj"]),
            ("terminal (U)", "union", terminal_ids),
            ("terminal (J)", "join",  terminal_ids),
        ],
        gt,
        pick["int_to_bn"],
    )


def plan_d(ctx: dict) -> None:
    """Plan D: semantic-first (cost-overridden) vs cheap-first Intersection chains, diffed."""
    print(f"\n############ PLAN D — semantic-first vs cheap-first (no MC) ############\n")
    pick, df, cols, tune, cheap = (
        ctx["pick"], ctx["df"], ctx["cols"], ctx["tune"], ctx["cheap"]
    )

    class _SULead(Seekers.SU):
        def cost(self): return 0
    class _SJLead(Seekers.SJ):
        def cost(self): return 0
    class _SemanticLeadUnion(Combiners.Union):
        def cost(self): return 0

    _K_SEM_LEAD = 100
    su_lead = _SULead(df, k=_K_SEM_LEAD)
    sj_lead = _SJLead(
        df[[pick["q_sj_col"]]], k=_K_SEM_LEAD, query_table_id=pick["T"],
    )
    plan_lead = Plan()
    plan_lead.add("su_lead",   su_lead)
    plan_lead.add("sj_lead",   sj_lead)
    plan_lead.add("sem_pool",  _SemanticLeadUnion(k=2 * _K_SEM_LEAD),
                  inputs=["su_lead", "sj_lead"])
    plan_lead.add("kw_rare",   cheap["kw_rare"])
    plan_lead.add("sc",        cheap["sc"])
    plan_lead.add("terminal",  Combiners.Intersection(k=_K),
                  inputs=["sem_pool", "kw_rare", "sc"])
    print(f"  D_lead plan: cost-sorted inputs of terminal Intersection:")
    print(f"    sem_pool(k={2*_K_SEM_LEAD})=0, kw_rare=3, sc=4  →  sem_pool runs FIRST")
    print(f"    su_lead.k={_K_SEM_LEAD}  sj_lead.k={_K_SEM_LEAD}  "
          f"sem_pool.k={2*_K_SEM_LEAD}")

    lead_ids, lead_ms = _timed(plan_lead.run)
    _print_row("  D_lead (semantic-first)", lead_ids, lead_ms)

    su_cheap = Seekers.SU(df, k=_K, exact_threshold=tune["T_thr"])
    sj_cheap = Seekers.SJ(
        df[[pick["q_sj_col"]]], k=_K, exact_threshold=tune["T_thr"],
        query_table_id=pick["T"],
    )
    kw_cheap = Seekers.Keyword(ctx["kw_rare"], k=_K)
    sc_cheap = Seekers.SC(_norm_drop_empty(df[cols["sc_col"]]), k=_K_CHEAP_HNSW)
    plan_cheap = Plan()
    plan_cheap.add("kw_rare", kw_cheap)
    plan_cheap.add("sc",      sc_cheap)
    plan_cheap.add("su",      su_cheap)
    plan_cheap.add("sj",      sj_cheap)
    plan_cheap.add("terminal", Combiners.Intersection(k=_K),
                   inputs=["kw_rare", "sc", "su", "sj"])
    print(f"\n  D_cheap plan: cost-sorted inputs of terminal Intersection:")
    print(f"    kw_rare=3, sc=4, su=7, sj=8  →  kw_rare runs FIRST")

    cheap_ids, cheap_ms = _timed(plan_cheap.run)
    _print_row("  D_cheap (cheap-first)", cheap_ids, cheap_ms)

    print(f"\n=== ordering diff ===")
    L, C = set(lead_ids), set(cheap_ids)
    print(f"  D_lead returned : {sorted(L)}")
    print(f"  D_cheap returned: {sorted(C)}")
    print(f"  ∩               : {sorted(L & C) or '∅'}  ({len(L & C)} tables)")
    print(f"  D_lead only     : {sorted(L - C) or '∅'}  "
          f"(semantic-first surfaced these, cheap-first dropped)")
    print(f"  D_cheap only    : {sorted(C - L) or '∅'}  "
          f"(cheap-first kept these, semantic-first dropped)")

    print(f"\n=== GT (k={_K}, in-split filtered) ===")
    gt = {"union": pick["union_gt"], "join": pick["join_gt"]}
    _print_gt_table(
        [
            ("D_lead (U-GT)",  "union", lead_ids),
            ("D_lead (J-GT)",  "join",  lead_ids),
            ("D_cheap (U-GT)", "union", cheap_ids),
            ("D_cheap (J-GT)", "join",  cheap_ids),
        ],
        gt,
        pick["int_to_bn"],
    )


_PLAN_FNS = {"a": plan_a, "b": plan_b, "c": plan_c, "d": plan_d}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", choices=["a","b","c","d","all"], default="all",
                    help="which plan(s) to run; 'all' runs a,b,c,d sequentially")
    args = ap.parse_args()
    cfg = SemanticConfig.load()
    ctx = _setup(cfg)
    plans = ["a","b","c","d"] if args.plan == "all" else [args.plan]
    for p in plans:
        _PLAN_FNS[p](ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
