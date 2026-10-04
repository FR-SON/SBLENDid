import csv
import ctypes
import statistics
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor

from src.optimizer_paths import model_path as _artifact_model_path
from src.Optimizer.features import compute_features
from src.Optimizer import sampling as smp
from src.Operators.OperatorBase import Operator
from src.Operators.Seekers.SingleColumnOverlap import SingleColumnOverlap
from src.Operators.Seekers.MultiColumnOverlap import MultiColumnOverlap
from src.Operators.Seekers.Correlation import Correlation
from src.Operators.Seekers.Keyword import Keyword

MODEL_CLASS = {
    "SC": "SingleColumnOverlap",
    "Keyword": "Keyword",
    "C": "Correlation",
    "MC": "MultiColumnOverlap",
}


def build_seeker(seeker_type, columns, k=10):
    if seeker_type == "SC":
        return SingleColumnOverlap(columns[0], k)
    if seeker_type == "Keyword":
        return Keyword(columns[0], k)
    if seeker_type == "C":
        target = pd.to_numeric(pd.Series(columns[1]), errors="coerce").tolist()
        return Correlation(columns[0], target, k)
    if seeker_type == "MC":
        df = pd.DataFrame({f"c{i}": col for i, col in enumerate(columns)})
        return MultiColumnOverlap(df, k)
    raise ValueError(f"unknown seeker_type {seeker_type}")


def cost_columns(seeker_type, seeker):
    if seeker_type in ("SC", "Keyword"):
        return [list(seeker.input)]
    if seeker_type == "C":
        return [[t for t in seeker.input_source]]
    if seeker_type == "MC":
        return [list(col) for col in seeker.input.values.T]
    raise ValueError(f"unknown seeker_type {seeker_type}")


class QueryTimeout(RuntimeError):
    """A measured query hit its wall cap and was cancelled client-side."""


def apply_statement_timeout(db, seconds):
    """Set a server-side per-query timeout on postgres; no-op elsewhere."""
    if seconds and getattr(db, "dbms", None) == "postgres":
        with db.connection.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(seconds) * 1000}")
        db.connection.commit()


@contextmanager
def query_deadline(db, seconds):
    """Cap a seeker's whole run at `seconds` (duckdb interrupt + async exception in this thread)."""
    # DuckDB: interrupting the parent connection does not cancel a query on its cursor.
    targets, seen = [], set()
    if getattr(db, "dbms", None) == "duckdb":
        for name in ("cursor", "connection"):
            h = getattr(db, name, None)
            if h is not None and hasattr(h, "interrupt") and id(h) not in seen:
                seen.add(id(h))
                targets.append(h)
    if not seconds:
        yield
        return
    fired = threading.Event()
    tid = threading.get_ident()
    lock = threading.Lock()
    done = False

    def _cancel():
        with lock:
            if done:
                return
            fired.set()
            for h in targets:
                try:
                    h.interrupt()
                except Exception:
                    pass
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid), ctypes.py_object(QueryTimeout))

    timer = threading.Timer(float(seconds), _cancel)
    timer.daemon = True
    timer.start()
    try:
        yield
    except QueryTimeout:
        raise
    except Exception as exc:
        if fired.is_set():
            raise QueryTimeout(f"cancelled after {seconds}s wall cap") from exc
        raise
    else:
        if fired.is_set():
            raise QueryTimeout(f"exceeded {seconds}s wall cap")
    finally:
        with lock:
            done = True
        timer.cancel()
        if fired.is_set():
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


def is_timeout(exc: BaseException) -> bool:
    """Return True if the wall cap, not a query error, stopped this query."""
    if isinstance(exc, QueryTimeout):
        return True
    return (type(exc).__name__ == "QueryCanceled"
            or "statement timeout" in str(exc).lower())


def _reset_failed_txn(db):
    if getattr(db, "dbms", None) == "postgres":
        db.connection.rollback()


def measure_runtime(seeker, repeats=3, warmup=True, db=None, timeout=None):
    """Return the median of `repeats` timed runs, each capped at `timeout` seconds."""
    if warmup:
        with query_deadline(db, timeout):
            seeker.run()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with query_deadline(db, timeout):
            seeker.run()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def training_row(seeker_type, columns, db, k=10, repeats=3, warmup=True):
    seeker = build_seeker(seeker_type, columns, k)
    feats = compute_features(cost_columns(seeker_type, seeker), db)
    runtime = measure_runtime(seeker, repeats, warmup)
    return feats, runtime


def train_model(rows):
    X = [f for f, _ in rows]
    y = [r for _, r in rows]
    model = XGBRegressor(n_estimators=100, objective="reg:squarederror", n_jobs=16)
    model.fit(X, y)
    return model


def model_path(seeker_type, dataset_dir, profile) -> Path:
    return _artifact_model_path(dataset_dir, profile, MODEL_CLASS[seeker_type])


def dump_model(model, seeker_type, dataset_dir, profile) -> Path:
    out = model_path(seeker_type, dataset_dir, profile)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out)
    return out


def measure_single_seekers(db, csv_dir, out_dir, seeker_types=smp.SEEKER_TYPES,
                           k=10, repeats=3, warmup=True, samples_dir=None,
                           timeout=None, limit=None):
    """Write one CSV per seeker type of median unfiltered runtimes per sampled query; return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_db = Operator.DB
    Operator.DB = db
    sdir = Path(samples_dir) if samples_dir else out_dir
    written = []
    try:
        for stype in seeker_types:
            specs = smp.load_specs(sdir / f"samples_{stype}.jsonl")
            if limit:
                specs = specs[:limit]
            path = out_dir / f"single_seeker_runtimes_{stype}.csv"
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["seeker_type", "csv", "cols", "runtime_s", "outcome"])
                for s in tqdm(specs, desc=f"measure {stype}", unit="q"):
                    try:
                        cols = smp.load_columns(csv_dir, s)
                        seeker = build_seeker(stype, cols, k)
                        rt = measure_runtime(seeker, repeats=repeats, warmup=warmup,
                                             db=db, timeout=timeout)
                        w.writerow([stype, s.csv, "|".join(s.cols), rt, "ok"])
                        fh.flush()
                    except Exception as e:
                        outcome = "censored" if is_timeout(e) else "error"
                        w.writerow([stype, s.csv, "|".join(s.cols), "", outcome])
                        fh.flush()
                        tqdm.write(f"{outcome} {stype} {s.csv} {s.cols!r}: {e!r}")
                        _reset_failed_txn(db)
            written.append(path)
    finally:
        Operator.DB = base_db
    return written


def train_all(lake, db, csv_dir, out_dir, dataset_dir, profile, k=10, repeats=3,
              warmup=True, seeker_types=smp.SEEKER_TYPES, type_db=None):
    """Measure, fit and dump an XGB cost model per seeker type; type_db overrides the DB per type."""
    type_db = type_db or {}
    base_db = Operator.DB
    results = {}
    try:
        for stype in seeker_types:
            active = type_db.get(stype, db)
            Operator.DB = active
            specs = smp.load_specs(f"{out_dir}/samples_{stype}.jsonl")
            rows = []
            for s in tqdm(specs, desc=f"measure {stype}", unit="q"):
                try:
                    rows.append(training_row(stype, smp.load_columns(csv_dir, s), active, k=k, repeats=repeats, warmup=warmup))
                except Exception as e:
                    tqdm.write(f"skip {stype} {s.csv} {s.cols!r}: {e!r}")
                    _reset_failed_txn(active)
            path = dump_model(train_model(rows), stype, dataset_dir, profile)
            results[stype] = (path, len(rows))
    finally:
        Operator.DB = base_db
    return results
