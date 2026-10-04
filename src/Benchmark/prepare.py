from __future__ import annotations

from pathlib import Path

from src.Benchmark.datasource import dataset_dir


def bench_task_dir(dataset: str, task: str) -> Path:
    return dataset_dir(dataset) / "bench" / task


def guard_outputs(paths: list[Path], force: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not force:
        raise FileExistsError(
            f"refusing to overwrite {', '.join(str(p) for p in existing)} "
            f"(pass --force)")


def dispatch(task: str, dataset: str, *, n: int, k: int, seed: int, force: bool,
             require_semantic: bool = False) -> None:
    out = bench_task_dir(dataset, task)
    out.mkdir(parents=True, exist_ok=True)
    if task == "imputation":
        from src.Benchmark.recipes.imputation import prepare_imputation
        prepare_imputation(dataset, out, n=n, seed=seed, force=force,
                           require_semantic=require_semantic)
    elif task == "correlation":
        from src.Benchmark.recipes.correlation import prepare_correlation
        prepare_correlation(dataset, out, n=n, k=k, seed=seed, force=force)
    elif task == "negex":
        from src.Benchmark.recipes.negative_example import prepare_negex
        prepare_negex(dataset, out, n_neg=1000, seed=seed, force=force)
    else:
        raise ValueError(task)
