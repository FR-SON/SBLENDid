from __future__ import annotations

import csv
import json
from pathlib import Path

from src.Benchmark.plans.registry import PlanResult


def write_plan_results(result: PlanResult, *, k: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "results"

    all_rows = [{"arm": a.name, **r} for a in result.arms for r in a.rows]
    csv_path = out_dir / f"{stem}.csv"
    if all_rows:
        fields = ["arm"] + sorted(c for c in all_rows[0] if c != "arm")
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
    else:
        csv_path.write_text("")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps({
        "plan": result.plan, "point": result.point, "dataset": result.dataset,
        "task": result.task,
        "arms": [{"name": a.name, "summary": a.summary, "rows": a.rows}
                 for a in result.arms],
        "deltas": result.deltas, "meta": result.meta,
    }, indent=2))

    (out_dir / f"{stem}.md").write_text(_markdown(result, k))
    return json_path


def _markdown(result: PlanResult, k: int) -> str:
    m = result.meta
    out = [
        f"# Bench plan: {result.plan} ({result.task})", "",
        f"**Point:** {result.point}", "",
        f"**Dataset:** {result.dataset} | **k:** {k} | "
        f"**evaluated:** {m.get('evaluated')} | **git:** {m.get('git_sha')}", "",
        "| arm | recall | precision | nDCG | MAP | MRR | ms/q |",
        "|-----|--------|-----------|------|-----|-----|------|",
    ]
    for a in result.arms:
        s = a.summary
        out.append(
            f"| {a.name} | {s['mean_recall']:.4f} | {s['mean_precision']:.4f} "
            f"| {s['mean_ndcg']:.4f} | {s['MAP']:.4f} | {s['MRR']:.4f} "
            f"| {s['mean_runtime_ms']:.1f} |")
    out += ["", "## Deltas vs seeker_only", ""]
    for pair, metrics in result.deltas.items():
        for name, val in metrics.items():
            out.append(f"- `{pair}.{name}`: {val:+.4f}")
    out += ["", "## Reproducibility", ""]
    for key, val in m.get("config_snapshot", {}).items():
        out.append(f"- {key}: {val}")
    out.append(f"- seed: {m.get('seed')}")
    return "\n".join(out) + "\n"
