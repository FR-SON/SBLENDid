# Blend with semantic seekers (master's thesis code)

This repository is the code of a master's thesis. It is a fork of **Blend**
([github.com/LUH-DBS/Blend](https://github.com/LUH-DBS/Blend)), the data discovery system
by Esmailoghli et al. Blend's index, operator algebra and SQL rewriting are the work of
the original authors. The thesis extends Blend with:

- semantic seekers `SemanticUnion` (SU) and `SemanticJoin` (SJ), which retrieve tables
  by column-embedding similarity, with LIFTus, Snoopy and DeepJoin plugged in as
  third-party encoders;
- a SimHash-overlap seeker (SHO) that follows SemDisc;
- DuckDB and Postgres (+ pgvector) backends, SQL pushdown into the semantic seekers,
  a cost model and a benchmark CLI.

Encoder training, embedding generation and the evaluation protocol live in **JUDIT**
([github.com/FR-SON/judit](https://github.com/FR-SON/judit)), a harness written for the
same thesis.

Rules, caveats and configuration details that the code does not make obvious are in
[docs/notes.md](docs/notes.md).

## Sources

If you use Blend, cite:

```bibtex
@inproceedings{esmailoghliBLENDUnifiedData2025,
  title = {{{BLEND}}: {{A Unified Data Discovery System}}},
  booktitle = {Proceedings {{ICDE}} 2025},
  author = {Esmailoghli, Mahdi and Schnell, Christoph and Miller, Ren{\'e}e and Abedjan, Ziawasch},
  year = 2025,
  publisher = {IEEE},
  doi = {10.1109/ICDE65448.2025.00061},
}
```

Third-party methods used by the thesis extensions:

| Method   | Used for          | Paper                                                                                      |
| -------- | ----------------- | ------------------------------------------------------------------------------------------ |
| LIFTus   | column encoder    | Qiu et al., ICDE 2025, [doi:10.1109/ICDE65448.2025.00165](https://doi.org/10.1109/ICDE65448.2025.00165) |
| Snoopy   | column encoder    | Guo et al., TKDE 2025, [doi:10.1109/TKDE.2025.3545176](https://doi.org/10.1109/TKDE.2025.3545176)       |
| DeepJoin | column encoder    | Dong et al., PVLDB 2023, [doi:10.14778/3603581.3603587](https://doi.org/10.14778/3603581.3603587)       |
| SemDisc  | SHO seeker design | Mohammad and Rezig, PACMMOD 2026, [doi:10.1145/3786682](https://doi.org/10.1145/3786682)                |

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

`uv sync` fetches the encoder packages `liftus` and `snoopy` from the
[judit repository](https://github.com/FR-SON/judit) (tag `v1.0.0`, `baselines/`), so `git`
and network access are needed the first time.

## Data

No datasets, indexes or checkpoints are tracked. All thesis data is on Zenodo:
[doi:10.5281/zenodo.22953090](https://doi.org/10.5281/zenodo.22953090). Each lake is one
archive `<lake>.tar.zst` that unpacks as-is into `datasets/`. It contains the lake CSVs
(`csvs/`), the union/join query and groundtruth files (`query/`, `groundtruth/`), the
encoder checkpoints with their LIFTus aspects or DeepJoin sentence caches
(`semantic/<approach>/default/`) and the cost profiles (`optimizer/`). The token index,
the semantic indexes and the SHO codes are not included and are built below.

## Build the indexes and run a benchmark (local, DuckDB + FAISS)

The checked-in `config/config.ini` targets DuckDB + FAISS with the `two_table` layout.
`<lake>` is the archive's directory name, `<prefix>` the prefix of its query and
groundtruth files (e.g. `opendata`).

```bash
zstd -dc <lake>.tar.zst | tar -xf - -C datasets/
# set [Dataset] name = <lake> in config/config.ini

# token index + basenames sidecar
uv run python scripts/create_blend_csv_index.py --csv-dir datasets/<lake>/csvs --dataset <lake> --layout two_table

# semantic index, once per approach the lake ships: liftus | snoopy | deepjoin
uv run python scripts/create_semantic_index.py --csv-dir datasets/<lake>/csvs --dataset <lake> \
    --approach liftus --index-name default

# optional: SHO code index (samples, encodes and hashes cell values locally;
# downloads the sentence-transformer model on first use)
uv run python scripts/create_sho_index.py --csv-dir datasets/<lake>/csvs --dataset <lake>

# union retrieval with LIFTus at k = 10
uv run python -m src.Benchmark.cli bench --dataset <prefix> --tasks union --seekers liftus --k 10
```

- Build the token index and every semantic index from the **same CSV directory**; table
  ids are joined to basenames through the sidecar. The token ingest of a lake with a few
  hundred tables takes minutes, not seconds.
- Snoopy needs the fastText model `cc.en.300.bin` at index-build time:
  `export BLEND_SNOOPY_FASTTEXT_PATH=/path/to/cc.en.300.bin`. DeepJoin needs nltk data
  (`BLEND_DEEPJOIN_NLTK_PATH`) only when a lake has no sentence cache.
- `--dataset` of `bench` is the query/groundtruth file **prefix**; the lake and the
  backend come from `config/config.ini` (or the file in `BLEND_CONFIG`).
- `bench` defaults to `--tasks union,join --seekers sc,sho,liftus,snoopy,deepjoin` and a
  full k-sweep; methods whose index is missing are skipped. `--repeat n` reports medians.
- Retrieval depth changes the reported numbers. The checked-in config pins
  `faiss_k_coarse = faiss_hnsw_ef_search = 500`; see
  [docs/notes.md](docs/notes.md#retrieval-depth-and-vector-backends) for what these
  settings mean and how `bench` overrides them.

Results go to `datasets/<lake>/bench_results/<subcommand>/<run-dir>/`. Each run directory
has a `manifest.json` (effective arguments, backend, status, wall time); `bench` adds
`aggregate.json`, `per_repeat.json` and `per_query.parquet`.

Other subcommands of `python -m src.Benchmark.cli` (each has `--help`): `run` (one task at
one `k`), `prepare`, `compare`, `correctness-sweep`, `correctness-theorem1`,
`tiebreak-bench`, `pushdown-bench`, `ann-frontier`, `regime-recall`, `condensed-recall`,
`plan`.

### Use the seekers in a Plan

Seekers compose with Blend's combiners. `Intersection` runs its inputs cheapest-first and
pushes each result into the next one as a `TableId IN (...)` filter, which the semantic
seekers apply to their vector search:

```python
import pandas as pd
from src.Operators import Combiners, Seekers
from src.Plan import Plan

q = pd.read_csv("datasets/<lake>/csvs/<table>.csv", dtype=str, keep_default_na=False)
q.attrs["table_id"] = "<table>.csv"   # semantic seekers look the query up by basename

plan = Plan()
plan.add("sc", Seekers.SC(q["<column>"], k=100))
plan.add("su", Seekers.SU(q, k=10))
plan.add("both", Combiners.Intersection(k=10), inputs=["sc", "su"])
ids = plan.run()   # list[int] of Blend TableIds
```

A query table that is not part of the indexed lake works too, as an opt-in: with
`[Semantic] query_encode = auto` (in the config or per seeker via `config_overrides`),
columns missing from the semantic index are encoded from their cells at query time (Snoopy
and DeepJoin; LIFTus raises `NotImplementedError`), the encoder is loaded lazily on the first
such query, and a `[SEMANTIC][QUERY-ENCODE]` line on stderr says which columns were encoded.
`table_id` is optional then. The default `query_encode = off` keeps every seeker, benchmark
and index script lookup-only: an unindexed query column is a hard error.

```python
q = pd.read_csv("my_table.csv")          # not in the lake
plan = Plan()
plan.add("sj", Seekers.SJ(q[["customer_id"]], k=10, config_overrides={"query_encode": "auto"}))
ids = plan.run()
```

### Postgres + pgvector (optional)

Start a local database with
`docker compose -f docker/postgres-compose.local.yml up -d`, use
`config/postgres_pgvector_example.ini` as the config (set `[Dataset] name = <lake>`), and
load the indexes built above. Make sure `shared_buffers` holds the vector tables
(see [docs/notes.md](docs/notes.md#postgres--pgvector)); otherwise pgvector timings are
not comparable to FAISS.

```bash
uv run python scripts/load_blend_index_pg.py    --csv-dir datasets/<lake>/csvs --dataset <lake>
uv run python scripts/load_semantic_index_pg.py --dataset <lake> --approach liftus --index-name default
uv run python scripts/load_sho_index_pg.py      --dataset <lake>
```

## Docker

The image is built from the repository root; the compose files live in `deploy/` and are
used from the repository root with `-f deploy/...`.

1. `cp deploy/env.example deploy/.env` and fill in: image name, `HOST_UID`/`HOST_GID`, and
   the absolute host paths `DATASETS_DIR`, `RUNS_DIR`, `CONFIG_FILE`, `FASTTEXT_BIN`,
   `NLTK_DATA_DIR` (all five must be set). Create `DATASETS_DIR` and `RUNS_DIR` owned by
   `HOST_UID`, unpack the lake into `DATASETS_DIR`, and set `[Dataset] name = <lake>` in
   `CONFIG_FILE`.
2. Build, then run the same steps inside the container. Always use `uv run --no-sync`
   there; the datasets are mounted at `/data/datasets`. `docker-compose.yaml` reserves an
   NVIDIA GPU; on a host without one, add the CPU overlay as below (with a GPU, drop
   `-f deploy/compose.cpu.yaml`).

```bash
docker compose -f deploy/docker-compose.yaml -f deploy/compose.cpu.yaml build
docker compose -f deploy/docker-compose.yaml -f deploy/compose.cpu.yaml run --rm blend \
    uv run --no-sync python scripts/create_blend_csv_index.py \
    --csv-dir /data/datasets/<lake>/csvs --dataset <lake> --layout two_table
docker compose -f deploy/docker-compose.yaml -f deploy/compose.cpu.yaml run --rm blend \
    uv run --no-sync python scripts/create_semantic_index.py \
    --csv-dir /data/datasets/<lake>/csvs --dataset <lake> --approach liftus --index-name default
docker compose -f deploy/docker-compose.yaml -f deploy/compose.cpu.yaml run --rm blend \
    uv run --no-sync python -m src.Benchmark.cli bench \
    --dataset <prefix> --tasks union --seekers liftus --k 10
```

For Postgres, run `docker network create blend-net`, create `PG_DATA_DIR` owned by
`HOST_UID` (default: `docker/pg-data/`; Docker would otherwise create it as root and
Postgres could not write to it), start `docker compose -f docker/postgres-compose.yml up -d`
(container `blend-pg`), and add `-f deploy/compose.pg.yaml` to the blend commands. Base
`CONFIG_FILE` on `config/postgres_pgvector_example.ini` with `[Database] host = blend-pg`,
`port = 5432`, and load the indexes with the `load_*_pg.py` scripts above.

## Acknowledgements

Parts of the code in this repository were written with the assistance of claude.ai by Anthropic.
