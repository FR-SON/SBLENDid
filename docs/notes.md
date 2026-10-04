# Notes

Reference material that the [README](../README.md) leaves out: rules and caveats that
the code does not make obvious. For flags, run `--help` on the scripts and CLIs.

## Consistency rules

1. **Build the token index and every semantic index from the same CSV directory.**
   Integer `TableId`s and table basenames are joined through the sidecar
   (`blend_index_basenames.parquet`). A basename missing on one side makes
   `IndexHandle.open` raise a `KeyError` listing the missing ids.
2. **Keep the sidecar and the token index from the same ingest run.** Re-ingesting
   renumbers `TableId`s; rebuild both with the same `--dataset`.
3. **LIFTus:** every CSV basename must be present in every aspect pickle, and the pickle
   filename prefix must equal the checkpoint sidecar's `train_dataset`.
4. **Query columns are looked up; encoding is opt-in.** At query time SU and SJ look up the
   stored vector of `(table_id, col_name)`. By default (`[Semantic] query_encode = off`) a
   column absent from the registry is a `KeyError` and the table id (`df.attrs["table_id"]`
   or `query_table_id=`) is required (`ValueError`). With `query_encode = auto` the column
   is encoded from the query DataFrame's cells instead (see *Queries outside the index*).

## Retrieval depth and vector backends

- `faiss_k_coarse`, `faiss_hnsw_ef_search` and `exact_threshold` count **columns**, not
  tables.
- `faiss_k_coarse` set in `[Semantic]` pins one fetch depth for every `k`. Unset, the
  depths are derived from `k` (`src/Semantic/depths.py`: vote depth =
  `vote_depth_factor` × k; `restricted_k_coarse` when a pushdown filter is present). The
  checked-in config pins `faiss_k_coarse = faiss_hnsw_ef_search = 500`.
- `bench` overrides: `--k-coarse` pins the per-column fetch depth, `--k-vote` the vote
  depth, `--vote-factor` sets vote depth = factor × k.
- FAISS stops after `ef_search` graph expansions, so it needs `ef_search >= k_coarse` to
  return the requested width. pgvector's iterative scan refills the `LIMIT` regardless.
  The same `ef_search` therefore buys different work on the two backends: compare them
  at equal recall (`bench ann-frontier`), not at equal `ef_search`.
- `exact_threshold` is FAISS-only: a *filtered* search that allows at most this many
  columns uses brute force. Postgres decides exact vs. HNSW itself.

## Postgres + pgvector

Each dataset is a Postgres schema. Get these wrong and backend comparisons are invalid,
not only slow:

- `shared_buffers` must hold the whole `semantic_columns__*` table family (heap, HNSW
  index, primary key). Both compose files set 2 GB. `bench ann-frontier --io-stats`
  checks it and exits 1 when the second pass still reads from disk.
- The `embedding` column uses `STORAGE main` (set by `load_semantic_index_pg.py`);
  otherwise high-dimensional vectors are stored out of line.
- Keep `[Semantic] pgvector_hnsw_iterative_scan` on (`relaxed_order` or
  `strict_order`, pgvector ≥ 0.8). With `off`, pgvector returns fewer rows than `LIMIT`
  whenever `ef_search < LIMIT`.

## Configuration

One INI file, `config/config.ini` or the path in `BLEND_CONFIG`. Operators connect to the
database at import time, so the config must be valid before any operator module is
imported. Examples: `config/postgres_pgvector_example.ini`,
`config/semantic_example.ini`; `duckdb_example.ini`, `postgres_example.ini` and
`vertica_example.ini` are upstream (this fork ignores the DuckDB example's `path` key).

- `[Dataset]`: `name`, `root` (default `<repo>/datasets`). Every per-dataset path is
  `<root>/<name>/…`.
- `[Database]`: `dbms` (`duckdb` | `postgres` | `vertica`), `index_table`, `db_filename`,
  connection parameters, and `layout` (`single` | `two_table`). `layout` only tells the
  ingest what to build; queries detect the layout from the tables present.
- `[Semantic]`: the vector backend (`faiss` | `pgvector`), the HNSW/PQ parameters and
  the depth keys above; `semantic_ml_cost = analytic` turns on the fitted semantic cost
  model as a tie-breaker; `query_encode = off | auto` (default `off`: an unindexed query
  column fails; `auto`: encode it from its cells).
- `[Semantic.SU]`, `[Semantic.SJ]`, `[Semantic.SHO]`: `approach` (`liftus` | `snoopy` |
  `deepjoin`; `simhash` for SHO) and `index_name`; optional `aggregator = munkres` with
  `munkres_threshold`, and a per-operator `vote_depth_factor`.
- `[Optimizer]`: `cost_basis` (`logical` = fixed integer costs, `measured` =
  `datasets/<name>/optimizer/<dbms>-<vector_backend>-<layout>/costs.json`),
  `pushdown_guard` (`A`: an approximate pushdown source runs with
  `k = pushdown_filter_width`; `B`: approximate inputs run after exact ones),
  `pushdown_filter_width` (default 500). The logical costs are SJ 1, SU 2, Keyword 3,
  SC 4, SHO 5, C 6, MC 10 (Keyword, SC, C and MC from upstream). `Intersection` runs the
  cheapest input first.

Environment variables: `BLEND_CONFIG`, `BLEND_DATASETS_DIR` (overrides `[Dataset] root`),
`BLEND_RUNS_DIR` (Optimizer outputs, default `<repo>/runs`), `BLEND_USE_ML_OPTIMIZER`
(upstream's XGBoost ordering), `BLEND_DUCKDB_THREADS` / `BLEND_DUCKDB_MEMORY_LIMIT`,
`BLEND_SNOOPY_FASTTEXT_PATH`, `BLEND_DEEPJOIN_NLTK_PATH`,
`BLEND_PG_MAINTENANCE_WORK_MEM` / `BLEND_PG_MAX_PARALLEL_MAINTENANCE_WORKERS` (pgvector
index build), and the standard libpq `PG*` variables for values missing from
`[Database]`.

## Queries outside the index

Opt in with `[Semantic] query_encode = auto` or `config_overrides={"query_encode": "auto"}`
on a seeker; the default `off` keeps every existing script lookup-only. Under `auto`,
`retrieve.search` encodes every query column whose `(table_id, col_name)` is not in the registry. A column that is in the registry is always
served from the stored vector, even if the DataFrame's cells differ. The table id may be
omitted; `__query__` is used then. Columns whose cells are all NaN or whitespace are
skipped. The first such query per table prints
`[SEMANTIC][QUERY-ENCODE] approach=<a>/<idx> table=<id> cols=<n> skipped=[...]` on stderr,
loads the encoder (kept for the process lifetime) and caches the vectors per
`(table id, column, cell content)`.

- Snoopy needs `BLEND_SNOOPY_FASTTEXT_PATH` (≈7 GB `cc.en.300.bin`, loaded once) and the
  lake's `csvs/` directory; DeepJoin needs the checkpoint and `BLEND_DEEPJOIN_NLTK_PATH`.
  LIFTus raises `NotImplementedError` (see the last bullet).
- Cells are stringified as the index build saw them: NaN becomes `""` for Snoopy and
  `"nan"` for DeepJoin; numeric dtypes stringify as pandas renders them (`1.0`, not `1`),
  so pass the frame as `pd.read_csv` reads it rather than a re-typed copy.
- Preparation matches the index build: per-table row cap and per-cell character cap from
  the Snoopy checkpoint sidecar, DeepJoin's frequency-sorted sentence with the 512-token
  cap. The lake-wide parallel passes of the baselines' own preprocessing are not part of
  the query path; the stored vectors came from the same sequential adapter code.
- Parity with the stored vector: Snoopy is deterministic given the sidecar caps. DeepJoin's
  sentence lists values by frequency, and the order among equal-frequency values follows the
  platform's sort (the lake caches were built on linux); re-encoding an enrolled column on
  another platform therefore reproduces the stored vector only when its value frequencies
  have no ties within the 512-token cut (observed on opendata-split_13, arm64: cosine 1.0000
  for tie-free columns, down to 0.81 for an all-unique 122k-row column). Blend's CSV
  fallback at index build time has the same property.
- `cost()` and `ml_cost` do not include encode time.
- **LIFTus: not implemented, and what it would take.** LIFTus is a batch preprocessing
  pipeline, not a per-column encoder: a column vector is assembled from five aspects
  (statistic, word, number, paragraph, pattern) that `liftus` computes over a whole dataset
  into pickles, and two of them (statistic, pattern) are sampled without a seed, so a vector
  encoded at query time is a different random draw from the stored one and cannot be
  verified against it. Every benchmark query table is a lake member, where the stored
  aspect vectors are the representation, and the released lakes ship exactly that (Net
  checkpoint + aspect pickles). A query-time path would need: the pattern-encoder checkpoint
  `<train_dataset>_pattern_encoder.pth` (a by-product of the LIFTus training preprocessing,
  not in the Zenodo lakes; on the server under
  `runs/<train_run>/liftus_layout/step_result/pattern_model/`), fastText
  `wiki-news-300d-1M.vec`, the `bert-base-uncased` directory and the nltk `words` corpus
  (new `BLEND_LIFTUS_*` environment variables and compose mounts); `row_cap`,
  `row_cap_seed` and `cell_char_cap` recorded in the LIFTus sidecar (JUDIT capped the LIFTus
  input tables the same way as Snoopy's); and glue in `LiftusAdapter.encode_one(cells=…)`
  that caps the table, round-trips it through CSV text so dtypes match preprocessing
  (`lineterminator="\n"` reads for statistic/tokens/number/word/pattern, `engine="python"`
  for paragraph), and per column calls `column_statistic(…, 200, en_words)`,
  `getWordsCount` (top 64 strings, 512 numbers), `number_encoder_by_column(…, 128)`,
  `word2vec_fasttext` per token, `col_paragraph2vectors(…, 32, "berttokenizer")` followed by
  BERT `last_hidden_state.mean(dim=1)` over all 32 padded positions (as
  `paragraph_encoder_batch` does; `convertFunc_bert` differs), and `getColPatterns(…, 10, 10)`
  → `string_to_onehot_matrix` → `SiameseNetwork(98, 128).inference`, seeding the two sampled
  aspects per table, then the existing `LiftusAdapter` reducers and `Net.inference`.
- Benchmark and Optimizer tooling additionally pins `query_encode = off` in the seekers it
  builds (`tests/test_query_encode_bench_guard.py` enforces it), so a config that opts in does
  not change what they measure: a query column that is not indexed is skipped and counted,
  never encoded.

## Optimizer

`python -m src.Optimizer.cli calibrate --lake <name>` runs the cost-model stages in
order. Profiles (`costs.json`, `semantic_cost_model.json`, XGBoost models) are keyed by
`<dbms>-<vector_backend>-<layout>`, so build them on the backend and layout you run.
`fit-semantic-cost` reports `ceiling_r2` (from `sweep-semantic --repeats` replicates)
next to `r2`. Read `r2` against that ceiling, not against 1. `r2` is scored on a
held-out split and the ceiling over all rows, so `r2` can sit slightly above it.

## Benchmark extras

- `scripts/bench_sc_seeker_metrics.py` runs SC variants that `bench` does not: the
  `Counter` combiner, `--paper-mode` (SC at k = 10·k, one sub-run per evaluation k, as in
  the Blend paper) and the full Blend union plan. `--nonnumeric-only` restricts queries,
  candidates and ground truth to the non-numeric columns the SHO index holds, so SC is
  compared with SHO on the same columns; it needs the SHO index and skips the other
  variants.

## Tracked data and dependencies

The only tracked data files are test fixtures and upstream figures:

| Path | Content | Origin |
| --- | --- | --- |
| `tests/fixtures/semantic/csvs/` | 3 lake tables | the opendata lake (Singapore government open data) |
| `tests/fixtures/semantic/aspects/` | LIFTus aspect features for those tables | sliced by `scripts/build_semantic_test_fixture.py` |
| `tests/fixtures/semantic/ckpt/` | LIFTus checkpoint + sidecar | the opendata checkpoint from the Zenodo archives |
| `tests/fixtures/semantic_join/ckpt/` | random 4-dim Snoopy model | `tests/fixtures/semantic_join/build_fixture.py` |
| `tests/fixtures/semantic_join/{target.csv,csvs/}` | 5 one-column tables | hand-written |
| `images/` | paper and user-study figures | upstream |

`pyproject.toml` and `uv.lock` define the dependencies. The encoder packages `liftus` and
`snoopy` are installed from the judit repository at tag `v1.0.0` (`baselines/liftus`,
`baselines/snoopy`), so `uv sync` needs `git` and network access the first time.

## Upstream material

Kept from upstream Blend and not maintained by this fork: the Vertica ingest
`scripts/create_index.py`, the task plans in `src/Tasks/` (except `UnionCounterSearch.py`), the paper benchmarks in
`benchmarks/`, the confidence interval calculation (in `confidence_interval.md`) and the user study (`UserStudy.md`, `images/`). Upstream's own README is
in the [upstream repository](https://github.com/LUH-DBS/Blend).
