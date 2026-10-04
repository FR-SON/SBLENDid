default_dataset := "santos"

# List available recipes.
default:
    @just --list

# Install python dependencies via uv.
install:
    uv sync

# Run the pytest suite.
test:
    uv run pytest

# Scaffold datasets/<dataset>/ + the approach-specific subtree under semantic/<approach>/<index>/ for the given op (SU|SJ); approach/index come from [Semantic.<op>].
init-dataset dataset=default_dataset op="SU":
    @uv run python -c "from src.Semantic.config import SemanticConfig, SemanticOp; \
    cfg = SemanticConfig.load(overrides={'dataset': '{{dataset}}'}); \
    op_cfg = cfg.operator(SemanticOp['{{op}}']); \
    ad = cfg.approach_dir(op_cfg.approach, op_cfg.index_name); \
    extras = {'liftus': [ad / 'aspects' / a for a in ('statistic', 'paragraph', 'word', 'number', 'pattern')], 'snoopy': []}; \
    paths = [cfg.dataset.dir(), ad / 'ckpt', ad / 'index', *extras.get(op_cfg.approach, [])]; \
    [p.mkdir(parents=True, exist_ok=True) for p in paths]; \
    print('created:'); [print(f'  {p}') for p in paths]; \
    print(f'duckdb -> {cfg.duckdb_path}'); \
    print(f'sidecar -> {cfg.blend_basenames_path}')"

# Drop blend.duckdb for <dataset>.
clean dataset=default_dataset:
    rm -f datasets/{{dataset}}/blend.duckdb datasets/{{dataset}}/blend.duckdb.wal
