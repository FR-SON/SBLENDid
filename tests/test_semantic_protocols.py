def test_encoder_base_imports():
    from src.Semantic.encoders.base import (
        SegmentEncoder, SidecarMeta, write_sidecar, read_sidecar,  # noqa: F401
    )
    assert hasattr(SegmentEncoder, "load")


def test_indexer_base_imports():
    from src.Semantic.indexers.base import (
        Indexer, BuiltIndex, IndexerConfig, validate_pq_alignment,  # noqa: F401
    )
    cfg = IndexerConfig()
    assert cfg.quant == "pq"
