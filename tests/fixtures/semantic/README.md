# Semantic seeker test fixture

Slice of the dev-local LIFTus reference artifacts used by tests under
`tests/test_semantic_*.py`. Source of truth is
`scripts/build_semantic_test_fixture.py`.

Rebuild after any change in the dev-local reference dir:

    python scripts/build_semantic_test_fixture.py --reference-dir _reference

The script filters each aspect pickle to the 3 sample tables only.
