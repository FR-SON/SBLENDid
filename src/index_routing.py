import re

_IN_RE = re.compile(r"(?<!NOT\s)TableId\s+IN\s*\(", re.IGNORECASE)
# (?<![\w.]) skips qualified joins such as `categorical.TableId = numerical.TableId`.
_EQ_RE = re.compile(r"(?<![\w.])tableid\s*=\s*[^=]", re.IGNORECASE)
_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_CODE_RE = re.compile(r"\bsimhash_code\b", re.IGNORECASE)


def route_suffix(query: str) -> str:
    """'tableid' if the query has a tableid-prunable predicate outside string literals, else 'token'."""
    bare = _LITERAL_RE.sub("''", query)
    if _IN_RE.search(bare) or _EQ_RE.search(bare):
        return "tableid"
    return "token"


def physical_table(base: str, layout: str, query: str) -> str:
    """Resolve the physical index table for a query under the given layout."""
    if _CODE_RE.search(_LITERAL_RE.sub("''", query)):
        return f"{base}_code"
    if layout == "two_table":
        return f"{base}_{route_suffix(query)}"
    return base
