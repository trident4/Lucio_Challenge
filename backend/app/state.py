"""Global in-memory state shared across the request lifecycle.

Includes corpus cache to avoid re-extracting/re-indexing the same
documents on repeated requests with the same corpus URL.
"""

import hashlib
import numpy as np

# chunk_id -> 256-dimensional numpy vector
vector_cache: dict[str, np.ndarray] = {}

# Per-document metadata collected during extraction (filename, title, type, page_count)
doc_metadata: list[dict] = []

# ── Corpus Cache ────────────────────────────────────────────────────────────
# Keyed by corpus hash. Caches chunks, index, metadata, and embeddings
# so that repeated requests with the same corpus skip Phases 1-3.

_corpus_cache: dict[str, dict] = {}


def corpus_hash(corpus_url: str, zip_bytes) -> str:
    """Compute a stable hash for corpus identity."""
    h = hashlib.sha256()
    h.update(corpus_url.encode())

    # zip_bytes is io.BytesIO
    buf = zip_bytes.getbuffer()
    h.update(str(buf.nbytes).encode())

    # Hash first 4KB + last 4KB for speed (don't hash entire zip)
    h.update(buf[:4096])
    if buf.nbytes > 4096:
        h.update(buf[-4096:])

    return h.hexdigest()[:16]


def get_cached_corpus(key: str) -> dict | None:
    """Get cached corpus data if available."""
    return _corpus_cache.get(key)


def set_cached_corpus(key: str, data: dict) -> None:
    """Store corpus data in cache. Only keeps one corpus cached."""
    _corpus_cache.clear()  # Only cache latest corpus (memory safety)
    _corpus_cache[key] = data
