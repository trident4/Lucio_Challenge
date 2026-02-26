"""Global in-memory state shared across the request lifecycle."""

import numpy as np

# chunk_id -> 256-dimensional numpy vector
vector_cache: dict[str, np.ndarray] = {}

# Per-document metadata collected during extraction (filename, title, page_count)
doc_metadata: list[dict] = []
