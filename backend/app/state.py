"""Global in-memory state shared across the request lifecycle."""

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# chunk_id -> 256-dimensional numpy vector
vector_cache: dict[str, np.ndarray] = {}

# Global lock to prevent simultaneous redundant downloads/parsing
corpus_lock = asyncio.Lock()

# corpus_url -> {"chunks": list, "metadata": list, "index": tantivy_index}
corpus_cache: dict[str, dict] = {}

# Persistent process pool — avoids 2-3s spawn overhead per request on macOS
process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())
