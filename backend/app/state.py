"""Global in-memory state shared across the request lifecycle."""

import asyncio
import os
import signal
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _pool_init():
    """Worker initializer: ignore SIGINT so only the main process handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# chunk_id -> 256-dimensional numpy vector
vector_cache: dict[str, np.ndarray] = {}

# Global lock to prevent simultaneous redundant downloads/parsing
corpus_lock = asyncio.Lock()

# corpus_url -> {"chunks": list, "metadata": list, "index": tantivy_index}
corpus_cache: dict[str, dict] = {}

# Persistent process pool — avoids 2-3s spawn overhead per request on macOS
process_pool = ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_pool_init)
