# Cold Batch Optimization: 1GB Corpus < 30s

## Context

Stress test on ~1GB corpus (68 files, 5168 PDF pages), 15 questions:
- **Batch cold: >600s TIMEOUT**
- Batch cached: 16.5s PASS (extract=0, embed=2.4, rerank=0.1, llm=13.9)

### Benchmarked Reality (corrected from initial estimates)

Extraction is **NOT** the bottleneck. Benchmarked on stress corpus:
- `get_text("layout")`: 8.3ms/page → **43s single-thread, ~5s with 8 cores**
- `get_text()`: 6.3ms/page → **33s single-thread, ~4s with 8 cores**
- Difference: only 1.3x (not 4x as initially assumed)

### Root Cause: Embedding Phase (~90% of cold time)

With `bm25_top_k=150` + dual search (primary + entity), each question retrieves up to 300 chunks. Across 15 questions, after deduplication: **~1500-2000 unique chunks** need cold embedding.

At current settings (`batch_size=20`, `concurrency=5`):
- 1500 chunks / 20 = 75 batches → 15 rounds of 5 concurrent API calls
- Each OpenRouter/Mac Studio embedding call: ~1-2s
- **Total: 15-30 rounds x ~2s = 30-60s+ for embedding alone**

Secondary factors causing the >600s timeout:
- `fetch_corpus` reads entire 1GB into `BytesIO` → peak ~3.4GB → possible **swap on 8GB Mac**
- ProcessPoolExecutor recreated per-request (macOS `spawn` start method) → 2-3s startup
- `unzip_to_tuples` and `build_index` block the event loop

## Changes (priority order)

### 1. [CRITICAL] Reduce `bm25_top_k` 150 → 50 — cuts embedding volume ~3x

**File:** `backend/app/config.py:29`

```python
bm25_top_k: int = 50   # was 150
```

With `rerank_top_k=5`, 50 BM25 candidates per query is more than sufficient. With dual search (primary + entity), effective max is 100 per question. After dedup across 15 questions: **~400-600 unique chunks** (down from ~1500-2000).

**Risk:** Potential recall drop if relevant chunks rank 51-150 in BM25. Mitigated by dual search (entity query catches vocabulary mismatches). Verify with eval.

### 2. [CRITICAL] Increase embedding throughput — 5x faster batch processing

**File:** `backend/app/config.py:33-34`

```python
embedding_batch_size: int = 100   # was 20
embedding_concurrency: int = 10   # was 5
```

With ~500 chunks: 500 / 100 = 5 batches → 1 round of 5 concurrent calls → **~2s** (down from 30-60s+).

**Risk:** Larger batches may hit API rate limits or payload size limits. OpenRouter/text-embedding-3-large should handle 100 texts fine (each ~2K chars x 100 = ~200K chars, well within limits).

### 3. [HIGH] Eliminate BytesIO for local files — prevents swap

**File:** `backend/app/extraction/fetcher.py`

Currently `fetch_corpus` reads the ENTIRE 1GB file into `io.BytesIO(f.read())`. Then `unzip_to_tuples` reads all entries into another ~1GB of tuples. Peak: ~3.4GB before extraction even starts → swap risk on 8GB Mac.

Fix: for local files, pass the path directly. `zipfile.ZipFile(path)` reads entries on demand from disk.

```python
async def fetch_corpus(corpus_url: str) -> str | io.BytesIO:
    """Return file path (local) or BytesIO (remote)."""
    if os.path.exists(corpus_url):
        return corpus_url  # No memory copy

    # Remote: download to temp file instead of holding in memory
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.get(corpus_url)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name
```

Update `unzip_to_tuples` to accept `str | io.BytesIO`:
```python
def unzip_to_tuples(source: str | io.BytesIO) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(source, "r") as zf:
        # ... same logic ...
```

Peak memory drops from ~3.4GB to ~2.2GB.

### 4. [MEDIUM] Persistent ProcessPoolExecutor — saves 2-3s

**File:** `backend/app/state.py`

```python
from concurrent.futures import ProcessPoolExecutor
import os

process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())
```

**File:** `backend/app/extraction/workers.py` — accept pool param

```python
def run_extraction(file_tuples, pool=None):
    _pool = pool or ProcessPoolExecutor(max_workers=os.cpu_count())
    try:
        results = list(_pool.map(_extract_document_wrapper, file_tuples))
    finally:
        if not pool:
            _pool.shutdown(wait=False)
```

### 5. [MEDIUM] Combine sync work in single executor call — unblocks event loop

**File:** `backend/app/main.py`

All sync work (unzip, extract, index) runs in one `run_in_executor` call:

```python
def _extract_pipeline(corpus_source, pool):
    """All sync work in one executor call."""
    import time
    t0 = time.perf_counter()
    file_tuples = unzip_to_tuples(corpus_source)
    t1 = time.perf_counter()
    chunks, metadata = run_extraction(file_tuples, pool)
    del file_tuples  # Free ~1GB
    t2 = time.perf_counter()
    index = build_index(chunks)
    t3 = time.perf_counter()
    return chunks, metadata, index, {
        "unzip": round(t1-t0,3), "extract": round(t2-t1,3), "index": round(t3-t2,3)
    }
```

### 6. [MEDIUM] Start question embedding during extraction — overlaps ~0.5s

**File:** `backend/app/main.py`

```python
q_embed_task = asyncio.create_task(
    embed_questions(embed_client, req.questions, settings)
)
# extraction runs in executor → event loop free → q_embed_task runs concurrently
async with corpus_lock:
    # ... extraction pipeline ...

search_results = await search_all(...)
q_vectors = await q_embed_task   # Already done
await embed_and_cache(...)       # Chunk embedding only
```

### 7. [LOW] `get_text("layout")` → `get_text()` — saves ~1s

**File:** `backend/app/extraction/workers.py:115`

```python
page_texts.append(page.get_text())  # was get_text("layout")
```

Only 1.3x speedup (benchmarked). Saves ~1s on 8-core extraction.

**Risk:** May lose table column alignment in financial PDFs. Since hackathon docs have same table formats as current corpus, test first: run eval with `get_text()` and check financial question scores. If they drop, revert to `get_text("layout")` — the 1s saving isn't worth quality loss.

## Estimated Timeline

| Phase | Before | After | Saving | Notes |
|-------|--------|-------|--------|-------|
| Fetch + unzip | 1-2s (1GB BytesIO) | **0s** (path) | ~2s | Change #3 |
| Pool startup | 2-3s | **0s** | ~2s | Change #4 |
| Extract | ~5s | ~4-5s | ~1s | Change #7 (optional) |
| Index | 1-2s | 1-2s | 0 | In executor now (Change #5) |
| Search | 0.5s | 0.5s | 0 | |
| **Embed chunks** | **30-60s+** | **~2s** | **28-58s** | Changes #1 + #2 (THE FIX) |
| Embed questions | 0.5s | **0s** | ~0.5s | Change #6 (overlapped) |
| Rerank | 0.1s | 0.1s | 0 | |
| LLM | 13-14s | 13-14s | 0 | API-bound |
| **Total** | **>600s** | **~20-24s** | | |

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `backend/app/config.py` | bm25_top_k=50, batch=100, concurrency=10 | CRITICAL |
| `backend/app/extraction/fetcher.py` | Return path for local, temp file for remote | HIGH |
| `backend/app/main.py` | Combined executor, early q embed, process_pool import | MEDIUM |
| `backend/app/state.py` | Add `process_pool` | MEDIUM |
| `backend/app/extraction/workers.py` | Accept pool param. Optionally `get_text()` | MEDIUM/LOW |

## Implementation Order

1. Config changes (#1, #2) — instant, no structural change
2. Restart server, run stress test cold — expect dramatic improvement
3. Run eval (`python3 eval/run_eval_batch.py --battle`) — verify 23/23
4. If eval passes: implement structural changes (#3-#6)
5. Test `get_text()` (#7) last — only if we need the extra 1s

## Verification

1. **Quality:** `python3 eval/run_eval_batch.py --battle` — must stay 23/23
   - If drops from `bm25_top_k=50` → try 80
   - If drops from `get_text()` → revert to `get_text("layout")`
2. **Cold stress:** Restart server → `python3 eval/run_stress_test.py` — target <30s
3. **Cached regression:** `python3 eval/run_stress_test.py --skip-cold` — must stay <17s
4. **Memory:** Monitor with `ps -o rss=` during cold run — should stay <3GB

## Safety Net

If cold lands at 31-35s after all changes, warmup protocol ensures hackathon hits cached path (7-16s):
```bash
python3 eval/run_stress_test.py --warmup --skip-cold --corpus <EXACT_URL>
```
