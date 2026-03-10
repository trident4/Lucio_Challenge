"""FastAPI application — the 30-second orchestrator.

Single endpoint POST /challenge/run wires all phases together
with per-phase wall-clock timing.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from app.assembly.assembler import assemble_response
from app.config import Settings
from app.embeddings.embedder import embed_and_cache, embed_questions
from app.extraction.fetcher import fetch_corpus, unzip_to_tuples
from app.extraction.workers import run_extraction
from app.llm.inference import run_inference
from app.reranker.reranker import rerank_all
from app.reranker.compressor import compress_context
from app.schemas import ChallengeRequest, ChallengeResponse
from app.search.indexer import build_index
from app.search.retriever import search_all
from collections import Counter

from app.state import vector_cache, corpus_cache, corpus_lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lucio")

# ── Timing helper ───────────────────────────────────────────────────────────


def log_phase(name: str, t: list[float]) -> None:
    """Append current time, log delta from previous checkpoint."""
    now = time.perf_counter()
    logger.info(f"{name}: {now - t[-1]:.3f}s (total: {now - t[0]:.3f}s)")
    t.append(now)


# ── App lifecycle ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load settings, create OpenAI client, probe embedding API."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    settings = Settings()
    app.state.settings = settings

    # 1. Always create Mac Studio client
    mac_studio_client = AsyncOpenAI(
        base_url=settings.mac_studio_base_url,
        api_key=settings.mac_studio_api_key,
    )

    # 2. Create OpenRouter client if key exists
    openrouter_client = None
    if settings.openrouter_api_key:
        openrouter_client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

    # 3. Route embedding client based on provider setting
    if settings.embedding_provider == "openrouter" and openrouter_client:
        app.state.embed_client = openrouter_client
        logger.info("Embedding → OpenRouter (%s)", settings.embedding_model)
    else:
        app.state.embed_client = mac_studio_client
        logger.info("Embedding → Mac Studio (%s)", settings.embedding_model)

    # 4. Route LLM client (OpenRouter override or fallback to local)
    if openrouter_client:
        app.state.llm_client = openrouter_client
        logger.info("LLM → OpenRouter (%s)", settings.llm_model)
    else:
        app.state.llm_client = mac_studio_client
        logger.info("LLM → Mac Studio (%s)", settings.llm_model)

    # Probe dimensions param support on the EMBEDDING client
    try:
        await app.state.embed_client.embeddings.create(
            input=["probe"],
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        logger.info("Embedding API supports dimensions param ✓")
    except Exception:
        settings.supports_dimensions_param = False
        logger.warning(
            "Embedding API: dimensions param unsupported → will truncate manually"
        )

    logger.info("Lucio ready")
    yield


app = FastAPI(title="Lucio Speedrun", lifespan=lifespan)

# Mount the static directory for the testing UI at /ui/
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/settings")
async def get_settings(request: Request):
    """Expose non-secret settings for eval metadata."""
    s: Settings = request.app.state.settings
    return {
        "embedding_model": s.embedding_model,
        "embedding_provider": s.embedding_provider,
        "embedding_dimensions": s.embedding_dimensions,
        "llm_model": s.llm_model,
        "bm25_top_k": s.bm25_top_k,
        "rerank_top_k": s.rerank_top_k,
        "llm_max_tokens": s.llm_max_tokens,
        "llm_temperature": s.llm_temperature,
    }


@app.post("/challenge/run", response_model=ChallengeResponse)
async def challenge_run(req: ChallengeRequest, request: Request):
    """Execute the full 30-second RAG pipeline."""
    embed_client: AsyncOpenAI = request.app.state.embed_client
    llm_client: AsyncOpenAI = request.app.state.llm_client
    settings: Settings = request.app.state.settings
    t = [time.perf_counter()]

    # ── Phase 1 & 2: Fetch, Extract, Index ─────────────────────────────
    # Lock globally to prevent 15 simultaneous 100MB downloads
    async with corpus_lock:
        if req.corpus_url in corpus_cache and not req.bypass_cache:
            cache_entry = corpus_cache[req.corpus_url]
            chunks = cache_entry["chunks"]
            metadata = cache_entry["metadata"]
            index = cache_entry["index"]
            log_phase("Phase 1+2: Cached Index", t)
        else:
            zip_bytes = await fetch_corpus(req.corpus_url)
            file_tuples = unzip_to_tuples(zip_bytes)
            loop = asyncio.get_event_loop()
            chunks, metadata = await loop.run_in_executor(
                None, run_extraction, file_tuples
            )
            type_dist = Counter(m.get("type", "?") for m in metadata)
            logger.info(f"Extraction: {len(metadata)} docs, types: {dict(type_dist)}")
            log_phase("Phase 1: Extract", t)

            index = build_index(chunks)
            log_phase("Phase 2: Index", t)

            corpus_cache[req.corpus_url] = {
                "chunks": chunks,
                "metadata": metadata,
                "index": index,
            }

    # ── Phase 3: Retrieve + Embed ───────────────────────────────────────
    search_results = await search_all(index, req.questions, settings.bm25_top_k)
    vec_cache_before = len(vector_cache)
    _, q_vectors = await asyncio.gather(
        embed_and_cache(embed_client, search_results, vector_cache, settings),
        embed_questions(embed_client, req.questions, settings),
    )
    log_phase("Phase 3: Retrieve+Embed", t)

    # ── Phase 4: Rerank ─────────────────────────────────────────────────
    actual_top_k = req.rerank_top_k or settings.rerank_top_k
    reranked = rerank_all(
        req.questions, q_vectors, search_results, vector_cache, chunks, actual_top_k
    )
    log_phase("Phase 4: Rerank", t)

    # ── Phase 4.5: Compress (BYPASSED) ──────────────────────────────────
    compressed = reranked
    log_phase("Phase 4.5: Compress (Bypassed)", t)

    # ── Phase 5: LLM Inference ──────────────────────────────────────────
    llm_answers, total_tokens = await run_inference(
        llm_client,
        req.questions,
        compressed,
        metadata,
        settings,
        model_override=req.llm_model,
    )
    log_phase("Phase 5: LLM", t)

    # ── Phase 6: Assemble ───────────────────────────────────────────────
    response = assemble_response(req.questions, llm_answers, reranked)

    total_time = t[-1] - t[0]
    response.total_time = round(total_time, 3)
    response.total_tokens = total_tokens
    response.cache_hit = vec_cache_before > 0

    log_phase("Phase 6: Assemble", t)

    logger.info(
        f"🏁 Total time: {total_time:.3f}s {'✓ UNDER 30s' if total_time < 30 else '⚠ OVER 30s!'}"
    )
    return response
