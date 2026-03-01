import asyncio
import os
from openai import AsyncOpenAI
import tiktoken
from app.config import Settings
from app.schemas import ChallengeRequest, Question
from app.extraction.fetcher import fetch_corpus, unzip_to_tuples
from app.extraction.workers import run_extraction
from app.search.indexer import build_index
from app.search.retriever import search_all
from app.embeddings.embedder import embed_and_cache, embed_questions
from app.reranker.reranker import rerank_all
from app.state import doc_metadata, vector_cache
from app.llm.inference import _build_user_prompt, SYSTEM_PROMPT


async def count_tokens():
    settings = Settings()
    client = AsyncOpenAI(
        base_url=settings.mac_studio_base_url,
        api_key=settings.mac_studio_api_key,
    )

    # Fake up Q4 request using the correct remote corpus URL
    req = ChallengeRequest(
        corpus_url="/Users/chetan/Downloads/Testing Set/Archive.zip",
        questions=[
            Question(id="q4", text="What was the bench in the Eastman Kodak Case?")
        ],
    )

    # Phases 1-4 (run quickly to get the exact context)
    vector_cache.clear()
    doc_metadata.clear()
    zip_bytes = await fetch_corpus(req.corpus_url)
    file_tuples = unzip_to_tuples(zip_bytes)
    chunks, metadata = run_extraction(file_tuples)
    doc_metadata.extend(metadata)
    index = build_index(chunks)
    search_results = await search_all(index, req.questions, settings.bm25_top_k)
    await embed_and_cache(client, search_results, vector_cache, settings)
    q_vectors = await embed_questions(client, req.questions, settings)
    reranked = rerank_all(
        q_vectors, search_results, vector_cache, settings.rerank_top_k
    )

    # Extract the exact prompt we would send to Qwen
    context = reranked.get("q4", {}).get("context", "")
    user_prompt = _build_user_prompt(req.questions[0].text, context, doc_metadata)

    full_text = SYSTEM_PROMPT + "\n\n" + user_prompt

    # Calculate sizes
    char_len = len(full_text)

    # tiktoken 'cl100k_base' is standard for OpenAI, very close to Qwen's tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(full_text)

    print("\n" + "=" * 50)
    print(f"PAYLOAD ANALYSIS FOR Q4")
    print("=" * 50)
    print(f"Characters Sent to LLM: {char_len:,}")
    print(f"Estimated Tokens (cl100k): {len(tokens):,}")
    print("=" * 50)

    # Print the structure
    header_count = full_text.count("[HEADER")
    source_count = full_text.count("[SOURCE")
    print(f"Number of distinct chunks/headers in prompt: {source_count}")
    print(
        f"Average tokens per chunk: {len(tokens) // source_count if source_count else 0}"
    )
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(count_tokens())
