import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from app.config import Settings
from app.search.indexer import build_index
from app.search.retriever import search_all
from app.embeddings.embedder import embed_questions, embed_and_cache
from app.reranker.reranker import rerank_all
from app.extraction.fetcher import fetch_corpus, unzip_to_tuples
from app.extraction.workers import run_extraction
from openai import AsyncOpenAI
import json

async def debug():
    try:
        settings = Settings()
        client = AsyncOpenAI(base_url=settings.mac_studio_base_url, api_key=settings.mac_studio_api_key)
        
        corpus_url = "/Users/chetan/Downloads/Testing Set/Archive.zip"
        print(f"Fetching corpus: {corpus_url}")
        zip_bytes = await fetch_corpus(corpus_url)
        print(f"Unzipping...")
        tuples = unzip_to_tuples(zip_bytes)
        print(f"Extracting {len(tuples)} files...")
        chunks, metadata = run_extraction(tuples)
        print(f"Building index from {len(chunks)} chunks...")
        index = build_index(chunks)
        
        q_text = "What are the revenue figures for Meta for Q1, Q2 and Q3?"
        class Q:
            def __init__(self, id, text):
                self.id = id
                self.text = text
        questions = [Q("q1", q_text)]
        
        print(f"Retrieving top {settings.bm25_top_k}...")
        search_results = await search_all(index, questions, settings.bm25_top_k)
        vector_cache = {}
        print(f"Embedding and caching chunks...")
        await embed_and_cache(client, search_results, vector_cache, settings)
        print(f"Embedding questions...")
        q_vectors = await embed_questions(client, questions, settings)
        
        print(f"Reranking top {settings.rerank_top_k}...")
        reranked = rerank_all(questions, q_vectors, search_results, vector_cache, settings.rerank_top_k)
        
        print("\n--- Q1 CONTEXT ---")
        print(reranked["q1"]["context"])
        print("--- END CONTEXT ---")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug())
