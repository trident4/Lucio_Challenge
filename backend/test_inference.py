import asyncio
from openai import AsyncOpenAI
from app.extraction.fetcher import fetch_corpus, unzip_to_tuples
from app.extraction.workers import run_extraction
from app.search.indexer import build_index
from app.search.retriever import search_all
from app.reranker.reranker import rerank_all
from app.config import Settings
from app.schemas import Question


async def run():
    settings = Settings()
    questions = [
        Question(
            id="q7",
            text="Pristine desires to acquire an Indian company having a turnover of INR 1 cr and no assets. Does pristine have to notify the deal to the CCI?",
        )
    ]

    zip_bytes = await fetch_corpus("http://127.0.0.1:8001/Archive.zip")
    files = unzip_to_tuples(zip_bytes)
    chunks, doc_meta = run_extraction(files)

    ix = build_index(chunks)

    client = AsyncOpenAI(api_key="fake", base_url=settings.mac_studio_base_url)

    retrieved = await search_all(client, questions, ix, title_map, settings)
    reranked = rerank_all(questions, retrieved, chunks, settings)

    for qid, data in reranked.items():
        print(f"--- Q7 CONTEXT ---\n{data['context']}\n--- END CONTEXT ---")


if __name__ == "__main__":
    asyncio.run(run())
