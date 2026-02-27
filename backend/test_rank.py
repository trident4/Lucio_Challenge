import asyncio
from app.extraction.fetcher import fetch_corpus, unzip_to_tuples
from app.extraction.workers import run_extraction
from app.search.indexer import build_index
from app.search.retriever import search_all


async def run():
    z = await fetch_corpus("/Users/chetan/Downloads/Testing Set/Archive.zip")
    file_tuples = unzip_to_tuples(z)

    # Run extraction in executor since it's sync
    loop = asyncio.get_event_loop()
    chunks, _ = await loop.run_in_executor(None, run_extraction, file_tuples)

    target_cid = None
    for c in chunks:
        if "40.6" in c["content"]:
            target_cid = c["chunk_id"]
            print(f"Target CID: {target_cid}")
            break

    if not target_cid:
        print("40.6 not found in any chunk!")
        return

    index = build_index(chunks)
    qs = [
        {"id": "q1", "text": "What are the revenue figures for Meta for Q1, Q2 and Q3?"}
    ]
    res = await search_all(index, qs, 150)

    hits = res["q1"]
    target_hit = next((h for h in hits if h["chunk_id"] == target_cid), None)
    if target_hit:
        print(
            f"BM25 Rank: {hits.index(target_hit) + 1} (Score: {target_hit['bm25_score']})"
        )
    else:
        print("Not in top 150 BM25")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        import traceback

        traceback.print_exc()
