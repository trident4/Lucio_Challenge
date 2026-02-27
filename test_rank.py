import asyncio
from app.extraction.fetcher import fetch_corpus, unzip_to_tuples
from app.extraction.workers import run_extraction
from app.search.indexer import build_index
from app.search.retriever import search_all

async def run():
    z = await fetch_corpus('/Users/chetan/Downloads/Testing Set/Archive.zip')
    chunks, _ = run_extraction(unzip_to_tuples(z))
    
    target_cid = None
    for c in chunks:
        if '40.6' in c['content']:
            target_cid = c['chunk_id']
            break
            
    index = build_index(chunks)
    qs = [{'id': 'q1', 'text': 'What are the revenue figures for Meta for Q1, Q2 and Q3?'}]
    res = await search_all(index, qs, 150)
    
    hits = res['q1']
    target_hit = next((h for h in hits if h["chunk_id"] == target_cid), None)
    if target_hit:
        print(f"BM25 Rank: {hits.index(target_hit) + 1}")
    else:
        print("Not in top 150 BM25")

asyncio.run(run())
