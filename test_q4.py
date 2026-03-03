import asyncio
import json
import httpx
import logging

logging.basicConfig(level=logging.INFO)


async def test_q4():
    url = "http://127.0.0.1:8000/challenge/run"
    payload = {
        "corpus_url": "/Users/chetan/Downloads/Testing Set/Archive.zip",
        "questions": [
            {"id": "q4", "text": "What was the bench in the Eastman Kodak Case?"}
        ],
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, json=payload)
        data = resp.json()
        print("====== LLM ANSWER ======")
        print(data["results"][0]["answer"])
        print("\n====== SOURCES ======")
        for s in data["results"][0].get("sources", []):
            print(s["filename"])


if __name__ == "__main__":
    asyncio.run(test_q4())
