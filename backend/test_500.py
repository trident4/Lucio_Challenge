import asyncio
import sys
import traceback
from fastapi.testclient import TestClient
from app.main import app


def run_test():
    with TestClient(app) as client:
        payload = {
            "corpus_url": "/Users/chetan/Downloads/Testing Set/Archive.zip",
            "questions": [{"id": "q1", "text": "What is the revenue?"}],
        }

    print("=== FIRST REQUEST (Cache Miss) ===")
    try:
        resp1 = client.post("/challenge/run", json=payload)
        print("Status:", resp1.status_code)
        if resp1.status_code != 200:
            print(resp1.text)
            return
    except Exception as e:
        print("Exception during first request:")
        traceback.print_exc()
        return

    print("\n=== SECOND REQUEST (Cache Hit) ===")
    try:
        resp2 = client.post("/challenge/run", json=payload)
        print("Status:", resp2.status_code)
        if resp2.status_code != 200:
            print(resp2.text)
    except Exception as e:
        print("Exception during second request:")
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
