import asyncio
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    # First request should be cache MIS
    print("=== FIRST REQUEST ===")
    response1 = client.post("/challenge/run", json={
        "corpus_url": "/Users/chetan/Downloads/Testing Set/Archive.zip",
        "questions": [{"id": "q1", "text": "What are the revenue figures for Meta for Q1, Q2 and Q3?"}]
    })
    print(response1.status_code)
    
    # Second request should be cache HIT and crash?
    print("=== SECOND REQUEST ===")
    response2 = client.post("/challenge/run", json={
        "corpus_url": "/Users/chetan/Downloads/Testing Set/Archive.zip",
        "questions": [{"id": "q1", "text": "What are the revenue figures for Meta for Q1, Q2 and Q3?"}]
    })
    print(response2.status_code)
    if response2.status_code != 200:
        print(response2.json())
