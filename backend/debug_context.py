import requests
import json

req = {
    "corpus_url": "/Users/chetan/Downloads/Testing Set/Archive.zip",
    "questions": [{"id": "q1", "text": "What are the revenue figures for Meta for Q1, Q2 and Q3?"}]
}
resp = requests.post("http://127.0.0.1:8000/challenge/run", json=req)
data = resp.json()
print(data["results"][0]["answer"])
# We don't return context in the challenge response, but we can look at the logs
