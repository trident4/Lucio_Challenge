import json
import urllib.request
import time

payload = {
    "corpus_url": "/Users/chetan/Downloads/Testing Set/Archive.zip",
    "questions": [
        {"id": "q4", "text": "What was the bench in the Eastman Kodak Case?"}
    ],
}

print("Running Q4: 'What was the bench in the Eastman Kodak Case?'")
start = time.time()
req = urllib.request.Request(
    "http://127.0.0.1:8000/challenge/run",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    print(f"Time taken: {time.time() - start:.2f}s")
    parsed = json.loads(data)
    print("\nANSWER:")
    print(parsed["results"][0]["answer"])
except Exception as e:
    print(f"Error: {e}")
    try:
        print(e.read().decode())
    except:
        pass
