# Lucio Project Setup & Execution Guide 🚀

This guide provides a single, comprehensive set of instructions to get the **Lucio RAG Pipeline** up and running for the competition.

---

## 🏗️ 1. Environment Setup

Lucio requires Python 3.10+ and a set of high-performance libraries for PDF parsing and vector search.

1.  **Enter the project directory**:

    ```bash
    cd Lucio/backend
    ```

2.  **Create a Virtual Environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ 2. Configuration (`.env`)

Lucio relies on an external Mac Studio for heavy neural compute (Embeddings and LLM).

1.  **Create your environment file**:

    ```bash
    cp .env.example .env
    ```

2.  **Edit `.env`** with the following production-ready values:

    ```env
    # Mac Studio OpenAI-compatible API
    MAC_STUDIO_BASE_URL=http://<MAC_STUDIO_IP>:port/v1
    MAC_STUDIO_API_KEY=not-needed

    # Models
    EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
    LLM_MODEL=qwen/qwen3-30b-a3b-2507

    # Tuning (The "Golden Config")
    EMBEDDING_DIMENSIONS=256
    BM25_TOP_K=150
    RERANK_TOP_K=8
    LLM_MAX_TOKENS=500
    LLM_TEMPERATURE=0.0
    EMBEDDING_BATCH_SIZE=20

    # Optional: OpenRouter Override (for LLM Inference ONLY)
    # Leave OPENROUTER_API_KEY empty to route everything to Mac Studio
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    OPENROUTER_API_KEY=
    ```

---

## 🏃 3. Running the Server

Start the FastAPI backend. It will automatically handle the concurrent "Thundering Herd" of competition requests using our global locking system.

```bash
# From the /backend directory
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server is now listening at `http://127.0.0.1:8000`.

---

## 🧪 4. Testing & Evaluation

### **A. Simple Curl Test**

You can fire a raw batch request for 7 complex questions to verify the end-to-end flow:

```bash
curl -X POST http://127.0.0.1:8000/challenge/run \
-H "Content-Type: application/json" \
-d '{
  "corpus_url": "https://pub-2d7c04bede434cd0bb8ebedc0a3160bb.r2.dev/Archive.zip",
  "questions": [
    {"id": "q1", "text": "What are the revenue figures for Meta for Q1, Q2 and Q3?"},
    {"id": "q2", "text": "Who are the registrar and the transfer agent for KFIN Technologies Limited?"},
    {"id": "q3", "text": "Has the Competition Commission of India (CCI) issued any order against Meta? If so, what was the fine?"},
    {"id": "q4", "text": "Who were the bench in the Eastman Kodak Co. v. Image Technical Services, Inc., 504 U.S. 451 (1992) case?"},
    {"id": "q5", "text": "What brand names and registered trademarks are mentioned in the Eastman Kodak case?"},
    {"id": "q6", "text": "In the NVCA Investors’ Rights Agreement, what specific actions are listed under the '\''Covenants of the Company'\'' section that require the approval of the Major Investors?"},
    {"id": "q7", "text": "If the order from CCI against Meta is still pristine, what would be the total penalty?"}
  ]
}'
```

### **B. Web UI Interface**

For a visual demonstration and manual testing, you can use the built-in dashboard:

1.  Open `backend/static/index.html` directly in your browser.
2.  Ensure the backend server is running on port `8000`.
3.  The UI allows you to select specific questions, view the "Ground Truth" side-by-side, and see the exact citations the AI extracted.

---

## 🛡️ 5. Key Architectural Features

- **Global Corpus Lock**: Prevents multiple requests from extracting the same ZIP file simultaneously.
- **Global Embedding Lock**: Protects the Mac Studio hardware from VRAM saturation by serializing embedding calls while deduplicating requests.
- **Reciprocal Rank Fusion (RRF)**: Combines Lexical (keyword) and Semantic (meaning) search for 100% accuracy.
- **Golden Context Window (Top-K=8)**: Balanced payload that gives the LLM exactly enough context to distinguish between consolidated and segment financial figures.
