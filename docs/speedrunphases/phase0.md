# Phase 0: System Orchestration & State Management

## 📌 Objective

Establish the bare-metal Python environment, set up the FastAPI entry point, define the exact Pydantic schemas for I/O validation, and establish the global in-memory state required to survive the 30-second constraint.

## 🏗 Architectural Requirements

### 1. Environment Setup (`requirements.txt`)

Create a `requirements.txt` file in the root directory. You must use these exact versions to ensure compatibility and maximum bare-metal performance on Apple Silicon. Do not generate a Dockerfile; this must run natively in a virtual environment.

```text
fastapi==0.110.0
uvicorn==0.27.1
httpx==0.27.0
PyMuPDF==1.23.26
python-docx==1.1.0
tantivy==0.21.0
numpy==1.26.4
openai==1.14.0
pydantic==2.6.4
```

### 2. Global State (`main.py`)

You must define a global, in-memory cache dictionary at the top of the application to prevent redundant vectorization.

```python
# Stores chunk_id -> 256-dimensional numpy array or list
global_vector_cache = {}
```

### 3. Pydantic Schemas (`schemas.py` or `main.py`)

Define the following strict schemas for the API.

- **Input:**
  - `Question`: `id` (str), `text` (str)
  - `ChallengeRequest`: `corpus_url` (str), `questions` (List[Question])
- **Output:**
  - `Source`: `filename` (str), `pages` (List[int])
  - `Answer`: `question_id` (str), `answer` (str), `sources` (List[Source])
  - `ChallengeResponse`: `results` (List[Answer])

### 4. FastAPI Endpoint (`main.py`)

- Create a single `POST` endpoint: `/challenge/run`.
- It must accept the `ChallengeRequest` schema.
- It must track execution time: `start_time = time.time()` at the very beginning, logging the total duration before returning the `ChallengeResponse`.
