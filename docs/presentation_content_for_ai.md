# Project Lucio: Architecture Deep Dive

Subtitle: A 6-Phase RAG Pipeline — 1GB Legal Corpus, 15 Questions, Under 30 Seconds

---

### Slide 1

**Title:** The Legal AI Challenge
**Key Points:**

- **The task:** Given a 1GB zip of legal PDFs/DOCX + 15 questions, return accurate, source-cited answers via a single POST request.
- **The hardware:** MacBook Air with 8GB RAM — no GPU, no beefy server.
- **The deadline:** 30 seconds wall-clock, cold start (no pre-processing).
- **The accuracy bar:** 33 automated assertions — exact revenue figures, SCOTUS bench composition, regulatory reasoning, and anti-hallucination detection.
- **The diversity:** Questions span financial earnings, Supreme Court cases, Indian competition law, and VC legal agreements — all in one corpus.

**Speaker Notes:** "This isn't just a speed challenge or an accuracy challenge — it's both, simultaneously, on consumer hardware. The system must find exact dollar figures like $42.3 billion, name all 9 justices on a SCOTUS bench, reason about Indian antitrust thresholds, AND refuse to answer when data isn't in the corpus. 33 automated assertions verify every run."

---

### Slide 2

**Title:** Split Architecture
**Key Points:**

- **MacBook Air (local):** Handles all CPU-bound work — extraction, indexing, caching.
- **OpenRouter API (cloud):** Handles all neural inference — embeddings via `text-embedding-3-large` (1024d) and LLM via `gpt-4o-mini`.
- **Concurrency everywhere:** All API calls are concurrent — 10 simultaneous embedding batches, 15 simultaneous LLM calls.
- **Cost:** $0.014 per run (15 questions, ~72K tokens total).

**Speaker Notes:** "The split lets us use the best tool for each job — local CPU for parsing 68 documents across all cores, cloud APIs for the neural math. Everything that can run in parallel does run in parallel. The entire run costs less than two cents."

---

### Slide 3

**Title:** Phase 1+2 — Ingestion
**Key Points:**

- **Streaming download:** Remote URLs streamed to temp file at 64KB chunks — ~64KB peak RAM, not 1GB buffered.
- **Multi-core extraction:** Persistent `ProcessPoolExecutor` spawned once at import — distributes PDF/DOCX parsing across all CPU cores. No per-request pool creation overhead.
- **Chunking:** 2,000-character chunks with 200-character overlap for continuous context.
- **Indexing:** Tantivy (Rust-based) builds an in-memory search index over 17K chunks in ~0.5s.

**Speaker Notes:** "The memory optimization is the story here. The original approach loaded the 1GB zip into Python's BytesIO — with Python overhead, that peaked at 3.4GB on an 8GB machine. macOS started swapping, and the run took over 600 seconds. Streaming to a temp file at 64KB chunks fixed it completely — same result, 64KB peak instead of 3.4GB."

---

### Slide 4

**Title:** Phase 3 — Dual BM25 + Embedding
**Key Points:**

- **Two BM25 strategies per question:** Full question text (keyword match) + extracted entities only (proper nouns, acronyms).
- **Why dual?** Single-strategy search missed documents where entity names differed from question phrasing.
- **Results merged** and deduplicated by chunk_id — ~100 candidates per question, ~1,000 unique chunks total.
- **Concurrent embedding:** Batches of 100 chunks, 10 concurrent API calls via `asyncio.Semaphore`. `text-embedding-3-large` at 1024 dimensions.

**Speaker Notes:** "The dual BM25 strategy is one of our key innovations. When you ask 'What was the bench in Eastman Kodak?', a keyword search finds 'Eastman Kodak' easily. But when you ask about CCI antitrust metrics, entity extraction pulls out 'CCI' and 'HHI' as separate search terms, catching documents that the full-text query misses."

---

### Slide 5

**Title:** Phase 4 — Reciprocal Rank Fusion
**Key Points:**

- **RRF (K=60):** Fuses BM25 rank + cosine similarity rank — no tuned weights needed, just ranks.
- **Top 8 chunks selected** per question from the ~100 candidates.
- **Context enrichment:** Each selected chunk gets ±1 neighbor chunks from the same document for continuous context.
- **Document headers** (chunk_0) injected if not already present — grounds the LLM in document identity.

**Speaker Notes:** "RRF is elegant because it doesn't need tuned weights — it just takes the rank position from each scoring method and combines them. This means we don't need to calibrate BM25 scores against cosine similarity scores, which operate on completely different scales. The neighbor chunk injection is critical too — without it, the LLM sometimes gets a chunk mid-paragraph with no context."

---

### Slide 6

**Title:** Anti-Hallucination & Evaluation
**Key Points:**

- **33 automated assertions** across 15 questions — tests exact figures, named entities, legal reasoning, AND "not available" detection.
- **Anti-hallucination test (b2):** "What was the gross margin for Apple Inc. in Q1 2025?" — system must refuse (Apple is not in the corpus). Any fabricated answer = failure.
- **Missing-info test (b4):** "What actions require Major Investor approval in the NVCA IRA?" — information not in the provided document. Must say "not available."
- **Regression testing:** Every code change is validated against the full assertion set.

**Speaker Notes:** "The b2 test is the most important assertion we have. Any RAG system can find data that exists. But a trustworthy system must also know when data is MISSING and say so. If you ask about Apple's gross margin and Apple isn't in the corpus, the system must say 'not available' — not hallucinate a number. We test for this on every single run."

---

### Slide 7

**Title:** Model Selection
**Key Points:**

- Benchmarked 5 models on identical retrieval results.
- **GPT-4o-mini:** 100% accuracy, $0.014/run — selected.
- **Mistral Nemo:** 82.6% accuracy, $0.012/run — missed complex reasoning questions.
- **Claude 3 Haiku:** 65.2% accuracy, $0.029/run — 2x cost, 35% less accurate.
- **Llama 3.1 8B:** 60.9% accuracy, $0.004/run — cheapest but suffered context starvation.
- The accuracy gap is massive: 100% vs next-best 83%.

**Speaker Notes:** "We didn't just pick a model — we proved it. All five models received identical retrieved chunks, so this is a pure generation benchmark. GPT-4o-mini scored perfect on all 23 assertions at the time. The nearest competitor missed nearly 1 in 5. At $0.014 per run, it's not even the most expensive option — Claude 3 Haiku costs twice as much and scores 35 points lower."

---

### Slide 8

**Title:** Phase 5+6 — Generation & Assembly
**Key Points:**

- **15 concurrent LLM calls** via `asyncio.gather` to gpt-4o-mini through OpenRouter.
- **Citation-forcing prompt:** System prompt demands `[Source: filename]` on every fact.
- **Regex source scrubbing:** Engine parses citations — only documents the LLM actually cited appear in the response.
- **Counting/listing guard:** For "how many?" questions, pre-computed document counts injected as ground truth to prevent LLM miscounting.

**Speaker Notes:** "Concurrent firing is critical — sequential calls would take 15 times as long, blowing past the 30-second deadline. The citation forcing is equally important: by requiring the LLM to cite its source for every fact, we can verify which documents actually contributed to the answer. If a document wasn't cited, it gets filtered out of the response — the user sees only what the AI actually used."

---

### Slide 9

**Title:** The Journey — From 527s to 22.3s (23x Speedup)
**Key Points:**

- **The timeline:**
  - Feb 27: 197s — first recorded run (self-hosted Nomic + Qwen-30B, 95.7% accuracy)
  - Mar 1: 188s — first perfect score (23/23, still self-hosted)
  - Mar 2: 527s — sentence compression experiment broke everything (60.9% accuracy)
  - Mar 9: 17.6s — the breakthrough (OpenRouter + concurrent batches + persistent pool, 100%)
  - Mar 11: 22.3s — expanded to 33 assertions, 97% accuracy (32/33)
- **The 5 breakthroughs:**
  1. OpenRouter switch: 200s → 20s (self-hosted → cloud API, same accuracy)
  2. Persistent ProcessPool: −3s (eliminated macOS spawn overhead)
  3. Concurrent embedding batches: embedding from 15s → 3.5s
  4. Early question embedding: −0.5s free (runs during extraction)
  5. Streaming download: memory crash → stable (3.4GB peak → 64KB peak)
- **Final: 97% accuracy, 22.3s, $0.014/run on a MacBook Air**

**Speaker Notes:** "The biggest lesson: no single optimization got us under 30 seconds. It took five independent breakthroughs, each attacking a different bottleneck. The most dramatic was switching from self-hosted models to OpenRouter — same accuracy, 10x faster. But without the memory fix (streaming download), we couldn't even finish on 8GB RAM. And the sentence compression experiment on March 2nd was a humbling reminder — clever ideas can make things 3x worse if you don't measure."

---

**Narrative thread across all slides:** The presentation tells a story of constraints breeding innovation. We couldn't just throw hardware at the problem (8GB RAM). We couldn't just use a bigger model (30-second deadline). Every optimization was a tradeoff — and we measured each one with 33 automated assertions to ensure accuracy never dropped.
