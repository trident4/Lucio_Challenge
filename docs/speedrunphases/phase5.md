# Phase 5: Unified LLM Inference

## 📌 Objective

Fire all 15 finalized prompts simultaneously to the remote Mac Studio for deep M&A/Legal reasoning using the Qwen-30B model.

## 🧰 Dependencies

- `asyncio`, `openai` (AsyncOpenAI)

## 🏗 Architectural Requirements

### 1. Prompt Construction

Create a highly structured prompt combining the question and the Top 5 chunks.

- **System Prompt:** "You are an expert M&A and Legal analyst. Answer the user's question using ONLY the provided context. If the answer is not in the context, state that clearly. Be concise and precise."
- **User Prompt:** ```text
  CONTEXT:
  [Insert concatenated Top 5 chunks here]

  QUESTION:
  [Insert question here]

  ```

  ```

### 2. Concurrent Execution

- Create an async function `ask_qwen(prompt_messages)`.
- Use the `AsyncOpenAI` client pointing to the Mac Studio.
  - Model: `qwen/qwen3-30b-a3b-2507`
  - `temperature=0.0` (Critical to prevent hallucination)
  - `max_tokens=250` (Keep answers focused to save time)
- Use `asyncio.gather` to execute all 15 calls simultaneously.
- Implement a basic try/except block to catch API timeouts, returning a formatted error string if the Mac Studio fails.
