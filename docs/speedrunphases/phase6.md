# Phase 6: Payload Assembly & Delivery

## 📌 Objective

Format the LLM outputs and their associated source tracking into the required Pydantic schema and prepare the final HTTP response.

## 🧰 Dependencies

- `pydantic`, `httpx`

## 🏗 Architectural Requirements

### 1. Schema Mapping

- Iterate through the 15 LLM responses.
- For each response, instantiate an `Answer` Pydantic model.
- Map the `question_id`.
- Map the `answer` string.
- Map the `sources`. (You must extract the `filename` and `page_nums` that were attached to the Top 5 winning chunks from Phase 4 and deduplicate them). Instantiate a list of `Source` models.

### 2. Final Output

- Wrap the list of `Answer` models into the final `ChallengeResponse` model.
- Stop the `start_time` tracker and print the total execution time to the console.
- Return the `ChallengeResponse` directly from the FastAPI endpoint.
- Add a prominent `# TODO:` comment indicating where the final `httpx.post()` to the Lucio submission webhook should be placed on Challenge Day.
