# Getting started

A practical walkthrough: install the library, bring up Qdrant with Docker, run the FastAPI
service, and call the pipeline — both from Python and over HTTP.

- [1. Prerequisites](#1-prerequisites)
- [2. Install](#2-install)
- [3. Bring up Qdrant (Docker)](#3-bring-up-qdrant-docker)
- [4. Use it as a library](#4-use-it-as-a-library)
- [5. Run the FastAPI service](#5-run-the-fastapi-service)
- [6. Call the HTTP API](#6-call-the-http-api)
- [7. Method & endpoint reference](#7-method--endpoint-reference)
- [8. Configuration](#8-configuration)

---

## 1. Prerequisites

- Python **3.11+**
- Docker (for Qdrant)
- An OpenAI API key
- `libsndfile` — a native dependency of SONAR's fairseq2 (pip can't install it):
  ```bash
  sudo apt install libsndfile1            # Debian/Ubuntu
  brew install libsndfile                 # macOS
  conda install -c conda-forge libsndfile # inside a conda env
  ```

## 2. Install

```bash
git clone https://github.com/vneshcheretov/prism-rag.git
cd prism-rag

# library only (bring your own embedder)
pip install -e .

# with the default SONAR embedder (pulls torch + fairseq2, ~3 GB model on first use)
pip install -e ".[sonar]"

# with the HTTP API as well
pip install -e ".[api,sonar]"
```

Copy the example env and add your key:

```bash
cp .env.example .env      # then edit OPENAI_API_KEY=sk-...
```

## 3. Bring up Qdrant (Docker)

Qdrant is the only required service for the library:

```bash
docker compose up -d qdrant      # http://localhost:6333
```

To run Qdrant **and** the API together, see [step 5](#5-run-the-fastapi-service).

Check it's healthy:

```bash
curl -s localhost:6333/healthz   # Qdrant's own health endpoint
```

## 4. Use it as a library

```python
import asyncio

from qdrant_client import AsyncQdrantClient
from prism import LLMClient, Prism, PrismGraph, QdrantBackend, SonarEmbedder


async def main() -> None:
    qdrant = QdrantBackend(
        AsyncQdrantClient(url="http://localhost:6333"),
        collection_name="my_kb",
    )
    # recreate=True wipes the collection for a clean run; use False to keep data
    graph = await PrismGraph.create(qdrant, SonarEmbedder(), recreate=True)
    prism = Prism(graph, LLMClient())

    # 1) Ingest markdown — language is auto-detected on the first ingest
    with open("demo/aiso_hotel.txt", encoding="utf-8") as f:
        await prism.ingest(f.read())

    # 2) Retrieve raw chunks (bring your own generation layer)
    found = await prism.search("во сколько заезд?")
    print(found.keypoints)
    print(found.paragraphs)

    # 3) Or get a grounded, synthesized answer
    ans = await prism.answer("можно ли с собакой?")
    print(ans.answer)        # -> e.g. "Да, проживание с животными до 5 кг допускается."
    if ans.note:
        print("note:", ans.note)   # set when the pipeline short-circuited

    await qdrant.client.close()


asyncio.run(main())
```

**Restarting?** If the collection already holds data, rehydrate the in-memory graph instead
of re-ingesting:

```python
graph = await PrismGraph.create(qdrant, SonarEmbedder(), recreate=False)
prism = await Prism.load(graph, LLMClient())   # restores nodes, language, summary
```

**Follow-up questions** — `ChatSession` keeps the dialogue for you:

```python
from prism import ChatSession

session = ChatSession(prism)
await session.ask("можно ли с собакой?")
await session.ask("а с кошкой?")   # resolved against the previous turn
```

## 5. Run the FastAPI service

**Locally:**

```bash
pip install -e ".[api,sonar]"
docker compose up -d qdrant
python -m prism.api               # serves on http://localhost:8000
```

**Everything in Docker (Qdrant + API):**

```bash
# .env must contain OPENAI_API_KEY
docker compose up -d              # starts qdrant and the api service
```

Open the interactive docs at <http://localhost:8000/docs>.

## 6. Call the HTTP API

```bash
# health / readiness
curl -s localhost:8000/health     # {"status":"ok"}
curl -s localhost:8000/ready      # {"status":"ready","nodes":N}  (503 if Qdrant is down)

# ingest a markdown document
curl -s localhost:8000/ingest -H 'content-type: application/json' -d '{
  "markdown": "# Hotel\n\n## Pets\n\nPets up to 5 kg are allowed."
}'

# retrieve chunks
curl -s localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "can I bring a pet?"
}'

# grounded answer
curl -s localhost:8000/answer -H 'content-type: application/json' -d '{
  "query": "can I bring a pet?"
}'

# follow-up question (stateless — pass the prior turns)
curl -s localhost:8000/answer -H 'content-type: application/json' -d '{
  "query": "what about a cat?",
  "history": [
    {"role": "user", "content": "can I bring a dog?"},
    {"role": "assistant", "content": "Yes, pets up to 5 kg are allowed."}
  ]
}'
```

Errors come back as `{"error": "...", "detail": "..."}`: **422** (empty/unprocessable
ingest), **502** (LLM down), **503** (Qdrant down), **500** (unexpected). Many non-fatal
cases instead return `200` with a `note` explaining why the result is empty.

## 7. Method & endpoint reference

**Python**

| Call | What it does |
|---|---|
| `await prism.ingest(markdown)` | chunk → extract keypoints → embed → index; returns nodes. Raises `IngestError` on empty input |
| `await prism.search(query, *, history=None)` | hybrid retrieval → `SearchResult(keypoints, paragraphs, nodes, note)` |
| `await prism.answer(query, *, history=None)` | `search()` + grounded synthesis → `AnswerResult(answer, final_summary, search, note)` |
| `await Prism.load(graph, llm)` | build a Prism with its graph rehydrated from Qdrant |
| `ChatSession(prism).ask(query)` | stateful follow-up wrapper over the stateless engine |

**HTTP**

| Endpoint | Body | Returns |
|---|---|---|
| `POST /ingest` | `{"markdown": "...", "summarize": true}` | indexed nodes, language, corpus summary |
| `POST /search` | `{"query": "...", "filter_relevance": true, "query_language": null, "history": []}` | keypoints + paragraphs |
| `POST /answer` | same as `/search` | grounded answer + underlying search result |
| `GET /health` | — | liveness |
| `GET /ready` | — | readiness (Qdrant reachable) |

## 8. Configuration

All via environment variables (or constructor arguments):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI auth (required) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `PRISM_LLM_FAST_MODEL` | `gpt-5.4-nano` | high-volume extraction calls |
| `PRISM_LLM_STRONG_MODEL` | `gpt-5.4-mini` | judgement & synthesis calls |
| `PRISM_SONAR_DEVICE` | auto (`cuda` if available) | force `cpu` / `cuda` for SONAR |
| `PRISM_COLLECTION` | `prism` | Qdrant collection name (API) |
| `PRISM_LANGUAGE` | unset (auto-detect) | lock the corpus language (API) |
| `PRISM_RECREATE_COLLECTION` | `false` | drop & recreate the collection on API startup |
| `PRISM_API_HOST` / `PRISM_API_PORT` | `0.0.0.0` / `8000` | API bind address |

> The API is unauthenticated by design — put it behind a gateway if you expose it publicly.
> See the [README](../README.md) for architecture and the full configuration reference.
