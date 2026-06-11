# Prism demo

End-to-end walkthrough of the Prism pipeline on a tiny **Russian** hotel corpus —
ingest with language auto-detection, hybrid search, grounded answers, and the
cross-language refusal policy.

| File | What it is |
|---|---|
| [`aiso_hotel_demo.ipynb`](aiso_hotel_demo.ipynb) | Annotated notebook — the best place to start, every step explained |
| [`aiso_hotel_demo.py`](aiso_hotel_demo.py) | Same flow as a plain script |
| [`aiso_hotel.txt`](aiso_hotel.txt) | The corpus: a fictional hotel fact sheet (description, amenities, house rules) |

## Run it

From the repository root:

```bash
# prerequisites (see the main README for details)
pip install -e ".[sonar]"
docker compose up -d qdrant
cp .env.example .env        # put your OPENAI_API_KEY here

# script
python demo/aiso_hotel_demo.py

# or the notebook
jupyter lab demo/aiso_hotel_demo.ipynb
```

> First run downloads the SONAR encoder (~3 GB) into `~/.cache/fairseq2`;
> subsequent runs load it from cache in seconds.

## What you'll see

1. **Ingest** — the corpus is chunked by markdown headers, an LLM extracts a header
   plus key phrases per chunk, everything is embedded and indexed into Qdrant.
   The corpus language (`ru`) is auto-detected and propagated to the embedder,
   prompts, and stemmer.
2. **Search** — four query types: a specific rule ("можно ли с собакой?"), a time
   lookup, a broad list question, and a near-miss ("тренажёрный зал" — the corpus
   only has a "мини-фитнес-зал", dense retrieval still finds it).
3. **Answer** — `prism.answer()` synthesizes a direct reply with the qualifying
   details (the 5 kg pet weight limit, not just "yes").
4. **Cross-language refusal** — an English query over the Russian corpus returns
   a polite localized message instead of degraded retrieval.
