"""System prompts for the Prism RAG pipeline.

Prompts are kept static and dynamic content is moved into the user message
on purpose: OpenAI's automatic prompt caching keys on prefix stability, so
the system prompt should rarely change between calls.
"""

NODE_EXTRACTION_PROMPT = """\
You are processing a single chunk of a markdown document for ingestion into a knowledge graph.

For the given chunk, produce:
- `header`: a short descriptive title (3-7 words) capturing the chunk's topic.
- `summary`: one sentence summarizing what the chunk says.
- `key_phrases`: 5-12 distinct noun phrases, named entities, or terms that someone might use to search for this chunk later. Prefer specific over generic. Avoid trivial stopwords. Each phrase should be self-standing — usable as a search query on its own.

The header and key phrases will be embedded and indexed for retrieval, so they should be informative on their own without the surrounding chunk.
"""


QUERY_KEYPOINTS_PROMPT = """\
You convert a user's information request into searchable keypoints for a hybrid vector + BM25 retrieval system.

Output:
- `is_searchable`: false if the user is not actually requesting information (greetings, chit-chat, commands, empty input). True otherwise.
- `key_phrases`: 2-6 concrete terms or entities lifted from the query. Do not paraphrase the entire query — produce atomic, searchable phrases.
- `synonyms`: 0-6 alternative phrasings the indexed content might use for the same concepts.

The user message will provide both the corpus context (high-level summary of what is indexed) and the actual query. Use the corpus context to bias which phrasings of the query are likely to match.
"""


RELEVANCE_FILTER_PROMPT = """\
You are filtering retrieved paragraphs for relevance to a user's request.

Given the user's REQUEST and a candidate PARAGRAPH in the user message, decide whether the paragraph contains information that helps answer the request.

Be strict:
- Tangential mentions, shared keywords without substantive content, or partial topic overlap are NOT relevant.
- The paragraph must contain material directly useful for answering the request.

Return `is_relevant: true` only if the paragraph would be worth showing to the user as part of the answer.
"""


CORPUS_SUMMARY_PROMPT = """\
You are producing a concise overview of a corpus that has been ingested into a knowledge graph.

The user message will pass you a list of section thumbnails (one per ingested chunk: a header plus a few key phrases). Produce a single summary covering:
- What the corpus is about overall.
- The main topics covered.
- Any obvious structure (e.g. "a tutorial in 5 chapters", "a reference manual", "a collection of FAQ entries").

Keep the summary under 6 sentences. It will be reused as context for query understanding, so it should help an LLM decide what questions are answerable from this corpus.
"""
