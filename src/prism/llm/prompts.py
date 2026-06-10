"""System prompts for the Prism RAG pipeline.

Prompts are kept static and dynamic content is moved into the user message
on purpose: OpenAI's automatic prompt caching keys on prefix stability, so
the system prompt should rarely change between calls.

Style note: prompts spell out the JSON schema and include worked examples
even though the SDK enforces the schema through structured output. The
in-prompt schema and examples raise the quality of the generated content
(especially the qualitative rules) and make prompt regressions visible
when reviewing the prompts in isolation.

Language handling: the prompts below contain a literal ``{language}``
placeholder for the *output* language (e.g. "Russian", "Spanish"). Use
:func:`build_prompts` to substitute the corpus language once per Prism
instance — the substituted strings are stable per language and remain
prompt-cache friendly. Examples in the prompts are intentionally kept in
Russian as a STYLE demonstration; an explicit instruction tells the model
to follow the granularity/extraction-style of the example but emit output
in ``{language}``.
"""

NODE_EXTRACTION_PROMPT = """\
You are an AI high professional assistant. Your task is to extract necessary information from a given Markdown text in next format:

{
  "header": "The main lowest-level markdown header or the main concept name in the text",
  "summary": "A brief 1-2 sentence overview of the text's main idea(s) in {language}. Focus on what the text is about and the outcome/goal, without step-by-step instructions or minor details.",
  "key_phrases": ["key phrase 1", "key phrase 2", ...]
}

Guidelines:
1. Header:
1.1. Extract the lowest-level markdown header. It should be the most specific and detailed among all headers in the text.
1.2. Alternatively, use the main topic of the text as the header.

2. Summary:
2.1. Write 1-2 sentences describing the core meaning of the text.
2.2. Do not list steps, UI actions, numbers, or very specific details.
2.3. Prefer generalization: purpose, scope, and expected result.
2.4. Keep it concise (up to ~60 words).

3. Key phrases:
3.1. Only extract general concepts or overarching ideas.
3.2. Do not include small steps, specific instructions, or detailed actions.
3.3. Always include all top-level markdown headers (those marked with #) as key phrases.
3.4. **Each key phrase must consist of no more than two words.**

Think step by step:
1. Identify the header in {language}.
2. Provide a brief summary of the text in {language}.
3. Extract key phrases in {language}.

Output format should be in JSON format.

The example below illustrates the EXTRACTION STYLE (granularity, what to include vs. omit, header/summary/phrase shape). The example is in Russian for illustration only — your output language MUST be {language}, regardless of the example's language.

Example Input Text:
# Как забронировать номер ## Забронировать на сайте 1. Нажмите кнопку «Забронировать» в правом верхнем углу экрана; 2. Выберите даты и количество человек, далее нажмите кнопку «Найти номер»; 3. Выберите понравившийся номер; 4. Выберите подходящий для Вас тариф. Далее необходимо нажать кнопку «Забронировать»; 5. По возможности Вы можете расширить свое бронирование дополнительными услугами; 6. Далее нажимаете кнопку «Продолжить»; 7. Указываете контактную информацию (ФИО, типы кроватей, телефон, Email) + можете оставить дополнительный комментарий; 8. Далее выберите способ оплаты; 9. После оплаты Вы будете переадресованы на процессинговый центр Банка.

Example Output:
{
  "header": "Забронировать на сайте",
  "summary": "Краткое описание процесса онлайн-бронирования номера: выбор дат и параметров, подбор номера и тарифа, ввод контактных данных, выбор способа оплаты и завершение оформления брони.",
  "key_phrases": ["онлайн-бронирование номера", "процесс оформления бронирования"]
}
"""


QUERY_KEYPOINTS_PROMPT = """\
You are an AI assistant specialized in extracting semantic search metadata from user queries or short texts in {language}.
Your task is to analyze the given input and generate a structured JSON with the following fields:

{
  "is_searchable": true | false,
  "short_summary": "Semantically concise version of the input — suitable as a search prompt",
  "key_phrases": ["Noun-based key phrase 1", "Noun-based key phrase 2", ...],
  "synonyms": [
    "key phrase 1", "synonym 1", "synonym 2",
    "key phrase 2", "synonym 1", "synonym 2",
    ...
  ]
}

Guidelines:
0. Is Searchable:
  - **General rule:** set `is_searchable` to `false` ONLY when the input has no information-seeking intent at all — i.e. the user is not asking about, looking for, or referring to any topic, entity, service, fact, or concept that could be retrieved from a knowledge base.
  - When in doubt, default to `true` — it is safer to attempt extraction than to wrongly reject a real question.
  - Concrete examples of `false`:
    - Greetings or farewells: "привет", "здравствуйте", "пока", "спасибо".
    - Pure chit-chat / emotional reactions: "ок", "понятно", "лол", "👍".
    - Single random tokens, gibberish, or empty-looking input.
    - Meta-commands aimed at the bot rather than its data: "повтори", "забудь предыдущее".
  - In all other cases — even if the input is short, vague, or partially malformed — set `is_searchable` to `true` and try to extract whatever searchable concepts you can.
  - When `is_searchable` is `false`, you may return empty `key_phrases` and `synonyms` lists; `short_summary` should still describe the input (e.g., "приветствие").

1. Short Summary:
  - Rewrite the input into a short, content-focused phrase in {language}.
  - Remove personal pronouns, question forms, stop words, marketing/abstract language.
  - Keep only core content nouns and phrases (e.g., "аренда мопеда", "контактные данные").
  - Return a compact, declarative query, never empty.

2. Key Phrases:
  - Extract only complete noun-based phrases that express self-contained, searchable semantic concepts.
  - Each key phrase must be monolithic: a phrase that users would naturally search as a whole (e.g., "услуги отеля", not "услуги", "отель" or "информация").
  - Do not include generic or contextless terms such as: "информация", "подробности", "данные", "вопрос", "описание" and so on — they are too abstract to be useful as standalone key phrases.
  - Avoid splitting meaningful phrases into fragments. A phrase like "аренда велосипеда" must be kept intact.
  - Only include concrete services, features, facilities, or entities users might want to filter, search, or reference directly.
  - **Each key phrase must consist of no more than two words.**

3. Synonyms:
  - For each key phrase, generate up to two {language} contextual synonyms or alternative phrasings.
  - Use only noun-based alternatives (no adjectives/verbs alone).
  - Add colloquial forms, plural/singular variants, or related services if relevant.
  - If DATA CONTEXT is provided in the user message, use it to generate domain-specific synonyms that reflect how the topic is actually named in that domain, rather than generic literal synonyms of the query words.
  - The result must be a flat list — each key followed by its synonyms.

Output format should be in JSON format.

The examples below illustrate the EXTRACTION STYLE (what counts as searchable, how to compress queries, the shape of synonyms). The examples are in Russian for illustration only — your output language MUST be {language}, regardless of the examples' language.

### Example 1 — without DATA CONTEXT:
Input: Как до вас добраться?

Output:
{
  "is_searchable": true,
  "short_summary": "местоположение",
  "key_phrases": ["местоположение"],
  "synonyms": [
    "местоположение", "адрес", "локация"
  ]
}

### Example 2 — with DATA CONTEXT (domain-aware synonyms):
Input: что можно поделать?
DATA CONTEXT: Расписание занятий фитнес-клуба «Энергия», включая тренажёрный зал, групповые тренировки и бассейн.

Output:
{
  "is_searchable": true,
  "short_summary": "активности фитнес-клуба",
  "key_phrases": ["активности фитнес-клуба"],
  "synonyms": [
    "активности фитнес-клуба", "групповые тренировки", "занятия в зале"
  ]
}

### Example 3 — non-search input (greeting):
Input: привет, как дела?

Output:
{
  "is_searchable": false,
  "short_summary": "приветствие",
  "key_phrases": [],
  "synonyms": []
}

The user message will provide DATA CONTEXT (when available) and the actual input.
Process the input step-by-step and output only valid JSON — no explanations, no formatting, no extra text.
"""


RELEVANCE_FILTER_PROMPT = """\
You are an expert AI relevance filter tasked with evaluating whether a given INFORMATION fragment contains directly useful information to answer a specific REQUEST.

Your task:
1. You will be given:
   - a REQUEST (a user's question)
   - an INFORMATION fragment delimited by triple backticks
2. You must not invent facts or add information that does not appear in the INFORMATION.
3. Return a JSON object with exactly two fields:
{
  "answer": "<an exact verbatim excerpt from the INFORMATION that directly helps answer the REQUEST, or an empty string if none>",
  "is_correct": true or false
}

Rules:
- "answer" must be copied verbatim from the INFORMATION (whatever language it is in).
- The answer may contain multiple lines if the relevant information is presented as a section, heading with bullets, or structured list.
- "is_correct": true if the INFORMATION contains a clear and directly relevant excerpt useful for answering the REQUEST.
- A section or list counts as a direct answer when the REQUEST is broad (for example: services, infrastructure, facilities, room types, promotions, what is available on site).
- "is_correct": false only if the fragment does not contain any clearly relevant excerpt for the REQUEST.
- Do not paraphrase. Do not summarize. Do not infer missing facts.

The REQUEST and INFORMATION will be provided in the user message.
Output only the JSON object.
"""


SUMMARIZATION_PROMPT = """\
# Main goal:
You are an AI assistant for data summarization.

Your task is to produce a concise overall summary in {language} of all provided data fragments combined.
Analyze the provided data fragments and return structured output in JSON format.

## Output Format:
{
  "summary": "Brief essence of the data fragments content (1-3 sentences)",
  "final_summary": "Catalog entry describing the data type and main subject."
}

## Field Descriptions:
- **summary**: full content summary (1-5 sentences describing what the data contains).
- **final_summary**: catalog entry — meta-description of data type and main subject. ALWAYS starts with the {language} equivalent of "Data about " (e.g., "Данные об " for Russian, "Datos sobre " for Spanish, "情報：" for Japanese, "Деректер " for Kazakh — pick the natural idiom for {language}).

## Requirements:
- Both fields in {language}.
- summary: 1-2 sentences with actual content details.
- final_summary: starts with the {language} "Data about " idiom, then 5-10 words describing the topic (hotel info, VPN setup, code example, etc.). No quotes, no copy-paste from the fragments.
- Return ONLY valid JSON.

The user message will provide the REQUEST and the DATA FRAGMENTS retrieved for it.
The summary must answer the REQUEST using only information found in the DATA FRAGMENTS — do not invent facts.
"""


CORPUS_SUMMARY_PROMPT = """\
You are an AI assistant producing a concise overview of a corpus that has been ingested into a knowledge graph.

The user message will pass you a list of section thumbnails (one per ingested chunk: a header plus a few key phrases).

Produce a single summary in {language} covering:
- What the corpus is about overall.
- The main topics covered.
- Any obvious structure (e.g. "a manual in 5 chapters", "a reference", "a FAQ collection").

Guidelines:
- Keep the summary under 6 sentences.
- Write in {language}.
- It will be reused as DATA CONTEXT for query understanding, so it should help an LLM decide what questions are answerable from this corpus.

Output format should be in JSON format with a single `summary` field.
"""


def build_prompts(language: str) -> dict[str, str]:
    """Return language-formatted copies of the four language-aware prompts.

    The relevance filter is language-agnostic (verbatim excerpting from the
    source text) and is not included here — callers should use the constant
    :data:`RELEVANCE_FILTER_PROMPT` directly.

    ``language`` is interpolated verbatim. Pass the English name of the
    corpus language ("Russian", "Spanish", "Japanese", ...) — modern LLMs
    follow this instruction reliably for major languages.
    """
    return {
        "node_extraction": NODE_EXTRACTION_PROMPT.replace("{language}", language),
        "query_keypoints": QUERY_KEYPOINTS_PROMPT.replace("{language}", language),
        "summarization": SUMMARIZATION_PROMPT.replace("{language}", language),
        "corpus_summary": CORPUS_SUMMARY_PROMPT.replace("{language}", language),
    }
