---
id: judge/qa_generate
tier: JUDGE
output_schema: QAPairSet
description: Generate k factual QA pairs probing the central claims of a source text.
---
Generate exactly {{k}} question/answer pairs from the source text below.

Requirements:
- Target the text's central claims, definitions, causal assertions, and named concepts — not trivia or incidental phrasing.
- Every question must be answerable from the source text alone, with a short, specific gold answer (one sentence or less).
- Cover different sections and different claim types; no two questions may probe the same fact.
- Do not reference "the text", "the author says", or section numbers in questions; ask about the subject matter directly.

## Source text

{{source_text}}

Respond with a JSON object: {"pairs": [{"question": "...", "answer": "..."}, ...]}
