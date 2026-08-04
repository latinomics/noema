---
id: judge/qa_answer
tier: JUDGE
output_schema: QAAnswer
description: Closed-book answering - answer a question using ONLY the provided artifact.
---
Answer the question using ONLY the artifact below. This is a closed-book fidelity probe: you must not use any outside knowledge, however confident you are.

Rules:
- If the artifact contains the information, answer in one short, specific sentence and set "answerable" to true.
- If the artifact does not contain the information needed, set "answerable" to false and leave a brief note in "answer" saying what is missing. Do not guess.

## Artifact

{{artifact}}

## Question

{{question}}

Respond with a JSON object: {"answer": "...", "answerable": <true|false>}
