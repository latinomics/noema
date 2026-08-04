---
id: judge/qa_faithful
tier: JUDGE
output_schema: FaithfulnessVerdict
description: Grade a predicted answer against the gold answer for a question.
---
Grade the predicted answer against the gold answer.

Scoring:
- 1.0 - the prediction conveys the same fact(s) as the gold answer; wording may differ.
- 0.5 - partially correct: right direction but missing a load-bearing element, or correct with a material addition the gold answer does not support.
- 0.0 - wrong, contradictory, evasive, or missing (e.g. "NOT IN ARTIFACT").

Judge factual content, not phrasing or length.

## Question

{{question}}

## Gold answer

{{gold}}

## Predicted answer

{{pred}}

Respond with a JSON object: {"score": <0.0 | 0.5 | 1.0>, "rationale": "<one sentence>"}
