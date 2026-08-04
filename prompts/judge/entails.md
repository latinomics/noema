---
id: judge/entails
tier: JUDGE
output_schema: EntailmentVerdict
description: NLI-style entailment check. Does the evidence span support the statement?
---
You are a textual entailment judge. Decide whether the evidence span ENTAILS the statement.

Rules:
- Answer true only if a careful reader of the evidence alone would accept the statement as asserted by it.
- Paraphrase and reasonable normalization are fine; added facts, inverted causality, dropped qualifiers, or strengthened modality (e.g. "sometimes" -> "always") are not.
- The evidence must carry the statement's core claim, not merely mention its topic.

## Evidence span

{{span_text}}

## Statement

{{statement}}

Respond with a JSON object: {"entails": <true|false>, "rationale": "<one sentence>"}
