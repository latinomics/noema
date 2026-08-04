# Quickstart

Get NOEMA installed, configured, and making its first judged LLM call.

## 1. Install

Requires [uv](https://docs.astral.sh/uv/) (Python ≥ 3.12 is fetched automatically):

```sh
git clone <this-repo> && cd noema
uv sync
```

Optional extras:

```sh
uv sync --extra local-embeddings   # BGE-M3 local embeddings (pulls torch)
uv sync --extra tracing            # Langfuse trace emission
```

Without `local-embeddings`, vector stores fall back to a deterministic hash embedder — the
plumbing works (fine for dev/tests), but there are no semantics. Install the extra or switch
to a hosted API before doing real extraction work.

## 2. Verify the install (no API keys needed)

The entire test suite runs offline against a mocked LLM transport:

```sh
uv run pytest          # 69 passed
uv run noema status    # "no stages recorded yet · llm calls: 0 · spend: $0.0000"
uv run noema doctor    # preflight: config, data plane, prompts, lineage, API keys
uv run noema --help
```

## 3. Configure models

Pipeline code speaks **tiers**, never model strings. The three tiers live in `noema.toml`:

```toml
[models.map]      # high-volume per-chunk extraction (cheap/fast class)
model = "anthropic/claude-haiku-4-5"

[models.reduce]   # synthesis, fusion, stance authoring (frontier class)
model = "anthropic/claude-sonnet-4-5"

[models.judge]    # eval rubrics, entailment (temperature pinned to 0 by the gateway)
model = "anthropic/claude-haiku-4-5"
```

Model strings are [LiteLLM names](https://docs.litellm.ai/docs/providers) — any provider works.
Credentials come from the provider's usual environment variable:

```sh
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY, GEMINI_API_KEY, ...
```

Any setting can be overridden per-invocation with `NOEMA_`-prefixed env vars (nested keys join
with `__`), and `NOEMA_CONFIG=/path/to/other.toml` points at an alternate config file:

```sh
NOEMA_MODELS__JUDGE__MODEL="openai/gpt-4o-mini" uv run noema status
```

Budget: each run warns at 80% of `[budget].run_usd` and again at 100% — it never hard-stops.

## 4. First live call: a judged rubric score

The pipelines land with M1+; today you drive the kernel directly. One real JUDGE-tier call,
schema-validated, traced to lineage — straight from the CLI:

```sh
uv run noema judge chapter.md --rubric "Headings survive conversion; prose is not garbled."
# score: 4/5
# rationale: clean structure, minor ...

uv run noema call judge/entails -v span_text=@evidence.txt -v statement="Delays cause oscillation."
# {"entails": true, "rationale": "..."}
```

`noema call` runs any versioned prompt asset through its tier (`noema prompts` lists them;
`-v key=@file` reads a value from a file). `noema judge` wraps the rubric-judge eval primitive
and takes `--min 4.2` to act as a gate in scripts. The same flow from Python:

```python
from noema.kernel import Gateway, Manifest, Settings
from noema.kernel.judge import rubric_judge

settings = Settings.load()  # noema.toml + NOEMA_* env overrides
settings.paths.ensure()  # creates the data/ tree
manifest = Manifest(settings)  # lineage + stage cache in data/indexes/lineage.duckdb
gw = Gateway(settings, manifest=manifest)

verdict = rubric_judge(
    gw,
    artifact="# Chapter 1\n\nStocks are accumulations. Flows fill and drain them...",
    rubric="Headings survive conversion; prose is not garbled; no dropped sentences.",
    stage="demo.gate",
)
print(verdict.score, "-", verdict.rationale)  # e.g. 4 - "clean structure, minor ..."
print(f"spent ${gw.spent_usd:.4f}")
```

Run it, then inspect the lineage:

```sh
uv run python demo.py
uv run noema status    # shows the call count and spend
```

Every call is traced: stage, tier, model, prompt id + revision, tokens, cost, latency.

## 5. The manifest cache in one minute

Stages run behind a content-addressed cache. A stage re-runs **iff** its key changes — the key
hashes inputs, the stage function's source, prompt revisions, tier→model bindings, and params:

```python
result = manifest.run(
    "demo.stage",
    lambda: {"answer": 42},  # must return a JSON-able dict
    inputs={"doc": "content-hash-or-path-or-model"},
    params={"k": 20},
)
print(result.cached)  # False the first time, True forever after — until the key changes
```

Pass `force=True` (or `--force` on CLI commands) to bust the cache deliberately.

## 6. What to read next

- [BLUEPRINT.md](BLUEPRINT.md) — the full build contract: pipelines, stages, gates, milestones.
- `src/noema/kernel/schemas.py` — the contract layer; start here to understand any artifact.
- `prompts/judge/*.md` — what a versioned prompt asset looks like (frontmatter declares
  `id`, `tier`, `output_schema`; the body hash becomes `prompt_rev` everywhere).
