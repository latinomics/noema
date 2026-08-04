# NOEMA Cheatsheet

Plain answers to "how do I run this?". Every command is copy-paste ready.
All commands use `uv run noema ...`; if you activate the venv (`source .venv/bin/activate`)
you can drop the `uv run` prefix.

## Set it up (once)

```sh
uv sync                            # installs everything (uv fetches Python too)
export ANTHROPIC_API_KEY=sk-ant-...   # noema.toml points all three tiers at Anthropic
```

Optional, only when you need them:

```sh
uv sync --extra local-embeddings   # real local embeddings (pulls torch, ~2GB)
uv sync --extra tracing            # send traces to Langfuse
```

## Check that everything works (free, no API key needed)

```sh
uv run noema doctor
```

Green `ok` lines mean that part is ready. Yellow `warn` means it will still run
(e.g. no API key yet, or embeddings falling back to the dev-only hash mode).
Red `fail` means something is actually broken and the command exits with code 1.

```sh
uv run pytest      # 69 tests, all offline — proves the code itself is healthy
```

## See what's configured

```sh
uv run noema config
```

Prints the final settings as JSON: which model each tier (MAP / REDUCE / JUDGE) uses,
the budget, paths, and gate thresholds. It also tells you which config file it read.

To change a setting for one command only, use a `NOEMA_` env var (nested keys join with `__`):

```sh
NOEMA_MODELS__JUDGE__MODEL="openai/gpt-4o-mini" uv run noema judge chapter.md --rubric "..."
NOEMA_CONFIG=/path/to/other.toml uv run noema status     # use a different config file
```

## See which prompts exist

```sh
uv run noema prompts
```

Lists every prompt asset under `prompts/`: its id (what you pass to `call`), tier,
revision hash, output schema, and — importantly — the variables it expects.

## Make a live LLM call

### Score a file against a rubric (the simplest real call)

```sh
uv run noema judge chapter.md --rubric "Headings survive conversion; prose is not garbled."
```

Output is a score from 1–5 plus a one-line rationale. Variations:

```sh
uv run noema judge chapter.md --rubric @rubric.txt      # rubric lives in a file
cat chapter.md | uv run noema judge - --rubric "..."    # read the artifact from stdin
uv run noema judge chapter.md --rubric "..." --min 4.2  # gate mode: exit 1 if score < 4.2
```

`--min` makes it usable in scripts and CI: pass/fail is the exit code.

### Run any prompt directly

```sh
uv run noema call judge/entails \
  -v span_text="Delays in feedback loops cause the system to oscillate." \
  -v statement="Delays cause oscillation."
```

Fill each variable with `-v name=value`. For long values, point at a file with `@`:

```sh
uv run noema call judge/rubric -v rubric="Faithful to source." -v artifact=@chapter.md
```

If the prompt declares an output schema you get validated JSON back; otherwise plain text.
Useful flags: `--tier MAP|REDUCE|JUDGE` to override which model runs it, `--text` to skip
schema validation, `--stage my.name` to label the call in the lineage log.

If you forget a variable, the error tells you exactly which ones the prompt expects.

## See what ran and what it cost

```sh
uv run noema status
```

Shows the cached pipeline stages (none yet, until M1 lands) plus the total number of
LLM calls and dollars spent. Every `call` and `judge` you make is recorded here —
check it after any live call.

Spend is also guarded per run: you get a warning at 80% of `[budget].run_usd`
(default $10) and again at 100%. It warns loudly but never hard-stops.

## Not runnable yet (lands with later milestones)

These commands exist but exit immediately with a note about their milestone:

| Command | What it will do | Lands with |
|---|---|---|
| `noema ingest <pdf...>` | turn PDFs into clean, addressable markdown | M1 |
| `noema extract <src_id>` | pull entities, claims, and mental models out of one book | M2 |
| `noema synthesize` | merge several books; build agent personas with positions | M3/M4 |
| `noema gym <frame_id>` | stress-test a persona before release | M4 |
| `noema export <frame_id>` | package a persona for Claude Agent SDK / LangGraph / MCP | M4 |

## When something fails

| Symptom | Fix |
|---|---|
| `warn model:* — set ANTHROPIC_API_KEY` in doctor | `export ANTHROPIC_API_KEY=...` (or the key for your provider) |
| `unknown prompt 'x'; available: ...` | run `uv run noema prompts` and copy an id from the list |
| `prompt 'x' missing variables: [...]` | add the listed `-v name=value` flags |
| `cannot read artifact ...` | check the file path; use `-` only when piping via stdin |
| auth error mid-call | wrong or expired API key for the model in `noema config` |
