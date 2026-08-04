# NOEMA

**A knowledge engineering foundry: books → domain ontologies → agents with a point of view.**

*Noema* (Husserl): the object as intended — the content of a directed mental act. The terminal
artifact of this system is not a knowledge base but an **intentional stance made executable**: an
agent frame with beliefs, desires, and inferential dispositions compiled from source texts.

The authoritative build contract lives in [BLUEPRINT.md](BLUEPRINT.md). To get running, see
[QUICKSTART.md](QUICKSTART.md).

## How it works

Three pipelines, each a pure-ish function between typed contracts:

| Pipeline | Contract | What it does |
|---|---|---|
| **P1 · DOCFORGE** | `SourceRef → CanonicalDoc` | PDF → canonical markdown + structure map, with VLM rescue for hard pages and a fidelity gate |
| **P2 · ONTOGEN** | `CanonicalDoc → DomainOntology` | Per source: layered abstract, entity thesaurus, entailment-validated axioms, mental models |
| **P3 · NOESIS** | `list[DomainOntology] → list[AgentFrame]` | Across sources: entity alignment, tension ledger, analogy bridges, intentional stance modeling, frame compilation, agent gym |

The output of P3 is a set of **agent frame bundles** — system prompt + retrieval pack + MCP tool
manifest + probe suite — drop-in for Claude Agent SDK / LangGraph runtimes. Two archetypes:
`author_simulacrum` (what one author believes, wants, and infers) and `synthetic_polymath`
(a frame that metabolized several sources and has taken positions on their disagreements).

## Status

| Milestone | Deliverable | State |
|---|---|---|
| **M0** | Shared kernel: schemas, gateway, promptlib, manifest, stores, judge | **Done** |
| M1 | DOCFORGE — 3 benchmark books through the gate | Next |
| M2 | ONTOGEN — 3 `DomainOntology` artifacts passing all gate numbers | Pending |
| M3 | NOESIS fusion — sameAs links, tension ledger, analogy bridges | Pending |
| M4 | Stance → compile → gym — gated frames, one exported and running | Pending |

## Design stance

Six principles govern every component (violations are bugs — see BLUEPRINT.md §0):

1. **Contracts over implementations.** Pydantic v2 schemas in `kernel/schemas.py` are the true API.
2. **Prompts are code.** Every prompt is a versioned asset in `prompts/`; its revision hash flows
   into provenance and cache keys.
3. **Evals are tests.** Every pipeline terminates in a cheap, LLM-judged gate. A gate score is a
   number, not a vibe.
4. **Provenance is type-checked.** No `Axiom`, `Entity`, or `MentalModel` exists without a
   `Provenance` pointing at character spans in the canonical markdown — enforced at validation.
5. **The intentional stance is the compilation target.** `AgentFrame` = doxastic base + telos +
   inferential repertoire.
6. **Local-first, swap-friendly.** Embedded stores (DuckDB, LanceDB, NetworkX-on-disk), LiteLLM
   behind one gateway. Nothing requires a server until you want one.

## Repository layout

```
noema.toml                  # config: model tiers, budget, paths, thresholds
prompts/                    # versioned prompt assets (markdown + YAML frontmatter)
data/                       # content-addressed data plane (gitignored)
src/noema/
├── kernel/                 # M0 — shared kernel
│   ├── schemas.py          #   all contracts (the system's true API)
│   ├── config.py           #   pydantic-settings over noema.toml
│   ├── gateway.py          #   LiteLLM + instructor behind MAP/REDUCE/JUDGE tiers
│   ├── promptlib.py        #   prompt loading + revision hashing
│   ├── manifest.py         #   content-addressed stage cache + lineage (DuckDB)
│   ├── stores.py           #   LanceDB vectors, NetworkX graphs, DuckDB tabular
│   └── judge.py            #   eval primitives: rubric judge, entailment, QA fidelity
├── p1_docforge/            # M1 — pdf → canonical markdown
├── p2_ontogen/             # M2 — source → DomainOntology
├── p3_noesis/              # M3/M4 — ontologies → agent frames
└── cli.py                  # typer entrypoint
tests/                      # contract tests (run offline, no API keys)
```

## CLI

```
noema status                           # lineage view: what's cached, spend
noema ingest <pdf ...>                 # register + run P1        (lands with M1)
noema extract <src_id>                 # run P2                   (lands with M2)
noema synthesize --frames 3 --authors  # run P3                   (lands with M3/M4)
noema gym <frame_id>                   # interrogate a frame      (lands with M4)
noema export <frame_id> --target ...   # export a frame bundle    (lands with M4)
```

Every command is resumable via the manifest cache; `--force` busts it.

## Development

```sh
uv sync                 # install (python >= 3.12, managed by uv)
uv run pytest           # 54 tests, fully offline
uv run ruff check .     # lint
uv run ruff format .    # format
```
