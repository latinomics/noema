# NOEMA — A Knowledge Engineering Foundry

**From books → domain ontologies → agents with a point of view.**

*Noema* (Husserl): the object as intended — the content of a directed mental act. Fitting, because the terminal artifact of this system is not a knowledge base but an **intentional stance made executable**: an agent frame with beliefs, desires, and inferential dispositions compiled from source texts.

This document is the build contract for downstream coding agents. Everything here is buildable by a small team (or a swarm of coding agents) in weeks, not quarters. No enterprise concerns. No speculative edge-case armor. Contracts, stages, tools, gates.

---

## 0. Design Stance

Six principles govern every component. Coding agents should treat violations as bugs.

1. **Contracts over implementations.** Pydantic v2 schemas in `kernel/schemas.py` are the system's true API. Pipelines are pure-ish functions between schemas. Any stage can be rewritten without touching its neighbors.
2. **Prompts are code.** Every prompt lives in `prompts/` as a markdown asset with frontmatter (id, revision, model tier, output schema ref). Prompt revisions hash into the lineage cache like source code.
3. **Evals are tests.** Every pipeline terminates in a gate — cheap, automatic, LLM-judged. A gate score is a number, not a vibe. Thresholds are startup-honest (catch regressions, don't chase nines).
4. **Provenance is type-checked.** No `Axiom`, `Entity`, or `MentalModel` exists without a `Provenance` pointing at character spans in the canonical markdown. Ungrounded knowledge is a validation error.
5. **The intentional stance is the compilation target.** Dennett: predict a system by ascribing the beliefs it *ought* to have and the desires it *ought* to pursue, assuming rationality. We invert this — we *author* agents by specifying exactly those three things. `AgentFrame` = doxastic base + telos + inferential repertoire.
6. **Local-first, swap-friendly.** Embedded stores (DuckDB, LanceDB, NetworkX-on-disk), LiteLLM behind one gateway. Nothing requires a server until you want one.

---

## 1. System Topology

```
                         ┌─────────────────────────────────────────────┐
  PDF shelf ────────────▶│  P1 · DOCFORGE                              │
  (content-addressed)    │  pdf → canonical markdown + structure map   │
                         └───────────────────┬─────────────────────────┘
                                             │ CanonicalDoc
                                             ▼
                         ┌─────────────────────────────────────────────┐
                         │  P2 · ONTOGEN  (runs once per source)       │
                         │  abstract · entity thesaurus · axioms ·     │
                         │  mental models  →  DomainOntology           │
                         └───────────────────┬─────────────────────────┘
                                             │ N × DomainOntology
                                             ▼
                         ┌─────────────────────────────────────────────┐
                         │  P3 · NOESIS  (runs across sources)         │
                         │  align → tension ledger → analogy bridges → │
                         │  intentional stance modeling → frame        │
                         │  compiler → agent gym                       │
                         └───────────────────┬─────────────────────────┘
                                             │
                                             ▼
                    AgentFrame bundles: system prompt + retrieval pack
                    + MCP tool manifest + probe suite  (drop-in for
                    Claude Agent SDK / LangGraph runtimes)
```

### Repository Layout

```
noema/
├── pyproject.toml              # uv-managed, python ≥3.12
├── noema.toml                  # pydantic-settings config (model tiers, thresholds, paths)
├── prompts/                    # versioned prompt assets (md + yaml frontmatter)
│   ├── docforge/               #   e.g. rescue_transcribe.md, page_judge.md
│   ├── ontogen/                #   e.g. axiom_mine.md, model_archaeology.md
│   └── noesis/                 #   e.g. tension_triage.md, stance_author.md
├── data/
│   ├── shelf/                  # raw PDFs, filed by content hash
│   ├── markdown/<src_id>/      # doc.md, structure.json, pages/, doc_report.json
│   ├── ontology/<src_id>/      # ontology.json (+ entities.parquet, axioms.jsonl, models.jsonl)
│   ├── noesis/<run_id>/        # fused.json, tensions.jsonl, bridges.jsonl
│   ├── frames/<frame_id>/      # frame.json, system_prompt.md, retrieval/, tools.json, probes/, frame_report.json
│   └── indexes/                # lancedb/, lineage.duckdb
├── src/noema/
│   ├── kernel/
│   │   ├── schemas.py          # ALL contracts live here (§2.1)
│   │   ├── gateway.py          # LiteLLM router + instructor + budget meter (§2.2)
│   │   ├── promptlib.py        # prompt loading, revision hashing (§2.3)
│   │   ├── manifest.py         # lineage + content-addressed stage cache (§2.4)
│   │   ├── stores.py           # DuckDB / LanceDB / NetworkX adapters (§2.5)
│   │   └── judge.py            # eval primitives: rubric judge, entailment, QA probe (§2.6)
│   ├── p1_docforge/            # stages.py, rescue.py, evals.py
│   ├── p2_ontogen/             # segment.py, abstracts.py, entities.py, thesaurus.py,
│   │                           # axioms.py, models.py, evals.py
│   ├── p3_noesis/              # align.py, tensions.py, bridges.py, stance.py,
│   │                           # compile.py, gym.py
│   └── cli.py                  # typer entrypoint (§6)
└── tests/                      # contract tests + golden fixtures (3 benchmark books)
```

---

## 2. Shared Kernel

The kernel is milestone zero. Everything else imports it; nothing in it imports pipelines.

### 2.1 `schemas.py` — the contract layer

Abbreviated but load-bearing. Coding agents: implement these exactly, extend freely, never weaken `Provenance` requirements.

```python
# ── addressing ────────────────────────────────────────────────────────────
class SourceRef(BaseModel):
    src_id: str  # sha256[:12] of the PDF bytes
    title: str
    authors: list[str]
    filed_at: datetime


class Provenance(BaseModel):
    src_id: str
    span: tuple[int, int]  # char offsets into canonical doc.md
    section_path: list[str]  # ["Ch 3", "3.2 Feedback Loops"]
    extractor: str  # stage name
    model: str  # e.g. "claude-sonnet-x"
    prompt_rev: str  # hash from promptlib


class StructureMap(BaseModel):  # the addressing system for ALL provenance
    toc: list[TocNode]  # heading tree with char offsets
    page_offsets: list[int]  # char offset of each source page start


# ── P1 output ─────────────────────────────────────────────────────────────
class CanonicalDoc(BaseModel):
    source: SourceRef
    md_path: Path
    structure: StructureMap
    report: DocReport  # fidelity scores, rescued pages, warnings


# ── P2 ontology parts ─────────────────────────────────────────────────────
class LayeredAbstract(BaseModel):
    one_liner: str  # ≤ 30 words
    paragraph: str  # ~150 words
    page: str  # ~600 words
    argument_spine: list[str]  # ordered chain of the book's core claims
    chapter_abstracts: dict[str, str]


class Entity(BaseModel):
    ent_id: str
    canonical_name: str
    aliases: list[str]
    kind: str  # open vocabulary: concept|person|method|system|metric|...
    definition: str  # synthesized, ≤ 60 words
    idiosyncratic: bool  # author-coined term? (e.g. "antifragile")
    provenance: list[Provenance]


class ThesaurusEdge(BaseModel):
    subject: str  # ent_id
    predicate: Literal["broader", "narrower", "related", "synonym", "antonym", "part_of"]
    object: str  # ent_id
    provenance: list[Provenance]


class Axiom(BaseModel):
    ax_id: str
    statement: str  # normalized declarative form
    formality: Literal["definition", "law", "principle", "heuristic", "empirical_claim"]
    modality: Literal["always", "typically", "sometimes"]
    scope: str  # conditions under which the author asserts it
    author_credence: float  # 0–1, the AUTHOR's confidence, rubric-elicited
    entities: list[str]  # ent_ids referenced
    provenance: list[Provenance]  # evidence spans; each must entail `statement`


class CausalLink(BaseModel):
    src_var: str
    dst_var: str
    polarity: Literal["+", "-", "~"]  # reinforcing, opposing, nonmonotonic


class MentalModel(BaseModel):
    mm_id: str
    name: str  # "Feedback Loops", "Antifragility", "Selection Pressure"
    one_liner: str
    variables: list[str]
    dynamics: list[CausalLink]
    predicts: list[str]  # what the model lets you anticipate
    fails_when: list[str]  # known breakdown conditions
    transfer_notes: str  # where the author claims/implies it generalizes
    worked_examples: list[Provenance]


class DomainOntology(BaseModel):
    source: SourceRef
    abstract: LayeredAbstract
    entities: list[Entity]
    thesaurus: list[ThesaurusEdge]
    axioms: list[Axiom]
    mental_models: list[MentalModel]
    fidelity: OntologyReport


# ── P3 synthesis + intentional stance ────────────────────────────────────
class SameAsLink(BaseModel):
    left: tuple[str, str]  # (src_id, ent_id)
    right: tuple[str, str]
    confidence: float


class Tension(BaseModel):
    t_id: str
    axioms: list[tuple[str, str]]  # (src_id, ax_id) pairs in conflict
    kind: Literal["contradiction", "scope_overlap", "value_conflict"]
    gloss: str  # one-paragraph statement of the disagreement


class AnalogyBridge(BaseModel):  # Gentner structure-mapping across domains
    b_id: str
    source_model: tuple[str, str]  # (src_id, mm_id)
    target_model: tuple[str, str]
    mapping: dict[str, str]  # variable-to-variable relational alignment
    strength: float  # relational overlap score
    breaks_when: list[str]  # disanalogy conditions — as important as the mapping


class Goal(BaseModel):
    statement: str
    priority: int
    derived_from: list[str]  # argument-spine claims that motivate it


class Belief(BaseModel):
    axiom_ref: tuple[str, str]
    credence: float  # the FRAME's credence (may differ from author's)
    justification: str


class TensionStance(BaseModel):
    tension_id: str
    stance: Literal["adopt_left", "adopt_right", "contextualize", "hold_open"]
    rationale: str


class EpistemicPolicy(BaseModel):
    humility_rule: str  # when to say "outside my doxastic base"
    citation_rule: str  # when answers must cite provenance spans
    uncertainty_expression: str  # how credences surface in language
    deference: list[str]  # topics on which the frame defers to the user/tools


class AgentFrame(BaseModel):
    frame_id: str
    archetype: Literal["author_simulacrum", "synthetic_polymath"]
    identity: str  # 2–3 sentence self-description
    telos: list[Goal]
    doxastic_base: list[Belief]
    inferential_repertoire: list[str]  # mm_ids + bridge b_ids, ranked by applicability
    lexicon: list[str]  # ent_ids the frame speaks in, idiosyncratic terms first
    tension_stances: list[TensionStance]
    epistemic_policy: EpistemicPolicy
    compiled: CompiledFrame | None


class CompiledFrame(BaseModel):
    system_prompt_path: Path  # ≤ ~2k tokens, generated
    retrieval_pack: Path  # LanceDB slice of provenance-linked chunks
    tool_manifest_path: Path  # MCP-style tools.json
    probe_suite_path: Path
    gym_report: FrameReport | None
```

### 2.2 `gateway.py` — one door to all models

LiteLLM router + `instructor` for schema-validated outputs. Three named tiers, configured in `noema.toml`, referenced by tier — never by model string — in pipeline code:

| Tier | Role | Default class |
|---|---|---|
| `MAP` | high-volume per-chunk extraction | cheap/fast (Haiku/Flash-class) |
| `REDUCE` | synthesis, fusion, stance authoring | frontier (Sonnet/Opus-class) |
| `JUDGE` | eval rubrics, entailment checks | mid-tier, temperature 0 |

Gateway responsibilities: retries with jitter, structured-output validation + one repair pass, per-run token/cost budget meter (log to lineage, warn at 80%), and trace emission to Langfuse.

### 2.3 `promptlib.py` — prompts as versioned assets

Loads `prompts/**/*.md`; frontmatter declares `id`, `tier`, `output_schema`. `prompt_rev = sha256(body)[:8]` flows into every `Provenance` and every cache key. Optional M4+ upgrade: wrap hot prompts in DSPy modules and optimize against the eval gates — the gates double as DSPy metrics for free.

### 2.4 `manifest.py` — AI-native make

Content-addressed stage cache in `lineage.duckdb`. A stage re-runs iff its key changes:

```
key = hash(input_artifact_hashes, code_rev, prompt_rev, model_tier_binding, params)
```

This is the whole orchestration story at this scale: `typer` CLI + this cache = resumable, incremental, parallelizable-later pipelines. Swap in Prefect 3 only when you need scheduling/fan-out across machines.

### 2.5 `stores.py`

| Store | Engine | Holds |
|---|---|---|
| Tabular / lineage | DuckDB | manifest, entity tables, eval scores |
| Vectors | LanceDB (embedded) | chunk embeddings, mention embeddings, retrieval packs |
| Graph ops | NetworkX, persisted JSONL + GraphML export | thesaurus, sameAs graph, causal graphs |

Embeddings default: BGE-M3 local via `sentence-transformers`; one-line swap to a hosted embedding API in `noema.toml`.

### 2.6 `judge.py` — eval primitives

Three reusable primitives every gate is built from:

- `rubric_judge(artifact, rubric_prompt) -> score 1–5` — LLM-as-judge with a written rubric.
- `entails(span_text, statement) -> bool` — NLI-style check via JUDGE tier; used to validate every axiom's provenance.
- `qa_probe(source_text, k) -> [(q, gold_a)]` then `answer_from(artifact_only)` then `faithful(pred, gold) -> score` — the closed-book fidelity loop used by P2 and P3.

---

## 3. Pipeline 1 · DOCFORGE — pdf → canonical markdown

**Contract:** `SourceRef → CanonicalDoc`. The canonical markdown + `StructureMap` is the substrate every later `Provenance` addresses into — treat its stability as sacred. Re-running P1 with different params produces a *new* `src_id` revision, never a mutation.

### Stages

| # | Stage | Responsibility | Tooling |
|---|---|---|---|
| 1 | `intake` | hash PDF → `data/shelf/`, register `SourceRef`; probe digital-vs-scanned ratio (embedded text layer %) | PyMuPDF |
| 2 | `fast_pass` | if ≥95% digital: cheap first draft for triage baseline | PyMuPDF4LLM |
| 3 | `parse` | full layout-aware conversion: headings, tables, figures, equations → markdown + layout JSON | **Marker** (primary — strongest on book-structured docs); MinerU for scan-heavy/CJK; Docling as swap |
| 4 | `triage` | per-page confidence: parser signals + heuristics (garbled-char ratio, table density, equation density, empty-page) | pure python |
| 5 | `rescue` | low-confidence pages → render page image → frontier VLM transcribes to markdown → splice | REDUCE-tier VLM via gateway; prompt `docforge/rescue_transcribe.md` |
| 6 | `refine` | normalize heading tree to single H1–H4 hierarchy; reflow footnotes inline as `[^n]`; VLM alt-text for figures; math to `$...$` LaTeX | MAP tier |
| 7 | `structure_map` | build `TocNode` tree + page char-offsets — the coordinate system for all provenance | pure python |
| 8 | `gate` | sample k=8 pages: VLM `rubric_judge` on structure fidelity + fuzzy-match on 20 random source spans; emit `DocReport` | judge.py |

**Gate:** mean fidelity ≥ 4.2/5 and span match ≥ 0.97 → pass. Failing pages loop through `rescue` once; still failing → flagged in `DocReport`, pipeline proceeds (gritty > perfect — a two-page casualty shouldn't block ontology extraction from a 400-page book).

---

## 4. Pipeline 2 · ONTOGEN — one source → one DomainOntology

**Contract:** `CanonicalDoc → DomainOntology`. Runs per source, embarrassingly parallel across sources. All LLM extraction is MAP-tier over heading-tree chunks (~1.5k tokens, chapter-aware, no mid-paragraph splits), REDUCE-tier for book-level synthesis.

### Stages

| # | Stage | Responsibility | Method |
|---|---|---|---|
| 0 | `segment` | heading-tree chunker off `StructureMap`; every chunk carries its `section_path` + span | custom (chonkie for fallback splitting inside oversized leaves) |
| 1 | `abstracts` | map: chapter abstracts → reduce: `LayeredAbstract` incl. **argument spine** (the ordered chain of core claims — later the raw material for telos derivation in P3) | MAP→REDUCE |
| 2 | `entity_harvest` | two-channel recall: GLiNER zero-shot sweep (high recall, cheap) ∪ per-chunk typed LLM extraction via instructor → `EntityMention` stream | GLiNER + MAP |
| 3 | `resolve` | embed mentions → HDBSCAN cluster → LLM adjudicates merges within clusters → canonical `Entity` with synthesized definition + alias set; flag `idiosyncratic` terms | BGE-M3 + hdbscan + MAP |
| 4 | `thesaurus` | build kNN graph over entity embeddings; elicit SKOS-lite relations only for candidate pairs (top-8 neighbors) — never O(n²) | MAP |
| 5 | `axiom_mine` | per-chunk declarative-claim mining with rubric for formality/modality/scope/`author_credence`; embed-dedupe; **every axiom passes `entails(span, statement)` or is dropped** | MAP + judge |
| 6 | `model_archaeology` | detect recurring causal schemas across chunks ("cognitive archaeology" prompt); cluster candidates; assemble `MentalModel` with causal graph + failure modes; **regeneration probe**: the extracted model, given only its own fields, must re-derive one of the book's worked examples | MAP→REDUCE + judge |
| 7 | `assemble + gate` | write `DomainOntology`; fidelity harness | judge.py |

**Gate (three numbers in `OntologyReport`):**
1. *QA fidelity*: 20 questions generated from source text, answered from the ontology **only** (closed book) → faithfulness ≥ 0.8.
2. *Groundedness*: 100% of axioms entailment-validated (enforced upstream, re-asserted here).
3. *Coverage proxy*: canonical entities capture ≥ 0.85 of GLiNER's high-confidence sweep.

---

## 5. Pipeline 3 · NOESIS — many ontologies → agents with a point of view

**Contract:** `list[DomainOntology] → list[AgentFrame]` (+ fusion artifacts). This is where the philosophy earns its keep. Three fusion moves, then the intentional stance modeling, then compilation, then interrogation.

### 5.1 Fusion stages

| # | Stage | Responsibility | Method |
|---|---|---|---|
| 1 | `registry` | load N ontologies, build joint LanceDB + NetworkX indexes | stores.py |
| 2 | `align` | cross-source entity matching: embed canonical entities, candidate pairs ≥ 0.86 cosine → LLM adjudication → `SameAsLink`s; merge thesauri over the linked lattice | MAP |
| 3 | `tensions` | candidate axiom pairs = pairs sharing aligned entities; LLM triage into `Tension` records (contradiction / scope-overlap / value-conflict). **Disagreement is a feature, not noise** — the tension ledger is what gives frames epistemic character instead of mushy averaging | MAP→REDUCE |
| 4 | `bridges` | Gentner structure-mapping over mental models: match *relational signatures* (causal-graph shapes), not surface vocabulary — "selection pressure" ↔ "market competition" aligns because both are variation→filter→retention loops. Emit `AnalogyBridge` with explicit `breaks_when` disanalogies | REDUCE |

### 5.2 Intentional stance modeling (`stance.py`)

Dennett's move, inverted into authorship. For each requested frame, the REDUCE tier is prompted as a *stance author* to specify:

- **Telos** — goals derived from the argument spines ("what would an agent who wrote/absorbed these arguments be *for*?"), prioritized.
- **Doxastic base** — a selected, coherent subset of axioms with frame-level credences (which may respectfully diverge from `author_credence` — the frame reports *its* confidence).
- **Inferential repertoire** — mental models + analogy bridges, ranked by applicability breadth; this operationalizes an inferentialist bet: the frame's *meaning* lives in how it reasons, not just what it stores.
- **Lexicon** — entity vocabulary the frame natively speaks, idiosyncratic terms foregrounded.
- **Tension stances** — one declared stance per relevant `Tension`: adopt / contextualize / hold-open, with rationale. A frame that has *taken positions on its own contradictions* is coherent in a way an averaged embedding soup never is.
- **Epistemic policy** — humility rule, citation rule, uncertainty expression, deference topics.

Two archetypes, both first-class:

- **`author_simulacrum`** — one frame per source: the intentional stance applied to the text's origin. "What Meadows believes, wants, and infers."
- **`synthetic_polymath`** — K fusion frames spanning sources: "the systems thinker who metabolized Meadows + Taleb + Darwin." Selection objective: high axiom coverage, low unresolved internal tension, high bridge utilization.

### 5.3 Frame compiler (`compile.py`)

Each `AgentFrame` compiles to a **drop-in bundle** under `data/frames/<frame_id>/`:

| Artifact | Contents |
|---|---|
| `system_prompt.md` | generated, ≤ ~2k tokens: identity, telos, lexicon glossary (idiosyncratic terms defined), epistemic policy, tension stances in prose |
| `retrieval/` | LanceDB slice: every chunk referenced by the frame's provenance closure — the frame's *evidence*, not the whole corpus |
| `tools.json` | MCP-style manifest: `ontology_lookup(ent_id)`, `cite_span(prov)`, `run_mental_model(mm_id, situation)`, `check_tension(topic)` |
| `probes/` | the gym suite (below) |
| `frame.json` | the full `AgentFrame` — machine-readable ground truth |

Export targets via `noema export`: Claude Agent SDK config, LangGraph node spec, bare MCP server.

### 5.4 Agent gym (`gym.py`)

Compiled frames are interrogated before release. Four probe families, all `rubric_judge`-scored into `FrameReport`:

1. **Consistency** — same question, five paraphrases; answers must agree with the doxastic base and each other.
2. **Humility** — questions deliberately outside the doxastic base; the frame must invoke its humility rule, not confabulate.
3. **Transfer** — novel scenarios solvable only by applying a mental model or analogy bridge across domains.
4. **Tension stability** — probing declared tension stances under pressure; the frame may elaborate, never silently flip.

**Gate:** all four families ≥ 4.0/5 → frame is releasable.

**Stretch (M4.5) — the dialectic harness:** run two `author_simulacrum` frames in structured debate on a shared `Tension`; the transcript, judged, feeds back as a richer `TensionStance` rationale for polymath frames. Synthesis by argument, not by averaging.

---

## 6. CLI Surface

```
noema ingest <pdf ...>                 # register + run P1
noema extract <src_id>                 # run P2
noema synthesize --frames 3 --authors  # run P3 fusion + stance modeling
noema gym <frame_id>                   # interrogate a compiled frame
noema export <frame_id> --target claude-agent-sdk | langgraph | mcp
noema status                           # lineage view: what's cached, what's stale, spend
```

Every command is resumable via the manifest cache; `--force` busts it.

---

## 7. Build Order (milestones for coding agents)

| M | Deliverable | Definition of done |
|---|---|---|
| **M0** | kernel: schemas, gateway, promptlib, manifest, stores, judge | round-trip a synthetic `DomainOntology` through stores; cache hit/miss behaves; one rubric_judge call traced end-to-end |
| **M1** | DOCFORGE | 3 benchmark books through the gate: one clean digital, one math-heavy, one scanned |
| **M2** | ONTOGEN | 3 `DomainOntology` artifacts passing all three gate numbers |
| **M3** | NOESIS fusion | sameAs links, tension ledger, ≥5 analogy bridges across the 3 ontologies, human-spot-checked |
| **M4** | stance → compile → gym | 3 author simulacra + 1 synthetic polymath, all gym-gated; one frame exported and running in Claude Agent SDK |
| M4.5 | dialectic harness | one judged debate transcript improving one polymath's tension stances |

Fixture books recommendation: Meadows *Thinking in Systems*, Taleb *Antifragile*, any well-scanned public-domain Darwin. Maximal mental-model density, maximal productive tension, one guaranteed OCR stress test.

---

## 8. Bill of Materials

| Concern | Primary | Swap / notes |
|---|---|---|
| Env / QA | uv, ruff, pytest, python ≥3.12 | — |
| Contracts | Pydantic v2 | export JSON Schema for non-python consumers |
| PDF parse | Marker | MinerU (scans/CJK), Docling (table-heavy), PyMuPDF4LLM (fast probe) |
| Hard-page rescue | frontier VLM via gateway | self-hosted OCR-VLM (olmOCR-class) if volume spikes |
| Structured outputs | instructor over LiteLLM | BAML; native provider structured outputs |
| Model routing | LiteLLM (3 tiers) | direct SDKs |
| NER pre-pass | GLiNER | spaCy-trf |
| Embeddings | BGE-M3 local | hosted embedding API, one config line |
| Chunking | heading-tree custom | chonkie for interior splits |
| Entity clustering | HDBSCAN (scikit) | leiden on kNN graph |
| Vectors | LanceDB | Chroma; pgvector when server-y |
| Tabular / lineage | DuckDB | SQLite |
| Graph | NetworkX + JSONL/GraphML | Neo4j only if you truly need Cypher |
| Orchestration | typer + manifest cache | Prefect 3 at multi-machine scale |
| Traces / evals | Langfuse (self-host) + pytest | Logfire, Braintrust |
| Prompt optimization | manual → DSPy at M4+ | gates double as DSPy metrics |

---

*End of blueprint. Build M0 first; everything downstream is a pure function away.*
