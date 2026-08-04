"""NOEMA contract layer (BLUEPRINT §2.1).

ALL cross-stage contracts live here. Pipelines are pure-ish functions between
these schemas; any stage can be rewritten without touching its neighbors.

Non-negotiable invariant (design stance #4): no ``Axiom``, ``Entity``, or
``MentalModel`` exists without at least one ``Provenance`` pointing at
character spans in the canonical markdown. Ungrounded knowledge is a
validation error, enforced right here with ``min_length=1``. Never weaken it.
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── base ─────────────────────────────────────────────────────────────────────


class NoemaModel(BaseModel):
    """Base for every contract: strict fields, assignment validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def make_id(prefix: str, *parts: str) -> str:
    """Deterministic short id, e.g. ``make_id("ax", src_id, statement)`` -> ``ax_1f3a9c2b7d``."""
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:10]}"


# ── addressing ───────────────────────────────────────────────────────────────


class SourceRef(NoemaModel):
    src_id: str  # sha256[:12] of the PDF bytes
    title: str
    authors: list[str]
    filed_at: datetime


class Provenance(NoemaModel):
    src_id: str
    span: tuple[int, int]  # char offsets into canonical doc.md
    section_path: list[str]  # ["Ch 3", "3.2 Feedback Loops"]
    extractor: str  # stage name
    model: str  # e.g. "claude-sonnet-x"; "" only for pure-python extractors
    prompt_rev: str  # hash from promptlib; "" only for pure-python extractors

    @field_validator("span")
    @classmethod
    def _span_is_sane(cls, v: tuple[int, int]) -> tuple[int, int]:
        start, end = v
        if start < 0 or end <= start:
            raise ValueError(f"span must satisfy 0 <= start < end, got {v}")
        return v


class TocNode(NoemaModel):
    title: str
    level: int = Field(ge=1, le=4)  # normalized H1-H4 hierarchy
    span: tuple[int, int]  # char range this heading's section covers
    children: list[TocNode] = Field(default_factory=list)


class StructureMap(NoemaModel):
    """The addressing system for ALL provenance."""

    toc: list[TocNode]
    page_offsets: list[int]  # char offset of each source page start


# ── P1 output ────────────────────────────────────────────────────────────────


class DocReport(NoemaModel):
    fidelity_mean: float = Field(ge=0.0, le=5.0)  # rubric_judge over sampled pages
    span_match: float = Field(ge=0.0, le=1.0)  # fuzzy-match rate on random spans
    sampled_pages: list[int] = Field(default_factory=list)
    rescued_pages: list[int] = Field(default_factory=list)
    flagged_pages: list[int] = Field(default_factory=list)  # failed rescue, shipped anyway
    warnings: list[str] = Field(default_factory=list)


class CanonicalDoc(NoemaModel):
    source: SourceRef
    md_path: Path
    structure: StructureMap
    report: DocReport


# ── P2 ontology parts ────────────────────────────────────────────────────────


class LayeredAbstract(NoemaModel):
    one_liner: str  # <= 30 words
    paragraph: str  # ~150 words
    page: str  # ~600 words
    argument_spine: list[str]  # ordered chain of the book's core claims
    chapter_abstracts: dict[str, str]


class EntityMention(NoemaModel):
    """Raw mention from the two-channel harvest (GLiNER sweep + LLM extraction)."""

    src_id: str
    surface: str  # exact text as it appears
    kind: str  # open vocabulary guess
    span: tuple[int, int]
    section_path: list[str]
    channel: Literal["gliner", "llm"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Entity(NoemaModel):
    ent_id: str
    canonical_name: str
    aliases: list[str]
    kind: str  # open vocabulary: concept|person|method|system|metric|...
    definition: str  # synthesized, <= 60 words
    idiosyncratic: bool  # author-coined term? (e.g. "antifragile")
    provenance: list[Provenance] = Field(min_length=1)


class ThesaurusEdge(NoemaModel):
    subject: str  # ent_id
    predicate: Literal["broader", "narrower", "related", "synonym", "antonym", "part_of"]
    object: str  # ent_id
    provenance: list[Provenance] = Field(default_factory=list)  # kNN-derived edges may lack spans


class Axiom(NoemaModel):
    ax_id: str
    statement: str  # normalized declarative form
    formality: Literal["definition", "law", "principle", "heuristic", "empirical_claim"]
    modality: Literal["always", "typically", "sometimes"]
    scope: str  # conditions under which the author asserts it
    author_credence: float = Field(ge=0.0, le=1.0)  # the AUTHOR's confidence, rubric-elicited
    entities: list[str]  # ent_ids referenced
    provenance: list[Provenance] = Field(min_length=1)  # each span must entail `statement`


class CausalLink(NoemaModel):
    src_var: str
    dst_var: str
    polarity: Literal["+", "-", "~"]  # reinforcing, opposing, nonmonotonic


class MentalModel(NoemaModel):
    mm_id: str
    name: str  # "Feedback Loops", "Antifragility", "Selection Pressure"
    one_liner: str
    variables: list[str]
    dynamics: list[CausalLink]
    predicts: list[str]  # what the model lets you anticipate
    fails_when: list[str]  # known breakdown conditions
    transfer_notes: str  # where the author claims/implies it generalizes
    worked_examples: list[Provenance] = Field(min_length=1)


class OntologyReport(NoemaModel):
    """The three gate numbers of P2 (BLUEPRINT §4)."""

    qa_fidelity: float = Field(ge=0.0, le=1.0)  # closed-book QA faithfulness
    groundedness: float = Field(ge=0.0, le=1.0)  # fraction of axioms entailment-validated
    coverage: float = Field(ge=0.0, le=1.0)  # canonical entities vs GLiNER sweep
    notes: list[str] = Field(default_factory=list)


class DomainOntology(NoemaModel):
    source: SourceRef
    abstract: LayeredAbstract
    entities: list[Entity]
    thesaurus: list[ThesaurusEdge]
    axioms: list[Axiom]
    mental_models: list[MentalModel]
    fidelity: OntologyReport


# ── P3 synthesis + intentional stance ────────────────────────────────────────


class SameAsLink(NoemaModel):
    left: tuple[str, str]  # (src_id, ent_id)
    right: tuple[str, str]
    confidence: float = Field(ge=0.0, le=1.0)


class Tension(NoemaModel):
    t_id: str
    axioms: list[tuple[str, str]] = Field(min_length=2)  # (src_id, ax_id) pairs in conflict
    kind: Literal["contradiction", "scope_overlap", "value_conflict"]
    gloss: str  # one-paragraph statement of the disagreement


class AnalogyBridge(NoemaModel):
    """Gentner structure-mapping across domains."""

    b_id: str
    source_model: tuple[str, str]  # (src_id, mm_id)
    target_model: tuple[str, str]
    mapping: dict[str, str]  # variable-to-variable relational alignment
    strength: float = Field(ge=0.0, le=1.0)  # relational overlap score
    breaks_when: list[str]  # disanalogy conditions — as important as the mapping


class Goal(NoemaModel):
    statement: str
    priority: int
    derived_from: list[str]  # argument-spine claims that motivate it


class Belief(NoemaModel):
    axiom_ref: tuple[str, str]  # (src_id, ax_id)
    credence: float = Field(ge=0.0, le=1.0)  # the FRAME's credence (may differ from author's)
    justification: str


class TensionStance(NoemaModel):
    tension_id: str
    stance: Literal["adopt_left", "adopt_right", "contextualize", "hold_open"]
    rationale: str


class EpistemicPolicy(NoemaModel):
    humility_rule: str  # when to say "outside my doxastic base"
    citation_rule: str  # when answers must cite provenance spans
    uncertainty_expression: str  # how credences surface in language
    deference: list[str]  # topics on which the frame defers to the user/tools


class FrameReport(NoemaModel):
    """Gym probe-family scores (BLUEPRINT §5.4). All rubric_judge-scored, 0-5."""

    consistency: float = Field(ge=0.0, le=5.0)
    humility: float = Field(ge=0.0, le=5.0)
    transfer: float = Field(ge=0.0, le=5.0)
    tension_stability: float = Field(ge=0.0, le=5.0)
    notes: list[str] = Field(default_factory=list)

    def releasable(self, min_score: float = 4.0) -> bool:
        return all(
            s >= min_score
            for s in (self.consistency, self.humility, self.transfer, self.tension_stability)
        )


class CompiledFrame(NoemaModel):
    system_prompt_path: Path  # <= ~2k tokens, generated
    retrieval_pack: Path  # LanceDB slice of provenance-linked chunks
    tool_manifest_path: Path  # MCP-style tools.json
    probe_suite_path: Path
    gym_report: FrameReport | None = None


class AgentFrame(NoemaModel):
    frame_id: str
    archetype: Literal["author_simulacrum", "synthetic_polymath"]
    identity: str  # 2-3 sentence self-description
    telos: list[Goal]
    doxastic_base: list[Belief]
    inferential_repertoire: list[str]  # mm_ids + bridge b_ids, ranked by applicability
    lexicon: list[str]  # ent_ids the frame speaks in, idiosyncratic terms first
    tension_stances: list[TensionStance]
    epistemic_policy: EpistemicPolicy
    compiled: CompiledFrame | None = None


# ── judge primitives output contracts (BLUEPRINT §2.6) ───────────────────────


class RubricVerdict(NoemaModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class EntailmentVerdict(NoemaModel):
    entails: bool
    rationale: str


class QAPair(NoemaModel):
    question: str
    answer: str


class QAPairSet(NoemaModel):
    pairs: list[QAPair] = Field(min_length=1)


class QAAnswer(NoemaModel):
    answer: str
    answerable: bool = True  # False -> artifact lacks the information


class FaithfulnessVerdict(NoemaModel):
    score: float = Field(ge=0.0, le=1.0)  # 1 faithful, 0.5 partial, 0 unfaithful/missing
    rationale: str


# ── schema registry (used by promptlib `output_schema` frontmatter refs) ─────


def schema_registry() -> dict[str, type[BaseModel]]:
    """Name -> class for every contract defined in this module."""
    return {
        name: obj
        for name, obj in globals().items()
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj.__module__ == __name__
    }


def resolve_schema(name: str) -> type[BaseModel]:
    registry = schema_registry()
    try:
        return registry[name]
    except KeyError:
        available = ", ".join(sorted(registry))
        raise KeyError(f"unknown output_schema {name!r}; available: {available}") from None
