"""Contract-layer tests: provenance is type-checked, bounds hold, registry resolves."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from noema.kernel.schemas import (
    AgentFrame,
    Axiom,
    Belief,
    DomainOntology,
    Entity,
    EpistemicPolicy,
    Goal,
    MentalModel,
    Provenance,
    RubricVerdict,
    Tension,
    make_id,
    resolve_schema,
)
from tests.conftest import SRC_ID, prov


def test_axiom_without_provenance_is_a_validation_error():
    with pytest.raises(ValidationError, match="provenance"):
        Axiom(
            ax_id="ax_x",
            statement="Ungrounded knowledge.",
            formality="principle",
            modality="typically",
            scope="anywhere",
            author_credence=0.5,
            entities=[],
            provenance=[],
        )


def test_entity_without_provenance_is_a_validation_error():
    with pytest.raises(ValidationError, match="provenance"):
        Entity(
            ent_id="ent_x",
            canonical_name="ghost",
            aliases=[],
            kind="concept",
            definition="An entity with no textual grounding.",
            idiosyncratic=False,
            provenance=[],
        )


def test_mental_model_requires_worked_example():
    with pytest.raises(ValidationError, match="worked_examples"):
        MentalModel(
            mm_id="mm_x",
            name="Ungrounded Model",
            one_liner="",
            variables=[],
            dynamics=[],
            predicts=[],
            fails_when=[],
            transfer_notes="",
            worked_examples=[],
        )


@pytest.mark.parametrize("span", [(5, 5), (10, 3), (-1, 4)])
def test_provenance_span_must_be_ordered_and_nonnegative(span):
    with pytest.raises(ValidationError):
        Provenance(
            src_id=SRC_ID,
            span=span,
            section_path=[],
            extractor="x",
            model="m",
            prompt_rev="r",
        )


def test_credence_bounds():
    with pytest.raises(ValidationError):
        Axiom(
            ax_id="ax_x",
            statement="s",
            formality="law",
            modality="always",
            scope="",
            author_credence=1.5,
            entities=[],
            provenance=[prov()],
        )
    with pytest.raises(ValidationError):
        Belief(axiom_ref=(SRC_ID, "ax_x"), credence=-0.1, justification="")


def test_tension_needs_at_least_two_axioms():
    with pytest.raises(ValidationError):
        Tension(t_id="t_1", axioms=[(SRC_ID, "ax_1")], kind="contradiction", gloss="only one side")


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        RubricVerdict(score=3, rationale="ok", vibes="excellent")


def test_ontology_json_roundtrip(ontology: DomainOntology):
    restored = DomainOntology.model_validate_json(ontology.model_dump_json())
    assert restored == ontology


def test_agent_frame_roundtrip():
    frame = AgentFrame(
        frame_id="frame_meadows",
        archetype="author_simulacrum",
        identity="A systems thinker who sees structure behind events.",
        telos=[Goal(statement="Expose structural causes.", priority=1, derived_from=["spine-2"])],
        doxastic_base=[
            Belief(axiom_ref=(SRC_ID, "ax_delay"), credence=0.85, justification="Well-evidenced.")
        ],
        inferential_repertoire=["mm_balancing_loop"],
        lexicon=["ent_stock", "ent_feedback"],
        tension_stances=[],
        epistemic_policy=EpistemicPolicy(
            humility_rule="Decline questions outside the doxastic base.",
            citation_rule="Cite provenance spans for every factual claim.",
            uncertainty_expression="State credences qualitatively.",
            deference=["current events"],
        ),
    )
    assert AgentFrame.model_validate_json(frame.model_dump_json()) == frame
    assert frame.compiled is None


def test_make_id_deterministic():
    a = make_id("ax", SRC_ID, "some statement")
    b = make_id("ax", SRC_ID, "some statement")
    c = make_id("ax", SRC_ID, "another statement")
    assert a == b
    assert a != c
    assert a.startswith("ax_") and len(a) == 13


def test_schema_registry_resolves():
    assert resolve_schema("RubricVerdict") is RubricVerdict
    with pytest.raises(KeyError, match="available"):
        resolve_schema("NotASchema")
