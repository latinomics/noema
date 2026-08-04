"""Shared fixtures: tmp-dir settings, a scripted LLM transport, and a synthetic
Meadows-flavored DomainOntology used for store round-trips."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import litellm
import pytest

from noema.kernel.config import (
    BudgetConfig,
    EmbeddingsConfig,
    ModelsConfig,
    PathsConfig,
    Settings,
    TierConfig,
)
from noema.kernel.gateway import Gateway
from noema.kernel.manifest import Manifest
from noema.kernel.schemas import (
    Axiom,
    CausalLink,
    DomainOntology,
    Entity,
    LayeredAbstract,
    MentalModel,
    OntologyReport,
    Provenance,
    SourceRef,
    ThesaurusEdge,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"

SRC_ID = "abc123def456"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        models=ModelsConfig(
            map=TierConfig(model="gpt-4o-mini", temperature=0.3),
            reduce=TierConfig(model="gpt-4o-mini", temperature=0.4),
            # deliberately nonzero: the gateway must force JUDGE calls to 0
            judge=TierConfig(model="gpt-4o-mini", temperature=0.7),
        ),
        budget=BudgetConfig(run_usd=5.0, warn_fraction=0.8),
        paths=PathsConfig(data_dir=tmp_path / "data", prompts_dir=PROMPTS_DIR),
        embeddings=EmbeddingsConfig(provider="hash", hash_dim=64),
    )
    s.paths.ensure()
    return s


@pytest.fixture
def manifest(settings: Settings) -> Manifest:
    return Manifest(settings, run_id="run-test")


class ScriptedCompletion:
    """Injectable stand-in for litellm.completion.

    Each script item is either a string (returned as a mock completion via
    litellm's offline mock_response path, so usage/cost behave realistically)
    or an exception instance (raised). Records every call's kwargs.
    """

    def __init__(self, script: list[str | Exception]) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> object:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("ScriptedCompletion: script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return litellm.completion(
            model=kwargs.get("model", "gpt-4o-mini"),
            messages=kwargs.get("messages", [{"role": "user", "content": "x"}]),
            mock_response=item,
        )


def make_gateway(
    settings: Settings, manifest: Manifest | None, script: list[str | Exception], **kw
) -> tuple[Gateway, ScriptedCompletion]:
    fake = ScriptedCompletion(script)
    gw = Gateway(settings, manifest=manifest, completion_fn=fake, backoff_base_s=0.01, **kw)
    return gw, fake


def prov(span: tuple[int, int] = (100, 480), section: str = "1.2 Feedback") -> Provenance:
    return Provenance(
        src_id=SRC_ID,
        span=span,
        section_path=["Ch 1", section],
        extractor="ontogen.axiom_mine",
        model="claude-sonnet-x",
        prompt_rev="deadbeef",
    )


@pytest.fixture
def ontology() -> DomainOntology:
    entities = [
        Entity(
            ent_id="ent_stock",
            canonical_name="stock",
            aliases=["level", "accumulation"],
            kind="concept",
            definition=(
                "An accumulation of material or information built up over time in a system."
            ),
            idiosyncratic=False,
            provenance=[prov((120, 260), "1.1 Stocks")],
        ),
        Entity(
            ent_id="ent_flow",
            canonical_name="flow",
            aliases=["rate"],
            kind="concept",
            definition="A rate of change that fills or drains a stock over time.",
            idiosyncratic=False,
            provenance=[prov((300, 410), "1.1 Stocks")],
        ),
        Entity(
            ent_id="ent_feedback",
            canonical_name="feedback loop",
            aliases=["loop"],
            kind="mechanism",
            definition=(
                "A closed chain of causal connections from a stock "
                "through decisions back to the stock."
            ),
            idiosyncratic=False,
            provenance=[prov((500, 700), "1.2 Feedback")],
        ),
    ]
    thesaurus = [
        ThesaurusEdge(
            subject="ent_flow", predicate="related", object="ent_stock", provenance=[prov()]
        ),
        ThesaurusEdge(
            subject="ent_feedback", predicate="broader", object="ent_stock", provenance=[]
        ),
    ]
    axioms = [
        Axiom(
            ax_id="ax_delay",
            statement="Delays in balancing feedback loops cause oscillation.",
            formality="principle",
            modality="typically",
            scope="systems whose corrective response lags the signal",
            author_credence=0.9,
            entities=["ent_feedback"],
            provenance=[prov((820, 1040), "1.3 Delays")],
        ),
        Axiom(
            ax_id="ax_stock_memory",
            statement="Stocks act as the memory of a system, decoupling inflows from outflows.",
            formality="definition",
            modality="always",
            scope="any stock-flow system",
            author_credence=0.95,
            entities=["ent_stock", "ent_flow"],
            provenance=[prov((150, 260), "1.1 Stocks")],
        ),
    ]
    models = [
        MentalModel(
            mm_id="mm_balancing_loop",
            name="Balancing Feedback Loop",
            one_liner="A goal-seeking structure that counteracts deviation from a target.",
            variables=["stock", "goal", "gap", "corrective_flow"],
            dynamics=[
                CausalLink(src_var="gap", dst_var="corrective_flow", polarity="+"),
                CausalLink(src_var="corrective_flow", dst_var="stock", polarity="+"),
                CausalLink(src_var="stock", dst_var="gap", polarity="-"),
            ],
            predicts=["convergence toward the goal", "oscillation when delays are long"],
            fails_when=["the corrective flow saturates", "the goal itself drifts"],
            transfer_notes="Thermostats, inventory management, glucose regulation.",
            worked_examples=[prov((1100, 1500), "1.4 Thermostat")],
        )
    ]
    return DomainOntology(
        source=SourceRef(
            src_id=SRC_ID,
            title="Thinking in Systems",
            authors=["Donella Meadows"],
            filed_at=datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC),
        ),
        abstract=LayeredAbstract(
            one_liner="Systems produce their own behavior through stocks, flows, and feedback.",
            paragraph="Systems are more than the sum of their parts. " * 5,
            page="Structure drives behavior. " * 60,
            argument_spine=[
                "Systems consist of stocks, flows, and feedback loops.",
                "System structure, not external events, produces system behavior.",
                "Delays in feedback create oscillation.",
                "Leverage comes from changing structure, not parameters.",
            ],
            chapter_abstracts={"Ch 1": "Stocks, flows, and feedback fundamentals."},
        ),
        entities=entities,
        thesaurus=thesaurus,
        axioms=axioms,
        mental_models=models,
        fidelity=OntologyReport(qa_fidelity=0.85, groundedness=1.0, coverage=0.9),
    )
