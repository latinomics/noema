"""Eval primitives. Includes the M0 DoD: one rubric_judge call traced end-to-end
(prompt asset -> gateway -> JUDGE tier -> validated verdict -> lineage row)."""

from __future__ import annotations

import json

import duckdb
import pytest

from noema.kernel import judge
from noema.kernel.schemas import RubricVerdict
from tests.conftest import make_gateway


def test_rubric_judge_traced_end_to_end(settings, manifest):
    gw, fake = make_gateway(
        settings, manifest, ['{"score": 4, "rationale": "headings preserved; one table mangled"}']
    )
    verdict = judge.rubric_judge(
        gw,
        artifact="# Ch 3\n\n3.2 Feedback Loops\n...",
        rubric="Structure fidelity: headings, tables, equations survive conversion.",
        stage="p1.gate",
    )
    assert isinstance(verdict, RubricVerdict)
    assert verdict.score == 4

    # the rendered prompt actually carried rubric + artifact to the model
    sent = json.dumps(fake.calls[0]["messages"])
    assert "Structure fidelity" in sent and "Feedback Loops" in sent

    # end-to-end trace: one lineage row with full provenance of the call
    rows = manifest.calls()
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "p1.gate"
    assert row["tier"] == "JUDGE"
    assert row["model"] == "gpt-4o-mini"
    assert row["prompt_id"] == "judge/rubric"
    assert row["prompt_rev"] == gw.prompts.get("judge/rubric").rev
    assert row["ok"]
    assert row["input_tokens"] > 0 and row["output_tokens"] > 0
    assert row["cost_usd"] > 0
    assert row["latency_ms"] > 0


def test_entails_true_and_false(settings, manifest):
    gw, _ = make_gateway(
        settings,
        manifest,
        [
            '{"entails": true, "rationale": "span states the claim directly"}',
            '{"entails": false, "rationale": "span only mentions the topic"}',
        ],
    )
    span = "Delays in balancing feedback loops make a system likelier to oscillate."
    assert judge.entails(gw, span, "Delays in balancing feedback loops cause oscillation.") is True
    assert judge.entails(gw, span, "Delays always destroy systems.") is False


def test_qa_fidelity_closed_book_loop(settings, manifest):
    pairs = json.dumps(
        {
            "pairs": [
                {"question": "What acts as a system's memory?", "answer": "Stocks."},
                {"question": "What causes oscillation?", "answer": "Delayed feedback."},
            ]
        }
    )
    gw, fake = make_gateway(
        settings,
        manifest,
        [
            pairs,
            '{"answer": "Stocks act as the memory of a system.", "answerable": true}',
            '{"score": 1.0, "rationale": "same fact"}',
            '{"answer": "the ontology does not cover this", "answerable": false}',
            '{"score": 0.0, "rationale": "missing"}',
        ],
    )
    score = judge.qa_fidelity(gw, source_text="...", artifact_text="...", k=2)
    assert score == pytest.approx(0.5)

    # unanswerable answers are graded as the NOT IN ARTIFACT sentinel
    last_grading_call = json.dumps(fake.calls[-1]["messages"])
    assert judge.NOT_IN_ARTIFACT in last_grading_call

    # the composite score lands in the lineage scores table
    with duckdb.connect(str(settings.paths.lineage_db)) as con:
        name, value = con.execute("SELECT name, value FROM scores").fetchone()
    assert name == "qa_fidelity"
    assert value == pytest.approx(0.5)
