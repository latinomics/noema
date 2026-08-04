"""Eval primitives (BLUEPRINT §2.6). Every pipeline gate is built from these.

- ``rubric_judge``: LLM-as-judge against a written rubric -> 1-5 + rationale
- ``entails``: NLI-style check; validates every axiom's provenance spans
- ``qa_probe`` / ``answer_from`` / ``faithful``: the closed-book fidelity loop

All run on the JUDGE tier (temperature 0, enforced by the gateway). A gate
score is a number, not a vibe.
"""

from __future__ import annotations

import logging

from noema.kernel.gateway import Gateway
from noema.kernel.schemas import (
    EntailmentVerdict,
    FaithfulnessVerdict,
    QAAnswer,
    QAPair,
    QAPairSet,
    RubricVerdict,
)

logger = logging.getLogger(__name__)

PROMPT_RUBRIC = "judge/rubric"
PROMPT_ENTAILS = "judge/entails"
PROMPT_QA_GENERATE = "judge/qa_generate"
PROMPT_QA_ANSWER = "judge/qa_answer"
PROMPT_QA_FAITHFUL = "judge/qa_faithful"

NOT_IN_ARTIFACT = "NOT IN ARTIFACT"


def rubric_judge(
    gw: Gateway, artifact: str, rubric: str, *, stage: str = "judge.rubric"
) -> RubricVerdict:
    """Score an artifact 1-5 against a written rubric."""
    return gw.structured(
        PROMPT_RUBRIC,
        {"rubric": rubric, "artifact": artifact},
        response_model=RubricVerdict,
        stage=stage,
    )


def entails_verdict(
    gw: Gateway, span_text: str, statement: str, *, stage: str = "judge.entails"
) -> EntailmentVerdict:
    return gw.structured(
        PROMPT_ENTAILS,
        {"span_text": span_text, "statement": statement},
        response_model=EntailmentVerdict,
        stage=stage,
    )


def entails(gw: Gateway, span_text: str, statement: str, *, stage: str = "judge.entails") -> bool:
    """True iff the span textually supports the statement. Used to validate
    every axiom's provenance — failures are dropped upstream, not stored."""
    return entails_verdict(gw, span_text, statement, stage=stage).entails


def qa_probe(
    gw: Gateway, source_text: str, k: int = 20, *, stage: str = "judge.qa_probe"
) -> list[QAPair]:
    """Generate k factual question/gold-answer pairs from source text."""
    result: QAPairSet = gw.structured(
        PROMPT_QA_GENERATE,
        {"source_text": source_text, "k": k},
        response_model=QAPairSet,
        stage=stage,
    )
    return result.pairs[:k]


def answer_from(
    gw: Gateway, artifact_text: str, question: str, *, stage: str = "judge.qa_answer"
) -> QAAnswer:
    """Answer a question using ONLY the artifact (closed book)."""
    return gw.structured(
        PROMPT_QA_ANSWER,
        {"artifact": artifact_text, "question": question},
        response_model=QAAnswer,
        stage=stage,
    )


def faithful(
    gw: Gateway, question: str, gold: str, pred: str, *, stage: str = "judge.qa_faithful"
) -> float:
    """Grade a predicted answer against gold: 1 faithful, 0.5 partial, 0 wrong/missing."""
    verdict: FaithfulnessVerdict = gw.structured(
        PROMPT_QA_FAITHFUL,
        {"question": question, "gold": gold, "pred": pred},
        response_model=FaithfulnessVerdict,
        stage=stage,
    )
    return verdict.score


def qa_fidelity(
    gw: Gateway,
    source_text: str,
    artifact_text: str,
    k: int = 20,
    *,
    stage: str = "judge.qa_fidelity",
) -> float:
    """The closed-book fidelity loop: questions from source, answers from the
    artifact only, mean faithfulness. P2 gate #1; reused by P3."""
    pairs = qa_probe(gw, source_text, k, stage=stage)
    scores: list[float] = []
    for pair in pairs:
        answer = answer_from(gw, artifact_text, pair.question, stage=stage)
        pred = answer.answer if answer.answerable else NOT_IN_ARTIFACT
        scores.append(faithful(gw, pair.question, pair.answer, pred, stage=stage))
    mean = sum(scores) / len(scores)
    if gw.manifest is not None:
        gw.manifest.log_score(stage=stage, name="qa_fidelity", value=mean, meta={"k": len(pairs)})
    return mean
