"""Gateway behavior: structured validation + repair, retries with jitter,
tier discipline, budget meter, trace emission."""

from __future__ import annotations

import logging

import litellm
import pytest

from noema.kernel.gateway import BudgetMeter, GatewayError
from noema.kernel.schemas import RubricVerdict
from tests.conftest import make_gateway

VALID_RUBRIC = '{"score": 4, "rationale": "solid structure"}'
RUBRIC_VARS = {"rubric": "Structure fidelity.", "artifact": "# Chapter 1\nText."}


def rate_limit() -> Exception:
    return litellm.exceptions.RateLimitError(
        message="rate limited", llm_provider="openai", model="gpt-4o-mini"
    )


def test_structured_resolves_schema_from_frontmatter(settings, manifest):
    gw, fake = make_gateway(settings, manifest, [VALID_RUBRIC])
    verdict = gw.structured("judge/rubric", RUBRIC_VARS, stage="test")
    assert isinstance(verdict, RubricVerdict)
    assert verdict.score == 4
    assert len(fake.calls) == 1


def test_judge_tier_temperature_forced_to_zero(settings, manifest):
    # fixture sets judge tier temperature to 0.7; the gateway must pin it to 0
    gw, fake = make_gateway(settings, manifest, [VALID_RUBRIC])
    gw.structured("judge/rubric", RUBRIC_VARS)
    assert fake.calls[0]["temperature"] == 0.0


def test_repair_pass_fixes_invalid_output(settings, manifest):
    gw, fake = make_gateway(settings, manifest, ["not json at all", VALID_RUBRIC])
    verdict = gw.structured("judge/rubric", RUBRIC_VARS)
    assert verdict.score == 4
    assert len(fake.calls) == 2  # initial + one repair pass


def test_validation_failure_after_repair_raises_and_traces(settings, manifest):
    gw, fake = make_gateway(settings, manifest, ["garbage", "still garbage"])
    with pytest.raises(GatewayError, match="judge/rubric"):
        gw.structured("judge/rubric", RUBRIC_VARS, stage="test.fail")
    assert len(fake.calls) == 2  # exactly one repair pass, then give up
    failed = [c for c in manifest.calls() if not c["ok"]]
    assert len(failed) == 1
    assert failed[0]["stage"] == "test.fail"
    assert failed[0]["error"]


def test_transient_errors_retry_with_backoff(settings, manifest):
    gw, fake = make_gateway(settings, manifest, [rate_limit(), rate_limit(), VALID_RUBRIC])
    verdict = gw.structured("judge/rubric", RUBRIC_VARS)
    assert verdict.score == 4
    assert len(fake.calls) == 3


def test_transient_exhaustion_raises_gateway_error(settings, manifest):
    gw, _ = make_gateway(settings, manifest, [rate_limit(), rate_limit()], max_attempts=2)
    with pytest.raises(GatewayError, match="exhausted 2 attempts"):
        gw.structured("judge/rubric", RUBRIC_VARS)


def test_text_path_returns_content_and_traces(settings, manifest):
    gw, _ = make_gateway(settings, manifest, ["transcribed page markdown"])
    out = gw.text("judge/rubric", RUBRIC_VARS, stage="p1.rescue")
    assert out == "transcribed page markdown"
    calls = manifest.calls()
    assert calls[-1]["stage"] == "p1.rescue"
    assert calls[-1]["ok"]


def test_budget_meter_warns_at_soft_and_hard_limits(caplog):
    meter = BudgetMeter(limit_usd=1.0, warn_fraction=0.8)
    with caplog.at_level(logging.WARNING, logger="noema.kernel.gateway"):
        meter.add(0.5)
        assert not caplog.records
        meter.add(0.35)  # 0.85 -> soft warn
        assert len(caplog.records) == 1
        assert "80%" in caplog.text
        meter.add(0.3)  # 1.15 -> hard warn
        assert len(caplog.records) == 2
        assert "EXCEEDED" in caplog.text
        meter.add(1.0)  # no repeat spam
        assert len(caplog.records) == 2


def test_gateway_accumulates_spend(settings, manifest):
    gw, _ = make_gateway(settings, manifest, [VALID_RUBRIC])
    assert gw.spent_usd == 0.0
    gw.structured("judge/rubric", RUBRIC_VARS)
    assert gw.spent_usd > 0.0  # litellm mock responses carry real usage/cost
    assert manifest.spend() == pytest.approx(gw.spent_usd)


def test_unknown_prompt_or_missing_schema_fail_loudly(settings, manifest):
    gw, _ = make_gateway(settings, manifest, [])
    with pytest.raises(KeyError, match="unknown prompt"):
        gw.structured("judge/does_not_exist", {})
