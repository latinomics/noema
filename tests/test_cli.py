"""CLI surface: `status` works against the lineage store; `doctor`, `config`,
`prompts` inspect the environment offline; `call` and `judge` push traffic
through the gateway (scripted transport here — no network); pipeline commands
declare their milestone instead of pretending."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import noema.cli
from noema.cli import app
from tests.conftest import PROMPTS_DIR, make_gateway

runner = CliRunner()

VALID_RUBRIC = '{"score": 4, "rationale": "solid structure"}'


def _env(tmp_path):
    return {
        "NOEMA_PATHS__DATA_DIR": str(tmp_path / "data"),
        "NOEMA_PATHS__PROMPTS_DIR": str(PROMPTS_DIR),
    }


@pytest.fixture
def scripted_gateway(monkeypatch):
    """Patch the CLI's gateway seam with a scripted offline transport."""

    def install(script):
        def factory(settings, manifest):
            gw, _ = make_gateway(settings, manifest, script)
            return gw

        monkeypatch.setattr(noema.cli, "_gateway", factory)

    return install


# ── status ────────────────────────────────────────────────────────────────────


def test_status_on_empty_lineage(tmp_path):
    result = runner.invoke(app, ["status"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "no stages recorded" in result.output
    assert "spend" in result.output


def test_status_shows_recorded_stages(tmp_path):
    from noema.kernel.config import Settings
    from noema.kernel.manifest import Manifest

    env = _env(tmp_path)
    settings = Settings(paths={"data_dir": env["NOEMA_PATHS__DATA_DIR"]})
    settings.paths.ensure()
    manifest = Manifest(settings)
    manifest.run("p1.intake", lambda: {"src_id": "abc123def456"}, inputs={"pdf": "somehash"})
    manifest.log_llm_call(stage="p1.gate", tier="JUDGE", model="m", cost_usd=0.5)

    result = runner.invoke(app, ["status"], env=env)
    assert result.exit_code == 0
    assert "p1.intake" in result.output
    assert "$0.5000" in result.output


# ── inspection: config / prompts / doctor ─────────────────────────────────────


def test_config_shows_resolved_settings_with_env_overrides(tmp_path):
    env = {**_env(tmp_path), "NOEMA_MODELS__JUDGE__MODEL": "test/judge-model"}
    result = runner.invoke(app, ["config"], env=env)
    assert result.exit_code == 0
    assert "config file" in result.output
    assert "test/judge-model" in result.output
    assert "thresholds" in result.output


def test_prompts_lists_versioned_assets(tmp_path):
    result = runner.invoke(app, ["prompts"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "judge/rubric" in result.output
    assert "RubricVerdict" in result.output
    assert "5 assets" in result.output


def test_prompts_fails_cleanly_on_missing_dir(tmp_path):
    env = {**_env(tmp_path), "NOEMA_PATHS__PROMPTS_DIR": str(tmp_path / "nope")}
    result = runner.invoke(app, ["prompts"], env=env)
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_doctor_passes_on_healthy_env(tmp_path):
    # Missing API keys / optional extras only warn; the kernel itself is fine.
    result = runner.invoke(app, ["doctor"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "config" in result.output
    assert "prompts" in result.output
    assert "lineage" in result.output


def test_doctor_fails_when_prompts_are_broken(tmp_path):
    env = {**_env(tmp_path), "NOEMA_PATHS__PROMPTS_DIR": str(tmp_path / "nope")}
    result = runner.invoke(app, ["doctor"], env=env)
    assert result.exit_code == 1
    assert "fail" in result.output


# ── running the kernel: call / judge ──────────────────────────────────────────


def test_call_structured_prompt_prints_validated_json(tmp_path, scripted_gateway):
    scripted_gateway([VALID_RUBRIC])
    result = runner.invoke(
        app,
        ["call", "judge/rubric", "-v", "rubric=Structure fidelity.", "-v", "artifact=# Ch 1"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert '"score": 4' in result.output
    assert "solid structure" in result.output


def test_call_reads_variable_values_from_files(tmp_path, scripted_gateway):
    scripted_gateway([VALID_RUBRIC])
    chapter = tmp_path / "chapter.md"
    chapter.write_text("# Chapter 1\n\nStocks are accumulations.")
    result = runner.invoke(
        app,
        ["call", "judge/rubric", "-v", "rubric=Fidelity.", "-v", f"artifact=@{chapter}"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert '"score": 4' in result.output


def test_call_text_mode_skips_schema(tmp_path, scripted_gateway):
    scripted_gateway(["plain prose out"])
    result = runner.invoke(
        app,
        ["call", "judge/rubric", "--text", "-v", "rubric=r", "-v", "artifact=a"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "plain prose out" in result.output


def test_call_unknown_prompt_fails_with_available_ids(tmp_path, scripted_gateway):
    scripted_gateway([])
    result = runner.invoke(app, ["call", "nope/nothing"], env=_env(tmp_path))
    assert result.exit_code == 1
    assert "unknown prompt" in result.output


def test_call_missing_variables_lists_them(tmp_path, scripted_gateway):
    scripted_gateway([])  # must fail before any transport call
    result = runner.invoke(app, ["call", "judge/rubric", "-v", "rubric=only"], env=_env(tmp_path))
    assert result.exit_code == 1
    assert "missing variables" in result.output
    assert "artifact" in result.output


def test_call_rejects_malformed_var(tmp_path, scripted_gateway):
    scripted_gateway([])
    result = runner.invoke(app, ["call", "judge/rubric", "-v", "novalue"], env=_env(tmp_path))
    assert result.exit_code == 2
    assert "key=value" in result.output


def test_call_rejects_unknown_tier(tmp_path, scripted_gateway):
    scripted_gateway([])
    result = runner.invoke(app, ["call", "judge/rubric", "--tier", "TURBO"], env=_env(tmp_path))
    assert result.exit_code == 2
    assert "unknown tier" in result.output


def test_judge_scores_artifact_and_traces(tmp_path, scripted_gateway):
    scripted_gateway([VALID_RUBRIC])
    chapter = tmp_path / "chapter.md"
    chapter.write_text("# Chapter 1\n\nStocks are accumulations. Flows fill and drain them.")
    result = runner.invoke(
        app,
        ["judge", str(chapter), "--rubric", "Headings survive; prose is not garbled."],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "4/5" in result.output
    assert "solid structure" in result.output


def test_judge_min_gate_fails_below_threshold(tmp_path, scripted_gateway):
    scripted_gateway(['{"score": 3, "rationale": "wobbly headings"}'])
    chapter = tmp_path / "chapter.md"
    chapter.write_text("# Ch 1\ntext")
    result = runner.invoke(
        app,
        ["judge", str(chapter), "--rubric", "Strict.", "--min", "4"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "3/5" in result.output
    assert "below --min 4" in result.output


def test_judge_missing_artifact_fails_cleanly(tmp_path, scripted_gateway):
    scripted_gateway([])
    result = runner.invoke(
        app, ["judge", str(tmp_path / "ghost.md"), "--rubric", "r"], env=_env(tmp_path)
    )
    assert result.exit_code == 1
    assert "cannot read artifact" in result.output


# ── pipelines still gated by milestones ───────────────────────────────────────


def test_pipeline_commands_point_at_their_milestone(tmp_path):
    result = runner.invoke(app, ["extract", "abc123def456"], env=_env(tmp_path))
    assert result.exit_code == 1
    assert "M2" in result.output
