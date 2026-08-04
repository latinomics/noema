"""CLI surface: `status` works against the lineage store; pipeline commands
declare their milestone instead of pretending."""

from __future__ import annotations

from typer.testing import CliRunner

from noema.cli import app
from tests.conftest import PROMPTS_DIR

runner = CliRunner()


def _env(tmp_path):
    return {
        "NOEMA_PATHS__DATA_DIR": str(tmp_path / "data"),
        "NOEMA_PATHS__PROMPTS_DIR": str(PROMPTS_DIR),
    }


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


def test_pipeline_commands_point_at_their_milestone(tmp_path):
    result = runner.invoke(app, ["extract", "abc123def456"], env=_env(tmp_path))
    assert result.exit_code == 1
    assert "M2" in result.output
