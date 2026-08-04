"""NOEMA CLI (BLUEPRINT §6). Every command is resumable via the manifest
cache; --force busts it. Pipeline commands land with their milestones; until
then the kernel (M0) is fully drivable from here: `doctor` preflights the
environment, `config`/`prompts` inspect it, and `call`/`judge` push real
traffic through the gateway with full lineage tracing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer
from rich.console import Console
from rich.table import Table

from noema.kernel.config import TIERS, Settings, Tier
from noema.kernel.gateway import Gateway, GatewayError
from noema.kernel.manifest import Manifest
from noema.kernel.promptlib import PromptError, PromptLib

app = typer.Typer(
    name="noema",
    help="NOEMA - books -> domain ontologies -> agents with a point of view.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _fail(msg: str) -> NoReturn:
    console.print(msg, style="red", markup=False)
    raise typer.Exit(code=1)


def _not_yet(command: str, milestone: str) -> None:
    console.print(
        f"[yellow]`noema {command}` lands with {milestone}. The kernel (M0) is live; "
        f"see BLUEPRINT.md for the build order.[/yellow]"
    )
    raise typer.Exit(code=1)


def _gateway(settings: Settings, manifest: Manifest) -> Gateway:
    """Gateway construction seam — tests swap in a scripted transport here."""
    return Gateway(settings, manifest=manifest)


def _kernel() -> tuple[Settings, Manifest, Gateway]:
    settings = Settings.load()
    settings.paths.ensure()
    manifest = Manifest(settings)
    return settings, manifest, _gateway(settings, manifest)


def _resolve_tier(tier: str | None) -> Tier | None:
    if tier is None:
        return None
    name = tier.upper()
    if name not in TIERS:
        raise typer.BadParameter(f"unknown tier {tier!r}; expected one of {', '.join(TIERS)}")
    return cast(Tier, name)


def _read_value(raw: str, *, what: str) -> str:
    """Values starting with @ are read from the named file."""
    if not raw.startswith("@"):
        return raw
    path = Path(raw[1:])
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise typer.BadParameter(f"cannot read {what} file {path}: {e}") from e


def _parse_vars(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected key=value (or key=@file), got {pair!r}")
        out[key] = _read_value(value, what=f"variable {key!r}")
    return out


def _spend_footer(gw: Gateway, tier: Tier, model: str) -> None:
    err_console.print(
        f"[dim]tier={tier} model={model} spend=${gw.spent_usd:.4f} "
        f"(run {gw.manifest.run_id if gw.manifest else '-'})[/dim]"
    )


# ── inspection ────────────────────────────────────────────────────────────────


@app.command()
def status(limit: int = typer.Option(50, help="Max stage rows to show.")) -> None:
    """Lineage view: what's cached, what's stale, spend."""
    settings = Settings.load()
    settings.paths.ensure()
    manifest = Manifest(settings)
    info = manifest.status(limit=limit)

    table = Table(title="stage cache", show_lines=False)
    for col in ("stage", "key", "status", "created_at", "duration_s"):
        table.add_column(col)
    for row in info["stages"]:
        table.add_row(
            str(row["stage"]),
            str(row["key"]),
            str(row["status"]),
            str(row["created_at"])[:19],
            f"{row['duration_s']:.2f}" if row["duration_s"] is not None else "-",
        )
    if info["stages"]:
        console.print(table)
    else:
        console.print("[dim]no stages recorded yet[/dim]")
    console.print(
        f"llm calls: [bold]{info['llm_calls']}[/bold]   "
        f"spend: [bold]${info['spend_usd']:.4f}[/bold]"
    )


@app.command()
def config() -> None:
    """Resolved settings: tiers, budget, paths, thresholds (after env overrides)."""
    toml_file = Path(os.environ.get("NOEMA_CONFIG", "noema.toml"))
    origin = "found" if toml_file.exists() else "missing -> defaults + NOEMA_* env only"
    settings = Settings.load()
    console.print(f"[dim]config file: {toml_file} ({origin})[/dim]")
    console.print_json(data=settings.model_dump(mode="json"))


@app.command()
def prompts() -> None:
    """List versioned prompt assets: id, tier, revision, schema, variables."""
    settings = Settings.load()
    try:
        lib = PromptLib(settings.paths.prompts_dir)
    except PromptError as e:
        _fail(str(e))
    assets = lib.all()
    if not assets:
        console.print(f"[dim]no prompt assets under {lib.root}[/dim]")
        return
    for asset in assets:
        schema = asset.output_schema or "-"
        console.print(f"[bold]{asset.id:<24}[/bold] {asset.tier:<7} rev={asset.rev}  {schema}")
        vars_note = ", ".join(sorted(asset.variables)) or "(none)"
        try:
            where = asset.path.relative_to(lib.root)
        except ValueError:
            where = asset.path
        console.print(f"    [dim]vars: {vars_note} · {where}[/dim]")
    console.print(f"[dim]{len(assets)} assets from {lib.root}[/dim]")


@app.command()
def doctor() -> None:
    """Preflight: config, data plane, prompts, lineage db, API keys, embeddings.

    Free and offline — no LLM calls. Exit 1 iff something is actually broken;
    missing credentials and optional extras only warn.
    """
    import litellm

    failed = False

    def report(status: str, name: str, detail: str) -> None:
        nonlocal failed
        color = {"ok": "green", "warn": "yellow", "fail": "red"}[status]
        failed = failed or status == "fail"
        console.print(f"[{color}]{status:<4}[/{color}] [bold]{name:<12}[/bold] {detail}")

    # config
    toml_file = Path(os.environ.get("NOEMA_CONFIG", "noema.toml"))
    try:
        settings = Settings.load()
        note = str(toml_file) if toml_file.exists() else f"{toml_file} missing (defaults + env)"
        report("ok", "config", note)
    except Exception as e:
        report("fail", "config", f"settings failed to load: {e}")
        raise typer.Exit(code=1) from e

    # data plane
    try:
        settings.paths.ensure()
        probe = settings.paths.data_dir / ".doctor-probe"
        probe.write_text("")
        probe.unlink()
        report("ok", "data", f"{settings.paths.data_dir} writable")
    except OSError as e:
        report("fail", "data", f"{settings.paths.data_dir}: {e}")

    # prompts
    try:
        lib = PromptLib(settings.paths.prompts_dir)
        report("ok", "prompts", f"{len(lib.all())} assets under {lib.root}")
    except PromptError as e:
        report("fail", "prompts", str(e))

    # lineage store
    try:
        Manifest(settings)
        report("ok", "lineage", str(settings.paths.lineage_db))
    except Exception as e:
        report("fail", "lineage", f"{settings.paths.lineage_db}: {e}")

    # model credentials, per tier
    for tier in TIERS:
        model = settings.models.for_tier(tier).model
        try:
            env_check = litellm.validate_environment(model=model)
            if env_check.get("keys_in_environment"):
                report("ok", f"model:{tier}", model)
            else:
                missing = ", ".join(env_check.get("missing_keys", [])) or "credentials"
                report("warn", f"model:{tier}", f"{model} — set {missing}")
        except Exception as e:
            report("warn", f"model:{tier}", f"{model} — could not validate: {e}")

    # embeddings
    emb = settings.embeddings
    if emb.provider == "local":
        from importlib.util import find_spec

        if find_spec("sentence_transformers") is not None:
            report("ok", "embeddings", f"local ({emb.model})")
        else:
            report(
                "warn",
                "embeddings",
                "provider=local but sentence-transformers missing — falls back to hash "
                "embedder (`uv sync --extra local-embeddings`)",
            )
    elif emb.provider == "api":
        if emb.api_model:
            report("ok", "embeddings", f"api ({emb.api_model})")
        else:
            report("fail", "embeddings", "provider=api but [embeddings].api_model is empty")
    else:
        report("ok", "embeddings", f"hash dim={emb.hash_dim} (deterministic, dev/tests only)")

    # tracing
    if settings.tracing.langfuse:
        langfuse_keys = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        missing = [k for k in langfuse_keys if not os.environ.get(k)]
        if missing:
            report("warn", "tracing", f"langfuse enabled — set {', '.join(missing)}")
        else:
            report("ok", "tracing", "langfuse enabled, keys present")

    if failed:
        raise typer.Exit(code=1)
    console.print("[dim]kernel is runnable — try `noema judge <file> --rubric ...`[/dim]")


# ── running the kernel ────────────────────────────────────────────────────────


@app.command()
def call(
    prompt_id: str = typer.Argument(..., help="Prompt asset id, e.g. judge/entails."),
    var: Annotated[
        list[str] | None,
        typer.Option(
            "--var", "-v", help="Template variable key=value; use key=@file to read a file."
        ),
    ] = None,
    tier: str | None = typer.Option(None, help="Override tier: MAP | REDUCE | JUDGE."),
    stage: str = typer.Option("cli.call", help="Stage name recorded in lineage."),
    text: bool = typer.Option(False, "--text", help="Plain-text completion, skip schema."),
) -> None:
    """Run one versioned prompt through the gateway (validated + traced).

    Structured when the asset declares an output_schema (JSON to stdout),
    plain text otherwise. Every call lands in lineage: `noema status` after.
    """
    _, _, gw = _kernel()
    try:
        asset = gw.prompts.get(prompt_id)
    except KeyError as e:
        _fail(str(e.args[0]) if e.args else str(e))
    variables = _parse_vars(var)
    tier_name = _resolve_tier(tier) or asset.tier

    try:
        if text or not asset.output_schema:
            out = gw.text(asset, variables, tier=tier_name, stage=stage)
            console.print(out, markup=False)
        else:
            parsed = gw.structured(asset, variables, tier=tier_name, stage=stage)
            console.print_json(data=parsed.model_dump(mode="json"))
    except (PromptError, GatewayError) as e:
        _fail(str(e))
    _spend_footer(gw, tier_name, gw.settings.models.for_tier(tier_name).model)


@app.command()
def judge(
    artifact: str = typer.Argument(..., help="Path to a text/markdown file, or '-' for stdin."),
    rubric: str = typer.Option(..., "--rubric", "-r", help="Rubric text, or @file to read one."),
    min_score: float | None = typer.Option(
        None, "--min", help="Exit 1 if the score falls below this (gate mode)."
    ),
    stage: str = typer.Option("cli.judge", help="Stage name recorded in lineage."),
) -> None:
    """Rubric-judge an artifact: one JUDGE-tier call -> score 1-5 + rationale.

    The QUICKSTART first-call as a one-liner; the verdict is traced to lineage.
    """
    from noema.kernel.judge import rubric_judge

    if artifact == "-":
        artifact_text = sys.stdin.read()
    else:
        path = Path(artifact)
        try:
            artifact_text = path.read_text(encoding="utf-8")
        except OSError as e:
            _fail(f"cannot read artifact {path}: {e}")
    if not artifact_text.strip():
        _fail("artifact is empty")
    rubric_text = _read_value(rubric, what="rubric")

    _, _, gw = _kernel()
    try:
        verdict = rubric_judge(gw, artifact_text, rubric_text, stage=stage)
    except (PromptError, GatewayError) as e:
        _fail(str(e))
    color = "green" if verdict.score >= 4 else "yellow" if verdict.score >= 3 else "red"
    console.print(f"score: [bold {color}]{verdict.score}/5[/bold {color}]")
    console.print(f"rationale: {verdict.rationale}", markup=False)
    _spend_footer(gw, "JUDGE", gw.settings.models.for_tier("JUDGE").model)
    if min_score is not None and verdict.score < min_score:
        console.print(f"[red]below --min {min_score:g}[/red]")
        raise typer.Exit(code=1)


# ── pipelines (land with their milestones) ────────────────────────────────────


@app.command()
def ingest(pdfs: list[Path]) -> None:
    """Register PDFs + run P1 (DOCFORGE)."""
    _not_yet("ingest", "M1 (DOCFORGE)")


@app.command()
def extract(src_id: str) -> None:
    """Run P2 (ONTOGEN) for one source."""
    _not_yet("extract", "M2 (ONTOGEN)")


@app.command()
def synthesize(
    frames: int = typer.Option(3, help="Number of synthetic_polymath frames."),
    authors: bool = typer.Option(False, "--authors", help="Also build author simulacra."),
) -> None:
    """Run P3 (NOESIS) fusion + intentional stance modeling."""
    _not_yet("synthesize", "M3/M4 (NOESIS)")


@app.command()
def gym(frame_id: str) -> None:
    """Interrogate a compiled frame (consistency / humility / transfer / tension)."""
    _not_yet("gym", "M4 (gym)")


@app.command()
def export(
    frame_id: str,
    target: str = typer.Option("claude-agent-sdk", help="claude-agent-sdk | langgraph | mcp"),
) -> None:
    """Export a frame bundle for an agent runtime."""
    _not_yet("export", "M4 (frame compiler)")


if __name__ == "__main__":
    app()
