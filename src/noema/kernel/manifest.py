"""AI-native make (BLUEPRINT §2.4): lineage + content-addressed stage cache.

A stage re-runs iff its key changes:

    key = hash(input_artifact_hashes, code_rev, prompt_revs, tier_bindings, params)

Backed by lineage.duckdb, which also holds the LLM call trace (every gateway
call: tokens, cost, latency) and gate scores. Connections are opened per
operation so the single DuckDB file is shared safely with stores.py.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, ConfigDict

from noema.kernel.config import Settings
from noema.kernel.promptlib import PromptAsset

# ── hashing helpers ──────────────────────────────────────────────────────────


def file_hash(path: Path | str) -> str:
    """sha256 hex digest of a file's bytes. src_id = file_hash(pdf)[:12]."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def obj_hash(obj: Any) -> str:
    """sha256 hex digest of a canonical JSON rendering of any jsonable/pydantic value."""
    if isinstance(obj, BaseModel):
        payload = obj.model_dump_json()
    else:
        payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_rev(fn: Callable[..., Any]) -> str:
    """Revision hash of a stage function's source; falls back to qualname."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = getattr(fn, "__qualname__", repr(fn))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]


def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, str]:
    """Coerce input values to stable hashes: Paths hash file bytes, models hash JSON."""
    out: dict[str, str] = {}
    for name, value in inputs.items():
        if isinstance(value, Path):
            out[name] = file_hash(value)
        elif isinstance(value, BaseModel):
            out[name] = obj_hash(value)
        elif isinstance(value, str):
            out[name] = value  # caller already supplies a hash/id
        else:
            out[name] = obj_hash(value)
    return out


def _output_default(value: Any) -> str:
    """Stage outputs may contain Paths/datetimes (stringified); anything else
    non-jsonable is a bug, not something to silently repr into the lineage."""
    if isinstance(value, Path | datetime):
        return str(value)
    raise TypeError(f"type {type(value).__name__} is not JSON-serializable")


def new_run_id() -> str:
    return f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


# ── records ──────────────────────────────────────────────────────────────────


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    stage: str
    outputs: dict[str, Any]
    cached: bool
    duration_s: float | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stage_runs (
    key         TEXT PRIMARY KEY,
    stage       TEXT NOT NULL,
    run_id      TEXT,
    inputs      TEXT,
    params      TEXT,
    code_rev    TEXT,
    prompt_revs TEXT,
    tiers       TEXT,
    outputs     TEXT,
    status      TEXT,
    created_at  TEXT,
    duration_s  DOUBLE
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id            TEXT,
    ts            TEXT,
    run_id        TEXT,
    stage         TEXT,
    tier          TEXT,
    model         TEXT,
    prompt_id     TEXT,
    prompt_rev    TEXT,
    input_tokens  BIGINT,
    output_tokens BIGINT,
    cost_usd      DOUBLE,
    latency_ms    DOUBLE,
    ok            BOOLEAN,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS scores (
    ts     TEXT,
    run_id TEXT,
    stage  TEXT,
    name   TEXT,
    value  DOUBLE,
    meta   TEXT
);
"""


class Manifest:
    """Lineage store + stage cache over lineage.duckdb."""

    def __init__(self, settings: Settings, run_id: str | None = None) -> None:
        self.settings = settings
        self.run_id = run_id or new_run_id()
        self.db_path = settings.paths.lineage_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.execute(_SCHEMA)

    def _con(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    # ── stage cache ──────────────────────────────────────────────────────────

    @staticmethod
    def stage_key(
        stage: str,
        inputs: dict[str, str],
        params: dict[str, Any],
        code: str,
        prompt_revs: dict[str, str],
        tier_bindings: dict[str, str],
    ) -> str:
        payload = json.dumps(
            {
                "stage": stage,
                "inputs": inputs,
                "params": params,
                "code_rev": code,
                "prompt_revs": prompt_revs,
                "tiers": tier_bindings,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def lookup(self, key: str) -> dict[str, Any] | None:
        with self._con() as con:
            row = con.execute(
                "SELECT outputs FROM stage_runs WHERE key = ? AND status = 'ok'", [key]
            ).fetchone()
        return json.loads(row[0]) if row else None

    def run(
        self,
        stage: str,
        fn: Callable[[], dict[str, Any]],
        *,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
        prompts: Iterable[PromptAsset] = (),
        tiers: Iterable[str] = (),
        force: bool = False,
    ) -> StageResult:
        """Run ``fn`` (returning a jsonable outputs dict) behind the cache.

        ``inputs`` values may be Paths (hashed by content), pydantic models
        (hashed by canonical JSON), or precomputed hash/id strings.
        """
        params = params or {}
        norm_inputs = _normalize_inputs(inputs)
        prompt_revs = {p.id: p.rev for p in prompts}
        bindings = self.settings.models.bindings(tuple(tiers)) if tiers else {}
        key = self.stage_key(stage, norm_inputs, params, code_rev(fn), prompt_revs, bindings)

        if not force:
            cached = self.lookup(key)
            if cached is not None:
                return StageResult(key=key, stage=stage, outputs=cached, cached=True)

        t0 = time.perf_counter()
        outputs = fn()
        duration = time.perf_counter() - t0
        try:
            outputs_json = json.dumps(outputs, default=_output_default)
        except TypeError as e:
            raise TypeError(f"stage {stage!r} outputs must be JSON-serializable: {e}") from e

        with self._con() as con:
            con.execute(
                "INSERT OR REPLACE INTO stage_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    key,
                    stage,
                    self.run_id,
                    json.dumps(norm_inputs, sort_keys=True),
                    json.dumps(params, sort_keys=True, default=str),
                    code_rev(fn),
                    json.dumps(prompt_revs, sort_keys=True),
                    json.dumps(bindings, sort_keys=True),
                    outputs_json,
                    "ok",
                    datetime.now(UTC).isoformat(),
                    duration,
                ],
            )
        return StageResult(key=key, stage=stage, outputs=outputs, cached=False, duration_s=duration)

    # ── lineage: LLM call trace + gate scores ────────────────────────────────

    def log_llm_call(
        self,
        *,
        stage: str,
        tier: str,
        model: str,
        prompt_id: str = "",
        prompt_rev: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        ok: bool = True,
        error: str = "",
    ) -> None:
        with self._con() as con:
            con.execute(
                "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    uuid.uuid4().hex,
                    datetime.now(UTC).isoformat(),
                    self.run_id,
                    stage,
                    tier,
                    model,
                    prompt_id,
                    prompt_rev,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_ms,
                    ok,
                    error,
                ],
            )

    def log_score(self, *, stage: str, name: str, value: float, meta: dict | None = None) -> None:
        with self._con() as con:
            con.execute(
                "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)",
                [
                    datetime.now(UTC).isoformat(),
                    self.run_id,
                    stage,
                    name,
                    value,
                    json.dumps(meta or {}, default=str),
                ],
            )

    def calls(self, run_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM llm_calls"
        args: list[Any] = []
        if run_id:
            query += " WHERE run_id = ?"
            args.append(run_id)
        query += " ORDER BY ts"
        with self._con() as con:
            cur = con.execute(query, args)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def spend(self, run_id: str | None = None) -> float:
        query = "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_calls"
        args: list[Any] = []
        if run_id:
            query += " WHERE run_id = ?"
            args.append(run_id)
        with self._con() as con:
            return float(con.execute(query, args).fetchone()[0])

    def status(self, limit: int = 50) -> dict[str, Any]:
        """Lineage view for `noema status`: cached stages + spend."""
        with self._con() as con:
            cur = con.execute(
                """
                SELECT stage, key, status, created_at, duration_s, run_id
                FROM stage_runs ORDER BY created_at DESC LIMIT ?
                """,
                [limit],
            )
            cols = [d[0] for d in cur.description]
            stages = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
            n_calls, total = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0.0) FROM llm_calls"
            ).fetchone()
        return {"stages": stages, "llm_calls": int(n_calls), "spend_usd": float(total)}
