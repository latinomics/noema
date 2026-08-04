"""One door to all models (BLUEPRINT §2.2).

LiteLLM router + instructor for schema-validated outputs. Pipeline code speaks
tiers (MAP / REDUCE / JUDGE), never model strings. Responsibilities:

- retries with jitter on transient provider errors
- structured-output validation + one repair pass (instructor feeds the
  validation error back to the model once)
- per-run token/cost budget meter — logged to lineage, warns at 80%
- trace emission: every call lands in lineage.duckdb; optional Langfuse
  emission via litellm callbacks when [tracing].langfuse = true
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

from noema.kernel.config import Settings, Tier
from noema.kernel.manifest import Manifest
from noema.kernel.promptlib import PromptAsset, PromptLib
from noema.kernel.schemas import resolve_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE = tuple(
    exc
    for exc in (
        getattr(litellm.exceptions, name, None)
        for name in (
            "RateLimitError",
            "APIConnectionError",
            "InternalServerError",
            "ServiceUnavailableError",
            "Timeout",
        )
    )
    if exc is not None
)


class GatewayError(RuntimeError):
    pass


class BudgetMeter:
    """Per-run spend accumulator. Warns at warn_fraction and again at 100%.

    Startup-honest: it never hard-stops a run — it makes overspend loud."""

    def __init__(self, limit_usd: float, warn_fraction: float = 0.8) -> None:
        self.limit_usd = limit_usd
        self.warn_fraction = warn_fraction
        self.spent_usd = 0.0
        self._warned_soft = False
        self._warned_hard = False

    def add(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        if not self._warned_soft and self.spent_usd >= self.limit_usd * self.warn_fraction:
            self._warned_soft = True
            logger.warning(
                "budget: %.0f%% of $%.2f consumed ($%.4f spent)",
                self.warn_fraction * 100,
                self.limit_usd,
                self.spent_usd,
            )
        if not self._warned_hard and self.spent_usd >= self.limit_usd:
            self._warned_hard = True
            logger.warning(
                "budget: run budget of $%.2f EXCEEDED ($%.4f spent)",
                self.limit_usd,
                self.spent_usd,
            )


class Gateway:
    """All LLM traffic flows through here.

    ``completion_fn`` is injectable (defaults to ``litellm.completion``) so
    tests and alternate transports never monkeypatch internals.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        manifest: Manifest | None = None,
        prompts: PromptLib | None = None,
        completion_fn: Callable[..., Any] | None = None,
        max_attempts: int = 4,
        backoff_base_s: float = 1.0,
    ) -> None:
        import instructor

        # Reasoning-class models (claude-*-5, o-series) reject sampling params
        # like temperature; drop what a model doesn't support instead of dying.
        litellm.drop_params = True
        self.settings = settings
        self.manifest = manifest
        self.prompts = prompts or PromptLib(settings.paths.prompts_dir)
        self.budget = BudgetMeter(settings.budget.run_usd, settings.budget.warn_fraction)
        self._completion = completion_fn or litellm.completion
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._instructor = instructor.from_litellm(
            self._retrying_completion, mode=instructor.Mode.JSON
        )
        if settings.tracing.langfuse:
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]

    # ── transport with jittered retries ──────────────────────────────────────

    def _retrying_completion(self, *args: Any, **kwargs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return self._completion(*args, **kwargs)
            except _RETRYABLE as e:  # type: ignore[misc]
                last = e
                delay = self._backoff_base_s * (2**attempt) * random.uniform(0.5, 1.5)
                logger.warning(
                    "transient LLM error (%s), retry %d/%d in %.1fs",
                    type(e).__name__,
                    attempt + 1,
                    self._max_attempts - 1,
                    delay,
                )
                if attempt < self._max_attempts - 1:
                    time.sleep(delay)
        raise GatewayError(f"exhausted {self._max_attempts} attempts: {last}") from last

    # ── public surface ───────────────────────────────────────────────────────

    def structured(
        self,
        prompt: PromptAsset | str,
        vars: dict[str, Any] | None = None,
        *,
        response_model: type[T] | None = None,
        tier: Tier | None = None,
        stage: str = "",
        **overrides: Any,
    ) -> T:
        """Render a prompt asset, call its tier, validate into its output schema.

        The response model defaults to the asset's ``output_schema`` frontmatter.
        One repair pass: on validation failure instructor re-asks with the error.
        """
        asset = prompt if isinstance(prompt, PromptAsset) else self.prompts.get(prompt)
        model_cls = response_model or (
            resolve_schema(asset.output_schema) if asset.output_schema else None
        )
        if model_cls is None:
            raise GatewayError(
                f"prompt {asset.id!r} declares no output_schema and no response_model given"
            )
        tier_name: Tier = tier or asset.tier
        cfg = self.settings.models.for_tier(tier_name)
        messages = [{"role": "user", "content": asset.render(**(vars or {}))}]
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "temperature": 0.0 if tier_name == "JUDGE" else cfg.temperature,
            **({"max_tokens": cfg.max_tokens} if cfg.max_tokens else {}),
            **overrides,
        }

        t0 = time.perf_counter()
        try:
            parsed, raw = self._instructor.chat.completions.create_with_completion(
                messages=messages,
                response_model=model_cls,
                max_retries=1,  # instructor counts retries: initial attempt + one repair pass
                **kwargs,
            )
        except Exception as e:
            self._trace(stage, tier_name, cfg.model, asset, None, t0, ok=False, error=repr(e))
            raise GatewayError(f"structured call failed for prompt {asset.id!r}: {e}") from e
        self._trace(stage, tier_name, cfg.model, asset, raw, t0, ok=True)
        return parsed  # type: ignore[return-value]

    def text(
        self,
        prompt: PromptAsset | str,
        vars: dict[str, Any] | None = None,
        *,
        tier: Tier | None = None,
        stage: str = "",
        system: str | None = None,
        **overrides: Any,
    ) -> str:
        """Plain-text completion (e.g. VLM page transcription, prose synthesis)."""
        asset = prompt if isinstance(prompt, PromptAsset) else self.prompts.get(prompt)
        tier_name: Tier = tier or asset.tier
        cfg = self.settings.models.for_tier(tier_name)
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": asset.render(**(vars or {}))})
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "temperature": 0.0 if tier_name == "JUDGE" else cfg.temperature,
            **({"max_tokens": cfg.max_tokens} if cfg.max_tokens else {}),
            **overrides,
        }

        t0 = time.perf_counter()
        try:
            raw = self._retrying_completion(messages=messages, **kwargs)
        except Exception as e:
            self._trace(stage, tier_name, cfg.model, asset, None, t0, ok=False, error=repr(e))
            raise
        self._trace(stage, tier_name, cfg.model, asset, raw, t0, ok=True)
        return raw.choices[0].message.content or ""

    @property
    def spent_usd(self) -> float:
        return self.budget.spent_usd

    # ── trace emission ───────────────────────────────────────────────────────

    def _trace(
        self,
        stage: str,
        tier: str,
        model: str,
        asset: PromptAsset,
        raw: Any,
        t0: float,
        *,
        ok: bool,
        error: str = "",
    ) -> None:
        latency_ms = (time.perf_counter() - t0) * 1000
        input_tokens = output_tokens = 0
        cost = 0.0
        if raw is not None:
            usage = getattr(raw, "usage", None)
            if usage is not None:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            try:
                cost = float(litellm.completion_cost(completion_response=raw))
            except Exception:  # unknown/mock models cost nothing
                cost = 0.0
        self.budget.add(cost)
        if self.manifest is not None:
            self.manifest.log_llm_call(
                stage=stage,
                tier=tier,
                model=model,
                prompt_id=asset.id,
                prompt_rev=asset.rev,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                ok=ok,
                error=error,
            )
        else:
            logger.debug(
                "llm call stage=%s tier=%s model=%s prompt=%s ok=%s cost=$%.5f",
                stage,
                tier,
                model,
                asset.id,
                ok,
                cost,
            )
