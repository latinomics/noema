"""Settings for NOEMA, sourced from noema.toml + NOEMA_* env overrides.

Pipeline code never mentions model strings — only tiers (MAP / REDUCE / JUDGE),
resolved here. Paths derive from one data_dir so the whole data plane relocates
with a single config line.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

Tier = Literal["MAP", "REDUCE", "JUDGE"]
TIERS: tuple[Tier, ...] = ("MAP", "REDUCE", "JUDGE")


class TierConfig(BaseModel):
    model: str
    temperature: float = 0.3
    max_tokens: int | None = None


class ModelsConfig(BaseModel):
    map: TierConfig = Field(
        default_factory=lambda: TierConfig(model="anthropic/claude-haiku-4-5", temperature=0.3)
    )
    reduce: TierConfig = Field(
        default_factory=lambda: TierConfig(model="anthropic/claude-sonnet-4-5", temperature=0.4)
    )
    judge: TierConfig = Field(
        default_factory=lambda: TierConfig(model="anthropic/claude-haiku-4-5", temperature=0.0)
    )

    def for_tier(self, tier: Tier | str) -> TierConfig:
        name = tier.lower()
        if name not in ("map", "reduce", "judge"):
            raise KeyError(f"unknown tier {tier!r}; expected one of {TIERS}")
        return getattr(self, name)

    def bindings(self, tiers: tuple[str, ...] | list[str] | None = None) -> dict[str, str]:
        """Tier -> model mapping; part of every manifest cache key."""
        selected = tiers if tiers is not None else TIERS
        return {t: self.for_tier(t).model for t in selected}


class BudgetConfig(BaseModel):
    run_usd: float = 10.0
    warn_fraction: float = 0.8


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    prompts_dir: Path = Path("prompts")

    @property
    def shelf_dir(self) -> Path:
        return self.data_dir / "shelf"

    @property
    def markdown_dir(self) -> Path:
        return self.data_dir / "markdown"

    @property
    def ontology_dir(self) -> Path:
        return self.data_dir / "ontology"

    @property
    def noesis_dir(self) -> Path:
        return self.data_dir / "noesis"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def lancedb_dir(self) -> Path:
        return self.indexes_dir / "lancedb"

    @property
    def lineage_db(self) -> Path:
        return self.indexes_dir / "lineage.duckdb"

    def ensure(self) -> PathsConfig:
        for d in (
            self.shelf_dir,
            self.markdown_dir,
            self.ontology_dir,
            self.noesis_dir,
            self.frames_dir,
            self.indexes_dir,
            self.lancedb_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


class EmbeddingsConfig(BaseModel):
    provider: Literal["local", "hash", "api"] = "local"
    model: str = "BAAI/bge-m3"  # sentence-transformers name when provider == "local"
    api_model: str = ""  # litellm embedding model string when provider == "api"
    hash_dim: int = 384


class ThresholdsConfig(BaseModel):
    docforge_fidelity: float = 4.2
    docforge_span_match: float = 0.97
    ontogen_qa_fidelity: float = 0.8
    ontogen_coverage: float = 0.85
    gym_min_score: float = 4.0


class TracingConfig(BaseModel):
    langfuse: bool = False


class Settings(BaseSettings):
    """Priority: explicit kwargs > NOEMA_* env > noema.toml > defaults."""

    model_config = SettingsConfigDict(
        env_prefix="NOEMA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_file = Path(os.environ.get("NOEMA_CONFIG", "noema.toml"))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=toml_file),
            file_secret_settings,
        )

    @classmethod
    def load(cls, config: Path | str | None = None) -> Settings:
        """Load settings, optionally from an explicit toml path."""
        if config is not None:
            os.environ["NOEMA_CONFIG"] = str(config)
        return cls()
