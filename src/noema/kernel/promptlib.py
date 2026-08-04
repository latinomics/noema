"""Prompts as versioned code (BLUEPRINT §2.3).

Every prompt is a markdown asset under prompts/ with YAML frontmatter declaring
``id``, ``tier``, and optionally ``output_schema`` (a name in kernel.schemas).
``prompt_rev = sha256(body)[:8]`` flows into every Provenance and every
manifest cache key — editing a prompt busts exactly the stages that use it.

Template variables use ``{{ name }}``; literal braces elsewhere (JSON examples)
are left untouched.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from noema.kernel.config import Tier

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptError(RuntimeError):
    pass


class PromptAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tier: Tier
    output_schema: str | None = None
    description: str = ""
    body: str
    rev: str  # sha256(body)[:8]
    path: Path

    @property
    def variables(self) -> set[str]:
        return set(_VAR_RE.findall(self.body))

    def render(self, **vars: object) -> str:
        missing = self.variables - vars.keys()
        if missing:
            raise PromptError(
                f"prompt {self.id!r} missing variables: {sorted(missing)} "
                f"(expects {sorted(self.variables)})"
            )
        return _VAR_RE.sub(lambda m: str(vars[m.group(1)]), self.body)


def prompt_rev(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:8]


def load_prompt(path: Path) -> PromptAsset:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise PromptError(f"{path}: missing YAML frontmatter (--- ... ---)")
    meta = yaml.safe_load(match.group(1)) or {}
    body = text[match.end() :].strip()
    if not body:
        raise PromptError(f"{path}: empty prompt body")
    for key in ("id", "tier"):
        if key not in meta:
            raise PromptError(f"{path}: frontmatter missing required key {key!r}")
    return PromptAsset(
        id=str(meta["id"]),
        tier=str(meta["tier"]).upper(),  # type: ignore[arg-type]
        output_schema=meta.get("output_schema"),
        description=str(meta.get("description", "")),
        body=body,
        rev=prompt_rev(body),
        path=path,
    )


class PromptLib:
    """Eagerly scans prompts/**/*.md; assets are addressed by frontmatter id."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._assets: dict[str, PromptAsset] = {}
        if not self.root.is_dir():
            raise PromptError(f"prompts root does not exist: {self.root}")
        for path in sorted(self.root.rglob("*.md")):
            asset = load_prompt(path)
            if asset.id in self._assets:
                raise PromptError(
                    f"duplicate prompt id {asset.id!r}: {path} and {self._assets[asset.id].path}"
                )
            self._assets[asset.id] = asset

    def get(self, prompt_id: str) -> PromptAsset:
        try:
            return self._assets[prompt_id]
        except KeyError:
            available = ", ".join(sorted(self._assets))
            raise KeyError(f"unknown prompt {prompt_id!r}; available: {available}") from None

    def all(self) -> list[PromptAsset]:
        return sorted(self._assets.values(), key=lambda a: a.id)
