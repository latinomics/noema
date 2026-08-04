"""Stage cache semantics (M0 DoD: cache hit/miss behaves) + lineage logging."""

from __future__ import annotations

from pathlib import Path

import pytest

from noema.kernel.manifest import Manifest, file_hash, obj_hash
from noema.kernel.promptlib import PromptAsset


def asset(rev: str) -> PromptAsset:
    return PromptAsset(
        id="ontogen/axiom_mine",
        tier="MAP",
        body="mine axioms {{chunk}}",
        rev=rev,
        path=Path("prompts/ontogen/axiom_mine.md"),
    )


def test_cache_hit_on_identical_key(manifest: Manifest):
    runs = {"n": 0}

    def stage():
        runs["n"] += 1
        return {"artifact": "data/ontology/x/ontology.json"}

    r1 = manifest.run("demo.stage", stage, inputs={"doc": "hash-a"}, params={"k": 1})
    r2 = manifest.run("demo.stage", stage, inputs={"doc": "hash-a"}, params={"k": 1})
    assert runs["n"] == 1
    assert not r1.cached and r2.cached
    assert r1.key == r2.key
    assert r2.outputs == {"artifact": "data/ontology/x/ontology.json"}


@pytest.mark.parametrize(
    "kwargs_a,kwargs_b",
    [
        ({"inputs": {"doc": "hash-a"}}, {"inputs": {"doc": "hash-b"}}),
        ({"params": {"k": 1}}, {"params": {"k": 2}}),
        ({"prompts": [asset("aaaa1111")]}, {"prompts": [asset("bbbb2222")]}),
        ({"tiers": ["MAP"]}, {"tiers": ["MAP", "JUDGE"]}),
    ],
)
def test_cache_miss_when_key_component_changes(manifest: Manifest, kwargs_a, kwargs_b):
    def stage():
        return {"ok": True}

    base = {"inputs": {"doc": "hash-a"}}
    r1 = manifest.run("demo.stage", stage, **{**base, **kwargs_a})
    r2 = manifest.run("demo.stage", stage, **{**base, **kwargs_b})
    assert r1.key != r2.key
    assert not r2.cached


def test_cache_miss_when_code_changes(manifest: Manifest):
    def stage_v1():
        return {"v": 1}

    def stage_v2():
        return {"v": 2}  # different source -> different code_rev

    r1 = manifest.run("demo.stage", stage_v1, inputs={"doc": "h"})
    r2 = manifest.run("demo.stage", stage_v2, inputs={"doc": "h"})
    assert r1.key != r2.key
    assert r2.outputs == {"v": 2}


def test_force_busts_cache(manifest: Manifest):
    runs = {"n": 0}

    def stage():
        runs["n"] += 1
        return {"n": runs["n"]}

    manifest.run("demo.stage", stage, inputs={"doc": "h"})
    r2 = manifest.run("demo.stage", stage, inputs={"doc": "h"}, force=True)
    assert runs["n"] == 2
    assert not r2.cached


def test_model_and_path_inputs_are_hashed(manifest: Manifest, ontology, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# canonical markdown")

    def stage():
        return {"ok": True}

    r1 = manifest.run("demo.stage", stage, inputs={"doc": f, "ont": ontology})
    f.write_text("# canonical markdown, edited")
    r2 = manifest.run("demo.stage", stage, inputs={"doc": f, "ont": ontology})
    assert r1.key != r2.key  # file content change busts the cache


def test_non_jsonable_outputs_rejected(manifest: Manifest):
    def stage():
        return {"bad": object()}

    with pytest.raises(TypeError, match="JSON-serializable"):
        manifest.run("demo.stage", stage, inputs={})


def test_lineage_logging_and_status(manifest: Manifest):
    manifest.log_llm_call(
        stage="s", tier="JUDGE", model="m", cost_usd=0.01, input_tokens=100, output_tokens=20
    )
    manifest.log_llm_call(stage="s", tier="MAP", model="m", cost_usd=0.02)
    manifest.log_score(stage="p2.gate", name="qa_fidelity", value=0.83, meta={"k": 20})

    assert manifest.spend() == pytest.approx(0.03)
    assert manifest.spend(run_id="run-test") == pytest.approx(0.03)
    assert manifest.spend(run_id="other-run") == 0.0

    def stage():
        return {}

    manifest.run("demo.stage", stage, inputs={})
    info = manifest.status()
    assert info["llm_calls"] == 2
    assert info["spend_usd"] == pytest.approx(0.03)
    assert info["stages"][0]["stage"] == "demo.stage"


def test_hash_helpers(tmp_path, ontology):
    f = tmp_path / "x.bin"
    f.write_bytes(b"pdf bytes")
    assert file_hash(f) == file_hash(f)
    assert len(file_hash(f)) == 64
    assert obj_hash(ontology) == obj_hash(ontology)
    assert obj_hash({"b": 1, "a": 2}) == obj_hash({"a": 2, "b": 1})  # key order irrelevant
