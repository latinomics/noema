"""Prompts-as-code: frontmatter contract, revision hashing, rendering."""

from __future__ import annotations

import hashlib

import pytest

from noema.kernel.promptlib import PromptError, PromptLib, load_prompt
from tests.conftest import PROMPTS_DIR

EXPECTED_JUDGE_PROMPTS = {
    "judge/rubric": "RubricVerdict",
    "judge/entails": "EntailmentVerdict",
    "judge/qa_generate": "QAPairSet",
    "judge/qa_answer": "QAAnswer",
    "judge/qa_faithful": "FaithfulnessVerdict",
}


@pytest.fixture
def lib() -> PromptLib:
    return PromptLib(PROMPTS_DIR)


def test_judge_prompts_load_with_declared_schemas(lib: PromptLib):
    for prompt_id, schema in EXPECTED_JUDGE_PROMPTS.items():
        asset = lib.get(prompt_id)
        assert asset.tier == "JUDGE"
        assert asset.output_schema == schema


def test_rev_is_sha256_of_body(lib: PromptLib):
    asset = lib.get("judge/rubric")
    assert asset.rev == hashlib.sha256(asset.body.encode("utf-8")).hexdigest()[:8]
    assert len(asset.rev) == 8


def test_render_fills_variables_and_leaves_json_braces(lib: PromptLib):
    asset = lib.get("judge/rubric")
    assert asset.variables == {"rubric", "artifact"}
    rendered = asset.render(rubric="Clarity above all.", artifact="The artifact text.")
    assert "Clarity above all." in rendered
    assert "The artifact text." in rendered
    assert "{{" not in rendered
    assert '{"score"' in rendered  # JSON example braces untouched


def test_render_missing_variable_raises(lib: PromptLib):
    with pytest.raises(PromptError, match="artifact"):
        lib.get("judge/rubric").render(rubric="only one of two")


def test_unknown_prompt_id_lists_available(lib: PromptLib):
    with pytest.raises(KeyError, match="judge/rubric"):
        lib.get("judge/nonexistent")


def test_body_edit_changes_rev(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("---\nid: x/a\ntier: MAP\n---\nBody one {{v}}\n")
    rev1 = load_prompt(p).rev
    p.write_text("---\nid: x/a\ntier: MAP\n---\nBody two {{v}}\n")
    rev2 = load_prompt(p).rev
    assert rev1 != rev2


def test_frontmatter_only_edit_keeps_rev(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("---\nid: x/a\ntier: MAP\n---\nSame body\n")
    rev1 = load_prompt(p).rev
    p.write_text("---\nid: x/a\ntier: MAP\ndescription: added later\n---\nSame body\n")
    assert load_prompt(p).rev == rev1


def test_missing_frontmatter_raises(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("no frontmatter here")
    with pytest.raises(PromptError, match="frontmatter"):
        load_prompt(p)


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "a.md").write_text("---\nid: dup\ntier: MAP\n---\nA\n")
    (tmp_path / "b.md").write_text("---\nid: dup\ntier: MAP\n---\nB\n")
    with pytest.raises(PromptError, match="duplicate"):
        PromptLib(tmp_path)
