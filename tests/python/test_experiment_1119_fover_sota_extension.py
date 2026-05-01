"""Tests for ``scripts/experiment_1119_fover_sota_extension_v5.py``.

Spec: REQ-VERIFY-083 (live_gpu provenance), REQ-LEARN-011 (corpus extension),
REQ-INFER-SOTA-001 (SOTA-tier model gate).

These tests cover the pure-function helpers: corpus counting, step splitting,
step labeling, final-answer checking, and artifact schema validation. We do NOT
exercise the live GGUF inference path (requires dual RTX 3090 + 21 GB of model
weights on disk) — that is the conductor's job when it runs the full experiment.

Why these specific tests:
  * ``split_cot_into_steps`` is the parser that determines FoVer corpus
    granularity — a bug here silently discards reasoning steps and produces
    spurious single-step pairs that erode training signal.
  * ``label_step`` is the labeling oracle. Its boundary decisions (score < 0.3
    → correct, score > 0.7 → incorrect, middle band → heuristic) directly
    affect label quality for exp1120. All three branches must be tested.
  * ``final_answer_correct`` drives the heuristic fallback inside ``label_step``
    when Z3 returns the indeterminate middle band. A wrong implementation here
    would invert labels in the heuristic branch.
  * ``_build_artifact`` has a fixed schema required by the conductor's failure
    ledger and the exp1120 training script. Every required field must be present
    and have the right Python type.
  * ``count_fover_jsonl`` and ``initialize_fover_jsonl_if_needed`` are the
    corpus-state observers; their correctness determines n_pairs_before / after
    in the artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1119_fover_sota_extension_v5.py"


def _load_module():
    """Load the experiment script as a module without executing ``main()``.

    importlib is used instead of a regular import so that pytest does not need
    ``scripts/`` on PYTHONPATH, and so that the module can be loaded even when
    it is not importable as a package.
    """
    spec = importlib.util.spec_from_file_location("exp1119", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1119"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp1119():
    return _load_module()


# ---------------------------------------------------------------------------
# split_cot_into_steps
# ---------------------------------------------------------------------------


def test_split_numbered_steps_returns_multiple(exp1119):
    """Numbered steps ("Step 1:", "Step 2:") should produce one entry each.

    Why: the FoVer corpus is step-level; failing to split a CoT into steps
    would produce a single giant pair instead of fine-grained labels.
    """
    cot = "Step 1: Multiply 4 × 5 = 20.\nStep 2: Add 3 to get 23.\nStep 3: The answer is 23."
    steps = exp1119.split_cot_into_steps(cot)
    assert len(steps) >= 3


def test_split_paragraph_fallback(exp1119):
    """Unstructured CoT with double newlines falls back to paragraph split."""
    cot = "First I compute 4 × 5 = 20.\n\nThen I add 3 to get 23.\n\nThe answer is 23."
    steps = exp1119.split_cot_into_steps(cot)
    assert len(steps) >= 2


def test_split_single_step_returns_one(exp1119):
    """A response with no step markers and no paragraph breaks gives one step."""
    cot = "The answer is 42."
    steps = exp1119.split_cot_into_steps(cot)
    assert len(steps) == 1
    assert steps[0] == "The answer is 42."


def test_split_empty_string_returns_empty(exp1119):
    """Empty input should return an empty list, not raise."""
    assert exp1119.split_cot_into_steps("") == []


def test_split_whitespace_only_returns_empty(exp1119):
    """Whitespace-only input should return empty list."""
    assert exp1119.split_cot_into_steps("   \n   ") == []


# ---------------------------------------------------------------------------
# final_answer_correct
# ---------------------------------------------------------------------------


def test_final_answer_correct_exact_integer(exp1119):
    """Last number in response matches expected integer."""
    assert exp1119.final_answer_correct("The result is 42", 42.0) is True


def test_final_answer_correct_float_tolerance(exp1119):
    """Float-rounded answers within 1e-6 are accepted."""
    assert exp1119.final_answer_correct("answer = 3.14", 3.14) is True


def test_final_answer_correct_wrong_number(exp1119):
    """Response ending in wrong number returns False."""
    assert exp1119.final_answer_correct("So 5 + 3 = 9", 8.0) is False


def test_final_answer_correct_no_numbers(exp1119):
    """Response with no numeric literals returns False."""
    assert exp1119.final_answer_correct("The answer is unknown.", 5.0) is False


def test_final_answer_correct_negative(exp1119):
    """Negative expected answers are handled correctly."""
    assert exp1119.final_answer_correct("balance is -10", -10.0) is True


# ---------------------------------------------------------------------------
# label_step
# ---------------------------------------------------------------------------


def test_label_step_z3_correct_low_score(exp1119):
    """Z3 violation score < 0.3 → label 'correct', verifier 'Z3Math'."""
    mock_verifier = MagicMock()
    mock_verifier.score.return_value = 0.1
    label, conf, verifier_name = exp1119.label_step(
        "3 + 4 = 7", mock_verifier, final_answer_correct=True
    )
    assert label == "correct"
    assert verifier_name == "Z3Math"
    assert 0.0 <= conf <= 1.0


def test_label_step_z3_incorrect_high_score(exp1119):
    """Z3 violation score > 0.7 → label 'incorrect', verifier 'Z3Math'."""
    mock_verifier = MagicMock()
    mock_verifier.score.return_value = 0.9
    label, conf, verifier_name = exp1119.label_step(
        "3 + 4 = 8", mock_verifier, final_answer_correct=False
    )
    assert label == "incorrect"
    assert verifier_name == "Z3Math"
    assert 0.0 <= conf <= 1.0


def test_label_step_middle_band_falls_to_heuristic(exp1119):
    """Z3 score in [0.3, 0.7] band falls through to heuristic verifier."""
    mock_verifier = MagicMock()
    mock_verifier.score.return_value = 0.5  # indeterminate
    label, conf, verifier_name = exp1119.label_step(
        "some step", mock_verifier, final_answer_correct=True
    )
    assert verifier_name == "heuristic"
    assert label in ("correct", "incorrect")


def test_label_step_no_verifier_uses_heuristic(exp1119):
    """When verifier is None the heuristic branch runs without error."""
    label, conf, verifier_name = exp1119.label_step(
        "The answer is 5.", None, final_answer_correct=True
    )
    assert verifier_name == "heuristic"
    assert label in ("correct", "incorrect")
    assert 0.0 <= conf <= 1.0


def test_label_step_verifier_exception_uses_heuristic(exp1119):
    """If verifier.score() raises, the heuristic path must still produce a label."""
    mock_verifier = MagicMock()
    mock_verifier.score.side_effect = RuntimeError("z3 crashed")
    label, conf, verifier_name = exp1119.label_step(
        "42 + 1 = 43", mock_verifier, final_answer_correct=True
    )
    assert label in ("correct", "incorrect")
    assert verifier_name == "heuristic"


# ---------------------------------------------------------------------------
# count_fover_jsonl and initialize_fover_jsonl_if_needed
# ---------------------------------------------------------------------------


def test_count_fover_jsonl_empty_file(exp1119, tmp_path):
    """An empty JSONL file has 0 entries."""
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text("")
    with patch.object(exp1119, "FOVER_JSONL", jsonl):
        assert exp1119.count_fover_jsonl() == 0


def test_count_fover_jsonl_counts_lines(exp1119, tmp_path):
    """count_fover_jsonl returns the number of non-empty lines."""
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(
        json.dumps({"question_id": "a", "step_text": "s", "label": "correct"})
        + "\n"
        + json.dumps({"question_id": "b", "step_text": "t", "label": "incorrect"})
        + "\n"
    )
    with patch.object(exp1119, "FOVER_JSONL", jsonl):
        assert exp1119.count_fover_jsonl() == 2


def test_count_fover_jsonl_missing_file_returns_zero(exp1119, tmp_path):
    """Missing JSONL file returns 0 without raising."""
    missing = tmp_path / "nonexistent.jsonl"
    with patch.object(exp1119, "FOVER_JSONL", missing):
        assert exp1119.count_fover_jsonl() == 0


def test_initialize_fover_jsonl_skips_if_exists(exp1119, tmp_path):
    """initialize_fover_jsonl_if_needed returns 0 and does not overwrite existing file."""
    existing = tmp_path / "corpus.jsonl"
    existing.write_text('{"question_id": "existing"}\n')
    with patch.object(exp1119, "FOVER_JSONL", existing):
        result = exp1119.initialize_fover_jsonl_if_needed()
    assert result == 0
    # File unchanged — still has the original content.
    assert "existing" in existing.read_text()


def test_initialize_fover_jsonl_creates_from_v4(exp1119, tmp_path):
    """When JSONL is absent and v4.json has entries, JSONL is created with them."""
    # Write a fake v4.json with 3 entries.
    v4_json = tmp_path / "fover_corpus_v4.json"
    entries = [
        {
            "question_id": f"gsm8k_{i}_0_0",
            "step_text": f"step {i}",
            "label": "correct",
            "confidence": 1.0,
        }
        for i in range(3)
    ]
    v4_json.write_text(json.dumps(entries))

    jsonl_path = tmp_path / "fover_corpus.jsonl"
    with (
        patch.object(exp1119, "FOVER_JSONL", jsonl_path),
        patch.object(exp1119, "FOVER_V4_JSON", v4_json),
    ):
        written = exp1119.initialize_fover_jsonl_if_needed()

    assert written == 3
    assert jsonl_path.exists()
    lines = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    # Source and model fields should be added.
    assert lines[0]["source"] == "fover_v4"
    assert lines[0]["model"] == "base_model"


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------

_REQUIRED_ARTIFACT_FIELDS = {
    "experiment",
    "title",
    "run_date",
    "schema_version",
    "n_pairs_before",
    "n_pairs_added",
    "n_pairs_after",
    "fover_sota_pairs_added_above_7000",
    "models_used",
    "labeling_verifiers",
    "label_positive_fraction",
    "inference_mode",
    "honest_verdict",
    "duration_s",
}


def test_build_artifact_has_all_required_fields(exp1119):
    """Every required schema field must appear in the artifact dict."""
    artifact = exp1119._build_artifact(
        n_pairs_before=6548,
        n_pairs_added=500,
        n_pairs_after=7048,
        models_used=["unsloth/Qwen3.6-35B-A3B-GGUF"],
        labeling_verifiers=["Z3Math"],
        label_positive_fraction=0.62,
        inference_mode="live_gpu",
        honest_verdict="corpus_extended_above_7000",
        duration_s=420.5,
    )
    missing = _REQUIRED_ARTIFACT_FIELDS - set(artifact.keys())
    assert not missing, f"Missing required fields: {missing}"


def test_build_artifact_fover_flag_true_when_above_7000(exp1119):
    """fover_sota_pairs_added_above_7000 is True when n_pairs_after >= 7000."""
    artifact = exp1119._build_artifact(
        n_pairs_before=6548,
        n_pairs_added=452,
        n_pairs_after=7000,
        models_used=[],
        labeling_verifiers=[],
        label_positive_fraction=0.0,
        inference_mode="live_gpu",
        honest_verdict="corpus_extended_above_7000",
        duration_s=10.0,
    )
    assert artifact["fover_sota_pairs_added_above_7000"] is True


def test_build_artifact_fover_flag_false_when_below_7000(exp1119):
    """fover_sota_pairs_added_above_7000 is False when n_pairs_after < 7000."""
    artifact = exp1119._build_artifact(
        n_pairs_before=6548,
        n_pairs_added=100,
        n_pairs_after=6648,
        models_used=[],
        labeling_verifiers=[],
        label_positive_fraction=0.0,
        inference_mode="live_gpu",
        honest_verdict="corpus_extended_below_7000",
        duration_s=10.0,
    )
    assert artifact["fover_sota_pairs_added_above_7000"] is False


def test_build_artifact_honest_verdict_in_allowed_set(exp1119):
    """honest_verdict must be one of the four canonical strings."""
    allowed = {
        "corpus_extended_above_7000",
        "corpus_extended_below_7000",
        "partial",
        "failed",
    }
    for verdict in allowed:
        artifact = exp1119._build_artifact(
            n_pairs_before=0,
            n_pairs_added=0,
            n_pairs_after=0,
            models_used=[],
            labeling_verifiers=[],
            label_positive_fraction=0.0,
            inference_mode="live_gpu",
            honest_verdict=verdict,
            duration_s=1.0,
        )
        assert artifact["honest_verdict"] == verdict


def test_build_artifact_experiment_id_is_1119(exp1119):
    """The experiment ID must be 1119 to match the roadmap entry."""
    artifact = exp1119._build_artifact(
        n_pairs_before=0,
        n_pairs_added=0,
        n_pairs_after=0,
        models_used=[],
        labeling_verifiers=[],
        label_positive_fraction=0.0,
        inference_mode="live_gpu",
        honest_verdict="failed",
        duration_s=0.0,
    )
    assert artifact["experiment"] == 1119
