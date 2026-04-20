"""Tests for Exp 570: FR-11 Tier 1 relay with CoACEExtractor.

Covers all helper functions introduced in
scripts/experiment_570_fr11_real_violations.py:
  - _load_gate()
  - _load_gsm8k_questions()
  - CoACEBackedPipeline.verify()
  - _run_batches_with_coace()
  - _compute_fp_rate_trend()
  - _build_artifact()

Spec: REQ-LEARN-053,
      SCENARIO-LEARN-084, SCENARIO-LEARN-085, SCENARIO-LEARN-086
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_570_fr11_real_violations import (
    CoACEBackedPipeline,
    _build_artifact,
    _compute_fp_rate_trend,
    _load_gate,
    _load_gsm8k_questions,
    _run_batches_with_coace,
)
from carnot.extraction.coace_extractor import CoACEExtractor
from scripts.experiment_template import ExperimentTemplate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tmpl(tmp_path: Path) -> ExperimentTemplate:
    """Build a minimal ExperimentTemplate pointing at tmp_path."""
    deliverable = str(tmp_path / "result.json")
    tmpl = ExperimentTemplate(
        exp_id=570,
        title="test",
        deliverable=deliverable,
        requires_gpu=True,
        repo_root=tmp_path,
    )
    tmpl.setup()
    return tmpl


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Tests: _load_gate
# ---------------------------------------------------------------------------


def test_load_gate_missing_file(tmp_path):
    """SCENARIO-LEARN-084: missing gate file returns None (blocked)."""
    result = _load_gate(tmp_path)
    assert result is None


def test_load_gate_open(tmp_path):
    """SCENARIO-LEARN-084: gate_open=True is returned correctly."""
    gate_path = tmp_path / "results" / "experiment_565_coace_live_diagnostic.json"
    _write_json(gate_path, {"gate_open": True, "status": "success"})
    result = _load_gate(tmp_path)
    assert result is not None
    assert result["gate_open"] is True


def test_load_gate_closed(tmp_path):
    """Gate file with gate_open=False is returned as-is (caller decides to block)."""
    gate_path = tmp_path / "results" / "experiment_565_coace_live_diagnostic.json"
    _write_json(gate_path, {"gate_open": False})
    result = _load_gate(tmp_path)
    assert result is not None
    assert result["gate_open"] is False


def test_load_gate_malformed_json(tmp_path):
    """Malformed JSON in gate file returns None without raising."""
    gate_path = tmp_path / "results" / "experiment_565_coace_live_diagnostic.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text("NOT JSON")
    result = _load_gate(tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# Tests: _load_gsm8k_questions
# ---------------------------------------------------------------------------


def test_load_gsm8k_questions_synthetic_fallback():
    """SCENARIO-LEARN-085: synthetic fallback returns 25 questions when datasets unavailable."""
    # Patch datasets to simulate offline environment
    with patch.dict(sys.modules, {"datasets": None}):
        questions = _load_gsm8k_questions(150, 174)
    assert len(questions) == 25
    assert all("question" in q and "answer" in q for q in questions)


def test_load_gsm8k_synthetic_even_questions_have_arithmetic_errors():
    """Even-indexed synthetic questions embed '3 + 3 = 7' so CoACE can fire."""
    with patch.dict(sys.modules, {"datasets": None}):
        questions = _load_gsm8k_questions(150, 174)
    # Even-indexed items (0, 2, 4...) should have '= 7' in answer (wrong arithmetic)
    even_answers = [questions[i]["answer"] for i in range(0, len(questions), 2)]
    assert all("3 + 3 = 7" in a for a in even_answers)


# ---------------------------------------------------------------------------
# Tests: CoACEBackedPipeline
# ---------------------------------------------------------------------------


def test_coace_pipeline_no_violation_verified():
    """SCENARIO-LEARN-084: response with no arithmetic error → verified=True."""
    extractor = CoACEExtractor(tolerance=1e-6, min_confidence=0.5)
    response_map = {"q1": "The answer is 5 because 2 + 3 = 5."}
    pipeline = CoACEBackedPipeline(extractor, response_map)
    verified, tier, energy = pipeline.verify("q1", question="q1")
    assert verified is True
    assert tier == "coace"
    assert energy == 0.0


def test_coace_pipeline_with_violation_not_verified():
    """SCENARIO-LEARN-085: response with arithmetic error → verified=False."""
    extractor = CoACEExtractor(tolerance=1e-6, min_confidence=0.5)
    # 47 + 28 = 76 is wrong (correct is 75)
    response_map = {"q2": "We compute 47 + 28 = 76, so the answer is 76."}
    pipeline = CoACEBackedPipeline(extractor, response_map)
    verified, tier, energy = pipeline.verify("q2", question="q2")
    assert verified is False
    assert energy > 0.0


def test_coace_pipeline_logs_calls():
    """Pipeline.call_log records one entry per verify() call."""
    extractor = CoACEExtractor()
    pipeline = CoACEBackedPipeline(extractor, {"q": "simple text"})
    pipeline.verify("q", question="q")
    pipeline.verify("q", question="q")
    assert len(pipeline.call_log) == 2


def test_coace_pipeline_fallback_to_response_param():
    """When question not in response_map, falls back to response argument."""
    extractor = CoACEExtractor()
    pipeline = CoACEBackedPipeline(extractor, {})
    # Pass a response text directly (not in map) — should not raise
    verified, tier, energy = pipeline.verify("some plain text", question="unknown_q")
    assert tier == "coace"


# ---------------------------------------------------------------------------
# Tests: _compute_fp_rate_trend
# ---------------------------------------------------------------------------


def test_fp_rate_trend_decreasing():
    """SCENARIO-LEARN-086: strictly decreasing FP rate returns 'decreasing'."""
    batch_results = [
        {"fp_rate": 0.5},
        {"fp_rate": 0.3},
        {"fp_rate": 0.1},
    ]
    assert _compute_fp_rate_trend(batch_results) == "decreasing"


def test_fp_rate_trend_flat():
    """Constant FP rate returns 'flat'."""
    batch_results = [
        {"fp_rate": 0.2},
        {"fp_rate": 0.2},
        {"fp_rate": 0.2},
    ]
    assert _compute_fp_rate_trend(batch_results) == "flat"


def test_fp_rate_trend_increasing():
    """Increasing FP rate returns 'flat'."""
    batch_results = [
        {"fp_rate": 0.1},
        {"fp_rate": 0.3},
        {"fp_rate": 0.5},
    ]
    assert _compute_fp_rate_trend(batch_results) == "flat"


def test_fp_rate_trend_single_batch():
    """Single batch returns 'flat' (no pairs to compare)."""
    assert _compute_fp_rate_trend([{"fp_rate": 0.3}]) == "flat"


def test_fp_rate_trend_empty():
    """Empty list returns 'flat'."""
    assert _compute_fp_rate_trend([]) == "flat"


def test_fp_rate_trend_partial_decrease():
    """At least one consecutive pair with strict decrease → 'decreasing'."""
    batch_results = [
        {"fp_rate": 0.5},
        {"fp_rate": 0.6},  # up
        {"fp_rate": 0.4},  # down — this is sufficient
    ]
    assert _compute_fp_rate_trend(batch_results) == "decreasing"


# ---------------------------------------------------------------------------
# Tests: _build_artifact
# ---------------------------------------------------------------------------


def test_build_artifact_required_fields(tmp_path):
    """Artifact contains all REQUIRED_RESULT_FIELDS from ExperimentTemplate."""
    from scripts.experiment_template import REQUIRED_RESULT_FIELDS

    tmpl = _make_tmpl(tmp_path)
    artifact = _build_artifact(
        tmpl=tmpl,
        batch_results=[{"batch_id": 0, "violations_found": 2, "fp_rate": 0.1, "accuracy": 0.6}],
        total_violations_found=2,
        n_constraints_added=1,
        fp_rate_trend="flat",
        inference_mode="live_gpu",
    )
    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


def test_build_artifact_fr11_confirmed_when_violations(tmp_path):
    """fr11_real_violations_confirmed=True when total_violations_found > 0."""
    tmpl = _make_tmpl(tmp_path)
    artifact = _build_artifact(
        tmpl,
        batch_results=[],
        total_violations_found=3,
        n_constraints_added=0,
        fp_rate_trend="flat",
        inference_mode="live_gpu",
    )
    assert artifact["fr11_real_violations_confirmed"] is True
    assert artifact["honest_verdict"] == "fr11_real_violations_confirmed"


def test_build_artifact_fr11_not_confirmed_zero_violations(tmp_path):
    """fr11_real_violations_confirmed=False when total_violations_found == 0."""
    tmpl = _make_tmpl(tmp_path)
    artifact = _build_artifact(
        tmpl,
        batch_results=[],
        total_violations_found=0,
        n_constraints_added=0,
        fp_rate_trend="flat",
        inference_mode="live_gpu",
    )
    assert artifact["fr11_real_violations_confirmed"] is False
    assert artifact["honest_verdict"] == "fr11_still_zero_violations"


def test_build_artifact_blocked_no_gate(tmp_path):
    """blocked_no_gate inference_mode yields blocked honest_verdict."""
    tmpl = _make_tmpl(tmp_path)
    artifact = _build_artifact(
        tmpl,
        batch_results=[],
        total_violations_found=0,
        n_constraints_added=0,
        fp_rate_trend="flat",
        inference_mode="blocked_no_gate",
        status="blocked",
    )
    assert artifact["honest_verdict"] == "blocked_no_gate"


def test_build_artifact_schema_field(tmp_path):
    """Artifact contains result_schema='carnot.fr11_relay_real.v1'; 'schema' is key list."""
    tmpl = _make_tmpl(tmp_path)
    artifact = _build_artifact(
        tmpl,
        batch_results=[],
        total_violations_found=1,
        n_constraints_added=0,
        fp_rate_trend="flat",
        inference_mode="live_gpu",
    )
    # build_result() always sets schema = sorted(keys) — verify it's a list
    assert isinstance(artifact["schema"], list)
    # Our named schema lives under result_schema
    assert artifact["result_schema"] == "carnot.fr11_relay_real.v1"


def test_build_artifact_batch_results_format(tmp_path):
    """batch_results in artifact is a list of 4-tuples."""
    tmpl = _make_tmpl(tmp_path)
    batch = [
        {"batch_id": 0, "violations_found": 1, "fp_rate": 0.2, "accuracy": 0.8},
        {"batch_id": 1, "violations_found": 2, "fp_rate": 0.1, "accuracy": 0.9},
    ]
    artifact = _build_artifact(
        tmpl,
        batch_results=batch,
        total_violations_found=3,
        n_constraints_added=0,
        fp_rate_trend="decreasing",
        inference_mode="live_gpu",
    )
    br = artifact["batch_results"]
    assert len(br) == 2
    assert br[0][0] == 0  # batch_id
    assert br[0][1] == 1  # violations_found
    assert br[1][2] == 0.1  # fp_rate


# ---------------------------------------------------------------------------
# Tests: _run_batches_with_coace (using synthetic generate_fn)
# ---------------------------------------------------------------------------


def _make_synthetic_generate_fn(inject_error_every: int = 2) -> Any:
    """Return a callable that generates responses with periodic arithmetic errors.

    Every inject_error_every-th call returns a response containing '47 + 28 = 76'
    (incorrect — actual value is 75) so CoACE can detect a violation.
    Otherwise returns a response with correct arithmetic.
    """
    counter = {"n": 0}

    def generate_fn(prompt: str) -> str:
        n = counter["n"]
        counter["n"] += 1
        if n % inject_error_every == 0:
            return "We compute 47 + 28 = 76, so the answer is 76. #### 76"
        return "The answer is 5. 2 + 3 = 5. #### 5"

    return generate_fn


# Type alias for the return type annotation above (not imported until runtime)
from typing import Any  # noqa: E402


def _make_questions(n: int) -> list[dict]:
    return [
        {"question": f"Question {i}?", "answer": f"#### {i * 2}"}
        for i in range(n)
    ]


def test_run_batches_finds_violations():
    """SCENARIO-LEARN-085: synthetic responses with errors → total_violations_found > 0."""
    questions = _make_questions(25)
    generate_fn = _make_synthetic_generate_fn(inject_error_every=2)
    batch_results, _ = _run_batches_with_coace(
        extractor=CoACEExtractor(),
        generate_fn=generate_fn,
        questions=questions,
        n_batches=3,
        cam_threshold=3,
    )
    total = sum(b["violations_found"] for b in batch_results)
    assert total > 0


def test_run_batches_returns_three_batches():
    """SCENARIO-LEARN-086: 3 batches returned for 25 questions."""
    questions = _make_questions(25)
    generate_fn = _make_synthetic_generate_fn()
    batch_results, _ = _run_batches_with_coace(
        extractor=CoACEExtractor(),
        generate_fn=generate_fn,
        questions=questions,
        n_batches=3,
        cam_threshold=3,
    )
    assert len(batch_results) == 3


def test_run_batches_constraints_added_after_threshold():
    """ConstraintAdditionFromMemory adds 'carry_check_constraint' after threshold.

    SCENARIO-LEARN-085: violations feed cam.observe('carry', ...) and once
    threshold observations accumulate, check_and_add() returns a new constraint.
    cam_threshold=3 and every response has an error → constraint added by batch 1.
    """
    questions = _make_questions(9)  # one batch of 9; all errors → 9 observations > threshold=3
    generate_fn = _make_synthetic_generate_fn(inject_error_every=1)
    batch_results, total_constraints = _run_batches_with_coace(
        extractor=CoACEExtractor(),
        generate_fn=generate_fn,
        questions=questions,
        n_batches=1,
        cam_threshold=3,  # 3 observations triggers carry_check_constraint addition
    )
    # At least one constraint should be added after 9 > 3 violations
    assert total_constraints >= 1


def test_run_batches_clean_responses_no_violations():
    """Responses with no arithmetic expressions → violations_found=0 for all batches."""
    questions = _make_questions(25)

    def clean_fn(prompt: str) -> str:
        return "The answer is forty-two. #### 42"

    batch_results, total_constraints = _run_batches_with_coace(
        extractor=CoACEExtractor(),
        generate_fn=clean_fn,
        questions=questions,
        n_batches=3,
        cam_threshold=3,
    )
    total_violations = sum(b["violations_found"] for b in batch_results)
    assert total_violations == 0
    assert total_constraints == 0


def test_run_batches_batch_result_fields():
    """Each batch result dict contains all expected keys."""
    questions = _make_questions(25)
    generate_fn = _make_synthetic_generate_fn()
    batch_results, _ = _run_batches_with_coace(
        extractor=CoACEExtractor(),
        generate_fn=generate_fn,
        questions=questions,
        n_batches=3,
        cam_threshold=3,
    )
    expected_keys = {
        "batch_id",
        "n_questions",
        "violations_found",
        "constraints_added_this_batch",
        "fp_rate",
        "accuracy",
    }
    for b in batch_results:
        assert expected_keys.issubset(b.keys()), f"Missing keys in batch {b}"


def test_run_batches_total_questions_matches():
    """Total questions across all batches equals len(questions) (capped at 25)."""
    questions = _make_questions(25)
    generate_fn = _make_synthetic_generate_fn()
    batch_results, _ = _run_batches_with_coace(
        extractor=CoACEExtractor(),
        generate_fn=generate_fn,
        questions=questions,
        n_batches=3,
        cam_threshold=3,
    )
    total_q = sum(b["n_questions"] for b in batch_results)
    # Total questions processed should be 25 (all questions distributed across batches)
    assert total_q == 25, f"Expected 25 questions total, got {total_q}"
