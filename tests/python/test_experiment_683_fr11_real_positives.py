"""Tests for scripts/experiment_683_fr11_real_positives.py.

Covers:
- VerifiedRepairPair dataclass construction (SCENARIO-LEARN-072, SCENARIO-LEARN-073)
- build_repair_pairs generates correct verified/unverified split
- wire_repairs_into_library wires only verified=True pairs (SCENARIO-LEARN-072)
- wire_repairs_into_library skips repair_verified_correct=False pairs (SCENARIO-LEARN-073)
- compute_honest_verdict covers all three branches (REQ-LEARN-043-4)
- build_synthetic_test_questions returns 10 questions
- load_exp668_questions reads live_pairs and n_post_correct correctly
- run_experiment produces artifact with all required fields
- run_experiment honest_verdict is correct for real 668 data

Spec: REQ-LEARN-042, REQ-LEARN-043, SCENARIO-LEARN-072, SCENARIO-LEARN-073
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_683_fr11_real_positives import (
    VerifiedRepairPair,
    build_repair_pairs,
    build_synthetic_test_questions,
    compute_honest_verdict,
    load_exp668_questions,
    run_experiment,
    wire_repairs_into_library,
)
from python.carnot.pipeline.constraint_template_library import ViolationPatternLibrary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_library(tmp_path):
    """Provide an isolated ViolationPatternLibrary backed by a temp file."""
    lib_path = str(tmp_path / "test_constraint_templates.json")
    return ViolationPatternLibrary(lib_path), lib_path


@pytest.fixture()
def fake_668_result(tmp_path):
    """Create a minimal Exp 668 result JSON with a matching live_pairs file.

    WHY A FIXTURE:
        The real 668 result on disk requires specific live_pairs path resolution.
        This fixture creates a fully self-contained fake so tests are hermetic.
    """
    pairs = [
        {
            "question_index": i,
            "question": f"What is {i} plus {i}?",
            "model": "test_model",
            "response": "The answer is X.",
            "is_correct": False,
            "fover_labels": ["not_verifiable"],
        }
        for i in range(5)
    ]
    pairs_path = tmp_path / "live_pairs_fake.json"
    pairs_path.write_text(json.dumps(pairs))

    result = {
        "experiment": 668,
        "schema": "carnot.vr_attempt_18_v2.v1",
        "n_questions": 5,
        "n_post_correct": 3,
        "live_pairs_source": str(pairs_path),
        "status": "success",
    }
    result_path = tmp_path / "experiment_668_fake.json"
    result_path.write_text(json.dumps(result))
    return result_path


# ---------------------------------------------------------------------------
# VerifiedRepairPair dataclass
# ---------------------------------------------------------------------------


def test_verified_repair_pair_fields():
    """VerifiedRepairPair holds expected fields with correct types. (SCENARIO-LEARN-072)"""
    pair = VerifiedRepairPair(
        question="What is 5 + 7?",
        violated_constraint="structured_arithmetic_forcing",
        repair_response="COMPUTE: 5 + 7 = 12",
        repair_verified_correct=True,
    )
    assert pair.question == "What is 5 + 7?"
    assert pair.violated_constraint == "structured_arithmetic_forcing"
    assert pair.repair_response == "COMPUTE: 5 + 7 = 12"
    assert pair.repair_verified_correct is True


def test_verified_repair_pair_unverified():
    """VerifiedRepairPair with repair_verified_correct=False is valid. (SCENARIO-LEARN-073)"""
    pair = VerifiedRepairPair(
        question="What is 2 + 2?",
        violated_constraint="carry_check",
        repair_response="COMPUTE: 2 + 2 = 5 (wrong)",
        repair_verified_correct=False,
    )
    assert pair.repair_verified_correct is False


# ---------------------------------------------------------------------------
# build_repair_pairs
# ---------------------------------------------------------------------------


def test_build_repair_pairs_length():
    """build_repair_pairs returns one pair per question."""
    questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    pairs = build_repair_pairs(questions, n_verified_correct=3)
    assert len(pairs) == 5


def test_build_repair_pairs_verified_split():
    """First n_verified_correct pairs are verified=True; rest are False."""
    questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    pairs = build_repair_pairs(questions, n_verified_correct=3)
    assert all(p.repair_verified_correct for p in pairs[:3])
    assert all(not p.repair_verified_correct for p in pairs[3:])


def test_build_repair_pairs_zero_verified():
    """n_verified_correct=0 produces all unverified pairs."""
    questions = ["Q1", "Q2"]
    pairs = build_repair_pairs(questions, n_verified_correct=0)
    assert all(not p.repair_verified_correct for p in pairs)


def test_build_repair_pairs_compute_tag():
    """Each repair_response contains 'COMPUTE:' tag."""
    questions = ["What is 5 + 7?"]
    pairs = build_repair_pairs(questions, n_verified_correct=1)
    assert "COMPUTE:" in pairs[0].repair_response


def test_build_repair_pairs_synthetic_label():
    """Each repair_response includes the synthetic marker."""
    questions = ["A question"]
    pairs = build_repair_pairs(questions, n_verified_correct=1)
    assert "synthetic_compute_pattern" in pairs[0].repair_response


# ---------------------------------------------------------------------------
# wire_repairs_into_library
# ---------------------------------------------------------------------------


def test_wire_repairs_wires_only_verified(temp_library):
    """Only pairs with repair_verified_correct=True are wired. (SCENARIO-LEARN-072)"""
    lib, _ = temp_library
    pairs = [
        VerifiedRepairPair("Q1", "c1", "COMPUTE: r1", True),
        VerifiedRepairPair("Q2", "c2", "COMPUTE: r2", False),
        VerifiedRepairPair("Q3", "c3", "COMPUTE: r3", True),
    ]
    n_wired = wire_repairs_into_library(pairs, lib)
    assert n_wired == 2
    assert len(lib.templates) == 2


def test_wire_repairs_skips_unverified(temp_library):
    """Pairs with repair_verified_correct=False are NOT added. (SCENARIO-LEARN-073)"""
    lib, _ = temp_library
    pairs = [
        VerifiedRepairPair("Q1", "c1", "COMPUTE: wrong", False),
    ]
    n_wired = wire_repairs_into_library(pairs, lib)
    assert n_wired == 0
    assert len(lib.templates) == 0


def test_wire_repairs_violation_type(temp_library):
    """Wired entries have violation_type='verified_repair'. (SCENARIO-LEARN-072)"""
    lib, _ = temp_library
    pairs = [VerifiedRepairPair("Q1", "c1", "COMPUTE: r1", True)]
    wire_repairs_into_library(pairs, lib)
    assert lib.templates[0].violation_type == "verified_repair"


def test_wire_repairs_source_experiment(temp_library):
    """Wired entries have source_experiment=683."""
    lib, _ = temp_library
    pairs = [VerifiedRepairPair("Q1", "c1", "COMPUTE: r1", True)]
    wire_repairs_into_library(pairs, lib)
    assert lib.templates[0].source_experiment == 683


def test_wire_repairs_deduplication(temp_library):
    """Wiring the same pattern twice does not create duplicates."""
    lib, _ = temp_library
    pairs = [
        VerifiedRepairPair("Q1", "c1", "COMPUTE: same_pattern", True),
        VerifiedRepairPair("Q1", "c1", "COMPUTE: same_pattern", True),
    ]
    n_wired = wire_repairs_into_library(pairs, lib)
    # n_wired counts calls, but library deduplicates storage
    assert n_wired == 2
    assert len(lib.templates) == 1


# ---------------------------------------------------------------------------
# compute_honest_verdict
# ---------------------------------------------------------------------------


def test_verdict_no_positives_available():
    """n_positives_wired=0 → 'no_positives_available'."""
    assert compute_honest_verdict(0, -0.1) == "no_positives_available"


def test_verdict_positives_wired_fp_reduced():
    """n_positives_wired>0 and fp_rate_delta<0 → 'positives_wired_fp_reduced'."""
    assert compute_honest_verdict(5, -0.05) == "positives_wired_fp_reduced"


def test_verdict_positives_wired_no_fp_change_zero():
    """fp_rate_delta=0.0 → 'positives_wired_no_fp_change' (not negative)."""
    assert compute_honest_verdict(5, 0.0) == "positives_wired_no_fp_change"


def test_verdict_positives_wired_no_fp_change_positive():
    """fp_rate_delta>0 → 'positives_wired_no_fp_change'."""
    assert compute_honest_verdict(3, 0.1) == "positives_wired_no_fp_change"


# ---------------------------------------------------------------------------
# build_synthetic_test_questions
# ---------------------------------------------------------------------------


def test_build_synthetic_test_questions_count():
    """Returns exactly 10 synthetic test questions."""
    qs = build_synthetic_test_questions()
    assert len(qs) == 10


def test_build_synthetic_test_questions_strings():
    """All returned items are non-empty strings."""
    qs = build_synthetic_test_questions()
    assert all(isinstance(q, str) and len(q) > 0 for q in qs)


def test_build_synthetic_test_questions_no_compute_tag():
    """Synthetic questions do not contain COMPUTE: (would create spurious FP matches)."""
    qs = build_synthetic_test_questions()
    assert not any("COMPUTE:" in q for q in qs)


# ---------------------------------------------------------------------------
# load_exp668_questions
# ---------------------------------------------------------------------------


def test_load_exp668_questions_count(fake_668_result):
    """Loads n_questions questions from live_pairs (fixture has 5)."""
    questions, n_verified = load_exp668_questions(fake_668_result)
    assert len(questions) == 5


def test_load_exp668_questions_n_verified(fake_668_result):
    """n_verified_correct matches n_post_correct from artifact (fixture: 3)."""
    _, n_verified = load_exp668_questions(fake_668_result)
    assert n_verified == 3


def test_load_exp668_questions_truncates_to_n_questions(tmp_path):
    """n_verified_correct is capped at len(questions) when n_post_correct exceeds pairs."""
    pairs = [{"question": "Q", "model": "m", "response": "r", "is_correct": False,
              "fover_labels": [], "question_index": 0}]
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps(pairs))
    result = {
        "experiment": 668, "schema": "s", "n_questions": 1,
        "n_post_correct": 999, "live_pairs_source": str(pairs_path), "status": "success",
    }
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps(result))
    questions, n_verified = load_exp668_questions(result_path)
    assert n_verified == 1  # capped at len(questions)


# ---------------------------------------------------------------------------
# run_experiment integration
# ---------------------------------------------------------------------------


def test_run_experiment_artifact_fields(fake_668_result, tmp_path):
    """run_experiment returns dict with all required fields."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=fake_668_result, library_path=lib_path)
    required = [
        "n_positives_wired",
        "fp_rate_before",
        "fp_rate_after",
        "fp_rate_delta",
        "n_constraints_updated",
        "fr11_real_positives_confirmed",
        "honest_verdict",
        "n_questions_from_668",
        "n_verified_correct_from_668",
        "n_test_questions",
    ]
    for field in required:
        assert field in result, f"Missing field: {field}"


def test_run_experiment_n_positives_wired(fake_668_result, tmp_path):
    """n_positives_wired equals n_verified_correct from fake 668 artifact (3)."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=fake_668_result, library_path=lib_path)
    assert result["n_positives_wired"] == 3


def test_run_experiment_fr11_confirmed(fake_668_result, tmp_path):
    """fr11_real_positives_confirmed is True when n_positives_wired > 0."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=fake_668_result, library_path=lib_path)
    assert result["fr11_real_positives_confirmed"] is True


def test_run_experiment_honest_verdict_is_string(fake_668_result, tmp_path):
    """honest_verdict is one of the expected string values."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=fake_668_result, library_path=lib_path)
    valid_verdicts = {
        "positives_wired_fp_reduced",
        "positives_wired_no_fp_change",
        "no_positives_available",
    }
    assert result["honest_verdict"] in valid_verdicts


def test_run_experiment_fp_rate_before_zero_on_empty_lib(fake_668_result, tmp_path):
    """fp_rate_before is 0.0 when library starts empty (no prior patterns)."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=fake_668_result, library_path=lib_path)
    assert result["fp_rate_before"] == 0.0


def test_run_experiment_fp_delta_equals_after_minus_before(fake_668_result, tmp_path):
    """fp_rate_delta == fp_rate_after - fp_rate_before."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=fake_668_result, library_path=lib_path)
    assert abs(result["fp_rate_delta"] - (result["fp_rate_after"] - result["fp_rate_before"])) < 1e-9


def test_run_experiment_n_test_questions(fake_668_result, tmp_path):
    """n_test_questions is 10 (the fixed synthetic set size)."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=fake_668_result, library_path=lib_path)
    assert result["n_test_questions"] == 10


def test_run_experiment_real_668_data(tmp_path):
    """run_experiment with real Exp 668 data wires n_post_correct=25 pairs.

    WHY THIS TEST:
        Ensures the experiment actually works on the real artifact on disk, not
        just the synthetic fixture.  This is the key integration check.
    """
    real_668 = Path(_REPO_ROOT) / "results" / "experiment_668_vr_attempt_18_v2.json"
    if not real_668.exists():
        pytest.skip("Real Exp 668 artifact not on disk — skipping integration test")
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(exp668_result_path=real_668, library_path=lib_path)
    # Exp 668 reports n_post_correct=25, n_questions=25
    assert result["n_positives_wired"] == 25
    assert result["n_verified_correct_from_668"] == 25
    assert result["fr11_real_positives_confirmed"] is True
    assert result["honest_verdict"] in {
        "positives_wired_fp_reduced",
        "positives_wired_no_fp_change",
    }
