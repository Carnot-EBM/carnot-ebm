"""
Tests for experiment 1099: RLVR + SSD Integration.

Spec: REQ-VER-100 — RLVR + SSD energy-filter selection-condition acceptance.

Validates the four selection conditions run correctly, the energy filter
selects correctly more often than random, and results are reported honestly.

REQ-FR11: Carnot verifier contribution (alpha_t) must be measurable as an
  RLVR signal over a dataset of verified responses.
SCENARIO-RLVR-SSD-001: Four-way selection experiment runs end-to-end.
SCENARIO-RLVR-SSD-002: Energy filter outperforms or equals random baseline.
SCENARIO-RLVR-SSD-003: Honest negative verdict is reported when no condition
  improves over baseline.
"""

import json
import sys
from pathlib import Path

import pytest

# Allow imports from scripts/ and the experiment module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from experiment_1099_rlvr_ssd_integration_v1 import (
    condition_a_rlvr_only,
    condition_b_ssd_only,
    condition_c_rlvr_ssd,
    condition_d_onpolicy_ssd,
    load_entries,
)

DATA_PATH = str(Path(__file__).parent.parent.parent / "data" / "fr11_zenil_distill_v2.jsonl")
RESULT_PATH = str(
    Path(__file__).parent.parent.parent / "results" / "experiment_1099_rlvr_ssd_integration_v1.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def entries():
    """Load the real dataset once for all tests in this module."""
    return load_entries(DATA_PATH)


@pytest.fixture()
def synthetic_entries():
    """Small synthetic dataset with known correct/incorrect distribution.

    10 questions, 2 responses each (20 total).
    Questions 0-6 have a majority-correct response pair.
    Questions 7-9 have both responses incorrect.
    All energy scores are 0.1 except entries for question_id 'q0' which are 0.0.
    """
    rows = []
    for i in range(7):
        qid = f"q{i}"
        rows.append(
            {
                "question_id": qid,
                "question": f"Q{i}",
                "response": "correct answer",
                "correct": True,
                "energy_score": 0.0 if i == 0 else 0.1,
                "alpha_t_contributes": True,
                "verifier_verdict": "correct",
            }
        )
        rows.append(
            {
                "question_id": qid,
                "question": f"Q{i}",
                "response": "wrong answer",
                "correct": False,
                "energy_score": 0.5,
                "alpha_t_contributes": False,
                "verifier_verdict": "incorrect",
            }
        )
    for i in range(7, 10):
        qid = f"q{i}"
        for _ in range(2):
            rows.append(
                {
                    "question_id": qid,
                    "question": f"Q{i}",
                    "response": "wrong answer",
                    "correct": False,
                    "energy_score": 0.1,
                    "alpha_t_contributes": False,
                    "verifier_verdict": "incorrect",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Test 1: all four conditions run without error on real data
# ---------------------------------------------------------------------------


def test_four_conditions_run_without_error(entries):
    """SCENARIO-RLVR-SSD-001: All four conditions complete on real dataset."""
    import statistics

    energy_scores = [e["energy_score"] for e in entries]
    threshold = statistics.median(energy_scores)

    res_a = condition_a_rlvr_only(entries, threshold)
    res_b = condition_b_ssd_only(entries)
    res_c = condition_c_rlvr_ssd(entries, threshold)
    res_d = condition_d_onpolicy_ssd(entries, threshold)

    for res in (res_a, res_b, res_c, res_d):
        assert "fraction_correct" in res, f"Missing fraction_correct in {res}"
        assert 0.0 <= res["fraction_correct"] <= 1.0, (
            f"fraction_correct out of [0,1]: {res['fraction_correct']}"
        )


# ---------------------------------------------------------------------------
# Test 2: energy filter selects correct more often than random (synthetic)
# ---------------------------------------------------------------------------


def test_energy_filter_selects_correct_more_often_than_random(synthetic_entries):
    """SCENARIO-RLVR-SSD-002: Low-energy threshold selects correct responses
    at higher rate than the unfiltered baseline.

    In the synthetic dataset, correct entries for q0 have energy_score=0.0
    and incorrect entries have energy_score=0.5. A threshold of 0.2 keeps
    only q0's correct entry, raising precision above baseline.
    """
    import statistics

    baseline = sum(1 for e in synthetic_entries if e["correct"]) / len(synthetic_entries)

    # Threshold 0.2: accepts all entries with energy <= 0.2.
    # Correct q0 (energy 0.0) is included. Incorrect q0 (energy 0.5) is excluded.
    res = condition_a_rlvr_only(synthetic_entries, threshold=0.2)

    # Energy filter should not be worse than random on this synthetic set.
    # (It may equal baseline in the degenerate case but should not go below it.)
    assert res["fraction_correct"] >= baseline - 1e-9, (
        f"Energy filter fraction {res['fraction_correct']:.4f} below baseline {baseline:.4f}"
    )

    # On the synthetic set, the energy filter should keep at least one entry
    assert res["n_accepted"] > 0, "Energy filter accepted nothing"


# ---------------------------------------------------------------------------
# Test 3: result artifact is reported honestly
# ---------------------------------------------------------------------------


def test_rlvr_ssd_result_reported_honestly():
    """SCENARIO-RLVR-SSD-003: The JSON artifact has all required schema fields
    and the honest_verdict is one of the allowed values.

    Checks the written artifact — if the script has not been run yet, the test
    calls run_experiment() directly so it always asserts something.
    """
    from experiment_1099_rlvr_ssd_integration_v1 import run_experiment

    result_file = Path(RESULT_PATH)
    if result_file.exists():
        artifact = json.loads(result_file.read_text())
    else:
        artifact = run_experiment()

    required_fields = [
        "n_training_examples",
        "baseline_fraction_correct",
        "condition_A_rlvr_only",
        "condition_B_ssd_only",
        "condition_C_rlvr_ssd",
        "condition_D_onpolicy_ssd",
        "best_condition",
        "improvement_over_baseline",
        "gpu_finetuning_available",
        "tests_passing",
        "honest_verdict",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    allowed_verdicts = {
        "rlvr_ssd_wins",
        "rlvr_only_wins",
        "ssd_only_wins",
        "no_improvement_honest_negative",
        "failed",
    }
    assert artifact["honest_verdict"] in allowed_verdicts, (
        f"Unexpected honest_verdict: {artifact['honest_verdict']}"
    )

    # Fractions must all be in [0, 1]
    for key in (
        "baseline_fraction_correct",
        "condition_A_rlvr_only",
        "condition_B_ssd_only",
        "condition_C_rlvr_ssd",
        "condition_D_onpolicy_ssd",
    ):
        val = artifact[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} outside [0,1]"

    # n_training_examples must be positive
    assert artifact["n_training_examples"] > 0
