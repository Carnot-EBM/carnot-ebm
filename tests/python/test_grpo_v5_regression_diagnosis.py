"""Tests for the Exp 1208 GRPO v5 regression diagnosis module.

Spec: REQ-LEARN-1219 (regression diagnosis must classify the Exp 1208
artifact as ``high_abstention_rate`` with the saturated-baseline finding
captured as a contributing factor, and must produce a non-trivial fix
recommendation for Exp 1220).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.training.grpo_v5_regression_diagnosis import (
    ALLOWED_HONEST_VERDICTS,
    ALLOWED_ROOT_CAUSES,
    REQUIRED_DIAGNOSIS_ARTIFACT_FIELDS,
    diagnose_exp1208_regression,
)


# REQ-LEARN-1219, SCENARIO-LEARN-1219: every required field is present and
# its value is in the allowed set when classifying the live Exp 1208 artifact.
def test_diagnose_real_exp1208_artifact_classifies_high_abstention() -> None:
    artifact_path = Path("results/experiment_1208_grpo_v5_tinyv_v2_dualgpu.json")
    payload = json.loads(artifact_path.read_text())
    diagnosis = diagnose_exp1208_regression(payload)

    for field in REQUIRED_DIAGNOSIS_ARTIFACT_FIELDS:
        assert field in diagnosis, f"missing required field {field}"
    assert diagnosis["root_cause"] in ALLOWED_ROOT_CAUSES
    assert diagnosis["honest_verdict"] in ALLOWED_HONEST_VERDICTS

    # The .94 artifact has 5/8 abstentions, pre=1.0, post=0.75 — the
    # classifier must call this out specifically as high-abstention,
    # not "unknown" or "implementation_bug".
    assert diagnosis["root_cause"] == "high_abstention_rate"
    assert diagnosis["honest_verdict"] == "root_cause_identified"
    assert diagnosis["diagnosis_complete"] is True
    assert pytest.approx(diagnosis["tinyv_abstention_rate_observed"], abs=1e-6) == 0.625
    assert pytest.approx(diagnosis["grpo_v5_improvement_pp"], abs=1e-6) == -35.0
    assert "saturated_baseline_eval" in " ".join(diagnosis["contributing_factors"])
    assert "exp 1220" in diagnosis["recommended_fix_for_exp1220"].lower()


# REQ-LEARN-1219, SCENARIO-LEARN-1219: when the abstention rate is
# moderate (0.3-0.5) but the regression is strong, the classifier
# returns ``threshold_misconfiguration`` — distinguishing it from the
# high-abstention case.
def test_threshold_misconfiguration_branch() -> None:
    payload = {
        "tinyv_abstention_rate": 0.4,
        "tinyv_abstention_count": 4,
        "n_train_questions": 10,
        "v5_fraction_correct_before": 0.6,
        "v5_fraction_correct_after": 0.55,
        "improvement_over_baseline_pp": -15.0,
        "dualgpu_gpu0_utilization_pct": 50.0,
        "dualgpu_gpu1_utilization_pct": 50.0,
    }
    diagnosis = diagnose_exp1208_regression(payload)
    assert diagnosis["root_cause"] == "threshold_misconfiguration"
    assert diagnosis["diagnosis_complete"] is True


# REQ-LEARN-1219, SCENARIO-LEARN-1219: a >25pp dualgpu utilization gap
# with one GPU above the imbalance threshold returns
# ``dualgpu_instability``.
def test_dualgpu_instability_branch() -> None:
    payload = {
        "tinyv_abstention_rate": 0.1,
        "tinyv_abstention_count": 1,
        "n_train_questions": 16,
        "v5_fraction_correct_before": 0.6,
        "v5_fraction_correct_after": 0.5,
        "improvement_over_baseline_pp": -12.0,
        "dualgpu_gpu0_utilization_pct": 95.0,
        "dualgpu_gpu1_utilization_pct": 20.0,
    }
    diagnosis = diagnose_exp1208_regression(payload)
    assert diagnosis["root_cause"] == "dualgpu_instability"


# REQ-LEARN-1219: a strong pre>post drop with no other signal returns
# ``reward_signal_collapse``.
def test_reward_signal_collapse_branch() -> None:
    payload = {
        "tinyv_abstention_rate": 0.0,
        "tinyv_abstention_count": 0,
        "n_train_questions": 16,
        "v5_fraction_correct_before": 0.7,
        "v5_fraction_correct_after": 0.4,
        "improvement_over_baseline_pp": -25.0,
        "dualgpu_gpu0_utilization_pct": 50.0,
        "dualgpu_gpu1_utilization_pct": 50.0,
    }
    diagnosis = diagnose_exp1208_regression(payload)
    assert diagnosis["root_cause"] == "reward_signal_collapse"
    assert diagnosis["diagnosis_complete"] is True


# REQ-LEARN-1219: when no hypothesis fires and the regression is small,
# the classifier returns ``unknown`` rather than fabricating a root cause.
def test_unknown_branch_when_no_hypothesis_fires() -> None:
    payload = {
        "tinyv_abstention_rate": 0.1,
        "tinyv_abstention_count": 1,
        "n_train_questions": 16,
        "v5_fraction_correct_before": 0.5,
        "v5_fraction_correct_after": 0.49,
        "improvement_over_baseline_pp": -1.0,
        "dualgpu_gpu0_utilization_pct": 50.0,
        "dualgpu_gpu1_utilization_pct": 50.0,
    }
    diagnosis = diagnose_exp1208_regression(payload)
    assert diagnosis["root_cause"] == "unknown"
    assert diagnosis["honest_verdict"] == "root_cause_unknown"
    assert diagnosis["diagnosis_complete"] is False


# REQ-LEARN-1219: a strong regression with no fitting hypothesis falls
# through to ``implementation_bug`` so the planner blocks Exp 1220 on a
# code review rather than re-running blind.
def test_implementation_bug_branch_for_unexplained_regression() -> None:
    payload = {
        "tinyv_abstention_rate": 0.1,
        "tinyv_abstention_count": 1,
        "n_train_questions": 16,
        "v5_fraction_correct_before": 0.6,
        "v5_fraction_correct_after": 0.55,
        "improvement_over_baseline_pp": -20.0,
        "dualgpu_gpu0_utilization_pct": 50.0,
        "dualgpu_gpu1_utilization_pct": 50.0,
    }
    diagnosis = diagnose_exp1208_regression(payload)
    assert diagnosis["root_cause"] == "implementation_bug"


# REQ-LEARN-1219: the diagnosis artifact must validate against the
# allowed-value enums even on a minimal/empty payload (defensive default).
def test_empty_payload_returns_unknown_without_crashing() -> None:
    diagnosis = diagnose_exp1208_regression({})
    assert diagnosis["root_cause"] == "unknown"
    assert diagnosis["honest_verdict"] == "root_cause_unknown"
    for field in REQUIRED_DIAGNOSIS_ARTIFACT_FIELDS:
        assert field in diagnosis
