"""Tests for Exp 1157 SECL cheap-tier calibration.

Spec: REQ-VERIFY-1157, SCENARIO-VERIFY-1157.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot.eval.goodfire_cheap_tier_distillation import CheapTierScore
from carnot.eval.secl_cheap_tier_calibration import (
    ALLOWED_HONEST_VERDICTS,
    CLASS_WEIGHT_CORRECT,
    CLASS_WEIGHT_FAILURE,
    FPR_BUDGET,
    REQUIRED_ARTIFACT_FIELDS,
    TP_TARGET,
    SECLCalibrationExample,
    build_calibration_examples,
    build_exp1157_artifact,
    cheap_tier_score,
    choose_operating_threshold,
    discriminative_signal,
    evaluate_secl_probe,
    run_experiment,
    train_secl_probe,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DELIVERABLE = _REPO_ROOT / "results" / "experiment_1157_secl_cheap_tier_calibration.json"


class _FakeThinkProbe:
    def spill_score(self, text: str, context: str = "") -> float:
        if "failure" in text:
            return 0.08
        if "mixed" in text:
            return 0.20
        return 0.60


class _FakeSemEnergyProbe:
    def score_response_proxy(self, text: str) -> float:
        if "failure" in text:
            return -0.40
        if "mixed" in text:
            return -0.90
        return -1.20


def _cheap_score(
    row_id: str,
    *,
    think: float,
    sem: float,
) -> CheapTierScore:
    return CheapTierScore(
        id=row_id,
        category="logic",
        text=row_id,
        thinkprm_score=think,
        semenergy_score=sem,
        entropy_proxy=0.5,
        embedding_distance=0.1,
    )


def test_discriminative_signal_uses_true_logprob_then_sigmoid_fallback() -> None:
    """REQ-VERIFY-1157: discriminative signal approximates P(True)."""
    assert discriminative_signal(0.25, thinkprm_true_logprob=math.log(0.8)) == pytest.approx(0.8)
    assert discriminative_signal(0.0) == pytest.approx(0.5)

    assert cheap_tier_score(0.50, -0.40) > cheap_tier_score(0.10, -1.20)


def test_build_calibration_examples_labels_goodfire_and_correct_rows() -> None:
    """REQ-VERIFY-1157: calibration rows contain cheap score and P(True)."""
    goodfire = [_cheap_score("bad", think=0.10, sem=-0.40)]
    correct = [_cheap_score("ok", think=0.00, sem=-1.20)]

    examples = build_calibration_examples(goodfire, correct)

    assert [example.label for example in examples] == [1, 0]
    assert examples[0].cheap_tier_score > examples[1].cheap_tier_score
    assert examples[0].discriminative_signal == pytest.approx(discriminative_signal(0.10))
    assert examples[1].source == "fover_correct"


def test_weighted_logistic_probe_meets_precision_budget_on_separable_rows() -> None:
    """SCENARIO-VERIFY-1157: weighted probe can satisfy TP/FPR targets."""
    failures = [
        SECLCalibrationExample(f"bad-{idx}", 1, 1.4 + idx * 0.05, 0.20, "goodfire")
        for idx in range(8)
    ]
    correct = [
        SECLCalibrationExample(f"ok-{idx}", 0, -0.4 + idx * 0.01, 0.80, "fover_correct")
        for idx in range(20)
    ]
    examples = [*failures, *correct]

    probe = train_secl_probe(examples)
    probabilities = probe.predict_proba(np.array([example.features for example in examples]))
    threshold = choose_operating_threshold(
        probabilities,
        np.array([example.label for example in examples]),
        fpr_budget=FPR_BUDGET,
    )
    metrics = evaluate_secl_probe(examples, probe, operating_threshold=threshold)

    assert probe.class_weights == {0: CLASS_WEIGHT_CORRECT, 1: CLASS_WEIGHT_FAILURE}
    assert metrics["secl_tp_rate"] >= TP_TARGET
    assert metrics["secl_fpr"] <= FPR_BUDGET
    assert metrics["discriminative_signal_used"] is True
    assert bool(probe.predict_flags(examples[0].features)[0]) is True


def test_build_artifact_documents_exp1145_thresholds_and_verdict() -> None:
    """REQ-VERIFY-1157: artifact records Exp1145 thresholds and SECL result."""
    examples = [
        SECLCalibrationExample("bad", 1, 1.2, 0.2, "goodfire"),
        SECLCalibrationExample("ok", 0, -0.2, 0.8, "fover_correct"),
    ]
    metrics = {
        "secl_tp_rate": 1.0,
        "secl_fpr": 0.0,
        "precision_recall_improved": True,
        "discriminative_signal_used": True,
        "cheap_tier_fpr_below_30pct": True,
        "cheap_tier_tp_above_80pct": True,
        "operating_threshold": 0.5,
    }

    artifact = build_exp1157_artifact(
        examples=examples,
        metrics=metrics,
        exp1145_artifact={
            "combined_cheap_tp_after": 0.916667,
            "false_positive_rate_after": 0.96,
            "thinkprm_default_threshold": 0.372,
            "thinkprm_threshold_after": 0.0,
            "thinkprm_threshold_adjusted": True,
            "semenergy_default_threshold": -0.5,
            "semenergy_threshold_after": -0.5,
            "semenergy_threshold_adjusted": False,
        },
        duration_s=0.01,
    )

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["n_exemplars"] == 1
    assert artifact["n_correct_examples"] == 1
    assert artifact["thinkprm_fpr_exp1145"] == pytest.approx(0.96)
    assert artifact["thinkprm_tp_exp1145"] == pytest.approx(0.917)
    assert artifact["exp1145_thresholds"]["thinkprm"]["default"] == pytest.approx(0.372)
    assert artifact["exp1145_thresholds"]["thinkprm"]["applied"] == pytest.approx(0.0)
    assert artifact["honest_verdict"] == "calibrated_tp_above_80_fpr_below_30"


def test_build_artifact_reports_all_honest_negative_verdict_branches() -> None:
    """REQ-VERIFY-1157: honest verdicts distinguish target miss modes."""
    examples = [
        SECLCalibrationExample("bad", 1, 1.2, 0.2, "goodfire"),
        SECLCalibrationExample("ok", 0, -0.2, 0.8, "fover_correct"),
    ]
    exp1145 = {
        "combined_cheap_tp_after": 0.916667,
        "false_positive_rate_after": 0.96,
        "thinkprm_default_threshold": 0.372,
        "thinkprm_threshold_after": 0.0,
        "thinkprm_threshold_adjusted": True,
        "semenergy_default_threshold": -0.5,
        "semenergy_threshold_after": -0.5,
        "semenergy_threshold_adjusted": False,
    }
    base_metrics = {
        "precision_recall_improved": False,
        "discriminative_signal_used": True,
        "cheap_tier_fpr_below_30pct": False,
        "cheap_tier_tp_above_80pct": True,
        "operating_threshold": 0.5,
    }

    reduced_not_gate = build_exp1157_artifact(
        examples=examples,
        metrics={**base_metrics, "secl_tp_rate": 0.85, "secl_fpr": 0.50},
        exp1145_artifact=exp1145,
        duration_s=0.01,
    )
    tradeoff = build_exp1157_artifact(
        examples=examples,
        metrics={**base_metrics, "secl_tp_rate": 0.50, "secl_fpr": 0.20},
        exp1145_artifact=exp1145,
        duration_s=0.01,
    )
    negative = build_exp1157_artifact(
        examples=examples,
        metrics={**base_metrics, "secl_tp_rate": 0.50, "secl_fpr": 0.99},
        exp1145_artifact=exp1145,
        duration_s=0.01,
    )

    assert reduced_not_gate["honest_verdict"] == "tp_improved_fpr_reduced_not_gate"
    assert tradeoff["honest_verdict"] == "trade_off_tp_dropped"
    assert negative["honest_verdict"] == "honest_negative_no_improvement"


def test_run_experiment_writes_tiny_secl_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1157: runner writes the SECL calibration artifact."""
    exemplar_path = tmp_path / "goodfire.jsonl"
    fover_path = tmp_path / "fover.json"
    exp1145_path = tmp_path / "exp1145.json"
    result_path = tmp_path / "experiment_1157.json"

    exemplar_rows = [
        {"id": "ex1", "category": "a", "buggy_response": "failure one"},
        {"id": "ex2", "category": "a", "buggy_response": "failure two"},
        {"id": "ex3", "category": "b", "buggy_response": "mixed failure"},
    ]
    fover_rows = [
        {"label": "correct", "step_text": "correct one"},
        {"label": "correct", "step_text": "correct two"},
        {"label": "incorrect", "step_text": "ignored"},
    ]
    exemplar_path.write_text(
        "\n".join(json.dumps(row) for row in exemplar_rows) + "\n",
        encoding="utf-8",
    )
    fover_path.write_text(json.dumps(fover_rows), encoding="utf-8")
    exp1145_path.write_text(
        json.dumps(
            {
                "combined_cheap_tp_after": 0.916667,
                "false_positive_rate_after": 0.96,
                "thinkprm_default_threshold": 0.372,
                "thinkprm_threshold_after": 0.0,
                "thinkprm_threshold_adjusted": True,
                "semenergy_default_threshold": -0.5,
                "semenergy_threshold_after": -0.5,
                "semenergy_threshold_adjusted": False,
            }
        ),
        encoding="utf-8",
    )

    artifact = run_experiment(
        exemplar_path=exemplar_path,
        fover_path=fover_path,
        exp1145_path=exp1145_path,
        result_path=result_path,
        think_probe=_FakeThinkProbe(),
        semenergy_probe=_FakeSemEnergyProbe(),
        fover_correct_n=2,
    )

    assert result_path.exists()
    assert artifact["n_exemplars"] == 3
    assert artifact["n_correct_examples"] == 2
    assert artifact["discriminative_signal_used"] is True
    assert artifact["class_weight_failure"] == pytest.approx(FPR_BUDGET / TP_TARGET)
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS


def test_deliverable_exists_and_validates_required_schema() -> None:
    """REQ-VERIFY-1157: on-disk Exp 1157 artifact exposes required fields."""
    if not _DELIVERABLE.exists():
        pytest.skip(f"deliverable has not been generated yet: {_DELIVERABLE}")
    artifact = json.loads(_DELIVERABLE.read_text(encoding="utf-8"))

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["n_exemplars"] == 36
    assert artifact["n_correct_examples"] >= 200
    assert artifact["thinkprm_fpr_exp1145"] == pytest.approx(0.96)
    assert artifact["thinkprm_tp_exp1145"] == pytest.approx(0.917)
    assert artifact["discriminative_signal_used"] is True
    assert artifact["cheap_tier_fpr_below_30pct"] is (artifact["secl_fpr"] <= 0.30)
    assert artifact["cheap_tier_tp_above_80pct"] is (artifact["secl_tp_rate"] >= 0.80)
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS
