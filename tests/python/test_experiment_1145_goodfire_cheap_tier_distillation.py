"""Tests for Exp 1145 Goodfire cheap-tier distillation.

Spec: REQ-VERIFY-1145, SCENARIO-VERIFY-1145.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.eval.goodfire_cheap_tier_distillation import (
    ALLOWED_HONEST_VERDICTS,
    DEFAULT_SEMENERGY_THRESHOLD,
    DEFAULT_THINKPRM_THRESHOLD,
    REQUIRED_ARTIFACT_FIELDS,
    CheapTierScore,
    ThresholdPolicy,
    build_exp1145_artifact,
    calibrate_policy,
    evaluate_policy,
    find_tp_maximizing_threshold,
    load_json_or_jsonl,
    run_experiment,
    summarize_halluguard_miss_features,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DELIVERABLE = _REPO_ROOT / "results" / "experiment_1145_goodfire_cheap_tier_distillation.json"


class _FakeThinkProbe:
    def spill_score(self, text: str, context: str = "") -> float:
        if "think-hit" in text:
            return 0.5
        if "near" in text:
            return 0.2
        return 0.0


class _FakeSemEnergyProbe:
    def score_response_proxy(self, text: str) -> float:
        return -0.4 if "sem-hit" in text else -1.0


def _score(
    row_id: str,
    *,
    category: str = "logic",
    think: float = 0.0,
    sem: float = -1.0,
    entropy: float = 1.0,
    distance: float = 0.2,
) -> CheapTierScore:
    return CheapTierScore(
        id=row_id,
        category=category,
        text=row_id,
        thinkprm_score=think,
        semenergy_score=sem,
        entropy_proxy=entropy,
        embedding_distance=distance,
    )


def test_threshold_search_handles_ge_and_strict_gt_boundaries() -> None:
    """REQ-VERIFY-1145: threshold search catches every positive exemplar."""
    assert find_tp_maximizing_threshold([0.4, 0.2, 0.8], direction="ge") == 0.2

    sem_threshold = find_tp_maximizing_threshold([-0.4, -1.2, -0.8], direction="gt")
    assert sem_threshold < -1.2
    assert sem_threshold == math.nextafter(-1.2, -math.inf)


def test_halluguard_summary_selects_entropy_for_combined_cheap_misses() -> None:
    """SCENARIO-VERIFY-1145: entropy_proxy can dominate cheap-tier misses."""
    rows = [
        _score("miss-1", entropy=0.95, distance=0.1),
        _score("miss-2", entropy=0.90, distance=0.2),
        _score("caught", think=0.7, entropy=0.1, distance=1.0),
    ]

    summary = summarize_halluguard_miss_features(
        rows,
        {
            "halluguard_features_explain_goodfire_failures": True,
            "entropy_threshold": 0.75,
            "embedding_distance_threshold": 0.85,
        },
    )

    assert summary["cheap_tier_miss_count"] == 2
    assert summary["dominant_halluguard_feature"] == "entropy_proxy"
    assert summary["entropy_proxy_miss_flag_rate"] == pytest.approx(1.0)
    assert summary["embedding_distance_miss_flag_rate"] == pytest.approx(0.0)


def test_calibration_policy_adjusts_thinkprm_when_entropy_dominates() -> None:
    """SCENARIO-VERIFY-1145: entropy-driven misses lower the ThinkPRM threshold."""
    goodfire_scores = [
        _score("a", category="arithmetic", think=0.0, sem=-1.0, entropy=0.90),
        _score("b", category="arithmetic", think=0.2, sem=-1.0, entropy=0.80),
        _score("c", category="logic", think=0.0, sem=-0.4, entropy=0.20),
        _score("d", category="logic", think=0.5, sem=-1.0, entropy=0.20),
    ]
    correct_scores = [
        _score("ok1", think=0.0, sem=-1.0, entropy=0.20),
        _score("ok2", think=0.1, sem=-1.0, entropy=0.30),
    ]
    feature_summary = {
        "halluguard_features_explain_goodfire_failures": True,
        "dominant_halluguard_feature": "entropy_proxy",
        "entropy_threshold": 0.75,
        "embedding_distance_threshold": 0.85,
    }

    policy = calibrate_policy(goodfire_scores, feature_summary)
    before = evaluate_policy(goodfire_scores, ThresholdPolicy())
    after = evaluate_policy(goodfire_scores, policy)
    correct_after = evaluate_policy(correct_scores, policy)

    assert policy.thinkprm_adjusted_threshold == 0.0
    assert policy.thinkprm_feature_gate == "entropy_proxy"
    assert policy.semenergy_adjusted_threshold is None
    assert before["combined_tp_rate"] == pytest.approx(0.5)
    assert after["combined_tp_rate"] == pytest.approx(1.0)
    assert correct_after["combined_tp_rate"] == pytest.approx(0.0)


def test_build_artifact_reports_category_persistence_and_verdict() -> None:
    """REQ-VERIFY-1145: artifact schema and verdict are deterministic."""
    goodfire_scores = [
        _score("a1", category="a", think=0.0, sem=-1.0, entropy=0.90),
        _score("a2", category="a", think=0.5, sem=-1.0, entropy=0.20),
        _score("b1", category="b", think=0.0, sem=-0.4, entropy=0.20),
        _score("b2", category="b", think=0.0, sem=-1.0, entropy=0.90),
    ]
    correct_scores = [_score("ok", think=0.0, sem=-1.0, entropy=0.20)]
    feature_summary = {
        "halluguard_features_explain_goodfire_failures": True,
        "dominant_halluguard_feature": "entropy_proxy",
        "entropy_threshold": 0.75,
        "embedding_distance_threshold": 0.85,
        "entropy_proxy_miss_flag_rate": 1.0,
        "embedding_distance_miss_flag_rate": 0.0,
        "cheap_tier_miss_count": 2,
    }
    policy = calibrate_policy(goodfire_scores, feature_summary)

    artifact = build_exp1145_artifact(
        goodfire_scores=goodfire_scores,
        correct_scores=correct_scores,
        feature_summary=feature_summary,
        policy=policy,
        exp1132_artifact={
            "per_tier_tp_rate": {
                "tier_0a_thinkprm": 0.25,
                "tier_0c_semenergy": 0.25,
            }
        },
        exp1143_artifact={
            "halluguard_features_added": ["entropy_proxy", "embedding_distance"],
            "halluguard_features_explain_goodfire_failures": True,
        },
        duration_s=0.1,
        fover_correct_n=1,
    )

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["thinkprm_threshold_adjusted"] is True
    assert artifact["semenergy_threshold_adjusted"] is False
    assert artifact["cheap_tier_tp_rate_improved"] is True
    assert artifact["honest_verdict"] == "cheap_tier_calibrated_tp_improved"
    assert artifact["category_improvement_summary"]["categories_no_worse"] == 2


def test_load_json_or_jsonl_supports_empty_array_and_jsonl(tmp_path: Path) -> None:
    """REQ-VERIFY-1145: local corpora can be JSON arrays or JSONL files."""
    empty = tmp_path / "empty.jsonl"
    array = tmp_path / "rows.json"
    jsonl = tmp_path / "rows.jsonl"
    rows = [{"id": "a"}, {"id": "b"}]
    empty.write_text("", encoding="utf-8")
    array.write_text(json.dumps(rows), encoding="utf-8")
    jsonl.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert load_json_or_jsonl(empty) == []
    assert load_json_or_jsonl(array) == rows
    assert load_json_or_jsonl(jsonl) == rows


def test_run_experiment_writes_tiny_distillation_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1145: runner writes the cheap-tier distillation artifact."""
    exemplar_path = tmp_path / "goodfire.jsonl"
    fover_path = tmp_path / "fover.json"
    exp1132_path = tmp_path / "exp1132.json"
    exp1143_path = tmp_path / "exp1143.json"
    result_path = tmp_path / "experiment_1145.json"

    exemplar_rows = [
        {"id": "ex1", "category": "a", "buggy_response": "alpha beta gamma delta"},
        {"id": "ex2", "category": "a", "buggy_response": "think-hit repeated repeated"},
        {"id": "ex3", "category": "b", "buggy_response": "sem-hit repeated repeated"},
        {"id": "ex4", "category": "b", "buggy_response": "near alpha beta gamma"},
    ]
    fover_rows = [
        {"label": "correct", "step_text": "ok ok ok ok"},
        {"label": "correct", "step_text": "safe safe safe safe"},
        {"label": "incorrect", "step_text": "ignored"},
    ]
    exemplar_path.write_text(
        "\n".join(json.dumps(row) for row in exemplar_rows) + "\n",
        encoding="utf-8",
    )
    fover_path.write_text(json.dumps(fover_rows), encoding="utf-8")
    exp1132_path.write_text(
        json.dumps(
            {
                "per_tier_tp_rate": {
                    "tier_0a_thinkprm": 0.25,
                    "tier_0c_semenergy": 0.25,
                }
            }
        ),
        encoding="utf-8",
    )
    exp1143_path.write_text(
        json.dumps(
            {
                "halluguard_features_added": ["entropy_proxy", "embedding_distance"],
                "halluguard_features_explain_goodfire_failures": True,
                "entropy_threshold": 0.75,
                "embedding_distance_threshold": 0.85,
            }
        ),
        encoding="utf-8",
    )

    artifact = run_experiment(
        exemplar_path=exemplar_path,
        fover_path=fover_path,
        exp1132_path=exp1132_path,
        exp1143_path=exp1143_path,
        result_path=result_path,
        think_probe=_FakeThinkProbe(),
        semenergy_probe=_FakeSemEnergyProbe(),
        fover_correct_n=2,
    )

    assert result_path.exists()
    assert artifact["n_exemplars"] == 4
    assert artifact["fover_correct_examples"] == 2
    assert artifact["thinkprm_threshold_adjusted"] is True
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS


def test_deliverable_exists_and_validates_required_schema() -> None:
    """REQ-VERIFY-1145: on-disk Exp 1145 artifact exposes required fields."""
    if not _DELIVERABLE.exists():
        pytest.skip(f"deliverable has not been generated yet: {_DELIVERABLE}")
    artifact = json.loads(_DELIVERABLE.read_text(encoding="utf-8"))

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["n_exemplars"] == 36
    assert artifact["thinkprm_tp_before"] == 0.138889
    assert artifact["semenergy_tp_before"] == 0.222222
    assert artifact["cheap_tier_tp_rate_improved"] is (
        artifact["combined_cheap_tp_after"] > artifact["combined_cheap_tp_before"]
    )
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS
    assert 0.0 <= artifact["false_positive_rate_after"] <= 1.0
