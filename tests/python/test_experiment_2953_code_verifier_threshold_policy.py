"""Tests for Exp 2953 code-verifier threshold policy.

Spec: REQ-CODE-2953, SCENARIO-CODE-2953,
SCENARIO-CODE-2953-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import code_verifier_threshold_policy as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exp2940(tmp_path: Path, *, include_distribution: bool = True) -> None:
    payload: dict[str, Any] = {
        "artifact": "experiment_2940_verifier_ensemble_auprc_code_corpora_v1",
        "code_corpus_auprc": 0.8888888888888888,
        "code_corpus_candidate_count": 320,
        "code_corpus_positive_count": 24,
        "code_corpus_empirical_positive_rate": 0.075,
        "code_corpus_baseline_random_auprc": {"value": 0.075},
        "max_f1_operating_point": {
            "threshold": 1.0,
            "ppv": 24 / 27,
            "recall": 1.0,
            "f1": 0.9411764705882353,
        },
        "precision_recall_curve": [
            {
                "threshold": 1.0,
                "ppv": 24 / 27,
                "recall": 1.0,
                "f1": 0.9411764705882353,
            },
            {
                "threshold": 0.5,
                "ppv": 24 / 33,
                "recall": 1.0,
                "f1": 0.8421052631578948,
            },
            {
                "threshold": 0.25,
                "ppv": 24 / 155,
                "recall": 1.0,
                "f1": 0.26815642458100564,
            },
            {"threshold": 0.0, "ppv": 0.075, "recall": 1.0, "f1": 0.13953488372093023},
        ],
        "paper_v6_recommendation": {"value": "retain"},
    }
    if include_distribution:
        payload["code_status_energy_values"] = [0.0] * 27 + [1.0] * 6 + [2.0] * 122 + [3.0] * 165
    _write_json(tmp_path / exp.EXP2940_REL_PATH, payload)


def _write_exp2943(tmp_path: Path) -> None:
    _write_json(
        tmp_path / exp.EXP2943_REL_PATH,
        {
            "artifact": "experiment_2943_cross_corpus_matrix_v11",
            "matrix_v11_ready": True,
            "rows_clean": ["exp2940_code_corpus_auprc_corrigendum"],
            "rows_flagged": ["exp2911_code_hallucination_verifier"],
            "per_corpus_auprc": {
                "code_corpora": {
                    "source_experiment_id": "exp2940",
                    "source_field": "code_corpus_auprc",
                    "baseline_random_auprc": 0.075,
                    "value": 0.8888888888888888,
                }
            },
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 14.5,
        tests_run=("focused-pytest",),
    )


def test_req_code_2953_spec_anchor_exists() -> None:
    """REQ-CODE-2953, SCENARIO-CODE-2953: Exp 2953 is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2953" in spec
    assert "SCENARIO-CODE-2953" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_scenario_code_2953_builds_threshold_policy_from_exp2940_summary(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2953: policy reports PPV/recall/FAR tradeoffs."""

    _write_exp2940(tmp_path)
    _write_exp2943(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["threshold_policy_ready"] is True
    assert artifact["missing_score_distribution"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert artifact["selected_default_threshold"] == pytest.approx(1.0)
    assert artifact["expected_ppv_at_default"] == pytest.approx(24 / 27)
    assert artifact["expected_recall_at_default"] == pytest.approx(1.0)
    assert artifact["expected_false_accept_rate_at_default"] == pytest.approx(3 / 296)
    assert "not a standalone correctness oracle" in artifact["deployment_boundary"]

    points = {point["policy_name"]: point for point in artifact["operating_points"]}
    assert set(points) == {"conservative", "balanced", "permissive"}
    assert points["conservative"]["threshold"] == pytest.approx(1.0)
    assert points["conservative"]["expected_false_accept_rate"] == pytest.approx(3 / 296)
    assert points["conservative"]["recommended_use"] == "automated_candidate_filtering"
    assert points["balanced"]["threshold"] == pytest.approx(0.5)
    assert points["balanced"]["expected_false_accept_rate"] == pytest.approx(9 / 296)
    assert points["balanced"]["recommended_use"] == "repair_queue_triage"
    assert points["permissive"]["threshold"] == pytest.approx(0.25)
    assert points["permissive"]["expected_false_accept_rate"] == pytest.approx(131 / 296)
    assert points["permissive"]["recommended_use"] == "diagnostic_review_only"

    source_by_id = {source["experiment_id"]: source for source in artifact["source_artifacts"]}
    assert source_by_id["exp2940"]["sha256"] == _sha256(tmp_path / exp.EXP2940_REL_PATH)
    assert source_by_id["exp2943"]["sha256"] == _sha256(tmp_path / exp.EXP2943_REL_PATH)


def test_scenario_code_2953_blocks_when_score_distribution_is_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2953-BLOCKED: missing score distribution stays partial."""

    _write_exp2940(tmp_path, include_distribution=False)
    _write_exp2943(tmp_path)

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["honest_verdict"] == "blocked_missing_exp2940_score_distribution"
    assert artifact["threshold_policy_ready"] is False
    assert artifact["missing_score_distribution"] is True
    assert artifact["operating_points"] == []
    assert artifact["selected_default_threshold"] is None
    assert artifact["missing_fields"] == ["code_status_energy_values"]
    assert "verifier_ensemble_auprc_code_corpora_2940" in artifact["next_command"]
    assert artifact["expected_false_accept_rate_at_default"] is None


def test_req_code_2953_requires_both_source_artifacts(tmp_path: Path) -> None:
    """REQ-CODE-2953: Exp 2943 is required for deployment boundaries."""

    _write_exp2940(tmp_path)

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert artifact["threshold_policy_ready"] is False
    assert artifact["missing_fields"] == ["source:exp2943"]
    assert artifact["missing_score_distribution"] is False
    assert artifact["next_command"] is None


def test_req_code_2953_rejects_invalid_threshold_inputs() -> None:
    """REQ-CODE-2953: invalid Exp 2940 summaries do not produce scores."""

    assert exp._baseline_auprc({"code_corpus_baseline_random_auprc": 0.125}) == pytest.approx(
        0.125
    )
    with pytest.raises(ValueError, match="both positive and negative"):
        exp._curve_points([], candidate_count=0, positive_count=0)
    with pytest.raises(ValueError, match="expected numeric"):
        exp._number("not-a-number")
