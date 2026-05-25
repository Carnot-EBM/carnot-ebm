"""Tests for Exp 3045 FR-11 governed self-learning boundary.

Spec refs: REQ-LEARN-3045, SCENARIO-LEARN-3045,
SCENARIO-LEARN-3045-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_governed_self_learning_boundary_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3045_fr11_governed_self_learning_boundary_v1.py"
EXP3032_SOURCE = REPO_ROOT / "results" / "experiment_3032_fr11_heldout_dvi_replay_v2.json"
EXP3033_SOURCE = (
    REPO_ROOT / "results" / "experiment_3033_fr11_nonforgetting_negative_control_stress_v1.json"
)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        exp3032_artifact_path=tmp_path / exp.EXP3032_ARTIFACT_REL_PATH,
        exp3033_artifact_path=tmp_path / exp.EXP3033_ARTIFACT_REL_PATH,
        started_at=100.0,
        clock=lambda: 102.5,
        tests_run=("focused-req-3045",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for source, rel_path in (
        (EXP3032_SOURCE, exp.EXP3032_ARTIFACT_REL_PATH),
        (EXP3033_SOURCE, exp.EXP3033_ARTIFACT_REL_PATH),
    ):
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _names(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["name"]) for row in rows}


def test_req_learn_3045_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3045: governance artifact is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3045" in spec
    assert "SCENARIO-LEARN-3045" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "fr11_governance_ready" in spec
    assert "contradiction graph size/rate" in spec
    assert "model weights marked out of scope" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3045_writes_complete_governance_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3045: terminal sources produce a complete gate protocol."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fr11_governance_ready"] is True
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["focused-req-3045"]

    substrate = artifact["inference_substrate"]
    assert substrate["cached_artifacts_only"] is True
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["mode"] == "cached_artifact_governance_aggregation"

    summary = artifact["prior_evidence_summary"]
    assert summary["exp3032"]["ready"] is True
    assert summary["exp3032"]["evidence_type"] == "heldout_cached_exact_trace_replay"
    assert summary["exp3032"]["limits"]["model_weight_training"] is False
    assert summary["exp3033"]["promotable"] is True
    assert summary["exp3033"]["evidence_type"] == "controller_only_nonforgetting_stress"
    assert summary["exp3033"]["limits"]["model_weight_training"] is False
    assert "controller-only" in summary["limits"]

    edit_targets = {row["name"]: row for row in artifact["allowed_edit_targets"]}
    assert set(edit_targets) == {
        "controller_weights",
        "trace_memory",
        "validator_thresholds",
        "kan_locality_anchors",
        "model_weights",
    }
    assert edit_targets["model_weights"]["scope"] == "out_of_scope"
    assert edit_targets["model_weights"]["requires_actual_training_experiment"] is True
    assert edit_targets["controller_weights"]["scope"] == "allowed_controller_side"

    assert _names(artifact["required_metrics"]) == {
        "family_holdout_delta",
        "prior_retention_delta",
        "no_feedback_delta",
        "shuffled_control_delta",
        "contradiction_graph_size_rate",
        "rollback_count",
        "delayed_regression_delta",
        "source_trace_completeness",
    }
    assert all(row["required_for_exp3046"] is True for row in artifact["required_metrics"])
    assert _names(artifact["non_promotion_criteria"]) == {
        "tautology",
        "self_confirming_labels",
        "family_leakage",
        "missing_negative_controls",
    }
    assert any("model-weight learning" in claim for claim in artifact["forbidden_claims"])
    assert "controller-only" in artifact["continuous_self_learning_scope"]

    exp.validate_artifact(artifact)


def test_scenario_learn_3045_blocked_without_prior_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3045-BLOCKED: missing evidence fails closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["fr11_governance_ready"] is False
    assert artifact["blocked_reason"] == "exp3032_artifact_missing_or_empty"
    assert artifact["prior_evidence_summary"]["exp3032"]["ready"] is False
    assert artifact["prior_evidence_summary"]["exp3033"]["ready"] is False
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_training"] is False
    assert artifact["honest_verdict"].startswith("blocked_")
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    exp.validate_artifact(artifact)


def test_req_learn_3045_precondition_blockers_are_explicit() -> None:
    """REQ-LEARN-3045-1: malformed or non-terminal inputs block by name."""

    exp3032 = {
        "fr11_heldout_replay_ready": True,
        "honest_verdict": "complete_fr11_heldout_replay_ready",
        "inference_substrate": {
            "live_llm_inference": False,
            "model_weight_training": False,
        },
    }
    exp3033 = {
        "fr11_nonforgetting_stress_ready": True,
        "fr11_self_learning_promotable": True,
        "honest_verdict": "complete_controller_only_promotable",
        "inference_substrate": {
            "live_llm_inference": False,
            "model_weight_training": False,
        },
    }

    assert exp.precondition_blocker(exp3032, exp3033) is None
    cases = [
        ({}, exp3033, "exp3032_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3033, "exp3032_artifact_malformed"),
        (exp3032 | {"honest_verdict": "blocked_missing"}, exp3033, "exp3032_not_terminal"),
        (exp3032 | {"fr11_heldout_replay_ready": False}, exp3033, "exp3032_not_ready"),
        (
            exp3032
            | {"inference_substrate": {"live_llm_inference": True, "model_weight_training": False}},
            exp3033,
            "exp3032_live_llm_inference_claimed",
        ),
        (
            exp3032
            | {"inference_substrate": {"live_llm_inference": False, "model_weight_training": True}},
            exp3033,
            "exp3032_model_weight_training_claimed",
        ),
        (
            exp3032 | {"inference_substrate": "missing"},
            exp3033,
            "exp3032_inference_substrate_missing",
        ),
        (exp3032, {}, "exp3033_artifact_missing_or_empty"),
        (exp3032, {"_malformed": True}, "exp3033_artifact_malformed"),
        (exp3032, exp3033 | {"honest_verdict": "blocked_missing"}, "exp3033_not_terminal"),
        (
            exp3032,
            exp3033 | {"fr11_nonforgetting_stress_ready": False},
            "exp3033_not_ready",
        ),
        (
            exp3032,
            exp3033 | {"fr11_self_learning_promotable": False},
            "exp3033_not_controller_promotable",
        ),
        (
            exp3032,
            exp3033
            | {"inference_substrate": {"live_llm_inference": True, "model_weight_training": False}},
            "exp3033_live_llm_inference_claimed",
        ),
        (
            exp3032,
            exp3033
            | {"inference_substrate": {"live_llm_inference": False, "model_weight_training": True}},
            "exp3033_model_weight_training_claimed",
        ),
    ]
    for source_3032, source_3033, expected in cases:
        assert exp.precondition_blocker(source_3032, source_3033) == expected


def test_req_learn_3045_validation_rejects_incomplete_protocol(tmp_path: Path) -> None:
    """REQ-LEARN-3045-6: readiness requires every governance gate."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete_incomplete"})
    with pytest.raises(ValueError, match="allowed_edit_targets"):
        exp.validate_artifact(artifact | {"allowed_edit_targets": "not-a-list"})
    with pytest.raises(ValueError, match="allowed_edit_targets"):
        exp.validate_artifact(artifact | {"allowed_edit_targets": [{"scope": "missing-name"}]})
    with pytest.raises(ValueError, match="allowed_edit_targets"):
        exp.validate_artifact(
            artifact | {"allowed_edit_targets": artifact["allowed_edit_targets"][:-1]}
        )
    with pytest.raises(ValueError, match="model_weights"):
        bad_targets = [dict(row) for row in artifact["allowed_edit_targets"]]
        bad_targets[-1]["scope"] = "allowed"
        exp.validate_artifact(artifact | {"allowed_edit_targets": bad_targets})
    with pytest.raises(ValueError, match="model_weights"):
        bad_targets = [dict(row) for row in artifact["allowed_edit_targets"]]
        bad_targets[-1]["requires_actual_training_experiment"] = False
        exp.validate_artifact(artifact | {"allowed_edit_targets": bad_targets})
    with pytest.raises(ValueError, match="required_metrics"):
        exp.validate_artifact(artifact | {"required_metrics": artifact["required_metrics"][:-1]})
    with pytest.raises(ValueError, match="required_metrics"):
        bad_metrics = [dict(row) for row in artifact["required_metrics"]]
        bad_metrics[0]["required_for_exp3046"] = False
        exp.validate_artifact(artifact | {"required_metrics": bad_metrics})
    with pytest.raises(ValueError, match="non_promotion_criteria"):
        exp.validate_artifact(
            artifact | {"non_promotion_criteria": artifact["non_promotion_criteria"][:-1]}
        )
    with pytest.raises(ValueError, match="non_promotion_criteria"):
        bad_criteria = [dict(row) for row in artifact["non_promotion_criteria"]]
        bad_criteria[0]["automatic"] = False
        exp.validate_artifact(artifact | {"non_promotion_criteria": bad_criteria})
    with pytest.raises(ValueError, match="forbidden_claims"):
        exp.validate_artifact(artifact | {"forbidden_claims": []})
    with pytest.raises(ValueError, match="forbidden_claims"):
        exp.validate_artifact(
            artifact
            | {
                "forbidden_claims": [
                    "live inference",
                    "broad learning",
                    "KAN retraining",
                    "promotion without gates",
                ]
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": "cached"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"cached_artifacts_only": False}
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_training": True}
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_mutation": True}
            }
        )
    with pytest.raises(ValueError, match="prior_evidence_summary"):
        exp.validate_artifact(artifact | {"prior_evidence_summary": {"limits": "missing sources"}})
    with pytest.raises(ValueError, match="prior_evidence_summary"):
        bad_summary = dict(artifact["prior_evidence_summary"])
        bad_summary["exp3032"] = bad_summary["exp3032"] | {"ready": False}
        exp.validate_artifact(artifact | {"prior_evidence_summary": bad_summary})
    with pytest.raises(ValueError, match="continuous_self_learning_scope"):
        exp.validate_artifact(
            artifact | {"continuous_self_learning_scope": "broad native learning"}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready_wrong_prefix"})
    with pytest.raises(ValueError, match="blocked_ prefix"):
        exp.validate_artifact(
            artifact
            | {
                "fr11_governance_ready": False,
                "honest_verdict": "complete_wrong_for_blocked",
            }
        )


def test_scenario_learn_3045_malformed_source_file_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3045-BLOCKED: malformed JSON evidence is not promoted."""

    cfg = _config(tmp_path)
    cfg.resolved_exp3032_artifact_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved_exp3032_artifact_path().write_text("{not json", encoding="utf-8")
    artifact = exp.run_experiment(cfg)

    assert artifact["fr11_governance_ready"] is False
    assert artifact["blocked_reason"] == "exp3032_artifact_malformed"
    assert artifact["honest_verdict"].startswith("blocked_")
    exp.validate_artifact(artifact)
