"""Tests for Exp 3046 governed solver-feedback self-learning loop.

Spec refs: REQ-LEARN-3046, SCENARIO-LEARN-3046,
SCENARIO-LEARN-3046-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_solver_feedback_self_learning_loop_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "experiment_3046_fr11_solver_feedback_self_learning_loop_v1.py"
)
EXP3044_SOURCE = (
    REPO_ROOT / "results" / "experiment_3044_smt_sat_validator_tree_exactness_upgrade_v1.json"
)
EXP3045_SOURCE = (
    REPO_ROOT / "results" / "experiment_3045_fr11_governed_self_learning_boundary_v1.json"
)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        loop_report_path=tmp_path / exp.LOOP_REPORT_REL_PATH,
        exp3044_artifact_path=tmp_path / exp.EXP3044_ARTIFACT_REL_PATH,
        exp3045_artifact_path=tmp_path / exp.EXP3045_ARTIFACT_REL_PATH,
        started_at=200.0,
        clock=lambda: 203.25,
        tests_run=("focused-req-3046",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for source, rel_path in (
        (EXP3044_SOURCE, exp.EXP3044_ARTIFACT_REL_PATH),
        (EXP3045_SOURCE, exp.EXP3045_ARTIFACT_REL_PATH),
    ):
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_learn_3046_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3046: solver-feedback loop is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3046" in spec
    assert "SCENARIO-LEARN-3046" in spec
    assert "SCENARIO-LEARN-3046-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "fr11_solver_feedback_ready" in spec
    assert "contradiction_rate_delta" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3046_writes_complete_solver_feedback_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3046: exact correction feedback improves held-out family."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))
    report_rows = _jsonl_rows(tmp_path / exp.LOOP_REPORT_REL_PATH)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fr11_solver_feedback_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "controller_only_solver_feedback_ready"
    assert artifact["edit_targets_used"] == ["controller_weights", "trace_memory"]
    assert artifact["family_holdout_delta"] == pytest.approx(0.5)
    assert artifact["prior_retention_delta"] == pytest.approx(0.0)
    assert artifact["no_feedback_delta"] == pytest.approx(0.0)
    assert artifact["shuffled_control_delta"] == pytest.approx(-0.5)
    assert artifact["contradiction_rate_delta"] == pytest.approx(-0.5)
    assert artifact["rollback_count"] == 1
    assert artifact["delayed_regression_delta"] == pytest.approx(0.0)
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["tests_run"] == ["focused-req-3046"]

    traces = artifact["source_trace_counts"]
    assert traces["exp3044_correction_set_count"] == 1
    assert traces["train_update_case_count"] == 2
    assert traces["family_holdout_case_count"] == 4
    assert traces["prior_exact_case_count"] == 2
    assert traces["no_feedback_control_count"] == 4
    assert traces["shuffled_control_count"] == 4
    assert traces["source_traced_update_count"] == 2

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "deterministic_exact_solver_feedback_controller_loop"
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["controller_weight_update"] is True
    assert substrate["trace_memory_update"] is True

    assert artifact["split_report"]["leakage_detected"] is False
    assert artifact["control_report"]["non_vacuous_controls"] is True
    assert artifact["control_report"]["no_feedback_case_count"] == 4
    assert artifact["control_report"]["shuffled_case_count"] == 4
    assert {row["section"] for row in report_rows} == {
        "baseline",
        "updated",
        "controls",
        "rollback",
    }
    exp.validate_artifact(artifact)


def test_req_learn_3046_split_controls_and_update_are_source_traced(tmp_path: Path) -> None:
    """REQ-LEARN-3046-2/3/4: split IDs, controls, and update traces are auditable."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    split = exp.build_family_split(sources.exp3044_artifact)
    baseline = exp.initial_controller_state()
    result = exp.run_governed_loop(split, sources.exp3044_artifact, sources.exp3045_artifact)

    assert exp.precondition_blocker(sources) is None
    assert {case.case_id for case in split.train_update}.isdisjoint(
        {case.case_id for case in split.family_holdout}
    )
    assert split.no_feedback_controls
    assert split.shuffled_feedback_controls
    assert all(case.source_trace_id for case in split.train_update)
    assert result.edit_targets_used == ("controller_weights", "trace_memory")
    assert result.metrics["family_holdout_delta"] > 0.0
    assert result.metrics["no_feedback_delta"] == 0.0
    assert result.metrics["shuffled_control_delta"] <= 0.0
    assert result.metrics["contradiction_rate_delta"] < 0.0
    assert exp.retention_score(baseline.weights, split.prior_exact) == 1.0
    assert exp.retention_score({}, ()) == 0.0
    assert exp.mean_signed_margin({}, ()) == 0.0
    assert exp.contradiction_rate({}, ()) == 0.0


def test_scenario_learn_3046_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3046-BLOCKED: missing sources fail closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["fr11_solver_feedback_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["edit_targets_used"] == []
    assert artifact["family_holdout_delta"] == 0.0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["no_feedback_delta"] == 0.0
    assert artifact["shuffled_control_delta"] == 0.0
    assert artifact["contradiction_rate_delta"] == 0.0
    assert artifact["rollback_count"] == 0
    assert artifact["delayed_regression_delta"] == 0.0
    assert artifact["honest_verdict"] == "blocked_missing_governance_or_exact_feedback"
    assert artifact["source_trace_counts"]["exp3044_correction_set_count"] == 0
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    assert not (tmp_path / exp.LOOP_REPORT_REL_PATH).exists()
    exp.validate_artifact(artifact)


def test_req_learn_3046_precondition_blockers_are_explicit() -> None:
    """REQ-LEARN-3046-1: malformed governance or feedback blocks before update."""

    exp3044 = {
        "validator_tree_exactness_ready": True,
        "correction_sets": [{"candidate_fields": ["total"]}],
        "honest_verdict": "complete: exact",
        "inference_substrate": {
            "live_llm_inference": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
        },
    }
    exp3045 = {
        "fr11_governance_ready": True,
        "honest_verdict": "complete_governance",
        "allowed_edit_targets": [
            {"name": "controller_weights", "scope": "allowed_controller_side"},
            {"name": "trace_memory", "scope": "allowed_controller_side"},
            {"name": "model_weights", "scope": "out_of_scope"},
        ],
        "inference_substrate": {
            "live_llm_inference": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
        },
    }

    assert exp.precondition_blocker(exp.SourceBundle(exp3044, exp3045)) is None
    cases = [
        ({}, exp3045, "exp3044_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3045, "exp3044_artifact_malformed"),
        (exp3044 | {"honest_verdict": "blocked_missing"}, exp3045, "exp3044_not_terminal"),
        (
            exp3044 | {"validator_tree_exactness_ready": False},
            exp3045,
            "exp3044_exact_feedback_not_ready",
        ),
        (exp3044 | {"correction_sets": []}, exp3045, "exp3044_correction_sets_missing"),
        (
            exp3044
            | {
                "inference_substrate": {
                    "live_llm_inference": True,
                    "model_weight_training": False,
                    "model_weight_mutation": False,
                }
            },
            exp3045,
            "exp3044_live_llm_inference_claimed",
        ),
        (exp3044, {}, "exp3045_artifact_missing_or_empty"),
        (exp3044, {"_malformed": True}, "exp3045_artifact_malformed"),
        (exp3044, exp3045 | {"honest_verdict": "waiting"}, "exp3045_not_terminal"),
        (
            exp3044,
            exp3045 | {"fr11_governance_ready": False},
            "exp3045_governance_not_ready",
        ),
        (
            exp3044,
            exp3045 | {"allowed_edit_targets": [{"name": "model_weights"}]},
            "exp3045_controller_edit_target_missing",
        ),
        (
            exp3044,
            exp3045
            | {
                "allowed_edit_targets": [
                    {"name": "controller_weights", "scope": "allowed_controller_side"},
                    {"name": "model_weights", "scope": "out_of_scope"},
                ]
            },
            "exp3045_trace_memory_edit_target_missing",
        ),
        (
            exp3044,
            exp3045
            | {
                "allowed_edit_targets": [
                    {"name": "controller_weights", "scope": "allowed_controller_side"},
                    {"name": "trace_memory", "scope": "allowed_controller_side"},
                    {"name": "model_weights", "scope": "allowed_controller_side"},
                ]
            },
            "exp3045_model_weights_not_out_of_scope",
        ),
        (
            exp3044 | {"inference_substrate": "missing"},
            exp3045,
            "exp3044_inference_substrate_missing",
        ),
        (
            exp3044
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": False,
                    "model_weight_mutation": True,
                }
            },
            exp3045,
            "exp3044_model_weight_mutation_claimed",
        ),
        (
            exp3044,
            exp3045
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": True,
                    "model_weight_mutation": False,
                }
            },
            "exp3045_model_weight_training_claimed",
        ),
        (
            exp3044,
            exp3045
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": False,
                    "model_weight_mutation": True,
                }
            },
            "exp3045_model_weight_mutation_claimed",
        ),
    ]
    for source_3044, source_3045, expected in cases:
        assert exp.precondition_blocker(exp.SourceBundle(source_3044, source_3045)) == expected


def test_req_learn_3046_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3046-5: readiness requires every metric and substrate gate."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete_incomplete"})
    with pytest.raises(ValueError, match="terminal success prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        exp.validate_artifact(artifact | {"continuous_self_learning_task": False})
    with pytest.raises(ValueError, match="edit_targets_used"):
        exp.validate_artifact(artifact | {"edit_targets_used": ["model_weights"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": "cached"})
    with pytest.raises(ValueError, match="live LLM"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            }
        )
    with pytest.raises(ValueError, match="model weights"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_mutation": True}
            }
        )
    with pytest.raises(ValueError, match="family_holdout_delta"):
        exp.validate_artifact(artifact | {"family_holdout_delta": 0.0})
    with pytest.raises(ValueError, match="prior_retention_delta"):
        exp.validate_artifact(artifact | {"prior_retention_delta": -0.25})
    with pytest.raises(ValueError, match="no_feedback_delta"):
        exp.validate_artifact(artifact | {"no_feedback_delta": 0.25})
    with pytest.raises(ValueError, match="shuffled_control_delta"):
        exp.validate_artifact(artifact | {"shuffled_control_delta": 0.25})
    with pytest.raises(ValueError, match="contradiction_rate_delta"):
        exp.validate_artifact(artifact | {"contradiction_rate_delta": 0.0})
    with pytest.raises(ValueError, match="delayed_regression_delta"):
        exp.validate_artifact(artifact | {"delayed_regression_delta": -0.25})
    with pytest.raises(ValueError, match="source_trace_counts"):
        exp.validate_artifact(
            artifact | {"source_trace_counts": artifact["source_trace_counts"] | {"x": 0}}
        )
    with pytest.raises(ValueError, match="source_trace_counts"):
        bad_counts = dict(artifact["source_trace_counts"])
        bad_counts["source_traced_update_count"] = 0
        exp.validate_artifact(artifact | {"source_trace_counts": bad_counts})
    with pytest.raises(ValueError, match="control_report"):
        exp.validate_artifact(
            artifact
            | {"control_report": artifact["control_report"] | {"non_vacuous_controls": False}}
        )
    with pytest.raises(ValueError, match="split_report"):
        exp.validate_artifact(
            artifact | {"split_report": artifact["split_report"] | {"leakage_detected": True}}
        )
    with pytest.raises(ValueError, match="blocked or terminal verdict"):
        exp.validate_artifact(
            artifact
            | {
                "fr11_solver_feedback_ready": False,
                "honest_verdict": "waiting",
            }
        )

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp._read_json(bad_json) == {"_malformed": True}
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
