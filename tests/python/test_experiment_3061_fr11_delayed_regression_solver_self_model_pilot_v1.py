"""Tests for Exp 3061 delayed-regression solver self-model pilot.

Spec refs: REQ-LEARN-3061, SCENARIO-LEARN-3061,
SCENARIO-LEARN-3061-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_delayed_regression_solver_self_model_pilot_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1.py"
)
SOURCE_FILES = (exp.EXP3060_ARTIFACT_REL_PATH, exp.EXP3058_ARTIFACT_REL_PATH)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        exp3060_artifact_path=tmp_path / exp.EXP3060_ARTIFACT_REL_PATH,
        exp3058_artifact_path=tmp_path / exp.EXP3058_ARTIFACT_REL_PATH,
        started_at=100.0,
        clock=lambda: 102.75,
        tests_run=("focused-req-3061",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in SOURCE_FILES:
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3061_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3061: delayed-regression pilot is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3061" in spec
    assert "SCENARIO-LEARN-3061" in spec
    assert "SCENARIO-LEARN-3061-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "fr11_delayed_regression_ready" in spec
    assert "self_model_trace_count" in spec
    assert "blocked_missing_trace_schema_or_formal_fallback" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3061_writes_ready_delayed_regression_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3061: exact feedback stores traces and measures delay."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fr11_delayed_regression_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "controller_only_delayed_regression_ready"
    assert artifact["edit_targets_used"] == [
        "controller_weights",
        "trace_memory",
        "rollback_policy",
    ]
    assert artifact["self_model_trace_count"] == 3
    assert artifact["family_holdout_delta"] == pytest.approx(0.4)
    assert artifact["prior_retention_delta"] == pytest.approx(0.0)
    assert artifact["no_feedback_delta"] == pytest.approx(0.0)
    assert artifact["shuffled_control_delta"] == pytest.approx(-0.4)
    assert artifact["contradiction_rate_delta"] == pytest.approx(-0.5)
    assert artifact["rollback_count"] == 1
    assert artifact["delayed_regression_delta"] == pytest.approx(0.4)
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(2.75)
    assert artifact["tests_run"] == ["focused-req-3061"]

    counts = artifact["source_trace_counts"]
    assert counts["exp3060_trace_schema_field_count"] == len(exp.REQUIRED_TRACE_FIELDS)
    assert counts["exp3058_fixture_count"] >= 1
    assert counts["train_update_case_count"] == 2
    assert counts["family_holdout_case_count"] == 4
    assert counts["prior_case_count"] == 2
    assert counts["delayed_regression_case_count"] == 4
    assert counts["no_feedback_control_count"] == 4
    assert counts["shuffled_control_count"] == 4
    assert counts["self_model_trace_count"] == 3
    assert counts["rolled_back_trace_count"] == 1

    traces = artifact["self_model_traces"]
    assert len(traces) == artifact["self_model_trace_count"]
    assert {row["trace_id"] for row in traces} == {
        "exp3061-trace-0001",
        "exp3061-trace-0002",
        "exp3061-trace-0003",
    }
    assert exp.trace_rows_are_schema_populated(traces, artifact["trace_schema"])
    assert all(row["controller_edit"]["model_weight_mutation"] is False for row in traces)
    assert all(row["correction_set"]["independent_label_authority"] for row in traces)
    assert traces[-1]["rollback_decision"]["rolled_back"] is True
    assert traces[-1]["controller_edit"]["target"] == "rollback_policy"
    assert traces[-1]["source_artifact"]["source_experiment_id"] == "exp3058"

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "deterministic_exact_solver_self_model_trace_pilot"
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["controller_weight_update"] is True
    assert substrate["trace_memory_update"] is True

    assert artifact["split_report"]["leakage_detected"] is False
    assert artifact["control_report"]["non_vacuous_controls"] is True
    assert artifact["control_report"]["no_feedback_case_count"] == 4
    assert artifact["control_report"]["shuffled_case_count"] == 4
    assert artifact["control_report"]["shuffled_candidate_rolled_back"] is True
    exp.validate_artifact(artifact)


def test_req_learn_3061_split_controls_and_traces_are_auditable(tmp_path: Path) -> None:
    """REQ-LEARN-3061-2/3/4: split, controls, and traces are source-traced."""

    _copy_sources(tmp_path)
    config = _config(tmp_path)
    sources = exp.load_source_bundle(config)
    split = exp.build_family_split(sources)
    baseline = exp.initial_controller_state()
    result = exp.run_self_model_pilot(split, sources, config)

    assert exp.precondition_blocker(sources) is None
    assert {case.case_id for case in split.train_update}.isdisjoint(
        {case.case_id for case in split.family_holdout}
    )
    assert split.prior_cases
    assert split.delayed_regression
    assert split.no_feedback_controls
    assert split.shuffled_feedback_controls
    assert result.edit_targets_used == ("controller_weights", "trace_memory", "rollback_policy")
    assert result.metrics["family_holdout_delta"] > 0.0
    assert result.metrics["prior_retention_delta"] == 0.0
    assert result.metrics["no_feedback_delta"] == 0.0
    assert result.metrics["shuffled_control_delta"] <= 0.0
    assert result.metrics["contradiction_rate_delta"] < 0.0
    assert result.metrics["delayed_regression_delta"] >= 0.0
    assert exp.retention_score(baseline.weights, split.prior_cases) == 1.0
    assert exp.retention_score({}, ()) == 0.0
    assert exp.mean_signed_margin({}, ()) == 0.0
    assert exp.contradiction_rate({}, ()) == 0.0
    assert exp.trace_rows_are_schema_populated(result.self_model_traces, sources.exp3060_artifact)
    assert exp._relative_to(tmp_path, tmp_path / "results" / "x.json") == Path("results/x.json")
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
    assert exp._round(1.2345678) == pytest.approx(1.234568)


def test_scenario_learn_3061_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3061-BLOCKED: missing sources fail closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["fr11_delayed_regression_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["edit_targets_used"] == []
    assert artifact["self_model_trace_count"] == 0
    assert artifact["family_holdout_delta"] == 0.0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["no_feedback_delta"] == 0.0
    assert artifact["shuffled_control_delta"] == 0.0
    assert artifact["contradiction_rate_delta"] == 0.0
    assert artifact["rollback_count"] == 0
    assert artifact["delayed_regression_delta"] == 0.0
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["blocked_reason"] == "exp3060_artifact_missing_or_empty"
    assert artifact["self_model_traces"] == []
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    exp.validate_artifact(artifact)


def test_req_learn_3061_precondition_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3061-1: trace-schema and formal-fallback blockers are explicit."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    exp3060_ready = dict(sources.exp3060_artifact)
    exp3058_ready = dict(sources.exp3058_artifact)

    assert exp.precondition_blocker(sources) is None

    cases = [
        ({}, exp3058_ready, "exp3060_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3058_ready, "exp3060_artifact_malformed"),
        (exp3060_ready | {"honest_verdict": "waiting"}, exp3058_ready, "exp3060_not_terminal"),
        (
            exp3060_ready | {"solver_self_model_trace_ready": False},
            exp3058_ready,
            "exp3060_trace_schema_not_ready",
        ),
        (
            exp3060_ready | {"trace_schema": {"fields": []}},
            exp3058_ready,
            "exp3060_trace_schema_missing_fields",
        ),
        (
            exp3060_ready
            | {
                "allowed_edit_targets": [{"name": "model_weights", "scope": "controller_side_only"}]
            },
            exp3058_ready,
            "exp3060_allowed_model_weight_target",
        ),
        (
            exp3060_ready
            | {
                "inference_substrate": {
                    "model_weight_training": False,
                    "model_weight_mutation": True,
                }
            },
            exp3058_ready,
            "exp3060_model_weight_learning_claimed",
        ),
        (exp3060_ready, {}, "exp3058_artifact_missing_or_empty"),
        (exp3060_ready, {"_malformed": True}, "exp3058_artifact_malformed"),
        (exp3060_ready, exp3058_ready | {"honest_verdict": "waiting"}, "exp3058_not_terminal"),
        (
            exp3060_ready | {"honest_verdict": "success_schema"},
            exp3058_ready | {"llm_guided_smt_pilot_ready": False},
            "exp3058_formal_fallback_not_ready",
        ),
        (
            exp3060_ready,
            exp3058_ready | {"formal_fallback_preserved": False},
            "exp3058_formal_fallback_not_preserved",
        ),
        (exp3060_ready, exp3058_ready | {"exact_solver_path": ""}, "exp3058_exact_solver_missing"),
        (exp3060_ready, exp3058_ready | {"fixture_count": 0}, "exp3058_fixture_count_missing"),
        (
            exp3060_ready,
            exp3058_ready
            | {
                "inference_substrate": {
                    "model_weight_training": True,
                    "model_weight_mutation": False,
                }
            },
            "exp3058_model_weight_learning_claimed",
        ),
    ]
    for exp3060_artifact, exp3058_artifact, expected in cases:
        assert (
            exp.precondition_blocker(exp.SourceBundle(exp3060_artifact, exp3058_artifact))
            == expected
        )


def test_req_learn_3061_validation_rejects_invalid_ready_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3061-5: readiness gates reject missing traces and bad claims."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"continuous_self_learning_task": False}, "continuous_self_learning_task"),
        (artifact | {"source_trace_counts": {}}, "source_trace_counts"),
        (artifact | {"inference_substrate": "bad"}, "inference_substrate"),
        (
            artifact | {"fr11_delayed_regression_ready": False, "honest_verdict": "waiting"},
            "blocked artifacts",
        ),
        (artifact | {"honest_verdict": "waiting"}, "honest_verdict"),
        (artifact | {"edit_targets_used": ["controller_weights"]}, "edit_targets_used"),
        (artifact | {"self_model_trace_count": 0}, "self_model_trace_count"),
        (artifact | {"self_model_traces": []}, "self_model_traces"),
        (
            artifact
            | {
                "inference_substrate": {
                    "live_llm_inference": True,
                    "model_weight_training": False,
                    "model_weight_mutation": False,
                }
            },
            "live LLM inference",
        ),
        (
            artifact
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": False,
                    "model_weight_mutation": True,
                }
            },
            "model weights",
        ),
        (artifact | {"family_holdout_delta": 0.0}, "family_holdout_delta"),
        (artifact | {"prior_retention_delta": -0.1}, "prior_retention_delta"),
        (artifact | {"no_feedback_delta": 0.1}, "no_feedback_delta"),
        (artifact | {"shuffled_control_delta": 0.1}, "shuffled_control_delta"),
        (artifact | {"contradiction_rate_delta": 0.0}, "contradiction_rate_delta"),
        (artifact | {"delayed_regression_delta": -0.1}, "delayed_regression_delta"),
        (
            artifact
            | {
                "source_trace_counts": artifact["source_trace_counts"]
                | {"exp3058_fixture_count": 0}
            },
            "source_trace_counts must be positive",
        ),
        (
            artifact
            | {"control_report": artifact["control_report"] | {"non_vacuous_controls": False}},
            "control_report",
        ),
        (
            artifact | {"split_report": artifact["split_report"] | {"leakage_detected": True}},
            "split_report",
        ),
    ]
    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(bad_artifact)

    assert (
        exp.trace_rows_are_schema_populated(artifact["self_model_traces"], {"fields": []}) is False
    )
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._read_json(malformed) == {"_malformed": True}
    assert exp._file_sha256(tmp_path / "missing.json") == ""
