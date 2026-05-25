"""Tests for Exp 3077 FR-11 soundness-bounded online self-learning pilot.

Spec refs: REQ-LEARN-3077, SCENARIO-LEARN-3077,
SCENARIO-LEARN-3077-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_soundness_bounded_online_self_learning_pilot_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3077_fr11_soundness_bounded_online_self_learning_pilot_v1.py"
)
SOURCE_FILES = (exp.EXP3076_ARTIFACT_REL_PATH, exp.EXP3060_ARTIFACT_REL_PATH)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        exp3076_artifact_path=tmp_path / exp.EXP3076_ARTIFACT_REL_PATH,
        exp3060_artifact_path=tmp_path / exp.EXP3060_ARTIFACT_REL_PATH,
        started_at=300.0,
        clock=lambda: 303.5,
        tests_run=("focused-req-3077",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in SOURCE_FILES:
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3077_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3077: pilot is OpenSpec anchored and has a CLI wrapper."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3077" in spec
    assert "SCENARIO-LEARN-3077" in spec
    assert "SCENARIO-LEARN-3077-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "fr11_soundness_bounded_ready" in spec
    assert "soundness_mistakes" in spec
    assert "completeness_mistakes" in spec
    assert "blocked_missing_soundness_budget" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3077_writes_bounded_budget_exceeded_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3077: completeness budget breach refuses promotion."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fr11_soundness_bounded_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == (
        "controller_only_budget_exceeded_no_stronger_promotion"
    )
    assert artifact["honest_verdict"] == "complete_fr11_soundness_bounded_budget_exceeded"
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["tests_run"] == ["focused-req-3077"]

    assert artifact["edit_targets_used"] == [
        "controller_weights",
        "trace_memory",
        "rollback_policy",
    ]
    assert artifact["self_model_trace_count"] == 20
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 1
    assert artifact["family_holdout_delta"] == pytest.approx(1.0)
    assert artifact["prior_retention_delta"] == pytest.approx(0.0)
    assert artifact["no_feedback_delta"] == pytest.approx(0.0)
    assert artifact["shuffled_control_delta"] == pytest.approx(0.0)
    assert artifact["contradiction_rate_delta"] == pytest.approx(-1.0)
    assert artifact["rollback_count"] == 1
    assert artifact["delayed_regression_delta"] == pytest.approx(0.666667)

    budget = artifact["mistake_budget_delta"]
    assert budget["soundness_mistakes"]["passed"] is True
    assert budget["soundness_mistakes"]["delta"] == 0
    assert budget["completeness_mistakes"]["passed"] is False
    assert budget["completeness_mistakes"]["observed"] == 1
    assert budget["completeness_mistakes"]["allowed"] == 0
    assert budget["completeness_mistakes"]["delta"] == 1
    assert budget["no_feedback_delta"]["passed"] is True
    assert budget["shuffled_control_delta"]["passed"] is True
    assert budget["prior_retention_score"]["passed"] is True
    assert budget["controls_non_vacuous"]["passed"] is True
    assert budget["all_gates_passed"] is False

    counts = artifact["source_trace_counts"]
    assert counts == {
        "exp3076_budget_gate_count": 7,
        "exp3060_trace_schema_field_count": len(exp.REQUIRED_TRACE_FIELDS),
        "train_update_case_count": 2,
        "family_holdout_case_count": 4,
        "prior_case_count": 2,
        "delayed_regression_case_count": 3,
        "no_feedback_control_count": 4,
        "shuffled_control_count": 4,
        "online_decision_count": 19,
        "self_model_trace_count": 20,
        "rolled_back_trace_count": 1,
    }

    traces = artifact["self_model_traces"]
    assert len(traces) == artifact["self_model_trace_count"]
    assert exp.trace_rows_are_schema_populated(traces, artifact["trace_schema"])
    assert all(row["controller_edit"]["model_weight_mutation"] is False for row in traces)
    assert all(row["correction_set"]["independent_label_authority"] for row in traces)
    assert traces[-1]["online_decision_label"] == "rollback"
    assert traces[-1]["rollback_decision"]["rolled_back"] is True
    assert "completeness_mistake" in {row["online_decision_label"] for row in traces}
    assert "soundness_mistake" in {row["online_decision_label"] for row in traces}

    online_labels = {row["decision_label"] for row in artifact["online_decisions"]}
    assert online_labels <= exp.ALLOWED_DECISION_LABELS
    assert artifact["decision_label_counts"]["main"]["completeness_mistake"] == 1
    assert artifact["decision_label_counts"]["main"]["soundness_mistake"] == 0
    assert artifact["control_report"]["non_vacuous_controls"] is True
    assert artifact["control_report"]["shuffled_candidate_rolled_back"] is True
    assert artifact["split_report"]["leakage_detected"] is False

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "deterministic_exact_controller_online_budget_pilot"
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["base_model_weights_updated"] is False
    assert substrate["controller_weight_update"] is True
    assert substrate["trace_memory_update"] is True
    exp.validate_artifact(artifact)


def test_req_learn_3077_split_decisions_and_budget_are_auditable(tmp_path: Path) -> None:
    """REQ-LEARN-3077-2/3/4/5: split, decisions, and gates are auditable."""

    _copy_sources(tmp_path)
    config = _config(tmp_path)
    sources = exp.load_source_bundle(config)
    split = exp.build_family_split()
    result = exp.run_online_pilot(split, sources, config)

    assert exp.precondition_blocker(sources) is None
    assert exp.budget_is_complete(sources.exp3076_artifact)
    assert {case.case_id for case in split.train_update}.isdisjoint(
        {case.case_id for case in split.family_holdout}
    )
    assert split.prior_cases
    assert split.delayed_regression
    assert split.no_feedback_controls
    assert split.shuffled_feedback_controls
    assert result.metrics["family_holdout_delta"] > 0.0
    assert result.metrics["prior_retention_delta"] == 0.0
    assert result.metrics["no_feedback_delta"] == 0.0
    assert result.metrics["shuffled_control_delta"] == 0.0
    assert result.metrics["contradiction_rate_delta"] < 0.0
    assert result.metrics["delayed_regression_delta"] > 0.0
    assert result.soundness_mistakes == 0
    assert result.completeness_mistakes == 1
    assert result.mistake_budget_delta["all_gates_passed"] is False
    assert result.mistake_budget_delta["completeness_mistakes"]["passed"] is False
    assert exp.trace_rows_are_schema_populated(result.self_model_traces, sources.exp3060_artifact)

    baseline = exp.initial_controller_state()
    updated = exp.apply_feedback_updates(baseline, split.train_update)
    assert exp.accuracy(baseline.weights, split.family_holdout) == 0.0
    assert exp.accuracy(updated.weights, split.family_holdout) == 1.0
    assert exp.retention_score(baseline.weights, split.prior_cases) == 1.0
    assert exp.accuracy({}, ()) == 0.0
    assert exp.contradiction_rate({}, ()) == 0.0
    regression_case = exp._prior_case("prior-regresses", True, "prior::consistent")
    assert (
        exp._delayed_regression_count(
            {"prior::consistent": 0.6},
            {"prior::consistent": 0.0},
            (regression_case,),
        )
        == 1
    )
    assert exp._relative_to(tmp_path, tmp_path / "results" / "x.json") == Path("results/x.json")
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
    assert exp._round(1.2345678) == pytest.approx(1.234568)


def test_scenario_learn_3077_blocked_without_soundness_budget(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3077-BLOCKED: missing Exp 3076 budget fails closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["fr11_soundness_bounded_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["edit_targets_used"] == []
    assert artifact["self_model_trace_count"] == 0
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["family_holdout_delta"] == 0.0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["no_feedback_delta"] == 0.0
    assert artifact["shuffled_control_delta"] == 0.0
    assert artifact["contradiction_rate_delta"] == 0.0
    assert artifact["rollback_count"] == 0
    assert artifact["delayed_regression_delta"] == 0.0
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["blocked_reason"] == "exp3076_artifact_missing_or_empty"
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    exp.validate_artifact(artifact)


def test_req_learn_3077_precondition_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3077-1: budget blockers name unsafe prior evidence."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    exp3076_ready = dict(sources.exp3076_artifact)
    exp3060_ready = dict(sources.exp3060_artifact)

    assert exp.precondition_blocker(sources) is None

    cases = [
        ({}, exp3060_ready, "exp3076_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3060_ready, "exp3076_artifact_malformed"),
        (exp3076_ready | {"honest_verdict": "waiting"}, exp3060_ready, "exp3076_not_terminal"),
        (
            exp3076_ready | {"soundness_completeness_budget_ready": False},
            exp3060_ready,
            "exp3076_soundness_budget_not_ready",
        ),
        (exp3076_ready | {"mistake_budget": {}}, exp3060_ready, "exp3076_budget_incomplete"),
        (
            exp3076_ready | {"required_controls": []},
            exp3060_ready,
            "exp3076_required_controls_incomplete",
        ),
        (
            exp3076_ready | {"inference_substrate": {"live_llm_inference": True}},
            exp3060_ready,
            "exp3076_live_model_inference_claimed",
        ),
        (
            exp3076_ready | {"inference_substrate": {"model_weight_training": True}},
            exp3060_ready,
            "exp3076_model_weight_training_claimed",
        ),
        (
            exp3076_ready | {"inference_substrate": {"model_weight_mutation": True}},
            exp3060_ready,
            "exp3076_model_weight_mutation_claimed",
        ),
        (exp3076_ready, {}, "exp3060_artifact_missing_or_empty"),
        (exp3076_ready, {"_malformed": True}, "exp3060_artifact_malformed"),
        (exp3076_ready, exp3060_ready | {"honest_verdict": "waiting"}, "exp3060_not_terminal"),
        (
            exp3076_ready,
            exp3060_ready | {"solver_self_model_trace_ready": False},
            "exp3060_trace_schema_not_ready",
        ),
        (
            exp3076_ready,
            exp3060_ready | {"trace_schema": {"fields": []}},
            "exp3060_trace_schema_missing_fields",
        ),
        (
            exp3076_ready,
            exp3060_ready | {"inference_substrate": {"model_weight_mutation": True}},
            "exp3060_model_weight_learning_claimed",
        ),
    ]
    for exp3076_artifact, exp3060_artifact, expected in cases:
        assert (
            exp.precondition_blocker(exp.SourceBundle(exp3076_artifact, exp3060_artifact))
            == expected
        )


def test_req_learn_3077_validation_rejects_invalid_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3077-6/7: readiness and promotion gates are enforced."""

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
            artifact | {"fr11_soundness_bounded_ready": False, "honest_verdict": "waiting"},
            "blocked artifacts",
        ),
        (artifact | {"honest_verdict": "waiting"}, "honest_verdict"),
        (artifact | {"edit_targets_used": ["controller_weights"]}, "edit_targets_used"),
        (artifact | {"self_model_trace_count": 0}, "self_model_trace_count"),
        (artifact | {"self_model_traces": []}, "self_model_traces"),
        (artifact | {"soundness_mistakes": -1}, "mistake counts"),
        (
            artifact
            | {
                "inference_substrate": {
                    "live_llm_inference": True,
                    "live_model_inference": False,
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
                    "live_model_inference": True,
                    "model_weight_training": False,
                    "model_weight_mutation": False,
                }
            },
            "live model inference",
        ),
        (
            artifact
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "live_model_inference": False,
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
            artifact | {"mistake_budget_delta": "bad"},
            "mistake_budget_delta",
        ),
        (
            artifact
            | {
                "mistake_budget_delta": artifact["mistake_budget_delta"]
                | {"all_gates_passed": True}
            },
            "promotion_decision",
        ),
        (
            artifact | {"promotion_decision": "controller_only_soundness_bounded_pilot_ready"},
            "promotion_decision",
        ),
        (
            artifact
            | {
                "source_trace_counts": artifact["source_trace_counts"]
                | {"exp3076_budget_gate_count": 0}
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
