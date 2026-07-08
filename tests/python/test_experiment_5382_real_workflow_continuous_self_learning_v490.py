"""Tests for Exp5382 real-workflow continuous self-learning.

Spec refs: REQ-LEARN-5382, SCENARIO-LEARN-5382-GATE,
SCENARIO-LEARN-5382-IDENTICAL-TASKS, SCENARIO-LEARN-5382-SAFETY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5382_real_workflow_continuous_self_learning_v490 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5382_spec_declares_real_workflow_contract() -> None:
    """REQ-LEARN-5382: OpenSpec anchors the gated real-workflow run."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5382") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5382",
        "SCENARIO-LEARN-5382-GATE",
        "SCENARIO-LEARN-5382-IDENTICAL-TASKS",
        "SCENARIO-LEARN-5382-SAFETY",
        str(exp.RESULT_RELATIVE_PATH),
        "budget_memory_corrigendum_clean=true",
        "baseline variant and a self-learning variant",
        "SHALL NOT load, fine-tune, write, or mutate model weights",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5382_gate_copies_exp5381_corrigendum() -> None:
    """SCENARIO-LEARN-5382-GATE: Exp5381 clean gate opens the workflow."""

    gate = exp.confirm_upstream_gate(root=REPO)

    assert gate["all_passed"] is True
    assert gate["budget_memory_corrigendum_clean"] is True
    assert gate["source_status"] == "complete"
    assert gate["source_unsafe_false_accepts_zero"] is True
    assert gate["source_no_weight_mutation"] is True
    assert gate["source_artifact"] == str(exp.EXP5381_RELATIVE_PATH)


def test_req_learn_5382_selected_workflow_has_repeated_real_decisions() -> None:
    """REQ-LEARN-5382-2: workflow has sessions, traces, and diverse decisions."""

    traces = exp.select_workflow_traces()
    description = exp.describe_workflow(traces)

    assert description["workflow_name"] == exp.WORKFLOW_NAME
    assert description["session_count"] >= exp.MIN_SESSIONS
    assert description["trace_count"] >= exp.MIN_TRACES
    assert description["checked_event_count"] >= exp.MIN_CHECKED_EVENTS
    assert description["decision_type_counts"]["retrieval"] > 0
    assert description["decision_type_counts"]["verifier_tool_use"] > 0
    assert description["decision_type_counts"]["rollback"] > 0
    assert description["decision_type_counts"]["constraint_selection"] > 0


def test_scenario_learn_5382_identical_tasks_and_metric_deltas() -> None:
    """SCENARIO-LEARN-5382-IDENTICAL-TASKS: variants share event IDs."""

    traces = exp.select_workflow_traces()
    evaluation = exp.evaluate_real_workflow(traces=traces, root=REPO)
    baseline = evaluation["baseline_variant"]
    learner = evaluation["self_learning_variant"]

    assert evaluation["same_event_ids"] is True
    assert baseline["event_ids"] == learner["event_ids"]
    assert evaluation["checked_event_count"] == len(baseline["event_ids"])
    assert evaluation["context_efficiency_delta"] == round(
        learner["context_efficiency"] - baseline["context_efficiency"],
        6,
    )
    assert evaluation["verifier_cost_delta"] == round(
        baseline["verifier_cost"] - learner["verifier_cost"],
        6,
    )
    assert evaluation["quality_delta"] == round(
        learner["quality"] - baseline["quality"],
        6,
    )
    assert evaluation["context_efficiency_delta"] > 0.0
    assert evaluation["verifier_cost_delta"] > 0.0
    assert evaluation["quality_delta"] >= 0.0


def test_scenario_learn_5382_safety_controls_deflect_bad_memory() -> None:
    """SCENARIO-LEARN-5382-SAFETY: stale and poisoned updates are rejected."""

    evaluation = exp.evaluate_real_workflow(root=REPO)
    controls = evaluation["safety_controls"]
    churn = evaluation["memory_churn"]

    assert controls["budget_limit_respected"] is True
    assert controls["retained_bytes"] == exp.BUDGET_LIMIT_BYTES
    assert evaluation["stale_memory_deflection_rate"] == 1.0
    assert evaluation["poison_memory_deflection_rate"] == 1.0
    assert evaluation["rollback_success_rate"] == 1.0
    assert evaluation["unsafe_false_accepts"] == 0
    assert evaluation["weight_mutation_receipt"]["no_weight_mutation"] is True
    assert evaluation["weight_mutation_receipt"]["model_weights_loaded"] is False
    assert churn["retained_memory_count"] == 3
    assert churn["rejected_or_quarantined_memory_count"] == 4
    assert churn["churn_rate"] == round(4 / 7, 6)


def test_req_learn_5382_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5382-5: run() writes the required terminal artifact."""

    tests_run = [
        {
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_5382_real_workflow_continuous_self_learning_v490.py "
                "-q --no-cov"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5382_real_workflow_continuous_self_learning_v490.py "
                "-m pytest "
                "tests/python/test_experiment_5382_real_workflow_continuous_self_learning_v490.py "
                "-q --no-cov -n 0 && .venv/bin/coverage report --fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["continuous_self_learning_real_workflow_ready"] is True
    assert artifact["upstream_budget_memory_corrigendum_clean"] is True
    assert artifact["workflow_name"] == exp.WORKFLOW_NAME
    assert artifact["session_count"] >= exp.MIN_SESSIONS
    assert artifact["trace_count"] >= exp.MIN_TRACES
    assert artifact["checked_event_count"] >= exp.MIN_CHECKED_EVENTS
    assert artifact["context_efficiency_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert artifact["quality_delta"] >= 0.0
    assert artifact["stale_memory_deflection_rate"] == 1.0
    assert artifact["poison_memory_deflection_rate"] == 1.0
    assert artifact["rollback_success_rate"] == 1.0
    assert artifact["no_weight_mutation"] is True
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5382_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5382: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["continuous_self_learning_real_workflow_ready"] is True
    assert result["unsafe_false_accepts"] == 0
    assert result["no_weight_mutation"] is True


def test_req_learn_5382_blocks_when_exp5381_gate_is_false(monkeypatch) -> None:
    """REQ-LEARN-5382-1: false Exp5381 readiness blocks the workflow."""

    failed_gate = {
        "all_passed": False,
        "budget_memory_corrigendum_clean": False,
        "source_status": "blocked",
        "source_unsafe_false_accepts_zero": True,
        "source_no_weight_mutation": True,
        "source_artifact": str(exp.EXP5381_RELATIVE_PATH),
        "failed_gates": ["budget_memory_corrigendum_clean"],
    }
    monkeypatch.setattr(exp, "confirm_upstream_gate", lambda root=REPO: failed_gate)

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5382", "outcome": "passed"}],
    )

    assert artifact["status"] == "blocked"
    assert artifact["continuous_self_learning_real_workflow_ready"] is False
    assert artifact["upstream_budget_memory_corrigendum_clean"] is False
    assert artifact["session_count"] == 0
    assert artifact["trace_count"] == 0
    assert artifact["checked_event_count"] == 0
    assert artifact["honest_verdict"].startswith("blocked_")
    exp.validate_artifact(artifact)


def test_req_learn_5382_validation_rejects_terminal_claim_drift() -> None:
    """REQ-LEARN-5382-5: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5382", "outcome": "passed"}],
    )

    bad_missing = deepcopy(artifact)
    bad_missing.pop("workflow_name")
    with pytest.raises(ValueError, match="workflow_name"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["status"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_bool = deepcopy(artifact)
    bad_bool["continuous_self_learning_target"] = "true"
    with pytest.raises(ValueError, match="continuous_self_learning_target"):
        exp.validate_artifact(bad_bool)

    bad_count = deepcopy(artifact)
    bad_count["checked_event_count"] = True
    with pytest.raises(ValueError, match="checked_event_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["context_efficiency_delta"] = {"value": 1.0}
    with pytest.raises(ValueError, match="context_efficiency_delta"):
        exp.validate_artifact(bad_numeric)

    bad_ready = deepcopy(artifact)
    bad_ready["status"] = "complete"
    bad_ready["continuous_self_learning_real_workflow_ready"] = False
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_ready)

    bad_ready_status = deepcopy(artifact)
    bad_ready_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_ready_status)

    bad_upstream = deepcopy(artifact)
    bad_upstream["upstream_budget_memory_corrigendum_clean"] = False
    with pytest.raises(ValueError, match="upstream_budget_memory_corrigendum_clean"):
        exp.validate_artifact(bad_upstream)

    bad_session_count = deepcopy(artifact)
    bad_session_count["session_count"] = exp.MIN_SESSIONS - 1
    with pytest.raises(ValueError, match="session_count"):
        exp.validate_artifact(bad_session_count)

    bad_trace_count = deepcopy(artifact)
    bad_trace_count["trace_count"] = exp.MIN_TRACES - 1
    with pytest.raises(ValueError, match="trace_count"):
        exp.validate_artifact(bad_trace_count)

    bad_checked_count = deepcopy(artifact)
    bad_checked_count["checked_event_count"] = exp.MIN_CHECKED_EVENTS - 1
    with pytest.raises(ValueError, match="checked_event_count"):
        exp.validate_artifact(bad_checked_count)

    bad_context = deepcopy(artifact)
    bad_context["context_efficiency_delta"] = 0.0
    with pytest.raises(ValueError, match="context_efficiency_delta"):
        exp.validate_artifact(bad_context)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_cost_delta"] = 0.0
    with pytest.raises(ValueError, match="verifier_cost_delta"):
        exp.validate_artifact(bad_verifier)

    bad_quality = deepcopy(artifact)
    bad_quality["quality_delta"] = -0.1
    with pytest.raises(ValueError, match="quality_delta"):
        exp.validate_artifact(bad_quality)

    bad_stale = deepcopy(artifact)
    bad_stale["stale_memory_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="stale_memory_deflection_rate"):
        exp.validate_artifact(bad_stale)

    bad_poison = deepcopy(artifact)
    bad_poison["poison_memory_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="poison_memory_deflection_rate"):
        exp.validate_artifact(bad_poison)

    bad_rollback = deepcopy(artifact)
    bad_rollback["rollback_success_rate"] = 0.5
    with pytest.raises(ValueError, match="rollback_success_rate"):
        exp.validate_artifact(bad_rollback)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    assert exp._rate(1, 0) == 0.0
    assert exp._json_ready(Path("results/example.json")) == "results/example.json"
