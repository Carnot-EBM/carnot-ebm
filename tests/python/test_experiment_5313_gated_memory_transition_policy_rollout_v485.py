"""Tests for Exp5313 gated memory transition policy rollout.

Spec refs: REQ-LEARN-5313, SCENARIO-LEARN-5313.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5313_gated_memory_transition_policy_rollout_v485 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5313_spec_declares_rollout_contract() -> None:
    """REQ-LEARN-5313: OpenSpec anchors rollout fields and safety gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5313") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5313",
        "SCENARIO-LEARN-5313",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5302 adaptive memory policy",
        "Exp5312 transition-level verifier",
        "always-full",
        "adaptive",
        "no-memory",
        "clean",
        "conflict",
        "forgetting",
        "stale evidence",
        "invalid premise",
        "rollback",
        "final task quality",
        "process-level transition quality",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5313_upstream_gates_are_confirmed() -> None:
    """REQ-LEARN-5313-1: rollout confirms Exp5302, Exp5303, and Exp5312 gates."""

    gates = exp.confirm_upstream_gates(root=REPO)

    assert gates["all_passed"] is True
    assert gates["exp5302_memory_policy_candidate_ready"] is True
    assert gates["exp5303_memory_stress_passed"] is True
    assert gates["exp5312_memory_transition_verifier_ready"] is True
    assert gates["no_weight_mutation_constraints"] == {
        "exp5302_no_weight_mutation": True,
        "exp5312_no_model_weight_mutation": True,
    }

    blocked = exp.confirm_upstream_gates(
        artifacts={
            "exp5302": {"memory_policy_candidate_ready": True, "no_weight_mutation": {"value": True}},
            "exp5303": {"memory_stress_passed": {"value": True}},
            "exp5312": {
                "memory_transition_verifier_ready": False,
                "no_model_weight_mutation": True,
            },
        }
    )

    assert blocked["all_passed"] is False
    assert blocked["failed_gates"] == ["exp5312_memory_transition_verifier_ready"]


def test_req_learn_5313_rollout_panel_covers_required_case_families() -> None:
    """REQ-LEARN-5313-2: panel covers every deterministic stress case family."""

    panel = exp.build_rollout_panel()

    assert [case.family for case in panel] == list(exp.REQUIRED_CASE_FAMILIES)
    assert len({case.case_id for case in panel}) == len(panel)
    assert {case.expected_decision for case in panel} == {"accept", "reject"}
    assert [case.proposal.label for case in panel] == [
        "useful_insert",
        "conflict_resolution",
        "forgetting",
        "stale_retention",
        "hallucinated_update",
        "rollback",
    ]
    assert sum(1 for case in panel if case.unsafe) == 2
    assert sum(1 for case in panel if case.rollback_expected) == 1


def test_scenario_learn_5313_adaptive_matches_quality_and_process_score() -> None:
    """SCENARIO-LEARN-5313: adaptive preserves safety and avoids verifier calls."""

    evaluation = exp.evaluate_policy_rollout(exp.build_rollout_panel())

    assert evaluation["transition_policy_rollout_complete"] is True
    assert evaluation["policy_metrics"]["always_full"]["final_quality_rate"] == 1.0
    assert evaluation["policy_metrics"]["adaptive"]["final_quality_rate"] == 1.0
    assert evaluation["policy_metrics"]["no_memory"]["final_quality_rate"] < 1.0
    assert evaluation["policy_metrics"]["always_full"]["transition_process_score"] == 1.0
    assert evaluation["policy_metrics"]["adaptive"]["transition_process_score"] == 1.0
    assert evaluation["quality_delta_vs_always_full"] == 0.0
    assert evaluation["transition_score_delta_vs_always_full"] == 0.0
    assert evaluation["full_verifier_calls_avoided"] == 3
    assert evaluation["unsafe_false_accepts"] == 0
    assert evaluation["unsafe_commits_rejected"] == 2
    assert evaluation["rollback_events"] == 1

    adaptive_rows = {row["case_id"]: row for row in evaluation["policy_rows"]["adaptive"]}
    assert adaptive_rows["rollout-clean-runtime"]["route"] == exp.ROUTE_MEMORY_POLICY
    assert adaptive_rows["rollout-conflict-registry"]["route"] == exp.ROUTE_MEMORY_POLICY
    assert adaptive_rows["rollout-forgetting-lexical"]["route"] == exp.ROUTE_MEMORY_POLICY
    assert adaptive_rows["rollout-stale-evidence"]["route"] == exp.ROUTE_FULL_VERIFIER
    assert adaptive_rows["rollout-invalid-premise"]["route"] == exp.ROUTE_FULL_VERIFIER
    assert adaptive_rows["rollout-rollback-autopatch"]["route"] == exp.ROUTE_FULL_VERIFIER


def test_req_learn_5313_safety_counters_and_cost_proxy_are_explicit() -> None:
    """REQ-LEARN-5313-3/4: safety counters and cost proxy are policy-separated."""

    evaluation = exp.evaluate_policy_rollout(exp.build_rollout_panel())
    adaptive_rows = {row["case_id"]: row for row in evaluation["policy_rows"]["adaptive"]}
    no_memory_rows = {row["case_id"]: row for row in evaluation["policy_rows"]["no_memory"]}
    proxy = evaluation["latency_or_cost_proxy"]

    for case_id in ("rollout-stale-evidence", "rollout-invalid-premise"):
        assert adaptive_rows[case_id]["accepted_transition"] is False
        assert adaptive_rows[case_id]["persistent_state_changed"] is False
        assert adaptive_rows[case_id]["unsafe_commit_rejected"] is True

    assert adaptive_rows["rollout-rollback-autopatch"]["rollback_event"] is True
    assert adaptive_rows["rollout-rollback-autopatch"]["accepted_transition"] is True
    assert no_memory_rows["rollout-conflict-registry"]["final_correct"] is False
    assert no_memory_rows["rollout-forgetting-lexical"]["transition_process_correct"] is False
    assert proxy["unit"] == "deterministic_cost_units"
    assert proxy["by_policy"]["adaptive"]["full_transition_verifier_calls"] == 3
    assert proxy["by_policy"]["always_full"]["full_transition_verifier_calls"] == 6
    assert proxy["adaptive_cost_units_saved_vs_always_full"] > 0


def test_req_learn_5313_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5313-5: run() writes the required terminal rollout artifact."""

    tests_run = [{"command": "unit gated rollout", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "rollout_complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["gates_confirmed"]["value"]["all_passed"] is True
    assert artifact["transition_policy_rollout_complete"] is True
    assert artifact["quality_delta_vs_always_full"] == 0.0
    assert artifact["transition_score_delta_vs_always_full"] == 0.0
    assert artifact["full_verifier_calls_avoided"] == 3
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["unsafe_commits_rejected"] == 2
    assert artifact["rollback_events"] == 1
    assert artifact["latency_or_cost_proxy"]["value"]["by_policy"]["adaptive"][
        "cost_units"
    ] < artifact["latency_or_cost_proxy"]["value"]["by_policy"]["always_full"]["cost_units"]
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["no_weight_mutation"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5313_blocked_artifact_when_upstream_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LEARN-5313-1/5: failed upstream gates produce blocked artifacts."""

    def blocked_gates(*, root: Path | str = exp.REPO_ROOT) -> dict[str, object]:
        assert root == REPO
        return {
            "exp5302_memory_policy_candidate_ready": True,
            "exp5303_memory_stress_passed": True,
            "exp5312_memory_transition_verifier_ready": False,
            "no_weight_mutation_constraints": {
                "exp5302_no_weight_mutation": True,
                "exp5312_no_model_weight_mutation": True,
            },
            "failed_gates": ["exp5312_memory_transition_verifier_ready"],
            "all_passed": False,
            "upstream_honest_verdicts": {},
        }

    monkeypatch.setattr(exp, "confirm_upstream_gates", blocked_gates)

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "blocked gate unit", "outcome": "passed"}],
    )

    assert artifact["status"]["value"] == "blocked_upstream_gate_or_tests"
    assert artifact["honest_verdict"]["value"].startswith("blocked_upstream_gate_not_ready:")
    assert artifact["transition_policy_rollout_complete"] is False
    assert artifact["policy_metrics"]["adaptive"]["n"] == 0
    assert artifact["latency_or_cost_proxy"]["value"]["by_policy"]["adaptive"][
        "cost_units"
    ] == 0
    assert artifact["no_weight_mutation"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5313_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5313: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["transition_policy_rollout_complete"] is True
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["unsafe_false_accepts"] == 0
    assert result["no_weight_mutation"] is True
    exp.validate_artifact(result)


def test_req_learn_5313_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5313-5: artifact validation rejects gate and schema drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit gated rollout", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_memory_transition_verifier_no_llm"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_rollout_gate = deepcopy(artifact)
    bad_rollout_gate["transition_policy_rollout_complete"] = {"value": True}
    with pytest.raises(ValueError, match="transition_policy_rollout_complete"):
        exp.validate_artifact(bad_rollout_gate)

    bad_numeric = deepcopy(artifact)
    bad_numeric["quality_delta_vs_always_full"] = "0.0"
    with pytest.raises(ValueError, match="quality_delta_vs_always_full"):
        exp.validate_artifact(bad_numeric)

    bad_integer = deepcopy(artifact)
    bad_integer["full_verifier_calls_avoided"] = 3.0
    with pytest.raises(ValueError, match="full_verifier_calls_avoided"):
        exp.validate_artifact(bad_integer)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_learn_5313_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-5313: helper edge paths stay deterministic and explicit."""

    clean = exp.build_rollout_panel()[0]

    assert exp._final_decision(clean, "adaptive", {"transition_process_correct": False}) == (
        "reject"
    )
    assert exp._json_ready(Path("local/path")) == "local/path"
    assert exp._json_ready(("x", Path("y"))) == ["x", "y"]


def test_req_learn_5313_validation_rejects_integer_drift_duplicate_guard() -> None:
    """REQ-LEARN-5313-5: integer gate rejects float drift specifically."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit gated rollout", "outcome": "passed"}],
    )

    bad_numeric = deepcopy(artifact)
    bad_numeric["full_verifier_calls_avoided"] = 3.0
    with pytest.raises(ValueError, match="full_verifier_calls_avoided"):
        exp.validate_artifact(bad_numeric)
