"""Tests for Exp 5302 adaptive memory policy self-learning.

Spec refs: REQ-LEARN-5302, SCENARIO-LEARN-5302.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_5302_adaptive_memory_policy_self_learning_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_learn_5302_spec_declares_adaptive_policy_contract() -> None:
    """REQ-LEARN-5302: OpenSpec anchors held-out adaptive memory policy replay."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5302") :]

    for marker in (
        "REQ-LEARN-5302",
        "SCENARIO-LEARN-5302",
        str(mod.RESULT_RELATIVE_PATH),
        "selection split",
        "held-out split",
        "supported",
        "unsupported",
        "stale",
        "contradictory",
        "harmful-memory",
        "rollback",
        "always-full verifier",
        "no-memory dose",
        "fixed governed-memory dose",
        "adaptive memory policy",
        "shuffled-memory control",
        "aggregation_from_upstream_artifacts",
        "offline_deterministic_fixture_no_llm",
        "no_weight_mutation",
        "cross-model transfer",
    ):
        assert marker in section

    normalized_section = " ".join(section.split())
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(mod.FIELD_PRINCIPLES[field].split()) in normalized_section
    assert "`memory_policy_candidate_ready`" in section
    assert mod.FIELD_PRINCIPLES["memory_policy_candidate_ready"] in normalized_section


def test_req_learn_5302_heldout_split_is_not_used_for_selection() -> None:
    """REQ-LEARN-5302-1: policy selection reads only selection case IDs."""

    splits = mod.build_policy_splits(root=REPO)
    selection_ids = {row.case_id for row in splits.selection}
    heldout_ids = {row.case_id for row in splits.heldout}

    assert selection_ids
    assert heldout_ids
    assert selection_ids.isdisjoint(heldout_ids)
    assert {row.split for row in splits.selection} == {"selection"}
    assert {row.split for row in splits.heldout} == {"heldout"}
    assert {row.case_type for row in splits.heldout} >= {
        "supported",
        "unsupported",
        "stale",
        "contradictory",
        "harmful-memory",
        "rollback",
    }

    selection = mod.select_adaptive_policy(splits.selection)

    assert set(selection["selection_case_ids"]) == selection_ids
    assert not (heldout_ids & set(selection["selection_case_ids"]))
    assert selection["selected_policy"]["confidence_threshold"] == pytest.approx(0.84)
    assert selection["selected_policy"]["policy_version"] == mod.POLICY_VERSION
    assert selection["selected_policy"]["optimized_on_split"] == "selection"
    assert selection["selected_policy"]["heldout_case_ids_seen_during_selection"] == []


def test_scenario_learn_5302_adaptive_arm_preserves_quality_and_avoids_calls() -> None:
    """SCENARIO-LEARN-5302: adaptive held-out policy beats fixed dosing safely."""

    splits = mod.build_policy_splits(root=REPO)
    selection = mod.select_adaptive_policy(splits.selection)
    evaluation = mod.evaluate_heldout_arms(splits.heldout, selection)

    metrics = evaluation["policy_metrics"]
    adaptive = metrics["adaptive_memory_policy"]

    assert metrics["always_full"]["quality_rate"] == 1.0
    assert metrics["no_memory"]["full_verifier_calls"] == 5
    assert metrics["fixed_governed_memory"]["full_verifier_calls"] == 5
    assert adaptive["quality_rate"] == 1.0
    assert adaptive["full_verifier_calls"] == 4
    assert adaptive["false_accepts"] == 0
    assert adaptive["unsafe_false_accepts"] == 0
    assert evaluation["heldout_quality_delta_vs_always_full"]["delta"] == 0.0
    assert evaluation["full_verifier_calls_avoided"]["vs_always_full"] == 3
    assert evaluation["full_verifier_calls_avoided"]["additional_vs_no_memory"] == 1
    assert evaluation["full_verifier_calls_avoided"]["additional_vs_fixed_governed_memory"] == 1
    assert evaluation["adaptive_memory_policy_positive"] is True
    assert evaluation["memory_policy_candidate_ready"] is True

    rows = {row["case_id"]: row for row in evaluation["policy_rows"]["adaptive_memory_policy"]}
    assert rows["heldout-unsupported-retrieval"]["route"] == mod.ROUTE_MEMORY_CHECK
    assert rows["heldout-unsupported-retrieval"]["selected_decision"] == "reject"
    assert rows["heldout-unsupported-retrieval"]["memory_answer_injection_blocked"] is True

    shuffled = evaluation["policy_metrics"]["shuffled_memory_control"]
    assert shuffled["false_accepts"] > adaptive["false_accepts"]
    assert shuffled["unsafe_false_accepts"] > adaptive["unsafe_false_accepts"]


def test_req_learn_5302_unsafe_stale_conflict_and_rollback_rows_block_memory_acceptance() -> None:
    """REQ-LEARN-5302-2/3/4: unsafe controls escalate and leave reversible state."""

    splits = mod.build_policy_splits(root=REPO)
    selection = mod.select_adaptive_policy(splits.selection)
    evaluation = mod.evaluate_heldout_arms(splits.heldout, selection)
    rows = {row["case_id"]: row for row in evaluation["policy_rows"]["adaptive_memory_policy"]}

    for case_id in (
        "heldout-stale-conflict",
        "heldout-contradictory-shuffled",
        "heldout-harmful-memory",
        "heldout-rollback",
    ):
        assert rows[case_id]["route"] == mod.ROUTE_FULL
        assert rows[case_id]["selected_decision"] == "reject"
        assert rows[case_id]["unsafe_false_accept"] is False

    assert rows["heldout-stale-conflict"]["escalation_reason"] == "stale_or_conflicting_memory"
    assert rows["heldout-contradictory-shuffled"]["escalation_reason"] == (
        "shuffled_scope_or_routing"
    )
    assert rows["heldout-harmful-memory"]["escalation_reason"] == (
        "safety_negative_or_harmful_memory"
    )
    assert rows["heldout-rollback"]["escalation_reason"] == "rollback_memory_control"

    stale = evaluation["stale_conflict_behavior"]
    assert stale["all_escalated"] is True
    assert stale["case_ids"] == ["heldout-stale-conflict", "heldout-contradictory-shuffled"]

    rollback = evaluation["rollback_exercised"]
    assert rollback["value"] is True
    assert rollback["trigger_count"] == 2
    assert rollback["case_ids"] == ["heldout-harmful-memory", "heldout-rollback"]

    state = evaluation["adaptive_policy_state_after_heldout"]
    assert state["policy_version"] == mod.POLICY_VERSION
    assert state["memory_entries"]
    assert state["rejected_promotions"]
    assert {row["case_id"] for row in state["rejected_promotions"]} >= {
        "heldout-stale-conflict",
        "heldout-contradictory-shuffled",
        "heldout-harmful-memory",
        "heldout-rollback",
    }
    for entry in state["memory_entries"]:
        assert entry["provenance"]
        assert entry["status"] in {"promoted", "blocked", "rolled_back"}
        assert entry["scope"]
        assert "counters" in entry
        assert entry["reversible"] is True


def test_req_learn_5302_route_and_verdict_branches_are_explicit() -> None:
    """REQ-LEARN-5302-2/5: non-positive routes and verdicts stay explicit."""

    splits = mod.build_policy_splits(root=REPO)
    selection = mod.select_adaptive_policy(splits.selection)
    evaluation = mod.evaluate_heldout_arms(splits.heldout, selection)
    no_weight = mod.no_weight_mutation_receipt(selection, evaluation)

    assert mod.choose_fixed_governed_route(splits.selection[0]) == mod.ROUTE_MEMORY_CHECK

    harmful_without_rollback = replace(splits.heldout[4], rollback_required=False)
    assert mod.choose_fixed_governed_route(harmful_without_rollback) == mod.ROUTE_FULL
    assert (
        mod.choose_adaptive_route(
            harmful_without_rollback,
            selection["memory_policy_state_after_selection"],
        )
        == mod.ROUTE_FULL
    )

    harmful_control_on_unsupported = replace(
        splits.heldout[1],
        memory_control_kind="harmful_memory",
        case_type="unsupported",
        rollback_required=False,
        unsafe=False,
    )
    assert mod.choose_fixed_governed_route(harmful_control_on_unsupported) == mod.ROUTE_FULL

    unsupported_unknown_scope = replace(
        splits.heldout[1],
        task_scope="verifier/unknown_heldout_scope",
    )
    assert (
        mod.choose_adaptive_route(
            unsupported_unknown_scope,
            selection["memory_policy_state_after_selection"],
        )
        == mod.ROUTE_FULL
    )

    untrusted = replace(splits.heldout[1], memory_control_kind="missing_provenance")
    assert mod._escalation_reason(untrusted, mod.ROUTE_FULL) == "untrusted_memory_control"

    unsafe = deepcopy(evaluation)
    unsafe["unsafe_false_accepts"]["count"] = 1
    assert mod._honest_verdict(unsafe, no_weight).startswith("harmful_unsafe_false_accepts")

    quality_drop = deepcopy(evaluation)
    quality_drop["heldout_quality_delta_vs_always_full"]["delta"] = -0.1
    assert mod._honest_verdict(quality_drop, no_weight).startswith("harmful_quality_regression")

    weight_blocked = deepcopy(no_weight)
    weight_blocked["no_weight_mutation"] = False
    assert mod._honest_verdict(evaluation, weight_blocked).startswith("blocked_weight_mutation")

    null_eval = deepcopy(evaluation)
    null_eval["adaptive_memory_policy_positive"] = False
    assert mod._honest_verdict(null_eval, no_weight).startswith("null:")


def test_req_learn_5302_artifact_schema_and_no_weight_mutation(tmp_path: Path) -> None:
    """REQ-LEARN-5302-5: run() writes the required artifact without weight mutation."""

    tests_run = [{"command": "unit adaptive memory", "outcome": "passed"}]
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "adaptive memory policy helped" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_task"]["value"] is True
    assert artifact["memory_policy_candidate_ready"] is True
    assert artifact["memory_policy_candidate_ready_principle"] == (
        mod.FIELD_PRINCIPLES["memory_policy_candidate_ready"]
    )
    assert artifact["adaptive_memory_policy_positive"]["value"] is True
    assert artifact["heldout_quality_delta_vs_always_full"]["value"]["delta"] == 0.0
    assert artifact["full_verifier_calls_avoided"]["value"]["additional_vs_fixed_governed_memory"] == 1
    assert artifact["unsafe_false_accepts"]["value"]["count"] == 0
    assert artifact["rollback_exercised"]["value"]["value"] is True
    assert artifact["no_weight_mutation"]["value"] is True
    assert artifact["tests_run"] == tests_run
    assert artifact["weight_mutation_receipt"]["model_weights_loaded"] is False
    assert artifact["weight_mutation_receipt"]["model_weight_hash_before"] == (
        artifact["weight_mutation_receipt"]["model_weight_hash_after"]
    )
    assert artifact["source_artifact_checksums"]["exp5290"].startswith("sha256:")

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_learn_5302_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5302: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["memory_policy_candidate_ready"] is True
    assert result["adaptive_memory_policy_positive"]["value"] is True
    assert result["unsafe_false_accepts"]["value"]["count"] == 0
    assert result["no_weight_mutation"]["value"] is True
    mod.validate_artifact(result)


def test_req_learn_5302_validation_rejects_gate_drift() -> None:
    """REQ-LEARN-5302: artifact validation rejects required gate drift."""

    artifact = mod.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit adaptive memory", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "live_llm_judge"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_ready = deepcopy(artifact)
    bad_ready["memory_policy_candidate_ready"] = "true"
    with pytest.raises(ValueError, match="memory_policy_candidate_ready"):
        mod.validate_artifact(bad_ready)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"]["value"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        mod.validate_artifact(bad_weight)
