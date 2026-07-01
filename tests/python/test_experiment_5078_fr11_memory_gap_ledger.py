"""Tests for Exp 5078 FR-11 memory-gap blocker ledger.

Spec refs: REQ-LEARN-5078, SCENARIO-LEARN-5078-MEMORY-GAP-LEDGER.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5078_fr11_memory_gap_ledger as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    _write_json(
        root / exp.EXP5051_RESULT_RELATIVE_PATH,
        {
            "experiment_id": 5051,
            "honest_verdict": "complete_verifier_trace_self_learning_replay_memory_minus_0p050",
            "duration_s": 0.037486,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "heldout_delta": -0.05,
            "pre_update_accuracy": 0.7,
            "post_update_accuracy": 0.65,
            "contamination_guard_passed": True,
            "self_learning_loop_executed": True,
            "update_type": "replay_memory_insertion",
            "trace_filter_diagnostics": {
                "generated_trace_count": 312,
                "rejected_trace_count": 87,
                "contamination_guard": {"passed": True, "violations": []},
            },
            "heldout_evaluation": {"heldout_n": 40, "heldout_delta": -0.05},
        },
    )
    _write_json(
        root / exp.EXP5064_RESULT_RELATIVE_PATH,
        {
            "experiment_id": 5064,
            "honest_verdict": "complete_guarded_no_promote_minus_0p050",
            "duration_s": 0.006432,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "heldout_delta": -0.05,
            "nonforgetting_delta": -0.142857,
            "pre_update_accuracy": 0.7,
            "post_update_accuracy": 0.65,
            "contamination_guard_passed": True,
            "self_learning_loop_executed": True,
            "promoted": False,
            "promoted_skill_ids": [],
            "candidate_skill_count": 2,
            "verified_skill_count": 2,
            "no_promote_reason": "heldout_delta_nonpositive;nonforgetting_regressed",
            "promotion_decision": {
                "promoted": False,
                "no_promote_reason": "heldout_delta_nonpositive;nonforgetting_regressed",
            },
            "positive_control": {"headroom_present": True, "oracle_at_k": 0.865},
            "heldout_evaluation": {
                "heldout_n": 40,
                "regressed_previously_correct_ids": ["q0167", "q0176"],
                "improved_previously_wrong_ids": ["q0166"],
            },
        },
    )
    _write_json(
        root / exp.EXP5077_RESULT_RELATIVE_PATH,
        {
            "experiment_id": 5077,
            "honest_verdict": "complete_fr11_group_sc_memory_guarded_no_promote_delta_minus_0p050",
            "duration_s": 0.005109,
            "inference_substrate": "deterministic_group_sc_memory_replay_no_live_llm",
            "heldout_delta": -0.05,
            "nonforgetting_delta": -0.142857,
            "contamination_guard_passed": True,
            "flagged_adversarial": False,
            "fr11_attempt_completed": True,
            "promoted_count": 0,
            "quarantined_count": 3,
            "rollback_guard_passed": True,
            "promotion_decision": {
                "promoted": False,
                "no_promote_reason": "heldout_delta_negative;nonforgetting_regressed",
                "gate_conditions": {
                    "contamination_guard_passed": True,
                    "heldout_delta_gte_zero": False,
                    "nonforgetting_delta_gte_zero": False,
                    "rollback_guard_passed": True,
                },
            },
            "memory_policy": {
                "policy_signature": "fallback_to_tuned_on_verifier_sc_disagreement",
                "trigger": "verifier_sc_disagreement",
                "action": "retrieve_verified_trace_then_fallback_to_tuned_self_consistency",
            },
            "group_self_consistency_summary": {
                "quarantined_nonconsensus_proposal_ids": [
                    "proposal_5077_abstention_trace_memory",
                    "proposal_5077_promote_all_verified_trace_memory",
                ],
                "tested_consensus_candidate_ids": ["candidate_5077_disagreement_fallback"],
            },
            "heldout_evaluation": {
                "n_rows": 20,
                "delta": -0.05,
                "regressed_previously_correct_ids": ["q0182", "q0198"],
                "improved_previously_wrong_ids": ["q0181"],
            },
            "dev_evaluation": {"n_rows": 8, "delta": 0.375},
            "upstream_flagged_adversarial_sources": [
                "results/experiment_5059_d1_sota_refresh_audit.json"
            ],
        },
    )
    return root


def test_req_learn_5078_spec_declares_memory_gap_ledger_contract() -> None:
    """REQ-LEARN-5078: OpenSpec anchors the blocker-ledger artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-LEARN-5078",
        "SCENARIO-LEARN-5078-MEMORY-GAP-LEDGER",
        "experiment_5078_fr11_memory_gap_ledger.py",
        "results/experiment_5078_fr11_memory_gap_ledger_v466.json",
        "complete_fr11_memory_gap_ledger_written_no_promotion",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for mode in exp.CONTROLLED_FAILURE_MODES:
        assert mode in spec


def test_scenario_learn_5078_builds_required_no_promotion_ledger(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5078-MEMORY-GAP-LEDGER: all attempts are summarized."""

    root = _fixture_root(tmp_path)
    artifact = exp.run(root=root, artifact_path=tmp_path / "ledger.json", now=lambda: 42.0, write=True)

    assert artifact["honest_verdict"].startswith(
        "complete_fr11_memory_gap_ledger_written_no_promotion"
    )
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.0
    assert artifact["promotion_executed"] is False
    assert artifact["promoted_updates"] == []
    assert artifact["flagged_adversarial"] is False
    assert [row["experiment_id"] for row in artifact["fr11_attempts_summarized"]] == [
        5051,
        5064,
        5077,
    ]
    assert artifact["upstream_flagged_adversarial_sources"] == [
        "results/experiment_5059_d1_sota_refresh_audit.json"
    ]
    assert set(artifact["field_principles"]) >= set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8")) == artifact


def test_scenario_learn_5078_classifies_controlled_failure_modes(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5078-MEMORY-GAP-LEDGER: failures use controlled labels."""

    artifact = exp.run(root=_fixture_root(tmp_path), write=False)
    modes = {row["mode"]: row for row in artifact["recurring_failure_modes"]}

    assert set(modes) == set(exp.CONTROLLED_FAILURE_MODES)
    assert modes["data_contamination"]["observed"] is False
    assert modes["irrelevant_replay"]["observed"] is True
    assert modes["retrieval_mismatch"]["affected_experiments"] == [5051, 5077]
    assert modes["overfitting"]["affected_experiments"] == [5077]
    assert modes["nonforgetting_regression"]["affected_experiments"] == [5064, 5077]
    assert modes["verifier_shortcut"]["affected_experiments"] == [5077]
    assert modes["insufficient_evaluation_power"]["affected_experiments"] == [5051, 5064, 5077]
    for mode in modes.values():
        assert mode["evidence"]
        assert mode["guard_or_blocker"]

    blockers = {row["blocker"] for row in artifact["promotion_blockers"]}
    assert {
        "heldout_delta_nonpositive",
        "nonforgetting_regressed",
        "upstream_adversarial_source_preserved",
        "insufficient_evaluation_power",
    } <= blockers


def test_scenario_learn_5078_next_and_retired_mechanisms_are_machine_readable(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5078: next mechanisms are explicit planner choices."""

    artifact = exp.run(root=_fixture_root(tmp_path), write=False)

    next_names = {row["mechanism"] for row in artifact["safe_next_mechanisms"]}
    assert next_names == {
        "retrieval_config_evolution",
        "skill_lifecycle_governance",
        "process_verifier_replay",
        "bounded_fr11_retirement_for_domain",
    }
    assert all(row["safe_to_try"] is True for row in artifact["safe_next_mechanisms"])
    retired_names = {row["mechanism"] for row in artifact["retired_mechanisms"]}
    assert {
        "blind_replay_memory_insertion",
        "promote_on_dev_or_consensus_without_heldout_gain",
        "group_sc_disagreement_fallback_policy",
    } <= retired_names
    doc_targets = {row["target"] for row in artifact["docs_update_recommendations"]}
    assert {"research-program.md", "ops/known-issues.md", "_bmad/prd.md"} <= doc_targets


def test_req_learn_5078_schema_errors_and_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5078: malformed ledgers and missing sources are rejected."""

    assert exp.read_json_object(tmp_path / "missing.json") == {}
    assert exp.sha256_file(tmp_path / "missing.json") is None
    assert exp.number(True) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp.read_json_object(bad) == {}
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(listed) == {}

    empty_attempt = exp.summarize_attempt({}, "missing.json")
    assert empty_attempt["experiment_id"] == 0
    assert empty_attempt["artifact_present"] is False
    assert exp.guard_that_worked({"rollback_guard_passed": True}) == "rollback_guard_preserved_baseline"
    assert (
        exp.guard_that_worked({"contamination_guard_passed": True, "heldout_delta": 0.0})
        == "contamination_guard_passed_no_leak"
    )
    assert exp.upstream_flagged_sources({"upstream_flagged_adversarial_sources": "bad"}) == []

    errors = exp.artifact_schema_errors(
        {
            "schema": "bad",
            "spec_refs": [],
            "honest_verdict": "bad",
            "duration_s": "bad",
            "inference_substrate": "live_llm_inference",
            "fr11_attempts_summarized": [],
            "recurring_failure_modes": [],
            "safe_next_mechanisms": [],
            "retired_mechanisms": [],
            "promotion_blockers": [],
            "docs_update_recommendations": [],
            "flagged_adversarial": "no",
            "field_principles": {},
            "promotion_executed": True,
            "promoted_updates": ["bad"],
        }
    )
    for field in (
        "schema",
        "spec_refs",
        "honest_verdict",
        "duration_s",
        "inference_substrate",
        "fr11_attempts_summarized",
        "recurring_failure_modes",
        "safe_next_mechanisms",
        "retired_mechanisms",
        "promotion_blockers",
        "docs_update_recommendations",
        "flagged_adversarial",
        "field_principles",
        "promotion_executed",
        "promoted_updates",
    ):
        assert field in errors


def test_req_learn_5078_run_aborts_on_internal_schema_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5078: run refuses to write an internally invalid ledger."""

    root = _fixture_root(tmp_path)

    def _bad_errors(_artifact: exp.JsonMap) -> list[str]:
        return ["forced_schema_error"]

    monkeypatch.setattr(exp, "artifact_schema_errors", _bad_errors)

    with pytest.raises(AssertionError, match="forced_schema_error"):
        exp.run(root=root, artifact_path=tmp_path / "bad-ledger.json", write=True)
