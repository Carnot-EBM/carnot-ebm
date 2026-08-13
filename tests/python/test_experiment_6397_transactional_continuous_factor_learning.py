"""Tests for Exp6397 transactional continuous factor learning.

Spec refs: REQ-LEARN-6397, SCENARIO-LEARN-6397-CHRONOLOGY,
SCENARIO-LEARN-6397-TRANSACTION, SCENARIO-LEARN-6397-ATTACKS,
SCENARIO-LEARN-6397-FUTURE, SCENARIO-LEARN-6397-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6397_transactional_continuous_factor_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data_6397",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6397_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6397: OpenSpec owns the transaction contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6397") : text.index("REQ-LEARN-6383")]
    for token in (
        "SCENARIO-LEARN-6397-CHRONOLOGY",
        "SCENARIO-LEARN-6397-TRANSACTION",
        "SCENARIO-LEARN-6397-ATTACKS",
        "SCENARIO-LEARN-6397-FUTURE",
        "SCENARIO-LEARN-6397-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6397_chronology_and_arm_matching(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6397-CHRONOLOGY: stream and arms are sealed."""

    artifact = _artifact(tmp_path)
    manifest = artifact["chronological_manifest_path_hash_license_balance_and_partition_seals"]
    arm_contract = artifact["preregistered_arm_contract"]
    preconditions = artifact["preconditions_checked"]

    assert manifest["event_count"] >= 48
    assert manifest["partition_counts"] == {
        "acquisition": 12,
        "release": 12,
        "retention": 12,
        "untouched_future": 12,
    }
    assert manifest["update_opportunity_count"] >= 3
    assert manifest["restart_boundary_count"] >= 2
    assert manifest["license_balance"]["balanced"] is True
    assert manifest["future_opened_before_head_freeze"] is False
    assert set(arm_contract["arms"]) == set(mod.ARMS)
    assert arm_contract["event_order_matched"] is True
    assert arm_contract["exact_check_budget_matched"] is True
    assert arm_contract["consumer_budget_matched"] is True
    assert preconditions["both_exp6396_gates_revalidated"] is True
    assert preconditions["all_preconditions_passed"] is True
    assert all(row["model_call_count"] == 0 for row in artifact["unlicensed_cell_abstention_records"])


def test_scenario_learn_6397_transaction_dispositions_and_head_history(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6397-TRANSACTION: commits bind exact predecessors."""

    artifact = _artifact(tmp_path)
    candidates = artifact["typed_candidate_records"]
    bindings = artifact["predecessor_candidate_evidence_checker_eprocess_and_effect_bindings"]
    dispositions = artifact["atomic_disposition_records"]
    history = artifact["factor_head_transition_history"]
    counts = artifact["commit_reject_quarantine_and_defer_counts"]

    assert len(candidates) == 6
    assert counts == {"Commit": 2, "Reject": 2, "Quarantine": 1, "Defer": 1}
    assert len(dispositions) == len({row["candidate_id"] for row in dispositions})
    assert len(bindings["by_candidate_id"]) == len(candidates)
    assert all(row["off_commit_evaluation"] is True for row in candidates)
    for row in candidates:
        binding = bindings["by_candidate_id"][row["candidate_id"]]
        assert binding["predecessor_head_hash"] == row["predecessor_head_hash"]
        assert binding["candidate_hash"] == row["candidate_hash"]
        assert binding["exact_release_receipt"]["released"] is True
        assert binding["exact_checker_receipt"]["checker_is_oracle"] is True
        assert binding["eprocess_state"]["state_hash"].startswith("sha256:")
        assert binding["proposed_effects_hash"].startswith("sha256:")

    assert history["initial_head_hash"] == artifact["factor_head_initial_hash"]
    assert history["commit_count"] == 2
    assert history["noncommit_head_change_count"] == 0
    assert history["terminal_head_hash"] != artifact["factor_head_initial_hash"]
    assert all(row["advanced_head"] is (row["disposition"] == "Commit") for row in dispositions)


def test_scenario_learn_6397_attack_matrix_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6397-ATTACKS: failed transactions do not change heads."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix"
    ]
    expected = {
        "stale_predecessor",
        "duplicate_effect",
        "replayed_evidence",
        "self_approval",
        "concurrent_proposal",
        "interrupted_write",
        "restart_recovery",
    }

    assert set(attacks["attacks"]) == expected
    assert attacks["all_fail_closed"] is True
    assert attacks["failed_transaction_head_change_count"] == 0
    assert attacks["restart_recovery"]["recovered_terminal_head_hash"] == artifact[
        "factor_head_transition_history"
    ]["terminal_head_hash"]
    assert all(row["failed_closed"] for row in attacks["attacks"].values())
    assert all(not row["head_changed"] for row in attacks["attacks"].values())

    head = mod.initial_factor_head()
    candidate = mod.build_candidate_records(head["head_hash"])[0]
    committed = mod.apply_transaction(
        head,
        candidate,
        seen_effect_hashes=set(),
        used_evidence_hashes=set(),
    )
    stale = {**candidate, "candidate_id": "stale-copy"}
    stale_result = mod.apply_transaction(
        committed["head_after"],
        stale,
        seen_effect_hashes={candidate["proposed_effects_hash"]},
        used_evidence_hashes={candidate["evidence_hashes"][0]},
    )
    assert stale_result["disposition"] == "Reject"
    assert stale_result["advanced_head"] is False
    assert stale_result["head_after"]["head_hash"] == committed["head_after"]["head_hash"]


def test_scenario_learn_6397_future_and_selective_rollback_carry(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6397-FUTURE: future exact utility opens once."""

    artifact = _artifact(tmp_path)
    future = artifact["untouched_future_evaluation_receipts"]
    yields = artifact["future_exact_yield_by_arm"]
    rollback = artifact["selective_rollback_control_path_hash_and_terminal_class"]

    assert future["open_count"] == 1
    assert future["opened_after_head_freeze"] is True
    assert future["future_outcomes_read_once"] is True
    assert yields["by_arm"][mod.LIVE_LEARNER_ARM]["future_exact_yield"] > yields["by_arm"][
        mod.FROZEN_BASELINE_ARM
    ]["future_exact_yield"]
    assert artifact["delta_future_exact_yield_over_frozen"] > 0
    assert rollback["terminal_class"] == "complete_positive"
    assert artifact["selective_rollback_control_ready_score"] == rollback["ready_score"] == 1.0


def test_scenario_learn_6397_cli_checksum_ready_and_negative_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6397-READY: readiness is fully conjunctive."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260813", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["transactional_continuous_self_learning_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["verifier_is_oracle"] is True
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["same_step_write_count"] == 0
    assert artifact["model_weight_change_count"] == 0
    assert artifact["protected_leakage_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    negative_cases = {
        "no_commit": lambda row: row["commit_reject_quarantine_and_defer_counts"].update(
            {"Commit": 0}
        ),
        "no_future_gain": lambda row: row.update({"delta_future_exact_yield_over_frozen": 0.0}),
        "retention_regression": lambda row: row[
            "backward_retention_and_forgetting_results"
        ].update({"harmful_retention_regression_count": 1}),
        "capacity_growth": lambda row: row["factor_growth_and_capacity_results"].update(
            {"growth_within_capacity": False}
        ),
        "attack_survivor": lambda row: row[
            "stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix"
        ].update({"all_fail_closed": False}),
        "protected_leak": lambda row: row.update({"protected_leakage_count": 1}),
        "weight_change": lambda row: row.update({"model_weight_change_count": 1}),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["transactional_continuous_self_learning_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6397_helpers_and_fail_closed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6397: helpers expose deterministic fail-closed paths."""

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.as_mapping([]) == {}
    assert mod.model_slug(mod.MANDATED_MODEL_IDS[0]).startswith("unsloth-")
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mod.read_json(malformed)

    blocked = tmp_path / "blocked_6396.json"
    blocked.write_text(
        json.dumps(
            {
                "status": "blocked_precondition",
                "honest_verdict": "blocked: fixture",
                "capability_qualified_frontier_ready_score": 0.0,
            }
        ),
        encoding="utf-8",
    )
    gate = mod.exp6396_gate_receipts(blocked)
    assert gate["gate_passed"] is False
    assert "exp6396_ready_score_not_one" in gate["blocked_reasons"]

    missing = mod.exp6396_gate_receipts(tmp_path / "missing_6396.json")
    assert missing["gate_passed"] is False
    assert missing["blocked_reasons"] == ["exp6396_artifact_missing"]

    bad_upstream = {
        "status": "complete_positive",
        "capability_qualified_frontier_ready_score": 1.0,
        "delta_verified_future_exact_yield": 0.1,
        "license_records_used_and_hashes": {"license_records": [{"license_key": "k"}]},
        "unlicensed_cell_abstention_records": [{"model_call_count": 1}],
        "untouched_future_evaluation_receipts": {
            "open_count": 1,
            "future_outcomes_read_once": True,
        },
        "protected_files_unchanged": {"unchanged": True},
        "autotokenizer_usage_count": 1,
        "protected_leakage_count": 1,
        "model_weight_change_count": 1,
    }
    bad_upstream_path = tmp_path / "bad_upstream_6396.json"
    bad_upstream_path.write_text(json.dumps(bad_upstream), encoding="utf-8")
    bad_gate = mod.exp6396_gate_receipts(bad_upstream_path)
    assert bad_gate["gate_passed"] is False
    assert {
        "external_tokenizer_used_upstream",
        "exp6396_protected_leakage",
        "exp6396_model_weight_change",
        "exp6396_unlicensed_cell_not_abstained",
    } <= set(bad_gate["blocked_reasons"])

    monkeypatch.setattr(
        mod.exp6396,
        "build_model_specs",
        lambda: {"MODEL_SPECS": [], "cached_sota_pair_receipts": {"fixture": True}},
    )
    assert mod.model_resolution_from_gate({})["cached_sota_pair_receipts"]["fixture"] is True
    monkeypatch.setattr(
        mod.exp6396,
        "tokenizer_receipts",
        lambda model_specs, tokenizer_func: [{"method": mod.TOKENIZER_METHOD}],
    )
    assert mod.tokenizer_receipts_from_gate({}, []) == [{"method": mod.TOKENIZER_METHOD}]
    monkeypatch.setattr(mod.exp6396, "host_environment_receipts", lambda: {"host": True})
    monkeypatch.setattr(
        mod.exp6396,
        "cuda_offload_and_runtime_receipts_by_model",
        lambda model_specs, host: {"complete_model_count": 0, "host": host},
    )
    assert mod.runtime_receipts_from_gate({}, [])["host"] == {"host": True}

    bad_rollback = tmp_path / "bad_6383.json"
    bad_rollback.write_text(
        json.dumps(
            {
                "status": "complete_null",
                "dependency_guided_rollback_ready_score": 0.0,
            }
        ),
        encoding="utf-8",
    )
    rollback = mod.selective_rollback_control_receipt(bad_rollback)
    assert rollback["ready_score"] == 0.0
    assert rollback["gate_passed"] is False
    missing_rollback = mod.selective_rollback_control_receipt(tmp_path / "missing_6383.json")
    assert missing_rollback["terminal_class"] == "absent"
    assert missing_rollback["gate_passed"] is False

    candidate = _artifact(tmp_path, write=False)
    candidate["preconditions_checked"]["all_preconditions_passed"] = False
    _refresh(candidate)
    assert candidate["status"] == "blocked_precondition"
    assert candidate["honest_verdict"].startswith("blocked:")

    no_delta = deepcopy(candidate)
    del no_delta["delta_future_exact_yield_over_frozen"]
    _refresh(no_delta)
    assert "delta_future_exact_yield_over_frozen" in no_delta

    failing_preconditions = mod.preconditions_checked(
        date="20260101",
        gate={"gate_passed": False},
        rollback={"gate_passed": False},
        model_resolution={"MODEL_SPECS": []},
        tokenizer_rows=[{"method": "wrong", "autotokenizer_used": True}],
        runtime={},
        bindings={},
        manifest={"event_count": 0, "license_balance": {"balanced": False}},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    assert {
        "wrong_planning_date",
        "exp6396_gates_not_ready",
        "exp6383_rollback_not_ready",
        "model_specs_wrong_ids",
        "embedded_tokenizer_method_mismatch",
        "external_tokenizer_used",
        "runtime_receipts_incomplete",
        "license_harness_hash_mismatch",
        "exact_checker_hash_missing",
        "evalue_ledger_not_ready",
        "chronological_stream_too_short",
        "license_balance_failed",
        "protected_hash_missing",
        "source_hash_missing",
    } == set(failing_preconditions["blocked_reasons"])

    head = mod.initial_factor_head()
    self_approved = mod._with_candidate_hash(
        {
            **mod._candidate_base(
                "candidate-self-approval",
                head["head_hash"],
                "self_approval_effect",
            ),
            "self_approved": True,
        }
    )
    assert (
        mod.apply_transaction(
            head,
            self_approved,
            seen_effect_hashes=set(),
            used_evidence_hashes=set(),
        )["reason"]
        == "self_approval_forbidden"
    )
    replayed = mod.build_candidate_records(head["head_hash"])[0]
    assert (
        mod.apply_transaction(
            head,
            replayed,
            seen_effect_hashes=set(),
            used_evidence_hashes={replayed["evidence_hashes"][0]},
        )["reason"]
        == "replayed_evidence"
    )
    protected_fail = mod._with_candidate_hash(
        {
            **mod._candidate_base(
                "candidate-protected-replay",
                head["head_hash"],
                "protected_effect",
                protected_replay_passed=False,
            )
        }
    )
    assert (
        mod.apply_transaction(
            head,
            protected_fail,
            seen_effect_hashes=set(),
            used_evidence_hashes=set(),
        )["reason"]
        == "protected_replay_failed"
    )
