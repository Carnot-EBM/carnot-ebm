"""Tests for Exp6398 default-off transactional factor consumer.

Spec refs: REQ-LEARN-6398, SCENARIO-LEARN-6398-READONLY,
SCENARIO-LEARN-6398-LICENSED, SCENARIO-LEARN-6398-MATCHED,
SCENARIO-LEARN-6398-ATTACKS, SCENARIO-LEARN-6398-ROLLBACK,
SCENARIO-LEARN-6398-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6398_default_off_transactional_factor_consumer as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data_6398",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6398_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6398: OpenSpec owns the consumer contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6398") : text.index("REQ-LEARN-6383")]
    for token in (
        "SCENARIO-LEARN-6398-READONLY",
        "SCENARIO-LEARN-6398-LICENSED",
        "SCENARIO-LEARN-6398-MATCHED",
        "SCENARIO-LEARN-6398-ATTACKS",
        "SCENARIO-LEARN-6398-ROLLBACK",
        "SCENARIO-LEARN-6398-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6398_readonly_preconditions_and_head_freeze(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6398-READONLY: consumer calls cannot mutate state."""

    artifact = _artifact(tmp_path)
    preconditions = artifact["preconditions_checked"]
    head = artifact["frozen_factor_head_and_transaction_log_hashes"]
    manifest = artifact[
        "untouched_consumer_manifest_path_hash_license_balance_and_prior_access_receipt"
    ]

    assert preconditions["all_preconditions_passed"] is True
    assert preconditions["both_exp6397_gates_revalidated"] is True
    assert preconditions["factor_head_hash_revalidated"] is True
    assert preconditions["transaction_log_revalidated"] is True
    assert preconditions["license_bindings_revalidated"] is True
    assert preconditions["rollback_receipt_revalidated"] is True
    assert head["retained_predecessor_bound_head_hash"].startswith("sha256:")
    assert head["consumer_read_only"] is True
    assert head["transaction_log_entry_count"] >= 6
    assert manifest["event_count"] >= 24
    assert manifest["license_balance"]["balanced"] is True
    assert manifest["prior_access_receipt"]["protected_outcomes_read_before_decision"] is False
    assert artifact["consumer_factor_write_count"] == 0
    assert artifact["factor_head_advance_count"] == 0
    assert artifact["license_renewal_count"] == 0
    assert artifact["silent_fallback_count"] == 0
    assert artifact["production_enable_count"] == 0
    assert artifact["protected_leakage_count"] == 0


def test_scenario_learn_6398_licensed_cells_abstain_without_family_switch(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6398-LICENSED: invalid cells abstain explicitly."""

    artifact = _artifact(tmp_path)
    results = artifact[
        "per_model_family_retrieval_license_abstention_checker_yield_and_cost_results"
    ]
    abstentions = results["abstention_records"]

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert all(
        row["method"] == mod.TOKENIZER_METHOD and row["autotokenizer_used"] is False
        for row in artifact["embedded_gguf_tokenizer_receipts"]
    )
    assert set(artifact["models_used"]) <= set(mod.MANDATED_MODEL_IDS)
    assert results["called_only_licensed_cells"] is True
    assert results["retry_switch_abstain_distinct"] is True
    assert results["abstentions_pooled_as_success"] is False
    assert {"unlicensed", "rejected", "expired", "stale", "revoked"} <= {
        row["abstention_reason"] for row in abstentions
    }
    assert all(row["terminal_decision"] == "abstain" for row in abstentions)
    assert all(row["model_call_count"] == 0 for row in abstentions)
    assert all(row["fallback_model_hf_id"] is None for row in abstentions)
    assert all(row["family_switch_approved"] is False for row in abstentions)
    assert all(row["inherited_license"] is False for row in abstentions)


def test_scenario_learn_6398_matched_work_yield_and_intervals(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6398-MATCHED: arms share future consumer work."""

    artifact = _artifact(tmp_path)
    arms = artifact["preregistered_arm_contract"]
    work = artifact["matched_work_receipts"]
    yields = artifact["exact_yield_by_arm"]
    intervals = artifact["confidence_intervals_and_effective_sample_sizes"]
    harm = artifact["false_accept_false_reject_negative_transfer_and_harm_results"]

    assert set(arms["arms"]) == set(mod.ARMS)
    assert arms["event_order_matched"] is True
    assert arms["token_budget_matched"] is True
    assert work["matched_event_count"] is True
    assert work["matched_model_call_count"] is True
    assert work["matched_exact_checker_call_count"] is True
    assert work["matched_token_budget"] is True
    assert yields["by_arm"][mod.V550_ARM]["exact_yield"] > yields["by_arm"][
        mod.FROZEN_BASELINE_ARM
    ]["exact_yield"]
    assert artifact["delta_exact_yield_over_frozen"] > 0.0
    assert harm["false_accepts_do_not_increase"] is True
    assert harm["by_arm"][mod.V550_ARM]["false_accept_count"] <= harm["by_arm"][
        mod.FROZEN_BASELINE_ARM
    ]["false_accept_count"]
    assert intervals["pooled"]["abstentions_counted_as_success"] is False
    assert intervals["pooled"]["effective_sample_size"] == yields["by_arm"][mod.V550_ARM][
        "exact_event_count"
    ]
    assert set(intervals["by_family"]) == set(manifest_family_counts(artifact))


def test_scenario_learn_6398_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6398-ATTACKS: attacks cannot write or fallback."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix"
    ]
    expected = {
        "stale_head",
        "revoked_descendant",
        "expired_license",
        "model_row_swap",
        "family_switch_request",
        "absent_licensed_model",
        "duplicated_evidence",
        "incomplete_rollback",
        "suppressed_abstention",
    }

    assert set(attacks["attacks"]) == expected
    assert attacks["all_fail_closed"] is True
    assert attacks["retry_switch_abstain_distinct"] is True
    assert attacks["failed_cell_factor_write_count"] == 0
    assert attacks["failed_cell_head_advance_count"] == 0
    assert attacks["failed_cell_license_renewal_count"] == 0
    assert attacks["failed_cell_production_enable_count"] == 0
    assert all(row["failed_closed"] for row in attacks["attacks"].values())
    assert all(row["fallback_model_hf_id"] is None for row in attacks["attacks"].values())
    assert all(not row["inherited_license"] for row in attacks["attacks"].values())
    assert attacks["attacks"]["family_switch_request"]["family_switch_approved"] is False
    assert attacks["attacks"]["suppressed_abstention"]["terminal_decision"] == "abstain"


def test_scenario_learn_6398_rollback_controls_scope(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6398-ROLLBACK: rollback control is inherited, not new."""

    artifact = _artifact(tmp_path)
    rollback = artifact["selective_rollback_full_reset_and_no_rollback_injected_cell_results"]

    assert rollback["scope"] == "injected_cells_only"
    assert rollback["source_exp6383_ready_score"] == 1.0
    assert rollback["new_rollback_method_claimed"] is False
    assert rollback["original_rollback_benchmark_rerun_count"] == 0
    assert rollback["selective_descendant_rollback"]["harmful_descendants_removed"] is True
    assert rollback["selective_descendant_rollback"]["unsafe_survivor_count"] == 0
    assert rollback["full_registry_reset"]["overrollback_count"] > 0
    assert rollback["no_rollback"]["unsafe_survivor_count"] > 0


def test_scenario_learn_6398_cli_checksum_ready_and_negative_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6398-READY: readiness is fully conjunctive."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260813", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["default_off_transactional_consumer_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["tests_run"]["all_passed"] is True
    assert mod.validate_artifact(artifact) is None

    negative_cases = {
        "no_gain": lambda row: row.update({"delta_exact_yield_over_frozen": 0.0}),
        "false_accept_increase": lambda row: row[
            "false_accept_false_reject_negative_transfer_and_harm_results"
        ].update({"false_accepts_do_not_increase": False}),
        "attack_survivor": lambda row: row[
            "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix"
        ].update({"all_fail_closed": False}),
        "rollback_failed": lambda row: row[
            "selective_rollback_full_reset_and_no_rollback_injected_cell_results"
        ]["selective_descendant_rollback"].update({"harmful_descendants_removed": False}),
        "production_enable": lambda row: row.update({"production_enable_count": 1}),
        "protected_leak": lambda row: row.update({"protected_leakage_count": 1}),
        "consumer_write": lambda row: row.update({"consumer_factor_write_count": 1}),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["default_off_transactional_consumer_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6398_helpers_and_fail_closed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6398: helpers expose deterministic fail-closed paths."""

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.as_mapping([]) == {}
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.write_payload_or_hash(tmp_path / "hash_only.json", {"x": 1}, write=False) == mod.sha256_json(
        {"x": 1}
    )
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mod.read_json(malformed)

    missing = mod.exp6397_gate_receipts(tmp_path / "missing_6397.json")
    assert missing["gate_passed"] is False
    assert missing["blocked_reasons"] == ["exp6397_artifact_missing"]
    missing_rollback = mod.exp6383_rollback_receipt(tmp_path / "missing_6383.json")
    assert missing_rollback["gate_passed"] is False
    assert missing_rollback["terminal_class"] == "absent"

    blocked_path = tmp_path / "blocked_6397.json"
    blocked_path.write_text(
        json.dumps(
            {
                "status": "complete_null",
                "transactional_continuous_self_learning_ready_score": 0.0,
                "factor_head_transition_history": {"commit_count": 0},
                "MODEL_SPECS": [],
            }
        ),
        encoding="utf-8",
    )
    blocked = mod.exp6397_gate_receipts(blocked_path)
    assert blocked["gate_passed"] is False
    assert "exp6397_ready_score_not_one" in blocked["blocked_reasons"]

    monkeypatch.setattr(
        mod.exp6397,
        "model_resolution_from_gate",
        lambda gate: {"MODEL_SPECS": [{"hf_id": "fixture"}], "cached_sota_pair_receipts": {}},
    )
    assert mod.model_resolution_from_gate({})["MODEL_SPECS"] == [{"hf_id": "fixture"}]
    monkeypatch.setattr(
        mod.exp6397,
        "tokenizer_receipts_from_gate",
        lambda gate, specs: [{"method": mod.TOKENIZER_METHOD}],
    )
    assert mod.tokenizer_receipts_from_gate({}, []) == [{"method": mod.TOKENIZER_METHOD}]
    monkeypatch.setattr(
        mod.exp6397,
        "runtime_receipts_from_gate",
        lambda gate, specs: {"complete_model_count": 0},
    )
    assert mod.runtime_receipts_from_gate({}, []) == {"complete_model_count": 0}
    assert mod._wald_interval(0, 0)["ci95"] == [0.0, 0.0]

    failing = mod.preconditions_checked(
        date="20260101",
        gate={"gate_passed": False},
        rollback={"gate_passed": False},
        model_specs=[],
        tokenizer_rows=[{"method": "wrong", "autotokenizer_used": True}],
        runtime={},
        bindings={},
        manifest={"event_count": 0, "license_balance": {"balanced": False}},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    assert {
        "wrong_planning_date",
        "exp6397_gates_not_ready",
        "retained_factor_head_missing",
        "transaction_log_missing",
        "exp6383_rollback_not_ready",
        "model_specs_wrong_ids",
        "embedded_tokenizer_method_mismatch",
        "external_tokenizer_used",
        "runtime_receipts_incomplete",
        "license_harness_hash_mismatch",
        "exact_checker_hash_missing",
        "consumer_manifest_too_short",
        "consumer_license_balance_failed",
        "protected_hash_missing",
        "source_hash_missing",
    } == set(failing["blocked_reasons"])

    candidate = _artifact(tmp_path, write=False)
    candidate["preconditions_checked"]["all_preconditions_passed"] = False
    _refresh(candidate)
    assert candidate["status"] == "blocked_precondition"
    assert candidate["honest_verdict"].startswith("blocked:")

    no_delta = deepcopy(candidate)
    del no_delta["delta_exact_yield_over_frozen"]
    _refresh(no_delta)
    assert "delta_exact_yield_over_frozen" in no_delta


def manifest_family_counts(artifact: dict[str, Any]) -> dict[str, int]:
    """Return the family counts used by interval assertions."""

    manifest = artifact[
        "untouched_consumer_manifest_path_hash_license_balance_and_prior_access_receipt"
    ]
    return dict(manifest["license_balance"]["events_by_family"])
