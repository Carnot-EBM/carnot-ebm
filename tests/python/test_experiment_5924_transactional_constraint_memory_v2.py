"""Tests for Exp5924 transactional constraint-memory V2.

Spec refs: REQ-LEARN-5924, REQ-STORE-5924,
SCENARIO-LEARN-5924-TRANSACTIONS,
SCENARIO-LEARN-5924-REJECTION,
SCENARIO-LEARN-5924-RECOVERY,
SCENARIO-LEARN-5924-CONTROLS,
SCENARIO-STORE-5924.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5924_transactional_constraint_memory_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
STORE_SPEC = REPO / "openspec/capabilities/constraint-store/spec.md"
FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5924_transactional_constraint_memory_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5924_transactional_constraint_memory_v2.py "
    "-m pytest tests/python/test_experiment_5924_transactional_constraint_memory_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5924_transactional_constraint_memory_v2.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5924_transactional_constraint_memory_v2.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5924_transactional_constraint_memory_v2.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
TEST_COMMANDS = [
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    ".venv/bin/python -m carnot.experiment_5924_transactional_constraint_memory_v2 --validate",
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    FULL_TEST_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5924: build the deterministic transaction artifact once."""

    base = tmp_path_factory.mktemp("exp5924")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5924_specs_declare_transaction_and_store_contracts() -> None:
    """REQ-LEARN-5924/REQ-STORE-5924: specs anchor fields and scenarios."""

    learn = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    store = STORE_SPEC.read_text(encoding="utf-8")
    learn_section = learn[learn.index("## REQ-LEARN-5924") : learn.index("## REQ-LEARN-5859")]
    store_section = store[store.index("### REQ-STORE-5924") :]
    normalized = " ".join(learn_section.split())

    for marker in (
        "REQ-LEARN-5924",
        "SCENARIO-LEARN-5924-TRANSACTIONS",
        "SCENARIO-LEARN-5924-REJECTION",
        "SCENARIO-LEARN-5924-RECOVERY",
        "SCENARIO-LEARN-5924-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`transactional_memory_fixture_ready_score`",
    ):
        assert marker in learn_section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in learn_section
        assert " ".join(principle.split()) in normalized
    for marker in (
        "REQ-STORE-5924",
        "SCENARIO-STORE-5924",
        "rejected updates have zero propagation",
        "validator-substitution tampering fails closed",
    ):
        assert marker in store_section


def test_scenario_learn_5924_artifact_is_ready_and_hash_bound(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5924-TRANSACTIONS: result JSON is stable and complete."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == replay == artifact
    assert mod.validate_artifact(replay) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(replay)
    assert replay["status"] == "complete_ready"
    assert replay["honest_verdict"].startswith("complete_ready:")
    assert replay["continuous_self_learning_task"] is True
    assert replay["transactional_memory_fixture_ready_score"] == pytest.approx(1.0)
    assert isinstance(replay["transactional_memory_fixture_ready_score"], float)
    assert replay["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert replay["verifier_is_oracle"] is True
    assert replay["reproducibility_checksum"] == mod.reproducibility_checksum(replay)
    assert replay["test_commands"] == TEST_COMMANDS
    assert replay["test_exit_codes"] == TEST_EXIT_CODES
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert replay["field_provenance"][field]["principle"] == principle


def test_scenario_learn_5924_replays_exp5920_gate_and_admitted_stream(
    artifact: dict[str, Any],
) -> None:
    """REQ-LEARN-5924: the fresh Exp5920 stream is the transaction source."""

    gate = artifact["gate_replay_receipt"]
    stream = artifact["admitted_stream_path_hash_rows_and_prefix_chain"]
    preconditions = artifact["preconditions_checked"]

    assert gate["exp5920_status"] == "complete_ready"
    assert gate["exp5920_ready_score"] == pytest.approx(1.0)
    assert gate["artifact_validates"] is True
    assert gate["stream_replay_ok"] is True
    assert gate["retired_exp5912_dependency_used"] is False
    assert stream["path"] == mod.EXP5920_ROWS_RELATIVE_PATH.as_posix()
    assert stream["row_count"] == 198
    assert stream["prefix_chain_valid"] is True
    assert stream["final_prefix_checksum"].startswith("sha256:")
    assert preconditions["preconditions_ready"] is True
    assert preconditions["exact_verifier_availability"]["available"] is True
    assert preconditions["atomic_writes"]["ok"] is True
    assert preconditions["disk"]["ok"] is True
    assert preconditions["ram"]["ok"] is True


def test_scenario_learn_5924_operation_ledger_enforces_order_and_hash_chain(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-5924: operation ledger and state hashes replay exactly."""

    schema = artifact["transaction_schema_and_version"]
    ledger = artifact["operation_ledger_and_state_hash_chain"]
    frozen = artifact["frozen_read_commit_validate_write_receipts"]
    authority = artifact["exact_promotion_authority"]

    assert set(schema["supported_operations"]) == set(mod.SUPPORTED_OPERATIONS)
    assert ledger["operation_count"] == len(ledger["sample_ledger"])
    assert set(ledger["operations_present"]) == set(mod.SUPPORTED_OPERATIONS)
    assert ledger["state_hash_chain_valid"] is True
    assert ledger["all_transitions_bind_event_receipt"] is True
    assert ledger["ledger_hash"].startswith("sha256:")
    assert ledger["initial_state_hash"].startswith("sha256:")
    assert ledger["final_state_hash"].startswith("sha256:")
    prior = ledger["initial_state_hash"]
    for entry in ledger["sample_ledger"]:
        assert entry["previous_state_hash"] == prior
        assert entry["resulting_state_hash"].startswith("sha256:")
        assert entry["exact_validator_receipt_hash"].startswith("sha256:")
        assert entry["row_prefix_checksum"].startswith("sha256:")
        prior = entry["resulting_state_hash"]
    assert prior == ledger["final_state_hash"]

    assert frozen["snapshots_before_lookup"] is True
    assert frozen["writes_after_commit_and_validate"] is True
    assert frozen["same_event_read_after_write_rejected"] is True
    assert frozen["future_label_visibility_rejected"] is True
    assert frozen["model_authored_label_rejected"] is True
    assert frozen["duplicate_commit_rejected"] is True
    assert frozen["stale_snapshot_rejected"] is True
    assert frozen["invalid_transition_order_rejected"] is True
    assert frozen["validator_substitution_rejected"] is True
    assert frozen["partial_state_write_rejected"] is True

    assert authority["authority"] == mod.EXACT_VALIDATOR_AUTHORITY
    assert authority["exact_verifier_promoted_update_count"] > 0
    assert authority["model_output_promoted_update_count"] == 0
    assert authority["memory_similarity_promoted_update_count"] == 0
    assert authority["validator_substitution_rejected"] is True
    assert authority["only_exact_verifier_authorized_promotion"] is True


def test_scenario_learn_5924_invalid_matrix_poison_controls_and_recovery(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5924-REJECTION/RECOVERY/CONTROLS: unsafe paths fail."""

    invalid = artifact["invalid_transition_and_leakage_rejection_matrix"]
    controls = artifact["fixed_no_memory_coupled_shuffled_and_corrupt_validator_controls"]
    poison = artifact["poison_burst_quarantine_recovery_and_retention"]
    rejected = artifact["rejected_update_non_propagation"]

    assert invalid["all_rejected"] is True
    assert invalid["state_hash_unchanged_for_all_rejections"] is True
    assert invalid["partial_state_write_count"] == 0
    assert {
        "same_event_read_after_write",
        "future_label_visibility",
        "model_authored_label",
        "duplicate_commit",
        "stale_snapshot",
        "invalid_transition_order",
        "validator_substitution",
        "partial_state_write",
    } <= {case["case"] for case in invalid["cases"]}

    budgets = {arm["query_budget"] for arm in controls["arms"].values()} | {
        arm["capacity_budget"] for arm in controls["arms"].values()
    }
    assert budgets == {mod.TRANSACTION_EVENT_BUDGET, mod.ACTIVE_CAPACITY}
    assert controls["matched_query_and_capacity_budgets"] is True
    assert controls["arms"]["transactional_memory"]["unsafe_propagation_count"] == 0
    assert controls["arms"]["immediate_coupled_writes"]["same_event_leakage_count"] > 0
    assert controls["arms"]["corrupted_validator"]["unsafe_propagation_count"] > 0
    assert controls["transactional_beats_controls_on_safety"] is True

    assert poison["poison_burst_count"] == len(mod.POISON_EVENT_INDICES)
    assert poison["semantic_near_miss_count"] == len(mod.NEAR_MISS_EVENT_INDICES)
    assert poison["deterministic_quarantine"] is True
    assert poison["poison_or_near_miss_promoted_count"] == 0
    assert poison["protected_prefix_retention_score"] == pytest.approx(1.0)
    assert poison["rollback_recovery_hash_matches"] is True
    assert poison["restart_recovery_hash_matches"] is True
    assert rejected["rejected_update_count"] > 0
    assert rejected["active_propagation_count"] == 0
    assert rejected["future_context_propagation_count"] == 0
    assert rejected["replay_context_propagation_count"] == 0


def test_scenario_learn_5924_supersession_capacity_restart_weights_and_boundary(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5924-RECOVERY: capacity, rollback, weights, and boundary."""

    recovery = artifact["supersession_capacity_rollback_and_restart"]
    weights = artifact["no_model_weight_mutation"]
    boundary = artifact["task_owned_test_boundary_and_global_failure_delta"]
    hardware = artifact["hardware_mapping_contract"]
    protected = artifact["protected_files_unchanged"]

    assert recovery["supersession_count"] > 0
    assert recovery["active_capacity"] == mod.ACTIVE_CAPACITY
    assert recovery["max_active_count"] <= mod.ACTIVE_CAPACITY
    assert recovery["quarantine_capacity"] == mod.QUARANTINE_CAPACITY
    assert recovery["max_quarantine_count"] <= mod.QUARANTINE_CAPACITY
    assert recovery["capacity_eviction_count"] > 0
    assert recovery["rollback_hash_matches"] is True
    assert recovery["restart_hash_matches"] is True
    assert recovery["rollback_mismatch_count"] == 0
    assert recovery["restart_mismatch_count"] == 0

    assert weights["all_unchanged"] is True
    assert weights["model_weight_mutation"] is False
    assert weights["before_hashes"] == weights["after_hashes"]
    assert boundary["all_task_owned_commands_clean"] is True
    assert boundary["global_suite_failure_delta"] <= 0
    assert boundary["ready_allowed"] is True
    assert protected["unchanged"] is True
    assert protected["changed_files"] == []
    assert hardware["finite_operation_set"] == list(mod.SUPPORTED_OPERATIONS)
    assert hardware["bounded_active_capacity"] == mod.ACTIVE_CAPACITY
    assert hardware["bounded_quarantine_capacity"] == mod.QUARANTINE_CAPACITY
    assert hardware["hardware_execution_claimed"] is False


def test_req_learn_5924_fail_closed_validation_paths(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5924: bad artifacts and failed commands cannot look ready."""

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm"
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = False
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verifier)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)

    bad_score = deepcopy(artifact)
    bad_score["transactional_memory_fixture_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "blocked: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    retired = deepcopy(artifact)
    retired["gate_replay_receipt"]["retired_exp5912_dependency_used"] = True
    assert mod.status(retired) == "retired"
    assert mod.honest_verdict(retired).startswith("retired:")

    failed_codes = dict(TEST_EXIT_CODES)
    failed_codes[FOCUSED_COMMAND] = 1
    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=failed_codes,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["transactional_memory_fixture_ready_score"] == 0.0
    assert mod.validate_artifact(blocked) is True

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod.read_json(scalar)

    rows = mod.exp5920.load_jsonl(REPO / mod.EXP5920_ROWS_RELATIVE_PATH)
    memory = mod.TransactionalConstraintMemory()
    row = rows[0]
    snapshot_id = memory.snapshot(row)
    with pytest.raises(mod.TransactionMemoryError, match="future label visibility"):
        memory.propose(row, snapshot_id, "model_candidate", {"future_label_visible": True})
    with pytest.raises(mod.TransactionMemoryError, match="stale snapshot"):
        memory.lookup(row, "missing-snapshot", "missing-key")
    with pytest.raises(mod.TransactionMemoryError, match="invalid transition order"):
        memory.commit(row, "missing-proposal")
    proposal_id = memory.propose(row, snapshot_id, "model_candidate", mod._model_payload(row, 0))
    with pytest.raises(mod.TransactionMemoryError, match="invalid transition order"):
        memory.validate(row, proposal_id)
    with pytest.raises(mod.TransactionMemoryError, match="rollback target missing"):
        memory.rollback(row, "sha256:" + "0" * 64)
    serialized = memory.serialize_state()
    serialized["state_hash"] = "sha256:" + "1" * 64
    with pytest.raises(mod.TransactionMemoryError, match="restart state hash mismatch"):
        mod.TransactionalConstraintMemory.from_serialized(serialized)

    no_reject = mod._invalid_case("noop", lambda mem, stream_rows: None, rows)
    assert no_reject["rejected"] is False
    assert (
        mod._ledger_chain_valid(
            "sha256:" + "a" * 64,
            [
                {
                    "previous_state_hash": "sha256:" + "b" * 64,
                    "resulting_state_hash": "sha256:" + "c" * 64,
                }
            ],
        )
        is False
    )
