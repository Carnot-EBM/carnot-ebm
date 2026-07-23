"""Tests for Exp5828 future-validated structural memory.

Spec refs: REQ-LEARN-5828, SCENARIO-LEARN-5828-FUTURE-PROMOTION,
SCENARIO-LEARN-5828-STRUCTURAL-OPS, SCENARIO-LEARN-5828-RESTART-CAP,
SCENARIO-LEARN-5828-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5828_future_validated_structural_memory as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5828_future_validated_structural_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5828_future_validated_structural_memory.py "
    "-m pytest tests/python/test_experiment_5828_future_validated_structural_memory.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5828_future_validated_structural_memory.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5828_future_validated_structural_memory.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": 512, "ok": True},
    )


def _walk(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.keys()) + [item for sub in value.values() for item in _walk(sub)]
    if isinstance(value, list):
        return [item for sub in value for item in _walk(sub)]
    return [value]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5828: build the deterministic lifecycle artifact once."""

    base = tmp_path_factory.mktemp("exp5828")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=base / "checkpoints",
        preconditions_checked=_preconditions(base),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5828_spec_declares_required_contract() -> None:
    """REQ-LEARN-5828: OpenSpec names fields, principles, and scenarios."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5828") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5828",
        "SCENARIO-LEARN-5828-FUTURE-PROMOTION",
        "SCENARIO-LEARN-5828-STRUCTURAL-OPS",
        "SCENARIO-LEARN-5828-RESTART-CAP",
        "SCENARIO-LEARN-5828-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`future_validated_lifecycle_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5828_artifact_is_terminal_hash_bound_and_immutable(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5828: artifact is complete, reproducible, and model-frozen."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        checkpoint_dir=tmp_path / "ckpt",
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert artifact == replay == loaded
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["future_validated_lifecycle_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["future_validated_lifecycle_ready_score"], float)
    assert artifact["model_weight_mutation"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    preconditions = artifact["preconditions_checked"]
    assert preconditions["preconditions_ready"] is True
    assert preconditions["structured_gate_replay"]["ok"] is True
    assert preconditions["structured_gate_replay"]["exp5827_structural_learner_ready_score"] == 1.0
    assert preconditions["multiple_change_coverage"]["ok"] is True
    assert preconditions["multiple_change_coverage"]["minimum_rows_per_family_change"] >= 30
    assert preconditions["sealed_future_batches"]["ok"] is True
    assert preconditions["exact_solvers"]["ok"] is True
    assert preconditions["output_paths"]["atomic_checkpoint_suffix"] == ".tmp"
    assert preconditions["llm_calls_made"] == 0


def test_scenario_learn_5828_future_promotion_uses_parity_and_sealed_suffixes(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5828-FUTURE-PROMOTION: promotions require future evidence."""

    arms = artifact["arm_definitions_and_parity"]
    assert set(arms["arms"]) == set(mod.CONTROL_ARMS)
    assert arms["arm_parity_passed"] is True
    assert arms["future_validated_arm"] == mod.FUTURE_VALIDATED_ARM
    assert arms["science_labels_assigned_after_arm_freeze"] is True

    budgets = {arms["definitions"][arm]["query_budget_per_row"] for arm in mod.CONTROL_ARMS}
    learners = {arms["definitions"][arm]["structure_learner_hash"] for arm in mod.CONTROL_ARMS}
    caps = {arms["definitions"][arm]["memory_cap_entries"] for arm in mod.CONTROL_ARMS}
    stopping_rules = {arms["definitions"][arm]["stopping_rule"] for arm in mod.CONTROL_ARMS}
    assert budgets == {mod.QUERY_BUDGET_PER_ROW}
    assert len(learners) == 1
    assert caps == {mod.MEMORY_CAP_ENTRIES}
    assert stopping_rules == {mod.STOPPING_RULE}

    validation = artifact["sealed_future_validation_receipts"]
    assert validation["future_suffix_count"] == 360
    assert validation["promoted_count"] == 360
    assert validation["validation_label_reuse_count"] == 0
    assert validation["future_label_leakage_count"] == 0
    assert validation["promotion_precision"] >= 0.95
    assert validation["promotion_recall"] == pytest.approx(1.0)
    assert validation["false_promotion_count"] == 0
    assert validation["all_future_batches_positive_lcb"] is True
    assert validation["all_promotions_passed_gates"] is True
    assert all(
        receipt["validation_label_reuse_count"] == 0
        for receipt in validation["sample_receipts"]
    )
    flattened = {str(item) for item in _walk(validation["sample_receipts"])}
    assert "future_labels_visible_to_learner_before_validation" in flattened
    assert "ground_truth_structure" not in flattened


def test_scenario_learn_5828_structural_operations_are_transactional(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5828-STRUCTURAL-OPS: every memory edit is receipt-backed."""

    ledger = artifact["quarantine_promotion_rollback_ledger"]
    structural = artifact["collision_supersession_recurrence_receipts"]

    assert ledger["quarantine_count"] == 360
    assert ledger["promotion_count"] == 360
    assert ledger["rollback_count"] >= 3
    assert ledger["rollback_hash_mismatch_count"] == 0
    assert ledger["transactional_replayable"] is True
    assert ledger["ledger_hash"].startswith("sha256:")
    assert set(ledger["quarantined_proposal_ids"]) == set(ledger["promoted_proposal_ids"])
    for receipt in (
        ledger["quarantine_receipts"][:3]
        + ledger["promotion_receipts"][:3]
        + ledger["rollback_receipts"][:3]
    ):
        assert receipt["pre_state_hash"].startswith("sha256:")
        assert receipt["post_state_hash"].startswith("sha256:")
        assert receipt["reason"]
        assert receipt["receipt_hash"].startswith("sha256:")

    assert structural["collision_split_count"] > 0
    assert structural["supersession_count"] == 120
    assert structural["recurrence_reactivation_count"] == 120
    assert structural["recurrence_recovery"] == pytest.approx(1.0)
    for receipt in (
        structural["collision_split_receipts"][:3]
        + structural["supersession_receipts"][:3]
        + structural["recurrence_reactivation_receipts"][:3]
    ):
        assert receipt["pre_state_hash"].startswith("sha256:")
        assert receipt["post_state_hash"].startswith("sha256:")
        assert receipt["reason"]


def test_scenario_learn_5828_restart_cap_and_metrics_gate_readiness(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5828-RESTART-CAP: restart, cap, safety, and CI gates pass."""

    paired = artifact["paired_deltas_and_ci95"]
    metrics = artifact["per_family_change_metrics"]

    assert paired["pooled"]["future_validated_minus_no_memory"]["n"] >= 30
    assert paired["pooled"]["future_validated_minus_no_memory"]["ci95"][0] > 0.0
    assert paired["family_harm_count"] == 0
    assert paired["no_family_harm"] is True
    assert artifact["protected_prefix_retention"] == pytest.approx(1.0)
    assert artifact["unsafe_update_count"] == 0
    assert artifact["rollback_hash_mismatch_count"] == 0
    assert artifact["restart_equivalence"]["restart_equivalence"] == pytest.approx(1.0)
    assert artifact["restart_equivalence"]["full_state_hash"] == artifact["restart_equivalence"]["resumed_state_hash"]
    assert artifact["restart_equivalence"]["full_event_hash"] == artifact["restart_equivalence"]["resumed_event_hash"]
    assert len(artifact["restart_equivalence"]["interruption_boundaries"]) >= 3
    assert artifact["memory_cap_receipts"]["cap_compliance"] == pytest.approx(1.0)
    assert artifact["memory_cap_receipts"]["max_entry_count"] <= mod.MEMORY_CAP_ENTRIES
    assert artifact["memory_cap_receipts"]["eviction_count"] > 0

    for family in mod.PRIMARY_FAMILIES:
        assert paired["family"][family]["future_validated_minus_no_memory"]["ci95"][0] > 0.0
        for change in mod.CHANGE_ORDER:
            cell = metrics[family][change]
            assert cell["row_count"] == 30
            assert cell["future_suffix_exact_accuracy"][mod.FUTURE_VALIDATED_ARM] == pytest.approx(1.0)
            assert (
                cell["future_suffix_exact_accuracy"][mod.FUTURE_VALIDATED_ARM]
                > cell["future_suffix_exact_accuracy"][mod.NO_MEMORY_ARM]
            )
            assert cell["promotion_precision"] >= 0.95
            assert cell["false_promotion_count"] == 0
            assert cell["unsafe_propagation_count"] == 0
            assert cell["protected_prefix_retention"] == pytest.approx(1.0)
            assert cell["dynamic_regret"][mod.FUTURE_VALIDATED_ARM] < cell["dynamic_regret"][mod.NO_MEMORY_ARM]


def test_scenario_learn_5828_fail_closed_for_bad_gates(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5828-FAIL-CLOSED: unsafe or stale evidence cannot look ready."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["future_validated_lifecycle_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "missing_upstream_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_exits = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
    )
    assert failed_exits["status"] == "blocked"
    assert failed_exits["future_validated_lifecycle_ready_score"] == 0.0
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_exits)

    for mutate, match in (
        (lambda item: item.update({"model_weight_mutation": True}), "model_weight_mutation"),
        (lambda item: item.update({"inference_substrate": "live_llm"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"unsafe_update_count": 1}), "future_validated_lifecycle_ready_score"),
        (lambda item: item.update({"rollback_hash_mismatch_count": 1}), "future_validated_lifecycle_ready_score"),
        (
            lambda item: item["restart_equivalence"].update({"restart_equivalence": 0.0}),
            "future_validated_lifecycle_ready_score",
        ),
        (
            lambda item: item["memory_cap_receipts"].update({"cap_compliance": 0.0}),
            "future_validated_lifecycle_ready_score",
        ),
        (
            lambda item: item["sealed_future_validation_receipts"].update(
                {"validation_label_reuse_count": 1}
            ),
            "future_validated_lifecycle_ready_score",
        ),
        (lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}), "reproducibility_checksum"),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    monkeypatch.setattr(
        mod,
        "_read_json",
        lambda path: (_ for _ in ()).throw(ValueError("corrupt")),
    )
    corrupt = mod.collect_preconditions(
        result_path=tmp_path / "corrupt.json",
        checkpoint_dir=tmp_path / "corrupt-ckpt",
        memory_probe=lambda: {"available_mb": 0, "required_mb": 512, "ok": False},
        disk_probe=lambda root: {"available_mb": 0, "required_mb": 512, "ok": False},
    )
    assert set(corrupt["blocked_reasons"]) >= {
        "corrupt_upstream_artifact",
        "insufficient_free_ram",
        "insufficient_free_disk",
    }

    reason_probe = deepcopy(artifact)
    reason_probe["model_weight_mutation"] = True
    reason_probe["inference_substrate"] = "wrong"
    reason_probe["verifier_is_oracle"] = False
    reason_probe["sealed_future_validation_receipts"]["validation_label_reuse_count"] = 1
    reason_probe["paired_deltas_and_ci95"]["no_family_harm"] = False
    reason_probe["unsafe_update_count"] = 1
    reason_probe["rollback_hash_mismatch_count"] = 1
    reason_probe["restart_equivalence"]["restart_equivalence"] = 0.0
    reason_probe["memory_cap_receipts"]["cap_compliance"] = 0.0
    assert set(mod.blocked_reasons(reason_probe)) >= {
        "model_weight_mutation",
        "inference_substrate",
        "verifier_is_oracle",
        "validation_label_reuse_count",
        "family_harm",
        "unsafe_update_count",
        "rollback_hash_mismatch_count",
        "restart_equivalence",
        "memory_cap_compliance",
    }

    generic_probe = deepcopy(artifact)
    generic_probe["arm_definitions_and_parity"]["arm_parity_passed"] = False
    assert mod.blocked_reasons(generic_probe) == ["future_validated_lifecycle_ready_score"]


def test_req_learn_5828_helper_edges_and_schema_validation(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5828: helper edge cases are deterministic and fail closed."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)

    assert mod.read_rows(tmp_path / "missing.rows.jsonl") == []
    assert mod.fixture_preconditions()["preconditions_ready"] is True
    assert mod._paired_summary([]) == {"n": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0]}
    assert mod._paired_summary([0.25]) == {"n": 1, "mean_delta": 0.25, "ci95": [0.25, 0.25]}
    assert mod._state_hash(mod._empty_memory_state()).startswith("sha256:")
    assert mod._exact_solver_receipt([])["ok"] is False
    assert mod._accuracy({}, {}) == 0.0

    rows = mod.read_rows(REPO / mod.EXP5826_ROWS_RELATIVE_PATH)
    proposal = mod._proposal_from_row(rows[0])
    validation = mod._future_validation_receipt(rows[0], proposal)
    assert validation["validation_label_reuse_count"] == 0
    assert validation["promote"] is True
    assert validation["future_batch_lcb95"] > 0.0

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    invalid_provenance_shape = deepcopy(artifact)
    invalid_provenance_shape["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(invalid_provenance_shape)

    invalid_provenance_principle = deepcopy(artifact)
    invalid_provenance_principle["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(invalid_provenance_principle)

    invalid_status = deepcopy(artifact)
    invalid_status["status"] = "blocked"
    invalid_status["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(invalid_status)

    invalid_complete_verdict = deepcopy(artifact)
    invalid_complete_verdict["honest_verdict"] = "blocked: forced"
    invalid_complete_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        invalid_complete_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_complete_verdict)

    invalid_blocked_verdict = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "blocked-verdict"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
    )
    invalid_blocked_verdict["honest_verdict"] = "ready"
    invalid_blocked_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        invalid_blocked_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_blocked_verdict)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    no_write = mod.run(
        result_path=tmp_path / "no-write.json",
        checkpoint_dir=tmp_path / "no-write-ckpt",
        preconditions_checked=_preconditions(tmp_path / "no-write"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert no_write["status"] == "complete"
    assert not (tmp_path / "no-write.json").exists()
