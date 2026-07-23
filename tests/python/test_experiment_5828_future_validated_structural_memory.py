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


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5828: build the deterministic future-validated artifact once."""

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


def _assert_operation_receipt(receipt: dict[str, Any]) -> None:
    assert receipt["pre_state_hash"].startswith("sha256:")
    assert receipt["post_state_hash"].startswith("sha256:")
    assert receipt["receipt_hash"].startswith("sha256:")
    assert receipt["reason"]


def test_req_learn_5828_spec_declares_future_validated_contract() -> None:
    """REQ-LEARN-5828: OpenSpec names the artifact fields and principles."""

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


def test_req_learn_5828_structured_gate_and_terminal_artifact(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5828: preconditions bind Exp5826, Exp5827, and the contract."""

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
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["structured_gate_replay"]["ok"] is True
    assert artifact["preconditions_checked"]["multiple_change_coverage"]["ok"] is True
    assert artifact["preconditions_checked"]["sealed_future_batches"]["ok"] is True
    assert artifact["preconditions_checked"]["exact_solvers"]["ok"] is True
    assert artifact["preconditions_checked"]["checkpoint_paths"]["checkpoint_atomic_suffix"] == ".tmp"
    assert artifact["upstream_artifact_hashes"]["exp5826_artifact"].startswith("sha256:")
    assert artifact["upstream_artifact_hashes"]["exp5827_artifact"].startswith("sha256:")
    assert artifact["upstream_artifact_hashes"]["exp5825_contract"].startswith("sha256:")
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_learn_5828_future_promotion_uses_sealed_nonreused_suffixes(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5828-FUTURE-PROMOTION: promotion opens only on suffix lift."""

    arms = artifact["arm_definitions_and_parity"]
    ledger = artifact["quarantine_promotion_rollback_ledger"]
    sealed = artifact["sealed_future_validation_receipts"]
    paired = artifact["paired_deltas_and_ci95"]

    assert arms["arms"] == list(mod.CONTROL_ARMS)
    assert arms["parity_passed"] is True
    assert {
        arms["definitions"][arm]["query_budget_per_row"] for arm in mod.CONTROL_ARMS
    } == {mod.QUERY_BUDGET_PER_ROW}
    assert {
        arms["definitions"][arm]["structural_learner"] for arm in mod.CONTROL_ARMS
    } == {mod.STRUCTURAL_LEARNER}
    assert {
        arms["definitions"][arm]["memory_cap"] for arm in mod.CONTROL_ARMS
    } == {mod.MEMORY_CAP}
    assert {
        arms["definitions"][arm]["stopping_rule"] for arm in mod.CONTROL_ARMS
    } == {mod.STOPPING_RULE}

    assert ledger["proposal_count"] == 360
    assert ledger["quarantine_count"] == 360
    assert ledger["promotion_count"] == 360
    assert ledger["false_promotion_count"] == 0
    assert ledger["control_rollback_count"] >= len(mod.PRIMARY_FAMILIES)
    assert ledger["validation_label_reuse_count"] == 0
    assert sealed["sealed_suffix_count"] == 360
    assert sealed["all_future_suffixes_sealed"] is True
    assert sealed["validation_label_reuse_count"] == 0
    assert sealed["pooled"]["future_validated_minus_no_memory"]["ci95"][0] > 0.0
    assert paired["pooled"]["future_validated_minus_no_memory"]["ci95"][0] > 0.0

    for sample in ledger["sample_quarantine_receipts"][:6]:
        _assert_operation_receipt(sample)
        assert sample["operation"] == "quarantine"
        assert sample["parent_state_hash"].startswith("sha256:")
        assert sample["proposal_hash"].startswith("sha256:")
        assert sample["evidence_receipts"]["minimal_core_receipt_hash"].startswith("sha256:")
    for sample in ledger["sample_promotion_receipts"][:6]:
        _assert_operation_receipt(sample)
        assert sample["operation"] == "promote"
        assert sample["promotion_gates"]["all_passed"] is True
        assert sample["promotion_gates"]["positive_paired_lower_bound"] is True


def test_scenario_learn_5828_structural_ops_and_per_family_metrics(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5828-STRUCTURAL-OPS: collisions and nonstationarity are explicit."""

    receipts = artifact["collision_supersession_recurrence_receipts"]
    metrics = artifact["per_family_change_metrics"]
    paired = artifact["paired_deltas_and_ci95"]

    assert receipts["collision_split_count"] >= len(mod.PRIMARY_FAMILIES)
    assert receipts["supersession_count"] >= len(mod.PRIMARY_FAMILIES)
    assert receipts["recurrence_reactivation_count"] >= len(mod.PRIMARY_FAMILIES)
    for family in mod.PRIMARY_FAMILIES:
        assert receipts["by_family"][family]["collision_split_count"] > 0
        assert receipts["by_family"][family]["supersession_count"] > 0
        assert receipts["by_family"][family]["recurrence_reactivation_count"] > 0
        assert paired["family"][family]["future_validated_minus_no_memory"]["ci95"][0] > 0.0
        assert paired["family"][family]["no_family_harm"] is True
        for change in mod.CHANGE_ORDER:
            cell = metrics[family][change]
            assert cell["row_count"] == 30
            assert cell["future_validated"]["future_suffix_exact_accuracy"] == pytest.approx(1.0)
            assert cell["future_validated"]["promotion_precision"] >= 0.95
            assert cell["future_validated"]["promotion_recall"] == pytest.approx(1.0)
            assert cell["future_validated"]["false_promotion_count"] == 0
            assert cell["future_validated"]["protected_prefix_retention"] == pytest.approx(1.0)
            assert cell["future_validated"]["unsafe_propagation_count"] == 0
            assert cell["future_validated"]["rollback_fidelity"] == pytest.approx(1.0)
            assert cell["future_validated"]["memory_growth"] <= mod.MEMORY_CAP
            assert cell["future_validated"]["recurrence_recovery"] in {0.0, 1.0}
            assert cell["future_validated"]["future_suffix_exact_accuracy"] > cell["no_adaptive_memory"][
                "future_suffix_exact_accuracy"
            ]

    for group in (
        receipts["sample_collision_split_receipts"],
        receipts["sample_supersession_receipts"],
        receipts["sample_recurrence_reactivation_receipts"],
    ):
        for sample in group[:6]:
            _assert_operation_receipt(sample)


def test_scenario_learn_5828_restart_rollback_retention_and_cap_are_exact(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5828-RESTART-CAP: rollback, restart, retention, and cap pass."""

    restart = artifact["restart_equivalence"]
    cap = artifact["memory_cap_receipts"]
    ledger = artifact["quarantine_promotion_rollback_ledger"]

    assert artifact["protected_prefix_retention"] == pytest.approx(1.0)
    assert artifact["unsafe_update_count"] == 0
    assert artifact["rollback_hash_mismatch_count"] == 0
    assert restart["restart_equivalence"] == pytest.approx(1.0)
    assert restart["full_state_hash"] == restart["resumed_state_hash"]
    assert restart["full_event_hash"] == restart["resumed_event_hash"]
    assert len(restart["interruption_boundaries"]) >= 3
    assert cap["cap_compliance"] == pytest.approx(1.0)
    assert cap["max_memory_size"] <= mod.MEMORY_CAP
    assert cap["eviction_count"] > 0
    assert cap["sample_eviction_receipts"]
    for sample in cap["sample_eviction_receipts"][:6]:
        _assert_operation_receipt(sample)
        assert sample["operation"] == "evict_quarantine_receipt"
    assert ledger["rollback_fidelity"] == pytest.approx(1.0)
    for sample in ledger["sample_rollback_receipts"][:6]:
        _assert_operation_receipt(sample)
        assert sample["operation"] == "rollback"
        assert sample["post_state_hash"] == sample["restored_state_hash"]


def test_scenario_learn_5828_fail_closed_for_bad_gates(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5828-FAIL-CLOSED: bad gates cannot look ready."""

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
        (
            lambda item: item["paired_deltas_and_ci95"]["pooled"]["future_validated_minus_no_memory"].update(
                {"ci95": [0.0, 0.1]}
            ),
            "future_validated_lifecycle_ready_score",
        ),
        (
            lambda item: item["quarantine_promotion_rollback_ledger"].update(
                {"promotion_precision": 0.9}
            ),
            "future_validated_lifecycle_ready_score",
        ),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
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


def test_req_learn_5828_low_level_helpers_and_schema_edges(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5828: helper edges remain deterministic and fail closed."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)
    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")
    scalar_jsonl = tmp_path / "scalar.jsonl"
    scalar_jsonl.write_text("1\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == [{"ok": True}]
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(scalar_jsonl)
    assert mod.read_row_file(tmp_path / "missing.rows.jsonl") == []
    assert mod.fixture_preconditions()["preconditions_ready"] is True

    assert mod._bootstrap_ci95([]) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25]) == [0.25, 0.25]
    assert mod._paired_summary([0.2, 0.4])["ci95"][0] > 0.0
    assert mod._future_suffix_labels({"exact_receipt": {"primary": {}}}, []) == {}
    assert mod._future_accuracy({}, {}) == 0.0

    rows = mod.read_row_file(REPO / mod.EXP5826_ROWS_RELATIVE_PATH)
    row = rows[0]
    proposal = mod._proposal_from_row(row)
    assert proposal["proposal_hash"].startswith("sha256:")
    assert mod._validation_label_reuse_count(row) == 0
    validation = mod._future_validation_receipt(row, proposal)
    assert validation["all_passed"] is True
    assert validation["future_validated_accuracy"] == pytest.approx(1.0)

    tampered = deepcopy(row)
    tampered["sealed_future_suffix"]["suffix_hash"] = mod.sha256_text("tamper")
    bad_validation = mod._future_validation_receipt(tampered, proposal)
    assert bad_validation["all_passed"] is False
    assert bad_validation["suffix_hash_ok"] is False

    stream_artifact = mod._read_json(REPO / mod.EXP5826_ARTIFACT_RELATIVE_PATH)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod.exp5826,
            "verify_row_file",
            lambda rows_arg, artifact_arg: (_ for _ in ()).throw(
                mod.exp5826.StreamReplayError("bad")
            ),
        )
        replay_receipt = mod._row_replay_receipt(rows, stream_artifact)
    assert replay_receipt["replay_ok"] is False

    state = mod._initial_state()
    pre = mod._state_hash(state)
    receipt = mod._state_transition(
        state,
        operation="probe",
        reason="coverage edge",
        payload={"ok": True},
        mutate=lambda memory: memory.update({"sequence": 1}),
    )
    assert receipt["pre_state_hash"] == pre
    assert receipt["post_state_hash"] == mod._state_hash(state)

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

    blocker_probe = deepcopy(artifact)
    blocker_probe["model_weight_mutation"] = True
    blocker_probe["inference_substrate"] = "wrong"
    blocker_probe["verifier_is_oracle"] = False
    blocker_probe["quarantine_promotion_rollback_ledger"]["false_promotion_count"] = 1
    blocker_probe["sealed_future_validation_receipts"]["validation_label_reuse_count"] = 1
    blocker_probe["unsafe_update_count"] = 1
    blocker_probe["rollback_hash_mismatch_count"] = 1
    assert set(mod.blocked_reasons(blocker_probe)) >= {
        "model_weight_mutation",
        "inference_substrate",
        "verifier_is_oracle",
        "false_promotion_count",
        "validation_label_reuse_count",
        "unsafe_update_count",
        "rollback_hash_mismatch_count",
    }

    generic_score_probe = deepcopy(artifact)
    generic_score_probe["arm_definitions_and_parity"]["parity_passed"] = False
    assert "future_validated_lifecycle_ready_score" in mod.blocked_reasons(
        generic_score_probe
    )
