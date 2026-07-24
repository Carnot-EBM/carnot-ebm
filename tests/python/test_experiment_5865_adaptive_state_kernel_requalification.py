"""Exp5865 adaptive-state requalification artifact tests.

Spec refs: REQ-LEARN-5865, SCENARIO-LEARN-5865-ATTRIBUTION,
SCENARIO-LEARN-5865-PARITY-PRESERVED.
"""

from __future__ import annotations

from pathlib import Path

from carnot import adaptive_state as base
from carnot import adaptive_state_requalification as req


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5859_adaptive_state_microkernel_parity.py "
    "-q --no-cov -n 0"
)


def _passing_parity_receipt() -> dict[str, object]:
    return {
        "binding_receipt": {
            "binding_available": True,
            "class_name": "RustAdaptiveStateKernel",
            "methods": list(base.ABI_OPERATIONS),
        },
        "canonical_state_and_hash_parity": {
            "canonical_form_parity": True,
            "hash_parity": True,
            "py_final_hashes": ["sha256:" + "1" * 64],
            "rust_final_hashes": ["sha256:" + "1" * 64],
        },
        "cross_language_operation_parity": {
            "accept_reject_parity": True,
            "operation_count": 3,
            "parity_failures": [],
            "trace_count": 1,
        },
        "invalid_input_and_capacity_controls": {
            "accepted_control_event": True,
            "capacity": 2,
            "fail_closed": True,
            "invalid_case_count": 1,
            "receipts": [
                {
                    "case": "duplicate_event",
                    "expected_code": "DUPLICATE_EVENT",
                    "observed_code": "DUPLICATE_EVENT",
                    "state_preserved": True,
                }
            ],
        },
        "serialization_restart_and_rollback_parity": {
            "restart_parity": True,
            "rollback_parity": True,
            "round_trip_parity": True,
        },
    }


def _base_reproduction_receipt() -> dict[str, object]:
    return {
        "command": FULL_PYTEST_COMMAND,
        "collection_items": 50873,
        "log_path": "/tmp/exp5865_pytest_before.log",
        "log_sha256": "sha256:" + "2" * 64,
        "natural_exit_reached": False,
        "observed_exit_code": None,
        "recorded_exp5859_exit_code": 2,
        "wrapper_exit_code_after_interrupt": 0,
    }


def _global_debt_receipt() -> dict[str, object]:
    return {
        "blocking": True,
        "classification": "unrelated_repository_debt",
        "owned_by_adaptive_state": False,
        "receipts": [
            {
                "classification": "unrelated_repository_debt",
                "nodeid": (
                    "tests/python/test_experiment_3494.py::"
                    "test_run_p01_gate_required_fields_present"
                ),
                "reason": "Fatal Python abort while JAX compiled phase3 sudoku gate.",
            }
        ],
    }


def test_req_learn_5865_spec_declares_requalification_contract() -> None:
    """REQ-LEARN-5865: OpenSpec preregisters the requalification artifact."""

    section = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = section[section.index("## REQ-LEARN-5865") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5865",
        "SCENARIO-LEARN-5865-ATTRIBUTION",
        "SCENARIO-LEARN-5865-PARITY-PRESERVED",
        "python/carnot/adaptive_state_requalification.py",
        "results/experiment_5865_adaptive_state_kernel_requalification.json",
        "`adaptive_state_microkernel_requalified_score`",
        "deterministic_cross_language_state_execution_no_llm",
    ):
        assert marker in section
    for field, principle in req.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5865_retired_artifact_for_unrelated_global_debt(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5865-ATTRIBUTION: unrelated global debt blocks score 1.0."""

    result_path = tmp_path / req.RESULT_RELATIVE_PATH.name
    test_exit_codes = {FOCUSED_COMMAND: 0, FULL_PYTEST_COMMAND: 2}
    artifact = req.build_artifact(
        result_path=result_path,
        duration_s=12.5,
        test_commands=list(test_exit_codes),
        test_exit_codes=test_exit_codes,
        original_nonzero_exit_reproduction=_base_reproduction_receipt(),
        failing_collection_and_node_receipts=_global_debt_receipt()["receipts"],
        global_suite_debt_classification=_global_debt_receipt(),
        applicable_e2e_receipts={"e2e_003": {"command": "binding", "exit_code": 0}},
        before_after_test_matrix={"focused_pytest": {"before": 0, "after": 0}},
        parity_receipt=_passing_parity_receipt(),
        write=True,
    )

    assert req.read_json(result_path) == artifact
    assert req.validate_artifact(artifact) is True
    assert set(req.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["adaptive_state_microkernel_requalified_score"] == 0.0
    assert artifact["semantic_change_scope"]["code_semantics_changed"] is False
    assert artifact["root_cause_classification"]["primary"] == "unrelated_repository_debt"
    assert artifact["global_suite_debt_classification"]["blocking"] is True
    assert artifact["cross_language_operation_parity"]["accept_reject_parity"] is True
    assert artifact["reproducibility_checksum"] == req.reproducibility_checksum(artifact)


def test_scenario_learn_5865_score_requires_zero_exits_and_no_global_blocker(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5865-PARITY-PRESERVED: score 1.0 requires clean commands."""

    test_exit_codes = {FOCUSED_COMMAND: 0, FULL_PYTEST_COMMAND: 0}
    artifact = req.build_artifact(
        result_path=tmp_path / req.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_commands=list(test_exit_codes),
        test_exit_codes=test_exit_codes,
        original_nonzero_exit_reproduction={
            **_base_reproduction_receipt(),
            "observed_exit_code": 0,
        },
        failing_collection_and_node_receipts=[],
        global_suite_debt_classification={"blocking": False, "classification": "none"},
        applicable_e2e_receipts={"e2e_003": {"command": "binding", "exit_code": 0}},
        before_after_test_matrix={"focused_pytest": {"before": 0, "after": 0}},
        parity_receipt=_passing_parity_receipt(),
        write=False,
    )

    assert artifact["status"] == "requalified"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["adaptive_state_microkernel_requalified_score"] == 1.0


def test_req_learn_5865_exp5859_input_receipt_is_hash_bound() -> None:
    """REQ-LEARN-5865: Exp5859 is consumed as immutable blocked input."""

    receipt = req.exp5859_input_receipt()

    assert receipt["path"] == base.RESULT_RELATIVE_PATH.as_posix()
    assert receipt["sha256"].startswith("sha256:")
    assert receipt["status"] == "blocked"
    assert receipt["adaptive_state_microkernel_ready_score"] == 0.0
    assert receipt["recorded_full_suite_exit_code"] == 2
    assert receipt["checksum_valid"] is True


def test_req_learn_5865_bookkeeping_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-5865: protected conflicts and E2E failures block promotion."""

    existing_result = tmp_path / req.RESULT_RELATIVE_PATH.name
    existing_result.write_text("prior", encoding="utf-8")
    test_exit_codes = {FOCUSED_COMMAND: 0, FULL_PYTEST_COMMAND: 0}
    artifact = req.build_artifact(
        result_path=existing_result,
        duration_s=1.0,
        test_commands=list(test_exit_codes),
        test_exit_codes=test_exit_codes,
        original_nonzero_exit_reproduction=_base_reproduction_receipt(),
        failing_collection_and_node_receipts=[],
        global_suite_debt_classification={
            "blocking": False,
            "classification": "none",
            "protected_user_work_conflict": True,
        },
        applicable_e2e_receipts={"e2e_003": {"command": "binding", "exit_code": 1}},
        before_after_test_matrix={"focused_pytest": {"before": 0, "after": 0}},
        parity_receipt=_passing_parity_receipt(),
        write=False,
    )
    protected = req._protected_files_unchanged()

    assert protected["unchanged"] is True
    assert artifact["field_provenance"]["source_hashes"]["output_path"] == str(existing_result)
    assert artifact["root_cause_classification"]["primary"] == "protected_user_work_conflict"
    assert artifact["adaptive_state_microkernel_requalified_score"] == 0.0
