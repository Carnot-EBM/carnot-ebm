"""Tests for Exp5892 headroom evidence escrow.

Spec refs: REQ-VERIFY-5892, SCENARIO-VERIFY-5892-IMMUTABLE-REPLAY,
SCENARIO-VERIFY-5892-ADMISSION-BOUNDARY,
SCENARIO-VERIFY-5892-NON-INTERFERENCE, SCENARIO-VERIFY-5892-FRESHNESS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5892_headroom_evidence_escrow as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5892_headroom_evidence_escrow.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5892_headroom_evidence_escrow.py "
    "-m pytest tests/python/test_experiment_5892_headroom_evidence_escrow.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5892_headroom_evidence_escrow.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5892_headroom_evidence_escrow.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5892_headroom_evidence_escrow.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {
    FOCUSED_TEST_COMMAND: 0,
    COVERAGE_COMMAND: 0,
    FULL_TEST_COMMAND: 2,
    SPEC_COMMAND: 0,
    ADVERSARIAL_COMMAND: 0,
    ROOT_CLUTTER_COMMAND: 0,
    PROTECTED_FILE_COMMAND: 0,
}
FULL_TEST_OUTPUT = (
    "FAILED tests/python/test_experiment_5890_transition_v524.py::"
    "test_preexisting_transition_debt - AssertionError\n"
)


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 32768, "required_mb": 1024, "ok": True},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 512, "ok": True},
    )


@pytest.fixture(scope="module")
def exp5892_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], Path]:
    """REQ-VERIFY-5892: build the admission escrow artifact once."""

    base = tmp_path_factory.mktemp("exp5892")
    before_hashes = {
        path.as_posix(): mod.sha256_file(REPO / path)
        for path in mod.PROTECTED_RELATIVE_PATHS
    }
    artifact = mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        check_outputs={FULL_TEST_COMMAND: FULL_TEST_OUTPUT},
        duration_s=7.0,
        write=True,
    )
    assert before_hashes == {
        path.as_posix(): mod.sha256_file(REPO / path)
        for path in mod.PROTECTED_RELATIVE_PATHS
    }
    return artifact, base


def test_req_verify_5892_spec_declares_evidence_escrow_contract() -> None:
    """REQ-VERIFY-5892: OpenSpec anchors every field and principle."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5892") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5892",
        "SCENARIO-VERIFY-5892-IMMUTABLE-REPLAY",
        "SCENARIO-VERIFY-5892-ADMISSION-BOUNDARY",
        "SCENARIO-VERIFY-5892-NON-INTERFERENCE",
        "SCENARIO-VERIFY-5892-FRESHNESS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`headroom_admission_ready_score`",
        "`unrelated_global_debt_receipts`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for principle in mod.REQUIRED_FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized


def test_req_verify_5892_terminal_artifact_is_ready_and_hash_bound(
    exp5892_artifact: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-5892: terminal JSON is fresh, complete, and non-retired."""

    artifact, base = exp5892_artifact
    rerun = mod.run(
        result_path=base / "rerun.json",
        preconditions_checked=_preconditions(base / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        check_outputs={FULL_TEST_COMMAND: FULL_TEST_OUTPUT},
        duration_s=99.0,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["headroom_admission_ready_score"] == 1.0
    assert artifact["duration_s"] == pytest.approx(7.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["test_exit_codes"][FULL_TEST_COMMAND] == 2
    assert artifact["owned_check_receipts"]["owned_checks_passed"] is True
    assert artifact["unrelated_global_debt_receipts"]["unrelated_failures_present"] is True
    assert artifact["gate_non_interference_receipts"]["all_unrelated_failures_safe"] is True
    assert artifact["terminal_artifact_freshness_receipt"]["freshness_ready"] is True
    assert artifact["terminal_artifact_freshness_receipt"]["bootstrap_running_artifact_written"] is False
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_5892_replays_rows_taxonomy_splits_and_oracle_boundary(
    exp5892_artifact: tuple[dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-5892-IMMUTABLE-REPLAY: all row predicates replay."""

    artifact, _base = exp5892_artifact
    replay = artifact["independent_row_and_certificate_replay"]
    boundary = artifact["taxonomy_and_oracle_boundary_receipt"]
    splits = artifact["leakage_safe_split_receipts"]
    non_oracle = artifact["non_oracle_nuisance_metrics"]
    oracle = artifact["oracle_derived_diagnostic_metrics"]

    assert replay["row_count"] == 84
    assert replay["label_counts"] == {"satisfiable": 42, "unsatisfiable": 42}
    assert replay["all_row_and_certificate_replay_passed"] is True
    assert replay["witness_replay"]["certificate_failure_count"] == 0
    assert replay["semantic_group_count"] > 0
    assert replay["relabel_stability_rate"] == 1.0
    assert replay["certificate_stability_rate"] == 1.0
    assert splits["all_splits_leakage_safe"] is True
    assert splits["duplicate_semantic_instances_across_splits"] == []

    assert boundary["exp5879_status_observed"] == "blocked"
    assert boundary["exp5879_status_is_admission_gate"] is False
    assert boundary["oracle_telemetry_separately_flagged_circular"] is True
    assert boundary["control_taxonomy"]["all_features_assigned_once"] is True
    assert boundary["control_taxonomy"]["class_overlap"] == []
    assert non_oracle["verifier_is_oracle"] is False
    assert non_oracle["max_non_oracle_nuisance_auroc"] == pytest.approx(0.583333)
    assert non_oracle["no_non_oracle_nuisance_control_exceeds_ceiling"] is True
    assert non_oracle["saturated_control_names"] == []
    assert oracle["verifier_is_oracle"] is True
    assert oracle["max_oracle_derived_auroc"] == pytest.approx(1.0)
    assert oracle["counts_as_learned_energy_win"] is False
    assert oracle["reduces_oracle_distinct_headroom"] is False


def test_scenario_verify_5892_global_debt_requires_exact_non_interference(
    exp5892_artifact: tuple[dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-5892-NON-INTERFERENCE: unrelated failures need nodes."""

    artifact, _base = exp5892_artifact
    failures = artifact["unrelated_global_debt_receipts"]["failure_receipts"]
    receipts = artifact["gate_non_interference_receipts"]["receipts"]

    assert len(failures) == 1
    assert failures[0]["command"] == FULL_TEST_COMMAND
    assert failures[0]["node_id"] == (
        "tests/python/test_experiment_5890_transition_v524.py::"
        "test_preexisting_transition_debt"
    )
    assert failures[0]["path_owner"] == "unrelated_global_suite"
    assert failures[0]["owner_path"] == "tests/python/test_experiment_5890_transition_v524.py"
    assert failures[0]["exit_code"] == 2
    assert failures[0]["can_alter_rows"] is False
    assert failures[0]["can_alter_audit_computations"] is False
    assert failures[0]["can_alter_schemas"] is False
    assert failures[0]["can_alter_gate_fields"] is False
    assert all(receipt["non_interference_passed"] for receipt in receipts)


def test_scenario_verify_5892_defensive_validation_branches(
    tmp_path: Path,
    exp5892_artifact: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-5892: missing receipts or owned failures fail closed."""

    artifact, _base = exp5892_artifact

    missing_node = mod.run(
        result_path=tmp_path / "missing-node" / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path / "missing-node"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        check_outputs={},
        duration_s=1.0,
        write=True,
    )
    assert missing_node["status"] == "blocked"
    assert missing_node["headroom_admission_ready_score"] == 0.0
    assert missing_node["unrelated_global_debt_receipts"]["missing_exact_node_path_evidence"] is True
    assert mod.validate_artifact(missing_node) is True

    owned_failed_codes = dict(TEST_EXIT_CODES)
    owned_failed_codes[FOCUSED_TEST_COMMAND] = 1
    owned_failed = mod.run(
        result_path=tmp_path / "owned-failed" / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path / "owned-failed"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=owned_failed_codes,
        check_outputs={FULL_TEST_COMMAND: FULL_TEST_OUTPUT},
        duration_s=1.0,
        write=True,
    )
    assert owned_failed["status"] == "blocked"
    assert owned_failed["owned_check_receipts"]["owned_checks_passed"] is False
    assert owned_failed["owned_check_receipts"]["failure_receipts"][0]["can_alter_gate_fields"] is True
    assert mod.validate_artifact(owned_failed) is True

    bad_preconditions = mod.collect_preconditions(
        root=tmp_path / "missing-root",
        result_path=tmp_path / "missing-root" / mod.RESULT_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 1, "required_mb": 1024, "ok": False},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 512, "ok": True},
    )
    blocked = mod.run(
        root=tmp_path / "missing-root",
        result_path=tmp_path / "blocked" / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=bad_preconditions,
        test_commands=TEST_COMMANDS,
        test_exit_codes={command: 0 for command in TEST_COMMANDS},
        duration_s=1.0,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    assert mod.extract_pytest_nodes("no pytest node here") == []
    crash_nodes = mod.extract_pytest_nodes(
        'Fatal Python error: Segmentation fault\n  File "/repo/carnot/'
        'python/carnot/verify/z3_math_verifier.py", line 64 in score\n'
    )
    assert crash_nodes == ["python/carnot/verify/z3_math_verifier.py::segmentation_fault"]
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")
    assert mod._path_stats(corrupt)["status"] == "unreadable_json"
    assert mod._command_node_id(FULL_TEST_COMMAND) == "global-python-suite"
    assert mod._command_node_id(SPEC_COMMAND) == "exp5892-spec-coverage"
    assert mod._command_node_id("custom check").startswith("command:")
    assert mod._command_owner_path(FULL_TEST_COMMAND) == "tests/python"
    assert mod._command_owner_path("custom check") == "unknown"

    full_suite_not_run = mod.unrelated_global_debt_receipts(
        [FOCUSED_TEST_COMMAND],
        {FOCUSED_TEST_COMMAND: 0},
    )
    assert full_suite_not_run["classification"] == "full_suite_not_run"
    owned_inside_global = mod.unrelated_global_debt_receipts(
        TEST_COMMANDS,
        TEST_EXIT_CODES,
        {
            FULL_TEST_COMMAND: (
                "FAILED tests/python/test_experiment_5892_headroom_evidence_escrow.py::"
                "test_owned_global_failure - AssertionError\n"
            )
        },
    )
    assert owned_inside_global["classification"] == "owned_failure_inside_global_suite"
    assert owned_inside_global["owned_failure_receipts"][0]["owned"] is True
    assert mod.gate_non_interference_receipts(owned_inside_global)[
        "all_unrelated_failures_safe"
    ] is False

    null_artifact = deepcopy(artifact)
    null_artifact["non_oracle_nuisance_metrics"][
        "no_non_oracle_nuisance_control_exceeds_ceiling"
    ] = False
    null_artifact["non_oracle_nuisance_metrics"]["max_non_oracle_nuisance_auroc"] = 0.9
    null_artifact["non_oracle_nuisance_metrics"]["saturated_control_names"] = ["row_order"]
    null_artifact["headroom_admission_ready_score"] = mod.headroom_admission_ready_score(
        null_artifact
    )
    null_artifact["status"] = mod.status(null_artifact)
    null_artifact["honest_verdict"] = mod.honest_verdict(null_artifact)
    null_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(null_artifact)
    assert null_artifact["status"] == "complete_null"
    assert null_artifact["honest_verdict"] == (
        "complete_null: non_oracle_nuisance_saturation=row_order"
    )
    assert mod.validate_artifact(null_artifact) is True

    retired = deepcopy(artifact)
    retired["status"] = "retired"
    retired["headroom_admission_ready_score"] = mod.headroom_admission_ready_score(retired)
    retired["honest_verdict"] = mod.honest_verdict(retired)
    retired["reproducibility_checksum"] = mod.reproducibility_checksum(retired)
    assert retired["honest_verdict"].startswith("retired:")
    assert mod.validate_artifact(retired) is True

    missing = deepcopy(artifact)
    missing.pop("immutable_upstream_hashes")
    with pytest.raises(ValueError, match="missing_fields"):
        mod.validate_artifact(missing)

    checksum_bad = deepcopy(artifact)
    checksum_bad["honest_verdict"] = "complete_ready: edited"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum_bad)

    score_bad = deepcopy(artifact)
    score_bad["terminal_artifact_freshness_receipt"]["freshness_ready"] = False
    score_bad["reproducibility_checksum"] = mod.reproducibility_checksum(score_bad)
    with pytest.raises(ValueError, match="headroom_admission_ready_score"):
        mod.validate_artifact(score_bad)

    status_bad = deepcopy(artifact)
    status_bad["status"] = "blocked"
    status_bad["reproducibility_checksum"] = mod.reproducibility_checksum(status_bad)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status_bad)

    verdict_bad = deepcopy(artifact)
    verdict_bad["honest_verdict"] = "retired: wrong"
    verdict_bad["reproducibility_checksum"] = mod.reproducibility_checksum(verdict_bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(verdict_bad)

    running = deepcopy(artifact)
    running["status"] = "running"
    running["headroom_admission_ready_score"] = mod.headroom_admission_ready_score(running)
    running["honest_verdict"] = mod.honest_verdict(running)
    running["reproducibility_checksum"] = mod.reproducibility_checksum(running)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(running)
