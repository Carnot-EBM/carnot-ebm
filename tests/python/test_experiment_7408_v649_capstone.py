"""Behavior tests for the V649 fourteen-task capstone.

Spec refs: REQ-REPORT-7408 and SCENARIO-REPORT-7408-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import pytest

from carnot import experiment_7408_v649_capstone as capstone
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one command receipt without starting a child process."""

    row: dict[str, Any] = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {},
        "scope": "test_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7408_v649_capstone": str((ROOT / capstone.MODULE_PATH).resolve())
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return the frozen affected receipt set expected from Exp7303."""

    receipts = [_receipt(name, passed) for name in REQUIRED_CHECK_NAMES]
    return {
        "validation_receipts": receipts,
        "required_checks_passed": passed,
        "terminal_validation_passed": passed,
        "plan_errors": [],
        "repository_health": {
            "status": "historical_observations_retained",
            "as_of": capstone.RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
    }


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load exact V649 authorities once for repository-backed checks."""

    contract = capstone.load_contract(ROOT)
    return contract, capstone.collect_evidence(ROOT, contract["tasks"])


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-DISPOSITIONS
def test_active_v649_contract_and_thirteen_predecessor_slots_are_authenticated(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """The active YAML and each prior disposition retain their real source kind."""

    contract, evidence = repository_state
    assert contract["milestone"] == capstone.MILESTONE
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["contract_match"] is False
    assert contract["contract_comparison"]["advisory_only"] is True
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    assert evidence["exp7397-delayed-adapter"]["source_kind"] == (
        "conductor_completion_without_artifact"
    )
    assert evidence["exp7399-online-trial"]["source_kind"] == (
        "conductor_completion_without_artifact"
    )
    assert evidence["exp7404-live-memory"]["source_kind"] == "conductor_pre_gate_record"
    assert evidence["exp7406-arc-generalization"]["verdict_class"] == "disqualified"
    assert evidence["exp7406-arc-generalization"]["flagged_adversarial"] is True
    assert all(row["authenticated"] for row in evidence.values())


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-DISPOSITIONS
def test_missing_evidence_and_unmatched_log_fail_closed(tmp_path: Path) -> None:
    """A completion label cannot replace missing declared producer bytes."""

    task = {
        "id": "exp7397-delayed-adapter",
        "title": "Prototype a bounded energy-offset learner with delayed feedback",
        "deliverable": "results/experiment_7397_v649_delayed_adapter.json",
    }
    missing = capstone.load_evidence_slot(tmp_path, task, {})
    assert missing["source_kind"] == "missing"
    assert missing["authenticated"] is False
    assert missing["verdict_class"] == "blocked"

    log = tmp_path / capstone.CONDUCTOR_PATH
    log.parent.mkdir(parents=True)
    log.write_text("similar completion wording without the exact record\n", encoding="utf-8")
    assert capstone.load_evidence_slot(tmp_path, task, {})["source_kind"] == "missing"

    malformed = tmp_path / "bad.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        capstone.load_json_object(malformed)


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-CLAIMS
def test_claim_matrix_preserves_seven_independent_boundaries(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Null, circular, blocked, disqualified, and host-cost evidence stay separate."""

    _, evidence = repository_state
    matrix = capstone.reduce_claim_matrix(evidence)
    assert list(matrix) == list(capstone.CLAIM_BRANCHES)
    assert matrix["static_calibration"]["verdict_class"] == "null"
    assert matrix["static_calibration"]["completion_score"] == 1
    assert matrix["static_calibration"]["value_score"] == 0
    assert matrix["online_learning"]["verdict_class"] == "blocked"
    assert matrix["synthetic_memory"]["verdict_class"] == "circular_positive"
    assert matrix["synthetic_memory"]["verifier_is_oracle"] is True
    assert matrix["live_memory"]["verdict_class"] == "blocked"
    assert matrix["proof_audit"]["audit_complete_score"] == 1
    assert matrix["arc"]["verdict_class"] == "disqualified"
    assert matrix["arc"]["official_score"] is None
    assert matrix["host_service_cost"]["verdict_class"] == "positive"
    assert matrix["host_service_cost"]["hardware_ready_score"] == 0
    assert matrix["host_service_cost"]["hardware_value_score"] == 0

    literature = capstone.literature_control_rows(evidence)
    by_name = {row["source_name"]: row for row in literature}
    assert set(by_name) == {"ORCA", "CORD", "Solver-Hard", "Memoir", "hardware_review"}
    assert by_name["ORCA"]["local_result_eligible"] is False
    assert by_name["Solver-Hard"]["paper_assumption_satisfied_locally"] is False
    assert by_name["Memoir"]["executed_controls"]
    assert by_name["hardware_review"]["local_result_eligible"] is True


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-RETIREMENT
def test_only_exact_repeated_verdicts_authorize_retirement(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Similar words and absent science do not activate exact retirement clauses."""

    contract, evidence = repository_state
    rows = capstone.retirement_rows(contract["tasks"], evidence)
    retired = {row["task_id"] for row in rows if row["decision"] == "retire-unchanged-mechanism"}
    assert retired == {"exp7404-live-memory", "exp7406-arc-generalization"}
    assert all(
        row["same_exact_verdict"] for row in rows if row["decision"] == "retire-unchanged-mechanism"
    )

    changed = deepcopy(evidence)
    changed["exp7406-arc-generalization"]["honest_verdict"] += "_similar_not_exact"
    changed_rows = capstone.retirement_rows(contract["tasks"], changed)
    assert not any(
        row["task_id"] == "exp7406-arc-generalization"
        and row["decision"] == "retire-unchanged-mechanism"
        for row in changed_rows
    )

    continuation = capstone.continuation_rows(evidence, rows)
    assert {row["decision"] for row in continuation} <= set(capstone.CONTINUATION_DECISIONS)
    assert len(continuation) == len(capstone.CLAIM_BRANCHES)


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-TERMINAL
def test_terminal_artifact_accounts_for_all_tasks_without_promotion(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Current checks complete the capstone while invalid ARC evidence disqualifies science."""

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T04:00:00+00:00",
        completed_at_utc="2026-09-19T04:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    assert artifact["schema"] == capstone.SCHEMA
    assert artifact["status"].startswith("complete_disqualified")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert isinstance(artifact["inference_substrate"], str)
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["promotion_score"] == 0
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_dispositions"]) == 14
    assert artifact["task_dispositions"][-1]["task_id"] == capstone.EXPERIMENT_ID
    assert artifact["task_dispositions"][-1]["source_kind"] == "self_current_checks"
    assert artifact["gate_check_summary"]["first_failure"]["upstream"] == (
        "exp7406-arc-generalization"
    )
    assert capstone.validate_artifact(artifact, root=ROOT) == []


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-ARTIFACT
def test_cold_validator_rejects_identity_rows_hash_score_and_checksum_drift(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Independent replay rejects every capstone field that controls the conclusion."""

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T04:00:00+00:00",
        completed_at_utc="2026-09-19T04:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    mutations = (
        ("identity_invalid", lambda value: value.__setitem__("milestone", "2026.09.000")),
        (
            "model_contract_invalid",
            lambda value: value.__setitem__("MODEL_SPECS", ["unexpected"]),
        ),
        (
            "task_dispositions_invalid",
            lambda value: value["task_dispositions"][0].__setitem__("verdict_class", "positive"),
        ),
        (
            "source_hash_mismatch",
            lambda value: value["source_artifact_hashes"]["exp7395-receipt-protocol"].__setitem__(
                "sha256", "sha256:" + "0" * 64
            ),
        ),
        (
            "capstone_score_invalid",
            lambda value: value.__setitem__("capstone_complete_score", 0),
        ),
        (
            "reproducibility_checksum_invalid",
            lambda value: value.__setitem__("reproducibility_checksum", "sha256:" + "0" * 64),
        ),
    )
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in capstone.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_invalid" in capstone.validate_artifact(changed, root=ROOT)
    del changed["schema"]
    assert capstone.validate_artifact(changed, root=ROOT)[0] == "missing_required_field:schema"
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-ARTIFACT
def test_exp7358_plan_is_exact_and_entrypoint_is_thin(tmp_path: Path) -> None:
    """The affected plan stays bounded and the launcher delegates once."""

    commands = capstone.build_validation_plan(ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert capstone.validate_validation_plan(ROOT, commands) == []
    assert all(
        capstone.TEST_PATH.as_posix() in command.argv
        for command in commands
        if command.name in {"focused_pytest", "changed_module_coverage"}
    )
    assert not any("full_python_suite" in command.name for command in commands)
    source = (ROOT / capstone.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7408_v649_capstone import main" in source
    assert source.count("main()") == 1

    duplicate = [*commands, commands[0]]
    assert "duplicate_command:worktree_imports" in capstone.validate_validation_plan(
        ROOT, duplicate
    )


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-ARTIFACT
def test_atomic_write_progress_and_date_boundaries(tmp_path: Path) -> None:
    """Utility boundaries keep publication atomic and the execution date frozen."""

    path = tmp_path / "nested" / "artifact.json"
    capstone.atomic_json(path, {"complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    assert capstone.utc_now().endswith("+00:00")
    capstone.progress(time.monotonic(), "test", "boundary", units=1)
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        capstone.date_argument("20260918")


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-ARTIFACT
def test_defensive_contract_gate_and_validator_paths(
    tmp_path: Path,
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Malformed authorities and private artifact changes fail at their owning check."""

    with pytest.raises(ValueError, match="numeric experiment identity missing"):
        capstone._numeric_experiment_id("task-without-number")

    (tmp_path / capstone.ROADMAP_PATH).write_text(
        "milestone: 2026.09.648\ntasks: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="milestone"):
        capstone.load_contract(tmp_path)
    (tmp_path / capstone.ROADMAP_PATH).write_text(
        f"milestone: {capstone.MILESTONE}\ntasks: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fourteen"):
        capstone.load_contract(tmp_path)

    assert capstone._gate_failures({"gate_check_summary": "failed"}) == ["failed"]
    assert capstone._gate_failures({"gate_check_summary": {"first_failure": {"check": "one"}}}) == [
        {"check": "one"}
    ]
    assert capstone._compare_gate("unsupported", 1, 1) is False

    manifest = tmp_path / capstone.EXCLUSION_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "retired:\n- ignored-scalar\n- experiment_id: exp7408\nretired_experiments: []\nretired_extras: []\n",
        encoding="utf-8",
    )
    assert capstone._retired_ids(tmp_path) == {7408}

    _, evidence = repository_state
    failed = capstone._terminal_state(evidence, _validation(False))
    assert failed["gate_check_summary"]["blocking_failed_count"] == 2
    blocked_evidence = deepcopy(evidence)
    blocked_evidence["exp7406-arc-generalization"]["verdict_class"] = "blocked"
    assert capstone._terminal_state(blocked_evidence, _validation())["verdict_class"] == ("blocked")
    clean_evidence = deepcopy(evidence)
    for row in clean_evidence.values():
        row["verdict_class"] = "null"
        row["available"] = True
    assert capstone._terminal_state(clean_evidence, _validation())["verdict_class"] == "null"


# REQ-REPORT-7408 / SCENARIO-REPORT-7408-ARTIFACT
def test_private_hash_and_terminal_mutations_are_rejected(
    tmp_path: Path,
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """The cold reader rejects malformed hash rows and every terminal control field."""

    assert capstone._hashes_match({"source_artifact_hashes": []}, ROOT) is False
    assert capstone._hashes_match({"source_artifact_hashes": {"bad": "row"}}, ROOT) is False

    staged = tmp_path / capstone.STAGED_ROADMAP_PATH
    staged.write_text("milestone: 2026.09.649\n", encoding="utf-8")
    assert (
        capstone._hashes_match(
            {
                "source_artifact_hashes": {
                    "staged": {
                        "path": capstone.STAGED_ROADMAP_PATH.as_posix(),
                        "sha256": None,
                        "source_kind": "expected_absent_staged_authority",
                    }
                }
            },
            tmp_path,
        )
        is False
    )
    assert (
        capstone._hashes_match(
            {
                "source_artifact_hashes": {
                    "record": {
                        "path": capstone.CONDUCTOR_PATH.as_posix(),
                        "sha256": "sha256:" + "0" * 64,
                        "source_file_sha256": "sha256:" + "0" * 64,
                        "source_records": [],
                        "source_kind": "conductor_pre_gate_record",
                    }
                }
            },
            tmp_path,
        )
        is False
    )
    log = tmp_path / capstone.CONDUCTOR_PATH
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("one exact record\n", encoding="utf-8")
    conductor_row = {
        "path": capstone.CONDUCTOR_PATH.as_posix(),
        "sha256": "sha256:" + "0" * 64,
        "source_file_sha256": capstone.sha256_file(log),
        "source_records": ["one exact record"],
        "source_kind": "conductor_pre_gate_record",
    }
    wrong_file_hash = deepcopy(conductor_row)
    wrong_file_hash["source_file_sha256"] = "sha256:" + "1" * 64
    assert (
        capstone._hashes_match({"source_artifact_hashes": {"record": wrong_file_hash}}, tmp_path)
        is False
    )
    assert (
        capstone._hashes_match({"source_artifact_hashes": {"record": conductor_row}}, tmp_path)
        is False
    )

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T04:00:00+00:00",
        completed_at_utc="2026-09-19T04:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    cases = (
        ("lifecycle_invalid", lambda value: value.__setitem__("status", "running")),
        (
            "substrate_invalid",
            lambda value: value.__setitem__("execution_venue", "host_cpu"),
        ),
        (
            "task_dispositions_invalid",
            lambda value: value["task_dispositions"].reverse(),
        ),
        (
            "claim_matrix_invalid",
            lambda value: value["claim_matrix"].pop("static_calibration"),
        ),
        ("promotion_invalid", lambda value: value.__setitem__("promotion_score", 1)),
        (
            "terminal_reduction_invalid",
            lambda value: value.__setitem__("verdict_class", "blocked"),
        ),
    )
    for expected, mutate in cases:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in capstone.validate_artifact(changed, root=ROOT)

    flagged = deepcopy(artifact)
    flagged["flagged_adversarial"] = True
    flagged["promotion_score"] = 1
    assert "flagged_promotion_invalid" in capstone.validate_artifact(flagged, root=ROOT)
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact, root=tmp_path)
