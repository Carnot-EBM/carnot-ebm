"""Behavior tests for the V647 capstone.

Spec refs: REQ-REPORT-7380 and SCENARIO-REPORT-7380-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import pytest

from carnot import experiment_7380_v647_capstone as capstone
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one bounded command receipt without running a nested child process."""

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
            "carnot.experiment_7380_v647_capstone": str((ROOT / capstone.MODULE_PATH).resolve())
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return the exact affected receipt set expected from Exp7303."""

    receipts = [_receipt(name, passed=passed) for name in REQUIRED_CHECK_NAMES]
    return {
        "validation_receipts": receipts,
        "required_checks_passed": passed,
        "plan_errors": [],
        "repository_health": {
            "status": "historical_observations_retained",
            "as_of": capstone.RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
    }


def _publication() -> dict[str, Any]:
    """Represent the unchanged four-gate FoVer publication result."""

    return {
        "paper_ready": True,
        "unmet_gates": [],
        "gates": {
            name: {"pass": True, "detail": f"canonical {name}"} for name in ("G1", "G2", "G3", "G4")
        },
        "note": "Stable 4-condition gate.",
    }


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load each immutable V647 authority once for repository-backed checks."""

    contract = capstone.load_contract(ROOT)
    return contract, capstone.collect_evidence(ROOT, contract["tasks"])


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-CONTRACT
def test_exact_contract_and_canonical_pregates_fill_first_eleven_slots(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """The capstone uses real artifacts or authenticated pre-gates, never invented JSON."""

    contract, evidence = repository_state
    assert contract["milestone"] == capstone.MILESTONE
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["contract_match"] is True
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    assert evidence["exp7373-proposal-capture"]["source_kind"] == "conductor_pre_gate_artifact"
    assert evidence["exp7373-proposal-capture"]["actual_path"] == (
        "results/experiment_7373_proposal_capture.json"
    )
    assert evidence["exp7374-prospective-memory"]["source_kind"] == "conductor_log_record"
    assert evidence["exp7375-memory-audit"]["source_kind"] == "conductor_log_record"
    assert evidence["exp7374-prospective-memory"]["actual_path"] is None
    assert evidence["exp7375-memory-audit"]["actual_path"] is None
    assert all(row["authenticated"] for row in evidence.values())


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-CONTRACT
def test_missing_or_drifted_evidence_fails_closed(tmp_path: Path) -> None:
    """An unavailable slot remains blocked and a false pre-gate line is rejected."""

    task = {
        "id": "exp7374-prospective-memory",
        "deliverable": "results/experiment_7374_v647_prospective_memory.json",
    }
    missing = capstone.load_evidence_slot(tmp_path, task)
    assert missing["source_kind"] == "missing"
    assert missing["verdict_class"] == "blocked"
    assert missing["authenticated"] is False
    assert missing["accepted_for_science"] is False

    (tmp_path / capstone.CONDUCTOR_PATH).parent.mkdir(parents=True)
    (tmp_path / capstone.CONDUCTOR_PATH).write_text("similar but invented gate line\n")
    drifted = capstone.load_evidence_slot(tmp_path, task)
    assert drifted["source_kind"] == "missing"
    assert drifted["authenticated"] is False

    ordinary = {
        "id": "exp7369-contract",
        "deliverable": "results/experiment_7369_v647_contract.json",
    }
    assert capstone.load_evidence_slot(tmp_path, ordinary)["source_kind"] == "missing"
    unknown = {"id": "exp0000-unknown", "deliverable": "results/missing.json"}
    assert capstone.load_evidence_slot(tmp_path, unknown)["source_kind"] == "missing"

    malformed = tmp_path / "list.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        capstone._load_json(malformed)


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-CONTRACT
def test_contract_rejects_wrong_milestone_and_task_order(tmp_path: Path) -> None:
    """Both authority identity and the ordered twelve-row roster fail closed."""

    (tmp_path / capstone.ROADMAP_PATH).write_text("milestone: 2026.09.646\ntasks: []\n")
    with pytest.raises(ValueError, match="milestone"):
        capstone.load_contract(tmp_path)

    (tmp_path / capstone.ROADMAP_PATH).write_text(
        f"milestone: {capstone.MILESTONE}\ntasks: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact twelve"):
        capstone.load_contract(tmp_path)


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-SCIENCE
def test_terminal_reduction_blocks_absence_but_disqualifies_required_failures(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """External absence is blocked; an observed required validation failure is stronger."""

    _, evidence = repository_state
    current = capstone.terminal_state(evidence, required_validation_passed=True)
    assert current["verdict_class"] == "disqualified"
    assert current["required_science_complete_score"] == 0
    failures = current["gate_check_summary"]["failures"]
    assert any(row["upstream"] == "exp7374-prospective-memory" for row in failures)
    assert any(row["upstream"] == "exp7378-ising-audit" for row in failures)
    assert all(row["observed"] != "retryable_partial" for row in failures)

    external_only = deepcopy(evidence)
    for task_id in ("exp7372-qwen-canary", "exp7376-arc-outcomes", "exp7378-ising-audit"):
        external_only[task_id]["verdict_class"] = "null"
        external_only[task_id]["flagged_adversarial"] = False
        external_only[task_id]["payload"] = {
            **external_only[task_id]["payload"],
            "verdict_class": "null",
            "flagged_adversarial": False,
        }
    blocked = capstone.terminal_state(external_only, required_validation_passed=True)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"]["upstream"] == (
        "exp7374-prospective-memory"
    )

    own_failure = capstone.terminal_state(external_only, required_validation_passed=False)
    assert own_failure["verdict_class"] == "disqualified"
    assert own_failure["gate_check_summary"]["first_failure"]["upstream"] == "exp7380-capstone"

    completed = deepcopy(external_only)
    for task_id in ("exp7374-prospective-memory", "exp7375-memory-audit"):
        completed[task_id]["accepted_for_science"] = True
    null = capstone.terminal_state(completed, required_validation_passed=True)
    assert null["verdict_class"] == "null"
    assert null["required_science_complete_score"] == 1


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-CLAIMS
def test_claim_reduction_keeps_learning_arc_ising_and_hardware_boundaries(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Summary rows report only measured evidence and do not fill absent learning rows."""

    _, evidence = repository_state
    rows = capstone.reduce_claim_rows(evidence)
    by_branch = {row["branch"]: row for row in rows}
    assert list(by_branch) == list(capstone.CLAIM_BRANCHES)

    proof = by_branch["proof_memory"]
    assert proof["completion_score"] == 0
    assert proof["metrics"]["prospective_measurement_available"] is False
    assert proof["metrics"]["independent_audit_available"] is False
    assert proof["metrics"]["development_safety_ready"] is True
    assert proof["verifier_is_oracle"] is True

    proposals = by_branch["fresh_model_proposals"]
    assert proposals["metrics"]["current_model_calls_counted_by_capstone"] == 0
    assert proposals["metrics"]["canary_transport_ready_score"] == 0
    assert proposals["metrics"]["capture_started"] is False

    arc = by_branch["arc_outcomes"]
    assert arc["metrics"] == {
        "planned_episodes": 6,
        "attempted_episodes": 6,
        "completed_episodes": 0,
        "censored_episodes": 6,
        "supervisor_firing_count": 0,
    }
    assert arc["completion_score"] == 0

    law = by_branch["ising_law_and_samples"]
    assert law["metrics"]["law_fixture_ready_score"] == 1
    assert law["metrics"]["completed_finite_law_rows"] == 216
    assert law["metrics"]["completed_source_cells"] == 144
    assert law["metrics"]["sample_audit_eligible"] is False

    hardware = by_branch["hardware_placement"]
    assert hardware["metrics"]["board_disposition_complete_score"] == 1
    assert hardware["metrics"]["placement_measurement_available"] is False
    assert hardware["value_score"] == 0


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-PUBLICATION
def test_publication_result_preserves_fover_scope_without_v647_authority() -> None:
    """Even four passing historical gates grant no V647 publication or deployment."""

    result = capstone.publication_gate_results(_publication())
    assert list(result["gates"]) == ["G1", "G2", "G3", "G4"]
    assert result["paper_ready"] is True
    assert result["scope"] == "historical_fover_paper_only"
    assert result["certifies_v647"] is False
    assert result["authorizes_external_publication"] is False
    assert result["authorizes_deployment"] is False

    malformed = capstone.publication_gate_results({"gates": {}})
    assert malformed["paper_ready"] is False
    assert malformed["unmet_gates"] == ["G1", "G2", "G3", "G4"]


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-ARTIFACT
def test_terminal_artifact_has_twelve_dispositions_and_replays(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """The self row appears only in the validated terminal artifact and checksum replay."""

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        _publication(),
        started_at_utc="2026-09-18T01:00:00+00:00",
        completed_at_utc="2026-09-18T01:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    assert artifact["schema"] == capstone.SCHEMA
    assert artifact["status"].startswith("complete_disqualified")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["promotion_score"] == 0
    assert artifact["milestone_disposition_complete_score"] == 1
    assert artifact["required_science_complete_score"] == 0
    assert len(artifact["disposition_rows"]) == 12
    assert artifact["disposition_rows"][-1]["task_id"] == capstone.EXPERIMENT_ID
    assert artifact["disposition_rows"][-1]["source_kind"] == "self"
    assert capstone.validate_artifact(artifact, root=ROOT) == []
    terminal = capstone.terminal_state(evidence, required_validation_passed=True)
    assert (
        len(
            capstone.build_disposition_rows(
                contract["tasks"], evidence, terminal, validation_complete=False
            )
        )
        == 11
    )

    changed = deepcopy(artifact)
    changed["disposition_rows"][4]["verdict_class"] = "positive"
    assert "disposition_rows_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["unexpected"]
    assert "model_contract_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    del changed["schema"]
    assert capstone.validate_artifact(changed, root=ROOT) == ["missing_required_field:schema"]
    changed = deepcopy(artifact)
    changed["milestone"] = "2026.09.000"
    assert "identity_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["status"] = "running"
    assert "lifecycle_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["execution_venue"] = "remote"
    assert "substrate_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["readiness_score"] = 1
    assert "failed_state_score_nonzero" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = []
    assert "source_hash_mismatch" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["active_roadmap"] = "not a row"
    assert "source_hash_mismatch" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["exp7374-prospective-memory"]["source_file_sha256"] = (
        "sha256:" + "0" * 64
    )
    assert "source_hash_mismatch" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["active_roadmap"]["sha256"] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in capstone.validate_artifact(changed, root=ROOT)
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-ARTIFACT
def test_exp7358_plan_is_exact_and_entrypoint_is_thin(tmp_path: Path) -> None:
    """Validation uses the actual bounded command plan and no whole-suite alias."""

    commands = capstone.build_validation_plan(ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert capstone.validate_validation_plan(ROOT, commands) == []
    assert all(
        capstone.TEST_PATH.as_posix() in command.argv
        for command in commands
        if command.name in {"focused_pytest", "changed_module_coverage"}
    )
    assert not any("full_python_suite" in command.name for command in commands)
    assert all(
        (tmp_path / "private").is_relative_to(Path(argument.split("=", 1)[1]).parents[1])
        for command in commands
        for argument in command.argv
        if argument.startswith("--basetemp=")
    )

    source = (ROOT / capstone.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7380_v647_capstone import main" in source
    assert source.count("main()") == 1

    duplicate = [*commands, commands[0]]
    assert "duplicate_command:worktree_imports" in capstone.validate_validation_plan(
        ROOT, duplicate
    )


# REQ-REPORT-7380 / SCENARIO-REPORT-7380-ARTIFACT
def test_atomic_write_and_date_boundary(tmp_path: Path) -> None:
    """The public boundary accepts only the frozen execution date and complete JSON."""

    path = tmp_path / "nested" / "artifact.json"
    capstone.atomic_json(path, {"complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    assert capstone.utc_now().endswith("+00:00")
    capstone.progress(time.monotonic(), "test", "boundary", units=1)
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        capstone.date_argument("20260917")
