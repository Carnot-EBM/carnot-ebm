"""Behavior tests for the V650 twelve-task capstone.

Spec refs: REQ-REPORT-7420 and SCENARIO-REPORT-7420-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7420_v650_capstone as capstone
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one exact-shaped receipt without starting a child process."""

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
            "carnot.experiment_7420_v650_capstone": str((ROOT / capstone.MODULE_PATH).resolve())
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return the frozen affected receipt set expected from Exp7303."""

    return {
        "validation_receipts": [_receipt(name, passed) for name in REQUIRED_CHECK_NAMES],
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
    """Load the exact authorities and predecessor evidence once."""

    contract = capstone.load_contract(ROOT)
    return contract, capstone.collect_evidence(ROOT, contract["tasks"])


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-DISPOSITIONS
def test_contract_and_eleven_predecessor_slots_are_authenticated(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """The two authorities agree and the absent audit keeps its pre-gate record."""

    contract, evidence = repository_state
    assert contract["milestone"] == capstone.MILESTONE
    assert contract["comparison"]["passed"] is True
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    assert all(row["authenticated"] for row in evidence.values())
    assert evidence["exp7417-extraction-audit"]["source_kind"] == ("conductor_pre_gate_record")
    assert evidence["exp7417-extraction-audit"]["available"] is False
    assert len(evidence["exp7417-extraction-audit"]["source_records"]) == 3
    assert evidence["exp7411-arc-call-budget"]["verdict_class"] == "disqualified"
    assert evidence["exp7411-arc-call-budget"]["flagged_adversarial"] is True
    assert evidence["exp7411-arc-call-budget"]["required_validation_passed"] is False


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-DISPOSITIONS
def test_missing_slot_and_malformed_json_fail_closed(tmp_path: Path) -> None:
    """Missing bytes and unrelated log text cannot become terminal evidence."""

    task = {
        "id": "exp7417-extraction-audit",
        "title": "Audit extraction coverage and semantic preservation independently",
        "deliverable": "results/experiment_7417_v650_extraction_audit.json",
        "gated_on": [],
    }
    missing = capstone.load_evidence_slot(tmp_path, task, {})
    assert missing["source_kind"] == "missing"
    assert missing["authenticated"] is False
    assert missing["verdict_class"] == "blocked"

    log = tmp_path / capstone.CONDUCTOR_PATH
    log.parent.mkdir(parents=True)
    log.write_text("unrelated completion text\n", encoding="utf-8")
    assert capstone.load_evidence_slot(tmp_path, task, {})["source_kind"] == "missing"

    malformed = tmp_path / "bad.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        capstone.load_json_object(malformed)
    assert capstone._compare_gate("unsupported", 1, 1) is False
    assert capstone._required_validation({"required_checks_passed": True}) is True


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-DISPOSITIONS
def test_contract_parser_rejects_mismatch_and_wrong_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Either authority disagreement or a changed raw order fails before reduction."""

    monkeypatch.setattr(
        capstone,
        "compare_contract_authorities",
        lambda *_args, **_kwargs: {"passed": False, "errors": ["changed"]},
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        capstone.load_contract(ROOT)

    roadmap = capstone.load_yaml(ROOT / capstone.ROADMAP_PATH)
    changed = deepcopy(roadmap)
    changed["tasks"] = list(changed["tasks"])
    changed["tasks"][0], changed["tasks"][1] = changed["tasks"][1], changed["tasks"][0]
    monkeypatch.setattr(capstone, "load_yaml", lambda _path: changed)
    monkeypatch.setattr(
        capstone,
        "compare_contract_authorities",
        lambda *_args, **_kwargs: {"passed": True, "errors": []},
    )
    with pytest.raises(ValueError, match="task order"):
        capstone.load_contract(ROOT)


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-CLAIMS
def test_nine_claim_rows_preserve_independent_boundaries(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Null, blocked, disqualified, oracle, cost, and board facts stay separate."""

    _, evidence = repository_state
    matrix = capstone.reduce_claim_matrix(evidence)
    assert list(matrix) == list(capstone.CLAIM_BRANCHES)
    assert matrix["corpus_authority"]["completion_score"] == 1
    assert matrix["corpus_authority"]["label_authority"] == "machine_annotations"
    assert matrix["static_calibration"]["verdict_class"] == "null"
    assert matrix["static_calibration"]["completion_score"] == 1
    assert matrix["static_calibration"]["benefit_score"] == 0
    assert matrix["online_calibration"]["audit_complete_score"] == 1
    assert matrix["arc_callback_invariants"]["verdict_class"] == "disqualified"
    assert matrix["arc_callback_invariants"]["flagged_adversarial"] is True
    assert matrix["qwen_extraction"]["verdict_class"] == "blocked"
    assert matrix["qwen_extraction"]["completion_score"] == 0
    assert matrix["extraction_audit"]["source_kind"] == "conductor_pre_gate_record"
    assert matrix["revised_proof_memory"]["completion_score"] == 1
    assert matrix["revised_proof_memory"]["verifier_is_oracle"] is True
    assert matrix["numeric_host_cost"]["benefit_score"] == 0
    assert matrix["board_status"]["host_science_invalidated"] is False
    assert matrix["board_status"]["boards"]["GateMate"] == ("blocked_changed_physical_state")

    controls = {row["source_name"]: row for row in capstone.literature_control_rows(evidence)}
    assert set(controls) == {"Enoki", "MARGIN", "corrupted-feedback", "Memoir"}
    assert controls["Enoki"]["paper_reproduction_claimed"] is False
    assert controls["MARGIN"]["executed_controls"]
    assert controls["corrupted-feedback"]["executed_controls"]
    assert controls["Memoir"]["local_result_eligible"] is True


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-RETIREMENT
def test_retirement_requires_the_exact_declared_prior_verdict(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """A changed capstone class and new branches cannot retire old mechanisms."""

    contract, evidence = repository_state
    current = "complete_blocked_required_v650_science_with_twelve_dispositions"
    rows = capstone.retirement_rows(contract["tasks"], evidence, current)
    assert len(rows) == 1
    assert rows[0]["prior_experiment_id"] == "exp7408-capstone"
    assert rows[0]["same_exact_verdict"] is False
    assert rows[0]["decision"] == "continue-with-measured-cause"

    repeated = capstone.retirement_rows(contract["tasks"], evidence, rows[0]["previous_verdict"])
    assert repeated[0]["same_exact_verdict"] is True
    assert repeated[0]["decision"] == "retire-unchanged-mechanism"
    retired_continuation = capstone.continuation_rows(
        capstone.reduce_claim_matrix(evidence), repeated
    )
    assert retired_continuation[0]["decision"] == "retire-unchanged-mechanism"

    continuation = capstone.continuation_rows(capstone.reduce_claim_matrix(evidence), rows)
    assert len(continuation) == len(capstone.CLAIM_BRANCHES)
    assert {row["decision"] for row in continuation} <= set(capstone.CONTINUATION_DECISIONS)
    assert not any(row["decision"] == "retire-unchanged-mechanism" for row in continuation)


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-CLASSIFY
def test_required_extraction_absence_blocks_without_erasing_valid_nulls(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Missing required extraction wins blocked while non-required ARC remains scoped."""

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        completed_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    assert artifact["schema"] == capstone.SCHEMA
    assert artifact["status"].startswith("complete_blocked")
    assert artifact["honest_verdict"].startswith("complete_blocked")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["flagged_adversarial"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == ("aggregation_from_exact_declared_artifacts")
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["promotion_score"] == 0
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_dispositions"]) == 12
    assert artifact["task_dispositions"][-1]["task_id"] == capstone.EXPERIMENT_ID
    assert artifact["gate_check_summary"]["first_failure"]["upstream"] == (
        "exp7416-anchored-extraction"
    )
    assert artifact["scope_reduction_compliance"]["standing_floors"] == {
        "corpus": True,
        "arc": False,
        "calibrated_decision": True,
        "self_learning": True,
        "hardware": True,
    }
    assert capstone.validate_artifact(artifact, root=ROOT) == []


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-CLASSIFY
def test_invalid_required_science_precedes_blocked_science(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """A required invalid producer disqualifies before another required absence."""

    contract, evidence = repository_state
    changed = deepcopy(evidence)
    changed["exp7413-source-calibration"]["verdict_class"] = "disqualified"
    changed["exp7413-source-calibration"]["flagged_adversarial"] = True
    terminal = capstone.classify_terminal(changed, _validation())
    assert terminal["verdict_class"] == "disqualified"
    assert terminal["gate_check_summary"]["first_failure"]["upstream"] == (
        "exp7413-source-calibration"
    )

    current_failure = capstone.classify_terminal(evidence, _validation(False))
    assert current_failure["verdict_class"] == "disqualified"
    assert any(
        row["category"] == "required_validation"
        for row in current_failure["gate_check_summary"]["failures"]
    )

    complete = deepcopy(evidence)
    for task_id in ("exp7416-anchored-extraction", "exp7417-extraction-audit"):
        complete[task_id]["available"] = True
        complete[task_id]["accepted_for_science"] = True
        complete[task_id]["verdict_class"] = "null"
        complete[task_id]["required_validation_passed"] = True
    assert capstone.classify_terminal(complete, _validation())["verdict_class"] == "null"


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-ARTIFACT
def test_cold_validator_rejects_identity_row_hash_score_and_checksum_drift(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Independent replay rejects every field that controls the conclusion."""

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        completed_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )

    def errors_after(change: Any, refresh: bool = True) -> list[str]:
        changed = deepcopy(artifact)
        change(changed)
        if refresh:
            changed["reproducibility_checksum"] = capstone.reproducibility_checksum(changed)
        return capstone.validate_artifact(changed, root=ROOT)

    assert "identity_invalid" in errors_after(
        lambda value: value.__setitem__("milestone", "2026.09.000")
    )
    assert "model_contract_invalid" in errors_after(
        lambda value: value.__setitem__("MODEL_SPECS", ["unexpected"])
    )
    assert "task_dispositions_invalid" in errors_after(
        lambda value: value["task_dispositions"][0].__setitem__("verdict_class", "positive")
    )
    assert "task_dispositions_invalid" in errors_after(
        lambda value: value.__setitem__("task_dispositions", [])
    )
    assert "source_hash_mismatch" in errors_after(
        lambda value: value["source_artifact_hashes"]["exp7409-evidence-custody"].__setitem__(
            "sha256", "sha256:" + "0" * 64
        )
    )
    assert "capstone_score_invalid" in errors_after(
        lambda value: value.__setitem__("capstone_complete_score", 0)
    )
    assert "claim_matrix_invalid" in errors_after(
        lambda value: value["claim_matrix"]["static_calibration"].__setitem__("benefit_score", 1)
    )
    assert "reproducibility_checksum_invalid" in errors_after(
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:" + "0" * 64),
        refresh=False,
    )

    assert capstone.validate_artifact({"schema": capstone.SCHEMA}, root=ROOT)[0].startswith(
        "missing_required_field:"
    )
    assert "lifecycle_invalid" in errors_after(lambda value: value.__setitem__("status", "running"))
    assert "substrate_invalid" in errors_after(
        lambda value: value.__setitem__("execution_venue", "host_cpu")
    )
    assert "retirement_rows_invalid" in errors_after(
        lambda value: value.__setitem__("retirement_rows", [])
    )
    assert "continuation_rows_invalid" in errors_after(
        lambda value: value.__setitem__("continuation_rows", [])
    )
    assert "literature_control_rows_invalid" in errors_after(
        lambda value: value.__setitem__("literature_control_rows", [])
    )
    assert "scope_reduction_invalid" in errors_after(
        lambda value: value.__setitem__("scope_reduction_compliance", {})
    )
    assert "terminal_reduction_invalid" in errors_after(
        lambda value: value.__setitem__("honest_verdict", "complete_wrong")
    )
    assert "promotion_invalid" in errors_after(
        lambda value: value.__setitem__("promotion_score", 1)
    )
    assert "field_principles_invalid" in errors_after(
        lambda value: value.__setitem__("field_principles", {})
    )

    flagged = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        completed_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
        flagged_adversarial=True,
    )
    assert flagged["verdict_class"] == "disqualified"
    assert capstone.validate_artifact(flagged, root=ROOT) == []

    monkeypatch_artifact = deepcopy(artifact)
    original = capstone.load_contract
    capstone.load_contract = lambda _root: (_ for _ in ()).throw(ValueError("changed"))
    try:
        assert "independent_reduction_failed" in capstone.validate_artifact(
            monkeypatch_artifact, root=ROOT
        )
    finally:
        capstone.load_contract = original


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-ARTIFACT
def test_hash_reader_rejects_malformed_and_changed_sources(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Source rows must remain file-backed and conductor records must match exactly."""

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        completed_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    assert capstone._hashes_match({"source_artifact_hashes": []}, ROOT) is False
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["bad"] = "not-a-row"
    assert capstone._hashes_match(changed, ROOT) is False

    conductor = changed = deepcopy(artifact)
    conductor["source_artifact_hashes"]["exp7417-extraction-audit"]["source_records"] = "wrong"
    assert capstone._hashes_match(conductor, ROOT) is False
    conductor = deepcopy(artifact)
    conductor["source_artifact_hashes"]["exp7417-extraction-audit"]["source_file_sha256"] = (
        "sha256:" + "0" * 64
    )
    assert capstone._hashes_match(conductor, ROOT) is False
    conductor = deepcopy(artifact)
    conductor["source_artifact_hashes"]["exp7417-extraction-audit"]["sha256"] = "sha256:" + "0" * 64
    assert capstone._hashes_match(conductor, ROOT) is False


# REQ-REPORT-7420 / SCENARIO-REPORT-7420-ARTIFACT
def test_scoped_plan_and_cli_cold_replay(
    tmp_path: Path,
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen plan stays narrow and the public reader validates one candidate."""

    private = tmp_path / "private"
    private.mkdir()
    commands = capstone.build_validation_plan(ROOT, private)
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert capstone.validate_validation_plan(ROOT, commands) == []
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert {"-n", "0", "-o", "addopts=", "--no-cov"}.issubset(focused.argv)
    coverage = next(command for command in commands if command.name == "changed_module_coverage")
    assert any(argument.startswith("--data-file=") for argument in coverage.argv)
    report = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert "COVERAGE_FILE" in dict(report.command_environment)
    assert all(command.name != "full_python_suite" for command in commands)

    contract, evidence = repository_state
    candidate = tmp_path / "candidate.json"
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        completed_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    capstone.atomic_json(candidate, artifact)
    assert capstone.main(["--date", capstone.RUN_DATE, "--validate", str(candidate)]) == 0
    assert capstone.main(["--date", capstone.RUN_DATE, "--validate", str(tmp_path / "none")]) == 1
    with pytest.raises(SystemExit):
        capstone.main(["--date", "wrong"])

    non_object = tmp_path / "non-object.json"
    non_object.write_text(json.dumps([]), encoding="utf-8")
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    assert capstone.main(["--date", capstone.RUN_DATE, "--validate", str(non_object)]) == 1

    monkeypatch.setattr(
        capstone,
        "run_experiment",
        lambda *_args: {
            "status": "complete",
            "verdict_class": "null",
            "capstone_complete_score": 1,
        },
    )
    assert capstone.main(["--date", capstone.RUN_DATE]) == 0
