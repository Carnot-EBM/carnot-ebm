"""Behavior tests for the V651 thirteen-task capstone.

Spec refs: REQ-REPORT-7433 and SCENARIO-REPORT-7433-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7433_v651_capstone as capstone
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one exact-shaped affected-check receipt without a subprocess."""

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
            "carnot.experiment_7433_v651_capstone": str((ROOT / capstone.MODULE_PATH).resolve())
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return the exact eight-command affected-check reduction."""

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
    """Load the active authorities and twelve predecessor evidence slots once."""

    contract = capstone.load_contract(ROOT)
    return contract, capstone.collect_evidence(ROOT, contract["tasks"])


@pytest.fixture(scope="module")
def artifact(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build one deterministic candidate for independent mutation checks."""

    contract, evidence = repository_state
    return capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        capstone.evaluate_publication_gates(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        completed_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-DISPOSITIONS
def test_exact_thirteen_contract_and_twelve_sources_are_authenticated(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Active YAML stays usable while the stale Markdown authority remains failed."""

    contract, evidence = repository_state
    assert contract["milestone"] == capstone.MILESTONE
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["comparison"]["passed"] is False
    assert contract["comparison"]["errors"]
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    assert all(row["authenticated"] for row in evidence.values())
    assert evidence["exp7430-extraction-audit"]["source_kind"] == "structured_pre_gate"
    assert evidence["exp7430-extraction-audit"]["available"] is False
    assert evidence["exp7430-extraction-audit"]["verdict_class"] == "blocked"
    assert evidence["exp7429-anchored-capture"]["verdict_class"] == "disqualified"
    assert evidence["exp7429-anchored-capture"]["flagged_adversarial"] is True
    assert evidence["exp7429-anchored-capture"]["required_validation_passed"] is False
    assert evidence["exp7426-static-decisions"]["raw_rows_available"] is True


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-DISPOSITIONS
def test_missing_and_malformed_evidence_fail_closed(tmp_path: Path) -> None:
    """Missing bytes and non-object JSON cannot manufacture science."""

    task = {
        "id": "exp7430-extraction-audit",
        "deliverable": "results/experiment_7430_extraction_audit.json",
        "gated_on": [],
    }
    missing = capstone.load_evidence_slot(tmp_path, task)
    assert missing["source_kind"] == "missing"
    assert missing["authenticated"] is False
    assert missing["verdict_class"] == "blocked"

    malformed = tmp_path / "bad.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        capstone.load_json_object(malformed)
    assert capstone._compare_gate("unsupported", 1, 1) is False
    assert capstone._required_validation({"required_checks_passed": True}) is True


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-DISPOSITIONS
def test_active_contract_order_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed active YAML order fails even though stale Markdown is retained."""

    roadmap = capstone.load_yaml(ROOT / capstone.ROADMAP_PATH)
    changed = deepcopy(roadmap)
    changed["tasks"] = list(changed["tasks"])
    changed["tasks"][0], changed["tasks"][1] = changed["tasks"][1], changed["tasks"][0]
    monkeypatch.setattr(capstone, "load_yaml", lambda _path: changed)
    with pytest.raises(ValueError, match="task order"):
        capstone.load_contract(ROOT)

    monkeypatch.setattr(capstone, "load_yaml", lambda _path: {"milestone": "old"})
    with pytest.raises(ValueError, match="active V651"):
        capstone.load_contract(ROOT)

    missing_path = deepcopy(roadmap)
    missing_path["tasks"] = deepcopy(missing_path["tasks"])
    del missing_path["tasks"][0]["deliverable"]
    monkeypatch.setattr(capstone, "load_yaml", lambda _path: missing_path)
    with pytest.raises(ValueError, match="path or milestone"):
        capstone.load_contract(ROOT)

    assert capstone._compare_gate("==", 1, 1) is True
    assert capstone._compare_gate(">=", 2, 1) is True
    assert capstone._compare_gate("in", "null", ["null"]) is True
    assert capstone._compare_gate("in", "null", "null") is False


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-CLAIMS
def test_twelve_claim_rows_keep_authority_and_limits_separate(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Null, circular, blocked, disqualified, sentinel, cost, and boards do not merge."""

    _, evidence = repository_state
    matrix = capstone.reduce_claim_matrix(evidence)
    assert list(matrix) == list(capstone.CLAIM_BRANCHES)
    assert matrix["human_label_protocol"]["completion_score"] == 1
    assert "human" in matrix["human_label_protocol"]["authority"]
    assert matrix["human_label_protocol"]["semantic_truth_claimed"] is False
    assert matrix["spline_equivalence"]["verdict_class"] == "circular_positive"
    assert matrix["spline_equivalence"]["general_ebm_superiority_claimed"] is False
    assert matrix["static_decisions"]["verdict_class"] == "null"
    assert matrix["static_decisions"]["benefit_score"] == 0
    assert matrix["online_learning"]["partial_feedback_bias_reported"] is True
    assert matrix["independent_audit"]["completion_score"] == 1
    assert matrix["qwen_extraction"]["verdict_class"] == "disqualified"
    assert matrix["qwen_extraction"]["flagged_adversarial"] is True
    assert matrix["extraction_audit"]["verdict_class"] == "blocked"
    assert matrix["arc_reachability"]["benefit_score"] == 0
    assert matrix["arc_reachability"]["hidden_performance_claimed"] is False
    assert matrix["host_update_cost"]["benefit_score"] == 0
    assert matrix["host_update_cost"]["hardware_benefit_claimed"] is False
    assert matrix["kv260"]["terminal_state"] == "graduated_preserved"
    assert matrix["gatemate"]["terminal_state"] == "blocked_changed_physical_state"
    assert matrix["polarfire"]["terminal_state"] == "graduated_cpu_dispatch_preserved"


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-DIAGNOSTICS
def test_diagnostics_retain_ablation_bias_sensitivity_support_and_costs(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """The capstone exposes the named diagnostic evidence without extrapolation."""

    diagnostics = capstone.diagnostic_summary(repository_state[1])
    assert diagnostics["static_source_conditions"]
    assert {"full_source", "source_masked", "source_swapped"} <= set(
        diagnostics["static_source_conditions"]
    )
    assert diagnostics["dense_basis_equivalence"]
    assert diagnostics["partial_feedback_selection_bias"]
    assert diagnostics["implicit_true_sensitivity"]
    assert set(diagnostics["per_domain_support"]) == {"Data2txt", "QA", "Summary"}
    assert diagnostics["complete_service_costs"]
    assert set(diagnostics["board_dispositions"]) == {"KV260", "GateMate", "PolarFire"}


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-CLASSIFY
def test_invalid_core_precedes_absence_and_current_failure_disqualifies(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Flagged extraction wins before blocked audit; absence alone remains blocked."""

    contract, evidence = repository_state
    terminal = capstone.classify_terminal(contract, evidence, _validation())
    assert terminal["verdict_class"] == "disqualified"
    assert terminal["gate_check_summary"]["first_failure"]["upstream"] == (
        "exp7429-anchored-capture"
    )

    valid_capture = deepcopy(evidence)
    valid_capture["exp7429-anchored-capture"].update(
        verdict_class="null",
        flagged_adversarial=False,
        required_validation_passed=True,
        available=True,
        valid=True,
    )
    matched_contract = deepcopy(contract)
    matched_contract["comparison"] = {"passed": True, "errors": []}
    blocked = capstone.classify_terminal(matched_contract, valid_capture, _validation())
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"]["upstream"] == (
        "exp7430-extraction-audit"
    )

    complete = deepcopy(valid_capture)
    complete["exp7430-extraction-audit"].update(
        verdict_class="null",
        required_validation_passed=True,
        available=True,
        valid=True,
    )
    stale_contract = capstone.classify_terminal(contract, complete, _validation())
    assert stale_contract["verdict_class"] == "disqualified"
    assert any(
        row["category"] == "contract_authority"
        for row in stale_contract["gate_check_summary"]["failures"]
    )

    fully_valid = capstone.classify_terminal(matched_contract, complete, _validation())
    assert fully_valid["verdict_class"] == "null"
    affected_failure = capstone.classify_terminal(matched_contract, complete, _validation(False))
    assert affected_failure["verdict_class"] == "disqualified"
    assert affected_failure["gate_check_summary"]["first_failure"]["category"] == (
        "required_validation"
    )


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-CONTINUATION
def test_continuations_compare_exact_verdicts_and_wait_for_external_absence(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Only an exact repeated mechanism retires; external and deferred work remain named."""

    contract, evidence = repository_state
    rows = capstone.retirement_rows(
        contract["tasks"], "complete_disqualified_required_v651_science"
    )
    assert rows
    assert rows[0]["same_exact_verdict"] is False
    assert rows[0]["decision"] == "continue-with-measured-cause"
    repeated = capstone.retirement_rows(contract["tasks"], rows[0]["previous_verdict"])
    assert repeated[0]["decision"] == "retire-unchanged-mechanism"

    capstone_prior = next(row for row in rows if row["prior_experiment_id"] == "exp7420-capstone")
    repeated_capstone = capstone.retirement_rows(
        contract["tasks"], capstone_prior["previous_verdict"]
    )
    retired = capstone.continuation_rows(capstone.reduce_claim_matrix(evidence), repeated_capstone)
    assert {row["claim"]: row["decision"] for row in retired}["qwen_extraction"] == (
        "retire-unchanged-mechanism"
    )

    continuation = capstone.continuation_rows(capstone.reduce_claim_matrix(evidence), rows)
    assert {row["decision"] for row in continuation} <= set(capstone.CONTINUATION_DECISIONS)
    by_claim = {row["claim"]: row for row in continuation}
    assert by_claim["extraction_audit"]["decision"] == "wait-for-prerequisite"
    assert by_claim["qwen_extraction"]["decision"] == "continue-with-measured-cause"
    assert by_claim["gatemate"]["decision"] == "wait-for-prerequisite"
    scope = capstone.scope_reduction_compliance(evidence)
    assert scope["v650_proof_memory"]["upper_cost_ratio"] == 3.7068
    assert scope["v650_proof_memory"]["decision"] == "defer-until-changed-cost-profile"
    assert scope["v649_proof_memory_chain"]["decision"] == "remain-retired"


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-PUBLICATION
def test_publication_gates_keep_their_existing_scope() -> None:
    """The unchanged G1-G4 result is retained but explicitly does not certify V651."""

    result = capstone.evaluate_publication_gates()
    assert set(result["gates"]) == {"G1", "G2", "G3", "G4"}
    assert result["paper_ready"] is True
    assert result["unmet_gates"] == []
    row = capstone.publication_gate_row(result)
    assert row["headline_scope"] == "FoVer dual-condition AUROC"
    assert row["certifies_v651"] is False


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-ARTIFACT
def test_artifact_has_thirteen_dispositions_zero_promotion_and_exact_schema(
    artifact: dict[str, Any],
) -> None:
    """All required fields are ordinary values with no current model work."""

    assert artifact["schema"] == capstone.SCHEMA
    assert artifact["status"].startswith("complete_disqualified")
    assert artifact["honest_verdict"].startswith("complete_disqualified")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["promotion_score"] == 0
    assert artifact["capstone_complete_score"] == 0
    assert len(artifact["task_dispositions"]) == 13
    assert [row["task_id"] for row in artifact["task_dispositions"]] == list(
        capstone.EXPECTED_TASK_IDS
    )
    assert list(artifact["claim_matrix"]) == list(capstone.CLAIM_BRANCHES)
    assert artifact["publication_gates"]["certifies_v651"] is False
    assert capstone.validate_artifact(artifact, root=ROOT) == []


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-ARTIFACT
def test_cold_validator_rejects_control_field_drift(artifact: dict[str, Any]) -> None:
    """Cold replay rejects identity, evidence, conclusion, principle, and checksum drift."""

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
        lambda value: value.__setitem__("task_dispositions", [])
    )
    assert "source_hash_mismatch" in errors_after(
        lambda value: value["source_artifact_hashes"]["exp7426-static-decisions"].__setitem__(
            "sha256", "sha256:" + "0" * 64
        )
    )
    assert "claim_matrix_invalid" in errors_after(
        lambda value: value["claim_matrix"]["host_update_cost"].__setitem__(
            "hardware_benefit_claimed", True
        )
    )
    assert "publication_gates_invalid" in errors_after(
        lambda value: value["publication_gates"].__setitem__("certifies_v651", True)
    )
    assert "capstone_score_invalid" in errors_after(
        lambda value: value.__setitem__("capstone_complete_score", 1)
    )
    assert "field_principles_invalid" in errors_after(
        lambda value: value.__setitem__("field_principles", {})
    )
    assert "lifecycle_invalid" in errors_after(lambda value: value.__setitem__("status", "running"))
    assert "substrate_invalid" in errors_after(
        lambda value: value.__setitem__("execution_venue", "host_cpu")
    )
    assert "promotion_invalid" in errors_after(
        lambda value: value.__setitem__("promotion_score", 1)
    )
    assert "retirement_rows_invalid" in errors_after(
        lambda value: value.__setitem__("retirement_rows", [])
    )
    assert "continuation_rows_invalid" in errors_after(
        lambda value: value.__setitem__("continuation_rows", [])
    )
    assert "scope_reduction_invalid" in errors_after(
        lambda value: value.__setitem__("scope_reduction_compliance", {})
    )
    assert "terminal_reduction_invalid" in errors_after(
        lambda value: value.__setitem__("honest_verdict", "complete_wrong")
    )
    assert "reproducibility_checksum_invalid" in errors_after(
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:" + "0" * 64),
        refresh=False,
    )
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    assert capstone.validate_artifact({"schema": capstone.SCHEMA}, root=ROOT)[0].startswith(
        "missing_required_field:"
    )

    flagged = deepcopy(artifact)
    flagged["flagged_adversarial"] = False
    contract = capstone.load_contract(ROOT)
    evidence = capstone.collect_evidence(ROOT, contract["tasks"])
    rebuilt = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        capstone.evaluate_publication_gates(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        completed_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
        flagged_adversarial=True,
    )
    assert rebuilt["flagged_adversarial"] is True

    original = capstone.load_contract
    capstone.load_contract = lambda _root: (_ for _ in ()).throw(ValueError("changed"))
    try:
        assert "independent_reduction_failed" in capstone.validate_artifact(artifact, root=ROOT)
    finally:
        capstone.load_contract = original


# REQ-REPORT-7433 / SCENARIO-REPORT-7433-ARTIFACT
def test_hash_reader_scoped_plan_and_public_cold_replay(
    tmp_path: Path,
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sources stay file-backed and the shared scoped runner remains narrow."""

    assert capstone._hashes_match({"source_artifact_hashes": []}, ROOT) is False
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["bad"] = "not-a-row"
    assert capstone._hashes_match(changed, ROOT) is False
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["contract-comparison"]["sha256"] = "sha256:" + "0" * 64
    assert capstone._hashes_match(changed, ROOT) is False
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["current-module"]["path"] = "bad + path"
    assert capstone._hashes_match(changed, ROOT) is False

    original = capstone.load_contract
    capstone.load_contract = lambda _root: (_ for _ in ()).throw(ValueError("changed"))
    try:
        assert capstone._hashes_match(artifact, ROOT) is False
    finally:
        capstone.load_contract = original

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

    candidate = tmp_path / "candidate.json"
    capstone.atomic_json(candidate, artifact)
    assert capstone.main(["--date", capstone.RUN_DATE, "--validate", str(candidate)]) == 0
    assert capstone.main(["--date", capstone.RUN_DATE, "--validate", str(tmp_path / "none")]) == 1
    with pytest.raises(SystemExit):
        capstone.main(["--date", "wrong"])

    non_object = tmp_path / "non-object.json"
    non_object.write_text(json.dumps([]), encoding="utf-8")
    assert capstone.main(["--date", capstone.RUN_DATE, "--validate", str(non_object)]) == 1

    monkeypatch.setattr(
        capstone,
        "run_experiment",
        lambda *_args: {
            "status": "complete",
            "verdict_class": "disqualified",
            "capstone_complete_score": 0,
        },
    )
    assert capstone.main(["--date", capstone.RUN_DATE]) == 0
