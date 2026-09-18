"""Behavior tests for the V648 capstone.

Spec refs: REQ-REPORT-7394 and SCENARIO-REPORT-7394-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import pytest

from carnot import experiment_7394_v648_capstone as capstone
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one bounded command receipt without starting a nested process."""

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
            "carnot.experiment_7394_v648_capstone": str((ROOT / capstone.MODULE_PATH).resolve())
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return the exact affected receipt set expected from Exp7303."""

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
    """Load the exact V648 authorities once for repository-backed checks."""

    contract = capstone.load_contract(ROOT)
    return contract, capstone.collect_evidence(ROOT, contract["tasks"])


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-DISPOSITIONS
def test_exact_contract_and_real_pregates_fill_thirteen_slots(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Each predecessor uses exact bytes or one authenticated conductor record."""

    contract, evidence = repository_state
    assert contract["milestone"] == capstone.MILESTONE
    assert contract["contract_match"] is True
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    assert evidence["exp7387-decision-audit"]["source_kind"] == ("conductor_pre_gate_artifact")
    assert evidence["exp7388-proposal-capture"]["actual_path"] == (
        "results/experiment_7388_proposal_capture.json"
    )
    assert evidence["exp7389-proof-learning"]["source_kind"] == "conductor_log_record"
    assert evidence["exp7390-proof-audit"]["source_kind"] == "conductor_log_record"
    assert evidence["exp7391-arc-generalization"]["source_kind"] == ("conductor_pre_gate_artifact")
    assert all(row["authenticated"] for row in evidence.values())


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-DISPOSITIONS
def test_missing_or_drifted_evidence_fails_closed(tmp_path: Path) -> None:
    """Missing producer bytes stay blocked and similar prose is not authenticated."""

    task = {
        "id": "exp7389-proof-learning",
        "title": "Measure prospective implication-memory value on sealed later requests",
        "deliverable": "results/experiment_7389_v648_proof_learning.json",
    }
    missing = capstone.load_evidence_slot(tmp_path, task)
    assert missing["source_kind"] == "missing"
    assert missing["verdict_class"] == "blocked"
    assert missing["authenticated"] is False

    log = tmp_path / capstone.CONDUCTOR_PATH
    log.parent.mkdir(parents=True)
    log.write_text("similar but invented gate line\n", encoding="utf-8")
    assert capstone.load_evidence_slot(tmp_path, task)["source_kind"] == "missing"

    ordinary = {
        "id": "exp7381-contract",
        "title": "Bind V648 sources and the exact fourteen-task contract",
        "deliverable": "results/missing.json",
    }
    assert capstone.load_evidence_slot(tmp_path, ordinary)["source_kind"] == "missing"
    unknown = {"id": "exp0000-unknown", "deliverable": "results/missing.json"}
    assert capstone.load_evidence_slot(tmp_path, unknown)["source_kind"] == "missing"
    with pytest.raises(ValueError, match="numeric experiment identity missing"):
        capstone._numeric_experiment_id("unknown")
    malformed = tmp_path / "list.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        capstone._load_json(malformed)


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-DISPOSITIONS
def test_contract_rejects_wrong_milestone_and_order(tmp_path: Path) -> None:
    """The active YAML and named design must contain the exact fourteen rows."""

    (tmp_path / capstone.ROADMAP_PATH).write_text("milestone: 2026.09.647\ntasks: []\n")
    (tmp_path / capstone.DESIGN_PATH).parent.mkdir(parents=True)
    (tmp_path / capstone.DESIGN_PATH).write_text("# wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="milestone"):
        capstone.load_contract(tmp_path)

    (tmp_path / capstone.ROADMAP_PATH).write_text(
        f"milestone: {capstone.MILESTONE}\ntasks: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fourteen"):
        capstone.load_contract(tmp_path)


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-TERMINAL
def test_terminal_reduction_prioritizes_required_failures_over_absence(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Failed required validation disqualifies; unchanged missing science blocks."""

    _, evidence = repository_state
    current = capstone.terminal_state(evidence, required_validation_passed=True)
    assert current["verdict_class"] == "disqualified"
    assert current["required_science_complete_score"] == 0
    failures = current["gate_check_summary"]["failures"]
    assert any(row["upstream"] == "exp7386-online-decisions" for row in failures)
    assert any(row["upstream"] == "exp7389-proof-learning" for row in failures)
    assert all(row["observed"] != "retryable_partial" for row in failures)

    external_only = deepcopy(evidence)
    for task_id in ("exp7386-online-decisions",):
        external_only[task_id]["verdict_class"] = "null"
        external_only[task_id]["flagged_adversarial"] = False
        external_only[task_id]["authenticated"] = True
        external_only[task_id]["accepted_for_science"] = True
        external_only[task_id]["payload"] = {
            **external_only[task_id]["payload"],
            "verdict_class": "null",
            "flagged_adversarial": False,
            "online_capture_complete_score": 1,
        }
    blocked = capstone.terminal_state(external_only, required_validation_passed=True)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"]["upstream"] == ("exp7387-decision-audit")

    own_failure = capstone.terminal_state(external_only, required_validation_passed=False)
    assert own_failure["verdict_class"] == "disqualified"
    assert own_failure["gate_check_summary"]["first_failure"]["upstream"] == capstone.EXPERIMENT_ID

    static_failed = deepcopy(external_only)
    static_failed["exp7385-decision-training"]["accepted_for_science"] = False
    static_failed["exp7385-decision-training"]["verdict_class"] = "disqualified"
    static_state = capstone.terminal_state(static_failed, required_validation_passed=True)
    assert any(
        row["upstream"] == "exp7385-decision-training"
        for row in static_state["gate_check_summary"]["failures"]
    )

    completed = deepcopy(external_only)
    for task_id in ("exp7387-decision-audit", "exp7389-proof-learning", "exp7390-proof-audit"):
        completed[task_id]["authenticated"] = True
        completed[task_id]["accepted_for_science"] = True
        completed[task_id]["verdict_class"] = "null"
    done = capstone.terminal_state(completed, required_validation_passed=True)
    assert done["verdict_class"] == "null"
    assert done["required_science_complete_score"] == 1


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-CLAIMS
def test_claim_reduction_preserves_branch_boundaries(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """Static null, unavailable proof work, ARC block, and Ising failure stay distinct."""

    _, evidence = repository_state
    rows = capstone.reduce_claim_rows(evidence)
    by_branch = {row["branch"]: row for row in rows}
    assert list(by_branch) == list(capstone.CLAIM_BRANCHES)

    static = by_branch["static_calibration"]
    assert static["completion_score"] == 1
    assert static["value_score"] == 0
    assert static["metrics"]["verdict_class"] == "null"
    assert static["metrics"]["independent_external_test"] is False

    online = by_branch["online_learning"]
    assert online["completion_score"] == 0
    assert online["metrics"]["online_capture_complete_score"] == 0
    assert online["metrics"]["audit_available"] is False

    proof = by_branch["proof_memory"]
    assert proof["completion_score"] == 0
    assert proof["metrics"]["measurement_available"] is False
    assert proof["metrics"]["audit_available"] is False
    assert proof["verifier_is_oracle"] is True

    arc = by_branch["arc_reachability_and_generalization"]
    assert arc["metrics"]["invocation_ready_score"] == 0
    assert arc["metrics"]["generalization_started"] is False

    finite = by_branch["archived_finite_laws"]
    assert finite["completion_score"] == 0
    assert finite["metrics"]["verdict_class"] == "disqualified"

    hardware = by_branch["hardware_placement"]
    assert hardware["metrics"]["board_disposition_complete_score"] == 1
    assert hardware["value_score"] == 0


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-PUBLICATION
def test_publication_result_preserves_fover_scope_without_v648_authority() -> None:
    """Passing historical G1-G4 values do not certify V648 or deployment."""

    result = capstone.publication_gate_results(_publication())
    assert list(result["gates"]) == ["G1", "G2", "G3", "G4"]
    assert result["paper_ready"] is True
    assert result["scope"] == "historical_fover_paper_only"
    assert result["certifies_v648"] is False
    assert result["authorizes_external_publication"] is False
    assert result["authorizes_deployment"] is False

    malformed = capstone.publication_gate_results({"gates": {}})
    assert malformed["paper_ready"] is False
    assert malformed["unmet_gates"] == ["G1", "G2", "G3", "G4"]


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-ARTIFACT
def test_terminal_artifact_has_fourteen_dispositions_and_replays(
    repository_state: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    """The terminal self-row, exact hashes, scores, and checksum replay together."""

    contract, evidence = repository_state
    artifact = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        _validation(),
        _publication(),
        started_at_utc="2026-09-18T13:31:00+00:00",
        completed_at_utc="2026-09-18T13:31:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )
    assert artifact["schema"] == capstone.SCHEMA
    assert artifact["status"].startswith("complete_disqualified")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert isinstance(artifact["inference_substrate"], str)
    assert "host CPU" in artifact["inference_substrate"]
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["promotion_score"] == 0
    assert artifact["milestone_disposition_complete_score"] == 1
    assert artifact["required_science_complete_score"] == 0
    assert len(artifact["disposition_rows"]) == 14
    assert artifact["disposition_rows"][-1]["task_id"] == capstone.EXPERIMENT_ID
    assert artifact["disposition_rows"][-1]["numeric_experiment_id"] == 7394
    assert capstone.validate_artifact(artifact, root=ROOT) == []

    assert (
        len(
            capstone.build_disposition_rows(
                contract["tasks"], evidence, capstone.terminal_state(evidence, True), False
            )
        )
        == 13
    )

    for field, value, error in (
        ("MODEL_SPECS", ["unexpected"], "model_contract_invalid"),
        ("execution_venue", "host_cpu", "substrate_invalid"),
        ("readiness_score", 1, "failed_state_score_nonzero"),
        ("reproducibility_checksum", "sha256:" + "0" * 64, "reproducibility_checksum_invalid"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["disposition_rows"][8]["verdict_class"] = "positive"
    assert "disposition_rows_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = []
    assert "source_hash_mismatch" in capstone.validate_artifact(changed, root=ROOT)
    for mutation in (
        lambda rows: rows.__setitem__("active_roadmap", "not a row"),
        lambda rows: rows["active_roadmap"].__setitem__("path", None),
        lambda rows: rows["exp7389-proof-learning"].__setitem__(
            "source_file_sha256", "sha256:" + "0" * 64
        ),
        lambda rows: rows["exp7389-proof-learning"].__setitem__("sha256", "sha256:" + "0" * 64),
        lambda rows: rows["exp7385-decision-training"].__setitem__("sha256", "sha256:" + "0" * 64),
    ):
        changed = deepcopy(artifact)
        mutation(changed["source_artifact_hashes"])
        assert "source_hash_mismatch" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["milestone"] = "2026.09.000"
    assert "identity_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["status"] = "running"
    assert "lifecycle_invalid" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    del changed["schema"]
    assert capstone.validate_artifact(changed, root=ROOT) == ["missing_required_field:schema"]
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    assert capstone._row_failures({"payload": {"gate_check_summary": "failed"}}) == ["failed"]


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-ARTIFACT
def test_exp7358_plan_is_exact_and_entrypoint_is_thin(tmp_path: Path) -> None:
    """The command plan stays affected-only and the launcher delegates once."""

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
    assert "experiment_7394_v648_capstone import main" in source
    assert source.count("main()") == 1

    duplicate = [*commands, commands[0]]
    assert "duplicate_command:worktree_imports" in capstone.validate_validation_plan(
        ROOT, duplicate
    )


# REQ-REPORT-7394 / SCENARIO-REPORT-7394-ARTIFACT
def test_atomic_write_and_public_boundaries(tmp_path: Path) -> None:
    """Utility boundaries keep JSON atomic, timestamps aware, and dates frozen."""

    path = tmp_path / "nested" / "artifact.json"
    capstone.atomic_json(path, {"complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    assert capstone.utc_now().endswith("+00:00")
    capstone.progress(time.monotonic(), "test", "boundary", units=1)
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        capstone.date_argument("20260917")
