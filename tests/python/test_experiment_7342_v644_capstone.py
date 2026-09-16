"""Behavior tests for the V644 independent capstone reducer.

Spec refs: REQ-REPORT-7342 and SCENARIO-REPORT-7342-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7342_v644_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the fixed contract and exact producer evidence once."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_evidence(ROOT, contract["tasks"])
    return contract, evidence


def test_contract_has_fourteen_exact_rows(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-CONTRACT."""

    contract, _ = repository_state
    assert contract["passed"] is True
    assert contract["selected_yaml_path"] == "research-roadmap.yaml"
    assert [row["unit_id"] for row in contract["contract_rows"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["tasks"][-1]["id"] == "exp7342-capstone"
    assert len(contract["tasks"]) == 14


def test_contract_mutation_fails_closed(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-CONTRACT."""

    contract, _ = repository_state
    changed = deepcopy(contract["yaml_document"])
    changed["tasks"][4]["deliverable"] += ".changed"
    evaluation = capstone.evaluate_contract(contract["markdown_text"], changed)
    assert evaluation["passed"] is False
    assert evaluation["contract_rows"][4]["failures"] == ["deliverable"]


def test_exact_artifacts_and_canonical_blocks_are_separate(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-EVIDENCE."""

    _, evidence = repository_state
    exact = {
        "exp7329-contract",
        "exp7330-executor-isolation",
        "exp7336-arc-resume",
        "exp7339-native-binding",
        "exp7340-native-cost",
        "exp7341-board-continuity",
    }
    blocked = {
        "exp7331-learning-adapter",
        "exp7332-plan-canary",
        "exp7333-plan-capture",
        "exp7334-prospective-learning",
        "exp7335-learning-audit",
        "exp7337-arc-transfer",
        "exp7338-arc-causal-audit",
    }
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    assert {
        task for task, row in evidence.items() if row["evidence_source"] == "declared_deliverable"
    } == exact
    assert {
        task
        for task, row in evidence.items()
        if row["evidence_source"] == "canonical_conductor_block"
    } == blocked
    assert all(evidence[task]["record_sha256"].startswith("sha256:") for task in blocked)
    assert evidence["exp7330-executor-isolation"]["disposition_class"] == "disqualified"
    assert evidence["exp7339-native-binding"]["disposition_class"] == "circular_positive"
    assert evidence["exp7340-native-cost"]["disposition_class"] == "null"
    assert evidence["exp7341-board-continuity"]["disposition_class"] == "blocked"


def test_fail_closed_classification_and_gate_replay() -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-EVIDENCE."""

    payload = {
        "status": "complete",
        "milestone": capstone.MILESTONE,
        "experiment_id": "exp7330-executor-isolation",
        "verdict_class": "disqualified",
        "executor_fixture_ready_score": 1,
    }
    assert capstone.classify_payload("exp7330-executor-isolation", payload, quarantined=False) == (
        True,
        "disqualified",
        False,
    )
    assert capstone.classify_payload("exp7330-executor-isolation", payload, quarantined=True) == (
        True,
        "quarantined",
        False,
    )

    task = {
        "id": "consumer",
        "gated_on": [{"upstream": "producer", "artifact_field": "ready", "op": "==", "value": 1}],
    }
    source = {
        "payload": {"ready": 1},
        "selected_evidence_path": "producer.json",
        "artifact_sha256": "sha256:abc",
        "disposition_class": "disqualified",
        "quarantined": False,
    }
    assert capstone.replay_gates([task], {"producer": source})[0]["outcome"] == "disqualified"
    source["disposition_class"] = "null"
    assert capstone.replay_gates([task], {"producer": source})[0]["outcome"] == "passed"
    source["payload"] = {}
    assert capstone.replay_gates([task], {"producer": source})[0]["outcome"] == "missing_field"


def test_claim_matrix_keeps_five_authority_classes_separate(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-CLAIMS."""

    _, evidence = repository_state
    rows = capstone.build_claim_matrix(evidence)
    assert [row["claim"] for row in rows] == list(capstone.CLAIM_NAMES)
    by_claim = {row["claim"]: row for row in rows}
    assert {name: row["verdict_class"] for name, row in by_claim.items()} == {
        "source_fidelity": "blocked",
        "acquired_constraint_learning": "blocked",
        "live_arc_causality": "blocked",
        "native_host_cost": "null",
        "board_evidence": "blocked",
    }
    assert by_claim["native_host_cost"]["metrics"]["paired_blocks"] == 90
    assert by_claim["native_host_cost"]["metrics"]["parity_mismatches"] == 0
    assert by_claim["native_host_cost"]["metrics"]["ten_x_lower_bound_passed"] is False
    assert by_claim["board_evidence"]["metrics"]["disposition_complete_score"] == 1
    assert by_claim["board_evidence"]["metrics"]["hardware_readiness_score"] == 0
    assert all(row["promotes_scientific_efficacy"] is False for row in rows)


def test_expected_board_block_does_not_define_capstone_block(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-BOARD."""

    _, evidence = repository_state
    terminal = capstone.terminal_state(evidence, capstone.build_claim_matrix(evidence), True)
    assert terminal["status"] == "complete"
    assert terminal["verdict_class"] == "blocked"
    assert terminal["honest_verdict"].startswith("blocked_exp7330_executor_isolation")
    failures = terminal["gate_check_summary"]["failures"]
    assert failures[0] == {
        "upstream": "exp7330-executor-isolation",
        "check": "executor_fixture_ready_score",
        "artifact_field": "executor_fixture_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "terminal_blocking": True,
    }
    assert all(row["upstream"] != "exp7341-board-continuity" for row in failures)


def test_prior_bytes_and_preserved_retirements_are_explicit(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-RETIREMENTS."""

    contract, evidence = repository_state
    terminal = capstone.terminal_state(evidence, capstone.build_claim_matrix(evidence), True)
    rows = capstone.prior_failure_rows(contract["tasks"], evidence, terminal)
    assert rows
    assert all(
        row["exact_repeat"]
        == (row["prior_honest_verdict_bytes"] == row["current_honest_verdict_bytes"])
        for row in rows
    )
    assert all(row["decision"] in capstone.PRIOR_DECISIONS for row in rows)

    decisions = capstone.next_branch_decisions(evidence, ROOT)
    assert [row["branch"] for row in decisions] == list(capstone.DECISION_BRANCHES)
    by_branch = {row["branch"]: row for row in decisions}
    assert by_branch["v643_source_comparison"]["action"] == "preserve_retirement"
    assert by_branch["v642_suffix_retirement"]["action"] == "preserve_retirement"
    assert by_branch["v642_storage_retirement"]["action"] == "preserve_retirement"
    assert all(row["reopening_condition"] for row in decisions)


def test_publication_gate_is_retained_without_action() -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-PUBLICATION."""

    publication, receipt = capstone.run_publication_gate(ROOT)
    assert publication["paper_ready"] is True
    assert publication["unmet_gates"] == []
    assert list(publication["gates"]) == ["G1", "G2", "G3", "G4"]
    assert receipt["exit_code"] == 0
    assert receipt["passed"] is True


def _passing_validation() -> dict[str, object]:
    receipts = [
        {
            "name": name,
            "command": f"check {name}",
            "scope": "explicit",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in capstone.REQUIRED_SCOPED_CHECKS
    ]
    return {
        "validation_receipts": receipts,
        "required_checks_passed": True,
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": {
            "status": "healthy",
            "incident_open": False,
            "historical_failures": [],
            "historical_failure_count": 0,
            "unresolved_collection_error_observation_count": 0,
            "affects_required_checks": False,
        },
    }


def _build(
    contract: dict[str, object],
    evidence: dict[str, dict[str, object]],
    validation: dict[str, object] | None = None,
) -> dict[str, object]:
    publication, receipt = capstone.run_publication_gate(ROOT)
    return capstone.build_artifact(
        root=ROOT,
        run_date=capstone.RUN_DATE,
        contract=contract,
        evidence=evidence,
        publication_gate=publication,
        publication_receipt=receipt,
        validation=validation or _passing_validation(),
        started_at="2026-09-16T00:00:00+00:00",
        completed_at="2026-09-16T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[capstone._phase_row("test", 0.0, 1.0, 14, None)],
    )


def test_artifact_build_and_cold_replay_detect_mutation(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-ARTIFACT."""

    contract, evidence = repository_state
    artifact = _build(contract, evidence)
    assert capstone.validate_artifact(artifact, root=ROOT, replay=True) == []
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capstone_readiness_score"] == 0
    assert artifact["capstone_value_score"] == 0
    assert artifact["capstone_promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == capstone.ZERO_INVOCATION_COUNTS
    assert artifact["publication_performed"] is False
    assert artifact["external_message_performed"] is False
    assert artifact["production_default_changed"] is False

    changed = deepcopy(artifact)
    changed["claim_matrix"][0]["verdict_class"] = "positive"
    assert "claim_matrix" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum" in capstone.validate_artifact(changed, root=ROOT)


def test_failed_validation_disqualifies_without_losing_dispositions(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-ARTIFACT."""

    contract, evidence = repository_state
    validation = _passing_validation()
    validation["required_checks_passed"] = False
    validation["failed_required_commands"] = ["focused_pytest"]
    artifact = _build(contract, evidence, validation)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_dispositions"]) == 14


def test_helpers_fail_closed_and_terminal_commands_are_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-EVIDENCE; SCENARIO-REPORT-7342-ARTIFACT."""

    target = tmp_path / "nested" / "artifact.json"
    capstone.atomic_write_json(target, {"value": 1})
    assert capstone.read_json(target) == {"value": 1}
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        capstone.read_json(bad)
    assert capstone.date_argument("20260916") == "20260916"
    with pytest.raises(ValueError, match="YYYYMMDD"):
        capstone.date_argument("2026-09-16")
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping"]
    assert capstone._receipt_set_passes({}) is False

    captured: list[capstone.scoped.CommandSpec] = []

    def fake_run(
        root: Path, commands: list[capstone.scoped.CommandSpec], *, log_dir: Path
    ) -> list[dict[str, object]]:
        assert root == ROOT
        assert log_dir == tmp_path / "terminal_validation"
        captured.extend(commands)
        return []

    monkeypatch.setattr(capstone.scoped, "run_commands", fake_run)
    assert capstone._run_terminal_commands(ROOT, tmp_path / "candidate.json", tmp_path) == []
    assert [command.name for command in captured] == [
        "independent_terminal_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]


def test_negative_evidence_and_terminal_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-EVIDENCE."""

    with pytest.raises(ValueError, match="no selected V644"):
        capstone.load_contract(tmp_path)
    assert capstone._task_number("other") is None
    assert capstone.classify_payload("other", {}, quarantined=False) == (
        False,
        "disqualified",
        False,
    )
    assert capstone._canonical_record(tmp_path, "unknown") is None

    log = tmp_path / capstone.CONDUCTOR_LOG_PATH
    log.parent.mkdir(parents=True)
    marker = capstone.CANONICAL_BLOCK_MARKERS["exp7331-learning-adapter"]
    log.write_text("unrelated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one canonical"):
        capstone._canonical_record(tmp_path, "exp7331-learning-adapter")
    log.write_text(marker + " | OK | detail |\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not GATE_BLOCK"):
        capstone._canonical_record(tmp_path, "exp7331-learning-adapter")
    good = marker + " | GATE_BLOCK | detail |\n"
    log.write_text(good + good, encoding="utf-8")
    with pytest.raises(ValueError, match="one canonical"):
        capstone._canonical_record(tmp_path, "exp7331-learning-adapter")

    missing = capstone.load_evidence(
        tmp_path,
        {"id": "exp7999-unknown", "deliverable": "results/missing.json"},
        {},
    )
    assert missing["evidence_source"] == "missing"

    task = {
        "id": "consumer",
        "gated_on": [{"upstream": "producer", "artifact_field": "ready", "op": "==", "value": 1}],
    }
    base = {
        "payload": {"ready": 1},
        "selected_evidence_path": "producer.json",
        "artifact_sha256": "sha256:abc",
        "disposition_class": "null",
        "quarantined": False,
    }
    outcomes = []
    for change in (
        {"selected_evidence_path": None},
        {"quarantined": True},
        {"disposition_class": "partial"},
    ):
        source = deepcopy(base)
        source.update(change)
        outcomes.append(capstone.replay_gates([task], {"producer": source})[0]["outcome"])
    assert outcomes == ["missing_file", "quarantined", "partial"]
    invalid = deepcopy(task)
    invalid["gated_on"][0]["op"] = "bad"
    assert capstone.replay_gates([invalid], {"producer": base})[0]["outcome"] == "value_mismatch"

    eligible = {
        "exp7330-executor-isolation": {
            "accepted_for_reduction": True,
            "payload": {"executor_fixture_ready_score": 1},
        },
        "exp7336-arc-resume": {
            "accepted_for_reduction": True,
            "payload": {"arc_resume_ready_score": 1},
        },
    }
    circular = capstone.terminal_state(
        eligible,
        [{"claim": "science", "verdict_class": "circular_positive"}],
        True,
    )
    assert circular["verdict_class"] == "circular_positive"
    null = capstone.terminal_state(eligible, [{"claim": "science", "verdict_class": "null"}], True)
    assert null["verdict_class"] == "null"

    repeated = capstone.prior_failure_rows(
        [
            {
                "id": "exp7999-example",
                "prior_failures": [
                    {
                        "experiment_id": "older",
                        "verdict": "same",
                        "retire_if_same_verdict": True,
                        "addressed_by": "changed control",
                    }
                ],
            }
        ],
        {
            "exp7999-example": {
                "payload": {"honest_verdict": "same"},
                "canonical_record": None,
                "disposition_class": "null",
            }
        },
        {},
    )
    assert repeated[0]["decision"] == "retire_exact_repeat"


def test_main_orchestrates_validation_and_atomic_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7342; SCENARIO-REPORT-7342-ARTIFACT."""

    contract, evidence = repository_state
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.json"
    raw = tmp_path / "raw"
    monkeypatch.setattr(capstone, "REPO_ROOT", ROOT)
    monkeypatch.setattr(capstone, "DEFAULT_OUTPUT_PATH", output)
    monkeypatch.setattr(capstone, "DEFAULT_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(capstone, "DEFAULT_RAW_DIR", raw)
    monkeypatch.setattr(capstone, "load_contract", lambda root: contract)
    monkeypatch.setattr(capstone, "load_repository_evidence", lambda root, tasks: evidence)
    monkeypatch.setattr(
        capstone,
        "run_publication_gate",
        lambda root: (
            {
                "paper_ready": True,
                "gates": {key: {"pass": True} for key in ("G1", "G2", "G3", "G4")},
                "unmet_gates": [],
            },
            {
                "name": "publication_gate",
                "command": "gate",
                "scope": "read_only",
                "exit_code": 0,
                "duration_s": 0.01,
                "log_sha256": "sha256:" + "b" * 64,
                "passed": True,
                "timed_out": False,
            },
        ),
    )
    monkeypatch.setattr(
        capstone.scoped, "run_scoped_validation", lambda *args, **kwargs: _passing_validation()
    )
    monkeypatch.setattr(
        capstone,
        "_run_terminal_commands",
        lambda *args: [{"name": "terminal", "passed": True, "exit_code": 0}],
    )
    monkeypatch.setattr(capstone, "validate_artifact", lambda *args, **kwargs: [])
    assert capstone.main(["--date", "20260916"]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["capstone_complete_score"] == 1
