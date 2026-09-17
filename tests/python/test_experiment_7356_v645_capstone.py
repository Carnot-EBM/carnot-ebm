"""Behavior tests for the V645 independent capstone reducer.

Spec refs: REQ-REPORT-7356 and SCENARIO-REPORT-7356-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7356_v645_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the fixed contract and exact evidence once for repository tests."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_evidence(ROOT, contract["tasks"])
    return contract, evidence


def test_contract_has_fourteen_exact_rows(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-CONTRACT."""

    contract, _ = repository_state
    assert contract["passed"] is True
    assert contract["selected_yaml_path"] == "research-roadmap.yaml"
    assert [row["unit_id"] for row in contract["contract_rows"]] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["tasks"]) == 14
    assert contract["tasks"][-1]["id"] == "exp7356-capstone"

    changed = deepcopy(contract["yaml_document"])
    changed["tasks"][11]["deliverable"] += ".wrong"
    replay = capstone.evaluate_contract(contract["markdown_text"], changed)
    assert replay["passed"] is False
    assert replay["contract_rows"][11]["failures"] == ["deliverable"]


def test_exact_artifacts_and_canonical_gate_records_are_distinct(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-EVIDENCE."""

    _, evidence = repository_state
    declared = {
        "exp7343-contract",
        "exp7344-executor-fixture",
        "exp7345-arc-resume-check",
        "exp7346-learning-adapter",
        "exp7347-plan-canary",
        "exp7348-plan-capture",
        "exp7351-acquisition-prototype",
        "exp7354-arc-transfer",
        "exp7355-board-state",
    }
    blocked = {
        "exp7349-prospective-learning",
        "exp7350-learning-audit",
        "exp7352-acquisition-cost",
        "exp7353-acquisition-audit",
    }
    assert set(evidence) == set(capstone.EXPECTED_TASK_IDS[:-1])
    assert {
        task for task, row in evidence.items() if row["evidence_source"] == "declared_deliverable"
    } == declared
    assert {
        task
        for task, row in evidence.items()
        if row["evidence_source"] == "canonical_conductor_block"
    } == blocked
    assert all(evidence[task]["record_sha256"].startswith("sha256:") for task in blocked)
    assert evidence["exp7344-executor-fixture"]["disposition_class"] == "circular_positive"
    assert evidence["exp7348-plan-capture"]["adversarially_flagged"] is True
    assert evidence["exp7354-arc-transfer"]["accepted_for_reduction"] is False
    assert evidence["exp7355-board-state"]["disposition_class"] == "blocked"


def test_terminal_eligibility_outranks_scores_and_gate_values() -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-EVIDENCE."""

    payload = {
        "status": "complete",
        "milestone": capstone.MILESTONE,
        "experiment_id": "exp7351-acquisition-prototype",
        "verdict_class": "positive",
        "flagged_adversarial": True,
        "acquisition_prototype_ready_score": 1,
    }
    result = capstone.classify_payload("exp7351-acquisition-prototype", payload, quarantined=False)
    assert result == (True, "disqualified", False, True)

    task = {
        "id": "consumer",
        "gated_on": [{"upstream": "producer", "artifact_field": "ready", "op": "==", "value": 1}],
    }
    source = {
        "payload": {"ready": 1},
        "selected_evidence_path": "producer.json",
        "artifact_sha256": "sha256:abc",
        "disposition_class": "positive",
        "quarantined": False,
        "adversarially_flagged": True,
    }
    assert capstone.replay_gates([task], {"producer": source})[0]["outcome"] == (
        "flagged_adversarial"
    )
    source["adversarially_flagged"] = False
    assert capstone.replay_gates([task], {"producer": source})[0]["outcome"] == "passed"
    source["payload"] = {}
    assert capstone.replay_gates([task], {"producer": source})[0]["outcome"] == "missing_field"
    source["selected_evidence_path"] = None
    assert capstone.replay_gates([task], {"producer": source})[0]["outcome"] == "missing_file"

    circular = capstone._base_claim(
        "example",
        "producer",
        {
            "accepted_for_reduction": True,
            "payload": {"verdict_class": "positive", "verifier_is_oracle": True},
            "artifact_sha256": "sha256:abc",
        },
        {},
        {},
        [],
    )
    assert circular["verdict_class"] == "circular_positive"


def test_raw_reducer_keeps_five_claim_boundaries(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-CLAIMS."""

    _, evidence = repository_state
    rows = capstone.build_claim_matrix(evidence)
    assert [row["claim"] for row in rows] == list(capstone.CLAIM_NAMES)
    by_claim = {row["claim"]: row for row in rows}
    assert {name: row["verdict_class"] for name, row in by_claim.items()} == {
        "source_fidelity": "blocked",
        "continuous_learning": "blocked",
        "acquisition_cost": "blocked",
        "live_arc_causality": "blocked",
        "hardware_disposition": "blocked",
    }
    assert by_claim["source_fidelity"]["diagnostic_metrics"]["raw_unit_count"] == 128
    assert by_claim["continuous_learning"]["diagnostic_metrics"]["adapter_raw_row_count"] == 1664
    assert by_claim["continuous_learning"]["diagnostic_metrics"]["prospective_unit_count"] == 0
    assert by_claim["acquisition_cost"]["diagnostic_metrics"]["prototype_raw_row_count"] == 126
    assert by_claim["acquisition_cost"]["diagnostic_metrics"]["evaluation_row_count"] == 90
    assert by_claim["live_arc_causality"]["diagnostic_metrics"] == {
        "planned_episode_count": 4,
        "completed_episode_count": 4,
        "game_count": 2,
        "complete_causal_chain_count": 0,
        "treatment_level_total": 0,
        "withheld_level_total": 0,
        "paired_level_delta": 0,
    }
    hardware = by_claim["hardware_disposition"]
    assert hardware["diagnostic_metrics"]["board_row_count"] == 3
    assert hardware["diagnostic_metrics"]["disposition_complete_score"] == 1
    assert hardware["disposition_complete"] is True
    assert all(row["metric"] is None for row in rows[:4])
    assert hardware["metric"] == 1
    assert hardware["metric_kind"] == "disposition_accounting_only"
    assert all(row["promotes_scientific_efficacy"] is False for row in rows)


def test_expected_hardware_block_does_not_define_science_block(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-HARDWARE."""

    _, evidence = repository_state
    terminal = capstone.terminal_state(evidence, capstone.build_claim_matrix(evidence), True)
    assert terminal["status"] == "complete"
    assert terminal["verdict_class"] == "blocked"
    assert terminal["honest_verdict"].startswith("blocked_exp7348_plan_capture")
    failures = terminal["gate_check_summary"]["failures"]
    assert failures[0] == {
        "upstream": "exp7348-plan-capture",
        "failed_check": "plan_capture_complete_score",
        "artifact_field": "plan_capture_complete_score",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
        "terminal_blocking": True,
    }
    assert all(row["upstream"] != "exp7355-board-state" for row in failures)


def test_prior_verdicts_use_recorded_predecessor_bytes(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-RETIREMENTS."""

    contract, evidence = repository_state
    terminal = capstone.terminal_state(evidence, capstone.build_claim_matrix(evidence), True)
    rows = capstone.prior_failure_rows(contract["tasks"], evidence, terminal, ROOT)
    assert len(rows) == 15
    mismatches = {
        row["prior_experiment_id"]
        for row in rows
        if row["declared_prior_matches_recorded_predecessor"] is False
    }
    assert mismatches == {
        "exp7333-plan-capture",
        "exp7334-prospective-learning",
        "exp7335-learning-audit",
    }
    assert all(row["recorded_predecessor_sha256"].startswith("sha256:") for row in rows)
    assert all(row["decision"] in capstone.PRIOR_DECISIONS for row in rows)
    native = next(row for row in rows if row["prior_experiment_id"] == "exp7340-native-cost")
    assert native["declared_prior_matches_recorded_predecessor"] is True
    assert native["decision"] == "preserve_prior_boundary_current_upstream_block"

    decisions = capstone.next_branch_decisions(evidence, ROOT)
    by_branch = {row["branch"]: row for row in decisions}
    assert set(by_branch) == set(capstone.DECISION_BRANCHES)
    assert by_branch["native_tenfold_null"]["action"] == "preserve_retirement"
    assert by_branch["retired_joint_claim_comparison"]["action"] == "preserve_retirement"
    assert by_branch["delayed_feedback_mixture_retirement"]["action"] == ("preserve_retirement")
    assert by_branch["delayed_feedback_suffix_retirement"]["action"] == "preserve_retirement"
    assert by_branch["unsupported_hardware_boundary"]["action"] == "preserve_boundary"
    assert all(row["reopening_condition"] for row in decisions)


def _passing_validation() -> dict[str, object]:
    receipts = [
        {
            "name": name,
            "command": f"check {name}",
            "command_argv": ["check", name],
            "scope": "explicit",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_path": None,
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
            "output_tail": "",
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
            "status": "degraded_open",
            "incident_open": True,
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
        phase_spans=[capstone.phase_row("test", 0.0, 1.0, 14, None)],
    )


def test_artifact_build_replay_and_scope_reduction(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-ARTIFACT."""

    contract, evidence = repository_state
    artifact = _build(contract, evidence)
    assert capstone.validate_artifact(artifact, root=ROOT, replay=True) == []
    assert capstone.RUN_DATE == "20260917"
    assert artifact["run_date"] == "20260917"
    assert artifact["capstone_complete_score"] == 1
    assert artifact["capstone_readiness_score"] == 0
    assert artifact["capstone_value_score"] == 0
    assert artifact["capstone_promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == capstone.ZERO_INVOCATION_COUNTS
    scope = artifact["scope_reduction_compliance"]
    assert len(scope["focused_science_questions"]) == 3
    assert len(scope["independent_infrastructure_prerequisites"]) == 2
    assert scope["new_model_families"] == []
    assert scope["new_hardware_integrations"] == []
    assert scope["compliant"] is True
    assert artifact["publication_performed"] is False
    assert artifact["production_default_changed"] is False

    changed = deepcopy(artifact)
    changed["claim_matrix"][0]["verdict_class"] = "positive"
    assert "claim_matrix" in capstone.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum" in capstone.validate_artifact(changed, root=ROOT)


def test_validation_failure_disqualifies_but_keeps_accounting(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-ARTIFACT."""

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
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-EVIDENCE; SCENARIO-REPORT-7356-ARTIFACT."""

    target = tmp_path / "nested" / "artifact.json"
    capstone.atomic_write_json(target, {"value": 1})
    assert capstone.read_json(target) == {"value": 1}
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        capstone.read_json(bad)
    assert capstone.date_argument("20260917") == "20260917"
    with pytest.raises(ValueError, match="YYYYMMDD"):
        capstone.date_argument("2026-09-16")
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping"]
    assert capstone.receipt_set_passes({}) is False

    captured: list[capstone.scoped.CommandSpec] = []

    def fake_run(
        root: Path, commands: list[capstone.scoped.CommandSpec], *, log_dir: Path
    ) -> list[dict[str, object]]:
        assert root == ROOT
        assert log_dir == tmp_path / "terminal_validation"
        captured.extend(commands)
        return []

    monkeypatch.setattr(capstone.scoped, "run_commands", fake_run)
    assert capstone.run_terminal_commands(ROOT, tmp_path / "candidate.json", tmp_path) == []
    assert [command.name for command in captured] == [
        "independent_terminal_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]


def test_negative_evidence_and_terminal_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-EVIDENCE."""

    with pytest.raises(ValueError, match="no selected V645"):
        capstone.load_contract(tmp_path)
    assert capstone.task_number("other") is None
    assert capstone.classify_payload("other", {}, quarantined=False) == (
        False,
        "disqualified",
        False,
        False,
    )
    assert capstone.canonical_record(tmp_path, "unknown") is None

    log = tmp_path / capstone.CONDUCTOR_LOG_PATH
    log.parent.mkdir(parents=True)
    marker = capstone.CANONICAL_BLOCK_SPECS["exp7349-prospective-learning"]["marker"]
    log.write_text("unrelated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one canonical"):
        capstone.canonical_record(tmp_path, "exp7349-prospective-learning")
    log.write_text(marker + " | OK | detail |\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not GATE_BLOCK"):
        capstone.canonical_record(tmp_path, "exp7349-prospective-learning")

    missing = capstone.load_evidence(
        tmp_path,
        {"id": "exp7999-unknown", "deliverable": "results/missing.json"},
        {},
    )
    assert missing["evidence_source"] == "missing"

    eligible = {
        producer: {
            "accepted_for_reduction": True,
            "payload": {field: expected},
            "disposition_class": "null",
            "adversarially_flagged": False,
        }
        for producer, field, expected in capstone.REQUIRED_SCIENCE_GATES
    }
    circular = capstone.terminal_state(
        eligible,
        [{"claim": "science", "verdict_class": "circular_positive"}],
        True,
    )
    assert circular["verdict_class"] == "circular_positive"
    null = capstone.terminal_state(
        eligible,
        [{"claim": "science", "verdict_class": "null"}],
        True,
    )
    assert null["verdict_class"] == "null"

    duplicate_root = tmp_path / "duplicate"
    duplicate_results = duplicate_root / "results"
    duplicate_results.mkdir(parents=True)
    (duplicate_results / "experiment_7998_a.json").write_text("{}", encoding="utf-8")
    (duplicate_results / "experiment_7998_b.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="one exact predecessor"):
        capstone._predecessor_record(duplicate_root, "exp7998-example")

    missing_root = tmp_path / "missing-predecessor"
    missing_history = missing_root / capstone.V644_CAPSTONE_PATH
    missing_history.parent.mkdir(parents=True)
    missing_history.write_text('{"task_dispositions": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="one recorded predecessor"):
        capstone._predecessor_record(missing_root, "exp7997-example")

    repeat_root = tmp_path / "repeat"
    repeat_results = repeat_root / "results"
    repeat_results.mkdir(parents=True)
    (repeat_results / "experiment_7996_prior.json").write_text(
        '{"honest_verdict": "same"}', encoding="utf-8"
    )
    repeated = capstone.prior_failure_rows(
        [
            {
                "id": "exp7995-current",
                "prior_failures": [
                    {
                        "experiment_id": "exp7996-prior",
                        "verdict": "same",
                        "retire_if_same_verdict": True,
                        "addressed_by": "changed control",
                    }
                ],
            }
        ],
        {
            "exp7995-current": {
                "payload": {"honest_verdict": "same"},
                "canonical_record": None,
                "disposition_class": "null",
            }
        },
        {},
        repeat_root,
    )
    assert repeated[0]["decision"] == "retire_exact_repeat"

    assert (
        capstone._historical_failures(
            {"producer": {"payload": {"repository_health": {"historical_failures": [42]}}}}
        )
        == []
    )


def test_main_orchestrates_scoped_validation_and_atomic_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7356; SCENARIO-REPORT-7356-ARTIFACT."""

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
                "command_argv": ["gate"],
                "scope": "read_only",
                "exit_code": 0,
                "duration_s": 0.01,
                "log_path": None,
                "log_sha256": "sha256:" + "b" * 64,
                "passed": True,
                "timed_out": False,
                "output_tail": "",
            },
        ),
    )
    monkeypatch.setattr(
        capstone.scoped, "run_scoped_validation", lambda *args, **kwargs: _passing_validation()
    )
    monkeypatch.setattr(
        capstone,
        "run_terminal_commands",
        lambda *args: [{"name": "terminal", "passed": True, "exit_code": 0}],
    )
    monkeypatch.setattr(capstone, "validate_artifact", lambda *args, **kwargs: [])
    assert capstone.main(["--date", "20260917"]) == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["capstone_complete_score"] == 1
    assert saved["verdict_class"] == "blocked"
