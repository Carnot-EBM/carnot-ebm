"""REQ-REPORT-7650: Keep source evidence and scientific benefit separate."""

from pathlib import Path
import json
import runpy
import sys

import pytest

import carnot.experiment_7650_v667_independent_source_audit as audit
from carnot.experiment_7650_v667_independent_source_audit import (
    ROOT,
    collect_sources,
    reduce_available,
    run_private_mutations,
    validate_terminal,
)


def test_source_custody_is_literal() -> None:
    """SCENARIO-REPORT-7650-CUSTODY: A pre-gate is not a producer."""

    sources, checks, hashes = collect_sources(ROOT)
    assert [source["disposition"] for source in sources] == [
        "circular_positive",
        "disqualified",
        "pre_gate_blocked",
        "missing",
        "missing",
    ]
    assert len(checks) >= 3
    assert all(
        set(check) >= {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        for check in checks
    )
    assert len(hashes["producer_files"]) == 2
    assert len(hashes["pre_gate_receipts"]) == 1
    assert len(hashes["missing_inputs"]) == 2
    assert not hashes["planned_outputs"]


def test_raw_rows_recompute_fixture_and_coverage() -> None:
    """SCENARIO-REPORT-7650-REDUCTION: Arms cannot multiply source groups."""

    sources, _, _ = collect_sources(ROOT)
    findings, rows, budget = reduce_available(ROOT, sources)
    assert findings["exp7644"]["fixture_matches"] == 48
    assert findings["exp7644"]["fixture_denominator"] == 48
    assert findings["exp7646"]["independent_groups"] == 240
    assert findings["exp7646"]["arm_rows"] == 720
    assert findings["exp7646"]["unknown_prose"] >= 1
    assert budget["observed_independent_groups"] == 240
    assert any(row["arm"] == "original_source" and row["denominator"] >= 1 for row in rows)


def test_private_mutations_reject_all_registered_overclaims() -> None:
    """SCENARIO-REPORT-7650-MUTATIONS: Changed operands must fail."""

    results = run_private_mutations()
    assert set(results) == {
        "label_in_predictor_input",
        "evidence_wrong_file",
        "same_name_wrong_scope",
        "semantic_overclaim",
        "future_label_shuffle",
        "duplicate_admission",
        "permuted_role",
        "aggregate_disagreement",
    }
    assert all(
        row["rejected"] and row["changed_sha256"] != row["baseline_sha256"]
        for row in results.values()
    )


def test_label_control_fails_if_predictor_guard_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7650-MUTATIONS: The leakage reader is necessary."""

    monkeypatch.setattr(audit, "validate_predictor_input", lambda record: None)
    with pytest.raises(AssertionError, match="label_mutation_not_rejected"):
        audit.run_private_mutations()


def test_terminal_rejects_promoted_benefit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7650-TERMINAL: Blocked science cannot be positive."""

    sources, checks, hashes = collect_sources(ROOT)
    findings, rows, budget = reduce_available(ROOT, sources)
    candidate = {
        "honest_verdict": "complete_blocked_source_producers_unavailable",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": checks,
        "source_artifact_hashes": hashes,
        "independent_findings": findings,
        "rows": rows,
        "sample_size_budget": budget,
        "MODEL_SPECS": [],
        "model_invoked": False,
    }
    assert validate_terminal(candidate, ROOT) == []
    candidate["verdict_class"] = "positive"
    assert "blocked_verdict_mismatch" in validate_terminal(candidate, ROOT)
    candidate["verdict_class"] = "blocked"
    candidate["rows"] = []
    assert "row_reduction_mismatch" in validate_terminal(candidate, ROOT)


def test_terminal_rejects_custody_gates_and_inference(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7650-TERMINAL: Each claim has its own guard."""

    value = audit.build_artifact(ROOT, "20260925", valid=True)
    assert validate_terminal(value, ROOT) == []
    assert [gate["gate"] for gate in value["acceptance_gate_results"]] == [
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    ]
    for field, replacement, expected_error in [
        ("gate_check_summary", [], "source_custody_mismatch"),
        ("independent_findings", {}, "finding_mismatch"),
        ("acceptance_gate_results", [], "acceptance_gate_mismatch"),
        ("private_mutations", {}, "mutation_mismatch"),
        ("flagged_adversarial", True, "terminal_claim_mismatch"),
        ("MODEL_SPECS", ["invented"], "current_inference_mismatch"),
    ]:
        changed = {**value, field: replacement}
        assert expected_error in validate_terminal(changed, ROOT)
    candidate = tmp_path / "candidate.json"
    audit.atomic_json(candidate, value)
    assert audit.cold_replay(candidate) == []
    assert audit.independent_replay(candidate) == []
    changed = {**value, "reproducibility_checksum": "wrong"}
    audit.atomic_json(candidate, changed)
    assert "reproducibility_checksum_mismatch" in audit.independent_replay(candidate)
    audit.atomic_json(candidate, value)
    original = audit.run_private_mutations
    audit.run_private_mutations = lambda: {
        "x": {"rejected": False, "baseline_sha256": "a", "changed_sha256": "b"}
    }
    try:
        assert "private_mutation_failure" in audit.independent_replay(candidate)
    finally:
        audit.run_private_mutations = original


def test_disqualified_validation_stays_disqualified() -> None:
    """REQ-REPORT-7650: Failed owned checks are not external absence."""

    value = audit.build_artifact(ROOT, "20260925", valid=False)
    assert value["verdict_class"] == "disqualified"
    assert value["honest_verdict"] == "complete_disqualified_required_validation"
    assert value["independent_audit_complete_score"] == 0
    assert audit.validate_terminal(value, ROOT) == []


def test_lifecycle_controls_and_cli_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7650-MUTATIONS: Release and use order matters."""

    assert (
        audit.causal_errors(
            [{"id": "a", "prediction_step": 1, "release_step": 2, "update_step": 3}]
        )
        == []
    )
    candidate = tmp_path / "candidate.json"
    audit.atomic_json(candidate, audit.build_artifact(ROOT, "20260925", valid=True))
    assert audit.main(["--cold", str(candidate)]) == 0
    assert audit.main(["--independent", str(candidate)]) == 0
    monkeypatch.setattr(audit, "run_experiment", lambda root, date, output: {})
    assert audit.main(["--date", "20260925", "--output", str(tmp_path / "out.json")]) == 0
    commands = audit.terminal_commands(ROOT, candidate)
    assert len(commands) == 4
    assert all(str(candidate) in command.argv for command in commands)


def test_owned_orchestration_publishes_exact_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7650-TERMINAL: Reader results precede atomic output."""

    monkeypatch.setattr(audit, "RAW", tmp_path / "raw")
    monkeypatch.setattr(audit, "NOTE", tmp_path / "note.md")
    monkeypatch.setattr(
        audit.checks,
        "run_scoped_validation",
        lambda *args, **kwargs: {"required_checks_passed": True, "validation_receipts": []},
    )
    monkeypatch.setattr(
        audit.checks,
        "run_commands",
        lambda *args, **kwargs: [{"name": "reader", "passed": True, "exit_code": 0}],
    )
    output = tmp_path / "result.json"
    value = audit.run_experiment(ROOT, "20260925", output)
    assert json.loads(output.read_text()) == value
    assert (tmp_path / "note.md").is_file()
    outcomes = json.loads((tmp_path / "raw/terminal_reader_outcomes.json").read_text())
    assert outcomes["all_passed"]
    assert outcomes["candidate_sha256"] == audit.sha256_file(
        tmp_path / "raw/terminal_candidate.json"
    )


def test_owned_orchestration_rejects_failed_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7650-TERMINAL: A failed reader blocks publication."""

    monkeypatch.setattr(audit, "RAW", tmp_path / "raw")
    monkeypatch.setattr(audit, "NOTE", tmp_path / "note.md")
    monkeypatch.setattr(
        audit.checks,
        "run_scoped_validation",
        lambda *args, **kwargs: {"required_checks_passed": True, "validation_receipts": []},
    )
    monkeypatch.setattr(
        audit.checks,
        "run_commands",
        lambda *args, **kwargs: [{"name": "reader", "passed": False, "exit_code": 1}],
    )
    with pytest.raises(RuntimeError, match="terminal_reader_failed"):
        audit.run_experiment(ROOT, "20260925", tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()


def test_module_entrypoint_runs_cold_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7650-TERMINAL: The declared module CLI checks bytes."""

    candidate = tmp_path / "candidate.json"
    audit.atomic_json(candidate, audit.build_artifact(ROOT, "20260925", valid=True))
    monkeypatch.setattr(sys, "argv", ["audit", "--cold", str(candidate)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(audit.__name__, run_name="__main__")
    assert exc.value.code == 0
