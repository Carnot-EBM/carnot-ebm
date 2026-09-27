"""REQ-REPORT-7732 and REQ-CL-7732-CAUSAL-ADMISSION tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.experiment_7732_v673_causal_admission import (
    admission_decision,
    cold_reduce,
    preflight,
    run_experiment,
    run_fixture,
)


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    """Keep one full fixture stream for independent assertions."""
    raw = tmp_path_factory.mktemp("exp7732")
    return run_fixture(raw), raw


def test_7732_eight_case_guard() -> None:
    """REQ-CL-7732-CAUSAL-ADMISSION: both guards use eight labels."""
    with pytest.raises(ValueError, match="eight_independent"):
        admission_decision([{"unit_id": "one", "base": 0.1, "candidate": 0.9}], [1])
    rows = [{"unit_id": str(i), "base": 0.1, "candidate": 0.5} for i in range(8)]
    assert admission_decision(rows, [1] * 8)["accepted"] is True
    assert admission_decision(rows, [0] * 8)["accepted"] is False
    duplicate = [dict(item) for item in rows]
    duplicate[1]["unit_id"] = "0"
    with pytest.raises(ValueError, match="eight_independent"):
        admission_decision(duplicate, [1] * 8)
    mixed = [{"unit_id": str(i), "base": 0.49, "candidate": 0.51} for i in range(8)]
    assert admission_decision(mixed, [0] + [1] * 7)["accepted"] is False
    with pytest.raises(ValueError, match="binary_admission"):
        admission_decision(rows, [True] + [1] * 7)


def test_7732_lifecycle_and_controls(measured: tuple[dict, Path]) -> None:
    """SCENARIO-CL-7732-LIFECYCLE: five outcomes and restart parity."""
    result, _ = measured
    assert len(result["rows"]) == 96 * 3
    assert len({row["unit_id"] for row in result["rows"]}) == 96
    assert result["lifecycle"] == {
        "beneficial_commit": True,
        "harmful_rejection": True,
        "duplicate_feedback": True,
        "pending_overflow": True,
        "interrupted_write_replay": True,
    }
    assert result["restart_exact_parity"] is True
    assert result["exactly_once"] is True
    assert result["pending_high_water"] <= 12
    assert result["proposal_count"] <= 8
    assert result["static_closure_complete"]["equality_check"] is True
    assert len(result["static_closure_complete"]["weights"]) == 36
    assert any(row["kind"] == "commit" for row in result["event_rows"])
    assert any(row["kind"] == "rollback" for row in result["event_rows"])
    assert all(row["exact_status"] == row["advisory_status"] for row in result["rows"])


def test_7732_cold_replay_mutation(measured: tuple[dict, Path], tmp_path: Path) -> None:
    """SCENARIO-REPORT-7732-TERMINAL: changed raw data fails cold replay."""
    result, _ = measured
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(result))
    assert cold_reduce(path)["valid"] is True
    changed = json.loads(path.read_text())
    changed["rows"][0]["probability"] = 0.7
    path.write_text(json.dumps(changed))
    assert cold_reduce(path)["valid"] is False


def test_7732_preflight_and_blocked_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-7732: absent external inputs give exact blocked operands."""
    checks, hashes = preflight(tmp_path)
    assert any(not row["passed"] and row["field"] == "exists" for row in checks)
    assert hashes["absent_sources"]
    artifact = run_experiment(tmp_path, "20260927", tmp_path / "out.json", validate=False)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]
    assert artifact["acquisition_protocol_ready_score"] == 0
    assert artifact["MODEL_SPECS"] == []


def test_7732_valid_artifact_without_validation(
    measured: tuple[dict, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7732-CAUSAL: fixture claims remain circular and scoped."""
    from carnot import experiment_7732_v673_causal_admission as exp

    monkeypatch.setattr(exp, "run_fixture", lambda _raw: measured[0])
    monkeypatch.setattr(exp, "preflight", lambda _root: ([], {"eligible_producers": {}}))
    artifact = run_experiment(tmp_path, "20260927", tmp_path / "out.json", validate=False)
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["acquisition_protocol_ready_score"] == 0
    assert artifact["claim_scope"]["fresh_generalization_eligible"] is False
    assert artifact["acceptance_gate_results"]["readiness"] is None


def test_7732_real_input_custody_and_malformed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7732-TERMINAL: authenticate all three producers."""
    from carnot import experiment_7732_v673_causal_admission as exp

    checks, hashes = preflight(exp.ROOT)
    assert all(row["passed"] for row in checks)
    assert len(hashes["eligible_producers"]) == 2
    assert len(hashes["flagged_historical_inputs"]) == 1
    for relative, _ in exp.INPUTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{")
    bad, _ = preflight(tmp_path)
    assert any(not row["passed"] and row["field"] == "schema" for row in bad)


def test_7732_reader_rejects_unknown_source_and_broken_bank(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7732-TERMINAL: changed family and ledger fail closed."""
    path = tmp_path / "candidate.json"
    raw = tmp_path / "rows.json"
    rows = [{"unit_id": "invented", "arm": "growth"}]
    raw.write_text(json.dumps(rows))
    import hashlib

    path.write_text(
        json.dumps(
            {
                "rows": rows,
                "arms": ["growth", "frozen", "complete_static"],
                "bank_state_paths": {"growth": str(tmp_path / "missing-bank.json")},
                "raw_rows_path": str(raw),
                "raw_rows_sha256": "sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
        )
    )
    (tmp_path / "missing-bank.json").write_text("{")
    result = cold_reduce(path)
    assert result["valid"] is False
    assert result["source_valid"] is False
    assert result["bank_valid"] is False


@pytest.mark.parametrize("failed_group", [None, "terminal"])
def test_7732_validation_gate_is_affected_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_group: str | None
) -> None:
    """REQ-REPORT-7732: global debt stays separate; terminal failure closes gate."""
    from carnot import experiment_7732_v673_causal_admission as exp

    arms = ("growth", "frozen", "complete_static")
    fake = {
        "rows": [{"unit_id": f"u{i}", "arm": arm} for arm in arms for i in range(96)],
        "arms": list(arms),
        "lifecycle": {
            name: True
            for name in (
                "beneficial_commit",
                "harmful_rejection",
                "duplicate_feedback",
                "pending_overflow",
                "interrupted_write_replay",
            )
        },
        "restart_exact_parity": True,
        "exactly_once": True,
        "pending_high_water": 8,
        "proposal_count": 2,
        "static_closure_complete": {"equality_check": True},
    }
    monkeypatch.setattr(exp, "run_fixture", lambda _raw: fake)
    monkeypatch.setattr(exp, "preflight", lambda _root: ([], {"eligible_producers": {}}))
    calls: list[str] = []

    def receipts(_root: Path, commands: list, *, log_dir: Path, **_kwargs: object) -> list[dict]:
        kind = log_dir.name
        calls.append(kind)
        if kind == "affected":
            coverage = next(
                command for command in commands if command.name == "changed_module_coverage"
            )
            focused = next(command for command in commands if command.name == "focused_pytest")
            assert exp.TEST in coverage.argv
            assert all(item not in coverage.argv for item in exp.REUSED_TESTS)
            assert all(item in focused.argv for item in exp.REUSED_TESTS)
        return [
            {
                "name": command.name,
                "passed": kind != "full"
                and not (kind == failed_group and command.name == "adversarial_verify"),
                "exit_code": 1
                if kind == failed_group and command.name == "adversarial_verify"
                else 0,
                "log_path": str(log_dir / f"{command.name}.log"),
            }
            for command in commands
        ]

    monkeypatch.setattr(exp, "run_commands", receipts)
    artifact = run_experiment(tmp_path, "20260927", tmp_path / "out.json")
    assert artifact["validation_receipts"]["repository_health"]["status"] == "degraded_open"
    if failed_group:
        assert artifact["verdict_class"] == "disqualified"
        assert artifact["flagged_adversarial"] is True
        assert artifact["acquisition_protocol_ready_score"] == 0
    else:
        assert artifact["verdict_class"] == "circular_positive"
        assert artifact["acquisition_protocol_ready_score"] == 1
        receipt = artifact["validation_receipts"]["repository_health"]["current_full_suite"][0]
        log = Path(receipt["log_path"])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("18 known collection errors")
        from carnot.reporting.current_work_receipt import sha256_file

        receipt["log_sha256"] = sha256_file(log)
        (tmp_path / "out.json").write_text(json.dumps(artifact))
        replay = run_experiment(tmp_path, "20260927", tmp_path / "out.json")
        assert replay["acquisition_protocol_ready_score"] == 1
        assert calls.count("full") == 1


def test_7732_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7732-TERMINAL: CLI returns cold-reader status."""
    from carnot import experiment_7732_v673_causal_admission as exp

    monkeypatch.setattr(exp, "cold_reduce", lambda _path: {"valid": True})
    assert exp.main(["--cold-reduce", str(tmp_path / "candidate.json")]) == 0
    monkeypatch.setattr(exp, "cold_reduce", lambda _path: {"valid": False})
    assert exp.main(["--cold-reduce", str(tmp_path / "candidate.json")]) == 1
    calls: list[tuple] = []
    monkeypatch.setattr(exp, "run_experiment", lambda *args: calls.append(args))
    assert exp.main(["--date", "20260927", "--output", "out.json"]) == 0
    assert calls
