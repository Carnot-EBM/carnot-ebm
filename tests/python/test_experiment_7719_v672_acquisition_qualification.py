"""REQ-REPORT-7719 and REQ-CL-7719-CAUSAL-ADMISSION qualification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.constraint_bank_protocol import Bank, grammar
from carnot.reporting.acquisition_qualification import (
    admission_decision,
    fit_static_closure,
    fixture_groups,
    run_fixture,
)
from carnot.experiment_7719_v672_acquisition_qualification import (
    cold_reduce,
    preflight,
    run_experiment,
)


@pytest.fixture(scope="module")
def measured_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    """Persist the expensive bank matrix once for independent reader tests."""
    raw = tmp_path_factory.mktemp("exp7719-bank")
    return run_fixture(raw), raw


def test_7719_fixture_is_independent_and_label_free() -> None:
    """SCENARIO-CL-7719-REACHABLE: three disjoint 32-family roles."""
    groups = fixture_groups()
    assert {key: len(value) for key, value in groups.items()} == {
        "development": 32,
        "admission": 32,
        "retention": 32,
    }
    assert len({case["unit_id"] for cases in groups.values() for case in cases}) == 96
    assert all(len(case["source_bytes_hex"]) > 0 for cases in groups.values() for case in cases)
    assert len(grammar()["primitives"]) == 8
    assert len(grammar()["pairs"]) == 28


def test_7719_batch_requires_frozen_counterfactuals(tmp_path: Path) -> None:
    """REQ-CL-7719-CAUSAL-ADMISSION: a five-case batch cannot borrow labels."""
    bank = Bank(tmp_path / "bank.json", grammar(), "priority", 0.1, 2)
    groups = fixture_groups()
    case = groups["development"][0]
    bank.predict(case["unit_id"], 0, case["features"], 0.1, "unknown", "update", case["unit_id"])
    bank.release(case["unit_id"], 1, 8)
    proposal = bank.propose()
    assert proposal is not None
    with pytest.raises(ValueError, match="five_frozen_forecasts_required"):
        admission_decision(bank, [])
    batch = []
    for index, item in enumerate(groups["admission"][:5]):
        result = bank.predict(
            item["unit_id"],
            index + 9,
            item["features"],
            item["base_probability"],
            "unknown",
            "admission",
            item["unit_id"],
        )
        batch.append(
            {
                "unit_id": item["unit_id"],
                "base_probability": result["probability"],
                "candidate_probability": min(1 - 1e-6, result["probability"] + proposal["weight"]),
                "freeze_tick": proposal["freeze_tick"],
            }
        )
    with pytest.raises(ValueError, match="admission_feedback_missing"):
        admission_decision(bank, batch)
    for item in groups["admission"][:5]:
        bank.release(item["unit_id"], item["label"], 22)
    assert admission_decision(bank, batch) is True
    bank.state["proposal"] = None
    with pytest.raises(ValueError, match="proposal_missing"):
        admission_decision(bank, batch)
    bank.state["proposal"] = proposal
    original_tick = proposal["freeze_tick"]
    bank.state["proposal"]["freeze_tick"] = 100
    with pytest.raises(ValueError, match="admission_not_postfreeze"):
        admission_decision(bank, batch)
    bank.state["proposal"]["freeze_tick"] = original_tick
    changed = [dict(item) for item in batch]
    changed[0]["base_probability"] = 0.7
    with pytest.raises(ValueError, match="frozen_forecast_mismatch"):
        admission_decision(bank, changed)


def test_7719_static_closure_is_fitted(tmp_path: Path) -> None:
    """SCENARIO-CL-7719-STATIC: all 36 features differ from empty control."""
    receipt = fit_static_closure(fixture_groups()["development"])
    assert len(receipt["coefficients"]) == 36
    assert receipt["fit_family_count"] == 32
    assert receipt["nonzero_coefficient_count"] > 0
    assert receipt["different_from_empty_bank"] is True


def test_7719_positive_negative_and_replay(measured_fixture: tuple[dict, Path]) -> None:
    """SCENARIO-REPORT-7719-CAUSAL: commit fires; rollback has no effect."""
    measured, _ = measured_fixture
    commits = measured["commit_reachability_rows"]
    assert any(row["decision"] == "commit" and row["later_forecast_delta"] > 0 for row in commits)
    assert any(
        row["decision"] == "rollback" and row["later_forecast_delta"] == 0 for row in commits
    )
    assert measured["restart_exact_parity"] is True
    assert measured["budget_valid"] is True
    assert measured["static_closure_receipt"]["different_from_empty_bank"] is True
    assert len(measured["rows"]) == 96 * len(measured["arms"])
    assert measured["v671_rollback_reproduced"] is True
    assert all(measured["feedback_diagnostics"].values())


def test_7719_missing_and_mutated_replay_fail(
    tmp_path: Path, measured_fixture: tuple[dict, Path]
) -> None:
    """SCENARIO-REPORT-7719-TERMINAL: absence and changed raw bytes fail."""
    checks, hashes = preflight(tmp_path)
    assert any(not row["passed"] and row["field"] == "exists" for row in checks)
    assert hashes["missing_custody"]
    measured, _ = measured_fixture
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(measured))
    assert cold_reduce(raw)["valid"] is True
    changed = json.loads(raw.read_text())
    changed["rows"].pop()
    raw.write_text(json.dumps(changed))
    assert cold_reduce(raw)["valid"] is False
    changed = json.loads(json.dumps(measured))
    changed["rows"][0]["raw_metrics"]["probability"] = 0.7
    changed["rows"][0]["raw_metrics"]["brier"] = (
        0.7 - changed["rows"][0]["raw_metrics"]["label"]
    ) ** 2
    raw.write_text(json.dumps(changed))
    assert cold_reduce(raw)["valid"] is False
    changed = json.loads(json.dumps(measured))
    changed["rows"][0]["provenance"]["source_bytes_hex"] = "00"
    raw.write_text(json.dumps(changed))
    assert cold_reduce(raw)["valid"] is False


def test_7719_blocked_artifact_has_exact_operands(tmp_path: Path) -> None:
    """REQ-REPORT-7719: missing upstream is terminal blocked with zero calls."""
    result = run_experiment(tmp_path, "20260926", tmp_path / "out.json", validate=False)
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert result["gate_check_summary"]
    assert result["acquisition_protocol_ready_score"] == 0
    assert result["model_invoked"] is False
    assert result["MODEL_SPECS"] == []


def test_7719_invalid_historical_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7719-TERMINAL: malformed historical JSON is blocked."""
    from carnot import experiment_7719_v672_acquisition_qualification as exp

    path = tmp_path / exp.V671
    path.parent.mkdir(parents=True)
    path.write_text("{")
    checks, _ = preflight(tmp_path)
    assert any(item["field"] == "schema" and not item["passed"] for item in checks)


def test_7719_progress_and_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CL-7719-REACHABLE: checkpoint callbacks and stale scratch replacement."""
    from carnot.reporting import acquisition_qualification as aq

    monkeypatch.setattr(aq, "ARMS", ("priority",))
    tmp_path.joinpath("bank_priority.json").write_text("{}")
    tmp_path.joinpath("bank_diagnostics.json").write_text("{}")
    events = []
    result = aq.run_fixture(tmp_path, lambda phase, event, units: events.append(event))
    assert result["budget_valid"] is True
    assert any(item.startswith("arm_start") for item in events)
    assert any(item.startswith("checkpoint") for item in events)
    assert any(item.startswith("arm_complete") for item in events)


def test_7719_qualified_publication_and_failed_reader(
    tmp_path: Path, measured_fixture: tuple[dict, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7719-TERMINAL: actual checks control readiness."""
    from carnot import experiment_7719_v672_acquisition_qualification as exp

    measured, _ = measured_fixture
    passed_check = exp.check("fixture", "current", "fixture", "exists", True, True)
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    monkeypatch.setattr(exp, "preflight", lambda _root: ([passed_check], hashes))
    monkeypatch.setattr(exp, "run_fixture", lambda _raw, _progress: measured)
    monkeypatch.setattr(
        exp,
        "build_scoped_commands",
        lambda *_args, **_kwargs: [exp.CommandSpec("focused_pytest", ("true",), "test")],
    )
    fail_reader = False
    fail_full = False

    def fake_commands(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        return [
            {
                "name": item.name,
                "command": "true",
                "exit_code": 2
                if (fail_reader and item.name == "adversarial_verify")
                or (fail_full and item.name == "full_python_suite")
                else 0,
                "log_path": str(tmp_path / item.name),
                "log_sha256": "sha256:fixture",
                "passed": not (
                    (fail_reader and item.name == "adversarial_verify")
                    or (fail_full and item.name == "full_python_suite")
                ),
            }
            for item in commands
        ]

    monkeypatch.setattr(exp, "run_commands", fake_commands)
    output = tmp_path / "result.json"
    result = exp.run_experiment(tmp_path, "20260926", output)
    assert result["verdict_class"] == "circular_positive"
    assert result["acquisition_protocol_ready_score"] == 1
    assert len(result["validation_receipts"]["terminal_readers"]) == 3
    assert output.is_file()
    fail_reader = True
    result = exp.run_experiment(tmp_path, "20260926", output)
    assert result["verdict_class"] == "disqualified"
    assert result["flagged_adversarial"] is True
    assert result["acquisition_protocol_ready_score"] == 0
    fail_reader = False
    fail_full = True
    result = exp.run_experiment(tmp_path, "20260926", output)
    assert result["validation_receipts"]["global_debt"]
    assert result["verdict_class"] == "disqualified"


def test_7719_entrypoint_and_current_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7719: only the declared current artifact can qualify."""
    from carnot import experiment_7719_v672_acquisition_qualification as exp

    checks, hashes = exp.preflight(exp.ROOT)
    assert all(item["passed"] for item in checks)
    assert str(exp.V671) in hashes["valid_producers"]
    calls = []
    monkeypatch.setattr(exp, "run_experiment", lambda root, date, output: calls.append(date))
    assert exp.main(["--date", "20260926", "--output", str(tmp_path / "out.json")]) == 0
    assert calls == ["20260926"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rows": [], "arms": [], "bank_state_paths": {}}))
    assert exp.main(["--cold-reduce", str(bad)]) == 1
    assert '"valid": false' in capsys.readouterr().out
