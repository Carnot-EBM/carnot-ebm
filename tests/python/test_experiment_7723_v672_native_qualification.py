"""REQ-REPORT-7723 and REQ-PYBIND-7723 qualification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_7723_v672_native_qualification as experiment
from carnot.pipeline.native_calibrated_decision_service import load_native_extension
from test_experiment_7710_v671_native_record_contract import extension_path


def test_scenario_report_7723_blocked_exact_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7723-BLOCKED names each absent producer operand."""

    checks, hashes = experiment.check_preconditions(tmp_path)
    failed = [row for row in checks if not row["passed"]]
    assert failed
    assert all(
        {"check", "upstream_id", "artifact_path", "field", "operator", "expected", "observed"}
        <= row.keys()
        for row in failed
    )
    assert hashes["missing_custody"]
    result = experiment.build_artifact(
        checks=checks,
        hashes=hashes,
        rows=[],
        parity=[],
        durability={},
        extension=None,
        affected_receipts=[],
        global_receipts=[],
        spans=[],
        frozen_scope="results/raw/experiment_7723_v672_native_qualification/frozen_scope.json",
    )
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert result["verdict_class"] == "blocked"
    assert result["native_service_ready_score"] == 0
    assert result["MODEL_SPECS"] == result["model_specs"] == []


def test_scenario_report_7723_current_inputs_are_authenticated() -> None:
    """REQ-REPORT-7723 checks the repaired checkout and immutable V671 bytes."""

    checks, hashes = experiment.check_preconditions(experiment.ROOT)
    assert all(row["passed"] for row in checks), checks
    assert hashes["valid_producers"]
    assert hashes["flagged_historical_evidence"]
    assert hashes["pre_gate_receipts"]


def test_scenario_pybind_7723_restart_and_parity(extension_path: Path, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7723-PARITY and SCENARIO-PYBIND-7723-RESTART."""

    binding = load_native_extension(extension_path)
    rows, parity, durability = experiment.measure_qualification(
        binding, tmp_path / "state.json", tmp_path, started=experiment.monotonic()
    )
    assert len([row for row in parity if row["unit_id"].startswith("unit-")]) == 128
    assert len(rows) == 256
    assert all(row["passed"] for row in parity)
    assert durability["normal_restart"]["passed"]
    assert durability["crash_before_ack"]["passed"]
    assert durability["crash_after_ack"]["passed"]
    assert durability["fsync_policy"] == "atomic_file_fsync_rename_directory_fsync_reload_ack"
    assert all("restart_outcome" in row for row in parity)


def test_scenario_report_7723_cold_reader_rejects_tampered_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7723-TERMINAL recomputes parity from raw operands."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "verdict_class": "null",
                "native_service_ready_score": 1,
                "parity_rows": [
                    {
                        "unit_id": "unit-0",
                        "python_probability": 0.1,
                        "rust_probability": 0.2,
                        "python_action": "accept",
                        "rust_action": "accept",
                        "python_error": None,
                        "rust_error": None,
                        "passed": True,
                    }
                ],
                "durable_replay": {"passed": True},
            }
        )
    )
    with pytest.raises(AssertionError):
        experiment.cold_reduce(candidate)


def test_scenario_report_7723_blocked_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7723-BLOCKED publishes a terminal exact-input receipt."""

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "frozen_scope.json").write_text("{}")
    monkeypatch.setattr(experiment, "RAW", raw)

    def fake_commands(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        return [
            {"name": spec.name, "passed": True, "exit_code": 0, "log_sha256": "sha256:test"}
            for spec in commands
        ]

    monkeypatch.setattr(experiment.validation, "run_commands", fake_commands)
    output = tmp_path / "blocked.json"
    result = experiment.run(tmp_path, output)
    assert result["verdict_class"] == "blocked"
    assert result["native_service_ready_score"] == 0
    assert json.loads(output.read_text()) == result
    assert (raw / "terminal_exact_receipts.json").is_file()
    experiment.cold_reduce(output)


def test_scenario_report_7723_producer_claim_and_terminal_failure(
    extension_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7723-TERMINAL gates readiness on exact reader exits."""

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "frozen_scope.json").write_text("{}")
    monkeypatch.setattr(experiment, "RAW", raw)
    real_commands = experiment.validation.run_commands

    def fake_build(_root: Path, _raw: Path, _private: Path, _started: float) -> tuple:
        return (
            extension_path,
            {"binary_hash": "sha256:test", "module_path": str(extension_path)},
            [{"name": "native_extension_build", "passed": True, "exit_code": 0}],
        )

    def fake_commands(root: Path, commands: list, **kwargs: object) -> list[dict]:
        if commands[0].name == "hard_exit_after_ack":
            return real_commands(root, commands, **kwargs)
        return [
            {"name": spec.name, "passed": True, "exit_code": 0, "log_sha256": "sha256:test"}
            for spec in commands
        ]

    monkeypatch.setattr(experiment.prior, "build_extension", fake_build)
    monkeypatch.setattr(experiment.validation, "run_commands", fake_commands)
    ready = experiment.run(experiment.ROOT, tmp_path / "ready.json")
    assert ready["native_service_ready_score"] == 1
    assert ready["verdict_class"] == "circular_positive"
    assert ready["inference_substrate"] == "deterministic_verifier_plus_replay"
    experiment.cold_reduce(tmp_path / "ready.json")

    def failed_reader(root: Path, commands: list, **kwargs: object) -> list[dict]:
        receipts = fake_commands(root, commands, **kwargs)
        if commands[0].name == "cold_reduction":
            receipts[1]["passed"] = False
            receipts[1]["exit_code"] = 1
        return receipts

    monkeypatch.setattr(experiment.validation, "run_commands", failed_reader)
    failed = experiment.run(experiment.ROOT, tmp_path / "failed.json")
    assert failed["verdict_class"] == "disqualified"
    assert failed["flagged_adversarial"] is True
    assert failed["native_service_ready_score"] == 0


def test_scenario_report_7723_main_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7723 rejects wrong dates and dispatches cold replay."""

    with pytest.raises(ValueError, match="run_date_invalid"):
        experiment.main(["--date", "20260925"])
    called = []
    monkeypatch.setattr(experiment, "cold_reduce", lambda path: called.append(path))
    experiment.main(["--cold-replay", str(tmp_path / "candidate.json")])
    assert called == [tmp_path / "candidate.json"]
    monkeypatch.setattr(experiment, "run", lambda root, output: called.append(output))
    experiment.main(["--output", str(tmp_path / "new.json")])
    assert called[-1] == tmp_path / "new.json"


def test_scenario_report_7723_frozen_scope_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7723 will not measure without its frozen scope."""

    monkeypatch.setattr(experiment, "RAW", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="frozen_scope_missing"):
        experiment.run(experiment.ROOT, tmp_path / "unused.json")


def test_scenario_report_7723_owned_build_failure_disqualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7723-TERMINAL keeps owned build failure distinct from blocked input."""

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "frozen_scope.json").write_text("{}")
    monkeypatch.setattr(experiment, "RAW", raw)

    def fail_build(*_args: object) -> None:
        raise RuntimeError("native_extension_build_failed")

    def pass_readers(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        return [
            {"name": spec.name, "passed": True, "exit_code": 0, "log_sha256": "sha256:test"}
            for spec in commands
        ]

    monkeypatch.setattr(experiment.prior, "build_extension", fail_build)
    monkeypatch.setattr(experiment.validation, "run_commands", pass_readers)
    output = tmp_path / "failed.json"
    result = experiment.run(experiment.ROOT, output)
    assert result["verdict_class"] == "disqualified"
    assert result["native_service_ready_score"] == 0
    assert any(row["check"] == "owned_execution" for row in result["gate_check_summary"])
    experiment.cold_reduce(output)
