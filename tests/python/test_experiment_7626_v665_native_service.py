"""Tests for REQ-REPORT-7626 and SCENARIO-REPORT-7626-*.

These tests distinguish a compiled Rust call from a Python substitute. They
also keep readiness separate from any speed or calibration benefit claim.
"""

from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest

from carnot import experiment_7626_v665_native_service as exp


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def binding(tmp_path_factory: pytest.TempPathFactory) -> object:
    """Load a real task build even when pytest is not launched by the producer."""

    value = os.environ.get("CARNOT_EXP7626_EXTENSION")
    if value:
        extension = Path(value)
    else:
        target = Path("/tmp/carnot-exp7626-pytest-target")
        lock_path = Path("/tmp/carnot-exp7626-pytest-build.lock")
        suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
        extension = tmp_path_factory.mktemp("exp7626-native") / f"_rust{suffix}"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for command in (
                (
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "carnot-python",
                    "--target-dir",
                    str(target),
                ),
                (
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "carnot-core",
                    "--bin",
                    "portable-recalibration-service",
                ),
            ):
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "PYO3_PYTHON": sys.executable},
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                assert completed.returncode == 0, completed.stdout + completed.stderr
            shutil.copy2(target / "release/libcarnot_python.so", extension)
    assert extension.is_file()
    module = exp.load_native_extension(extension)
    assert Path(module.__file__).resolve() == extension.resolve()
    return module


def test_req_report_7626_freezes_exp7598_workload() -> None:
    """REQ-REPORT-7626 authenticates one fixed historical workload."""

    workload = exp.frozen_workload()
    assert [row["seed"] for row in workload] == [7_598_101, 7_598_102]
    assert all(len(row["events"]) == 4 for row in workload)
    assert all(row["historical_source"] == exp.EXP7598_PATH.as_posix() for row in workload)
    assert exp.workload_checksum(workload) == exp.FROZEN_WORKLOAD_SHA256


def test_scenario_report_7626_parity_reducer_requires_all_callers() -> None:
    """SCENARIO-REPORT-7626-PARITY requires complete exact typed parity."""

    rows = exp.synthetic_parity_rows()
    reduced = exp.reduce_parity_rows(rows)
    assert reduced == {
        "complete": True,
        "independent_units": 2,
        "request_count": 30,
        "typed_mismatch_count": 0,
        "numeric_mismatch_count": 0,
        "max_abs_delta": 0.0,
        "durable_reload_match": True,
    }

    rows.pop()
    with pytest.raises(ValueError, match="parity_row_count"):
        exp.reduce_parity_rows(rows)

    wrong = exp.synthetic_parity_rows()
    wrong[0]["action"] = "reject"
    with pytest.raises(ValueError, match="typed_decision_mismatch"):
        exp.reduce_parity_rows(wrong)


def test_scenario_report_7626_durability_reducer_fails_closed() -> None:
    """SCENARIO-REPORT-7626-INTERRUPT forbids false durable acknowledgments."""

    rows = exp.synthetic_durability_rows()
    reduced = exp.reduce_durability_rows(rows)
    assert reduced["complete"] is True
    assert reduced["interrupted_write_acknowledged"] is False
    assert reduced["prior_state_survived"] is True

    wrong = deepcopy(rows)
    wrong[-1]["acknowledged"] = True
    with pytest.raises(ValueError, match="interrupted_write_false_ack"):
        exp.reduce_durability_rows(wrong)


def test_req_report_7626_terminal_validator_rejects_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-7626 keeps readiness, benefit, and native identity separate."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    assert exp.validate_artifact(artifact, check_files=False) == []

    wrong = deepcopy(artifact)
    wrong["native_service_ready_score"] = 0
    wrong["reproducibility_checksum"] = exp.reproducibility_checksum(wrong)
    assert "native_service_ready_score" in exp.validate_artifact(wrong, check_files=False)

    wrong = deepcopy(artifact)
    wrong["acceptance_gate_results"][2]["passed"] = True
    wrong["reproducibility_checksum"] = exp.reproducibility_checksum(wrong)
    assert "benefit_gate" in exp.validate_artifact(wrong, check_files=False)


def test_scenario_report_7626_actual_binding_invalid_and_cold_reload(
    binding: object, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7626-INVALID/COLD cross the compiled durable core."""

    state = tmp_path / "native.json"
    client = exp.NativeServiceClient(binding, state)
    decision = client.predict("native-one", 0.31)
    ack = client.release_feedback("native-one", 1)
    assert decision.available and ack.durable
    before = state.read_bytes()
    invalid = client.release_feedback("native-one", 2)
    assert invalid.durable is False and invalid.error == "binary_label_required"
    assert state.read_bytes() == before

    restarted = exp.NativeServiceClient(binding, state)
    resumed = restarted.predict("native-two", 0.37)
    assert resumed.available
    assert restarted.state_summary()["sample_count"] == 1


def test_scenario_report_7626_actual_three_caller_parity(binding: object, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7626-PARITY crosses Python, JSONL, and PyO3 callers."""

    rows, durability = exp.exercise_three_callers(ROOT, binding, tmp_path)
    parity = exp.reduce_parity_rows(rows)
    durable = exp.reduce_durability_rows(durability)
    assert parity["complete"] is True
    assert parity["max_abs_delta"] <= exp.PARITY_TOLERANCE
    assert durable["complete"] is True


def test_scenario_report_7626_process_interruption_preserves_checkpoint(
    binding: object, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7626-INTERRUPT checks E2E-004 in an owned child."""

    extension = Path(binding.__file__)
    state = tmp_path / "interrupted.json"
    client = exp.NativeServiceClient(binding, state)
    assert client.predict("kept", 0.21).available
    assert client.release_feedback("kept", 1).durable
    before = exp.sha256_file(state)

    row = exp.run_interrupted_write_probe(ROOT, extension, state)
    assert row["owned_process"] is True
    assert row["exit_code"] == exp.INTERRUPTED_WRITE_EXIT
    assert row["acknowledged"] is False
    assert row["prior_state_survived"] is True
    assert exp.sha256_file(state) == before


def test_scenario_report_7626_fresh_process_cold_reduction(binding: object, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7626-COLD requires a new interpreter and exact module."""

    artifact_path = tmp_path / "artifact.json"
    artifact = exp.build_test_artifact(ROOT, tmp_path)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'python'}:{ROOT}"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_7626_v665_native_service",
            "--cold-replay",
            str(artifact_path),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["valid"] is True


def test_req_report_7626_reducers_reject_each_incomplete_contract() -> None:
    """REQ-REPORT-7626 rejects arm, number, ack, crash, and reload drift."""

    wrong_arm = exp.synthetic_parity_rows()
    wrong_arm[0]["arm"] = "unknown"
    with pytest.raises(ValueError, match="parity_arms"):
        exp.reduce_parity_rows(wrong_arm)

    wrong_number = exp.synthetic_parity_rows()
    wrong_number[0]["output_probability"] += 1e-4
    with pytest.raises(ValueError, match="numeric_parity_mismatch"):
        exp.reduce_parity_rows(wrong_number)

    wrong_ack = exp.synthetic_parity_rows()
    wrong_ack[0]["acknowledged"] = False
    with pytest.raises(ValueError, match="durable_update_missing"):
        exp.reduce_parity_rows(wrong_ack)

    with pytest.raises(ValueError, match="interrupted_write_row_count"):
        exp.reduce_durability_rows(exp.synthetic_durability_rows()[:-1])

    lost = exp.synthetic_durability_rows()
    lost[-1]["prior_state_survived"] = False
    with pytest.raises(ValueError, match="interrupted_write_lost_prior_state"):
        exp.reduce_durability_rows(lost)

    missing_reload = exp.synthetic_durability_rows()
    missing_reload.pop(1)
    with pytest.raises(ValueError, match="durability_events_missing"):
        exp.reduce_durability_rows(missing_reload)


def test_req_report_7626_client_rejects_native_contract_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7626-INVALID converts native exceptions to escalation."""

    class BadService:
        def __init__(self, _path: str) -> None:
            pass

        def predict(self, event_id: str, _probability: float) -> tuple[str, float, str]:
            if event_id == "error":
                raise ValueError("probability_not_finite")
            return "wrong-id", 0.2, "reject"

        def release_feedback(self, event_id: str, _label: int) -> tuple[str, bool, bool, None]:
            return event_id, False, False, None

        def state_summary(self) -> tuple[int, list[str], str]:
            return 0, [], "schema"

    class BadBinding:
        RustPortableRecalibrationService = BadService

    client = exp.NativeServiceClient(BadBinding, tmp_path / "unused.json")
    assert client.predict("drift", 0.2).error == "prediction_contract_invalid"
    assert client.predict("error", float("nan")).available is False
    assert client.release_feedback("event", 1).error == "durable_acknowledgment_invalid"
    assert client.state_summary()["sample_count"] == 0


def test_req_report_7626_preconditions_and_command_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7626 authenticates tools and freezes narrow commands."""

    context = exp.collect_preconditions(ROOT)
    assert context["blocker"] is None
    assert all(row["passed"] for row in context["rows"])
    names = {command.name for command in exp.build_validation_commands(ROOT, tmp_path)}
    assert {"focused_pytest", "cargo_test_portable_recalibration", "cargo_clippy_binding"} <= names
    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert [row.name for row in terminal] == [
        "fresh_process_cold_replay",
        "independent_raw_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]

    def fail_run(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("tool", 1)

    monkeypatch.setattr(exp.subprocess, "run", fail_run)
    assert exp._tool_version(["missing"]) is None


def test_req_report_7626_validator_and_replays_cover_invalid_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7626-TERMINAL rejects each governed field mutation."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    assert exp.validate_artifact(artifact, check_files=True) == []
    mutations = {
        "artifact_identity": ("schema", "wrong"),
        "honest_verdict": ("honest_verdict", "unfinished"),
        "verdict_class": ("verdict_class", "unknown"),
        "model_invocation": ("model_invoked", True),
        "invocation_counts": ("invocation_counts", {}),
        "parity_reduction": ("parity_reduction", {}),
        "durability_reduction": ("durability_reduction", {}),
        "production_default": ("production_default_changed", True),
    }
    for expected, (field, replacement) in mutations.items():
        wrong = deepcopy(artifact)
        wrong[field] = replacement
        wrong["reproducibility_checksum"] = exp.reproducibility_checksum(wrong)
        assert expected in exp.validate_artifact(wrong, check_files=False)

    wrong = deepcopy(artifact)
    wrong["parity_rows"] = []
    wrong["reproducibility_checksum"] = exp.reproducibility_checksum(wrong)
    assert any(row.startswith("independent_reduction") for row in exp.validate_artifact(wrong))

    wrong = deepcopy(artifact)
    wrong["native_build_manifest_path"] = str(tmp_path / "missing.json")
    wrong["reproducibility_checksum"] = exp.reproducibility_checksum(wrong)
    assert "native_build_manifest_sha256" in exp.validate_artifact(wrong)

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    assert exp.cold_replay(list_path)["valid"] is False
    assert exp.independent_replay(list_path)["valid"] is False

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--cold-replay", str(artifact_path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert exp.main(["--independent-replay", str(artifact_path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE


def test_req_report_7626_checksum_span_and_producer_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7626 covers checksum failure and the thin producer dispatch."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    artifact["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum" in exp.validate_artifact(artifact, check_files=False)
    span = exp._span("unit", 10.0, 9.0, 2)
    assert span["completed_units"] == 2 and span["checkpoint_position"] == 2

    called: list[tuple[Path, str, Path | None]] = []

    def fake_run(
        root: Path, run_date: str, *, output_path: Path | None = None
    ) -> dict[str, object]:
        called.append((root, run_date, output_path))
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    assert exp.main(["--root", str(ROOT), "--date", exp.RUN_DATE]) == 0
    assert called == [(ROOT, exp.RUN_DATE, None)]
