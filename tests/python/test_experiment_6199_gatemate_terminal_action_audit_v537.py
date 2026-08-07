"""Tests for Exp6199 GateMate terminal-action audit.

Spec refs: REQ-HW-6199, SCENARIO-HW-6199-1, SCENARIO-HW-6199-2,
SCENARIO-HW-6199-3, SCENARIO-HW-6199-4, SCENARIO-HW-6199-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6199_gatemate_terminal_action_audit_v537 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "hardware" / "spec.md"


class RecordingRunner:
    """REQ-HW-6199 fake runner; tests assert exactly which commands would run."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandReceipt]] | None = None):
        self.probes = {key: list(value) for key, value in (probes or {}).items()}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandReceipt:
        assert timeout_s > 0
        self.calls.append(tuple(command))
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class StepClock:
    """Deterministic clock for stable duration assertions."""

    def __init__(self, *values: float):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _probe(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    duration_s: float = 0.015,
) -> mod.CommandReceipt:
    return mod.CommandReceipt(
        command=mod.DETECT_COMMAND,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
    )


def _changed_receipt() -> dict:
    receipt = deepcopy(mod.DEFAULT_OPERATOR_RECEIPT)
    receipt.update(
        {
            "exists": True,
            "receipt_date": "20260807",
            "source": "unit-test operator receipt after USB-C and port change",
            "cable": "new short USB-C cable attached directly to host",
            "port": "moved from cached 3-2.3 hub path to host-root-port-1",
            "power": "GateMate power LED observed after cable change",
            "changes": [
                {
                    "field": "cable",
                    "before": mod.CANONICAL_PRIOR_PHYSICAL_STATE["cable"],
                    "after": "new short USB-C cable attached directly to host",
                },
                {
                    "field": "port",
                    "before": mod.CANONICAL_PRIOR_PHYSICAL_STATE["port"],
                    "after": "moved from cached 3-2.3 hub path to host-root-port-1",
                },
            ],
        }
    )
    return receipt


def _unchanged_receipt() -> dict:
    receipt = deepcopy(mod.DEFAULT_OPERATOR_RECEIPT)
    receipt.update(
        {
            "exists": True,
            "receipt_date": "20260807",
            "source": "unit-test operator receipt confirms no physical change",
            "changes": [],
        }
    )
    return receipt


def _hit_stdout() -> str:
    return (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )


def test_req_hw_6199_spec_defines_terminal_action_audit() -> None:
    """REQ-HW-6199: OpenSpec anchors the Exp6199 audit contract."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("### REQ-HW-6199", maxsplit=1)[1]

    for marker in (
        "REQ-HW-6199",
        "SCENARIO-HW-6199-1",
        "SCENARIO-HW-6199-2",
        "SCENARIO-HW-6199-3",
        "SCENARIO-HW-6199-4",
        "SCENARIO-HW-6199-5",
        mod.OUTPUT_REL_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_hw_6199_unchanged_state_runs_zero_commands_and_exact_packet() -> None:
    """SCENARIO-HW-6199-1: unchanged physical state executes no hardware command."""

    runner = RecordingRunner()
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(10.0, 10.5),
        run_date="20260807",
        current_dated_operator_receipt=_unchanged_receipt(),
        protected_before_hashes=mod.protected_file_hashes(REPO_ROOT),
    )

    assert runner.calls == []
    assert artifact["status"] == "blocked_no_change"
    assert artifact["physical_state_changed"] is False
    assert artifact["hardware_command_authorized"] is False
    assert artifact["detect_attempt_count_command_stdout_stderr_exit_code"]["attempt_count"] == 0
    assert artifact["detect_attempt_count_command_stdout_stderr_exit_code"]["stdout"] == ""
    assert artifact["detect_attempt_count_command_stdout_stderr_exit_code"]["stderr"] == ""
    assert artifact["mutation_command_counts"] == mod.ZERO_MUTATION_COMMAND_COUNTS
    assert artifact["operator_action_packet"] == mod.EXACT_OPERATOR_ACTION_PACKET
    assert artifact["historical_flagged_terminal_evidence_excluded"]["excluded"] is True
    assert artifact["hardware_execution_authenticated"]["authenticated"] is False
    assert artifact["speed_power_energy_terminal_tsu_kona_claim_counts"] == mod.ZERO_CLAIM_COUNTS
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["honest_verdict"].startswith("blocked_no_change:")
    mod.validate_artifact(artifact)


def test_scenario_hw_6199_missing_and_stale_receipts_do_not_authorize_detect() -> None:
    """SCENARIO-HW-6199-2: missing or stale physical receipts run zero commands."""

    missing_runner = RecordingRunner()
    missing = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=missing_runner,
        clock=StepClock(20.0, 20.25),
        run_date="20260807",
    )

    stale_receipt = _changed_receipt()
    stale_receipt["receipt_date"] = mod.BASELINE_RECEIPT_DATE
    stale_runner = RecordingRunner()
    stale = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=stale_runner,
        clock=StepClock(21.0, 21.25),
        run_date="20260807",
        current_dated_operator_receipt=stale_receipt,
    )

    assert missing_runner.calls == []
    assert stale_runner.calls == []
    assert missing["status"] == "blocked_missing_receipt"
    assert stale["status"] == "blocked_stale_receipt"
    assert missing["hardware_command_authorized"] is False
    assert stale["hardware_command_authorized"] is False
    assert missing["detect_attempt_count_command_stdout_stderr_exit_code"]["attempt_count"] == 0
    assert stale["detect_attempt_count_command_stdout_stderr_exit_code"]["attempt_count"] == 0
    assert missing["honest_verdict"].startswith("blocked_missing_receipt:")
    assert stale["honest_verdict"].startswith("blocked_stale_receipt:")
    mod.validate_artifact(missing)
    mod.validate_artifact(stale)


def test_scenario_hw_6199_changed_receipt_wrong_idcode_blocks_execution() -> None:
    """SCENARIO-HW-6199-3: one changed-state detect with wrong IDCODE stays blocked."""

    runner = RecordingRunner(
        {mod.DETECT_COMMAND: [_probe(stdout="index 0:\n\tidcode 0x12345678\n")]}
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(30.0, 31.0),
        run_date="20260807",
        current_dated_operator_receipt=_changed_receipt(),
    )

    assert runner.calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "blocked_idcode"
    assert artifact["physical_state_changed"] is True
    assert artifact["hardware_command_authorized"] is True
    assert artifact["detect_attempt_count_command_stdout_stderr_exit_code"]["attempt_count"] == 1
    assert artifact["detect_attempt_count_command_stdout_stderr_exit_code"]["command"] == (
        "openFPGALoader -c dirtyJtag --detect"
    )
    assert artifact["expected_and_observed_idcode"]["observed_idcode"] == "0x12345678"
    assert artifact["expected_and_observed_idcode"]["matches"] is False
    assert artifact["hardware_execution_authenticated"]["authenticated"] is False
    assert artifact["mutation_command_counts"] == mod.ZERO_MUTATION_COMMAND_COUNTS
    assert artifact["honest_verdict"].startswith("blocked_idcode:")
    mod.validate_artifact(artifact)


def test_scenario_hw_6199_changed_receipt_matching_idcode_records_visibility_only() -> None:
    """SCENARIO-HW-6199-4: matching IDCODE proves visibility, not execution."""

    runner = RecordingRunner({mod.DETECT_COMMAND: [_probe(stdout=_hit_stdout())]})
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(40.0, 41.0),
        run_date="20260807",
        current_dated_operator_receipt=_changed_receipt(),
    )

    assert runner.calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "complete_visible"
    assert artifact["expected_and_observed_idcode"]["matches"] is True
    assert artifact["hardware_execution_authenticated"]["idcode_visibility_authenticated"] is True
    assert artifact["hardware_execution_authenticated"]["authenticated"] is False
    assert artifact["speed_power_energy_terminal_tsu_kona_claim_counts"] == mod.ZERO_CLAIM_COUNTS
    assert "passively cooled" in artifact["passive_cooling_note"]
    assert artifact["honest_verdict"].startswith("complete_visible:")
    mod.validate_artifact(artifact)


def test_scenario_hw_6199_schema_rejects_command_budget_and_claim_violations() -> None:
    """SCENARIO-HW-6199-5: validation fails closed on command budget violations."""

    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(50.0, 50.5),
        run_date="20260807",
    )

    too_many = deepcopy(artifact)
    too_many["physical_state_changed"] = True
    too_many["hardware_command_authorized"] = True
    too_many["detect_attempt_count_command_stdout_stderr_exit_code"]["attempt_count"] = 2
    assert any("at most one detect" in err for err in mod.artifact_schema_errors(too_many))

    wrong_command = deepcopy(artifact)
    wrong_command["physical_state_changed"] = True
    wrong_command["hardware_command_authorized"] = True
    wrong_command["detect_attempt_count_command_stdout_stderr_exit_code"].update(
        {"attempt_count": 1, "command": "openFPGALoader -c dirtyJtag --flash x"}
    )
    assert any(
        "detect command allowlist" in err for err in mod.artifact_schema_errors(wrong_command)
    )

    unauthorized = deepcopy(artifact)
    unauthorized["detect_attempt_count_command_stdout_stderr_exit_code"].update(
        {"attempt_count": 1, "command": "openFPGALoader -c dirtyJtag --detect"}
    )
    assert any("unauthorized state" in err for err in mod.artifact_schema_errors(unauthorized))

    mutated = deepcopy(artifact)
    mutated["mutation_command_counts"]["flash"] = 1
    with pytest.raises(ValueError, match="mutation"):
        mod.validate_artifact(mutated)

    claim = deepcopy(artifact)
    claim["speed_power_energy_terminal_tsu_kona_claim_counts"]["energy"] = 1
    assert any("claim counts" in err for err in mod.artifact_schema_errors(claim))


def test_req_hw_6199_defensive_fallbacks_and_schema_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-HW-6199: fallback receipts and malformed artifacts fail closed."""

    assert mod.path_receipt(tmp_path, "missing.json") == {
        "path": "missing.json",
        "present": False,
        "bytes": 0,
        "sha256": None,
    }
    assert mod.read_json_object(tmp_path, "missing.json") == {}
    assert mod.canonical_prior_physical_state(tmp_path) == mod.CANONICAL_PRIOR_PHYSICAL_STATE

    unchanged_material = _changed_receipt()
    for field in mod.MATERIAL_PHYSICAL_FIELDS:
        unchanged_material[field] = mod.CANONICAL_PRIOR_PHYSICAL_STATE[field]
    prior, current, changed, reason = mod.physical_state_comparison(tmp_path, unchanged_material)
    assert prior == current
    assert changed is False
    assert reason == "no_material_physical_change"

    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(55.0, 55.5),
        run_date="20260807",
    )

    missing = deepcopy(artifact)
    del missing["status"]
    assert any("missing required fields" in err for err in mod.artifact_schema_errors(missing))

    metadata = deepcopy(artifact)
    metadata["schema"] = "wrong"
    metadata["spec_refs"] = []
    metadata["random_seed"] = 0
    metadata["inference_substrate"] = "wrong"
    metadata["field_principles"] = {}
    errors = mod.artifact_schema_errors(metadata)
    assert "schema mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "random_seed mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert "field_principles mismatch" in errors

    bad_detect = deepcopy(artifact)
    bad_detect["detect_attempt_count_command_stdout_stderr_exit_code"] = []
    assert any("detect receipt" in err for err in mod.artifact_schema_errors(bad_detect))

    stale_stdout = deepcopy(artifact)
    stale_stdout["detect_attempt_count_command_stdout_stderr_exit_code"]["stdout"] = "stale"
    stale_stdout["operator_action_packet"] = {}
    errors = mod.artifact_schema_errors(stale_stdout)
    assert any("stdout/stderr" in err for err in errors)
    assert any("exact operator action packet" in err for err in errors)

    historical = deepcopy(artifact)
    historical["historical_flagged_terminal_evidence_excluded"]["excluded"] = False
    assert any("historical flagged" in err for err in mod.artifact_schema_errors(historical))

    auth = deepcopy(artifact)
    auth["hardware_execution_authenticated"]["authenticated"] = True
    assert any("hardware execution" in err for err in mod.artifact_schema_errors(auth))

    protected = deepcopy(artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    assert any("protected files" in err for err in mod.artifact_schema_errors(protected))

    verdict = deepcopy(artifact)
    verdict["honest_verdict"] = "success: wrong"
    assert any("honest_verdict prefix" in err for err in mod.artifact_schema_errors(verdict))


def test_req_hw_6199_run_experiment_writes_requested_artifact(tmp_path: Path) -> None:
    """REQ-HW-6199: run_experiment writes the terminal audit JSON atomically."""

    out = mod.run_experiment(
        repo_root=tmp_path,
        source_root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(60.0, 60.5),
        run_date="20260807",
    )
    artifact = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["status"] == "blocked_missing_receipt"
    assert artifact["detect_attempt_count_command_stdout_stderr_exit_code"]["attempt_count"] == 0
    mod.validate_artifact(artifact)


def test_req_hw_6199_main_prints_audit_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    """REQ-HW-6199: CLI prints the artifact path and command count."""

    real_run_experiment = mod.run_experiment

    def fake_run_experiment(*, repo_root: Path, run_date: str) -> Path:
        return real_run_experiment(
            repo_root=tmp_path,
            source_root=REPO_ROOT,
            command_runner=RecordingRunner(),
            clock=StepClock(70.0, 70.5),
            run_date=run_date,
        )

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    rc = mod.main(["--date", "20260807", "--repo-root", str(tmp_path)])
    captured = capsys.readouterr().out

    assert rc == 0
    assert mod.OUTPUT_REL_PATH.name in captured
    assert "physical_state_changed: False" in captured
    assert "detect_attempt_count: 0" in captured
