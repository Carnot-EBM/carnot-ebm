"""Tests for Exp6325 GateMate dated-receipt single detect.

Spec refs: REQ-HW-6325, SCENARIO-HW-6325-1, SCENARIO-HW-6325-2,
SCENARIO-HW-6325-3, SCENARIO-HW-6325-4, SCENARIO-HW-6325-5,
SCENARIO-HW-6325-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6325_gatemate_dated_receipt_single_detect as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "hardware" / "spec.md"


class RecordingRunner:
    """REQ-HW-6325 fake runner records all command calls for budget checks."""

    def __init__(self, receipts: dict[tuple[str, ...], list[mod.CommandReceipt]] | None = None):
        self.receipts = {key: list(value) for key, value in (receipts or {}).items()}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandReceipt:
        assert timeout_s > 0
        command = tuple(command)
        self.calls.append(command)
        if command not in self.receipts or not self.receipts[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.receipts[command].pop(0)

    @property
    def detect_calls(self) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call == mod.DETECT_COMMAND]


class StepClock:
    """Deterministic UTC clock for stable timestamp assertions."""

    def __init__(self, *values: float):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _receipt() -> dict:
    return deepcopy(mod.EXPECTED_DATED_PHYSICAL_RECEIPT)


def _command_receipt(
    command: tuple[str, ...],
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timeout: bool = False,
    duration_s: float = 0.02,
) -> mod.CommandReceipt:
    return mod.CommandReceipt(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
        timeout=timeout,
    )


def _usb_stdout() -> str:
    return "Bus 003 Device 011: ID 1209:c0ca Generic DirtyJTAG\n"


def _hit_stdout() -> str:
    return (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )


def _runner_for_detect(detect_receipt: mod.CommandReceipt) -> RecordingRunner:
    return RecordingRunner(
        {
            mod.LSUSB_COMMAND: [
                _command_receipt(mod.LSUSB_COMMAND, stdout=_usb_stdout()),
                _command_receipt(mod.LSUSB_COMMAND, stdout=_usb_stdout()),
            ],
            mod.VERSION_COMMAND: [
                _command_receipt(mod.VERSION_COMMAND, stdout="openFPGALoader v1.1.1\n")
            ],
            mod.DETECT_COMMAND: [detect_receipt],
        }
    )


def test_req_hw_6325_spec_defines_single_detect_contract() -> None:
    """REQ-HW-6325: OpenSpec anchors the single-detect artifact."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("### REQ-HW-6325", maxsplit=1)[1]

    for marker in (
        "REQ-HW-6325",
        "SCENARIO-HW-6325-1",
        "SCENARIO-HW-6325-2",
        "SCENARIO-HW-6325-3",
        "SCENARIO-HW-6325-4",
        "SCENARIO-HW-6325-5",
        "SCENARIO-HW-6325-6",
        mod.OUTPUT_REL_PATH.as_posix(),
        mod.EXACT_AUTHORIZED_COMMAND,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_hw_6325_matching_idcode_records_one_visible_detect() -> None:
    """SCENARIO-HW-6325-1: matching IDCODE is visibility only."""

    runner = _runner_for_detect(_command_receipt(mod.DETECT_COMMAND, stdout=_hit_stdout()))
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(0.0, 1.0, 2.0, 3.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
        protected_before_hashes=mod.protected_file_hashes(REPO_ROOT),
    )

    assert runner.detect_calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "complete_visible"
    assert artifact["detect_command_count"] == 1
    assert artifact["exact_authorized_command"] == mod.EXACT_AUTHORIZED_COMMAND
    assert artifact["detect_stdout"] == _hit_stdout()
    assert artifact["detect_stderr"] == ""
    assert artifact["detect_exit_code"] == 0
    assert artifact["detect_timeout"] is False
    assert artifact["detected_chain_and_device_ids"]["idcodes"] == ["0x20000001"]
    assert artifact["detected_chain_and_device_ids"]["expected_gatemate_idcode_seen"] is True
    assert artifact["hardware_state_changed_from_prior_attempts"]["changed"] is True
    assert artifact["receipt_newer_than_prior_attempts"]["newer_than_all_prior_attempts"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete_visible:")
    for field in mod.FORBIDDEN_COUNT_FIELDS:
        assert artifact[field] == 0
    mod.validate_artifact(artifact)


def test_scenario_hw_6325_empty_chain_records_one_attempt_and_stops() -> None:
    """SCENARIO-HW-6325-2: empty chain is preserved as the raw outcome."""

    runner = _runner_for_detect(
        _command_receipt(mod.DETECT_COMMAND, stdout="Jtag frequency ok\nfound 0 devices\n")
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(10.0, 11.0, 12.0, 13.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
    )

    assert runner.detect_calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "blocked_empty_chain"
    assert artifact["detect_command_count"] == 1
    assert artifact["detected_chain_and_device_ids"]["chain_empty"] is True
    assert artifact["stop_after_single_attempt_receipt"]["stopped_after_single_attempt"] is True
    assert artifact["honest_verdict"].startswith("blocked_empty_chain:")
    mod.validate_artifact(artifact)


def test_scenario_hw_6325_timeout_records_one_attempt_and_no_retry() -> None:
    """SCENARIO-HW-6325-3: timeout is terminal and does not retry."""

    runner = _runner_for_detect(
        _command_receipt(
            mod.DETECT_COMMAND,
            stderr="timed out after 30.0s",
            exit_code=124,
            timeout=True,
            duration_s=30.0,
        )
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(20.0, 21.0, 52.0, 53.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
    )

    assert runner.detect_calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "blocked_timeout"
    assert artifact["detect_command_count"] == 1
    assert artifact["detect_timeout"] is True
    assert "timed out" in artifact["detect_stderr"]
    assert artifact["stop_after_single_attempt_receipt"]["retry_count"] == 0
    assert artifact["honest_verdict"].startswith("blocked_timeout:")
    mod.validate_artifact(artifact)


def test_scenario_hw_6325_wrong_idcode_and_tool_failure_stop_after_one_attempt() -> None:
    """SCENARIO-HW-6325-2: wrong IDs and tool failures are terminal outcomes."""

    wrong_runner = _runner_for_detect(
        _command_receipt(mod.DETECT_COMMAND, stdout="index 0:\n\tidcode 0x12345678\n")
    )
    wrong = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=wrong_runner,
        clock=StepClock(24.0, 25.0, 26.0, 27.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
    )

    failed_runner = _runner_for_detect(
        _command_receipt(mod.DETECT_COMMAND, stderr="dirtyjtag open failed\n", exit_code=2)
    )
    failed = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=failed_runner,
        clock=StepClock(28.0, 29.0, 30.0, 31.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
    )

    assert wrong["status"] == "blocked_idcode"
    assert wrong["detected_chain_and_device_ids"]["idcodes"] == ["0x12345678"]
    assert wrong["honest_verdict"].startswith("blocked_idcode:")
    assert failed["status"] == "blocked_detect_failed"
    assert failed["detect_exit_code"] == 2
    assert failed["honest_verdict"].startswith("blocked_detect_failed:")
    mod.validate_artifact(wrong)
    mod.validate_artifact(failed)


def test_scenario_hw_6325_stale_receipt_blocks_before_detect() -> None:
    """SCENARIO-HW-6325-4: stale receipts run zero hardware commands."""

    stale = _receipt()
    stale["receipt_date"] = "20260807"
    runner = RecordingRunner(
        {
            mod.LSUSB_COMMAND: [_command_receipt(mod.LSUSB_COMMAND, stdout=_usb_stdout())],
            mod.VERSION_COMMAND: [
                _command_receipt(mod.VERSION_COMMAND, stdout="openFPGALoader v1.1.1\n")
            ],
        }
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(30.0, 31.0),
        run_date="20260812",
        dated_physical_receipt=stale,
    )

    assert runner.detect_calls == []
    assert artifact["status"] == "blocked_stale_receipt"
    assert artifact["detect_command_count"] == 0
    assert artifact["detect_stdout"] == ""
    assert artifact["detect_stderr"] == ""
    assert artifact["receipt_newer_than_prior_attempts"]["newer_than_all_prior_attempts"] is False
    assert artifact["honest_verdict"].startswith("blocked_stale_receipt:")
    mod.validate_artifact(artifact)


def test_scenario_hw_6325_wrong_target_blocks_before_detect() -> None:
    """SCENARIO-HW-6325-5: wrong board or cable target runs zero detects."""

    wrong = _receipt()
    wrong["board"] = "AMD Xilinx KV260"
    wrong["usb_dirtyjtag"] = "03fd:0008 Xilinx JTAG"
    runner = RecordingRunner(
        {
            mod.LSUSB_COMMAND: [_command_receipt(mod.LSUSB_COMMAND, stdout=_usb_stdout())],
            mod.VERSION_COMMAND: [
                _command_receipt(mod.VERSION_COMMAND, stdout="openFPGALoader v1.1.1\n")
            ],
        }
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(40.0, 41.0),
        run_date="20260812",
        dated_physical_receipt=wrong,
    )

    assert runner.detect_calls == []
    assert artifact["status"] == "blocked_wrong_target"
    assert artifact["detect_command_count"] == 0
    assert artifact["board_and_cable_target"]["target_ok"] is False
    assert artifact["kv260_command_count"] == 0
    assert artifact["polarfire_command_count"] == 0
    assert artifact["honest_verdict"].startswith("blocked_wrong_target:")
    mod.validate_artifact(artifact)


def test_req_hw_6325_missing_receipt_and_tool_version_fail_closed() -> None:
    """REQ-HW-6325: missing receipts and missing tool versions block detects."""

    missing_runner = RecordingRunner(
        {
            mod.LSUSB_COMMAND: [_command_receipt(mod.LSUSB_COMMAND, stdout=_usb_stdout())],
            mod.VERSION_COMMAND: [
                _command_receipt(mod.VERSION_COMMAND, stdout="openFPGALoader v1.1.1\n")
            ],
        }
    )
    missing = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=missing_runner,
        clock=StepClock(44.0, 45.0),
        run_date="20260812",
        dated_physical_receipt={"exists": False, "receipt_date": None, "changes": []},
    )

    tool_runner = RecordingRunner(
        {
            mod.LSUSB_COMMAND: [_command_receipt(mod.LSUSB_COMMAND, stdout=_usb_stdout())],
            mod.VERSION_COMMAND: [
                _command_receipt(
                    mod.VERSION_COMMAND,
                    stderr="openFPGALoader: command not found",
                    exit_code=127,
                )
            ],
        }
    )
    tool = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=tool_runner,
        clock=StepClock(46.0, 47.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
    )

    assert missing_runner.detect_calls == []
    assert tool_runner.detect_calls == []
    assert missing["status"] == "blocked_missing_receipt"
    assert missing["honest_verdict"].startswith("blocked_missing_receipt:")
    assert tool["status"] == "blocked_tool_version"
    assert tool["openfpgaloader_version_receipt"]["exit_code"] == 127
    assert tool["honest_verdict"].startswith("blocked_tool_version:")
    mod.validate_artifact(missing)
    mod.validate_artifact(tool)


def test_scenario_hw_6325_schema_refuses_second_detect_and_forbidden_counts() -> None:
    """SCENARIO-HW-6325-6: validation fails closed on budget violations."""

    runner = _runner_for_detect(_command_receipt(mod.DETECT_COMMAND, stdout=_hit_stdout()))
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(50.0, 51.0, 52.0, 53.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
    )

    too_many = deepcopy(artifact)
    too_many["detect_command_count"] = 2
    assert any("detect_command_count" in err for err in mod.artifact_schema_errors(too_many))

    bad_command = deepcopy(artifact)
    bad_command["exact_authorized_command"] = "openFPGALoader -c dirtyJtag --flash x"
    assert any("exact_authorized_command" in err for err in mod.artifact_schema_errors(bad_command))

    flash = deepcopy(artifact)
    flash["flash_command_count"] = 1
    with pytest.raises(ValueError, match="forbidden command counts"):
        mod.validate_artifact(flash)

    principles = deepcopy(artifact)
    del principles["field_principles"]["status"]
    assert any("field_principles" in err for err in mod.artifact_schema_errors(principles))

    missing = deepcopy(artifact)
    del missing["status"]
    assert any("missing required fields" in err for err in mod.artifact_schema_errors(missing))

    metadata = deepcopy(artifact)
    metadata["schema"] = "wrong"
    metadata["spec_refs"] = []
    metadata["random_seed"] = 0
    metadata["inference_substrate"] = "wrong"
    metadata["field_provenance"] = {}
    errors = mod.artifact_schema_errors(metadata)
    assert "schema mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "random_seed mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert any("field_provenance" in err for err in errors)

    missing_utc = deepcopy(artifact)
    missing_utc["detect_started_utc"] = None
    assert any("UTC" in err for err in mod.artifact_schema_errors(missing_utc))

    blocked = deepcopy(artifact)
    blocked["status"] = "blocked_stale_receipt"
    blocked["detect_command_count"] = 1
    blocked["detect_stdout"] = "stale"
    errors = mod.artifact_schema_errors(blocked)
    assert any("zero detect" in err for err in errors)
    assert any("stdout/stderr" in err for err in errors)

    stop = deepcopy(artifact)
    stop["stop_after_single_attempt_receipt"]["retry_count"] = 1
    assert any("zero retries" in err for err in mod.artifact_schema_errors(stop))

    protected = deepcopy(artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    assert any("protected files" in err for err in mod.artifact_schema_errors(protected))

    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = False
    assert any("verifier_is_oracle" in err for err in mod.artifact_schema_errors(oracle))

    chain = deepcopy(artifact)
    chain["detected_chain_and_device_ids"] = []
    assert any("detected_chain" in err for err in mod.artifact_schema_errors(chain))

    visible_without_id = deepcopy(artifact)
    visible_without_id["detected_chain_and_device_ids"]["expected_gatemate_idcode_seen"] = False
    assert any("complete_visible" in err for err in mod.artifact_schema_errors(visible_without_id))

    verdict = deepcopy(artifact)
    verdict["honest_verdict"] = "success: wrong"
    assert any("honest_verdict" in err for err in mod.artifact_schema_errors(verdict))


def test_req_hw_6325_helper_fallbacks_and_default_receipt_parser(tmp_path: Path) -> None:
    """REQ-HW-6325: helper fallbacks stay explicit and receipt parsing is dated."""

    assert mod._coerce_timeout_text(None) == ""
    assert mod._coerce_timeout_text(b"abc") == "abc"
    assert mod._coerce_timeout_text("abc") == "abc"
    assert mod.read_json_object(tmp_path, "missing.json") == {}
    assert mod.extracted_receipt_text(tmp_path) == (
        {"exists": False, "receipt_date": None, "changes": []},
        "",
    )

    missing_anchor = tmp_path / mod.KNOWN_ISSUES_REL_PATH
    missing_anchor.parent.mkdir(parents=True)
    missing_anchor.write_text("no receipt here\n", encoding="utf-8")
    assert mod.extracted_receipt_text(tmp_path) == (
        {"exists": False, "receipt_date": None, "changes": []},
        "",
    )

    missing_anchor.write_text("2026-08-11 operator physical action\nno json block\n", encoding="utf-8")
    assert mod.extracted_receipt_text(tmp_path) == (
        {"exists": False, "receipt_date": None, "changes": []},
        "",
    )

    parsed, field = mod.dated_physical_receipt_path_hash_date_and_text(REPO_ROOT, None)
    assert parsed["receipt_date"] == "20260811"
    assert parsed["board"] == mod.EXPECTED_BOARD
    assert field["source"] == "ops_known_issues_structured_json_block"
    assert field["text_sha256"].startswith("sha256:")


def test_req_hw_6325_run_experiment_writes_requested_artifact(tmp_path: Path) -> None:
    """REQ-HW-6325: run_experiment writes the deliverable JSON."""

    runner = _runner_for_detect(_command_receipt(mod.DETECT_COMMAND, stdout=_hit_stdout()))
    out = mod.run_experiment(
        repo_root=tmp_path,
        source_root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(60.0, 61.0, 62.0, 63.0),
        run_date="20260812",
        dated_physical_receipt=_receipt(),
    )

    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert out == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["detect_command_count"] == 1
    assert artifact["result_path"] == mod.OUTPUT_REL_PATH.as_posix()
    mod.validate_artifact(artifact)


def test_req_hw_6325_main_prints_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    """REQ-HW-6325: CLI prints the artifact path and detect count."""

    real_run_experiment = mod.run_experiment

    def fake_run_experiment(*, repo_root: Path, run_date: str) -> Path:
        runner = _runner_for_detect(_command_receipt(mod.DETECT_COMMAND, stdout=_hit_stdout()))
        return real_run_experiment(
            repo_root=tmp_path,
            source_root=REPO_ROOT,
            command_runner=runner,
            clock=StepClock(70.0, 71.0, 72.0, 73.0),
            run_date=run_date,
            dated_physical_receipt=_receipt(),
        )

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    rc = mod.main(["--date", "20260812", "--repo-root", str(tmp_path)])
    captured = capsys.readouterr().out

    assert rc == 0
    assert mod.OUTPUT_REL_PATH.name in captured
    assert "detect_command_count: 1" in captured
    assert "detect_timeout: False" in captured
