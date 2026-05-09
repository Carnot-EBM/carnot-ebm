"""Tests for Exp 1584 Tenstorrent Wormhole block-Gibbs preflight.

Spec traces: REQ-SAMPLE-064, SCENARIO-SAMPLE-092.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import carnot.hardware.tenstorrent_wormhole_preflight as exp1584
from carnot.hardware.tenstorrent_wormhole_preflight import (
    REQUIRED_ARTIFACT_FIELDS,
    AvailabilitySummary,
    CommandResult,
    build_artifact,
    probe_availability,
    run_preflight,
    validate_artifact,
    write_in_progress_artifact,
)


def _result(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> CommandResult:
    return CommandResult(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_s=0.01,
    )


def test_req_sample_064_spec_anchor_exists() -> None:
    """REQ-SAMPLE-064, SCENARIO-SAMPLE-092: Exp 1584 is spec-anchored."""

    spec = (
        exp1584.PROJECT_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-SAMPLE-064" in spec
    assert "SCENARIO-SAMPLE-092" in spec
    assert "experiment_1584_tenstorrent_wormhole_n150d_block_gibbs_preflight.json" in spec


def test_req_sample_064_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-064: the deliverable starts with a schema-shaped marker."""

    output = tmp_path / "experiment_1584_tenstorrent_wormhole_n150d_block_gibbs_preflight.json"

    marker = write_in_progress_artifact(output)

    assert REQUIRED_ARTIFACT_FIELDS <= set(marker)
    assert marker["status"] == "in_progress"
    assert marker["wormhole_access_available"] is False
    assert marker["tt_metalium_available"] is False
    assert marker["wormhole_preflight_ready"] is False
    assert marker["no_hardware_claim_without_transcript"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_req_sample_064_probe_records_missing_local_and_remote_access(tmp_path: Path) -> None:
    """REQ-SAMPLE-064: safe probes do not turn public reachability into access."""

    commands: list[list[str]] = []

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        commands.append(command)
        if command[0] == "python3":
            return _result(command, returncode=1, stderr="ModuleNotFoundError: ttnn")
        return _result(command, stdout="reachable")

    summary = probe_availability(
        project_root=tmp_path,
        env={},
        runner=_runner,
        command_lookup=lambda name: "/usr/bin/git" if name == "git" else None,
        device_paths=[],
    )

    assert commands[0] == ["python3", "-c", exp1584.TT_IMPORT_PROBE_CODE]
    assert commands[1][:2] == ["git", "ls-remote"]
    assert summary.tt_metalium_available is False
    assert summary.wormhole_access_available is False
    assert summary.remote_reachability["tt_metal_github"]["reachable"] is True


def test_scenario_sample_092_ready_requires_smoke_transcript(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-092: readiness is gated by access, TT-Metalium, and smoke."""

    env = {"TT_METAL_HOME": str(tmp_path / "tt-metal"), "TT_REMOTE_HOST": "tt-box"}
    (tmp_path / "tt-metal").mkdir()

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if command[0] == "python3":
            return _result(command, stdout='["ttnn"]\n')
        if command[0] == "git":
            return _result(command, stdout="reachable")
        return _result(command, stdout="Wormhole n150d ready\n")

    summary = probe_availability(
        project_root=tmp_path,
        env=env,
        runner=_runner,
        command_lookup=lambda name: "/usr/bin/tt-smi" if name == "tt-smi" else "/usr/bin/git",
        device_paths=[tmp_path / "tenstorrent0"],
    )
    transcript = tmp_path / "logs" / "tt.txt"
    report = tmp_path / "docs" / "tt.md"
    artifact = build_artifact(
        summary=summary,
        smoke_result={"status": "smoke_passed", "successful": True, "command": ["tt-smi"]},
        hardware_transcript_path=transcript,
        benchmark_protocol_path=report,
    )

    validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["wormhole_access_available"] is True
    assert artifact["tt_metalium_available"] is True
    assert artifact["wormhole_preflight_ready"] is True
    assert artifact["blocked_reason"] == "not_blocked"


def test_req_sample_064_validator_rejects_dishonest_readiness(tmp_path: Path) -> None:
    """REQ-SAMPLE-064: ready artifacts must include transcript evidence."""

    valid = {
        "status": "blocked",
        "wormhole_access_available": False,
        "tt_metalium_available": False,
        "hardware_transcript_path": str(tmp_path / "transcript.txt"),
        "benchmark_protocol_path": str(tmp_path / "protocol.md"),
        "wormhole_preflight_ready": False,
        "blocked_reason": "TT-Metalium not importable; Wormhole hardware not visible.",
        "no_hardware_claim_without_transcript": True,
        "honest_verdict": "complete: wormhole_preflight_blocked_no_access_no_hardware_claim",
    }
    validate_artifact(valid)

    missing = dict(valid)
    missing.pop("hardware_transcript_path")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    dishonest = dict(valid)
    dishonest["no_hardware_claim_without_transcript"] = False
    with pytest.raises(ValueError, match="no_hardware_claim_without_transcript"):
        validate_artifact(dishonest)

    ready_without_transcript = dict(valid)
    ready_without_transcript.update(
        {
            "status": "complete",
            "wormhole_access_available": True,
            "tt_metalium_available": True,
            "wormhole_preflight_ready": True,
            "hardware_transcript_path": "",
        }
    )
    with pytest.raises(ValueError, match="transcript"):
        validate_artifact(ready_without_transcript)


def test_scenario_sample_092_run_preflight_writes_blocked_artifacts(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-092: runner writes JSON, transcript, and protocol report."""

    output = tmp_path / "results" / "experiment_1584_tenstorrent.json"
    transcript = tmp_path / "logs" / "exp1584.txt"
    report = tmp_path / "docs" / "tenstorrent.md"

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if command[0] == "python3":
            return _result(command, returncode=1, stderr="No module named 'ttnn'")
        if command[0] == "git":
            return _result(command, returncode=2, stderr="network blocked")
        return _result(command, returncode=127, stderr="missing command")

    artifact = run_preflight(
        output_path=output,
        hardware_transcript_path=transcript,
        benchmark_protocol_path=report,
        project_root=tmp_path,
        env={},
        runner=_runner,
        command_lookup=lambda name: "/usr/bin/git" if name == "git" else None,
        device_paths=[],
        run_date="20260509",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["status"] == "blocked"
    assert payload["wormhole_preflight_ready"] is False
    assert payload["hardware_transcript_path"] == str(transcript)
    assert "KL to THRML" in report.read_text(encoding="utf-8")
    assert "TT-Metalium import probe" in transcript.read_text(encoding="utf-8")


def test_req_sample_064_command_payload_and_text_compaction() -> None:
    """REQ-SAMPLE-064: command and probe payloads remain artifact-sized."""

    completed = exp1584.run_command(["python3", "-c", "print('ok')"], timeout_s=10.0)
    long_text = "x" * 5000
    payload = exp1584.command_payload(
        _result(["cmd"], stdout=long_text, stderr=long_text, timed_out=True)
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "ok"
    assert payload["stdout"].endswith("...")
    assert payload["stderr"].endswith("...")
    assert payload["timed_out"] is True
    assert exp1584._json_list_from_stdout("not json") == []
    assert exp1584._json_list_from_stdout("{}") == []
    assert isinstance(exp1584._device_path_strings(None), list)


def test_req_sample_064_blocked_reason_covers_partial_access(tmp_path: Path) -> None:
    """REQ-SAMPLE-064: partial access explains exactly what is missing."""

    summary = AvailabilitySummary(
        tt_metalium_available=True,
        wormhole_access_available=False,
        tt_metalium_signals=[{"name": "TT_METAL_HOME", "available": True}],
        wormhole_access_signals=[],
        remote_reachability={},
    )

    artifact = build_artifact(
        summary=summary,
        smoke_result={"status": "skipped_unavailable", "successful": False},
        hardware_transcript_path=tmp_path / "transcript.txt",
        benchmark_protocol_path=tmp_path / "protocol.md",
    )

    assert artifact["blocked_reason"] == "Wormhole hardware or cloud access was not detected."


def test_req_sample_064_remote_skip_and_smoke_failure_paths(tmp_path: Path) -> None:
    """REQ-SAMPLE-064: absent git and failed safe smoke remain explicit."""

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if command[0] == "python3":
            return _result(command, stdout="{}\n")
        return _result(command, returncode=4, stderr="device query failed")

    summary = probe_availability(
        project_root=tmp_path,
        env={},
        runner=_runner,
        command_lookup=lambda name: None,
        device_paths=[],
    )
    assert summary.remote_reachability["tt_metal_github"]["status"] == "skipped_git_not_found"

    ready_summary = AvailabilitySummary(
        tt_metalium_available=True,
        wormhole_access_available=True,
        tt_metalium_signals=[],
        wormhole_access_signals=[],
        remote_reachability={},
    )
    no_command = exp1584.run_non_destructive_smoke(
        summary=ready_summary,
        runner=_runner,
        command_lookup=lambda name: None,
    )
    failed = exp1584.run_non_destructive_smoke(
        summary=ready_summary,
        runner=_runner,
        command_lookup=lambda name: "/usr/bin/tt-smi" if name == "tt-smi" else None,
    )
    artifact = build_artifact(
        summary=ready_summary,
        smoke_result=failed,
        hardware_transcript_path=tmp_path / "transcript.txt",
        benchmark_protocol_path=tmp_path / "protocol.md",
    )

    assert no_command["status"] == "skipped_no_safe_smoke_command"
    assert failed["status"] == "smoke_failed"
    assert artifact["blocked_reason"] == (
        "Wormhole and TT-Metalium were detected, but the non-destructive smoke did not pass."
    )


def test_req_sample_064_validator_rejects_remaining_bad_states(tmp_path: Path) -> None:
    """REQ-SAMPLE-064: validator guards all terminal consistency edges."""

    ready = {
        "status": "complete",
        "wormhole_access_available": True,
        "tt_metalium_available": True,
        "hardware_transcript_path": str(tmp_path / "transcript.txt"),
        "benchmark_protocol_path": str(tmp_path / "protocol.md"),
        "wormhole_preflight_ready": True,
        "blocked_reason": "not_blocked",
        "no_hardware_claim_without_transcript": True,
        "honest_verdict": "complete: ready",
    }
    validate_artifact(ready)

    no_protocol = dict(ready)
    no_protocol["benchmark_protocol_path"] = ""
    with pytest.raises(ValueError, match="benchmark_protocol_path"):
        validate_artifact(no_protocol)

    bad_status = dict(ready)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status=complete"):
        validate_artifact(bad_status)

    missing_access = dict(ready)
    missing_access["wormhole_access_available"] = False
    with pytest.raises(ValueError, match="both access"):
        validate_artifact(missing_access)

    bad_reason = dict(ready)
    bad_reason["blocked_reason"] = "smoke failed"
    with pytest.raises(ValueError, match="not_blocked"):
        validate_artifact(bad_reason)

    bad_terminal = dict(ready)
    bad_terminal.update(
        {
            "status": "pending",
            "wormhole_preflight_ready": False,
            "blocked_reason": "still running",
        }
    )
    with pytest.raises(ValueError, match="terminal artifacts"):
        validate_artifact(bad_terminal)

    no_blocked_reason = dict(ready)
    no_blocked_reason.update(
        {
            "status": "blocked",
            "wormhole_preflight_ready": False,
            "blocked_reason": "",
        }
    )
    with pytest.raises(ValueError, match="blocked_reason"):
        validate_artifact(no_blocked_reason)

    bad_verdict = dict(ready)
    bad_verdict["honest_verdict"] = "blocked_no_prefix"
    with pytest.raises(ValueError, match="terminal prefix"):
        validate_artifact(bad_verdict)
