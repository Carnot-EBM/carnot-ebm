"""Artifact and subprocess tests for Exp6633.

Spec refs: REQ-REPORT-6633, SCENARIO-REPORT-6633-READY,
SCENARIO-REPORT-6633-BLOCKED, SCENARIO-REPORT-6633-ATOMIC-EVIDENCE,
REQ-INFRA-6633, SCENARIO-INFRA-6633-ATOMIC-RACE,
SCENARIO-INFRA-6633-INDEPENDENT-DEVICES, and
SCENARIO-INFRA-6633-CRASH-RECOVERY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_6633_gpu_lease_phase_journal as exp


REPO = Path(__file__).resolve().parents[2]


def _tests_run() -> list[dict]:
    return [
        {"command": "focused", "exit_code": 0, "summary": "focused tests passed"},
        {"command": "coverage", "exit_code": 0, "summary": "new code covered 100%"},
    ]


def _ready_artifact(tmp_path: Path) -> dict:
    fixtures = [
        {
            "fixture_id": fixture_id,
            "passed": True,
            "owner": {
                "device_uuid": "GPU-test",
                "pid": 123,
                "pid_start_ticks": 456,
                "expected_model": "fixture/model.gguf",
                "token_digest": "sha256:test",
            },
        }
        for fixture_id in exp.REQUIRED_PROCESS_FIXTURE_IDS
    ]
    attacks = [
        {"attack_id": attack_id, "accepted": False, "fail_closed": True}
        for attack_id in exp.REQUIRED_ATTACK_IDS
    ]
    return exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=1.0,
        tests_run=_tests_run(),
        process_fixtures={"rows": fixtures, "all_passed": True, "failed_checks": []},
        phase_transition_rows=[{"accepted": True}, {"accepted": False}],
        attack_rows=attacks,
        protected_before=exp.protected_hashes(REPO),
        lease_api_receipts={"all_owner_bound": True},
    )


def test_req_report_6633_spec_and_principled_fields() -> None:
    """REQ-REPORT-6633: the result schema and no-LLM boundary are explicit."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6633") :]
    for anchor in (
        "SCENARIO-REPORT-6633-READY",
        "SCENARIO-REPORT-6633-BLOCKED",
        "SCENARIO-REPORT-6633-ATOMIC-EVIDENCE",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert anchor in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_process_fixtures_cover_race_crash_stale_tamper_and_restart(tmp_path: Path) -> None:
    """Bounded fixtures replay atomic ownership and crash recovery."""

    result = exp.run_process_fixtures(tmp_path)
    rows = {row["fixture_id"]: row for row in result["rows"]}
    assert set(rows) == set(exp.REQUIRED_PROCESS_FIXTURE_IDS)
    assert result["all_passed"] is True
    assert rows["same_device_race"]["acquired_count"] == 1
    assert rows["same_device_race"]["busy_count"] == 1
    assert rows["independent_devices"]["acquired_count"] == 2
    assert rows["owner_crash"]["crash_exit_code"] == exp.FIXTURE_CRASH_EXIT
    assert rows["stale_heartbeat"]["live_contender_outcome"] == "busy"
    assert rows["tamper"]["outcome"] == "fail_closed"
    assert rows["partial_write"]["final_checksum_unchanged"] is True
    assert rows["pid_reuse"]["outcome"] == "fail_closed"
    assert rows["restart_recovery"]["recovery_performed"] is True
    assert all(row["bounded"] is True for row in rows.values())
    assert all(row["signals_sent"] == [] for row in rows.values())


def test_phase_and_attack_rows_cover_every_required_failure(tmp_path: Path) -> None:
    """Rejected transitions and owner attacks cannot open readiness."""

    transitions = exp.build_phase_transition_rows(tmp_path / "phases")
    attacks = exp.build_attack_rows(tmp_path / "attacks")
    assert any(row["accepted"] is True for row in transitions)
    assert {row["attack_id"] for row in attacks} == set(exp.REQUIRED_ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in attacks)
    assert all(row["accepted"] is False for row in attacks)


def test_scenario_report_6633_ready_artifact_replays_and_writes_atomically(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6633-READY and ATOMIC-EVIDENCE produce valid JSON."""

    output = tmp_path / exp.RESULT_PATH.name
    artifact = exp.run(
        date="20260826",
        root=REPO,
        result_path=output,
        work_dir=tmp_path / "work",
        tests_run=_tests_run(),
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "terminal_complete"
    assert artifact["verdict_class"] is None
    assert artifact["gate_check_summary"] == []
    assert artifact["gpu_lease_scheduler_ready_score"] == 1.0
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["no_llm"]["model_load_attempt_count"] == 0
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)


def test_scenario_report_6633_blocked_and_validator_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6633-BLOCKED names every failed observed value."""

    artifact = exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=1.0,
        tests_run=_tests_run(),
        process_fixtures={
            "rows": [],
            "all_passed": False,
            "failed_checks": [{"check": "process_fixtures", "expected": True, "observed": False}],
        },
        phase_transition_rows=[],
        attack_rows=[],
        protected_before=exp.protected_hashes(REPO),
    )
    assert artifact["status"].startswith("blocked_")
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gpu_lease_scheduler_ready_score"] == 0.0
    assert artifact["gate_check_summary"][0] == {
        "check": "process_fixtures",
        "expected": sorted(exp.REQUIRED_PROCESS_FIXTURE_IDS),
        "observed": [],
    }
    assert exp.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["gpu_lease_scheduler_ready_score"] = 1.0
    bad["reproducibility_checksum"] = exp.payload_checksum(bad)
    assert "readiness_score_mismatch" in exp.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = exp.payload_checksum(bad)
    assert "field_provenance_mismatch" in exp.validate_artifact(bad)


def test_req_report_6633_cli_run_and_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6633: the documented module command runs end to end."""

    output = tmp_path / "cli.json"
    work = tmp_path / "cli-work"
    assert (
        exp.main(
            [
                "--date",
                "20260826",
                "--output",
                str(output),
                "--work-dir",
                str(work),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["gpu_lease_scheduler_ready_score"] == 1.0
    assert exp.main(["--validate", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["valid"] is True

    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", "--output", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["valid"] is False

    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(unreadable)]) == 1
    error = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert error["errors"] == ["artifact_unreadable:JSONDecodeError"]


def test_scenario_report_6633_subprocess_and_diagnostic_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6633-ATOMIC-EVIDENCE bounds child failure paths."""

    assert exp.sha256_file(tmp_path / "missing") == "missing"

    class NoStdout:
        stdout = None

    with pytest.raises(RuntimeError, match="fixture_stdout_missing"):
        exp._readline_bounded(NoStdout())  # type: ignore[arg-type]

    class EmptySelector:
        def register(self, *_args: object) -> None:
            return None

        def select(self, _timeout: float) -> list:
            return []

        def close(self) -> None:
            return None

    sleeping = exp._start_worker([sys.executable, "-c", "import time; time.sleep(5)"])
    monkeypatch.setattr(exp.selectors, "DefaultSelector", EmptySelector)
    with pytest.raises(TimeoutError, match="fixture_first_line_timeout"):
        exp._readline_bounded(sleeping)
    monkeypatch.undo()

    silent = exp._start_worker([sys.executable, "-c", ""])
    with pytest.raises(RuntimeError, match="fixture_first_line_missing"):
        exp._readline_bounded(silent)
    silent.wait(timeout=exp.FIXTURE_TIMEOUT_S)

    class SlowProcess:
        returncode = None

        def __init__(self) -> None:
            self.calls = 0
            self.killed = False

        def communicate(self, timeout: float) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["fixture"], timeout)
            self.returncode = -9
            return "", "timed out"

        def kill(self) -> None:
            self.killed = True

    slow = SlowProcess()
    with pytest.raises(TimeoutError, match="fixture_completion_timeout"):
        exp._finish_worker(slow)  # type: ignore[arg-type]
    assert slow.killed is True

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing tool")),
    )
    diagnostic = exp._command_diagnostic(["missing-tool"])
    assert diagnostic["exit_code"] is None
    assert "missing tool" in diagnostic["error"]

    accepted = exp._attack_row("accepted-probe", lambda: None)
    assert accepted["fail_closed"] is False
    row = exp._rejected_transition(
        tmp_path,
        case="accepted-transition",
        prepare=lambda _lease: None,
        target="admitted",
    )
    assert row["accepted"] is True


def test_scenario_report_6633_validator_rejects_every_schema_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6633-BLOCKED exercises every validator rejection."""

    ready = _ready_artifact(tmp_path)
    assert exp.validate_artifact(ready) == []

    def errors_for(**changes: object) -> list[str]:
        bad = deepcopy(ready)
        bad.update(changes)
        bad["reproducibility_checksum"] = exp.payload_checksum(bad)
        return exp.validate_artifact(bad)

    bad = deepcopy(ready)
    bad.pop("duration_s")
    bad["reproducibility_checksum"] = exp.payload_checksum(bad)
    assert "required_fields_mismatch" in exp.validate_artifact(bad)
    assert "ready_status_mismatch" in errors_for(status="wrong")
    assert "ready_verdict_class_mismatch" in errors_for(verdict_class="blocked")
    assert "ready_gate_summary_not_empty" in errors_for(gate_check_summary=[{"bad": True}])
    assert "inference_substrate_mismatch" in errors_for(inference_substrate="wrong")
    assert "verifier_is_oracle_mismatch" in errors_for(verifier_is_oracle=False)

    invalid_provenance = deepcopy(ready["field_provenance"])
    invalid_provenance["status"] = {}
    assert "field_provenance_invalid:status" in errors_for(field_provenance=invalid_provenance)

    blocked = exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=1.0,
        tests_run=[],
        process_fixtures={"rows": [], "all_passed": False},
        phase_transition_rows=[],
        attack_rows=[],
        protected_before=exp.protected_hashes(REPO),
    )

    def blocked_errors(**changes: object) -> list[str]:
        bad = deepcopy(blocked)
        bad.update(changes)
        bad["reproducibility_checksum"] = exp.payload_checksum(bad)
        return exp.validate_artifact(bad)

    assert "blocked_status_mismatch" in blocked_errors(status="wrong")
    assert "blocked_verdict_mismatch" in blocked_errors(honest_verdict="wrong")
    assert "blocked_verdict_class_mismatch" in blocked_errors(verdict_class=None)
    assert "blocked_gate_summary_mismatch" in blocked_errors(gate_check_summary=[])


def test_req_report_6633_run_refuses_invalid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6633: the writer refuses an artifact rejected by its oracle."""

    monkeypatch.setattr(exp, "collect_preconditions", lambda *_args: {})
    monkeypatch.setattr(
        exp,
        "run_process_fixtures",
        lambda *_args: {"rows": [], "all_passed": False},
    )
    monkeypatch.setattr(exp, "build_phase_transition_rows", lambda *_args: [])
    monkeypatch.setattr(exp, "build_attack_rows", lambda *_args: [])
    monkeypatch.setattr(exp, "build_lease_api_receipts", lambda *_args: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_invalid"])
    with pytest.raises(ValueError, match="forced_invalid"):
        exp.run(
            date="20260826",
            root=REPO,
            result_path=tmp_path / "must-not-exist.json",
            work_dir=tmp_path / "work",
            tests_run=_tests_run(),
        )
