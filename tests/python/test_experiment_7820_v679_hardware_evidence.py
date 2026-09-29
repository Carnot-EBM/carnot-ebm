"""REQ-REPORT-7820: attached-board custody and immutable validation receipts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.experiment_7820_v679_hardware_evidence import (
    build_audit,
    cold_reduce,
    dispatch,
    main,
    run_child,
)
from carnot.reporting.experiment_7806_v678_hardware_evidence import build_audit as prior_audit


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "results/raw/experiment_7820_v679_hardware_evidence/validation_command_manifest.json"
)


def test_missing_service_keeps_three_authentic_board_rows() -> None:
    """SCENARIO-REPORT-7820-CUSTODY: old board truth does not open service."""
    audit = build_audit(ROOT, "20260928", MANIFEST)
    assert audit["experiment_id"] == 7820
    assert audit["milestone"] == "2026.09.679"
    assert audit["verdict_class"] == "blocked"
    assert audit["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert audit["service_source_status"] == "missing"
    assert audit["gate_check_summary"] == [
        {
            "upstream_id": "Exp7819",
            "artifact_path": "results/experiment_7819_v679_service_cost.json",
            "artifact_hash": None,
            "field": "service_evidence_ready_score",
            "operator": "==",
            "expected": 1,
            "observed": None,
            "passed": False,
        }
    ]
    boards = {row["board"]: row for row in audit["board_rows"]}
    assert set(boards) == {"KV260", "PolarFire", "GateMate"}
    assert boards["KV260"]["processor_class"] == "fpga_fabric"
    assert boards["KV260"]["k_max"] == 5
    assert boards["PolarFire"]["processor_class"] == "linux_cpu"
    assert boards["GateMate"]["blocker"] == "0xffffffff"
    assert "Dated operator" in boards["GateMate"]["next_missing_prerequisite"]
    assert audit["acquisition_recommendation"]["decision"] == "defer"
    assert audit["preconditions_checked"]["board_operations_issued"] == []
    assert audit["acceptance_gate_results"]["efficiency"] is None


def test_current_service_fraction_requires_qualified_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7820-SERVICE: only a qualified science producer supplies time."""
    old = prior_audit(ROOT, "20260928")
    monkeypatch.setattr(
        "carnot.reporting.experiment_7820_v679_hardware_evidence.prior_build_audit",
        lambda root, date: deepcopy(old),
    )
    service = tmp_path / "results/experiment_7819_v679_service_cost.json"
    service.parent.mkdir(parents=True)
    data = {
        "experiment_id": 7819,
        "schema": "carnot.experiment_7819.v1",
        "run_date": "20260928",
        "service_evidence_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "stage_times_ms": {"host_stage": 2, "whole_service": 10},
    }
    service.write_text(json.dumps(data))
    audit = build_audit(tmp_path, "20260928", MANIFEST)
    assert audit["service_source_status"] == "qualified"
    assert audit["host_stage_fraction"] == pytest.approx(0.2)
    assert audit["acceleration_bound"] == pytest.approx(1.25)
    assert audit["stage_opportunity_map"]["transfer_ms"] is None
    data["flagged_adversarial"] = True
    service.write_text(json.dumps(data))
    audit = build_audit(tmp_path, "20260928", MANIFEST)
    assert audit["service_source_status"] == "disqualified"
    assert audit["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert audit["host_stage_fraction"] is None


def test_missing_parent_reproduced_then_repaired_with_sealed_retry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7820-PARENT: pytest needs its nested basetemp parent."""
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_ok(tmp_path):\n    assert tmp_path.is_dir()\n")
    missing = tmp_path / "absent/parent/basetemp"
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        f"--basetemp={missing}",
        str(test_file),
        "-q",
    ]
    old = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    assert old.returncode != 0
    assert "FileNotFoundError" in old.stdout + old.stderr
    spec = {
        "name": "missing_parent",
        "argv": argv,
        "classification": "required",
        "private_root": str(tmp_path / "attempt1"),
        "timeout_s": 30,
    }
    first = run_child(ROOT, spec, tmp_path / "durable")
    assert first["passed"]
    assert Path(first["log_path"]).is_file()
    spec["private_root"] = str(tmp_path / "attempt2")
    second = run_child(ROOT, spec, tmp_path / "durable")
    assert second["passed"]
    assert first["log_path"] != second["log_path"]
    assert (
        Path(first["log_path"]).read_bytes() == first["log_bytes"] if "log_bytes" in first else True
    )


def test_dispatch_manifest_is_complete_and_rejects_undeclared_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7820-DISPATCH: recording executor sees every exact argv."""
    manifest = json.loads(MANIFEST.read_text())
    seen = []

    def record(root: Path, spec: dict, raw: Path) -> dict:
        seen.append((spec["name"], spec["argv"], spec["classification"]))
        return {
            "name": spec["name"],
            "command_argv": spec["argv"],
            "classification": spec["classification"],
            "exit_code": 0,
            "passed": True,
            "log_path": "recorded",
            "log_sha256": "sha256:recorded",
        }

    receipts = dispatch(ROOT, MANIFEST, tmp_path, executor=record)
    assert seen == [(c["name"], c["argv"], c["classification"]) for c in manifest["commands"]]
    assert len(receipts) == len(manifest["commands"])
    assert [c["name"] for c in manifest["commands"]][-3:] == [
        "fresh_process_cold_replay",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    changed = deepcopy(manifest)
    changed["commands"].append(
        {
            "name": "undeclared",
            "argv": [sys.executable, "-c", "pass"],
            "classification": "required",
            "private_root": str(tmp_path / "bad"),
            "timeout_s": 1,
        }
    )
    private = tmp_path / "changed.json"
    private.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="undeclared"):
        dispatch(ROOT, private, tmp_path, executor=record)


def test_real_cli_dispatch_records_manifest_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7820-DISPATCH: the CLI records all required and diagnostic exits."""
    manifest = json.loads(MANIFEST.read_text())
    manifest["run_root"] = str(tmp_path / "private")
    manifest["candidate_path"] = str(tmp_path / "private/candidate.json")
    for index, command in enumerate(manifest["commands"]):
        command["private_root"] = str(tmp_path / f"private/command_{index:02d}")
        command["argv"] = [
            arg.replace(json.loads(MANIFEST.read_text())["run_root"], manifest["run_root"])
            for arg in command["argv"]
        ]
    private_manifest = tmp_path / "manifest.json"
    private_manifest.write_text(json.dumps(manifest))
    original = prior_audit(ROOT, "20260928")
    monkeypatch.setattr(
        "carnot.reporting.experiment_7820_v679_hardware_evidence.prior_build_audit",
        lambda root, date: deepcopy(original),
    )
    seen = []

    def record(root: Path, spec: dict, raw: Path) -> dict:
        seen.append((spec["name"], spec["argv"], spec["classification"]))
        failed = spec["name"] == "repository_health"
        return {
            "name": spec["name"],
            "command_argv": spec["argv"],
            "classification": spec["classification"],
            "exit_code": 1 if failed else 0,
            "passed": not failed,
            "log_path": "recorded",
            "log_sha256": "sha256:recorded",
        }

    output = tmp_path / "result.json"
    assert (
        main(
            ["--date", "20260928", "--manifest", str(private_manifest), "--output", str(output)],
            executor=record,
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert seen == [(c["name"], c["argv"], c["classification"]) for c in manifest["commands"]]
    assert result["observed_child_commands"] == manifest["commands"]
    assert result["repository_health"]["exit_code"] == 1
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert result["validation_receipts"]["required_checks_passed"] is True


def test_cold_replay_rejects_changed_row_venue_and_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7820-CUSTODY: row, venue, and log bytes are sealed."""
    original = prior_audit(ROOT, "20260928")
    monkeypatch.setattr(
        "carnot.reporting.experiment_7820_v679_hardware_evidence.prior_build_audit",
        lambda root, date: deepcopy(original),
    )
    audit = build_audit(ROOT, "20260928", MANIFEST)
    log = tmp_path / "log"
    log.write_bytes(b"exact\n")
    from carnot.reporting.current_work_receipt import sha256_file

    audit["validation_receipts"] = {
        "checks": [
            {
                "name": "probe",
                "log_path": str(log),
                "log_sha256": sha256_file(log),
                "passed": True,
                "exit_code": 0,
            }
        ]
    }
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(audit))
    assert cold_reduce(candidate, ROOT, MANIFEST)["row_count"] == 3
    changed = deepcopy(audit)
    changed["board_rows"][1]["processor_class"] = "fpga_fabric"
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="board_row_mismatch"):
        cold_reduce(candidate, ROOT, MANIFEST)
    candidate.write_text(json.dumps(audit))
    log.write_bytes(b"Exact\n")
    with pytest.raises(ValueError, match="log_hash_mismatch"):
        cold_reduce(candidate, ROOT, MANIFEST)


def test_manifest_rejects_wrong_class_and_executor_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7820-DISPATCH: neither class nor returned argv can drift."""
    manifest = json.loads(MANIFEST.read_text())
    manifest["commands"][0]["classification"] = "diagnostic"
    path = tmp_path / "wrong_class.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="undeclared"):
        dispatch(ROOT, path, tmp_path, executor=lambda root, spec, raw: {})

    def lie(root: Path, spec: dict, raw: Path) -> dict:
        return {
            "name": spec["name"],
            "command_argv": ["changed"],
            "classification": spec["classification"],
        }

    with pytest.raises(ValueError, match="undeclared child receipt"):
        dispatch(ROOT, MANIFEST, tmp_path, executor=lie)


def test_timeout_and_sealed_log_rejection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7820-PARENT: only the owned child dies at its timeout."""
    import carnot.reporting.experiment_7820_v679_hardware_evidence as module

    class Child:
        returncode = -9
        calls = 0

        def wait(self, timeout: float | None = None) -> int:
            self.calls += 1
            if self.calls <= 2:
                raise subprocess.TimeoutExpired("owned", timeout)
            return self.returncode

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

    events: list[str] = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: Child())
    spec = {
        "name": "timeout",
        "argv": [sys.executable, "-c", "pass"],
        "classification": "required",
        "private_root": str(tmp_path / "attempt/command"),
        "timeout_s": 0,
    }
    receipt = run_child(ROOT, spec, tmp_path / "durable")
    assert events == ["terminate", "kill"]
    assert receipt["timed_out"] is True
    assert receipt["passed"] is False
    from carnot.reporting.current_work_receipt import sha256_file

    spec["private_root"] = str(tmp_path / "retry/command")
    collision = (
        tmp_path
        / "durable/retry/command/timeout"
        / (sha256_file(Path(receipt["log_path"])).split(":", 1)[1] + ".log")
    )
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"")
    with pytest.raises(FileExistsError, match="sealed log"):
        run_child(ROOT, spec, tmp_path / "durable")


def test_cold_replay_rejects_source_rows_gate_summary_and_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7820-CUSTODY: every asserted summary reopens its source."""
    from carnot.reporting.current_work_receipt import sha256_file

    original = prior_audit(ROOT, "20260928")
    monkeypatch.setattr(
        "carnot.reporting.experiment_7820_v679_hardware_evidence.prior_build_audit",
        lambda root, date: deepcopy(original),
    )
    audit = build_audit(ROOT, "20260928", MANIFEST)
    candidate = tmp_path / "candidate.json"
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(audit["rows"]))
    audit["raw_rows_path"] = str(raw)
    audit["raw_rows_sha256"] = sha256_file(raw)
    candidate.write_text(json.dumps(audit))
    assert cold_reduce(candidate, ROOT, MANIFEST)["row_count"] == 3
    cases = [
        ("source_artifact_hashes", "source_hash_mismatch"),
        ("rows", "row_mismatch"),
        ("gate_check_summary", "gate_mismatch"),
        ("host_stage_fraction", "summary_mismatch"),
    ]
    for field, error in cases:
        changed = deepcopy(audit)
        if field == "source_artifact_hashes":
            changed[field]["results/experiment_7819_v679_service_cost.json"]["eligible"] = True
        elif field == "rows":
            changed[field][3]["venue"] = "wrong"
        elif field == "gate_check_summary":
            changed[field] = []
        else:
            changed[field] = 0.8
        candidate.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=error):
            cold_reduce(candidate, ROOT, MANIFEST)
    candidate.write_text(json.dumps(audit))
    raw.write_text("[]")
    with pytest.raises(ValueError, match="raw_hash_mismatch"):
        cold_reduce(candidate, ROOT, MANIFEST)
    audit["raw_rows_sha256"] = sha256_file(raw)
    candidate.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="raw_row_mismatch"):
        cold_reduce(candidate, ROOT, MANIFEST)


def test_copy_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7820-PARENT: the sealed bytes must equal the closed live log."""
    import carnot.reporting.experiment_7820_v679_hardware_evidence as module

    def wrong_copy(source: Path, target: Path) -> None:
        target.write_bytes(b"one-byte-mutation")

    monkeypatch.setattr(module.shutil, "copyfile", wrong_copy)
    spec = {
        "name": "copy",
        "argv": [sys.executable, "-c", "print('original')"],
        "classification": "required",
        "private_root": str(tmp_path / "attempt/copy"),
        "timeout_s": 5,
    }
    with pytest.raises(ValueError, match="sealed log copy"):
        run_child(ROOT, spec, tmp_path / "durable")


def test_actual_dispatch_and_cli_modes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7820-DISPATCH: real child exits seal and replay."""
    from carnot.reporting.current_work_receipt import sha256_file

    original = prior_audit(ROOT, "20260928")
    monkeypatch.setattr(
        "carnot.reporting.experiment_7820_v679_hardware_evidence.prior_build_audit",
        lambda root, date: deepcopy(original),
    )
    manifest = json.loads(MANIFEST.read_text())
    manifest["run_root"] = str(tmp_path / "run")
    manifest["candidate_path"] = str(tmp_path / "run/candidate.json")
    for index, command in enumerate(manifest["commands"]):
        command["private_root"] = str(tmp_path / f"run/command_{index:02d}_{command['name']}")
        command["argv"] = [sys.executable, "-c", "print('exit')"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    prepared = tmp_path / "prepared.json"
    args = ["--root", str(tmp_path), "--manifest", str(path), "--date", "20260928"]
    assert main([*args, "--prepare", str(prepared)]) == 0
    assert prepared.is_file()
    output = tmp_path / "result.json"
    assert main([*args, "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["validation_receipts"]["required_checks_passed"] is True
    assert result["raw_rows_sha256"] == sha256_file(Path(result["raw_rows_path"]))
    assert all(
        sha256_file(Path(r["log_path"])) == r["log_sha256"]
        for r in result["validation_receipts"]["checks"]
    )
    assert main([*args, "--cold-replay", str(output)]) == 0
    assert Path(result["raw_rows_path"]).is_file()
    Path(result["raw_rows_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable raw row path"):
        main([*args, "--output", str(output)])


def test_required_failure_disqualifies_and_script_is_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7820-DISPATCH: failed required work zeros every gate."""
    import runpy

    old = prior_audit(ROOT, "20260928")
    monkeypatch.setattr(
        "carnot.reporting.experiment_7820_v679_hardware_evidence.prior_build_audit",
        lambda root, date: deepcopy(old),
    )
    manifest = json.loads(MANIFEST.read_text())
    manifest["candidate_path"] = str(tmp_path / "candidate.json")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    def fail_one(root: Path, spec: dict, raw: Path) -> dict:
        failed = spec["name"] == "focused_pytest"
        return {
            "name": spec["name"],
            "command_argv": spec["argv"],
            "classification": spec["classification"],
            "exit_code": 2 if failed else 0,
            "passed": not failed,
            "log_path": "recorded",
            "log_sha256": "sha256:recorded",
        }

    output = tmp_path / "result.json"
    assert (
        main(
            ["--root", str(tmp_path), "--manifest", str(path), "--output", str(output)],
            executor=fail_one,
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["verdict_class"] == "disqualified"
    assert set(result["acceptance_gate_results"].values()) == {0}
    assert result["gate_check_summary"][-1]["field"] == "focused_pytest"
    monkeypatch.setattr("carnot.reporting.experiment_7820_v679_hardware_evidence.main", lambda: 0)
    with pytest.raises(SystemExit) as done:
        runpy.run_path(
            str(ROOT / "scripts/experiments/experiment_7820_v679_hardware_evidence.py"),
            run_name="__main__",
        )
    assert done.value.code == 0
