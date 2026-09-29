"""REQ-REPORT-7806: attached-board evidence and whole-service gate."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import shutil
import subprocess
import time
from copy import deepcopy

import pytest

from carnot.reporting.experiment_7806_v678_hardware_evidence import (
    _validate,
    build_audit,
    cold_reduce,
    main,
)


ROOT = Path(__file__).resolve().parents[2]


def test_authentic_missing_service(capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7806-CUSTODY: authenticated boards do not open service."""
    audit = build_audit(ROOT, "20260928")
    progress_lines = capsys.readouterr().out
    assert progress_lines.index("phase=preconditions event=inputs_complete") < progress_lines.index(
        "[exp7793] phase=preflight"
    )
    assert audit["experiment_id"] == 7806
    assert audit["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert audit["verdict_class"] == "blocked"
    assert audit["service_source_status"] == "missing"
    assert audit["gate_check_summary"] == [
        {
            "upstream_id": "Exp7805",
            "artifact_path": "results/experiment_7805_v678_service_cost.json",
            "artifact_hash": None,
            "field": "service_evidence_ready_score",
            "operator": "==",
            "expected": 1,
            "observed": None,
            "passed": False,
        }
    ]
    rows = {row["board"]: row for row in audit["board_rows"]}
    assert set(rows) == {"KV260", "PolarFire", "GateMate"}
    assert rows["KV260"]["k_max"] == 5
    assert (
        rows["KV260"]["source_hash"]
        == "sha256:acdfd841c75649279515fab6b91115d07587094b8703cc72a04febae123236d2"
    )
    assert rows["KV260"]["processor_class"] == "fpga_fabric"
    assert (
        rows["PolarFire"]["source_hash"]
        == "sha256:341ca079f8ca42ed26dcda1d57edba21cede9c0ca965265fed05121feec2b588"
    )
    assert rows["PolarFire"]["processor_class"] == "linux_cpu"
    assert (
        rows["GateMate"]["source_hash"]
        == "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66"
    )
    assert rows["GateMate"]["blocker"] == "0xffffffff"
    assert "Dated operator" in rows["GateMate"]["next_missing_prerequisite"]
    assert all(row["service_fraction"] is None for row in audit["board_rows"])
    assert audit["acceptance_gate_results"]["efficiency"] is None
    assert audit["acquisition_recommendation"]["decision"] == "defer"
    assert audit["preconditions_checked"]["board_operations_issued"] == []
    assert all(check["passed"] for check in audit["preconditions_checked"]["schema_checks"])
    assert len(audit["preconditions_checked"]["immutable_board_hash_checks"]) == 3
    assert all(
        check["passed"] for check in audit["preconditions_checked"]["immutable_board_hash_checks"]
    )
    assert audit["claim_scope"]["source_family_exposure"] == "640 exposed development families"
    assert audit["model_invocation_counts"]["loads"] == 0
    assert (
        audit["source_artifact_hashes"]["results/experiment_7793_v677_hardware_evidence.json"][
            "eligible"
        ]
        is False
    )


def test_private_wrong_venue_and_digest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7806-CUSTODY: private source changes fail closed."""
    original = build_audit(ROOT, "20260928")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(original))
    assert cold_reduce(candidate, ROOT)["row_count"] == 3
    changed = json.loads(candidate.read_text())
    changed["board_rows"][1]["processor_class"] = "fpga_fabric"
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="board_row_mismatch"):
        cold_reduce(candidate, ROOT)
    candidate.write_text(json.dumps(original))
    private = tmp_path / "private"
    for rel, receipt in original["source_artifact_hashes"].items():
        source = ROOT / rel
        if source.is_file() and not rel.startswith("git:"):
            dest = private / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    board = private / "results/experiment_7231_v636_board_continuity.json"
    board.write_bytes(board.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(candidate, private)


def test_private_service_fraction_and_disqualification(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7806-SERVICE: only qualified measured times bound speed."""
    service = tmp_path / "results/experiment_7805_v678_service_cost.json"
    service.parent.mkdir(parents=True)
    service.write_text(
        json.dumps(
            {
                "experiment_id": 7805,
                "run_date": "20260928",
                "service_evidence_ready_score": 1,
                "flagged_adversarial": False,
                "verdict_class": "null",
                "stage_times_ms": {"host_stage": 2.0, "whole_service": 10.0},
            }
        )
    )
    # Reuse only the authenticated historical source bytes.
    baseline = build_audit(ROOT, "20260928")
    for rel in baseline["source_artifact_hashes"]:
        source = ROOT / rel
        if source.is_file():
            dest = tmp_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    for check in baseline["preconditions_checked"]["input_checks"]:
        source = ROOT / check["path"]
        dest = tmp_path / check["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    cli = tmp_path / "scripts/experiments/experiment_7806_v678_hardware_evidence.py"
    cli.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / cli.relative_to(tmp_path), cli)
    blob = "7766c92e07b4cee74ece8d9e0ddfbea14fb69173"
    old_bytes = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", blob])
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    stored = subprocess.run(
        ["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"],
        input=old_bytes,
        capture_output=True,
        check=True,
    )
    assert stored.stdout.decode().strip() == blob
    audit = build_audit(tmp_path, "20260928")
    assert audit["service_source_status"] == "qualified"
    assert audit["honest_verdict"] == "complete_null_hardware_accounting_only"
    assert audit["host_stage_fraction"] == pytest.approx(0.2)
    assert audit["acceleration_bound"] == pytest.approx(1.25)
    assert audit["acquisition_recommendation"]["decision"] == "defer"
    service_data = json.loads(service.read_text())
    service_data["flagged_adversarial"] = True
    service.write_text(json.dumps(service_data))
    audit = build_audit(tmp_path, "20260928")
    assert audit["service_source_status"] == "disqualified"
    assert audit["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert audit["host_stage_fraction"] is None


def test_cli_prepare_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7806-TERMINAL: the task CLI can prepare and replay."""
    candidate = tmp_path / "candidate.json"
    assert main(["--date", "20260928", "--prepare", str(candidate)]) == 0
    assert main(["--cold-replay", str(candidate)]) == 0


def test_cold_replay_rejects_summary_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7806-CUSTODY: every public summary is replay-bound."""
    original = build_audit(ROOT, "20260928")
    path = tmp_path / "candidate.json"
    cases = [
        ("source_artifact_hashes", "source_set_mismatch"),
        ("rows", "row_mismatch"),
        ("gate_check_summary", "gate_mismatch"),
        ("service_source_status", "service_status_mismatch"),
        ("host_stage_fraction", "summary_mismatch"),
        ("acquisition_recommendation", "summary_mismatch"),
        ("reproducibility_checksum", "summary_mismatch"),
    ]
    for field, expected in cases:
        changed = deepcopy(original)
        if field == "source_artifact_hashes":
            changed[field].pop("results/experiment_7793_v677_hardware_evidence.json")
        elif field == "rows":
            changed[field][3]["state"] = "fabric_speedup"
        elif field == "gate_check_summary":
            changed[field] = []
        elif field == "host_stage_fraction":
            changed[field] = 0.5
        elif field == "acquisition_recommendation":
            changed[field]["decision"] = "buy"
        elif field == "reproducibility_checksum":
            changed[field] = "sha256:changed"
        else:
            changed[field] = "qualified"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=expected):
            cold_reduce(path, ROOT)
    changed = deepcopy(original)
    changed["source_artifact_hashes"]["results/experiment_7793_v677_hardware_evidence.json"][
        "eligible"
    ] = True
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(path, ROOT)


def test_frozen_validation_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7806-TERMINAL: required subprocesses use fixed argv."""
    seen: list[str] = []

    def scoped(*args: object, **kwargs: object) -> dict[str, object]:
        seen.append("affected")
        assert kwargs["static_paths"] == [
            "scripts/experiments/experiment_7806_v678_hardware_evidence.py"
        ]
        return {"validation_receipts": [{"name": "affected", "passed": True, "exit_code": 0}]}

    def commands(root: Path, specs: object, **kwargs: object) -> list[dict[str, object]]:
        names = [spec.name for spec in specs]  # type: ignore[attr-defined]
        seen.extend(names)
        assert names == [
            "new_code_coverage",
            "new_code_coverage_report",
            "fresh_process_cold_replay",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        ]
        return [{"name": name, "passed": True, "exit_code": 0} for name in names]

    monkeypatch.setattr(
        "carnot.reporting.experiment_7806_v678_hardware_evidence.run_scoped_validation", scoped
    )
    monkeypatch.setattr(
        "carnot.reporting.experiment_7806_v678_hardware_evidence.run_commands", commands
    )
    assert len(_validate(ROOT, tmp_path / "candidate.json")) == 6
    assert seen[0] == "affected"


def test_terminal_publication_classifies_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7806-TERMINAL: failed checks zero every readiness gate."""
    original = build_audit(ROOT, "20260928")
    monkeypatch.setattr(
        "carnot.reporting.experiment_7806_v678_hardware_evidence.build_audit",
        lambda root, date: deepcopy(original),
    )
    monkeypatch.setenv("CARNOT_7806_CANDIDATE", str(tmp_path / "candidate.json"))
    receipt = {
        "name": "fresh_process_cold_replay",
        "passed": True,
        "exit_code": 0,
        "log_path": "private.log",
        "log_sha256": "sha256:ok",
    }
    monkeypatch.setattr(
        "carnot.reporting.experiment_7806_v678_hardware_evidence._validate",
        lambda root, candidate: (time.sleep(0.01), [receipt])[1],
    )
    assert main(["--root", str(tmp_path), "--date", "20260928"]) == 0
    published = json.loads(
        (tmp_path / "results/experiment_7806_v678_hardware_evidence.json").read_text()
    )
    assert published["validation_receipts"]["required_checks_passed"] is True
    assert published["duration_s"] >= 0.01
    assert published["phase_spans"][-1]["phase"] == "validation_and_publication"
    receipt["passed"] = False
    receipt["exit_code"] = 2
    assert main(["--root", str(tmp_path), "--date", "20260928"]) == 0
    published = json.loads(
        (tmp_path / "results/experiment_7806_v678_hardware_evidence.json").read_text()
    )
    assert published["verdict_class"] == "disqualified"
    assert set(published["acceptance_gate_results"].values()) == {0}
    assert (
        cold_reduce(tmp_path / "results/experiment_7806_v678_hardware_evidence.json", ROOT)[
            "row_count"
        ]
        == 3
    )
    receipt["name"] = "adversarial_verify"
    assert main(["--root", str(tmp_path), "--date", "20260928"]) == 0
    published = json.loads(
        (tmp_path / "results/experiment_7806_v678_hardware_evidence.json").read_text()
    )
    assert published["flagged_adversarial"] is True


def test_script_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7806-TERMINAL: the declared script calls the new main."""
    monkeypatch.setattr("carnot.reporting.experiment_7806_v678_hardware_evidence.main", lambda: 0)
    with pytest.raises(SystemExit) as done:
        runpy.run_path(
            str(ROOT / "scripts/experiments/experiment_7806_v678_hardware_evidence.py"),
            run_name="__main__",
        )
    assert done.value.code == 0
