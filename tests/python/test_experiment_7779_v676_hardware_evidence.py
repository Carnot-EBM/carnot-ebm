"""REQ-REPORT-7779: historical bytes, board limits, and terminal accounting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import pytest

from carnot.reporting.experiment_7779_v676_hardware_evidence import (
    build_audit,
    cold_reduce,
    main,
    resolve_historical_blob,
)


def _write(root: Path, name: str, value: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    return path


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    """SCENARIO-REPORT-7779-CUSTODY: private historical Git and board files."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "config", "user.name", "Fixture")
    historical = _write(tmp_path, "research-references.md", "historical NPU TSU Kona\n")
    expected = _hash(historical)
    _git(tmp_path, "add", "research-references.md")
    _git(tmp_path, "commit", "-qm", "historical reference")
    _write(tmp_path, "research-references.md", "historical NPU TSU Kona\nnew paper\n")
    _write(tmp_path, "research-hardware-wishlist.md", "board wishlist\n")
    boards = {}
    for name, venue in (
        ("KV260", "kv260_fpga_fabric_historical"),
        ("PolarFire", "polarfire_linux_cpu_historical"),
        ("GateMate", "none_read_only"),
    ):
        path = f"results/{name.lower()}.json"
        receipt = _write(tmp_path, path, {"board": name, "venue": venue})
        boards[name] = {
            "evidence_date": "20260913",
            "evidence_hash": _hash(receipt),
            "venue": venue,
            "supported_workload": name,
            "blocker": "0xffffffff" if name == "GateMate" else None,
            "changed_prerequisite": "physical change" if name == "GateMate" else "new workload",
            "bounded_next_experiment": "physical change" if name == "GateMate" else "new workload",
        }
    for name in ("AMD XDNA NPU", "Extropic Z1T/TSU", "Kona"):
        boards[name] = {
            "evidence_date": "20260927",
            "evidence_hash": expected,
            "venue": "none",
            "supported_workload": "disclosure only",
            "blocker": "no local execution",
            "changed_prerequisite": "local evidence",
            "bounded_next_experiment": "local evidence",
        }
    sources = {
        f"results/{name.lower()}.json": {"sha256": boards[name]["evidence_hash"]}
        for name in ("KV260", "PolarFire", "GateMate")
    }
    _write(
        tmp_path,
        "results/experiment_7751_v674_hardware_continuity.json",
        {
            "hardware_continuity": boards,
            "source_artifact_hashes": sources,
            "hardware_continuity_complete_score": 1,
        },
    )
    _write(
        tmp_path,
        "results/experiment_7765_v675_service_cost.json",
        {"service_evidence_ready_score": 0},
    )
    return tmp_path


def test_historical_blob_survives_current_append(evidence: Path) -> None:
    """SCENARIO-REPORT-7779-CUSTODY: original digest resolves through Git."""
    audit = build_audit(evidence, "20260927", evidence / "snapshots")
    assert audit["immutable_evidence_ready_score"] == 1
    assert audit["hardware_continuity_rows"][0]["scope"].find("k_max<=5") >= 0
    assert audit["hardware_continuity"]["PolarFire"]["venue"] == "polarfire_linux_cpu_historical"
    assert audit["hardware_continuity"]["GateMate"]["blocker"] == "0xffffffff"
    assert audit["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert audit["acceptance_gate_results"]["decision_benefit"] is None
    assert audit["acceptance_gate_results"]["readiness"] == 0
    assert all(row["source_hash"] for row in audit["rows"])
    assert audit["source_artifact_hashes"]["research-references.md"]["sha256"] == _hash(
        evidence / "research-references.md"
    )
    assert (evidence / "snapshots/research-references.md").is_file()


def test_unresolved_blob_is_excluded(evidence: Path) -> None:
    """SCENARIO-REPORT-7779-CUSTODY: missing original bytes stay unresolved."""
    inventory = evidence / "results/experiment_7751_v674_hardware_continuity.json"
    value = json.loads(inventory.read_text())
    value["hardware_continuity"]["Kona"]["evidence_hash"] = "sha256:" + "0" * 64
    inventory.write_text(json.dumps(value))
    audit = build_audit(evidence, "20260927", evidence / "snapshots")
    assert audit["immutable_evidence_ready_score"] == 0
    assert audit["hardware_continuity_rows"][-1]["state"] == "evidence_unresolved"
    assert any(c["upstream_id"] == "Exp7751:Kona" for c in audit["gate_check_summary"])
    assert resolve_historical_blob(evidence, "sha256:" + "0" * 64) is None


def test_board_hash_and_cold_replay(evidence: Path) -> None:
    """SCENARIO-REPORT-7779-TERMINAL: raw bytes govern board rows."""
    audit = build_audit(evidence, "20260927", evidence / "snapshots")
    candidate = _write(evidence, "candidate.json", audit)
    raw = _write(evidence, "raw.json", audit["rows"])
    assert cold_reduce(candidate, raw, evidence)["row_count"] == 6
    _write(evidence, "results/kv260.json", "tampered")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(candidate, raw, evidence)


def test_replay_rejects_row_change(evidence: Path) -> None:
    """SCENARIO-REPORT-7779-TERMINAL: a changed row cannot pass reduction."""
    audit = build_audit(evidence, "20260927", evidence / "snapshots")
    candidate = _write(evidence, "candidate.json", audit)
    rows = audit["rows"]
    rows[0]["state"] = "positive"
    raw = _write(evidence, "raw.json", rows)
    with pytest.raises(ValueError, match="row_mismatch"):
        cold_reduce(candidate, raw, evidence)


def test_replay_rejects_blob_and_snapshot_change(evidence: Path) -> None:
    """SCENARIO-REPORT-7779-TERMINAL: Git and snapshot identities are binding."""
    audit = build_audit(evidence, "20260927", evidence / "snapshots")
    raw = _write(evidence, "raw.json", audit["rows"])
    git_key = next(key for key in audit["source_artifact_hashes"] if key.startswith("git:"))
    audit["source_artifact_hashes"][git_key]["git_blob"] = "invalid_blob"
    candidate = _write(evidence, "candidate.json", audit)
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(candidate, raw, evidence)
    audit["source_artifact_hashes"][git_key]["git_blob"] = resolve_historical_blob(
        evidence, audit["source_artifact_hashes"][git_key]["sha256"]
    )["blob"]
    _write(evidence, "candidate.json", audit)
    _write(evidence, "snapshots/research-references.md", "changed")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(candidate, raw, evidence)


def test_replay_rejects_summary_and_missing_git(evidence: Path) -> None:
    """SCENARIO-REPORT-7779-TERMINAL: derived views cannot drift from rows."""
    with tempfile.TemporaryDirectory() as empty:
        assert resolve_historical_blob(Path(empty), "sha256:" + "0" * 64) is None
    audit = build_audit(evidence, "20260927", evidence / "snapshots")
    raw = _write(evidence, "raw.json", audit["rows"])
    audit["hardware_continuity_rows"] = []
    candidate = _write(evidence, "candidate.json", audit)
    with pytest.raises(ValueError, match="row_mismatch"):
        cold_reduce(candidate, raw, evidence)
    audit["hardware_continuity_rows"] = audit["rows"]
    audit["hardware_continuity"]["KV260"]["state"] = "invented"
    _write(evidence, "candidate.json", audit)
    with pytest.raises(ValueError, match="row_mismatch"):
        cold_reduce(candidate, raw, evidence)


def test_cli_prepare_replay_and_validation_gate(evidence: Path) -> None:
    """SCENARIO-REPORT-7779-TERMINAL: real CLI paths and failed checks."""
    args = ["--root", str(evidence), "--date", "20260927"]
    assert main([*args, "--prepare"]) == 0
    raw_dir = evidence / "results/raw/experiment_7779_v676_hardware_evidence"
    assert main([*args, "--cold-reduce", str(raw_dir / "candidate.json")]) == 0
    with pytest.raises(FileNotFoundError, match="validation receipts required"):
        main(args)
    receipts = {
        "checks": [],
        "required_checks_passed": True,
        "terminal_reader": {"flagged_adversarial": False},
        "phase_spans": [],
        "run_started_monotonic_s": time.monotonic(),
    }
    _write(
        evidence,
        "results/raw/experiment_7779_v676_hardware_evidence/validation_receipts.json",
        receipts,
    )
    assert main(args) == 0
    final = evidence / "results/experiment_7779_v676_hardware_evidence.json"
    assert (
        json.loads(final.read_text())["honest_verdict"]
        == "complete_blocked_missing_service_evidence"
    )
    receipts["checks"] = [
        {
            "name": "full_python_suite",
            "exit_code": 2,
            "log_path": "/tmp/full.log",
            "log_sha256": "sha256:example",
        }
    ]
    receipts["required_checks_passed"] = False
    _write(
        evidence,
        "results/raw/experiment_7779_v676_hardware_evidence/validation_receipts.json",
        receipts,
    )
    assert main(args) == 0
    final_value = json.loads(final.read_text())
    assert final_value["honest_verdict"] == "complete_disqualified_required_checks"
    assert final_value["acceptance_gate_results"]["readiness"] == 0
    assert any(
        check["upstream_id"] == "Exp7779:validation" for check in final_value["gate_check_summary"]
    )
