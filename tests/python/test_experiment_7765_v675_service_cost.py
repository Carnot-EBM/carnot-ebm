"""REQ-REPORT-7765: blocked timing custody and response-level decisions."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.experiment_7765_v675_service_cost import (
    aggregate_risk,
    build_candidate,
    cold_reduce,
    main,
    score_rows,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """Give every replay private files with explicit byte custody."""
    _json(
        tmp_path / "results/experiment_7757_view_energy_fit.json",
        {
            "schema": "blocked_gate_check_v1",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "failed_field": "sentence_protocol_ready_score",
        },
    )
    refs = {}
    for name in ("KV260", "PolarFire", "GateMate"):
        rel = f"results/{name.lower()}.json"
        _json(tmp_path / rel, {"board": name})
        refs[rel] = {"sha256": sha256_file(tmp_path / rel), "scope": "historical_board_evidence"}
    hardware = {
        name: {
            "evidence_date": "20260913",
            "evidence_hash": refs[f"results/{name.lower()}.json"]["sha256"],
            "evidence_path": f"results/{name.lower()}.json",
            "venue": "kv260_fpga_fabric_historical" if name == "KV260" else "none_read_only",
            "supported_workload": "quadratic k_max<=5" if name == "KV260" else "historical only",
            "blocker": "0xffffffff" if name == "GateMate" else None,
            "changed_prerequisite": "new authenticated complete-service receipt",
        }
        for name in ("KV260", "PolarFire", "GateMate")
    }
    hardware["Extropic Z1T/TSU"] = {
        "evidence_date": "20260904",
        "venue": "none",
        "blocker": "no access",
        "changed_prerequisite": "SDK and full service run",
        "supported_workload": "projection",
        "evidence_hash": None,
        "evidence_path": None,
    }
    hardware["AMD XDNA NPU"] = dict(hardware["Extropic Z1T/TSU"])
    _json(
        tmp_path / "results/experiment_7751_v674_hardware_continuity.json",
        {
            "experiment_id": 7751,
            "run_date": "20260927",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "hardware_continuity_complete_score": 1,
            "hardware_continuity": hardware,
            "source_artifact_hashes": refs,
        },
    )
    return tmp_path


def test_response_risk_and_shared_invalid_policy() -> None:
    """SCENARIO-REPORT-7765-DECISION: average risks before one temperature."""
    assert aggregate_risk("paired_energy", [0.1, 0.9], 2) == pytest.approx(0.5)
    assert aggregate_risk("canonical_energy", [0.1, 0.9], 2) == pytest.approx(0.25)
    assert aggregate_risk("paired_energy", [0.1, 0.9], 1) == pytest.approx(0.5)
    assert aggregate_risk(
        "local_logistic", [[0.0, 2.0], [2.0, 0.0]], 1, logistic_head=lambda x: x[0] / 4
    ) == pytest.approx(0.25)
    rows = score_rows(
        [
            {
                "family": "a",
                "label": 1,
                "risks": [0.1, 0.9],
                "arm": "paired_energy",
                "invalid": False,
            },
            {"family": "b", "label": 0, "risks": [], "arm": "paired_energy", "invalid": True},
        ],
        2,
    )
    assert rows[1]["risk"] == 0.5
    assert rows[1]["decision"] == "escalate"
    assert rows[1]["brier"] == rows[1]["realized_cost"] == 0.25
    assert rows[0]["risk"] == 0.5
    with pytest.raises(ValueError, match="temperature"):
        aggregate_risk("paired_energy", [0.1, 0.9], 0)
    with pytest.raises(ValueError, match="logistic head"):
        aggregate_risk("local_logistic", [[0.1], [0.9]], 1)
    with pytest.raises(ValueError, match="risk must"):
        aggregate_risk("canonical_energy", [1.0], 1)


def test_blocked_custody_and_cold_replay(inputs: Path) -> None:
    """SCENARIO-REPORT-7765-BLOCKED/REPLAY: receipt cannot stand in for heads."""
    candidate = build_candidate(inputs, "20260927")
    assert candidate["honest_verdict"].startswith("complete_blocked_")
    assert candidate["service_evidence_ready_score"] == 0
    assert candidate["service_cost_rows"] == []
    assert (
        "heads"
        in candidate["source_artifact_hashes"]["results/experiment_7757_view_energy_fit.json"][
            "imported_fields"
        ]
    )
    assert any(r["field"] == "fitted_heads" for r in candidate["gate_check_summary"])
    assert any(
        r["field"] == "artifact_exists" and r["passed"]
        for r in candidate["preconditions_checked"]["checks"]
    )
    assert (
        candidate["cost_input_eligibility"]["results/experiment_7761_v675_online_state.json"][
            "reason"
        ]
        == "optional_absent"
    )
    assert candidate["hardware_continuity"]["GateMate"]["blocker"] == "0xffffffff"
    assert candidate["hardware_continuity"]["KV260"]["venue"] == "kv260_fpga_fabric_historical"
    assert candidate["hardware_continuity"]["KV260"]["authenticated"] is True
    assert candidate["hardware_continuity"]["Extropic Z1T/TSU"]["authenticated"] is False
    assert all(not row["started"] for row in candidate["rows"] if row["unit_type"] == "timing")
    output = inputs / "candidate.json"
    _json(output, candidate)
    assert cold_reduce(output, inputs)["row_count"] == len(candidate["rows"])
    altered = dict(candidate)
    altered["rows_checksum"] = "tampered"
    _json(output, altered)
    with pytest.raises(ValueError, match="summary_mismatch"):
        cold_reduce(output, inputs)
    _json(output, candidate)
    (inputs / "results/kv260.json").write_text("tampered")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(output, inputs)


def test_entrypoint_and_real_child(inputs: Path) -> None:
    """SCENARIO-REPORT-7765-REPLAY: host request and cold process agree."""
    output = inputs / "result.json"
    raw = inputs / "raw.json"
    assert (
        main(
            [
                "--root",
                str(inputs),
                "--date",
                "20260927",
                "--output",
                str(output),
                "--raw",
                str(raw),
            ]
        )
        == 0
    )
    original_raw = raw.read_text()
    raw.write_text("{}")
    with pytest.raises(ValueError, match="raw_rows_mismatch"):
        cold_reduce(output, inputs, raw)
    raw.write_text(original_raw)
    assert main(["--root", str(inputs), "--cold-reduce", str(output)]) == 0
    child = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "scripts/experiments/experiment_7765_v675_service_cost.py"
            ),
            "--root",
            str(inputs),
            "--cold-reduce",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr
    assert "row_count" in child.stdout
