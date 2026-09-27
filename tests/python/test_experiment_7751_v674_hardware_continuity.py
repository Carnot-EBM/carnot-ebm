"""REQ-REPORT-7751 and REQ-HW-7751: private continuity fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.experiment_7751_v674_hardware_continuity import (
    build_audit,
    cold_reduce,
    default_root,
    main,
)


def _write(root: Path, name: str, value: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else str(value))
    return path


@pytest.fixture
def private_root(tmp_path: Path) -> Path:
    """SCENARIO-REPORT-7751-CUSTODY: construct dated source bytes."""
    for name in (
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "research-hardware-wishlist.md",
        "research-references.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/hardware/spec.md",
    ):
        _write(tmp_path, name, "fixture evidence REQ-REPORT-7751")
    boards = []
    for board, venue in (
        ("KV260", "kv260_fpga_fabric_historical"),
        ("PolarFire", "polarfire_linux_cpu_historical"),
        ("GateMate", "none_read_only"),
    ):
        path = f"results/{board.lower()}_evidence.json"
        source = _write(tmp_path, path, {"board": board, "run_date": "20260913"})
        boards.append(
            {
                "board": board,
                "execution_venue": venue,
                "evidence_artifact_path": path,
                "evidence_artifact_sha256": sha256_file(source),
                "evidence_date": "20260913",
                "k_max": 5 if board == "KV260" else None,
                "last_diagnostic": {"observed": "0xffffffff"} if board == "GateMate" else None,
            }
        )
    _write(
        tmp_path,
        "results/experiment_7599_v663_board_continuity.json",
        {
            "run_date": "20260924",
            "board_rows": boards,
            "honest_verdict": "complete_null_board_continuity_placement_unmeasured",
            "verdict_class": "null",
            "flagged_adversarial": False,
        },
    )
    return tmp_path


def test_boundaries_and_missing_cost(private_root: Path) -> None:
    """SCENARIO-HW-7751-NO-PROBE: no new execution or invented fractions."""
    artifact = build_audit(private_root, "20260927")
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["hardware_continuity_complete_score"] == 1
    assert artifact["cost_input_eligibility"]["eligible"] is False
    assert any(g["upstream_id"] == "Exp7750" for g in artifact["gate_check_summary"])
    assert artifact["hardware_continuity"]["KV260"]["k_max"] == 5
    assert artifact["hardware_continuity"]["PolarFire"]["venue"] == "polarfire_linux_cpu_historical"
    assert artifact["hardware_continuity"]["GateMate"]["blocker"] == "0xffffffff"
    assert artifact["hardware_continuity"]["Extropic Z1T/TSU"]["evidence_date"] == "20260904"
    assert all(row["measured_service_fraction"] is None for row in artifact["rows"])
    assert artifact["acceptance_gate_results"]["decision_benefit"] is None
    assert artifact["model_invoked"] is False


def test_default_root_resolves_repository() -> None:
    """SCENARIO-REPORT-7751-CUSTODY: default paths belong to this repo."""
    assert (default_root() / "AGENTS.md").is_file()
    assert (
        default_root() / "scripts/experiments/experiment_7751_v674_hardware_continuity.py"
    ).is_file()


def test_reject_changed_source(private_root: Path) -> None:
    """SCENARIO-REPORT-7751-REPLAY: a source hash change breaks custody."""
    artifact = build_audit(private_root, "20260927")
    raw = _write(
        private_root,
        "results/raw/experiment_7751_v674_hardware_continuity/rows.json",
        artifact["rows"],
    )
    candidate = _write(private_root, "candidate.json", artifact)
    cold_reduce(candidate, raw, private_root)
    (private_root / "results/kv260_evidence.json").write_text("changed")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(candidate, raw, private_root)


def test_reject_changed_summary(private_root: Path) -> None:
    """SCENARIO-REPORT-7751-REPLAY: raw rows are the summary authority."""
    artifact = build_audit(private_root, "20260927")
    raw = _write(private_root, "raw.json", artifact["rows"])
    artifact["sample_size_budget"]["completed"] += 1
    candidate = _write(private_root, "candidate.json", artifact)
    with pytest.raises(ValueError, match="summary_mismatch"):
        cold_reduce(candidate, raw, private_root)


@pytest.mark.parametrize("change", ["rows", "score", "hardware_hash"])
def test_reject_other_summary_tampering(private_root: Path, change: str) -> None:
    """SCENARIO-REPORT-7751-REPLAY: each published reduction is checked."""
    artifact = build_audit(private_root, "20260927")
    raw = _write(private_root, "raw.json", artifact["rows"])
    if change == "rows":
        artifact["rows"][0]["numerator"] = 999
    elif change == "score":
        artifact["hardware_continuity_complete_score"] = 0
    else:
        artifact["hardware_continuity"]["KV260"]["evidence_hash"] = "sha256:tampered"
    candidate = _write(private_root, "candidate.json", artifact)
    with pytest.raises(ValueError, match="summary_mismatch"):
        cold_reduce(candidate, raw, private_root)


def test_private_cli_cold_process(private_root: Path) -> None:
    """SCENARIO-REPORT-7751-TERMINAL: CLI and fresh reader use private bytes."""
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_7751_v674_hardware_continuity.py"
    )
    output = private_root / "candidate.json"
    env = {
        **os.environ,
        "PYTHONPATH": f"{Path(__file__).resolve().parents[2] / 'python'}:{Path(__file__).resolve().parents[2]}",
    }
    command = [
        sys.executable,
        str(script),
        "--date",
        "20260927",
        "--root",
        str(private_root),
        "--output",
        str(output),
        "--fixture",
    ]
    assert (
        subprocess.run(command, env=env, capture_output=True, text=True, check=False).returncode
        == 0
    )
    raw = private_root / "results/raw/experiment_7751_v674_hardware_continuity/rows.json"
    assert (
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--cold-reduce",
                str(output),
                "--root",
                str(private_root),
                "--raw",
                str(raw),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )


def test_direct_cli_modes(private_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7751-TERMINAL: both entrypoint branches report progress."""
    candidate = private_root / "candidate.json"
    assert (
        main(
            [
                "--date",
                "20260927",
                "--root",
                str(private_root),
                "--output",
                str(candidate),
                "--fixture",
            ]
        )
        == 0
    )
    value = json.loads(candidate.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["verifier_is_oracle"] is True
    assert main(["--root", str(private_root), "--cold-reduce", str(candidate)]) == 0
    assert "after cold_reduce" in capsys.readouterr().out


def test_missing_and_malformed_inventory(private_root: Path) -> None:
    """SCENARIO-REPORT-7751-CUSTODY: broken required evidence is blocked."""
    source = private_root / "results/experiment_7599_v663_board_continuity.json"
    source.unlink()
    absent = build_audit(private_root, "20260927")
    assert absent["verdict_class"] == "blocked"
    assert any(g["field"] == "byte_count" for g in absent["gate_check_summary"])
    _write(private_root, "results/experiment_7599_v663_board_continuity.json", {"board_rows": []})
    malformed = build_audit(private_root, "20260927")
    assert malformed["verdict_class"] == "blocked"
    assert any(g["field"] == "board_rows" for g in malformed["gate_check_summary"])


def test_cost_only_maps_measured_quadratic_kernel(private_root: Path) -> None:
    """SCENARIO-HW-7751-NO-PROBE: finite normalization is not a fabric kernel."""
    cost = "results/experiment_7750_v674_service_cost.json"
    _write(
        private_root,
        cost,
        {
            "verdict_class": "null",
            "honest_verdict": "complete_null_cost",
            "flagged_adversarial": False,
            "service_fractions": {"finite_set_normalization": 0.4},
        },
    )
    no_kernel = build_audit(private_root, "20260927")
    assert no_kernel["cost_input_eligibility"]["eligible"] is True
    assert all(row["measured_service_fraction"] is None for row in no_kernel["rows"])
    _write(
        private_root,
        cost,
        {
            "verdict_class": "null",
            "honest_verdict": "complete_null_cost",
            "flagged_adversarial": False,
            "service_fractions": {"quadratic_ising_kernel": 0.2},
        },
    )
    mapped = build_audit(private_root, "20260927")
    assert mapped["hardware_continuity_complete_score"] == 1
    assert (
        next(row for row in mapped["rows"] if row["substrate"] == "KV260")[
            "measured_service_fraction"
        ]
        == 0.2
    )
    assert all(
        row["measured_service_fraction"] is None
        for row in mapped["rows"]
        if row["substrate"] != "KV260"
    )


def test_reject_ineligible_or_invalid_cost(private_root: Path) -> None:
    """SCENARIO-REPORT-7751-CUSTODY: disqualified cost stays out of placement."""
    cost = "results/experiment_7750_v674_service_cost.json"
    _write(
        private_root,
        cost,
        {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified",
            "flagged_adversarial": True,
            "service_fractions": {"quadratic_ising_kernel": 0.9},
        },
    )
    rejected = build_audit(private_root, "20260927")
    assert rejected["cost_input_eligibility"]["reason"] == "producer_ineligible"
    (private_root / cost).write_text("{")
    invalid = build_audit(private_root, "20260927")
    assert invalid["cost_input_eligibility"]["reason"] == "invalid_json"
