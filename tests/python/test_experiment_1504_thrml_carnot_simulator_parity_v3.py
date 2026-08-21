"""Tests for the gated Exp 1504 THRML/Carnot simulator parity audit.

Spec refs: REQ-SAMPLE-045, SCENARIO-SAMPLE-073.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.samplers import thrml_carnot_simulator_parity_v3 as exp1504


def _write_gate(path: Path, *, ready: bool) -> None:
    path.write_text(
        json.dumps({"status": "complete", "thrml_import_ready": ready}) + "\n",
        encoding="utf-8",
    )


def test_write_in_progress_artifact_preserves_no_hardware_claim(tmp_path: Path) -> None:
    """REQ-SAMPLE-045: bootstrap artifact is simulator-only before probing."""

    output_path = tmp_path / "experiment_1504.json"

    artifact = exp1504.write_in_progress_artifact(output_path)

    assert artifact["status"] == "in_progress"
    assert artifact["parity_experiment_ran"] is False
    assert artifact["simulator_only"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_gate_closed_writes_terminal_blocker_without_importing_thrml(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-073: closed Exp 1503 gate prevents parity execution."""

    gate_path = tmp_path / "exp1503.json"
    output_path = tmp_path / "experiment_1504.json"
    _write_gate(gate_path, ready=False)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when the gate is closed")

    artifact = exp1504.run_parity_audit(
        output_path=output_path,
        gate_path=gate_path,
        importer=importer,
    )

    assert artifact["status"] == "blocked"
    assert artifact["parity_experiment_ran"] is False
    assert artifact["thrml_import_ready"] is False
    assert artifact["parity_pass_count"] == 0
    assert artifact["parity_fail_count"] == 0
    assert artifact["blockers"] == [
        {
            "blocker": "thrml_import_gate_closed",
            "detail": "Exp 1503 did not report thrml_import_ready=true",
        }
    ]
    assert artifact["hardware_claim_allowed"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_missing_or_malformed_gate_writes_terminal_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-045: unreadable gate artifacts block parity honestly."""

    missing_gate = tmp_path / "missing_exp1503.json"
    malformed_gate = tmp_path / "malformed_exp1503.json"
    malformed_gate.write_text("{not json", encoding="utf-8")

    missing_artifact = exp1504.run_parity_audit(
        output_path=tmp_path / "missing_output.json",
        gate_path=missing_gate,
    )
    malformed_artifact = exp1504.run_parity_audit(
        output_path=tmp_path / "malformed_output.json",
        gate_path=malformed_gate,
    )

    assert missing_artifact["blockers"][0]["detail"].startswith("missing gate artifact:")
    assert malformed_artifact["blockers"][0]["detail"].startswith("malformed gate artifact:")
    assert exp1504._round_metric(None) is None


def test_import_ready_but_incompatible_api_records_api_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-045: THRML API limitations are recorded without hardware claims."""

    gate_path = tmp_path / "exp1503.json"
    output_path = tmp_path / "experiment_1504.json"
    _write_gate(gate_path, ready=True)
    fake_thrml = SimpleNamespace(__version__="fake-thrml-no-ising-api")

    artifact = exp1504.run_parity_audit(
        output_path=output_path,
        gate_path=gate_path,
        importer=lambda _name: fake_thrml,
    )

    assert artifact["status"] == "blocked"
    assert artifact["parity_experiment_ran"] is False
    assert artifact["thrml_import_ready"] is True
    assert artifact["cases_compared"] == []
    assert artifact["blockers"][0]["blocker"] == "thrml_api_incompatible"
    assert "SpinNode" in artifact["blockers"][0]["detail"]
    assert artifact["simulator_only"] is True
    assert artifact["hardware_claim_allowed"] is False


def test_installed_thrml_runs_energy_and_stochastic_parity(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-073: import-ready THRML gets fixed-seed simulator parity."""

    thrml_module = pytest.importorskip("thrml")
    try:
        exp1504._require_thrml_api(thrml_module)
    except AttributeError as exc:
        pytest.skip(f"installed thrml lacks Exp1504 Ising APIs: {exc}")
    gate_path = tmp_path / "exp1503.json"
    output_path = tmp_path / "experiment_1504.json"
    _write_gate(gate_path, ready=True)

    artifact = exp1504.run_parity_audit(
        output_path=output_path,
        gate_path=gate_path,
        seed=1504,
        n_samples=64,
        n_warmup=64,
        steps_per_sample=4,
        importer=lambda _name: thrml_module,
    )

    assert artifact["status"] == "complete"
    assert artifact["parity_experiment_ran"] is True
    assert artifact["thrml_import_ready"] is True
    assert artifact["parity_pass_count"] == 2
    assert artifact["parity_fail_count"] == 0
    assert artifact["max_observed_delta"] <= artifact["tolerance"]["stochastic_mean_energy_abs"]
    assert [case["case"] for case in artifact["cases_compared"]] == [
        "tiny_ising:n4_signed_ring_chord:exact_energy",
        "tiny_ising:n4_signed_ring_chord:fixed_seed_sample_mean_energy",
    ]
    assert all(case["passed"] for case in artifact["cases_compared"])
    assert artifact["hardware_claim_allowed"] is False


def test_validate_artifact_rejects_missing_fields_and_bad_claims() -> None:
    """REQ-SAMPLE-045: schema validation enforces terminal no-hardware boundary."""

    valid = {
        "status": "blocked",
        "parity_experiment_ran": False,
        "thrml_import_ready": False,
        "simulator_only": True,
        "cases_compared": [],
        "parity_pass_count": 0,
        "parity_fail_count": 0,
        "tolerance": exp1504.DEFAULT_TOLERANCE.copy(),
        "max_observed_delta": None,
        "hardware_claim_allowed": False,
        "blockers": [{"blocker": "gate", "detail": "closed"}],
        "honest_verdict": "complete_gate_closed_no_hardware_claim",
    }

    exp1504.validate_artifact(valid)

    missing = valid.copy()
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1504.validate_artifact(missing)

    bad_claim = valid.copy()
    bad_claim["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1504.validate_artifact(bad_claim)

    bad_simulator = valid.copy()
    bad_simulator["simulator_only"] = False
    with pytest.raises(ValueError, match="simulator_only"):
        exp1504.validate_artifact(bad_simulator)


def test_validate_artifact_rejects_bad_terminal_shapes() -> None:
    """REQ-SAMPLE-045: terminal parity counts and verdict prefixes stay honest."""

    complete = {
        "status": "complete",
        "parity_experiment_ran": True,
        "thrml_import_ready": True,
        "simulator_only": True,
        "cases_compared": [{"case": "x", "passed": True}],
        "parity_pass_count": 1,
        "parity_fail_count": 0,
        "tolerance": exp1504.DEFAULT_TOLERANCE.copy(),
        "max_observed_delta": 0.0,
        "hardware_claim_allowed": False,
        "blockers": [],
        "honest_verdict": "complete_thrml_carnot_simulator_parity_passed_no_hardware_claim",
    }

    bad_status = complete.copy()
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="invalid status"):
        exp1504.validate_artifact(bad_status)

    bad_verdict = complete.copy()
    bad_verdict["honest_verdict"] = "thrml_parity_passed"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1504.validate_artifact(bad_verdict)

    bad_counts = complete.copy()
    bad_counts["parity_pass_count"] = 0
    with pytest.raises(ValueError, match="parity pass/fail counts"):
        exp1504.validate_artifact(bad_counts)

    bad_empty = complete.copy()
    bad_empty["cases_compared"] = []
    bad_empty["parity_pass_count"] = 0
    with pytest.raises(ValueError, match="complete parity artifact"):
        exp1504.validate_artifact(bad_empty)
