"""Tests for Exp 1422 Discrete SB KV260 RTL specification.

Spec traces: REQ-ISING-023, SCENARIO-ISING-033
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.hardware import discrete_sb_rtl_spec as exp1422


def _minimal_exp1399() -> dict:
    """Return the Exp 1399 fields that Exp 1422 must carry forward."""

    return {
        "status": "complete",
        "algorithm": (
            "Discrete Simulated Bifurcation CPU sign-pressure model "
            "x_i(t+1)=sign(x_i(t)+eta*sum_j J_ij*x_j(t)-pressure(t))"
        ),
        "convergence_speedup_discrete_sb": 1.3846153846153846,
        "bram_estimate_kb_for_256var": 64.0,
        "kv260_bram_budget_kb": 648,
        "lut_estimate_per_update_unit": 2000,
        "kv260_lut_budget_fits": True,
        "bram_budget_feasible": True,
        "metadata": {
            "hardware_execution_performed": False,
            "synthesis_performed": False,
        },
    }


def test_req_ising_023_resource_estimate_carries_forward_exp1399() -> None:
    """REQ-ISING-023: RTL estimate preserves Exp 1399 BRAM/LUT arithmetic."""

    estimate = exp1422.build_resource_estimate(_minimal_exp1399())

    assert estimate["n_variables"] == 256
    assert estimate["bits_per_coupling"] == 8
    assert estimate["estimated_bram_kb"] == 64.0
    assert estimate["kv260_bram_budget_kb"] == 648
    assert estimate["estimated_lut"] == 2000
    assert estimate["kv260_lut_budget"] == 117000
    assert estimate["kv260_budget_fits"] is True


def test_req_ising_023_markdown_documents_rtl_contract_and_synthesis() -> None:
    """REQ-ISING-023: spec text includes datapath, memory, RNG, schedule, and host API."""

    estimate = exp1422.build_resource_estimate(_minimal_exp1399())
    markdown = exp1422.build_rtl_spec_markdown(
        estimate=estimate,
        exp1399=_minimal_exp1399(),
        run_date="20260506",
    )

    for section in (
        "## Datapath",
        "## Memory Layout",
        "## Random/Noise Source Assumptions",
        "## Update Schedule",
        "## Host Interface",
        "## Synthesis Plan",
        "## Claim Boundary",
    ):
        assert section in markdown

    assert "256 x 256 x 8 bits = 65536 bytes = 64.0 KB" in markdown
    assert "vivado -mode batch -source hardware/kv260/synth_discrete_sb.tcl" in markdown
    assert "No hardware execution was performed" in markdown


def test_scenario_ising_033_write_outputs_persists_spec_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-ISING-033: writer persists complete spec and hardware-gated JSON."""

    exp1399_path = tmp_path / "results" / "experiment_1399_discrete_sb_kv260_cpu_simulation.json"
    exp1399_path.parent.mkdir(parents=True)
    exp1399_path.write_text(json.dumps(_minimal_exp1399()), encoding="utf-8")
    artifact_path = tmp_path / "results" / "experiment_1422_discrete_sb_kv260_rtl_spec.json"
    spec_path = tmp_path / "hardware" / "kv260" / "discrete_sb_rtl_spec.md"

    artifact = exp1422.write_outputs(
        project_root=tmp_path,
        run_date="20260506",
        exp1399_path=exp1399_path,
        artifact_path=artifact_path,
        rtl_spec_path=spec_path,
    )

    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    markdown = spec_path.read_text(encoding="utf-8")

    assert persisted == artifact
    assert exp1422.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["rtl_spec_complete"] is True
    assert artifact["rtl_spec_path"] == "hardware/kv260/discrete_sb_rtl_spec.md"
    assert artifact["estimated_lut"] == 2000
    assert artifact["estimated_bram"] == 64.0
    assert artifact["kv260_budget_fits"] is True
    assert artifact["synthesis_command_documented"] is True
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == exp1422.CPU_ONLY_HONEST_VERDICT
    assert "Discrete SB KV260 RTL Specification" in markdown


def test_req_ising_023_validation_rejects_incomplete_or_dishonest_artifacts() -> None:
    """REQ-ISING-023: validator rejects missing fields and unsupported hardware claims."""

    artifact = exp1422.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260506",
        exp1399=_minimal_exp1399(),
    )

    missing = dict(artifact)
    missing.pop("rtl_spec_complete")
    with pytest.raises(ValueError, match="missing"):
        exp1422.validate_artifact(missing)

    incomplete = dict(artifact)
    incomplete["rtl_spec_complete"] = False
    with pytest.raises(ValueError, match="rtl_spec_complete"):
        exp1422.validate_artifact(incomplete)

    dishonest_execution = dict(artifact)
    dishonest_execution["hardware_execution_performed"] = True
    dishonest_execution["hardware_claim_allowed"] = False
    with pytest.raises(ValueError, match="hardware_execution_performed"):
        exp1422.validate_artifact(dishonest_execution)

    dishonest_claim = dict(artifact)
    dishonest_claim["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1422.validate_artifact(dishonest_claim)

    undocumented_synthesis = dict(artifact)
    undocumented_synthesis["synthesis_command_documented"] = False
    with pytest.raises(ValueError, match="synthesis_command_documented"):
        exp1422.validate_artifact(undocumented_synthesis)

    bad_status = dict(artifact)
    bad_status["status"] = "in_progress"
    with pytest.raises(ValueError, match="status"):
        exp1422.validate_artifact(bad_status)

    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "hardware_execution_claimed"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1422.validate_artifact(bad_verdict)
