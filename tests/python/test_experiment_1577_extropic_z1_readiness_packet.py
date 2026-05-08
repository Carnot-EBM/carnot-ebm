"""Tests for Exp 1577 Extropic Z1 readiness packet THRML alignment.

Spec refs: REQ-REPORT-065, SCENARIO-REPORT-065.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.reporting.extropic_z1_readiness_packet_thrml_alignment as exp1577


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1545_payload(*, no_hardware: bool = True) -> dict[str, object]:
    return {
        "status": "complete",
        "extropic_z1_readiness_packet_ready": True,
        "readiness_packet_path": "ops/extropic_z1_readiness_packet.md",
        "no_hardware_execution_claim": no_hardware,
        "access_blockers": ["no_authenticated_extropic_z1_or_xtr0_device_access"],
        "honest_verdict": "complete: extropic_z1_access_readiness_packet_ready_no_hardware_execution_claim",
    }


def _exp1564_payload(*, no_hardware: bool = True) -> dict[str, object]:
    return {
        "status": "complete",
        "thrml_vendoring_complete": True,
        "thrml_version": "0.1.3",
        "thrml_license": "Apache-2.0",
        "candidate_warm_start_implemented": True,
        "kl_to_thrml_after_vendoring": 0.0,
        "simulator_only": no_hardware,
        "no_tsu_hardware_claim": no_hardware,
        "honest_verdict": "complete: vendored_thrml_block_gibbs_candidate_warm_start_complete",
    }


def _exp1565_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "soft_gibbs_residual_implemented": True,
        "soft_brs_decay_confirmed": True,
        "hard_brs_acceptance_rate": 0.0,
        "honest_verdict": "complete: soft_gibbs_residual_operational_hard_brs_empty_intersection_falsified",
    }


def _exp1566_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "candidate_warm_start_validated": True,
        "recommended_deployment_policy": "candidate_warm_start",
        "cold_start_accuracy_drop_percent_at_k100": 51.052632,
        "cached_state_worse_than_cold_start": True,
        "honest_verdict": "complete: candidate_warm_start_validated_cold_and_cached_state_rejected",
    }


def test_req_report_065_spec_mentions_exp1577_contract() -> None:
    """REQ-REPORT-065, SCENARIO-REPORT-065: Exp1577 is spec-anchored."""

    spec = (exp1577.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-065" in spec
    assert "SCENARIO-REPORT-065" in spec
    assert "experiment_1577_extropic_z1_readiness_packet_thrml_alignment_resumed.json" in spec
    assert "extropic-z1-readiness-packet-2026-05-121.md" in spec
    assert "detailed-balance drift correction" in spec


def test_req_report_065_builds_thrml_aligned_packet_and_artifact() -> None:
    """REQ-REPORT-065: packet records THRML, warm-start, Soft-Gibbs, and drift gates."""

    artifact, packet_text = exp1577.build_artifact(
        exp1545=_exp1545_payload(),
        exp1564=_exp1564_payload(),
        exp1565=_exp1565_payload(),
        exp1566=_exp1566_payload(),
        packet_path="docs/research-notes/extropic-z1-readiness-packet-2026-05-121.md",
    )

    assert exp1577.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["packet_path"] == "docs/research-notes/extropic-z1-readiness-packet-2026-05-121.md"
    assert artifact["extropic_z1_packet_updated"] is True
    assert artifact["thrml_vendoring_reflected"] is True
    assert artifact["analog_drift_correction_required"] is True
    assert artifact["simulator_only_no_hardware_claim"] is True
    assert artifact["honest_verdict"] == exp1577.COMPLETE_VERDICT
    assert "THRML 0.1.3" in packet_text
    assert "candidate warm-start" in packet_text
    assert "Soft-Gibbs Residual" in packet_text
    assert "pre-silicon correction prerequisites" in packet_text
    assert "detailed-balance drift correction" in packet_text
    assert "simulator-only" in packet_text
    assert "No Z1, XTR, or TSU hardware access is claimed" in packet_text


def test_req_report_065_rejects_hardware_claim_drift() -> None:
    """REQ-REPORT-065: source and packet validation fail closed on hardware claims."""

    with pytest.raises(ValueError, match="Exp1564"):
        exp1577.build_artifact(
            exp1545=_exp1545_payload(),
            exp1564=_exp1564_payload(no_hardware=False),
            exp1565=_exp1565_payload(),
            exp1566=_exp1566_payload(),
            packet_path="docs/research-notes/extropic-z1-readiness-packet-2026-05-121.md",
        )

    artifact, packet_text = exp1577.build_artifact(
        exp1545=_exp1545_payload(),
        exp1564=_exp1564_payload(),
        exp1565=_exp1565_payload(),
        exp1566=_exp1566_payload(),
        packet_path="docs/research-notes/extropic-z1-readiness-packet-2026-05-121.md",
    )

    weakened = dict(artifact)
    weakened["simulator_only_no_hardware_claim"] = False
    with pytest.raises(ValueError, match="simulator_only_no_hardware_claim"):
        exp1577.validate_artifact(weakened)

    with pytest.raises(ValueError, match="pre-silicon correction prerequisites"):
        exp1577.validate_packet_text(packet_text.replace("pre-silicon correction prerequisites", "blocked"))

    with pytest.raises(ValueError, match="hardware claim"):
        exp1577.validate_packet_text(packet_text + "\nZ1 hardware execution completed.\n")


def test_scenario_report_065_run_writes_packet_and_terminal_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-065: run writes final packet and JSON without hardware claims."""

    _write_json(
        tmp_path / "results" / "experiment_1545_extropic_z1_access_readiness_packet.json",
        _exp1545_payload(),
    )
    _write_json(
        tmp_path / "results" / "experiment_1564_thrml_vendored_block_gibbs_replacement.json",
        _exp1564_payload(),
    )
    _write_json(
        tmp_path / "results" / "experiment_1565_soft_gibbs_residual_implementation.json",
        _exp1565_payload(),
    )
    _write_json(
        tmp_path / "results" / "experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json",
        _exp1566_payload(),
    )

    artifact = exp1577.run(repo_root=tmp_path)
    result_path = (
        tmp_path
        / "results"
        / "experiment_1577_extropic_z1_readiness_packet_thrml_alignment_resumed.json"
    )
    packet_path = tmp_path / "docs" / "research-notes" / "extropic-z1-readiness-packet-2026-05-121.md"
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    packet_text = packet_path.read_text(encoding="utf-8")

    assert loaded == artifact
    assert loaded["status"] == "complete"
    assert loaded["extropic_z1_packet_updated"] is True
    assert loaded["analog_drift_correction_required"] is True
    assert loaded["simulator_only_no_hardware_claim"] is True
    assert "No Z1, XTR, or TSU hardware access is claimed" in packet_text
    assert "hardware execution completed" not in packet_text.lower()


def test_req_report_065_in_progress_artifact_is_bootstrap_only(tmp_path: Path) -> None:
    """REQ-REPORT-065: the first write is explicit in_progress state."""

    output_path = tmp_path / "results" / "experiment_1577.json"
    artifact = exp1577.write_in_progress_artifact(output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert loaded["status"] == "in_progress"
    assert loaded["extropic_z1_packet_updated"] is False
    assert loaded["honest_verdict"] == "in_progress"
