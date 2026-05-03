"""Tests for Exp 1150 Extropic integration packet artifact.

Spec refs: REQ-SAMPLE-040, SCENARIO-SAMPLE-068.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1150_extropic_integration_packet as exp1150


def test_packet_writer_documents_required_interface_sections(tmp_path: Path) -> None:
    """REQ-SAMPLE-040: packet documents workload, backend API, gates, and fallback."""
    packet_path = tmp_path / "extropic_integration_packet.md"

    exp1150.write_integration_packet(
        packet_path=packet_path,
        thrml_available=False,
        thrml_version=None,
        thrml_latency_us=None,
    )

    text = packet_path.read_text()
    assert "Ising sampling" in text
    assert "spin state read/write" in text
    assert "energy evaluation" in text
    assert "ThermoSamplerBackend" in text
    assert "KL(Z1 || CPU_Gibbs) < 0.05" in text
    assert "latency < 1ms" in text
    assert "THRML CPU simulation" in text
    assert "pip install thrml" in text


def test_build_artifact_has_required_unavailable_schema() -> None:
    """SCENARIO-SAMPLE-068: artifact records unavailable THRML honestly."""
    artifact = exp1150.build_artifact(
        thrml_available=False,
        thrml_version=None,
        thrml_latency_us=None,
        packet_written=True,
        backend_stub_written=True,
    )

    assert exp1150.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["thrml_available"] is False
    assert artifact["thrml_version"] is None
    assert artifact["thrml_latency_us"] is None
    assert artifact["integration_packet_written"] is True
    assert artifact["extropic_integration_packet_written"] is True
    assert artifact["sampler_backend_interface_documented"] is True
    assert artifact["honest_verdict"] == "thrml_not_available_packet_written"


def test_build_artifact_records_available_benchmark_schema() -> None:
    """SCENARIO-SAMPLE-068: available THRML run records latency and verdict."""
    artifact = exp1150.build_artifact(
        thrml_available=True,
        thrml_version="0.1.0",
        thrml_latency_us=1234.5,
        packet_written=True,
        backend_stub_written=True,
    )

    assert artifact["thrml_available"] is True
    assert artifact["thrml_version"] == "0.1.0"
    assert artifact["thrml_latency_us"] == pytest.approx(1234.5)
    assert artifact["honest_verdict"] == "thrml_available_benchmark_run"


def test_validate_artifact_rejects_incomplete_payload() -> None:
    """REQ-SAMPLE-040: schema validation catches partial packet artifacts."""
    artifact = exp1150.build_artifact(
        thrml_available=False,
        thrml_version=None,
        thrml_latency_us=None,
        packet_written=False,
        backend_stub_written=True,
    )

    assert artifact["honest_verdict"] == "partial_packet_written"
    with pytest.raises(ValueError, match="integration packet"):
        exp1150.validate_artifact(artifact)


def test_write_artifact_round_trips_json(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-068: artifact writer emits stable JSON."""
    artifact = exp1150.build_artifact(
        thrml_available=False,
        thrml_version=None,
        thrml_latency_us=None,
        packet_written=True,
        backend_stub_written=True,
    )
    output_path = tmp_path / "artifact.json"

    exp1150.write_artifact(artifact, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded == artifact


@pytest.mark.skipif(not exp1150.DELIVERABLE.exists(), reason="artifact not yet generated")
def test_deliverable_json_has_required_fields() -> None:
    """SCENARIO-SAMPLE-068: generated deliverable satisfies the roadmap contract."""
    payload = json.loads(exp1150.DELIVERABLE.read_text())

    assert exp1150.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["packet_path"] == "docs/hardware/extropic_integration_packet.md"
    assert payload["thrml_backend_path"] == "python/carnot/samplers/thrml_backend.py"
    assert payload["honest_verdict"] in exp1150.HONEST_VERDICTS
