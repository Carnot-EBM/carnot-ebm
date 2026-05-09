"""Tests for Exp 1598 Z1 drift SamplerBackend compatibility.

Spec traces: REQ-SAMPLE-065, SCENARIO-SAMPLE-093.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.samplers import z1_drift_compatibility as exp1598


def test_req_sample_065_spec_anchor_exists() -> None:
    """REQ-SAMPLE-065, SCENARIO-SAMPLE-093: Exp 1598 work is spec-anchored."""

    spec = (
        exp1598.PROJECT_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-SAMPLE-065" in spec
    assert "SCENARIO-SAMPLE-093" in spec
    assert "results/experiment_1598_z1_drift.json" in spec


def test_req_sample_065_backend_boundary_runs_sample_and_minimize() -> None:
    """REQ-SAMPLE-065: corrected Z1 drift simulator satisfies SamplerBackend shape."""

    config = exp1598.CompatibilityConfig(
        n_spins=16,
        n_samples=8,
        n_warmup_sweeps=4,
        sweeps_per_sample=1,
        minimize_samples=3,
        minimize_steps=5,
    )

    artifact = exp1598.build_artifact(config)

    exp1598.validate_artifact(artifact)
    assert exp1598.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["z1_drift_compatibility_test_passed"] is True
    assert artifact["sampler_backend_protocol"] == "carnot.samplers.backend.SamplerBackend"
    assert artifact["sampler_backend_boundary"].endswith("SyntheticDriftIsingBackend")
    assert artifact["backend_name"] == "synthetic-drift-ising-hastings"
    assert artifact["sample_shape"] == [8, 16]
    assert artifact["minimize_shape"] == [3, 16]
    assert artifact["sample_dtype"] == "bool"
    assert artifact["minimize_dtype"] == "bool"
    assert 0.0 <= artifact["sample_acceptance_rate"] <= 1.0
    assert 0.0 <= artifact["minimize_acceptance_rate"] <= 1.0


def test_scenario_sample_093_strict_transcript_fields_are_absent() -> None:
    """SCENARIO-SAMPLE-093: simulator artifact excludes strict Z1 transcript fields."""

    artifact = exp1598.build_artifact(exp1598.CompatibilityConfig(n_spins=12, n_samples=4))

    assert artifact["strict_transcript_fields_absent"] is True
    assert artifact["strict_transcript_absent_fields"] == list(
        exp1598.STRICT_TRANSCRIPT_ABSENT_FIELDS
    )
    assert set(exp1598.STRICT_TRANSCRIPT_ABSENT_FIELDS).isdisjoint(artifact)
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["z1_hardware_execution"] is False
    assert artifact["tsu_hardware_execution"] is False
    assert artifact["simulator_only_no_hardware_claim"] is True

    dishonest = dict(artifact)
    dishonest["device_identifier"] = "claimed-z1"
    with pytest.raises(ValueError, match="strict transcript"):
        exp1598.validate_artifact(dishonest)


def test_scenario_sample_093_write_artifact_round_trips_json(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-093: writer persists the Exp 1598 terminal artifact."""

    output = tmp_path / "experiment_1598_z1_drift.json"
    artifact = exp1598.write_artifact(
        output,
        exp1598.CompatibilityConfig(
            n_spins=16,
            n_samples=6,
            n_warmup_sweeps=4,
            minimize_samples=2,
            minimize_steps=3,
        ),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == artifact
    assert payload["honest_verdict"] == (
        "complete: z1_drift_samplerbackend_compatibility_simulator_only_no_execution_claim"
    )


def test_req_sample_065_validator_rejects_claim_drift() -> None:
    """REQ-SAMPLE-065: validation rejects hardware claims and schema drift."""

    artifact = exp1598.build_artifact(exp1598.CompatibilityConfig(n_spins=12, n_samples=4))
    exp1598.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("backend_name")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1598.validate_artifact(missing)

    bad_claim = dict(artifact)
    bad_claim["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1598.validate_artifact(bad_claim)

    bad_strict_flag = dict(artifact)
    bad_strict_flag["strict_transcript_fields_absent"] = False
    with pytest.raises(ValueError, match="strict_transcript_fields_absent"):
        exp1598.validate_artifact(bad_strict_flag)

    bad_shape = dict(artifact)
    bad_shape["sample_shape"] = [4, 11]
    with pytest.raises(ValueError, match="sample_shape"):
        exp1598.validate_artifact(bad_shape)


def test_scenario_sample_093_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-SAMPLE-093: CLI main writes artifact and reports claim boundary."""

    calls: list[str] = []

    def fake_write(path: Path = exp1598.DEFAULT_RESULT_PATH) -> dict:
        calls.append(path.name)
        return {
            "compatibility_test_ready": True,
            "hardware_claim_allowed": False,
            "honest_verdict": "complete: fake",
        }

    monkeypatch.setattr(exp1598, "write_artifact", fake_write)

    exp1598.main()

    assert calls == ["experiment_1598_z1_drift.json"]
    assert "True False complete: fake" in capsys.readouterr().out
