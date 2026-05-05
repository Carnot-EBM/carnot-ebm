"""Tests for Exp 1347 THRML compatibility parity audit.

Spec refs: REQ-SAMPLE-041, SCENARIO-SAMPLE-069.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.analysis import thrml_compatibility_parity as exp1347


def _missing_thrml_import(name: str) -> Any:
    raise ModuleNotFoundError(name)


def _fake_thrml_import(name: str) -> Any:
    class SpinNode:
        pass

    class IsingEBM:
        def __init__(
            self,
            nodes: list[SpinNode],
            edges: list[tuple[SpinNode, SpinNode]],
            biases: np.ndarray,
            weights: np.ndarray,
            beta: float,
        ) -> None:
            self.nodes = nodes
            self.edges = edges
            self.biases = np.asarray(biases, dtype=np.float64)
            self.weights = np.asarray(weights, dtype=np.float64)
            self.beta = float(beta)

        def energy(self, spins: np.ndarray) -> float:
            spin_vec = np.asarray(spins, dtype=np.float64)
            node_index = {id(node): idx for idx, node in enumerate(self.nodes)}
            energy = -float(self.biases @ spin_vec)
            for weight, (left, right) in zip(self.weights, self.edges):
                i = node_index[id(left)]
                j = node_index[id(right)]
                energy -= float(weight) * float(spin_vec[i] * spin_vec[j])
            return energy

    if name == "thrml":
        return SimpleNamespace(__version__="fake-0.1", SpinNode=SpinNode)
    if name == "thrml.models":
        return SimpleNamespace(IsingEBM=IsingEBM)
    raise ModuleNotFoundError(name)


def _fake_thrml_without_models_import(name: str) -> Any:
    if name == "thrml":
        return SimpleNamespace(__version__="fake-0.1")
    raise ModuleNotFoundError(name)


def test_scenario_sample_069_missing_thrml_records_mapping_notes_only() -> None:
    """SCENARIO-SAMPLE-069: unavailable THRML produces an honest blocked audit."""
    artifact = exp1347.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        import_module=_missing_thrml_import,
    )

    exp1347.validate_artifact(artifact)
    assert exp1347.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["metadata"]["run_date"] == "20260505"
    assert artifact["thrml_import_available"] is False
    assert artifact["energy_parity_max_abs_error"] is None
    assert artifact["sample_quality_proxy"] is None
    assert "thrml" in artifact["missing_api_or_dependency"]
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_thrml_not_importable_no_hardware_claim"
    assert {case["case"] for case in artifact["cases_attempted"]} == {
        "tiny_ising:n4_signed_ring_chord",
        "tiny_kan:univariate_kaem_note",
    }


def test_scenario_sample_069_partial_thrml_api_blocks_parity() -> None:
    """SCENARIO-SAMPLE-069: importable THRML without models stays claim-blocked."""
    artifact = exp1347.build_artifact(import_module=_fake_thrml_without_models_import)

    exp1347.validate_artifact(artifact)
    assert artifact["thrml_import_available"] is True
    assert artifact["energy_parity_max_abs_error"] is None
    assert artifact["sample_quality_proxy"] is None
    assert "thrml.models" in artifact["missing_api_or_dependency"]
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_thrml_api_missing_no_hardware_claim"


def test_req_sample_041_fake_thrml_measures_tiny_ising_parity() -> None:
    """REQ-SAMPLE-041: available local THRML energy API records parity metrics."""
    artifact = exp1347.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        import_module=_fake_thrml_import,
    )

    exp1347.validate_artifact(artifact)
    assert artifact["thrml_import_available"] is True
    assert artifact["thrml_version"] == "fake-0.1"
    assert artifact["energy_parity_max_abs_error"] == pytest.approx(0.0)
    assert artifact["sample_quality_proxy"]["proxy_name"] == "exact_energy_parity_score"
    assert artifact["sample_quality_proxy"]["proxy_value"] == pytest.approx(1.0)
    assert artifact["hardware_claim_allowed"] is True
    assert artifact["honest_verdict"] == "local_thrml_parity_measured_no_tsu_execution_claim"
    assert artifact["metadata"]["tsu_hardware_execution_confirmed"] is False


def test_req_sample_041_missing_ising_classes_are_explicit() -> None:
    """REQ-SAMPLE-041: missing SpinNode/IsingEBM is not treated as parity."""
    probe = exp1347.ThrmlProbe(
        import_available=True,
        module=SimpleNamespace(),
        models_module=SimpleNamespace(),
        version="fake-0.1",
        missing_api_or_dependency=None,
    )

    with pytest.raises(exp1347.MissingThrmlApi, match="SpinNode"):
        exp1347.measure_tiny_ising_thrml_parity(probe)


def test_req_sample_041_missing_energy_method_is_explicit() -> None:
    """REQ-SAMPLE-041: an Ising model without energy(spins) does not measure parity."""

    class SpinNode:
        pass

    class IsingEBM:
        def __init__(self, *_args: Any) -> None:
            pass

    probe = exp1347.ThrmlProbe(
        import_available=True,
        module=SimpleNamespace(SpinNode=SpinNode),
        models_module=SimpleNamespace(IsingEBM=IsingEBM),
        version="fake-0.1",
        missing_api_or_dependency=None,
    )

    with pytest.raises(exp1347.MissingThrmlApi, match="energy"):
        exp1347.measure_tiny_ising_thrml_parity(probe)


def test_req_sample_041_validator_rejects_incomplete_or_dishonest_artifacts() -> None:
    """REQ-SAMPLE-041: validation enforces schema and claim gates."""
    artifact = exp1347.build_artifact(import_module=_missing_thrml_import)

    missing = dict(artifact)
    missing.pop("cases_attempted")
    with pytest.raises(ValueError, match="missing"):
        exp1347.validate_artifact(missing)

    dishonest = dict(artifact)
    dishonest["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1347.validate_artifact(dishonest)

    unavailable_with_parity = dict(artifact)
    unavailable_with_parity["energy_parity_max_abs_error"] = 0.0
    with pytest.raises(ValueError, match="unavailable THRML"):
        exp1347.validate_artifact(unavailable_with_parity)

    unknown_verdict = dict(artifact)
    unknown_verdict["honest_verdict"] = "claimed_z1_speedup_without_local_evidence"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1347.validate_artifact(unknown_verdict)


def test_scenario_sample_069_write_artifact_round_trips_json(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-069: writer persists the final audit JSON."""
    output_path = tmp_path / "experiment_1347_thrml_compatibility_parity_audit.json"
    artifact = exp1347.build_artifact(import_module=_missing_thrml_import)

    written = exp1347.write_artifact(output_path, artifact)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert payload == artifact
    assert payload["hardware_claim_allowed"] is False
