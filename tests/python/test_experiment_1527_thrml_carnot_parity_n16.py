"""Tests for Exp 1527 exact n=16 THRML/Carnot parity.

Spec refs: REQ-SAMPLE-048, SCENARIO-SAMPLE-076.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_carnot_parity_n16 as exp1527


class FakeBackend:
    """Deterministic SamplerBackend-shaped fake for fixed-seed evidence rows."""

    def __init__(self, seed: int, backend_name: str) -> None:
        self.seed = int(seed)
        self.backend_name = backend_name

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        del biases, couplings, config
        base = np.array([(idx + self.seed) % 2 == 0 for idx in range(16)], dtype=bool)
        return np.tile(base, (int(n_samples), 1))


def _fake_backend_factory(name: str) -> Any:
    def factory(seed: int) -> FakeBackend:
        return FakeBackend(seed=seed, backend_name=name)

    return factory


def _write_exp1526(path: Path, *, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_parity_n8_passed": passed,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "exact_states_enumerated": 256,
            }
        )
        + "\n",
        encoding="utf-8",
    )


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
        return SimpleNamespace(
            __version__="fake-0.1", __file__="/fake/thrml/__init__.py", SpinNode=SpinNode
        )
    if name == "thrml.models":
        return SimpleNamespace(IsingEBM=IsingEBM)
    raise ModuleNotFoundError(name)


def _missing_thrml_import(name: str) -> Any:
    raise ModuleNotFoundError(name)


def _fake_thrml_without_nodes(name: str) -> Any:
    class IsingEBM:
        pass

    if name == "thrml":
        return SimpleNamespace(__version__="fake-0.1", __file__="/fake/thrml/__init__.py")
    if name == "thrml.models":
        return SimpleNamespace(IsingEBM=IsingEBM)
    raise ModuleNotFoundError(name)


def test_spec_mentions_exp1527_contract() -> None:
    """REQ-SAMPLE-048, SCENARIO-SAMPLE-076: Exp1527 is spec-anchored."""

    spec = (exp1527.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-048" in spec
    assert "SCENARIO-SAMPLE-076" in spec
    assert "experiment_1527_thrml_carnot_parity_n16.json" in spec
    assert "thrml_carnot_parity_n16_1527.jsonl" in spec


def test_n16_case_is_deterministic_signed_ring_chord() -> None:
    """REQ-SAMPLE-048: the parity helper defines the deterministic n=16 case."""

    case = exp1527.n16_signed_ring_chord_case()
    states = exp1527.enumerate_spin_states(case.n_spins)

    assert case.n_spins == 16
    assert case.name == "n16_signed_ring_chord"
    assert case.topology == "signed_ring_chord"
    assert states.shape == (65536, 16)
    assert np.allclose(case.j_matrix, case.j_matrix.T)
    assert np.allclose(np.diag(case.j_matrix), 0.0)
    assert np.count_nonzero(np.triu(case.j_matrix, 1)) == 32
    assert exp1527.ising_energy(case, states[0]) == pytest.approx(0.11)


def test_write_in_progress_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-SAMPLE-048: bootstrap artifact is written before parity execution."""

    output_path = tmp_path / "experiment_1527.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp1527.write_in_progress_artifact(output_path, manifest_path)

    assert artifact["status"] == "in_progress"
    assert artifact["thrml_parity_n16_passed"] is False
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["n_spins"] == 16
    assert artifact["exact_states_enumerated"] == 0
    assert artifact["parity_manifest_path"] == str(manifest_path)
    assert artifact["honest_verdict"].startswith("success_")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_complete_run_writes_exact_65536_and_sampling_metrics(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-076: exact n=16 metrics and JSONL evidence are written."""

    exp1526_path = tmp_path / "exp1526.json"
    output_path = tmp_path / "experiment_1527.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1526(exp1526_path)

    artifact = exp1527.run_parity_n16(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1526_path=exp1526_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        sample_count=6,
        n_warmup=4,
        steps_per_sample=1,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["thrml_parity_n16_passed"] is True
    assert artifact["n_spins"] == 16
    assert artifact["exact_states_enumerated"] == 65536
    assert artifact["partition_relative_error"] == pytest.approx(0.0)
    assert artifact["mean_energy_delta"] == pytest.approx(0.0)
    assert artifact["kl_divergence"] == pytest.approx(0.0)
    assert artifact["sample_mean_energy_delta"] == pytest.approx(0.0)
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert {row["case_type"] for row in rows} == {
        "exact_distribution_parity",
        "fixed_seed_sampling_secondary_check",
    }
    assert rows[0]["case_id"] == "exp1527:n16_signed_ring_chord:exact_distribution"
    assert rows[0]["state_count"] == 65536
    assert rows[1]["case_id"] == "exp1527:n16_signed_ring_chord:fixed_seed_sampling"
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_upstream_exp1526_evidence_blocks_before_thrml_import(tmp_path: Path) -> None:
    """REQ-SAMPLE-048: Exp1526 passed parity evidence gates the n=16 run."""

    malformed_exp1526 = tmp_path / "malformed_exp1526.json"
    not_passed_exp1526 = tmp_path / "not_passed_exp1526.json"
    malformed_exp1526.write_text("{not-json", encoding="utf-8")
    _write_exp1526(not_passed_exp1526, passed=False)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when Exp1526 is not ready")

    missing_artifact = exp1527.run_parity_n16(
        output_path=tmp_path / "missing_1527.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        exp1526_path=tmp_path / "missing_exp1526.json",
        importer=importer,
    )
    malformed_artifact = exp1527.run_parity_n16(
        output_path=tmp_path / "malformed_1527.json",
        manifest_path=tmp_path / "malformed_manifest.jsonl",
        exp1526_path=malformed_exp1526,
        importer=importer,
    )
    not_passed_artifact = exp1527.run_parity_n16(
        output_path=tmp_path / "not_passed_1527.json",
        manifest_path=tmp_path / "not_passed_manifest.jsonl",
        exp1526_path=not_passed_exp1526,
        importer=importer,
    )

    assert missing_artifact["blockers"][0]["blocker"] == "exp1526_evidence_missing"
    assert malformed_artifact["blockers"][0]["blocker"] == "exp1526_evidence_malformed"
    assert not_passed_artifact["blockers"][0]["blocker"] == "exp1526_parity_not_passed"
    assert missing_artifact["simulator_only"] is True
    assert missing_artifact["no_tsu_hardware_claim"] is True


def test_missing_thrml_import_and_api_write_terminal_blockers(tmp_path: Path) -> None:
    """REQ-SAMPLE-048: unavailable THRML remains terminal data, not a fake pass."""

    exp1526_path = tmp_path / "exp1526.json"
    _write_exp1526(exp1526_path)

    missing_import = exp1527.run_parity_n16(
        output_path=tmp_path / "missing_import_1527.json",
        manifest_path=tmp_path / "missing_import.jsonl",
        exp1526_path=exp1526_path,
        importer=_missing_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )
    missing_api = exp1527.run_parity_n16(
        output_path=tmp_path / "missing_api_1527.json",
        manifest_path=tmp_path / "missing_api.jsonl",
        exp1526_path=exp1526_path,
        importer=_fake_thrml_without_nodes,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )

    assert missing_import["status"] == "blocked"
    assert missing_import["thrml_parity_n16_passed"] is False
    assert missing_import["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert missing_import["exact_states_enumerated"] == 0
    assert missing_api["blockers"][0]["blocker"] == "thrml_ising_energy_api_unavailable"
    assert "SpinNode" in missing_api["blockers"][0]["detail"]


def test_validate_artifact_rejects_bad_claims_and_metrics(tmp_path: Path) -> None:
    """REQ-SAMPLE-048: schema validation enforces pass metrics and no-TSU gates."""

    artifact = exp1527.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "thrml_parity_n16_passed": True,
            "exact_states_enumerated": 65536,
            "topology": "signed_ring_chord",
            "carnot_partition_function": 10.0,
            "thrml_partition_function": 10.0,
            "partition_relative_error": 0.0,
            "mean_energy_delta": 0.0,
            "kl_divergence": 0.0,
            "sample_mean_energy_delta": 0.0,
            "thresholds": {
                "partition_relative_error_max": 1e-8,
                "mean_energy_delta_abs_max": 1e-8,
                "kl_divergence_max": 1e-8,
                "sample_mean_energy_delta_abs_max": 0.35,
            },
            "blockers": [],
            "honest_verdict": "complete_thrml_carnot_parity_n16_passed_no_tsu_hardware_claim",
        }
    )
    exp1527.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1527.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1527.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1527.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1527.validate_artifact(bad_tsu)

    bad_metric = dict(artifact, partition_relative_error=1e-4)
    with pytest.raises(ValueError, match="pass metrics"):
        exp1527.validate_artifact(bad_metric)

    bad_sample_metric = dict(artifact, sample_mean_energy_delta=0.5)
    with pytest.raises(ValueError, match="pass metrics"):
        exp1527.validate_artifact(bad_sample_metric)

    bad_verdict = dict(artifact, honest_verdict="claimed hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1527.validate_artifact(bad_verdict)
