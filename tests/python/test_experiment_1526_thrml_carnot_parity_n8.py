"""Tests for Exp 1526 exact n=8 THRML/Carnot parity.

Spec refs: REQ-SAMPLE-047, SCENARIO-SAMPLE-075.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_carnot_parity_n8 as exp1526


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
        base = np.array([True, False, True, False, True, False, True, self.seed % 2 == 0])
        return np.tile(base, (int(n_samples), 1))


def _fake_backend_factory(name: str) -> Any:
    def factory(seed: int) -> FakeBackend:
        return FakeBackend(seed=seed, backend_name=name)

    return factory


def _write_exp1515(path: Path, *, ready: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_import_ready": ready,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "metadata": {"thrml_version": "fake-0.1"},
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


def _fake_thrml_without_energy(name: str) -> Any:
    class SpinNode:
        pass

    class IsingEBM:
        def __init__(self, *_args: Any) -> None:
            pass

    if name == "thrml":
        return SimpleNamespace(
            __version__="fake-0.1", __file__="/fake/thrml/__init__.py", SpinNode=SpinNode
        )
    if name == "thrml.models":
        return SimpleNamespace(IsingEBM=IsingEBM)
    raise ModuleNotFoundError(name)


def _fake_thrml_block_energy_import(name: str) -> Any:
    class SpinNode:
        pass

    class Block:
        def __init__(self, nodes: list[SpinNode]) -> None:
            self.nodes = nodes

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

        def energy(self, state: list[np.ndarray], blocks: list[Block]) -> float:
            spin_vec = np.where(np.asarray(state[0], dtype=bool), 1.0, -1.0)
            node_index = {id(node): idx for idx, node in enumerate(blocks[0].nodes)}
            energy = -float(self.biases @ spin_vec)
            for weight, (left, right) in zip(self.weights, self.edges):
                i = node_index[id(left)]
                j = node_index[id(right)]
                energy -= float(weight) * float(spin_vec[i] * spin_vec[j])
            return energy

    if name == "thrml":
        return SimpleNamespace(
            __version__="fake-0.1",
            __file__="/fake/thrml/__init__.py",
            SpinNode=SpinNode,
            Block=Block,
        )
    if name == "thrml.models":
        return SimpleNamespace(IsingEBM=IsingEBM)
    raise ModuleNotFoundError(name)


def _fake_thrml_block_energy_without_block(name: str) -> Any:
    class SpinNode:
        pass

    class IsingEBM:
        def __init__(self, *_args: Any) -> None:
            pass

        def energy(self, state: list[np.ndarray], blocks: list[Any]) -> float:
            del state, blocks
            return 0.0

    if name == "thrml":
        return SimpleNamespace(
            __version__="fake-0.1", __file__="/fake/thrml/__init__.py", SpinNode=SpinNode
        )
    if name == "thrml.models":
        return SimpleNamespace(IsingEBM=IsingEBM)
    raise ModuleNotFoundError(name)


def test_spec_mentions_exp1526_contract() -> None:
    """REQ-SAMPLE-047, SCENARIO-SAMPLE-075: Exp1526 is spec-anchored."""

    spec = (exp1526.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-047" in spec
    assert "SCENARIO-SAMPLE-075" in spec
    assert "experiment_1526_thrml_carnot_parity_n8.json" in spec
    assert "thrml_carnot_parity_n8_1526.jsonl" in spec


def test_n8_case_is_deterministic_signed_ring_chord() -> None:
    """REQ-SAMPLE-047: the parity helper defines the deterministic n=8 case."""

    case = exp1526.n8_signed_ring_chord_case()
    states = exp1526.enumerate_spin_states(case.n_spins)

    assert case.n_spins == 8
    assert case.name == "n8_signed_ring_chord"
    assert case.topology == "signed_ring_chord"
    assert states.shape == (256, 8)
    assert np.allclose(case.j_matrix, case.j_matrix.T)
    assert np.allclose(np.diag(case.j_matrix), 0.0)
    assert np.count_nonzero(np.triu(case.j_matrix, 1)) == 16
    assert exp1526.ising_energy(case, states[0]) == pytest.approx(0.36)


def test_write_in_progress_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-SAMPLE-047: bootstrap artifact is written before parity execution."""

    output_path = tmp_path / "experiment_1526.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp1526.write_in_progress_artifact(output_path, manifest_path)

    assert artifact["status"] == "in_progress"
    assert artifact["thrml_parity_n8_passed"] is False
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["parity_manifest_path"] == str(manifest_path)
    assert artifact["honest_verdict"].startswith("success_")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_complete_run_writes_exact_and_sampling_metrics(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-075: exact n=8 metrics and JSONL evidence are written."""

    exp1515_path = tmp_path / "exp1515.json"
    output_path = tmp_path / "experiment_1526.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1515(exp1515_path)

    artifact = exp1526.run_parity_n8(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1515_path=exp1515_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        sample_count=6,
        n_warmup=4,
        steps_per_sample=1,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["thrml_parity_n8_passed"] is True
    assert artifact["n_spins"] == 8
    assert artifact["exact_states_enumerated"] == 256
    assert artifact["partition_relative_error"] == pytest.approx(0.0)
    assert artifact["mean_energy_delta"] == pytest.approx(0.0)
    assert artifact["kl_divergence"] == pytest.approx(0.0)
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert {row["case_type"] for row in rows} == {
        "exact_distribution_parity",
        "fixed_seed_sampling_secondary_check",
    }
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_block_energy_api_also_measures_exact_metrics() -> None:
    """REQ-SAMPLE-047: installed THRML-style Block energy calls are supported."""

    modules, details, blocker = exp1526._import_thrml(_fake_thrml_block_energy_import)

    assert blocker is None
    assert details["thrml_version"] == "fake-0.1"
    metrics = exp1526.exact_parity_metrics(modules, exp1526.n8_signed_ring_chord_case())
    assert metrics["state_count"] == 256
    assert metrics["partition_relative_error"] == pytest.approx(0.0)


def test_missing_thrml_import_writes_terminal_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-047: unavailable THRML is terminal data, not a fake pass."""

    exp1515_path = tmp_path / "exp1515.json"
    output_path = tmp_path / "experiment_1526.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1515(exp1515_path)

    artifact = exp1526.run_parity_n8(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1515_path=exp1515_path,
        importer=_missing_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )

    assert artifact["status"] == "blocked"
    assert artifact["thrml_parity_n8_passed"] is False
    assert artifact["exact_states_enumerated"] == 0
    assert artifact["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert not manifest_path.exists()


def test_exp1515_not_ready_blocks_before_thrml_import(tmp_path: Path) -> None:
    """REQ-SAMPLE-047: exp1515 readiness must gate the n=8 parity run."""

    exp1515_path = tmp_path / "exp1515.json"
    _write_exp1515(exp1515_path, ready=False)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when Exp1515 is not ready")

    artifact = exp1526.run_parity_n8(
        output_path=tmp_path / "experiment_1526.json",
        manifest_path=tmp_path / "manifest.jsonl",
        exp1515_path=exp1515_path,
        importer=importer,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )

    assert artifact["status"] == "blocked"
    assert artifact["blockers"][0]["blocker"] == "exp1515_thrml_import_not_ready"


def test_missing_or_malformed_exp1515_evidence_blocks_before_import(tmp_path: Path) -> None:
    """REQ-SAMPLE-047: missing/malformed exp1515 evidence is terminal data."""

    malformed_exp1515 = tmp_path / "malformed_exp1515.json"
    malformed_exp1515.write_text("{not-json", encoding="utf-8")

    missing_artifact = exp1526.run_parity_n8(
        output_path=tmp_path / "missing_experiment_1526.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        exp1515_path=tmp_path / "missing_exp1515.json",
        importer=lambda _name: (_ for _ in ()).throw(AssertionError("must not import")),
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )
    malformed_artifact = exp1526.run_parity_n8(
        output_path=tmp_path / "malformed_experiment_1526.json",
        manifest_path=tmp_path / "malformed_manifest.jsonl",
        exp1515_path=malformed_exp1515,
        importer=lambda _name: (_ for _ in ()).throw(AssertionError("must not import")),
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )

    assert missing_artifact["blockers"][0]["blocker"] == "exp1515_evidence_missing"
    assert malformed_artifact["blockers"][0]["blocker"] == "exp1515_evidence_malformed"


def test_missing_thrml_ising_api_writes_terminal_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-047: missing THRML Ising API does not become parity evidence."""

    exp1515_path = tmp_path / "exp1515.json"
    _write_exp1515(exp1515_path)

    missing_nodes = exp1526.run_parity_n8(
        output_path=tmp_path / "missing_nodes_1526.json",
        manifest_path=tmp_path / "missing_nodes.jsonl",
        exp1515_path=exp1515_path,
        importer=_fake_thrml_without_nodes,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )
    missing_energy = exp1526.run_parity_n8(
        output_path=tmp_path / "missing_energy_1526.json",
        manifest_path=tmp_path / "missing_energy.jsonl",
        exp1515_path=exp1515_path,
        importer=_fake_thrml_without_energy,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )
    missing_block = exp1526.run_parity_n8(
        output_path=tmp_path / "missing_block_1526.json",
        manifest_path=tmp_path / "missing_block.jsonl",
        exp1515_path=exp1515_path,
        importer=_fake_thrml_block_energy_without_block,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
    )

    assert missing_nodes["blockers"][0]["blocker"] == "thrml_ising_energy_api_unavailable"
    assert "SpinNode" in missing_nodes["blockers"][0]["detail"]
    assert missing_energy["blockers"][0]["blocker"] == "thrml_ising_energy_api_unavailable"
    assert "energy" in missing_energy["blockers"][0]["detail"]
    assert missing_block["blockers"][0]["blocker"] == "thrml_ising_energy_api_unavailable"
    assert "Block" in missing_block["blockers"][0]["detail"]


def test_validate_artifact_rejects_bad_claims_and_metrics(tmp_path: Path) -> None:
    """REQ-SAMPLE-047: schema validation enforces pass metrics and no-TSU gates."""

    artifact = exp1526.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "thrml_parity_n8_passed": True,
            "exact_states_enumerated": 256,
            "topology": "signed_ring_chord",
            "carnot_partition_function": 10.0,
            "thrml_partition_function": 10.0,
            "partition_relative_error": 0.0,
            "mean_energy_delta": 0.0,
            "kl_divergence": 0.0,
            "thresholds": {
                "partition_relative_error_max": 1e-8,
                "mean_energy_delta_abs_max": 1e-8,
                "kl_divergence_max": 1e-8,
            },
            "blockers": [],
            "honest_verdict": "complete_thrml_carnot_parity_n8_passed_no_tsu_hardware_claim",
        }
    )
    exp1526.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1526.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1526.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1526.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1526.validate_artifact(bad_tsu)

    bad_metric = dict(artifact, partition_relative_error=1e-4)
    with pytest.raises(ValueError, match="pass metrics"):
        exp1526.validate_artifact(bad_metric)

    bad_verdict = dict(artifact, honest_verdict="claimed hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1526.validate_artifact(bad_verdict)
