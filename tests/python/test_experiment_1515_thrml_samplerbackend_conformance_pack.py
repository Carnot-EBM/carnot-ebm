"""Tests for Exp 1515 THRML SamplerBackend conformance pack.

Spec refs: REQ-SAMPLE-046, SCENARIO-SAMPLE-074.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_samplerbackend_conformance_pack as exp1515


class FakeBackend:
    """Small deterministic backend that satisfies the SamplerBackend shape."""

    backend_name = "thrml_cpu_fallback"

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        del biases, couplings, config
        row = np.array([self.seed % 2 == 0, True, False, self.seed % 3 == 0])
        return np.tile(row, (int(n_samples), 1))

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        del biases, couplings, n_steps, beta
        row = np.array([True, self.seed % 2 == 1, True, False])
        return np.tile(row, (int(n_samples), 1))


def _write_gate(path: Path, *, ready: bool) -> None:
    path.write_text(json.dumps({"prior_thrml_parity_ready": ready}) + "\n", encoding="utf-8")


def _write_parity(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_import_ready": True,
                "simulator_only": True,
                "cases_compared": [
                    {
                        "case": "tiny_ising:n4_signed_ring_chord:exact_energy",
                        "type": "exact_enumerated_energy",
                        "passed": True,
                        "carnot_output": {"min_energy": -1.0},
                        "thrml_output": {"min_energy": -1.0},
                        "delta": 0.0,
                        "tolerance": 1.0e-6,
                    },
                    {
                        "case": "tiny_ising:n4_signed_ring_chord:sample_mean_energy",
                        "type": "fixed_seed_sample_mean_energy",
                        "passed": False,
                        "carnot_output": {"mean_energy": -0.5},
                        "thrml_output": {"mean_energy": -0.1},
                        "delta": 0.4,
                        "tolerance": 0.35,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _fake_thrml() -> SimpleNamespace:
    return SimpleNamespace(__version__="0.1.fake", __file__="/local/venv/thrml/__init__.py")


def test_spec_mentions_exp1515_contract() -> None:
    """REQ-SAMPLE-046, SCENARIO-SAMPLE-074: Exp1515 is spec-anchored."""

    spec = (exp1515.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-046" in spec
    assert "SCENARIO-SAMPLE-074" in spec
    assert "experiment_1515_thrml_samplerbackend_conformance_pack.json" in spec
    assert "thrml_samplerbackend_conformance_1515.jsonl" in spec


def test_write_in_progress_artifact_has_required_no_tsu_fields(tmp_path: Path) -> None:
    """REQ-SAMPLE-046: bootstrap artifact is simulator-only before probing."""

    output_path = tmp_path / "experiment_1515.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp1515.write_in_progress_artifact(output_path, manifest_path)

    assert artifact["status"] == "in_progress"
    assert artifact["thrml_samplerbackend_conformance_ready"] is False
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["conformance_manifest_path"] == str(manifest_path)
    assert artifact["honest_verdict"].startswith("success_")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_missing_prior_gate_blocks_without_importing_thrml(tmp_path: Path) -> None:
    """REQ-SAMPLE-046: Exp1506 prior parity gate must be present and true."""

    output_path = tmp_path / "experiment_1515.json"
    manifest_path = tmp_path / "manifest.jsonl"

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when the Exp1506 gate is closed")

    artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=tmp_path / "missing_exp1506.json",
        parity_path=tmp_path / "unused_exp1504.json",
        importer=importer,
        backend_factory=FakeBackend,
    )

    assert artifact["status"] == "blocked"
    assert artifact["gated_inputs_present"] is False
    assert artifact["thrml_import_ready"] is False
    assert artifact["thrml_samplerbackend_conformance_ready"] is False
    assert artifact["blockers"][0]["blocker"] == "prior_thrml_parity_gate_missing"
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert not manifest_path.exists()


def test_closed_or_malformed_prior_gate_blocks_before_import(tmp_path: Path) -> None:
    """REQ-SAMPLE-046: malformed or false Exp1506 gates stay terminal."""

    closed_gate = tmp_path / "closed_exp1506.json"
    malformed_gate = tmp_path / "malformed_exp1506.json"
    output_path = tmp_path / "experiment_1515.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_gate(closed_gate, ready=False)
    malformed_gate.write_text("{not json", encoding="utf-8")

    closed_artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=closed_gate,
        parity_path=tmp_path / "unused_exp1504.json",
        importer=lambda _name: _fake_thrml(),
        backend_factory=FakeBackend,
    )
    malformed_artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=malformed_gate,
        parity_path=tmp_path / "unused_exp1504.json",
        importer=lambda _name: _fake_thrml(),
        backend_factory=FakeBackend,
    )

    assert closed_artifact["blockers"][0]["blocker"] == "prior_thrml_parity_gate_closed"
    assert malformed_artifact["blockers"][0]["blocker"] == "prior_thrml_parity_gate_missing"
    assert "malformed JSON input" in malformed_artifact["blockers"][0]["detail"]


def test_import_failure_records_simulator_dependency_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-046: local THRML dependency failure is terminal data."""

    gate_path = tmp_path / "exp1506.json"
    output_path = tmp_path / "experiment_1515.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_gate(gate_path, ready=True)

    def importer(_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'thrml'")

    artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=gate_path,
        parity_path=tmp_path / "unused_exp1504.json",
        importer=importer,
        backend_factory=FakeBackend,
    )

    assert artifact["status"] == "blocked"
    assert artifact["gated_inputs_present"] is True
    assert artifact["thrml_import_ready"] is False
    assert artifact["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert "ModuleNotFoundError" in artifact["blockers"][0]["detail"]
    assert not manifest_path.exists()


def test_complete_pack_writes_manifest_rows_and_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-074: ready artifact requires rows, shapes, seed, and parity."""

    gate_path = tmp_path / "exp1506.json"
    parity_path = tmp_path / "exp1504.json"
    output_path = tmp_path / "experiment_1515.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_gate(gate_path, ready=True)
    _write_parity(parity_path)

    artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=gate_path,
        parity_path=parity_path,
        importer=lambda _name: _fake_thrml(),
        backend_factory=FakeBackend,
        seed=1515,
        n_samples=5,
        n_steps=7,
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["thrml_samplerbackend_conformance_ready"] is True
    assert artifact["gated_inputs_present"] is True
    assert artifact["thrml_import_ready"] is True
    assert artifact["parity_cases_passed"] == 1
    assert artifact["seed_reproducibility_checked"] is True
    assert len(rows) == len(artifact["conformance_cases"]) == 5
    assert {row["case_type"] for row in rows} == {
        "accepted_model_shape",
        "sample_shape_contract",
        "minimize_shape_contract",
        "seed_reproducibility",
        "carnot_thrml_parity_vector",
    }
    assert artifact["sample_shape_contracts"] == [
        {
            "method": "sample",
            "expected_shape": [5, 4],
            "observed_shape": [5, 4],
            "dtype": "bool",
            "passed": True,
        },
        {
            "method": "minimize_energy",
            "expected_shape": [5, 4],
            "observed_shape": [5, 4],
            "dtype": "bool",
            "passed": True,
        },
    ]
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_missing_or_failed_parity_vectors_block_readiness(tmp_path: Path) -> None:
    """REQ-SAMPLE-046: readiness requires at least one passed Exp1504 parity row."""

    gate_path = tmp_path / "exp1506.json"
    output_path = tmp_path / "experiment_1515.json"
    manifest_path = tmp_path / "manifest.jsonl"
    missing_parity = tmp_path / "missing_exp1504.json"
    failed_parity = tmp_path / "failed_exp1504.json"
    _write_gate(gate_path, ready=True)
    failed_parity.write_text(
        json.dumps({"cases_compared": [{"case": "x", "type": "y", "passed": False}]}) + "\n",
        encoding="utf-8",
    )

    missing_artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=gate_path,
        parity_path=missing_parity,
        importer=lambda _name: _fake_thrml(),
        backend_factory=FakeBackend,
    )
    failed_artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=gate_path,
        parity_path=failed_parity,
        importer=lambda _name: _fake_thrml(),
        backend_factory=FakeBackend,
    )

    assert missing_artifact["blockers"][0]["blocker"] == "prior_parity_vectors_missing"
    assert failed_artifact["blockers"][0]["blocker"] == "prior_parity_vectors_not_passed"


def test_adapter_exception_blocks_without_writing_partial_manifest(tmp_path: Path) -> None:
    """REQ-SAMPLE-046: adapter errors are blockers, not partial ready artifacts."""

    class BrokenBackend(FakeBackend):
        def sample(
            self,
            biases: np.ndarray,
            couplings: np.ndarray,
            n_samples: int,
            config: dict[str, Any],
        ) -> np.ndarray:
            del biases, couplings, n_samples, config
            raise RuntimeError("backend failed")

    gate_path = tmp_path / "exp1506.json"
    parity_path = tmp_path / "exp1504.json"
    output_path = tmp_path / "experiment_1515.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_gate(gate_path, ready=True)
    _write_parity(parity_path)

    artifact = exp1515.run_conformance_pack(
        output_path=output_path,
        manifest_path=manifest_path,
        gate_path=gate_path,
        parity_path=parity_path,
        importer=lambda _name: _fake_thrml(),
        backend_factory=BrokenBackend,
    )

    assert artifact["status"] == "blocked"
    assert artifact["blockers"][0]["blocker"] == "samplerbackend_conformance_failed"
    assert "RuntimeError" in artifact["blockers"][0]["detail"]
    assert not manifest_path.exists()


def test_validate_artifact_rejects_bad_schema_and_claims() -> None:
    """REQ-SAMPLE-046: schema validation enforces the no-TSU boundary."""

    valid = {
        "status": "complete",
        "thrml_samplerbackend_conformance_ready": True,
        "gated_inputs_present": True,
        "thrml_import_ready": True,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "conformance_cases": [{"case_id": "x", "passed": True}],
        "parity_cases_passed": 1,
        "sample_shape_contracts": [{"passed": True}],
        "seed_reproducibility_checked": True,
        "conformance_manifest_path": "manifest.jsonl",
        "blockers": [],
        "honest_verdict": "complete_thrml_samplerbackend_conformance_ready_no_tsu_hardware_claim",
    }

    exp1515.validate_artifact(valid)

    missing = valid.copy()
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1515.validate_artifact(missing)

    bad_status = valid.copy()
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="invalid status"):
        exp1515.validate_artifact(bad_status)

    bad_ready = valid.copy()
    bad_ready["sample_shape_contracts"] = []
    with pytest.raises(ValueError, match="ready artifact"):
        exp1515.validate_artifact(bad_ready)

    bad_tsu = valid.copy()
    bad_tsu["no_tsu_hardware_claim"] = False
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1515.validate_artifact(bad_tsu)

    bad_simulator = valid.copy()
    bad_simulator["simulator_only"] = False
    with pytest.raises(ValueError, match="simulator_only"):
        exp1515.validate_artifact(bad_simulator)

    bad_verdict = valid.copy()
    bad_verdict["honest_verdict"] = "ready"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1515.validate_artifact(bad_verdict)
