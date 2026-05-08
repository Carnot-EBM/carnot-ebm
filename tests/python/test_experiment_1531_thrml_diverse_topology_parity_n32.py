"""Tests for Exp 1531 n=32 diverse-topology THRML/Carnot parity.

Spec refs: REQ-SAMPLE-052, SCENARIO-SAMPLE-080.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_diverse_topology_parity_n32 as exp1531


class FakeBackend:
    """Deterministic SamplerBackend-shaped fake for topology parity rows."""

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
        del couplings, config
        n_spins = int(np.asarray(biases).shape[0])
        base = np.array([(idx + self.seed) % 2 == 0 for idx in range(n_spins)], dtype=bool)
        return np.vstack([np.roll(base, shift % n_spins) for shift in range(int(n_samples))])


def _fake_backend_factory(name: str) -> Any:
    def factory(seed: int) -> FakeBackend:
        return FakeBackend(seed=seed, backend_name=name)

    return factory


def _write_exp1528(path: Path, *, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_parity_n32_passed": passed,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "n_spins": 32,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _fake_thrml_import(name: str) -> Any:
    if name == "thrml":
        return SimpleNamespace(__version__="fake-0.1", __file__="/fake/thrml/__init__.py")
    if name == "thrml.models":
        return SimpleNamespace()
    raise ModuleNotFoundError(name)


def _missing_thrml_import(name: str) -> Any:
    raise ModuleNotFoundError(name)


def _edge_count(case: exp1531.ParityIsingCase) -> int:
    return int(np.count_nonzero(np.triu(case.j_matrix, 1)))


def test_spec_mentions_exp1531_contract() -> None:
    """REQ-SAMPLE-052, SCENARIO-SAMPLE-080: Exp1531 is spec-anchored."""

    spec = (exp1531.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-052" in spec
    assert "SCENARIO-SAMPLE-080" in spec
    assert "experiment_1531_thrml_diverse_topology_parity_n32.json" in spec
    assert "thrml_diverse_topology_parity_n32_1531.jsonl" in spec
    for topology in ("complete", "sparse random", "lattice", "scale-free"):
        assert topology in spec


def test_n32_topology_cases_are_deterministic_and_diverse() -> None:
    """REQ-SAMPLE-052: four deterministic n=32 topology families are generated."""

    cases = exp1531.n32_diverse_topology_cases()
    repeated = exp1531.n32_diverse_topology_cases()
    edge_counts = {case.topology: _edge_count(case) for case in cases}

    assert [case.topology for case in cases] == [
        "complete",
        "sparse_random",
        "lattice",
        "scale_free",
    ]
    assert edge_counts == {
        "complete": 496,
        "sparse_random": 80,
        "lattice": 64,
        "scale_free": 61,
    }
    for case, repeated_case in zip(cases, repeated, strict=True):
        assert case.n_spins == 32
        assert case.beta == pytest.approx(1.10)
        assert np.allclose(case.j_matrix, repeated_case.j_matrix)
        assert np.allclose(case.bias, repeated_case.bias)
        assert np.allclose(case.j_matrix, case.j_matrix.T)
        assert np.allclose(np.diag(case.j_matrix), 0.0)
        assert case.bias.shape == (32,)


def test_topology_rows_aggregate_by_family_and_gate_three_of_four() -> None:
    """REQ-SAMPLE-052: manifest aggregation records per-topology pass/fail metrics."""

    rows: list[dict[str, Any]] = []
    schedule = {"beta": 1.10, "n_warmup": 2, "steps_per_sample": 1, "use_checkerboard": True}
    thresholds = {**exp1531.THRESHOLDS, "kl_min_samples_per_backend": 1}
    for case in exp1531.n32_diverse_topology_cases():
        samples = np.zeros((4, case.n_spins), dtype=bool)
        shifted = samples if case.topology != "scale_free" else np.ones((4, case.n_spins), dtype=bool)
        rows.append(
            exp1531.sampled_topology_backend_row(
                case,
                seed=11,
                backend_label="carnot",
                backend_name="cpu",
                samples=samples,
                schedule=schedule,
            )
        )
        rows.append(
            exp1531.sampled_topology_backend_row(
                case,
                seed=11,
                backend_label="thrml",
                backend_name="thrml_cpu_fallback",
                samples=shifted,
                schedule=schedule,
            )
        )

    summary = exp1531.summarize_diverse_topology_rows(
        rows,
        topologies=("complete", "sparse_random", "lattice", "scale_free"),
        seeds=(11,),
        thresholds=thresholds,
        energy_bin_count=4,
    )

    assert summary["case_type"] == "diverse_topology_summary"
    assert summary["topologies_tested"] == ["complete", "sparse_random", "lattice", "scale_free"]
    assert summary["topologies_passed"] == ["complete", "sparse_random", "lattice"]
    assert summary["diverse_topology_parity_ready"] is True
    assert summary["topology_results"]["scale_free"]["passed_thresholds"] is False
    assert set(summary["mean_energy_delta_by_topology"]) == set(summary["topologies_tested"])
    assert set(summary["kl_divergence_by_topology"]) == set(summary["topologies_tested"])
    assert summary["simulator_only"] is True
    assert summary["no_tsu_hardware_claim"] is True


def test_complete_run_writes_topology_backend_rows_and_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-080: diverse topology JSONL evidence and artifact are written."""

    exp1528_path = tmp_path / "exp1528.json"
    output_path = tmp_path / "experiment_1531.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1528(exp1528_path)

    artifact = exp1531.run_diverse_topology_parity_n32(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1528_path=exp1528_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11,),
        sample_count_per_seed=4,
        n_warmup=2,
        steps_per_sample=1,
        thresholds={**exp1531.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["diverse_topology_parity_ready"] is True
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["n_spins"] == 32
    assert artifact["topologies_tested"] == ["complete", "sparse_random", "lattice", "scale_free"]
    assert artifact["topologies_passed"] == artifact["topologies_tested"]
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert [row["case_type"] for row in rows].count("sampled_topology_seed_backend") == 8
    assert rows[-1]["case_type"] == "diverse_topology_summary"
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_upstream_exp1528_and_thrml_import_blockers_are_terminal(tmp_path: Path) -> None:
    """REQ-SAMPLE-052: blockers are recorded before fake parity can pass."""

    malformed_exp1528 = tmp_path / "malformed_exp1528.json"
    not_passed_exp1528 = tmp_path / "not_passed_exp1528.json"
    ready_exp1528 = tmp_path / "ready_exp1528.json"
    malformed_exp1528.write_text("{not-json", encoding="utf-8")
    _write_exp1528(not_passed_exp1528, passed=False)
    _write_exp1528(ready_exp1528)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when Exp1528 is not ready")

    missing_artifact = exp1531.run_diverse_topology_parity_n32(
        output_path=tmp_path / "missing_1531.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        exp1528_path=tmp_path / "missing_exp1528.json",
        importer=importer,
    )
    malformed_artifact = exp1531.run_diverse_topology_parity_n32(
        output_path=tmp_path / "malformed_1531.json",
        manifest_path=tmp_path / "malformed_manifest.jsonl",
        exp1528_path=malformed_exp1528,
        importer=importer,
    )
    not_passed_artifact = exp1531.run_diverse_topology_parity_n32(
        output_path=tmp_path / "not_passed_1531.json",
        manifest_path=tmp_path / "not_passed_manifest.jsonl",
        exp1528_path=not_passed_exp1528,
        importer=importer,
    )
    import_blocked_artifact = exp1531.run_diverse_topology_parity_n32(
        output_path=tmp_path / "import_blocked_1531.json",
        manifest_path=tmp_path / "import_blocked_manifest.jsonl",
        exp1528_path=ready_exp1528,
        importer=_missing_thrml_import,
        thresholds={**exp1531.THRESHOLDS, "kl_min_samples_per_backend": 1},
    )

    assert missing_artifact["blockers"][0]["blocker"] == "exp1528_evidence_missing"
    assert malformed_artifact["blockers"][0]["blocker"] == "exp1528_evidence_malformed"
    assert not_passed_artifact["blockers"][0]["blocker"] == "exp1528_parity_not_passed"
    assert import_blocked_artifact["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert import_blocked_artifact["status"] == "blocked"
    assert import_blocked_artifact["simulator_only"] is True
    assert import_blocked_artifact["no_tsu_hardware_claim"] is True
    assert not (tmp_path / "import_blocked_manifest.jsonl").exists()


def test_validate_artifact_rejects_bad_claims_and_metrics(tmp_path: Path) -> None:
    """REQ-SAMPLE-052: readiness validation enforces metrics and no-TSU gates."""

    artifact = exp1531.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "diverse_topology_parity_ready": True,
            "topologies_tested": list(exp1531.TOPOLOGIES),
            "topologies_passed": list(exp1531.TOPOLOGIES[:3]),
            "topology_results": {
                topology: {
                    "passed_thresholds": True,
                    "mean_energy_delta": 0.0,
                    "magnetization_delta": 0.0,
                    "kl_divergence": 0.0,
                    "kl_estimate_stable": True,
                }
                for topology in exp1531.TOPOLOGIES
            },
            "mean_energy_delta_by_topology": {topology: 0.0 for topology in exp1531.TOPOLOGIES},
            "kl_divergence_by_topology": {topology: 0.0 for topology in exp1531.TOPOLOGIES},
            "thresholds": {
                "mean_energy_delta_abs_max": 1e-8,
                "magnetization_delta_abs_max": 1e-8,
                "kl_divergence_max": 1e-8,
                "kl_min_samples_per_backend": 1,
            },
            "blockers": [],
            "honest_verdict": "complete_thrml_diverse_topology_parity_n32_passed_no_tsu_hardware_claim",
        }
    )
    exp1531.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1531.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1531.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1531.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1531.validate_artifact(bad_tsu)

    bad_n = dict(artifact, n_spins=64)
    with pytest.raises(ValueError, match="n_spins=32"):
        exp1531.validate_artifact(bad_n)

    bad_verdict = dict(artifact, honest_verdict="claimed hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1531.validate_artifact(bad_verdict)

    bad_topology_count = dict(artifact, topologies_passed=["complete", "sparse_random"])
    with pytest.raises(ValueError, match="at least three passing topologies"):
        exp1531.validate_artifact(bad_topology_count)

    bad_metric = dict(artifact)
    bad_metric["topology_results"] = dict(artifact["topology_results"])
    bad_metric["topology_results"]["complete"] = dict(bad_metric["topology_results"]["complete"])
    bad_metric["topology_results"]["complete"]["kl_estimate_stable"] = False
    with pytest.raises(ValueError, match="at least three passing topologies"):
        exp1531.validate_artifact(bad_metric)
