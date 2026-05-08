"""Tests for Exp 1544 n=64 diverse-topology THRML/Carnot parity.

Spec refs: REQ-SAMPLE-054, SCENARIO-SAMPLE-082.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_diverse_topology_parity_n64 as exp1544


class FakeBackend:
    """Deterministic SamplerBackend-shaped fake for n=64 topology parity rows."""

    def __init__(self, seed: int, backend_name: str, invert: bool = False) -> None:
        self.seed = int(seed)
        self.backend_name = backend_name
        self.invert = bool(invert)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        del config
        n_spins = int(np.asarray(biases).shape[0])
        offset = int(round(float(np.sum(np.abs(couplings))) * 1000.0)) % n_spins
        base = np.array(
            [((idx + self.seed + offset) % 5) in (0, 2) for idx in range(n_spins)],
            dtype=bool,
        )
        if self.invert:
            base = np.logical_not(base)
        return np.vstack([np.roll(base, shift % n_spins) for shift in range(int(n_samples))])


def _fake_backend_factory(name: str, *, invert: bool = False) -> Any:
    def factory(seed: int) -> FakeBackend:
        return FakeBackend(seed=seed, backend_name=name, invert=invert)

    return factory


def _write_exp1531(path: Path, *, ready: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "diverse_topology_parity_ready": ready,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "n_spins": 32,
                "thresholds": {
                    "mean_energy_delta_abs_max": 0.15,
                    "magnetization_delta_abs_max": 0.025,
                    "kl_divergence_max": 0.05,
                    "kl_min_samples_per_backend": 10000,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_exp1543(path: Path, *, ready: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_parity_n256_schedule_ready": ready,
                "parity_passed": ready,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "n_spins": 256,
                "thresholds": {
                    "mean_energy_delta_abs_max": 0.6,
                    "max_energy_delta_abs_max": 0.6,
                    "magnetization_delta_abs_max": 0.035,
                    "kl_divergence_max": 0.15,
                    "kl_min_samples_per_backend": 10000,
                    "autocorrelation_lag1_delta_abs_max": 0.15,
                },
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


def _edge_count(case: exp1544.ParityIsingCase) -> int:
    return int(np.count_nonzero(np.triu(case.j_matrix, 1)))


def test_spec_mentions_exp1544_contract() -> None:
    """REQ-SAMPLE-054, SCENARIO-SAMPLE-082: Exp1544 is spec-anchored."""

    spec = (exp1544.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-054" in spec
    assert "SCENARIO-SAMPLE-082" in spec
    assert "experiment_1544_thrml_diverse_topology_parity_n64.json" in spec
    assert "diverse_topology_parity_n64_ready" in spec
    assert "complete, sparse random, lattice, and scale-free" in spec


def test_n64_topology_cases_are_deterministic_and_diverse() -> None:
    """REQ-SAMPLE-054: four deterministic n=64 topology families are generated."""

    cases = exp1544.n64_diverse_topology_cases()
    repeated = exp1544.n64_diverse_topology_cases()
    edge_counts = {case.topology: _edge_count(case) for case in cases}

    assert [case.topology for case in cases] == [
        "complete",
        "sparse_random",
        "lattice",
        "scale_free",
    ]
    assert edge_counts == {
        "complete": 2016,
        "sparse_random": 192,
        "lattice": 128,
        "scale_free": 125,
    }
    for case, repeated_case in zip(cases, repeated, strict=True):
        assert case.n_spins == 64
        assert case.beta == pytest.approx(1.05)
        assert np.allclose(case.j_matrix, repeated_case.j_matrix)
        assert np.allclose(case.bias, repeated_case.bias)
        assert np.allclose(case.j_matrix, case.j_matrix.T)
        assert np.allclose(np.diag(case.j_matrix), 0.0)
        assert case.bias.shape == (64,)


def test_topology_rows_compute_per_topology_and_aggregate_metrics() -> None:
    """REQ-SAMPLE-054: manifest aggregation records per-topology pass/fail metrics."""

    rows: list[dict[str, Any]] = []
    schedule = {"beta": 1.05, "n_warmup": 2, "steps_per_sample": 1, "use_checkerboard": True}
    thresholds = {**exp1544.THRESHOLDS, "kl_min_samples_per_backend": 1}
    for case in exp1544.n64_diverse_topology_cases():
        samples = np.zeros((4, case.n_spins), dtype=bool)
        rows.append(
            exp1544.sampled_topology_backend_row(
                case,
                seed=11,
                backend_label="carnot",
                backend_name="cpu",
                samples=samples,
                schedule=schedule,
            )
        )
        rows.append(
            exp1544.sampled_topology_backend_row(
                case,
                seed=11,
                backend_label="thrml",
                backend_name="thrml_cpu_fallback",
                samples=samples,
                schedule=schedule,
            )
        )

    summary = exp1544.summarize_diverse_topology_rows(
        rows,
        topologies=("complete", "sparse_random", "lattice", "scale_free"),
        seeds=(11,),
        thresholds=thresholds,
        energy_bin_count=4,
    )

    assert summary["case_type"] == "diverse_topology_n64_summary"
    assert summary["topologies_tested"] == ["complete", "sparse_random", "lattice", "scale_free"]
    assert summary["topologies_passed"] == summary["topologies_tested"]
    assert set(summary["per_topology_results"]) == set(summary["topologies_tested"])
    assert summary["mean_energy_delta"] == pytest.approx(0.0)
    assert summary["max_energy_delta"] == pytest.approx(0.0)
    assert summary["kl_divergence"] == pytest.approx(0.0)
    assert summary["parity_passed"] is True
    assert summary["diverse_topology_parity_n64_ready"] is True
    assert summary["simulator_only"] is True
    assert summary["no_tsu_hardware_claim"] is True


def test_complete_run_writes_required_artifact_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-082: complete n=64 topology evidence is written."""

    exp1531_path = tmp_path / "exp1531.json"
    exp1543_path = tmp_path / "exp1543.json"
    output_path = tmp_path / "experiment_1544.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1531(exp1531_path)
    _write_exp1543(exp1543_path)

    artifact = exp1544.run_diverse_topology_parity_n64(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1531_path=exp1531_path,
        exp1543_path=exp1543_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11,),
        sample_count_per_seed=4,
        n_warmup=2,
        steps_per_sample=1,
        thresholds={**exp1544.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
        thrml_seed_offset=0,
        focused_tests_passed=True,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.118"
    assert artifact["diverse_topology_parity_n64_ready"] is True
    assert artifact["n_spins"] == 64
    assert artifact["topologies_tested"] == ["complete", "sparse_random", "lattice", "scale_free"]
    assert artifact["per_topology_results"].keys() == artifact["topology_results"].keys()
    assert artifact["mean_energy_delta"] == pytest.approx(0.0)
    assert artifact["max_energy_delta"] == pytest.approx(0.0)
    assert artifact["kl_divergence"] == pytest.approx(0.0)
    assert artifact["parity_passed"] is True
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["parity_report_path"] == str(manifest_path)
    assert artifact["focused_tests_passed"] is True
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert [row["case_type"] for row in rows].count("sampled_topology_seed_backend") == 8
    assert rows[-1]["case_type"] == "diverse_topology_n64_summary"
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_upstream_and_thrml_import_blockers_are_terminal(tmp_path: Path) -> None:
    """REQ-SAMPLE-054: prior artifacts and THRML import gate fake passes."""

    malformed_exp1531 = tmp_path / "malformed_exp1531.json"
    not_ready_exp1531 = tmp_path / "not_ready_exp1531.json"
    malformed_exp1543 = tmp_path / "malformed_exp1543.json"
    not_ready_exp1543 = tmp_path / "not_ready_exp1543.json"
    ready_exp1531 = tmp_path / "ready_exp1531.json"
    ready_exp1543 = tmp_path / "ready_exp1543.json"
    malformed_exp1531.write_text("{not-json", encoding="utf-8")
    _write_exp1531(not_ready_exp1531, ready=False)
    malformed_exp1543.write_text("{not-json", encoding="utf-8")
    _write_exp1543(not_ready_exp1543, ready=False)
    _write_exp1531(ready_exp1531)
    _write_exp1543(ready_exp1543)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when prior parity is not ready")

    missing_1531 = exp1544.run_diverse_topology_parity_n64(
        output_path=tmp_path / "missing_1531.json",
        manifest_path=tmp_path / "missing_1531.jsonl",
        exp1531_path=tmp_path / "missing_exp1531_source.json",
        exp1543_path=ready_exp1543,
        importer=importer,
    )
    malformed_1531 = exp1544.run_diverse_topology_parity_n64(
        output_path=tmp_path / "malformed_1531.json",
        manifest_path=tmp_path / "malformed_1531.jsonl",
        exp1531_path=malformed_exp1531,
        exp1543_path=ready_exp1543,
        importer=importer,
    )
    not_ready_1531 = exp1544.run_diverse_topology_parity_n64(
        output_path=tmp_path / "not_ready_1531.json",
        manifest_path=tmp_path / "not_ready_1531.jsonl",
        exp1531_path=not_ready_exp1531,
        exp1543_path=ready_exp1543,
        importer=importer,
    )
    missing_1543 = exp1544.run_diverse_topology_parity_n64(
        output_path=tmp_path / "missing_1543.json",
        manifest_path=tmp_path / "missing_1543.jsonl",
        exp1531_path=ready_exp1531,
        exp1543_path=tmp_path / "missing_exp1543_source.json",
        importer=importer,
    )
    malformed_1543 = exp1544.run_diverse_topology_parity_n64(
        output_path=tmp_path / "malformed_1543.json",
        manifest_path=tmp_path / "malformed_1543.jsonl",
        exp1531_path=ready_exp1531,
        exp1543_path=malformed_exp1543,
        importer=importer,
    )
    not_ready_1543 = exp1544.run_diverse_topology_parity_n64(
        output_path=tmp_path / "not_ready_1543.json",
        manifest_path=tmp_path / "not_ready_1543.jsonl",
        exp1531_path=ready_exp1531,
        exp1543_path=not_ready_exp1543,
        importer=importer,
    )
    import_blocked = exp1544.run_diverse_topology_parity_n64(
        output_path=tmp_path / "import_blocked.json",
        manifest_path=tmp_path / "import_blocked.jsonl",
        exp1531_path=ready_exp1531,
        exp1543_path=ready_exp1543,
        importer=_missing_thrml_import,
        thresholds={**exp1544.THRESHOLDS, "kl_min_samples_per_backend": 1},
    )

    assert missing_1531["blockers"][0]["blocker"] == "exp1531_evidence_missing"
    assert malformed_1531["blockers"][0]["blocker"] == "exp1531_evidence_malformed"
    assert not_ready_1531["blockers"][0]["blocker"] == "exp1531_parity_not_ready"
    assert missing_1543["blockers"][0]["blocker"] == "exp1543_evidence_missing"
    assert malformed_1543["blockers"][0]["blocker"] == "exp1543_evidence_malformed"
    assert not_ready_1543["blockers"][0]["blocker"] == "exp1543_parity_not_ready"
    assert import_blocked["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert import_blocked["status"] == "blocked"
    assert import_blocked["diverse_topology_parity_n64_ready"] is False
    assert import_blocked["simulator_only"] is True
    assert import_blocked["no_tsu_hardware_claim"] is True


def test_validate_artifact_rejects_bad_claims_metrics_and_verdict(tmp_path: Path) -> None:
    """REQ-SAMPLE-054: artifact validation enforces metrics and no-TSU gates."""

    artifact = exp1544.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "diverse_topology_parity_n64_ready": True,
            "topologies_tested": list(exp1544.TOPOLOGIES),
            "topologies_passed": list(exp1544.TOPOLOGIES),
            "per_topology_results": {
                topology: {
                    "passed_thresholds": True,
                    "mean_energy_delta": 0.0,
                    "mean_energy_delta_percent": 0.0,
                    "magnetization_delta": 0.0,
                    "kl_divergence": 0.0,
                    "kl_estimate_stable": True,
                    "stability_diagnostics_present": True,
                    "n_samples_per_backend": 4,
                    "autocorrelation_summary": {
                        "carnot_energy_lag1_mean": 0.0,
                        "thrml_energy_lag1_mean": 0.0,
                        "lag1_delta": 0.0,
                    },
                }
                for topology in exp1544.TOPOLOGIES
            },
            "topology_results": {},
            "mean_energy_delta": 0.0,
            "max_energy_delta": 0.0,
            "kl_divergence": 0.0,
            "parity_passed": True,
            "focused_tests_passed": True,
            "thresholds": {**exp1544.THRESHOLDS, "kl_min_samples_per_backend": 1},
            "blockers": [],
            "honest_verdict": (
                "complete_thrml_diverse_topology_parity_n64_passed_simulator_only_"
                "no_tsu_hardware_claim"
            ),
        }
    )
    artifact["topology_results"] = artifact["per_topology_results"]
    exp1544.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("per_topology_results")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1544.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1544.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1544.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1544.validate_artifact(bad_tsu)

    bad_n = dict(artifact, n_spins=32)
    with pytest.raises(ValueError, match="n_spins=64"):
        exp1544.validate_artifact(bad_n)

    bad_verdict = dict(artifact, honest_verdict="claims hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1544.validate_artifact(bad_verdict)

    bad_metric = dict(artifact, kl_divergence=1.0)
    with pytest.raises(ValueError, match="n=64 readiness"):
        exp1544.validate_artifact(bad_metric)

    bad_topology_count = dict(artifact, topologies_passed=["complete", "sparse_random"])
    with pytest.raises(ValueError, match="n=64 readiness"):
        exp1544.validate_artifact(bad_topology_count)
