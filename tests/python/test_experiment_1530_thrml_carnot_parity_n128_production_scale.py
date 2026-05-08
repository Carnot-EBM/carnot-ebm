"""Tests for Exp 1530 sampled n=128 THRML/Carnot parity.

Spec refs: REQ-SAMPLE-051, SCENARIO-SAMPLE-079.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_carnot_parity_n128_production_scale as exp1530


class FakeBackend:
    """Deterministic SamplerBackend-shaped fake for bounded parity tests."""

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
        del couplings, config
        n_spins = int(np.asarray(biases).shape[0])
        base = np.array(
            [(idx + self.seed) % 3 == 0 for idx in range(n_spins)],
            dtype=bool,
        )
        if self.invert:
            base = np.logical_not(base)
        return np.vstack([np.roll(base, shift % n_spins) for shift in range(int(n_samples))])


def _fake_backend_factory(name: str, *, invert: bool = False) -> Any:
    def factory(seed: int) -> FakeBackend:
        return FakeBackend(seed=seed, backend_name=name, invert=invert)

    return factory


def _write_exp1529(path: Path, *, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_parity_n64_passed": passed,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "n_samples_per_backend": 10240,
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


def test_spec_mentions_exp1530_contract() -> None:
    """REQ-SAMPLE-051, SCENARIO-SAMPLE-079: Exp1530 is spec-anchored."""

    spec = (exp1530.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-051" in spec
    assert "SCENARIO-SAMPLE-079" in spec
    assert "experiment_1530_thrml_carnot_parity_n128_production_scale.json" in spec
    assert "thrml_carnot_parity_n128_1530.jsonl" in spec


def test_n128_case_is_deterministic_signed_ring_chord_production_scale() -> None:
    """REQ-SAMPLE-051: the n=128 case is deterministic and sample-only."""

    case = exp1530.n128_signed_ring_chord_case()
    repeated = exp1530.n128_signed_ring_chord_case()

    assert case.n_spins == 128
    assert case.name == "n128_signed_ring_chord"
    assert case.topology == "signed_ring_chord"
    assert case.beta == pytest.approx(1.0)
    assert exp1530.DEFAULT_SEEDS == (20260508, 20260509, 20260510, 20260511, 20260512)
    assert np.allclose(case.j_matrix, repeated.j_matrix)
    assert np.allclose(case.bias, repeated.bias)
    assert np.allclose(case.j_matrix, case.j_matrix.T)
    assert np.allclose(np.diag(case.j_matrix), 0.0)
    assert np.count_nonzero(np.triu(case.j_matrix, 1)) == 256


def test_sampled_backend_rows_and_summary_are_exp1530_distributional() -> None:
    """REQ-SAMPLE-051: sampled metrics include stability, percent delta, and KL."""

    case = exp1530.n128_signed_ring_chord_case()
    schedule = {"beta": case.beta, "n_warmup": 2, "steps_per_sample": 1, "use_checkerboard": True}
    base = np.array([(idx % 3) == 0 for idx in range(case.n_spins)], dtype=bool)
    samples = np.vstack([np.roll(base, shift) for shift in range(6)])

    carnot_row = exp1530.sampled_backend_row(
        case,
        seed=11,
        backend_label="carnot",
        backend_name="cpu",
        samples=samples,
        schedule=schedule,
    )
    thrml_row = exp1530.sampled_backend_row(
        case,
        seed=11,
        backend_label="thrml",
        backend_name="thrml_cpu_fallback",
        samples=samples,
        schedule=schedule,
    )
    summary = exp1530.summarize_sampled_rows(
        [carnot_row, thrml_row],
        seeds=(11,),
        thresholds={**exp1530.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )

    assert carnot_row["case_id"] == "exp1530:n128_signed_ring_chord:seed_11:carnot"
    assert carnot_row["case_type"] == "sampled_seed_backend"
    assert carnot_row["sample_count"] == 6
    assert summary["case_id"] == "exp1530:n128_signed_ring_chord:sampled_summary"
    assert summary["case_type"] == "sampled_distribution_summary"
    assert summary["n_samples_per_backend"] == 6
    assert summary["mean_energy_delta"] == pytest.approx(0.0)
    assert summary["mean_energy_delta_percent"] == pytest.approx(0.0)
    assert summary["magnetization_delta"] == pytest.approx(0.0)
    assert summary["kl_divergence"] == pytest.approx(0.0)
    assert summary["stability_diagnostics_present"] is True
    assert summary["passed_thresholds"] is True


def test_write_in_progress_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-SAMPLE-051: bootstrap artifact is written before parity execution."""

    output_path = tmp_path / "experiment_1530.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp1530.write_in_progress_artifact(output_path, manifest_path)

    assert artifact["status"] == "in_progress"
    assert artifact["thrml_parity_n128_passed"] is False
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["n_spins"] == 128
    assert artifact["seeds"] == list(exp1530.DEFAULT_SEEDS)
    assert artifact["n_samples_per_backend"] == 0
    assert artifact["runtime_seconds_by_backend"] == {}
    assert artifact["memory_summary"] == {}
    assert artifact["parity_manifest_path"] == str(manifest_path)
    assert artifact["honest_verdict"].startswith("success_")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_complete_run_writes_seed_backend_rows_summary_runtime_and_memory(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-079: n=128 sampled metrics and JSONL evidence are written."""

    exp1529_path = tmp_path / "exp1529.json"
    output_path = tmp_path / "experiment_1530.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1529(exp1529_path)

    artifact = exp1530.run_parity_n128(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1529_path=exp1529_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11, 12),
        sample_count_per_seed=4,
        n_warmup=2,
        steps_per_sample=1,
        thresholds={**exp1530.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["thrml_parity_n128_passed"] is True
    assert artifact["n_spins"] == 128
    assert artifact["seeds"] == [11, 12]
    assert artifact["n_samples_per_backend"] == 8
    assert artifact["mean_energy_delta"] == pytest.approx(0.0)
    assert artifact["mean_energy_delta_percent"] == pytest.approx(0.0)
    assert artifact["magnetization_delta"] == pytest.approx(0.0)
    assert artifact["kl_divergence"] == pytest.approx(0.0)
    assert artifact["runtime_seconds_by_backend"]["carnot"] >= 0.0
    assert artifact["runtime_seconds_by_backend"]["thrml"] >= 0.0
    assert artifact["memory_summary"]["tracemalloc_peak_bytes"] >= 0
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert [row["case_type"] for row in rows].count("sampled_seed_backend") == 4
    assert rows[-1]["case_type"] == "sampled_distribution_summary"
    assert rows[-1]["passed_thresholds"] is True
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_threshold_failure_is_complete_not_passed(tmp_path: Path) -> None:
    """REQ-SAMPLE-051: n=128 pass is false when sampled thresholds fail."""

    exp1529_path = tmp_path / "exp1529.json"
    _write_exp1529(exp1529_path)

    artifact = exp1530.run_parity_n128(
        output_path=tmp_path / "failed_1530.json",
        manifest_path=tmp_path / "failed_manifest.jsonl",
        exp1529_path=exp1529_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback", invert=True),
        seeds=(11,),
        sample_count_per_seed=4,
        n_warmup=2,
        steps_per_sample=1,
        thresholds={
            **exp1530.THRESHOLDS,
            "mean_energy_delta_abs_max": 0.0,
            "mean_energy_delta_percent_max": 0.0,
            "magnetization_delta_abs_max": 0.0,
            "kl_divergence_max": 0.0,
            "kl_min_samples_per_backend": 1,
        },
        energy_bin_count=4,
    )

    assert artifact["status"] == "complete"
    assert artifact["thrml_parity_n128_passed"] is False
    assert artifact["blockers"][0]["blocker"] == "sampled_parity_threshold_failed"
    assert artifact["honest_verdict"].startswith("complete_")


def test_upstream_exp1529_evidence_blocks_before_thrml_import(tmp_path: Path) -> None:
    """REQ-SAMPLE-051: Exp1529 passed parity evidence gates the n=128 run."""

    malformed_exp1529 = tmp_path / "malformed_exp1529.json"
    not_passed_exp1529 = tmp_path / "not_passed_exp1529.json"
    malformed_exp1529.write_text("{not-json", encoding="utf-8")
    _write_exp1529(not_passed_exp1529, passed=False)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when Exp1529 is not ready")

    missing_artifact = exp1530.run_parity_n128(
        output_path=tmp_path / "missing_1530.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        exp1529_path=tmp_path / "missing_exp1529.json",
        importer=importer,
    )
    malformed_artifact = exp1530.run_parity_n128(
        output_path=tmp_path / "malformed_1530.json",
        manifest_path=tmp_path / "malformed_manifest.jsonl",
        exp1529_path=malformed_exp1529,
        importer=importer,
    )
    not_passed_artifact = exp1530.run_parity_n128(
        output_path=tmp_path / "not_passed_1530.json",
        manifest_path=tmp_path / "not_passed_manifest.jsonl",
        exp1529_path=not_passed_exp1529,
        importer=importer,
    )

    assert missing_artifact["blockers"][0]["blocker"] == "exp1529_evidence_missing"
    assert malformed_artifact["blockers"][0]["blocker"] == "exp1529_evidence_malformed"
    assert not_passed_artifact["blockers"][0]["blocker"] == "exp1529_parity_not_passed"
    assert missing_artifact["simulator_only"] is True
    assert missing_artifact["no_tsu_hardware_claim"] is True


def test_missing_thrml_import_writes_terminal_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-051: unavailable THRML remains terminal data, not a fake pass."""

    exp1529_path = tmp_path / "exp1529.json"
    _write_exp1529(exp1529_path)

    artifact = exp1530.run_parity_n128(
        output_path=tmp_path / "missing_import_1530.json",
        manifest_path=tmp_path / "missing_import.jsonl",
        exp1529_path=exp1529_path,
        importer=_missing_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11,),
        sample_count_per_seed=2,
        thresholds={**exp1530.THRESHOLDS, "kl_min_samples_per_backend": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["thrml_parity_n128_passed"] is False
    assert artifact["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert artifact["n_samples_per_backend"] == 0
    assert not (tmp_path / "missing_import.jsonl").exists()


def test_validate_artifact_rejects_bad_claims_metrics_and_diagnostics(tmp_path: Path) -> None:
    """REQ-SAMPLE-051: schema validation enforces sampled metrics and no-TSU gates."""

    artifact = exp1530.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "thrml_parity_n128_passed": True,
            "seeds": [11],
            "n_samples_per_backend": 4,
            "mean_energy_delta": 0.0,
            "mean_energy_delta_percent": 0.0,
            "magnetization_delta": 0.0,
            "autocorrelation_summary": {
                "carnot_energy_lag1_mean": 0.0,
                "thrml_energy_lag1_mean": 0.0,
                "lag1_delta": 0.0,
            },
            "kl_divergence": 0.0,
            "thresholds": {
                "mean_energy_delta_abs_max": 1e-8,
                "mean_energy_delta_percent_max": 1e-8,
                "magnetization_delta_abs_max": 1e-8,
                "kl_divergence_max": 1e-8,
                "kl_min_samples_per_backend": 1,
                "autocorrelation_lag1_delta_abs_max": 1e-8,
            },
            "kl_estimate_stable": True,
            "stability_diagnostics_present": True,
            "runtime_seconds_by_backend": {"carnot": 0.1, "thrml": 0.2},
            "memory_summary": {
                "method": "python_tracemalloc_resource",
                "rss_max_kib_start": 1,
                "rss_max_kib_end": 2,
                "tracemalloc_peak_bytes": 3,
            },
            "blockers": [],
            "honest_verdict": "complete_thrml_carnot_parity_n128_passed_no_tsu_hardware_claim",
        }
    )
    exp1530.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("runtime_seconds_by_backend")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1530.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1530.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1530.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1530.validate_artifact(bad_tsu)

    bad_n = dict(artifact, n_spins=64)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1530.validate_artifact(bad_n)

    bad_energy = dict(artifact, mean_energy_delta=1e-4, mean_energy_delta_percent=1e-4)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1530.validate_artifact(bad_energy)

    bad_runtime = dict(artifact, runtime_seconds_by_backend={"carnot": 0.1})
    with pytest.raises(ValueError, match="runtime diagnostics"):
        exp1530.validate_artifact(bad_runtime)

    bad_memory = dict(artifact, memory_summary={})
    with pytest.raises(ValueError, match="memory diagnostics"):
        exp1530.validate_artifact(bad_memory)

    bad_verdict = dict(artifact, honest_verdict="claimed hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1530.validate_artifact(bad_verdict)
