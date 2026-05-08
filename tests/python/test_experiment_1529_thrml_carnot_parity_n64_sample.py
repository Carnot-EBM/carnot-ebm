"""Tests for Exp 1529 sampled n=64 THRML/Carnot parity.

Spec refs: REQ-SAMPLE-050, SCENARIO-SAMPLE-078.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_carnot_parity_n64_sample as exp1529


class FakeBackend:
    """Deterministic SamplerBackend-shaped fake for repeated-chain sample rows."""

    def __init__(self, seed: int, backend_name: str, offset: int = 0) -> None:
        self.seed = int(seed)
        self.backend_name = backend_name
        self.offset = int(offset)

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
            [(idx + self.seed + self.offset) % 3 == 0 for idx in range(n_spins)],
            dtype=bool,
        )
        return np.vstack([np.roll(base, shift % n_spins) for shift in range(int(n_samples))])


def _fake_backend_factory(name: str, *, offset: int = 0) -> Any:
    def factory(seed: int) -> FakeBackend:
        return FakeBackend(seed=seed, backend_name=name, offset=offset)

    return factory


def _write_exp1528(path: Path, *, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_parity_n32_passed": passed,
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


def test_spec_mentions_exp1529_contract() -> None:
    """REQ-SAMPLE-050, SCENARIO-SAMPLE-078: Exp1529 is spec-anchored."""

    spec = (exp1529.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-050" in spec
    assert "SCENARIO-SAMPLE-078" in spec
    assert "experiment_1529_thrml_carnot_parity_n64_sample.json" in spec
    assert "thrml_carnot_parity_n64_1529.jsonl" in spec


def test_n64_case_is_deterministic_signed_ring_chord_sample_only() -> None:
    """REQ-SAMPLE-050: the n=64 case is deterministic and sample-only."""

    case = exp1529.n64_signed_ring_chord_case()
    repeated = exp1529.n64_signed_ring_chord_case()

    assert case.n_spins == 64
    assert case.name == "n64_signed_ring_chord"
    assert case.topology == "signed_ring_chord"
    assert case.beta == pytest.approx(1.05)
    assert exp1529.DEFAULT_SEEDS == (20260508, 20260509, 20260510, 20260511, 20260512)
    assert np.allclose(case.j_matrix, repeated.j_matrix)
    assert np.allclose(case.bias, repeated.bias)
    assert np.allclose(case.j_matrix, case.j_matrix.T)
    assert np.allclose(np.diag(case.j_matrix), 0.0)
    assert np.count_nonzero(np.triu(case.j_matrix, 1)) == 128


def test_sampled_backend_rows_and_summary_are_exp1529_distributional() -> None:
    """REQ-SAMPLE-050: sampled metrics include stability, percent delta, and KL."""

    case = exp1529.n64_signed_ring_chord_case()
    schedule = {"beta": case.beta, "n_warmup": 2, "steps_per_sample": 1, "use_checkerboard": True}
    base = np.array([(idx % 3) == 0 for idx in range(case.n_spins)], dtype=bool)
    samples = np.vstack([np.roll(base, shift) for shift in range(6)])

    carnot_row = exp1529.sampled_backend_row(
        case,
        seed=11,
        backend_label="carnot",
        backend_name="cpu",
        samples=samples,
        schedule=schedule,
    )
    thrml_row = exp1529.sampled_backend_row(
        case,
        seed=11,
        backend_label="thrml",
        backend_name="thrml_cpu_fallback",
        samples=samples,
        schedule=schedule,
    )
    summary = exp1529.summarize_sampled_rows(
        [carnot_row, thrml_row],
        seeds=(11,),
        thresholds={**exp1529.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )

    assert carnot_row["case_id"] == "exp1529:n64_signed_ring_chord:seed_11:carnot"
    assert carnot_row["case_type"] == "sampled_seed_backend"
    assert carnot_row["sample_count"] == 6
    assert summary["case_id"] == "exp1529:n64_signed_ring_chord:sampled_summary"
    assert summary["case_type"] == "sampled_distribution_summary"
    assert summary["n_samples_per_backend"] == 6
    assert summary["mean_energy_delta"] == pytest.approx(0.0)
    assert summary["mean_energy_delta_percent"] == pytest.approx(0.0)
    assert summary["magnetization_delta"] == pytest.approx(0.0)
    assert summary["kl_divergence"] == pytest.approx(0.0)
    assert summary["stability_diagnostics_present"] is True
    assert summary["passed_thresholds"] is True
    assert exp1529._stability_diagnostics_present({"autocorrelation_summary": None}) is False


def test_write_in_progress_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-SAMPLE-050: bootstrap artifact is written before parity execution."""

    output_path = tmp_path / "experiment_1529.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp1529.write_in_progress_artifact(output_path, manifest_path)

    assert artifact["status"] == "in_progress"
    assert artifact["thrml_parity_n64_passed"] is False
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["n_spins"] == 64
    assert artifact["seeds"] == list(exp1529.DEFAULT_SEEDS)
    assert artifact["n_samples_per_backend"] == 0
    assert artifact["parity_manifest_path"] == str(manifest_path)
    assert artifact["honest_verdict"].startswith("success_")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_complete_run_writes_seed_backend_rows_and_summary(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-078: n=64 sampled metrics and JSONL evidence are written."""

    exp1528_path = tmp_path / "exp1528.json"
    output_path = tmp_path / "experiment_1529.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1528(exp1528_path)

    artifact = exp1529.run_parity_n64(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1528_path=exp1528_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11, 12),
        sample_count_per_seed=4,
        n_warmup=2,
        steps_per_sample=1,
        thresholds={**exp1529.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["thrml_parity_n64_passed"] is True
    assert artifact["n_spins"] == 64
    assert artifact["seeds"] == [11, 12]
    assert artifact["n_samples_per_backend"] == 8
    assert artifact["mean_energy_delta"] == pytest.approx(0.0)
    assert artifact["mean_energy_delta_percent"] == pytest.approx(0.0)
    assert artifact["magnetization_delta"] == pytest.approx(0.0)
    assert artifact["kl_divergence"] == pytest.approx(0.0)
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert [row["case_type"] for row in rows].count("sampled_seed_backend") == 4
    assert rows[-1]["case_type"] == "sampled_distribution_summary"
    assert rows[-1]["passed_thresholds"] is True
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_threshold_failure_is_complete_not_passed(tmp_path: Path) -> None:
    """REQ-SAMPLE-050: n=64 pass is false when sampled thresholds fail."""

    exp1528_path = tmp_path / "exp1528.json"
    _write_exp1528(exp1528_path)

    artifact = exp1529.run_parity_n64(
        output_path=tmp_path / "failed_1529.json",
        manifest_path=tmp_path / "failed_manifest.jsonl",
        exp1528_path=exp1528_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu", offset=0),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback", offset=1),
        seeds=(11,),
        sample_count_per_seed=4,
        n_warmup=2,
        steps_per_sample=1,
        thresholds={
            **exp1529.THRESHOLDS,
            "mean_energy_delta_abs_max": 0.0,
            "mean_energy_delta_percent_max": 0.0,
            "magnetization_delta_abs_max": 0.0,
            "kl_divergence_max": 0.0,
            "kl_min_samples_per_backend": 1,
        },
        energy_bin_count=4,
    )

    assert artifact["status"] == "complete"
    assert artifact["thrml_parity_n64_passed"] is False
    assert artifact["blockers"][0]["blocker"] == "sampled_parity_threshold_failed"
    assert artifact["honest_verdict"].startswith("complete_")


def test_upstream_exp1528_evidence_blocks_before_thrml_import(tmp_path: Path) -> None:
    """REQ-SAMPLE-050: Exp1528 passed parity evidence gates the n=64 run."""

    malformed_exp1528 = tmp_path / "malformed_exp1528.json"
    not_passed_exp1528 = tmp_path / "not_passed_exp1528.json"
    malformed_exp1528.write_text("{not-json", encoding="utf-8")
    _write_exp1528(not_passed_exp1528, passed=False)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when Exp1528 is not ready")

    missing_artifact = exp1529.run_parity_n64(
        output_path=tmp_path / "missing_1529.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        exp1528_path=tmp_path / "missing_exp1528.json",
        importer=importer,
    )
    malformed_artifact = exp1529.run_parity_n64(
        output_path=tmp_path / "malformed_1529.json",
        manifest_path=tmp_path / "malformed_manifest.jsonl",
        exp1528_path=malformed_exp1528,
        importer=importer,
    )
    not_passed_artifact = exp1529.run_parity_n64(
        output_path=tmp_path / "not_passed_1529.json",
        manifest_path=tmp_path / "not_passed_manifest.jsonl",
        exp1528_path=not_passed_exp1528,
        importer=importer,
    )

    assert missing_artifact["blockers"][0]["blocker"] == "exp1528_evidence_missing"
    assert malformed_artifact["blockers"][0]["blocker"] == "exp1528_evidence_malformed"
    assert not_passed_artifact["blockers"][0]["blocker"] == "exp1528_parity_not_passed"
    assert missing_artifact["simulator_only"] is True
    assert missing_artifact["no_tsu_hardware_claim"] is True


def test_missing_thrml_import_writes_terminal_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-050: unavailable THRML remains terminal data, not a fake pass."""

    exp1528_path = tmp_path / "exp1528.json"
    _write_exp1528(exp1528_path)

    artifact = exp1529.run_parity_n64(
        output_path=tmp_path / "missing_import_1529.json",
        manifest_path=tmp_path / "missing_import.jsonl",
        exp1528_path=exp1528_path,
        importer=_missing_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11,),
        sample_count_per_seed=2,
        thresholds={**exp1529.THRESHOLDS, "kl_min_samples_per_backend": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["thrml_parity_n64_passed"] is False
    assert artifact["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert artifact["n_samples_per_backend"] == 0
    assert not (tmp_path / "missing_import.jsonl").exists()


def test_validate_artifact_rejects_bad_claims_and_metrics(tmp_path: Path) -> None:
    """REQ-SAMPLE-050: schema validation enforces sampled metrics and no-TSU gates."""

    artifact = exp1529.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "thrml_parity_n64_passed": True,
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
            "blockers": [],
            "honest_verdict": "complete_thrml_carnot_parity_n64_passed_no_tsu_hardware_claim",
        }
    )
    exp1529.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1529.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1529.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1529.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1529.validate_artifact(bad_tsu)

    bad_energy = dict(artifact, mean_energy_delta=1e-4, mean_energy_delta_percent=1e-4)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1529.validate_artifact(bad_energy)

    bad_magnetization = dict(artifact, magnetization_delta=1e-4)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1529.validate_artifact(bad_magnetization)

    bad_kl = dict(artifact, kl_divergence=1e-4)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1529.validate_artifact(bad_kl)

    unstable_kl = dict(artifact, kl_estimate_stable=False)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1529.validate_artifact(unstable_kl)

    missing_stability = dict(artifact, stability_diagnostics_present=False)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1529.validate_artifact(missing_stability)

    missing_autocorr = dict(artifact, autocorrelation_summary={})
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1529.validate_artifact(missing_autocorr)

    bad_verdict = dict(artifact, honest_verdict="claimed hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1529.validate_artifact(bad_verdict)
