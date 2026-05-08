"""Tests for Exp 1528 sampled n=32 THRML/Carnot parity.

Spec refs: REQ-SAMPLE-049, SCENARIO-SAMPLE-077.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_carnot_parity_n32_sample as exp1528


class FakeBackend:
    """Deterministic SamplerBackend-shaped fake for repeated-chain sample rows."""

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


def _write_exp1527(path: Path, *, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_parity_n16_passed": passed,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "exact_states_enumerated": 65536,
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


def test_spec_mentions_exp1528_contract() -> None:
    """REQ-SAMPLE-049, SCENARIO-SAMPLE-077: Exp1528 is spec-anchored."""

    spec = (exp1528.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-049" in spec
    assert "SCENARIO-SAMPLE-077" in spec
    assert "experiment_1528_thrml_carnot_parity_n32_sample.json" in spec
    assert "thrml_carnot_parity_n32_1528.jsonl" in spec


def test_n32_case_is_deterministic_signed_ring_chord_sample_only() -> None:
    """REQ-SAMPLE-049: the n=32 case is deterministic and sample-only."""

    case = exp1528.n32_signed_ring_chord_case()
    all_minus = -np.ones(case.n_spins, dtype=np.int8)

    assert case.n_spins == 32
    assert case.name == "n32_signed_ring_chord"
    assert case.topology == "signed_ring_chord"
    assert case.beta == pytest.approx(1.10)
    assert exp1528.DEFAULT_SEEDS == (20260508, 20260509, 20260510, 20260511, 20260512)
    assert np.allclose(case.j_matrix, case.j_matrix.T)
    assert np.allclose(np.diag(case.j_matrix), 0.0)
    assert np.count_nonzero(np.triu(case.j_matrix, 1)) == 64
    assert exp1528.ising_energy(case, all_minus) == pytest.approx(-1.44)


def test_sampled_backend_rows_and_summary_are_distributional() -> None:
    """REQ-SAMPLE-049: sampled metrics include energy, magnetization, autocorr, KL."""

    case = exp1528.n32_signed_ring_chord_case()
    schedule = {"beta": case.beta, "n_warmup": 2, "steps_per_sample": 1, "use_checkerboard": True}
    base = np.array([(idx % 2) == 0 for idx in range(case.n_spins)], dtype=bool)
    samples = np.vstack([np.roll(base, shift) for shift in range(4)])

    carnot_row = exp1528.sampled_backend_row(
        case,
        seed=11,
        backend_label="carnot",
        backend_name="cpu",
        samples=samples,
        schedule=schedule,
    )
    thrml_row = exp1528.sampled_backend_row(
        case,
        seed=11,
        backend_label="thrml",
        backend_name="thrml_cpu_fallback",
        samples=samples,
        schedule=schedule,
    )
    summary = exp1528.summarize_sampled_rows(
        [carnot_row, thrml_row],
        seeds=(11,),
        thresholds={**exp1528.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )
    short_autocorr = exp1528._lag_one_autocorrelation(np.array([1.0]))
    constant_autocorr = exp1528._lag_one_autocorrelation(np.array([2.0, 2.0, 2.0]))

    assert carnot_row["case_type"] == "sampled_seed_backend"
    assert carnot_row["sample_count"] == 4
    assert carnot_row["backend"] == "carnot"
    assert carnot_row["energy_quantiles"]["q50"] == pytest.approx(carnot_row["mean_energy"])
    assert carnot_row["simulator_only"] is True
    assert carnot_row["no_tsu_hardware_claim"] is True
    assert summary["case_type"] == "sampled_distribution_summary"
    assert summary["n_samples_per_backend"] == 4
    assert summary["mean_energy_delta"] == pytest.approx(0.0)
    assert summary["magnetization_delta"] == pytest.approx(0.0)
    assert summary["kl_divergence"] == pytest.approx(0.0)
    assert summary["kl_estimate_stable"] is True
    assert summary["passed_thresholds"] is True
    assert short_autocorr == pytest.approx(0.0)
    assert constant_autocorr == pytest.approx(0.0)


def test_write_in_progress_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-SAMPLE-049: bootstrap artifact is written before parity execution."""

    output_path = tmp_path / "experiment_1528.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp1528.write_in_progress_artifact(output_path, manifest_path)

    assert artifact["status"] == "in_progress"
    assert artifact["thrml_parity_n32_passed"] is False
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["n_spins"] == 32
    assert artifact["seeds"] == list(exp1528.DEFAULT_SEEDS)
    assert artifact["n_samples_per_backend"] == 0
    assert artifact["parity_manifest_path"] == str(manifest_path)
    assert artifact["honest_verdict"].startswith("success_")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_complete_run_writes_seed_backend_rows_and_summary(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-077: n=32 sampled metrics and JSONL evidence are written."""

    exp1527_path = tmp_path / "exp1527.json"
    output_path = tmp_path / "experiment_1528.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1527(exp1527_path)

    artifact = exp1528.run_parity_n32(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1527_path=exp1527_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11, 12),
        sample_count_per_seed=4,
        n_warmup=2,
        steps_per_sample=1,
        thresholds={**exp1528.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["thrml_parity_n32_passed"] is True
    assert artifact["n_spins"] == 32
    assert artifact["seeds"] == [11, 12]
    assert artifact["n_samples_per_backend"] == 8
    assert artifact["mean_energy_delta"] == pytest.approx(0.0)
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


def test_upstream_exp1527_evidence_blocks_before_thrml_import(tmp_path: Path) -> None:
    """REQ-SAMPLE-049: Exp1527 passed parity evidence gates the n=32 run."""

    malformed_exp1527 = tmp_path / "malformed_exp1527.json"
    not_passed_exp1527 = tmp_path / "not_passed_exp1527.json"
    malformed_exp1527.write_text("{not-json", encoding="utf-8")
    _write_exp1527(not_passed_exp1527, passed=False)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when Exp1527 is not ready")

    missing_artifact = exp1528.run_parity_n32(
        output_path=tmp_path / "missing_1528.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        exp1527_path=tmp_path / "missing_exp1527.json",
        importer=importer,
    )
    malformed_artifact = exp1528.run_parity_n32(
        output_path=tmp_path / "malformed_1528.json",
        manifest_path=tmp_path / "malformed_manifest.jsonl",
        exp1527_path=malformed_exp1527,
        importer=importer,
    )
    not_passed_artifact = exp1528.run_parity_n32(
        output_path=tmp_path / "not_passed_1528.json",
        manifest_path=tmp_path / "not_passed_manifest.jsonl",
        exp1527_path=not_passed_exp1527,
        importer=importer,
    )

    assert missing_artifact["blockers"][0]["blocker"] == "exp1527_evidence_missing"
    assert malformed_artifact["blockers"][0]["blocker"] == "exp1527_evidence_malformed"
    assert not_passed_artifact["blockers"][0]["blocker"] == "exp1527_parity_not_passed"
    assert missing_artifact["simulator_only"] is True
    assert missing_artifact["no_tsu_hardware_claim"] is True


def test_missing_thrml_import_writes_terminal_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-049: unavailable THRML remains terminal data, not a fake pass."""

    exp1527_path = tmp_path / "exp1527.json"
    _write_exp1527(exp1527_path)

    artifact = exp1528.run_parity_n32(
        output_path=tmp_path / "missing_import_1528.json",
        manifest_path=tmp_path / "missing_import.jsonl",
        exp1527_path=exp1527_path,
        importer=_missing_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        seeds=(11,),
        sample_count_per_seed=2,
        thresholds={**exp1528.THRESHOLDS, "kl_min_samples_per_backend": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["thrml_parity_n32_passed"] is False
    assert artifact["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert artifact["n_samples_per_backend"] == 0
    assert not (tmp_path / "missing_import.jsonl").exists()


def test_validate_artifact_rejects_bad_claims_and_metrics(tmp_path: Path) -> None:
    """REQ-SAMPLE-049: schema validation enforces sampled metrics and no-TSU gates."""

    artifact = exp1528.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "thrml_parity_n32_passed": True,
            "seeds": [11],
            "n_samples_per_backend": 4,
            "mean_energy_delta": 0.0,
            "magnetization_delta": 0.0,
            "autocorrelation_summary": {"lag1_delta": 0.0},
            "kl_divergence": 0.0,
            "thresholds": {
                "mean_energy_delta_abs_max": 1e-8,
                "magnetization_delta_abs_max": 1e-8,
                "kl_divergence_max": 1e-8,
                "kl_min_samples_per_backend": 1,
            },
            "kl_estimate_stable": True,
            "blockers": [],
            "honest_verdict": "complete_thrml_carnot_parity_n32_passed_no_tsu_hardware_claim",
        }
    )
    exp1528.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1528.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1528.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1528.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1528.validate_artifact(bad_tsu)

    bad_energy = dict(artifact, mean_energy_delta=1e-4)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1528.validate_artifact(bad_energy)

    bad_magnetization = dict(artifact, magnetization_delta=1e-4)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1528.validate_artifact(bad_magnetization)

    bad_kl = dict(artifact, kl_divergence=1e-4)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1528.validate_artifact(bad_kl)

    unstable_kl = dict(artifact, kl_estimate_stable=False)
    with pytest.raises(ValueError, match="sampled pass metrics"):
        exp1528.validate_artifact(unstable_kl)

    bad_verdict = dict(artifact, honest_verdict="claimed hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1528.validate_artifact(bad_verdict)
