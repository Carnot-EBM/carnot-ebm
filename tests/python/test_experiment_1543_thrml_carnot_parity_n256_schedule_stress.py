"""Tests for Exp 1543 n=256 THRML/Carnot schedule-stress parity.

Spec refs: REQ-SAMPLE-053, SCENARIO-SAMPLE-081.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_carnot_parity_n256_schedule_stress as exp1543


class FakeBackend:
    """Deterministic SamplerBackend-shaped fake for schedule-stress tests."""

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
        del biases, config
        n_spins = int(np.asarray(couplings).shape[0])
        base = np.array([(idx + self.seed) % 5 in (0, 2) for idx in range(n_spins)], dtype=bool)
        if self.invert:
            base = np.logical_not(base)
        return np.vstack([np.roll(base, shift % n_spins) for shift in range(int(n_samples))])


def _fake_backend_factory(name: str, *, invert: bool = False) -> Any:
    def factory(seed: int) -> FakeBackend:
        return FakeBackend(seed=seed, backend_name=name, invert=invert)

    return factory


def _write_exp1530(path: Path, *, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "thrml_parity_n128_passed": passed,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
                "n_spins": 128,
                "thresholds": {
                    "mean_energy_delta_abs_max": 0.6,
                    "mean_energy_delta_percent_max": 0.1,
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


def _fake_thrml_import(name: str) -> Any:
    if name == "thrml":
        return SimpleNamespace(__version__="fake-0.1", __file__="/fake/thrml/__init__.py")
    if name == "thrml.models":
        return SimpleNamespace()
    raise ModuleNotFoundError(name)


def _missing_thrml_import(name: str) -> Any:
    raise ModuleNotFoundError(name)


def test_spec_mentions_exp1543_contract() -> None:
    """REQ-SAMPLE-053, SCENARIO-SAMPLE-081: Exp1543 is spec-anchored."""

    spec = (exp1543.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-053" in spec
    assert "SCENARIO-SAMPLE-081" in spec
    assert "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json" in spec
    assert "status" in spec and "thrml_parity_n256_schedule_ready" in spec
    assert "simulator-only" in spec


def test_n256_case_and_schedule_manifest_are_deterministic() -> None:
    """REQ-SAMPLE-053: n=256 case and schedule variants are explicit."""

    case = exp1543.n256_signed_ring_chord_case()
    repeated = exp1543.n256_signed_ring_chord_case()
    schedules = exp1543.default_schedule_variants()

    assert case.n_spins == 256
    assert case.name == "n256_signed_ring_chord"
    assert case.topology == "signed_ring_chord"
    assert np.allclose(case.j_matrix, repeated.j_matrix)
    assert np.allclose(case.bias, repeated.bias)
    assert np.allclose(case.j_matrix, case.j_matrix.T)
    assert np.allclose(np.diag(case.j_matrix), 0.0)
    assert np.count_nonzero(np.triu(case.j_matrix, 1)) == 768
    assert exp1543.validate_schedule_manifest(schedules) == schedules
    assert len({schedule["schedule_id"] for schedule in schedules}) == 3

    duplicate = (dict(schedules[0]), dict(schedules[0]), dict(schedules[2]))
    with pytest.raises(ValueError, match="unique schedule_id"):
        exp1543.validate_schedule_manifest(duplicate)

    too_few = (dict(schedules[0]), dict(schedules[1]))
    with pytest.raises(ValueError, match="at least three"):
        exp1543.validate_schedule_manifest(too_few)

    empty_id = tuple({**schedule, "schedule_id": ""} for schedule in schedules)
    with pytest.raises(ValueError, match="non-empty schedule_id"):
        exp1543.validate_schedule_manifest(empty_id)

    invalid_beta = tuple({**schedule, "beta": 0.0} for schedule in schedules)
    with pytest.raises(ValueError, match="positive beta"):
        exp1543.validate_schedule_manifest(invalid_beta)

    negative_warmup = tuple({**schedule, "n_warmup": -1} for schedule in schedules)
    with pytest.raises(ValueError, match="non-negative n_warmup"):
        exp1543.validate_schedule_manifest(negative_warmup)

    invalid_steps = tuple({**schedule, "steps_per_sample": 0} for schedule in schedules)
    with pytest.raises(ValueError, match="positive steps_per_sample"):
        exp1543.validate_schedule_manifest(invalid_steps)

    missing_checkerboard = tuple(
        {key: value for key, value in schedule.items() if key != "use_checkerboard"}
        for schedule in schedules
    )
    with pytest.raises(ValueError, match="use_checkerboard"):
        exp1543.validate_schedule_manifest(missing_checkerboard)


def test_schedule_rows_and_summary_compute_parity_metrics() -> None:
    """REQ-SAMPLE-053: schedule-stress rows aggregate metric gates."""

    case = exp1543.n256_signed_ring_chord_case()
    schedules = exp1543.default_schedule_variants()
    rows: list[dict[str, Any]] = []
    base = np.array([(idx % 4) == 0 for idx in range(case.n_spins)], dtype=bool)

    for schedule in schedules:
        samples = np.vstack([np.roll(base, shift) for shift in range(6)])
        rows.append(
            exp1543.sampled_schedule_backend_row(
                case,
                schedule=schedule,
                backend_label="carnot",
                backend_name="cpu",
                seed=11,
                samples=samples,
            )
        )
        rows.append(
            exp1543.sampled_schedule_backend_row(
                case,
                schedule=schedule,
                backend_label="thrml",
                backend_name="thrml_cpu_fallback",
                seed=11,
                samples=samples,
            )
        )

    summary = exp1543.summarize_schedule_stress_rows(
        rows,
        schedules=schedules,
        thresholds={**exp1543.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
    )

    assert rows[0]["case_type"] == "schedule_backend_sampled"
    assert rows[0]["schedule_id"] == schedules[0]["schedule_id"]
    assert summary["case_type"] == "schedule_stress_summary"
    assert summary["schedules_tested"] == 3
    assert summary["samples_per_schedule"] == 6
    assert summary["mean_energy_delta"] == pytest.approx(0.0)
    assert summary["max_energy_delta"] == pytest.approx(0.0)
    assert summary["kl_divergence"] == pytest.approx(0.0)
    assert summary["autocorrelation_delta"] == pytest.approx(0.0)
    assert summary["parity_passed"] is True
    assert summary["simulator_only"] is True
    assert summary["no_tsu_hardware_claim"] is True


def test_complete_run_writes_required_artifact_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-081: complete n=256 schedule-stress evidence is written."""

    exp1530_path = tmp_path / "exp1530.json"
    exp1531_path = tmp_path / "exp1531.json"
    output_path = tmp_path / "experiment_1543.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_exp1530(exp1530_path)
    _write_exp1531(exp1531_path)

    artifact = exp1543.run_schedule_stress_n256(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1530_path=exp1530_path,
        exp1531_path=exp1531_path,
        importer=_fake_thrml_import,
        carnot_backend_factory=_fake_backend_factory("carnot_cpu"),
        thrml_backend_factory=_fake_backend_factory("thrml_cpu_fallback"),
        samples_per_schedule=4,
        thresholds={**exp1543.THRESHOLDS, "kl_min_samples_per_backend": 1},
        energy_bin_count=4,
        thrml_seed_offset=0,
        focused_tests_passed=True,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "20260508"
    assert artifact["thrml_parity_n256_schedule_ready"] is True
    assert artifact["n_spins"] == 256
    assert artifact["schedules_tested"] == 3
    assert artifact["samples_per_schedule"] == 4
    assert artifact["mean_energy_delta"] == pytest.approx(0.0)
    assert artifact["max_energy_delta"] == pytest.approx(0.0)
    assert artifact["kl_divergence"] == pytest.approx(0.0)
    assert artifact["autocorrelation_delta"] == pytest.approx(0.0)
    assert artifact["parity_passed"] is True
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["parity_report_path"] == str(manifest_path)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete_")
    assert [row["case_type"] for row in rows].count("schedule_backend_sampled") == 6
    assert rows[-1]["case_type"] == "schedule_stress_summary"
    assert all(row["simulator_only"] is True for row in rows)
    assert all(row["no_tsu_hardware_claim"] is True for row in rows)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_upstream_and_thrml_import_blockers_are_terminal(tmp_path: Path) -> None:
    """REQ-SAMPLE-053: prior artifacts and THRML import gate fake passes."""

    malformed_exp1530 = tmp_path / "malformed_exp1530.json"
    not_passed_exp1530 = tmp_path / "not_passed_exp1530.json"
    malformed_exp1531 = tmp_path / "malformed_exp1531.json"
    not_ready_exp1531 = tmp_path / "not_ready_exp1531.json"
    ready_exp1530 = tmp_path / "ready_exp1530.json"
    ready_exp1531 = tmp_path / "ready_exp1531.json"
    malformed_exp1530.write_text("{not-json", encoding="utf-8")
    _write_exp1530(not_passed_exp1530, passed=False)
    malformed_exp1531.write_text("{not-json", encoding="utf-8")
    _write_exp1531(not_ready_exp1531, ready=False)
    _write_exp1530(ready_exp1530)
    _write_exp1531(ready_exp1531)

    def importer(_name: str) -> Any:
        raise AssertionError("THRML import must not run when prior parity is not ready")

    missing = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "missing.json",
        manifest_path=tmp_path / "missing.jsonl",
        exp1530_path=tmp_path / "missing_exp1530.json",
        exp1531_path=ready_exp1531,
        importer=importer,
    )
    not_passed = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "not_passed.json",
        manifest_path=tmp_path / "not_passed.jsonl",
        exp1530_path=not_passed_exp1530,
        exp1531_path=ready_exp1531,
        importer=importer,
    )
    malformed_exp1530_artifact = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "malformed_exp1530_artifact.json",
        manifest_path=tmp_path / "malformed_exp1530.jsonl",
        exp1530_path=malformed_exp1530,
        exp1531_path=ready_exp1531,
        importer=importer,
    )
    missing_exp1531 = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "missing_exp1531_artifact.json",
        manifest_path=tmp_path / "missing_exp1531.jsonl",
        exp1530_path=ready_exp1530,
        exp1531_path=tmp_path / "missing_exp1531_source.json",
        importer=importer,
    )
    malformed = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "malformed.json",
        manifest_path=tmp_path / "malformed.jsonl",
        exp1530_path=ready_exp1530,
        exp1531_path=malformed_exp1531,
        importer=importer,
    )
    not_ready = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "not_ready.json",
        manifest_path=tmp_path / "not_ready.jsonl",
        exp1530_path=ready_exp1530,
        exp1531_path=not_ready_exp1531,
        importer=importer,
    )
    invalid_schedule = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "invalid_schedule.json",
        manifest_path=tmp_path / "invalid_schedule.jsonl",
        exp1530_path=ready_exp1530,
        exp1531_path=ready_exp1531,
        importer=importer,
        schedules=exp1543.default_schedule_variants()[:2],
    )
    import_blocked = exp1543.run_schedule_stress_n256(
        output_path=tmp_path / "import_blocked.json",
        manifest_path=tmp_path / "import_blocked.jsonl",
        exp1530_path=ready_exp1530,
        exp1531_path=ready_exp1531,
        importer=_missing_thrml_import,
        thresholds={**exp1543.THRESHOLDS, "kl_min_samples_per_backend": 1},
    )

    assert missing["blockers"][0]["blocker"] == "exp1530_evidence_missing"
    assert not_passed["blockers"][0]["blocker"] == "exp1530_parity_not_passed"
    assert malformed_exp1530_artifact["blockers"][0]["blocker"] == "exp1530_evidence_malformed"
    assert missing_exp1531["blockers"][0]["blocker"] == "exp1531_evidence_missing"
    assert malformed["blockers"][0]["blocker"] == "exp1531_evidence_malformed"
    assert not_ready["blockers"][0]["blocker"] == "exp1531_parity_not_ready"
    assert invalid_schedule["blockers"][0]["blocker"] == "invalid_schedule_manifest"
    assert import_blocked["blockers"][0]["blocker"] == "thrml_local_import_unavailable"
    assert import_blocked["status"] == "blocked"
    assert import_blocked["thrml_parity_n256_schedule_ready"] is False
    assert import_blocked["simulator_only"] is True
    assert import_blocked["no_tsu_hardware_claim"] is True


def test_validate_artifact_rejects_bad_claims_metrics_and_verdict(tmp_path: Path) -> None:
    """REQ-SAMPLE-053: artifact validation enforces gates and claim boundaries."""

    artifact = exp1543.write_in_progress_artifact(
        path=tmp_path / "unused.json",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    artifact.update(
        {
            "status": "complete",
            "thrml_parity_n256_schedule_ready": True,
            "schedules_tested": 3,
            "samples_per_schedule": 4,
            "mean_energy_delta": 0.0,
            "max_energy_delta": 0.0,
            "kl_divergence": 0.0,
            "autocorrelation_delta": 0.0,
            "parity_passed": True,
            "focused_tests_passed": True,
            "thresholds": {**exp1543.THRESHOLDS, "kl_min_samples_per_backend": 1},
            "schedule_results": {
                schedule["schedule_id"]: {
                    "passed_thresholds": True,
                    "mean_energy_delta": 0.0,
                    "magnetization_delta": 0.0,
                    "kl_divergence": 0.0,
                    "autocorrelation_delta": 0.0,
                    "n_samples_per_backend": 4,
                }
                for schedule in exp1543.default_schedule_variants()
            },
            "blockers": [],
            "honest_verdict": (
                "complete_thrml_parity_n256_schedule_passed_simulator_only_no_tsu_hardware_claim"
            ),
        }
    )
    exp1543.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("parity_report_path")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp1543.validate_artifact(missing)

    bad_status = dict(artifact, status="done")
    with pytest.raises(ValueError, match="invalid status"):
        exp1543.validate_artifact(bad_status)

    bad_simulator = dict(artifact, simulator_only=False)
    with pytest.raises(ValueError, match="simulator_only"):
        exp1543.validate_artifact(bad_simulator)

    bad_tsu = dict(artifact, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp1543.validate_artifact(bad_tsu)

    bad_n = dict(artifact, n_spins=128)
    with pytest.raises(ValueError, match="n_spins=256"):
        exp1543.validate_artifact(bad_n)

    bad_metric = dict(artifact, kl_divergence=1.0)
    with pytest.raises(ValueError, match="schedule readiness"):
        exp1543.validate_artifact(bad_metric)

    bad_verdict = dict(artifact, honest_verdict="claims hardware parity")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1543.validate_artifact(bad_verdict)
