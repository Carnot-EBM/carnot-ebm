"""Tests for Exp 2916 THRML/KV260 simulator parity.

REQ-HW-067: Exp 2916 must import THRML, sample the matched Exp 2898/2912
Ising basis where supported, and compare simulator energy summaries against
CPU Gibbs plus KV260 final-energy evidence without making a TSU hardware claim.
SCENARIO-HW-067: ready same-basis evidence produces a bounded simulator parity
artifact with distributional summaries and no hardware claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.hardware import thrml_kv260_sampler_parity as exp


def _upload(n_spins: int = 64, max_degree: int = 2) -> dict[str, Any]:
    adjacency = [
        [int((row + offset + 1) % n_spins) for offset in range(max_degree)]
        for row in range(n_spins)
    ]
    couplings = [[64, -32] for _ in range(n_spins)]
    return {
        "layout": "ising_sampler_v2_n64_sparse_axi_q8_8",
        "max_degree": max_degree,
        "h_q88": [0 for _ in range(n_spins)],
        "adjacency": adjacency,
        "couplings_q88": couplings,
    }


def _kv260_payload(*, seeds: list[int] | None = None) -> dict[str, Any]:
    seeds = seeds or [42, 137]
    upload_by_seed = {seed: _upload() for seed in seeds}
    return {
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "random_seeds_used": seeds,
        "ising_problem_spec": {"n_spins": 64},
        "problem_payload": {
            "n_spins": 64,
            "random_seeds_used": seeds,
            "n_sample_counts": [100],
            "ising_problem_specs": [
                {
                    "n_spins": 64,
                    "random_seed": seed,
                    "j_matrix_sha256": f"{seed:064x}"[-64:],
                    "h_vector_sha256": "0" * 64,
                }
                for seed in seeds
            ],
            "problems": [
                {
                    "random_seed": seed,
                    "n_spins": 64,
                    "beta_final_q88": 256,
                    "h_vector": [0.0 for _ in range(64)],
                    "j_matrix": [[0.0 for _ in range(64)] for _ in range(64)],
                    "upload": upload_by_seed[seed],
                }
                for seed in seeds
            ],
        },
        "sample_count_sweep_results": [
            {
                "seed": seed,
                "n_samples": 100,
                "final_energy": -float(seed % 10),
                "per_sample_wall_clock_us_median": 24.0,
                "per_sample_wall_clock_us_p95": 25.0,
            }
            for seed in seeds
        ],
        "per_seed_results": [
            {"seed": seed, "n_samples": 100, "final_energy": -float(seed % 10)}
            for seed in seeds
        ],
        "reproducibility_checksum": "a" * 64,
    }


def _cpu_payload(*, seeds: list[int] | None = None, ready: bool = True) -> dict[str, Any]:
    seeds = seeds or [42, 137]
    return {
        "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
        "same_basis_cpu_baseline_ready": ready,
        "n_spins": 64,
        "matched_sparse_topology": ready,
        "matched_coupling_tensor": ready,
        "matched_field_tensor": ready,
        "random_seeds_used": seeds,
        "sample_count_sweep": [100],
        "cpu_per_seed_results": [
            {
                "seed": seed,
                "sample_count": 100,
                "n_spins": 64,
                "final_energy": -float((seed % 10) + 1),
            }
            for seed in seeds
        ],
        "speedup_claim_made": False,
        "inference_substrate": "cpu_sampler",
        "run_date": "20260523",
    }


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_upstreams(
    root: Path,
    *,
    kv260: dict[str, Any] | None = None,
    cpu: dict[str, Any] | None = None,
) -> None:
    _write_json(root, exp.KV260_ARTIFACT_REL_PATH, kv260 or _kv260_payload())
    _write_json(root, exp.CPU_ARTIFACT_REL_PATH, cpu or _cpu_payload())


class _FakeThrml:
    __version__ = "0.9.1"
    __file__ = "/tmp/thrml/__init__.py"


def _importer(name: str) -> Any:
    if name == "thrml":
        return _FakeThrml()
    raise ModuleNotFoundError(name)


def _deterministic_sampler(case: exp.ThrmlIsingCase, seed: int, n_samples: int) -> np.ndarray:
    del seed
    rows = []
    for row in range(n_samples):
        rows.append([(row + spin) % 2 == 0 for spin in range(case.n_spins)])
    return np.asarray(rows, dtype=bool)


def test_req_hw_067_spec_anchor_exists() -> None:
    """REQ-HW-067: OpenSpec defines the Exp 2916 simulator-parity contract."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/fpga/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-HW-067" in spec
    assert "SCENARIO-HW-067" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert exp.INFERENCE_SUBSTRATE in spec


def test_req_hw_067_blocks_before_thrml_import_when_cpu_baseline_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-HW-067: unready Exp 2912 evidence stops before simulator import."""

    _write_json(tmp_path, exp.CPU_ARTIFACT_REL_PATH, _cpu_payload(ready=False))

    def forbidden_importer(name: str) -> Any:
        raise AssertionError(f"unexpected import: {name}")

    artifact = exp.run_experiment(
        root_path=tmp_path,
        importer=forbidden_importer,
        sampler=_deterministic_sampler,
        started_s=10.0,
        now_s=11.0,
    )

    assert artifact["honest_verdict"] == "blocked_cpu_baseline_not_ready"
    assert artifact["thrml_kv260_parity_ready"] is False
    assert artifact["thrml_import_ok"] is False
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()


def test_req_hw_067_blocks_on_thrml_import_failure(tmp_path: Path) -> None:
    """REQ-HW-067: missing THRML is an honest terminal block, not fake parity."""

    _write_upstreams(tmp_path)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        sampler=_deterministic_sampler,
        started_s=1.0,
        now_s=2.25,
    )

    assert artifact["honest_verdict"] == "blocked_thrml_import_unavailable"
    assert artifact["thrml_kv260_parity_ready"] is False
    assert artifact["thrml_import_ok"] is False
    assert artifact["thrml_version"] == "unavailable"
    assert artifact["cpu_vs_thrml_distance"] == pytest.approx(0.0)


def test_scenario_hw_067_matched_n64_writes_distribution_summary(tmp_path: Path) -> None:
    """SCENARIO-HW-067: full n=64 THRML samples are compared with CPU/KV260 evidence."""

    _write_upstreams(tmp_path)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=_deterministic_sampler,
        sample_count_per_seed=4,
        started_s=3.0,
        now_s=7.5,
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["thrml_kv260_parity_ready"] is True
    assert artifact["thrml_import_ok"] is True
    assert artifact["thrml_version"] == "0.9.1"
    assert artifact["matched_full_n64_basis"] is True
    assert artifact["fallback_subset_used"] is False
    assert artifact["random_seeds_used"] == [42, 137]
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["duration_s"] == pytest.approx(4.5)

    summary = artifact["energy_distribution_summary"]
    assert summary["thrml"]["n_spins"] == 64
    assert summary["thrml"]["sample_count"] == 8
    assert set(summary["thrml"]) >= {"mean", "variance", "min", "histogram"}
    assert summary["cpu_final_energy"]["sample_count"] == 2
    assert artifact["cpu_vs_thrml_distance"] >= 0.0
    assert artifact["kv260_vs_thrml_summary"]["histogram_distance"] >= 0.0

    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_hw_067_falls_back_to_bounded_n16_subset_when_full_basis_is_unsupported(
    tmp_path: Path,
) -> None:
    """REQ-HW-067: unsupported n=64 THRML representation emits an n=16 limitation."""

    _write_upstreams(tmp_path, kv260=_kv260_payload(seeds=[42]), cpu=_cpu_payload(seeds=[42]))
    calls: list[int] = []

    def sampler(case: exp.ThrmlIsingCase, seed: int, n_samples: int) -> np.ndarray:
        del seed
        calls.append(case.n_spins)
        if case.n_spins == 64:
            raise exp.ThrmlBasisUnsupportedError("full sparse edge list unsupported")
        return np.ones((n_samples, case.n_spins), dtype=bool)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=sampler,
        sample_count_per_seed=3,
        started_s=4.0,
        now_s=5.0,
    )

    assert calls == [64, 16]
    assert artifact["thrml_kv260_parity_ready"] is True
    assert artifact["matched_full_n64_basis"] is False
    assert artifact["fallback_subset_used"] is True
    assert artifact["energy_distribution_summary"]["thrml"]["n_spins"] == 16
    assert "full sparse edge list unsupported" in artifact["basis_limitation"]


def test_req_hw_067_sparse_case_energy_matches_upload_energy() -> None:
    """REQ-HW-067: recovered THRML case preserves the uploaded sparse energy."""

    basis = exp.recover_problem_basis(_kv260_payload(seeds=[42])).problems[0]
    case = exp.thrml_case_from_sparse_basis(basis)
    state = np.asarray([1 if idx % 2 == 0 else -1 for idx in range(64)], dtype=np.int8)

    assert case.n_spins == 64
    assert case.edge_count == 128
    assert exp.energy_for_spin_state(case, state) == pytest.approx(
        exp.sparse_upload_energy(basis, state)
    )


def test_req_hw_067_validation_and_helper_failures_are_explicit(tmp_path: Path) -> None:
    """REQ-HW-067: schema and metric helpers fail loudly on invalid artifacts."""

    _write_upstreams(tmp_path)
    valid = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=_deterministic_sampler,
        sample_count_per_seed=2,
    )
    exp.validate_artifact(valid)

    missing = dict(valid)
    missing.pop("run_date")
    with pytest.raises(ValueError, match="missing"):
        exp.validate_artifact(missing)

    bad_claim = dict(valid, no_tsu_hardware_claim=False)
    with pytest.raises(ValueError, match="no_tsu_hardware_claim"):
        exp.validate_artifact(bad_claim)

    bad_substrate = dict(valid, inference_substrate="hardware_smoke")
    with pytest.raises(ValueError, match="simulator_parity"):
        exp.validate_artifact(bad_substrate)

    with pytest.raises(ValueError, match="empty"):
        exp.energy_distribution_summary([], bin_count=4)
    with pytest.raises(ValueError, match="positive"):
        exp.histogram_distance([1.0], [2.0], bin_count=0)


def test_req_hw_067_probe_metric_and_energy_edge_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-067: import, histogram, and sparse conversion edge paths are bounded."""

    assert exp._round_metric(None) is None
    assert exp._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    assert exp._read_json(bad_json) == {}

    class NoVersionThrml:
        __file__ = "/tmp/no-version-thrml.py"

    monkeypatch.setattr(
        exp.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(exp.importlib.metadata.PackageNotFoundError(name)),
    )
    probe = exp.probe_thrml_import(lambda name: NoVersionThrml())
    assert probe.ok is True
    assert probe.version == "unknown"

    payload = _kv260_payload(seeds=[42])
    for row in payload["problem_payload"]["problems"][0]["upload"]["couplings_q88"]:
        row[1] = 0
    basis = exp.recover_problem_basis(payload).problems[0]
    case = exp.thrml_case_from_sparse_basis(basis)
    assert case.edge_count == 64

    with pytest.raises(ValueError, match="basis range"):
        exp.thrml_case_from_sparse_basis(basis, n_spins=65)
    with pytest.raises(ValueError, match="state shape"):
        exp.energy_for_spin_state(case, np.ones(63, dtype=np.int8))
    with pytest.raises(ValueError, match="returned shape"):
        exp._samples_to_energies(case, np.ones((2, 63), dtype=bool))

    with pytest.raises(ValueError, match="positive"):
        exp.energy_distribution_summary([1.0], bin_count=0)
    assert exp.histogram_distance([], [1.0]) == pytest.approx(0.0)
    assert exp.histogram_distance([1.0], [1.0]) == pytest.approx(0.0)
    assert exp._final_energies_from_cpu({"cpu_per_seed_results": "bad"}) == []
    assert exp._final_energies_from_kv260({"sample_count_sweep_results": "bad"}) == (
        [],
        "per_seed_results.final_energy",
    )
    assert exp._final_energies_from_kv260(
        {"sample_count_sweep_results": "bad", "per_seed_results": "also-bad"}
    ) == ([], "per_seed_results.final_energy")
    assert exp._final_energies_from_kv260(
        {"per_seed_results": [{"final_energy": -1.0}]}
    ) == ([-1.0], "per_seed_results.final_energy")


def test_req_hw_067_unrecoverable_kv260_basis_and_seed_fallback(tmp_path: Path) -> None:
    """REQ-HW-067: malformed KV260 evidence blocks, and seed fallback is explicit."""

    _write_json(tmp_path, exp.CPU_ARTIFACT_REL_PATH, _cpu_payload())
    _write_json(tmp_path, exp.KV260_ARTIFACT_REL_PATH, {"problem_payload": {}})

    blocked = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=_deterministic_sampler,
        started_s=1.0,
        now_s=2.0,
    )
    assert blocked["honest_verdict"] == "blocked_kv260_problem_basis_unrecoverable"
    assert blocked["thrml_import_ok"] is True

    _write_upstreams(tmp_path, kv260=_kv260_payload(seeds=[42]), cpu=_cpu_payload(seeds=[999]))
    artifact = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=_deterministic_sampler,
        sample_count_per_seed=2,
        started_s=2.0,
        now_s=3.0,
    )
    assert artifact["thrml_kv260_parity_ready"] is True
    assert artifact["random_seeds_used"] == [42]


def test_req_hw_067_sampling_failure_blocks_are_terminal(tmp_path: Path) -> None:
    """REQ-HW-067: unsupported fallback and generic sampler failures fail closed."""

    _write_upstreams(tmp_path, kv260=_kv260_payload(seeds=[42]), cpu=_cpu_payload(seeds=[42]))

    def always_unsupported(case: exp.ThrmlIsingCase, seed: int, n_samples: int) -> np.ndarray:
        del case, seed, n_samples
        raise exp.ThrmlBasisUnsupportedError("unsupported")

    unsupported = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=always_unsupported,
        started_s=3.0,
        now_s=4.0,
    )
    assert unsupported["honest_verdict"] == "blocked_thrml_basis_unsupported"
    assert "fallback also failed" in unsupported["basis_limitation"]

    def broken_sampler(case: exp.ThrmlIsingCase, seed: int, n_samples: int) -> np.ndarray:
        del case, seed, n_samples
        raise RuntimeError("boom")

    failed = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=broken_sampler,
        started_s=4.0,
        now_s=5.0,
    )
    assert failed["honest_verdict"] == "blocked_thrml_sampling_failed"
    assert "RuntimeError: boom" in failed["basis_limitation"]


def test_req_hw_067_validate_artifact_rejects_ready_state_corruption(tmp_path: Path) -> None:
    """REQ-HW-067: ready artifacts cannot be corrupted into invalid claim states."""

    _write_upstreams(tmp_path)
    valid = exp.run_experiment(
        root_path=tmp_path,
        importer=_importer,
        sampler=_deterministic_sampler,
        sample_count_per_seed=2,
    )

    for payload, pattern in [
        (dict(valid, run_date="20260522"), "20260523"),
        (dict(valid, duration_s=-0.1), "duration_s"),
        (dict(valid, cpu_vs_thrml_distance=-1.0), "cpu_vs_thrml_distance"),
        (
            dict(valid, matched_full_n64_basis=True, fallback_subset_used=True),
            "full n64",
        ),
        (dict(valid, thrml_import_ok=False), "thrml_import_ok"),
        (dict(valid, random_seeds_used=[]), "random_seeds_used"),
        (dict(valid, energy_distribution_summary={}), "THRML energy summary"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            exp.validate_artifact(payload)


def test_req_hw_067_main_reports_output_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-067: the CLI reports the terminal artifact path."""

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root_path: {"honest_verdict": "complete: cli-ok"},
    )

    assert exp.main(["--root", str(tmp_path)]) == 0
    assert "complete: cli-ok" in capsys.readouterr().out

    assert exp.main(["--root", str(tmp_path), "--print-result-path"]) == 0
    assert str(tmp_path / exp.OUTPUT_REL_PATH) in capsys.readouterr().out
