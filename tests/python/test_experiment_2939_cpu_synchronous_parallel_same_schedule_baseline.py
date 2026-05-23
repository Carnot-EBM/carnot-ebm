"""Tests for Exp 2939 CPU synchronous-parallel same-schedule baseline.

REQ-HW-072: Exp 2939 must replace the invalid sequential-Gibbs speedup
comparison with a CPU synchronous checkerboard baseline for the same Exp 2898
n=64 problems.
SCENARIO-HW-072: the artifact records the apples-to-apples CPU/KV260 timing
ratio and a same-schedule energy equivalence cross-check.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.hardware import kv260_cpu_synchronous_parallel_same_schedule_baseline as exp
from carnot.hardware import kv260_mmd_vs_cpu_sequential_gibbs as exp2938


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def _upload(n_spins: int = 64, max_degree: int = 2) -> dict[str, Any]:
    return {
        "layout": "ising_sampler_v2_n64_sparse_axi_q8_8",
        "max_degree": max_degree,
        "h_q88": [0 for _ in range(n_spins)],
        "adjacency": [
            [int((row + offset + 1) % n_spins) for offset in range(max_degree)]
            for row in range(n_spins)
        ],
        "couplings_q88": [[64, -32] for _ in range(n_spins)],
    }


def _exp2898_payload(seeds: list[int] | None = None) -> dict[str, Any]:
    seeds = seeds or list(exp.RANDOM_SEEDS)
    problems = []
    specs = []
    for seed in seeds:
        problem = exp2938.generate_ising_problem(seed)
        problem["upload"] = _upload()
        problem["beta_final_q88"] = 256
        problems.append(problem)
        specs.append(exp2938.problem_spec(problem))
    return {
        "experiment_id": 2898,
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "bitstream_sha256": "a" * 64,
        "random_seeds_used": seeds,
        "ising_problem_spec": {
            "n_spins": 64,
            "all_seed_specs": specs,
        },
        "problem_payload": {
            "n_spins": 64,
            "max_degree_uploaded": 16,
            "random_seeds_used": seeds,
            "n_sample_counts": [100, 1000, 10000],
            "ising_problem_specs": specs,
            "problems": problems,
        },
        "reproducibility_checksum": "b" * 64,
    }


def _exp2912_payload(*, ready: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
        "same_basis_cpu_baseline_ready": ready,
        "random_seeds_used": list(exp.RANDOM_SEEDS),
        "sample_count_sweep": [100, 1000, 10000],
        "cpu_update_schedule": "cpu_sequential_round_robin_uploaded_sparse_rows_one_sweep_per_sample",
    }


def _write_json(root: Path, relative_path: Path, payload: dict[str, Any]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_preconditions(root: Path) -> None:
    _write_json(root, exp.EXP2898_REL_PATH, _exp2898_payload())
    _write_json(root, exp.EXP2912_REL_PATH, _exp2912_payload())


def _fake_cpu_runner(
    problem: exp2938.DenseIsingProblem,
    *,
    n_samples: int,
    timer_ns: Any | None = None,
) -> exp.CpuSynchronousRunResult:
    energies = [float(problem.seed + index) for index in range(n_samples)]
    return exp.CpuSynchronousRunResult(
        seed=problem.seed,
        energies=energies,
        energy_sha256=exp.sha256_canonical(energies),
        latency_us_median=12.0,
        latency_us_p95=18.0,
        update_schedule=exp.CPU_UPDATE_SCHEDULE,
    )


def test_req_hw_072_spec_anchor_exists() -> None:
    """REQ-HW-072: OpenSpec anchors the same-schedule CPU baseline artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-072" in spec
    assert "SCENARIO-HW-072" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "kv260_speedup_vs_same_schedule_cpu" in spec


def test_checkerboard_sweep_updates_odd_phase_from_fresh_even_state() -> None:
    """REQ-HW-072: the CPU updater uses the KV260 even-then-odd schedule."""

    state = np.array([-1, -1], dtype=np.int8)
    sparse_j = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    fields = np.array([10.0, 0.0], dtype=np.float64)

    updated = exp.checkerboard_sweep(
        state,
        sparse_j_matrix=sparse_j,
        fields=fields,
        beta=10.0,
        rng=np.random.default_rng(7),
    )

    assert updated.tolist() == [1, 1]


def test_cpu_synchronous_run_is_reproducible_and_timed() -> None:
    """SCENARIO-HW-072: CPU synchronous runs record reproducible energies and timings."""

    problem = exp2938.recover_exp2898_problems(_exp2898_payload())[0]
    ticks = iter(range(0, 20_000, 1_000))

    first = exp.run_cpu_synchronous_parallel_glauber(
        problem,
        n_samples=4,
        timer_ns=lambda: next(ticks),
    )
    second = exp.run_cpu_synchronous_parallel_glauber(problem, n_samples=4)

    assert first.seed == 42
    assert len(first.energies) == 4
    assert first.latency_us_median == pytest.approx(1.0)
    assert first.latency_us_p95 == pytest.approx(1.0)
    assert first.energy_sha256 == second.energy_sha256
    assert first.update_schedule == exp.CPU_UPDATE_SCHEDULE


def test_run_experiment_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-072: same-schedule timing ratio drives the paper-v6 verdict."""

    _write_preconditions(tmp_path)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        n_samples=4,
        cpu_runner=_fake_cpu_runner,
        kv260_energy_provider=exp.same_schedule_reference_energies,
        started_s=0.0,
        now_s=20.5,
        enforce_min_duration=False,
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == "complete: kv260_slower_than_same_schedule_cpu_at_n64"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["cpu_synchronous_parallel_per_sample_us_median"] == pytest.approx(12.0)
    assert artifact["cpu_synchronous_parallel_per_sample_us_p95"] == pytest.approx(18.0)
    assert artifact["kv260_per_sample_us_cited"] == pytest.approx(24.0)
    assert artifact["kv260_speedup_vs_same_schedule_cpu"]["value"] == pytest.approx(0.5)
    assert artifact["energy_distribution_equivalence_test"]["ks_pvalue"] >= 0.01
    assert artifact["energy_distribution_equivalence_test"]["mmd_squared"] == pytest.approx(0.0)
    assert artifact["random_seed"] == 2939
    assert artifact["random_seeds_used"] == [42, 137, 271]
    assert artifact["paper_v6_recommendation"].startswith("retract")
    assert artifact["duration_s"] == pytest.approx(20.5)

    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_success_artifact_can_retain_narrow_speedup_claim(tmp_path: Path) -> None:
    """REQ-HW-072: ratios >=1 keep only a narrow n=64 speedup recommendation."""

    _write_preconditions(tmp_path)

    def slow_cpu_runner(
        problem: exp2938.DenseIsingProblem,
        *,
        n_samples: int,
        timer_ns: Any | None = None,
    ) -> exp.CpuSynchronousRunResult:
        row = _fake_cpu_runner(problem, n_samples=n_samples, timer_ns=timer_ns)
        return exp.CpuSynchronousRunResult(
            seed=row.seed,
            energies=row.energies,
            energy_sha256=row.energy_sha256,
            latency_us_median=48.0,
            latency_us_p95=60.0,
            update_schedule=row.update_schedule,
        )

    artifact = exp.run_experiment(
        root_path=tmp_path,
        n_samples=4,
        cpu_runner=slow_cpu_runner,
        kv260_energy_provider=exp.same_schedule_reference_energies,
        started_s=0.0,
        now_s=21.0,
        enforce_min_duration=False,
    )

    assert artifact["honest_verdict"] == "complete: kv260_faster_than_same_schedule_cpu_at_n64"
    assert artifact["kv260_speedup_vs_same_schedule_cpu"]["value"] == pytest.approx(2.0)
    assert artifact["paper_v6_recommendation"].startswith("retain narrow")


def test_run_experiment_blocks_missing_or_unready_preconditions(tmp_path: Path) -> None:
    """REQ-HW-072: Exp 2898 and ready Exp 2912 are mandatory preconditions."""

    missing = exp.run_experiment(
        root_path=tmp_path,
        n_samples=1,
        started_s=0.0,
        now_s=1.0,
        enforce_min_duration=False,
    )
    assert missing["honest_verdict"] == "blocked_exp2898_artifact_missing"

    _write_json(tmp_path, exp.EXP2898_REL_PATH, _exp2898_payload())
    no_2912 = exp.run_experiment(
        root_path=tmp_path,
        n_samples=1,
        started_s=0.0,
        now_s=1.0,
        enforce_min_duration=False,
    )
    assert no_2912["honest_verdict"] == "blocked_exp2912_artifact_missing"

    _write_json(tmp_path, exp.EXP2912_REL_PATH, _exp2912_payload(ready=False))
    unready = exp.run_experiment(
        root_path=tmp_path,
        n_samples=1,
        started_s=0.0,
        now_s=1.0,
        enforce_min_duration=False,
    )
    assert unready["honest_verdict"] == "blocked_exp2912_cpu_baseline_not_ready"


def test_validation_and_helper_failures(tmp_path: Path) -> None:
    """REQ-HW-072: schema and helper validation reject corrupt artifacts."""

    problem = exp2938.recover_exp2898_problems(_exp2898_payload())[0]
    bad_fields = replace(problem, upload={**problem.upload, "h_q88": [0]})
    with pytest.raises(ValueError, match="h_q88"):
        exp.build_sparse_upload_matrix(bad_fields)

    bad_couplings = replace(problem, upload={**problem.upload, "couplings_q88": [[1]]})
    with pytest.raises(ValueError, match="adjacency"):
        exp.build_sparse_upload_matrix(bad_couplings)

    with pytest.raises(ValueError, match="n_samples"):
        exp.run_cpu_synchronous_parallel_glauber(problem, n_samples=0)
    with pytest.raises(ValueError, match="median"):
        exp._median([])
    with pytest.raises(ValueError, match="p95"):
        exp._p95([])
    assert exp._deterministic_subset([0, 1, 2, 3, 4], 3).tolist() == [0.0, 2.0, 4.0]

    _write_preconditions(tmp_path)
    artifact = exp.run_experiment(
        root_path=tmp_path,
        n_samples=4,
        cpu_runner=_fake_cpu_runner,
        kv260_energy_provider=exp.same_schedule_reference_energies,
        started_s=0.0,
        now_s=20.0,
        enforce_min_duration=False,
    )

    missing = dict(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing"):
        exp.validate_artifact(missing)

    bad_substrate = dict(artifact, inference_substrate="hardware_smoke")
    with pytest.raises(ValueError, match="live_llm_inference"):
        exp.validate_artifact(bad_substrate)

    bad_timing = dict(artifact, cpu_synchronous_parallel_per_sample_us_median=0.0)
    with pytest.raises(ValueError, match="positive"):
        exp.validate_artifact(bad_timing)

    bad_p95 = dict(artifact, cpu_synchronous_parallel_per_sample_us_p95=0.0)
    with pytest.raises(ValueError, match="positive"):
        exp.validate_artifact(bad_p95)

    bad_speedup = dict(artifact, kv260_speedup_vs_same_schedule_cpu={})
    with pytest.raises(ValueError, match="principle"):
        exp.validate_artifact(bad_speedup)

    bad_equivalence_object = dict(artifact, energy_distribution_equivalence_test=[])
    with pytest.raises(ValueError, match="energy_distribution"):
        exp.validate_artifact(bad_equivalence_object)

    bad_duration = dict(artifact, duration_s=19.0)
    with pytest.raises(ValueError, match="duration"):
        exp.validate_artifact(bad_duration)

    bad_ks = dict(artifact)
    bad_ks["energy_distribution_equivalence_test"] = {
        **artifact["energy_distribution_equivalence_test"],
        "ks_pvalue": 0.0,
    }
    with pytest.raises(ValueError, match="KS"):
        exp.validate_artifact(bad_ks)

    bad_seed = dict(artifact, random_seed=1)
    with pytest.raises(ValueError, match="random_seed"):
        exp.validate_artifact(bad_seed)

    bad_seeds = dict(artifact, random_seeds_used=[42])
    with pytest.raises(ValueError, match="random_seeds_used"):
        exp.validate_artifact(bad_seeds)

    bad_checksum = dict(artifact, reproducibility_checksum="short")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)


def test_malformed_exp2898_blocks_before_cpu_run(tmp_path: Path) -> None:
    """REQ-HW-072: unreproducible Exp 2898 provenance fails closed."""

    payload = _exp2898_payload()
    payload["problem_payload"]["random_seeds_used"] = [42]
    _write_json(tmp_path, exp.EXP2898_REL_PATH, payload)
    _write_json(tmp_path, exp.EXP2912_REL_PATH, _exp2912_payload())

    artifact = exp.run_experiment(
        root_path=tmp_path,
        n_samples=1,
        started_s=0.0,
        now_s=1.0,
        enforce_min_duration=False,
    )

    assert artifact["honest_verdict"] == "blocked_exp2898_problem_reproduction_failed"


def test_min_duration_extension_branch_is_exercised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-072: successful runs can enforce the duration gate with extra work."""

    _write_preconditions(tmp_path)
    monkeypatch.setattr(exp, "_extend_runtime_with_same_schedule_work", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_duration", lambda started_s, now_s: 20.0)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        n_samples=4,
        cpu_runner=_fake_cpu_runner,
        kv260_energy_provider=exp.same_schedule_reference_energies,
    )

    assert artifact["duration_s"] == pytest.approx(20.0)


def test_main_reports_summary_and_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-072: the CLI reports the deliverable path."""

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root_path: {"honest_verdict": "complete: cli-ok"},
    )

    assert exp.main(["--root", str(tmp_path)]) == 0
    assert "complete: cli-ok" in capsys.readouterr().out

    assert exp.main(["--root", str(tmp_path), "--print-result-path"]) == 0
    assert str(tmp_path / exp.OUTPUT_REL_PATH) in capsys.readouterr().out
