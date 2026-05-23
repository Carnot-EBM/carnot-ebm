"""Tests for Exp 2938 KV260 MMD versus exact CPU sequential Gibbs.

Spec refs: REQ-HW-071, SCENARIO-HW-071.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.hardware import kv260_mmd_vs_cpu_sequential_gibbs as exp


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
        problem = exp.generate_ising_problem(seed)
        problem["upload"] = _upload()
        problem["beta_final_q88"] = 256
        problems.append(problem)
        specs.append(exp.problem_spec(problem))
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


def _write_exp2898(root: Path, payload: dict[str, Any] | None = None) -> None:
    path = root / exp.EXP2898_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _exp2898_payload(), sort_keys=True), encoding="utf-8")


def test_req_hw_071_spec_anchor_exists() -> None:
    """REQ-HW-071: OpenSpec anchors the MMD comparison artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-071" in spec
    assert "SCENARIO-HW-071" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "per_seed_mmd_pvalue" in spec


def test_recover_exp2898_problems_verifies_regenerated_checksums() -> None:
    """REQ-HW-071: Exp 2898 dense J/h checksums must reproduce from seeds."""

    problems = exp.recover_exp2898_problems(_exp2898_payload())

    assert [problem.seed for problem in problems] == exp.RANDOM_SEEDS
    assert problems[0].j_matrix.shape == (64, 64)
    assert problems[0].h_vector.shape == (64,)
    assert len(problems[0].j_matrix_sha256) == 64
    assert len(problems[0].h_vector_sha256) == 64

    corrupt = _exp2898_payload()
    corrupt["problem_payload"]["ising_problem_specs"][0]["j_matrix_sha256"] = "0" * 64
    with pytest.raises(exp.ProblemReproductionError, match="j_matrix_sha256"):
        exp.recover_exp2898_problems(corrupt)


def test_cpu_sequential_gibbs_is_reproducible_and_uses_random_order() -> None:
    """REQ-HW-071: CPU baseline uses reproducible random-order sequential Gibbs."""

    problem = exp.recover_exp2898_problems(_exp2898_payload())[0]

    first = exp.run_cpu_sequential_gibbs(problem, n_samples=8, burn_in_sweeps=3)
    second = exp.run_cpu_sequential_gibbs(problem, n_samples=8, burn_in_sweeps=3)

    assert first.energies == second.energies
    assert first.energy_sha256 == second.energy_sha256
    assert first.update_schedule == exp.CPU_UPDATE_SCHEDULE
    assert first.spin_orders_sha256 == second.spin_orders_sha256
    assert len(first.energies) == 8


def test_mmd_and_ks_detect_separated_energy_distributions() -> None:
    """SCENARIO-HW-071: MMD/KS p-values distinguish incompatible energies."""

    cpu = np.linspace(-3.0, -2.0, 30)
    kv260 = np.linspace(3.0, 4.0, 30)

    comparison = exp.compare_energy_distributions(
        cpu,
        kv260,
        seed=42,
        n_permutations=99,
        max_permutation_samples=60,
    )

    assert comparison["mmd_squared"] > 0.1
    assert comparison["mmd_pvalue"] < 0.05
    assert comparison["ks_statistic"] == pytest.approx(1.0)
    assert comparison["ks_pvalue"] < 0.01


def test_run_experiment_blocks_missing_exp2898(tmp_path: Path) -> None:
    """REQ-HW-071: missing Exp 2898 provenance fails closed."""

    artifact = exp.run_experiment(
        root_path=tmp_path,
        cpu_n_samples=4,
        cpu_burn_in_sweeps=1,
        n_permutations=9,
        started_s=10.0,
        now_s=11.0,
    )

    assert artifact["honest_verdict"] == "blocked_exp2898_artifact_missing"
    assert artifact["distributions_distinguishable"] is False
    assert artifact["paper_v6_recommendation"].startswith("blocked")
    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()


def test_run_experiment_with_fake_hardware_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-071: three-seed MMD/KS verdict drives the paper recommendation."""

    _write_exp2898(tmp_path)

    def fake_preconditions(_: list[exp.DenseIsingProblem]) -> exp.HardwareRunResult:
        return exp.HardwareRunResult(
            preconditions_checked=[
                {"resource": "kv260_ssh", "available": True, "detail": "test"},
                {"resource": "active_bitstream_sha256", "available": True, "detail": "a" * 64},
            ],
            bitstream_sha256="a" * 64,
            energies_by_seed={
                42: [4.0] * 12,
                137: [4.5] * 12,
                271: [5.0] * 12,
            },
        )

    artifact = exp.run_experiment(
        root_path=tmp_path,
        hardware_runner=fake_preconditions,
        cpu_energy_runner=lambda problem, n_samples, burn_in_sweeps: exp.EnergyRunResult(
            seed=problem.seed,
            energies=[-float(index + 1) for index in range(n_samples)],
            energy_sha256=exp.sha256_canonical([problem.seed, n_samples]),
            update_schedule=exp.CPU_UPDATE_SCHEDULE,
            spin_orders_sha256=exp.sha256_canonical([problem.seed, "orders"]),
        ),
        cpu_n_samples=12,
        cpu_burn_in_sweeps=2,
        n_permutations=19,
        max_permutation_samples=24,
        started_s=0.0,
        now_s=61.0,
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert len(artifact["per_seed_mmd_squared"]) == 3
    assert len(artifact["per_seed_mmd_pvalue"]) == 3
    assert len(artifact["per_seed_ks_statistic"]) == 3
    assert len(artifact["per_seed_ks_pvalue"]) == 3
    assert artifact["distributions_distinguishable"] is True
    assert artifact["paper_v6_recommendation"].startswith("retract")
    assert artifact["random_seeds_used"] == [42, 137, 271]
    assert artifact["duration_s"] == pytest.approx(61.0)

    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_validate_artifact_rejects_incomplete_success() -> None:
    """REQ-HW-071: success artifacts require three-seed statistics and duration."""

    artifact = exp.blocked_artifact(
        verdict="blocked_test",
        preconditions_checked=[],
        duration_s=1.0,
        recommendation="blocked test",
    )
    artifact.update(
        {
            "honest_verdict": "complete: bad",
            "per_seed_mmd_squared": [1.0],
            "per_seed_mmd_pvalue": [0.5],
            "per_seed_ks_statistic": [0.1],
            "per_seed_ks_pvalue": [0.5],
            "duration_s": 1.0,
        }
    )

    with pytest.raises(ValueError, match="three seed"):
        exp.validate_artifact(artifact)


def test_recover_exp2898_problems_rejects_malformed_payloads() -> None:
    """REQ-HW-071: malformed Exp 2898 provenance never becomes comparison data."""

    with pytest.raises(exp.ProblemReproductionError, match="problem_payload"):
        exp.recover_exp2898_problems({})

    wrong_n = _exp2898_payload()
    wrong_n["problem_payload"]["n_spins"] = 63
    with pytest.raises(exp.ProblemReproductionError, match="n_spins"):
        exp.recover_exp2898_problems(wrong_n)

    wrong_seeds = _exp2898_payload()
    wrong_seeds["problem_payload"]["random_seeds_used"] = [42]
    with pytest.raises(exp.ProblemReproductionError, match="random_seeds_used"):
        exp.recover_exp2898_problems(wrong_seeds)

    missing_seed = _exp2898_payload()
    missing_seed["problem_payload"]["problems"] = missing_seed["problem_payload"]["problems"][:2]
    with pytest.raises(exp.ProblemReproductionError, match="seed 271"):
        exp.recover_exp2898_problems(missing_seed)

    bad_j = _exp2898_payload()
    bad_j["problem_payload"]["problems"][0]["j_matrix"][0][1] += 1.0
    with pytest.raises(exp.ProblemReproductionError, match="artifact j_matrix_sha256"):
        exp.recover_exp2898_problems(bad_j)

    bad_h = _exp2898_payload()
    bad_h["problem_payload"]["problems"][0]["h_vector"][0] = 1.0
    with pytest.raises(exp.ProblemReproductionError, match="artifact h_vector_sha256"):
        exp.recover_exp2898_problems(bad_h)

    missing_upload = _exp2898_payload()
    missing_upload["problem_payload"]["problems"][0].pop("upload")
    with pytest.raises(exp.ProblemReproductionError, match="upload"):
        exp.recover_exp2898_problems(missing_upload)


def test_sampler_and_statistic_error_paths() -> None:
    """REQ-HW-071: helper functions fail loudly on invalid statistical inputs."""

    problem = exp.recover_exp2898_problems(_exp2898_payload())[0]

    with pytest.raises(ValueError, match="n_samples"):
        exp.run_cpu_sequential_gibbs(problem, n_samples=0)
    with pytest.raises(ValueError, match="burn_in"):
        exp.run_cpu_sequential_gibbs(problem, n_samples=1, burn_in_sweeps=-1)

    assert exp.median_pairwise_distance(np.array([1.0])) == pytest.approx(1.0)
    assert exp.median_pairwise_distance(np.array([2.0, 2.0])) == pytest.approx(1.0)
    assert exp.median_pairwise_distance(np.array([0.0, 1.0, 2.0])) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="non-empty"):
        exp.mmd_squared_rbf([], [1.0], 1.0)
    with pytest.raises(ValueError, match="n_permutations"):
        exp.mmd_permutation_pvalue([1.0], [2.0], bandwidth=1.0, seed=1, n_permutations=0)

    rng = np.random.default_rng(1)
    xs, ys = exp._balanced_subset(
        np.arange(20.0),
        np.arange(20.0) + 1.0,
        max_permutation_samples=10,
        rng=rng,
    )
    assert len(xs) == 5
    assert len(ys) == 5

    assert (
        exp.mmd_permutation_pvalue(
            np.arange(700.0),
            np.arange(700.0),
            bandwidth=1.0,
            seed=2,
            n_permutations=2,
            max_permutation_samples=800,
        )
        >= 0.0
    )
    assert exp.mmd_permutation_pvalue(
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        bandwidth=1.0,
        seed=3,
        n_permutations=2,
        max_permutation_samples=6,
    ) == pytest.approx(1.0)


def test_artifact_validation_and_small_helpers() -> None:
    """REQ-HW-071: schema helpers enforce terminal fields and provenance parsing."""

    assert exp._recommendation(False).startswith("retain")
    assert exp._precondition("thing", 1, "ok") == {
        "resource": "thing",
        "available": True,
        "detail": "ok",
    }
    assert exp._detect_overlay("loaded carnot_ising_v4") == "carnot_ising_v4"
    assert exp._detect_overlay("none") is None
    sha = "a" * 64
    assert exp._parse_sha256sum(f"{sha}  /tmp/a.bit\n") == (sha, "/tmp/a.bit")
    assert exp._parse_sha256sum("no sha here") == (None, None)

    problem = exp.recover_exp2898_problems(_exp2898_payload())[0]
    payload = exp._problem_payload_for_board([problem], 7)
    assert payload["n_samples"] == 7
    assert payload["problems"][0]["random_seed"] == 42

    valid_blocked = exp.blocked_artifact(
        verdict="blocked_test",
        preconditions_checked=[],
        duration_s=0.1,
        recommendation="blocked",
    )
    exp.validate_artifact(valid_blocked)

    missing = dict(valid_blocked)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing"):
        exp.validate_artifact(missing)

    bad_prefix = dict(valid_blocked, honest_verdict="bad")
    with pytest.raises(ValueError, match="terminal"):
        exp.validate_artifact(bad_prefix)

    bad_substrate = dict(valid_blocked, honest_verdict="complete: ok", inference_substrate="cpu")
    with pytest.raises(ValueError, match="hardware_smoke"):
        exp.validate_artifact(bad_substrate)

    bad_duration = dict(valid_blocked, honest_verdict="complete: ok", duration_s=1.0)
    bad_duration.update(
        {
            "per_seed_mmd_squared": [0.0, 0.0, 0.0],
            "per_seed_mmd_pvalue": [1.0, 1.0, 1.0],
            "per_seed_ks_statistic": [0.0, 0.0, 0.0],
            "per_seed_ks_pvalue": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="duration"):
        exp.validate_artifact(bad_duration)

    bad_seeds = dict(bad_duration, duration_s=60.0, random_seeds_used=[42])
    with pytest.raises(ValueError, match="random_seeds"):
        exp.validate_artifact(bad_seeds)

    bad_sha = dict(bad_duration, duration_s=60.0)
    bad_sha["cpu_sequential_gibbs_energies_sha256"] = "short"
    with pytest.raises(ValueError, match="sha256"):
        exp.validate_artifact(bad_sha)


def test_run_experiment_blocked_branches(tmp_path: Path) -> None:
    """REQ-HW-071: precondition failures stop before success recommendation."""

    malformed = _exp2898_payload()
    malformed["problem_payload"]["random_seeds_used"] = [42]
    _write_exp2898(tmp_path, malformed)
    artifact = exp.run_experiment(root_path=tmp_path, started_s=0.0, now_s=1.0)
    assert artifact["honest_verdict"] == "blocked_exp2898_problem_reproduction_failed"

    _write_exp2898(tmp_path)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        hardware_runner=lambda _: exp.HardwareRunResult(
            preconditions_checked=[],
            bitstream_sha256="",
            energies_by_seed={},
            blocked_verdict="blocked_kv260_ssh_unreachable",
        ),
        started_s=0.0,
        now_s=1.0,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"

    artifact = exp.run_experiment(
        root_path=tmp_path,
        hardware_runner=lambda _: exp.HardwareRunResult(
            preconditions_checked=[],
            bitstream_sha256="c" * 64,
            energies_by_seed={},
        ),
        started_s=0.0,
        now_s=1.0,
    )
    assert artifact["honest_verdict"] == "blocked_active_bitstream_sha256_mismatch"

    artifact = exp.run_experiment(
        root_path=tmp_path,
        hardware_runner=lambda _: exp.HardwareRunResult(
            preconditions_checked=[],
            bitstream_sha256="a" * 64,
            energies_by_seed={42: [1.0]},
        ),
        cpu_energy_runner=lambda problem, n_samples, burn_in_sweeps: exp.EnergyRunResult(
            seed=problem.seed,
            energies=[0.0] * n_samples,
            energy_sha256="d" * 64,
            update_schedule=exp.CPU_UPDATE_SCHEDULE,
        ),
        cpu_n_samples=2,
        cpu_burn_in_sweeps=0,
        started_s=0.0,
        now_s=1.0,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_energy_trace_incomplete"


def test_main_outputs_summary_and_result_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-071: the module CLI reports the deliverable path."""

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root_path: {"honest_verdict": "complete: cli-ok"},
    )

    assert exp.main(["--root", str(tmp_path)]) == 0
    assert "complete: cli-ok" in capsys.readouterr().out

    assert exp.main(["--root", str(tmp_path), "--print-result-path"]) == 0
    assert str(tmp_path / exp.OUTPUT_REL_PATH) in capsys.readouterr().out
