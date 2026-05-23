"""Tests for Exp 2912 KV260 same-basis CPU Gibbs baseline.

REQ-HW-064: Exp 2912 must recover the exact Exp 2898 uploaded sparse
Ising basis, run a CPU Gibbs baseline on that basis, and write a no-speedup
artifact.
SCENARIO-HW-064: a recoverable Exp 2898 artifact produces same-basis CPU
latency and energy provenance for the matched sample counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import kv260_same_basis_cpu_gibbs_baseline as exp


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


def _exp2898_payload(
    *,
    seeds: list[int] | None = None,
    sample_counts: list[int] | None = None,
    upload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seeds = seeds or [42]
    sample_counts = sample_counts or [2, 3]
    upload = upload or _upload()
    problems = []
    specs = []
    for seed in seeds:
        problems.append(
            {
                "n_spins": 64,
                "random_seed": seed,
                "beta_final_q88": 256,
                "h_vector": [0.0 for _ in range(64)],
                "j_matrix": [[0.0 for _ in range(64)] for _ in range(64)],
                "upload": upload,
            }
        )
        specs.append(
            {
                "n_spins": 64,
                "random_seed": seed,
                "j_matrix_sha256": f"{seed:064x}"[-64:],
                "h_vector_sha256": "0" * 64,
            }
        )
    return {
        "experiment_id": 2898,
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "ising_problem_spec": {"n_spins": 64, "all_seed_specs": specs},
        "problem_payload": {
            "n_spins": 64,
            "max_degree_uploaded": upload["max_degree"],
            "random_seeds_used": seeds,
            "n_sample_counts": sample_counts,
            "ising_problem_specs": specs,
            "problems": problems,
        },
        "random_seeds_used": seeds,
        "sample_count_sweep_results": [
            {"seed": seed, "n_samples": count}
            for seed in seeds
            for count in sample_counts
        ],
        "reproducibility_checksum": "a" * 64,
    }


def _write_upstream(root: Path, payload: dict[str, Any]) -> Path:
    path = root / exp.UPSTREAM_KV260_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_missing_exp2898_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-HW-064: missing upstream artifacts fail closed."""
    artifact = exp.run_experiment(root_path=tmp_path)

    assert artifact["honest_verdict"] == "blocked_kv260_latency_artifact_missing"
    assert artifact["same_basis_cpu_baseline_ready"] is False
    assert artifact["speedup_claim_made"] is False
    assert (tmp_path / exp.RESULT_ARTIFACT).exists()


def test_unrecoverable_basis_names_missing_tensor(tmp_path: Path) -> None:
    """REQ-HW-064: unrecoverable exact tensors are not synthesized silently."""
    payload = _exp2898_payload()
    del payload["problem_payload"]["problems"][0]["upload"]["couplings_q88"]
    _write_upstream(tmp_path, payload)

    artifact = exp.run_experiment(root_path=tmp_path, sample_counts=(2,))

    assert artifact["honest_verdict"] == "blocked_kv260_problem_basis_unrecoverable"
    assert artifact["same_basis_cpu_baseline_ready"] is False
    assert "problem_payload.problems[0].upload.couplings_q88" in artifact[
        "missing_problem_basis_fields"
    ]


def test_recover_basis_checks_uploaded_tensors() -> None:
    """SCENARIO-HW-064: recovered basis carries sparse topology and q8.8 checksums."""
    payload = _exp2898_payload(seeds=[42, 137])
    basis = exp.recover_problem_basis(payload)

    assert [item.seed for item in basis.problems] == [42, 137]
    assert basis.n_spins == 64
    assert basis.sample_count_sweep == [2, 3]
    assert basis.problems[0].adjacency.shape == (64, 2)
    assert basis.problems[0].couplings_q88.shape == (64, 2)
    assert len(basis.problems[0].coupling_tensor_checksum) == 64
    assert len(basis.problems[0].field_tensor_checksum) == 64


def test_cpu_gibbs_result_is_reproducible_for_same_basis() -> None:
    """REQ-HW-064: CPU Gibbs provenance is reproducible for a fixed seed and basis."""
    basis = exp.recover_problem_basis(_exp2898_payload()).problems[0]

    first = exp.run_cpu_gibbs_for_problem(basis, sample_count=4)
    second = exp.run_cpu_gibbs_for_problem(basis, sample_count=4)

    assert first["seed"] == 42
    assert first["sample_count"] == 4
    assert first["update_schedule"] == exp.CPU_UPDATE_SCHEDULE
    assert first["energy_trace_checksum"] == second["energy_trace_checksum"]
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert first["cpu_latency_us_median"] > 0.0
    assert first["cpu_latency_us_p95"] > 0.0


def test_run_experiment_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-064: recoverable Exp 2898 tensors produce the deliverable schema."""
    _write_upstream(tmp_path, _exp2898_payload(seeds=[42, 137], sample_counts=[2, 3]))

    artifact = exp.run_experiment(root_path=tmp_path, sample_counts=(2, 3))

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["same_basis_cpu_baseline_ready"] is True
    assert artifact["upstream_kv260_artifact"] == exp.UPSTREAM_KV260_ARTIFACT.as_posix()
    assert artifact["n_spins"] == 64
    assert artifact["matched_sparse_topology"] is True
    assert artifact["matched_coupling_tensor"] is True
    assert artifact["matched_field_tensor"] is True
    assert artifact["random_seeds_used"] == [42, 137]
    assert artifact["sample_count_sweep"] == [2, 3]
    assert len(artifact["cpu_per_seed_results"]) == 4
    assert set(artifact["cpu_latency_us_median_by_sample_count"]) == {"2", "3"}
    assert set(artifact["cpu_latency_us_p95_by_sample_count"]) == {"2", "3"}
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["speedup_claim_made"] is False
    assert artifact["inference_substrate"] == "cpu_sampler"
    assert artifact["run_date"] == "20260523"

    saved = json.loads((tmp_path / exp.RESULT_ARTIFACT).read_text(encoding="utf-8"))
    assert saved == artifact


def test_validate_success_artifact_rejects_speedup_claim() -> None:
    """REQ-HW-064: terminal artifacts must not make a hardware speedup claim."""
    payload = {
        field: None for field in exp.REQUIRED_ARTIFACT_FIELDS
    }
    payload.update(
        {
            "honest_verdict": "complete: cpu_baseline_ready_no_speedup_claim",
            "same_basis_cpu_baseline_ready": True,
            "upstream_kv260_artifact": exp.UPSTREAM_KV260_ARTIFACT.as_posix(),
            "n_spins": 64,
            "matched_sparse_topology": True,
            "matched_coupling_tensor": True,
            "matched_field_tensor": True,
            "random_seeds_used": [42],
            "sample_count_sweep": [100],
            "cpu_per_seed_results": [{"sample_count": 100}],
            "cpu_latency_us_median_by_sample_count": {"100": 1.0},
            "cpu_latency_us_p95_by_sample_count": {"100": 2.0},
            "reproducibility_checksum": "b" * 64,
            "speedup_claim_made": True,
            "inference_substrate": "cpu_sampler",
            "duration_s": 0.1,
            "run_date": "20260523",
        }
    )

    with pytest.raises(ValueError, match="speedup"):
        exp.validate_artifact(payload)


def test_helper_error_and_fallback_paths() -> None:
    """REQ-HW-064: helper validation paths fail loudly instead of guessing."""
    with pytest.raises(ValueError, match="median"):
        exp._median([])
    with pytest.raises(ValueError, match="p95"):
        exp._p95([])

    assert exp._sigmoid(-1.0) == pytest.approx(0.26894142137)

    missing: list[str] = []
    exp._int_array("not-int", (1,), "bad.int", missing)
    assert missing == ["bad.int"]

    missing = []
    exp._int_array([1, 2], (1,), "bad.shape", missing)
    assert missing == ["bad.shape"]

    assert exp._sample_counts_from_sweep({}) == []
    assert exp._sample_counts_from_sweep(
        {"sample_count_sweep_results": [{"n_samples": 3}, {"n_samples": 2}, {"x": 1}]}
    ) == [2, 3]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda payload: payload.pop("problem_payload"), "problem_payload"),
        (
            lambda payload: payload["problem_payload"].update({"problems": []}),
            "problem_payload.problems",
        ),
        (
            lambda payload: payload["problem_payload"].update({"problems": [None]}),
            "problem_payload.problems[0]",
        ),
        (
            lambda payload: payload["problem_payload"]["problems"][0].pop("upload"),
            "problem_payload.problems[0].upload",
        ),
    ],
)
def test_recover_problem_basis_rejects_structural_payload_errors(
    mutate: Any, expected: str
) -> None:
    """REQ-HW-064: malformed Exp 2898 structure blocks recovery."""
    payload = _exp2898_payload()
    mutate(payload)

    with pytest.raises(exp.ProblemBasisUnrecoverableError) as exc:
        exp.recover_problem_basis(payload)

    assert expected in exc.value.missing_fields


def test_recover_problem_basis_reports_malformed_tensor_metadata() -> None:
    """REQ-HW-064: malformed tensor metadata names every unrecoverable field."""
    payload = _exp2898_payload()
    problem = payload["problem_payload"]["problems"][0]
    payload["problem_payload"]["n_spins"] = 63
    payload["problem_payload"]["random_seeds_used"] = ["bad"]
    payload["problem_payload"].pop("n_sample_counts")
    payload["sample_count_sweep_results"] = []
    payload["problem_payload"]["ising_problem_specs"] = []
    problem["random_seed"] = "bad"
    problem["upload"]["max_degree"] = 0
    problem["beta_final_q88"] = "bad"

    with pytest.raises(exp.ProblemBasisUnrecoverableError) as exc:
        exp.recover_problem_basis(payload)

    fields = set(exc.value.missing_fields)
    assert "problem_payload.n_spins" in fields
    assert "problem_payload.random_seeds_used" in fields
    assert "problem_payload.n_sample_counts" in fields
    assert "problem_payload.problems[0].random_seed" in fields
    assert "problem_payload.problems[0].upload.max_degree" in fields
    assert "problem_payload.problems[0].j_matrix_sha256" in fields
    assert "problem_payload.problems[0].h_vector_sha256" in fields
    assert "problem_payload.problems[0].beta_final_q88" in fields


def test_recover_problem_basis_rejects_seed_order_mismatch() -> None:
    """REQ-HW-064: seed order must match the upstream Exp 2898 run order."""
    payload = _exp2898_payload(seeds=[42, 137])
    payload["problem_payload"]["random_seeds_used"] = [137, 42]

    with pytest.raises(exp.ProblemBasisUnrecoverableError) as exc:
        exp.recover_problem_basis(payload)

    assert exc.value.missing_fields == ["problem_payload.problems.random_seed_order"]


def test_cpu_gibbs_rejects_non_positive_sample_count() -> None:
    """REQ-HW-064: CPU Gibbs runs must use a positive sample count."""
    basis = exp.recover_problem_basis(_exp2898_payload()).problems[0]

    with pytest.raises(ValueError, match="sample_count"):
        exp.run_cpu_gibbs_for_problem(basis, sample_count=0)


def test_validate_artifact_failure_modes(tmp_path: Path) -> None:
    """REQ-HW-064: schema validation catches corrupt terminal artifacts."""
    _write_upstream(tmp_path, _exp2898_payload())
    valid = exp.run_experiment(root_path=tmp_path, sample_counts=(2,))
    exp.validate_artifact(valid)

    missing = dict(valid)
    missing.pop("run_date")
    with pytest.raises(ValueError, match="missing"):
        exp.validate_artifact(missing)

    bad_substrate = dict(valid, inference_substrate="hardware_smoke")
    with pytest.raises(ValueError, match="cpu_sampler"):
        exp.validate_artifact(bad_substrate)

    bad_date = dict(valid, run_date="20260522")
    with pytest.raises(ValueError, match="20260523"):
        exp.validate_artifact(bad_date)

    bad_n = dict(valid, n_spins=63)
    with pytest.raises(ValueError, match="64"):
        exp.validate_artifact(bad_n)

    bad_match = dict(valid, matched_sparse_topology=False)
    with pytest.raises(ValueError, match="matched_sparse_topology"):
        exp.validate_artifact(bad_match)

    no_results = dict(valid, cpu_per_seed_results=[])
    with pytest.raises(ValueError, match="cpu_per_seed_results"):
        exp.validate_artifact(no_results)

    bad_checksum = dict(valid, reproducibility_checksum="short")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)


def test_main_outputs_summary_and_result_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-064: the module CLI reports the written artifact path."""
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root_path: {"honest_verdict": "complete: cli-ok"},
    )

    assert exp.main(["--root", str(tmp_path)]) == 0
    assert "complete: cli-ok" in capsys.readouterr().out

    assert exp.main(["--root", str(tmp_path), "--print-result-path"]) == 0
    assert str(tmp_path / exp.RESULT_ARTIFACT) in capsys.readouterr().out
