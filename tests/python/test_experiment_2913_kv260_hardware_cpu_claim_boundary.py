"""Tests for Exp 2913 KV260 hardware/CPU claim boundary.

REQ-HW-065: Exp 2913 must compare only matched KV260 and CPU per-sample
latency evidence before allowing a numeric hardware speedup claim.
SCENARIO-HW-065: matched Exp 2898 and Exp 2912 artifacts produce a bounded
matrix row and paper claim boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import kv260_hardware_cpu_claim_boundary as exp


def _upload(seed: int, *, n_spins: int = 64) -> dict[str, Any]:
    max_degree = 2
    adjacency = [
        [int((row + offset + 1 + seed) % n_spins) for offset in range(max_degree)]
        for row in range(n_spins)
    ]
    couplings = [[64 + (seed % 5), -32] for _ in range(n_spins)]
    return {
        "max_degree": max_degree,
        "h_q88": [seed % 7 for _ in range(n_spins)],
        "adjacency": adjacency,
        "couplings_q88": couplings,
    }


def _exp2898_payload(
    *,
    seeds: list[int] | None = None,
    sample_counts: list[int] | None = None,
    n_spins: int = 64,
) -> dict[str, Any]:
    seeds = seeds or [42, 137]
    sample_counts = sample_counts or [100, 1000]
    kv_latencies = {
        (42, 100): (20.0, 25.0),
        (137, 100): (30.0, 32.0),
        (42, 1000): (40.0, 44.0),
        (137, 1000): (50.0, 55.0),
    }
    return {
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "ising_problem_spec": {"n_spins": n_spins},
        "random_seeds_used": seeds,
        "problem_payload": {
            "n_spins": n_spins,
            "random_seeds_used": seeds,
            "n_sample_counts": sample_counts,
            "problems": [
                {
                    "random_seed": seed,
                    "n_spins": n_spins,
                    "upload": _upload(seed, n_spins=n_spins),
                }
                for seed in seeds
            ],
        },
        "sample_count_sweep_results": [
            {
                "seed": seed,
                "n_samples": count,
                "per_sample_wall_clock_us_median": kv_latencies.get((seed, count), (20.0, 25.0))[0],
                "per_sample_wall_clock_us_p95": kv_latencies.get((seed, count), (20.0, 25.0))[1],
            }
            for seed in seeds
            for count in sample_counts
        ],
    }


def _exp2912_payload(
    *,
    seeds: list[int] | None = None,
    sample_counts: list[int] | None = None,
    ready: bool = True,
    n_spins: int = 64,
) -> dict[str, Any]:
    seeds = seeds or [42, 137]
    sample_counts = sample_counts or [100, 1000]
    cpu_latencies = {
        (42, 100): (200.0, 250.0),
        (137, 100): (330.0, 320.0),
        (42, 1000): (440.0, 528.0),
        (137, 1000): (600.0, 660.0),
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        upload = _upload(seed, n_spins=n_spins)
        topo = exp.sha256_canonical(upload["adjacency"])
        couplings = exp.sha256_canonical(upload["couplings_q88"])
        fields = exp.sha256_canonical(upload["h_q88"])
        for count in sample_counts:
            median_us, p95_us = cpu_latencies.get((seed, count), (200.0, 250.0))
            rows.append(
                {
                    "seed": seed,
                    "sample_count": count,
                    "n_spins": n_spins,
                    "cpu_latency_us_median": median_us,
                    "cpu_latency_us_p95": p95_us,
                    "matched_sparse_topology_checksum": topo,
                    "matched_coupling_tensor_checksum": couplings,
                    "matched_field_tensor_checksum": fields,
                }
            )
    return {
        "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
        "same_basis_cpu_baseline_ready": ready,
        "n_spins": n_spins,
        "matched_sparse_topology": ready,
        "matched_coupling_tensor": ready,
        "matched_field_tensor": ready,
        "random_seeds_used": seeds,
        "sample_count_sweep": sample_counts,
        "cpu_per_seed_results": rows,
        "inference_substrate": "cpu_sampler",
        "speedup_claim_made": False,
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
    _write_json(root, exp.KV260_ARTIFACT_REL_PATH, kv260 or _exp2898_payload())
    _write_json(root, exp.CPU_ARTIFACT_REL_PATH, cpu or _exp2912_payload())


def test_req_hw_065_spec_anchor_exists() -> None:
    """REQ-HW-065: OpenSpec defines the claim-boundary contract."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/fpga/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-HW-065" in spec
    assert "SCENARIO-HW-065" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert exp.INFERENCE_SUBSTRATE in spec


def test_req_hw_065_blocks_when_cpu_baseline_is_not_ready(tmp_path: Path) -> None:
    """REQ-HW-065: an unready Exp 2912 artifact stops before speedup math."""

    _write_json(
        tmp_path,
        exp.CPU_ARTIFACT_REL_PATH,
        _exp2912_payload(ready=False, sample_counts=[]),
    )

    artifact = exp.run_experiment(tmp_path, started_s=10.0, now_s=11.5)

    assert artifact["honest_verdict"] == "blocked_cpu_baseline_not_ready"
    assert artifact["kv260_claim_boundary_ready"] is False
    assert artifact["same_basis_verified"] is False
    assert artifact["hardware_speedup_claim_eligible"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["speedup_ratio_median_by_sample_count"] == {}
    assert "same_basis_cpu_baseline_ready is not true" in artifact["comparison_notes"][0]
    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()


def test_scenario_hw_065_matched_artifacts_emit_speedup_boundary(tmp_path: Path) -> None:
    """SCENARIO-HW-065: matched latency rows allow a scoped numeric speedup."""

    _write_upstreams(tmp_path)

    artifact = exp.run_experiment(tmp_path, started_s=20.0, now_s=23.25)

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["kv260_claim_boundary_ready"] is True
    assert artifact["same_basis_verified"] is True
    assert artifact["hardware_speedup_claim_eligible"] is True
    assert artifact["speedup_claim_made"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["run_date"] == "20260523"
    assert artifact["speedup_ratio_median_by_sample_count"] == {
        "100": pytest.approx(10.5),
        "1000": pytest.approx(11.5),
    }
    assert artifact["speedup_ratio_p95_by_sample_count"] == {
        "100": pytest.approx(10.0),
        "1000": pytest.approx(12.0),
    }
    assert len(artifact["per_seed_speedup_ratios"]) == 4
    assert artifact["matrix_row_candidate"]["eligible_for_matrix_v9"] is True
    assert artifact["matrix_row_candidate"]["claim_scope"] == (
        "matched n=64 sparse Ising KV260 hardware-smoke versus CPU Gibbs baseline"
    )
    assert "matched n=64 sparse Ising workload" in artifact["paper_claim_boundary"]
    assert "not a broad FPGA acceleration claim" in artifact["paper_claim_boundary"]

    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_hw_065_mismatch_forbids_numeric_speedup(tmp_path: Path) -> None:
    """REQ-HW-065: sample-count mismatches name the failed gate and clear ratios."""

    _write_upstreams(tmp_path, cpu=_exp2912_payload(sample_counts=[100]))

    artifact = exp.run_experiment(tmp_path, started_s=5.0, now_s=5.75)

    assert artifact["honest_verdict"] == "complete: kv260_claim_boundary_ready_no_speedup_claim"
    assert artifact["kv260_claim_boundary_ready"] is True
    assert artifact["same_basis_verified"] is False
    assert artifact["hardware_speedup_claim_eligible"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["speedup_ratio_median_by_sample_count"] == {}
    assert "sample_counts_mismatch" in artifact["comparison_notes"]
    assert "No numeric KV260/CPU speedup claim is eligible" in artifact[
        "paper_claim_boundary"
    ]


def test_req_hw_065_timing_unit_mismatch_forbids_speedup(tmp_path: Path) -> None:
    """REQ-HW-065: latency ratios require per-sample microsecond fields."""

    cpu = _exp2912_payload()
    cpu["cpu_per_seed_results"][0].pop("cpu_latency_us_median")
    _write_upstreams(tmp_path, cpu=cpu)

    artifact = exp.run_experiment(tmp_path)

    assert artifact["same_basis_verified"] is True
    assert artifact["hardware_speedup_claim_eligible"] is False
    assert "timing_units_or_latency_fields_mismatch" in artifact["comparison_notes"]
    assert artifact["speedup_claim_made"] is False


def test_req_hw_065_missing_kv260_artifact_blocks_boundary(tmp_path: Path) -> None:
    """REQ-HW-065: CPU evidence alone cannot create a KV260 claim boundary."""

    _write_json(tmp_path, exp.CPU_ARTIFACT_REL_PATH, _exp2912_payload())

    artifact = exp.run_experiment(tmp_path)

    assert artifact["honest_verdict"] == "blocked_kv260_artifact_not_ready"
    assert artifact["kv260_claim_boundary_ready"] is False
    assert artifact["matrix_row_candidate"] == {}
    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()


def test_validate_artifact_rejects_schema_drift(tmp_path: Path) -> None:
    """REQ-HW-065: terminal artifacts keep the required claim-boundary schema."""

    _write_upstreams(tmp_path)
    valid = exp.run_experiment(tmp_path, write=False)
    exp.validate_artifact(valid)

    missing = dict(valid)
    missing.pop("run_date")
    with pytest.raises(ValueError, match="missing"):
        exp.validate_artifact(missing)

    bad_substrate = dict(valid, inference_substrate="hardware_smoke")
    with pytest.raises(ValueError, match=exp.INFERENCE_SUBSTRATE):
        exp.validate_artifact(bad_substrate)

    bad_date = dict(valid, run_date="20260522")
    with pytest.raises(ValueError, match="20260523"):
        exp.validate_artifact(bad_date)

    bad_claim = dict(valid, hardware_speedup_claim_eligible=False)
    with pytest.raises(ValueError, match="speedup_claim_made"):
        exp.validate_artifact(bad_claim)

    bad_same_basis = dict(valid, same_basis_verified=False)
    with pytest.raises(ValueError, match="same_basis_verified"):
        exp.validate_artifact(bad_same_basis)

    bad_ratios = dict(valid, speedup_ratio_median_by_sample_count={})
    with pytest.raises(ValueError, match="eligible speedup"):
        exp.validate_artifact(bad_ratios)

    bad_p95 = dict(valid, speedup_ratio_p95_by_sample_count={})
    with pytest.raises(ValueError, match="p95 ratios"):
        exp.validate_artifact(bad_p95)


def test_helper_and_cli_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-065: helpers and CLI fail closed without guessing."""

    assert exp._median([1.0, 3.0]) == pytest.approx(2.0)
    assert exp._median([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert exp._p95([1.0, 3.0]) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="median"):
        exp._median([])
    with pytest.raises(ValueError, match="p95"):
        exp._p95([])
    assert exp._positive_float({"value": 1.5}, "value") is True
    assert exp._positive_float({"value": 0.0}, "value") is False
    assert exp._seed_list({"problem_payload": {"random_seeds_used": [7, 8]}}) == [7, 8]
    assert exp._seed_list({"random_seeds_used": ["bad"]}) == []
    assert exp._n_spins({"n_spins": 12}) == 12
    assert exp._n_spins({"ising_problem_spec": {"n_spins": 13}}) == 13
    assert exp._n_spins({}) is None
    assert exp._kv260_latency_rows({"sample_count_sweep_results": "bad"}) == {}
    assert exp._cpu_latency_rows({"cpu_per_seed_results": "bad"}) == {}
    assert exp._kv260_basis_checksums({"problem_payload": []}) == {}
    assert exp._kv260_basis_checksums({"problem_payload": {"problems": "bad"}}) == {}
    assert (
        exp._kv260_basis_checksums(
            {
                "problem_payload": {
                    "problems": [
                        None,
                        {"random_seed": "bad"},
                        {"random_seed": 1},
                        {"random_seed": 2, "upload": []},
                        {"random_seed": 3, "upload": {"adjacency": []}},
                    ]
                }
            }
        )
        == {}
    )
    basis_notes = exp._basis_match_notes(
        {1: {"topology": "topo", "couplings": "couplings", "fields": "fields"}},
        {
            (1, 100): {
                "matched_sparse_topology_checksum": "wrong",
                "matched_coupling_tensor_checksum": "wrong",
                "matched_field_tensor_checksum": "wrong",
            },
            (2, 100): {},
        },
        {
            "matched_sparse_topology": False,
            "matched_coupling_tensor": False,
            "matched_field_tensor": False,
        },
    )
    assert "sparse_topology_not_marked_matched_by_exp2912" in basis_notes
    assert "coupling_tensor_not_marked_matched_by_exp2912" in basis_notes
    assert "field_tensor_not_marked_matched_by_exp2912" in basis_notes
    assert "sparse_topology_mismatch" in basis_notes
    assert "coupling_tensor_mismatch" in basis_notes
    assert "field_tensor_mismatch" in basis_notes
    mismatch = exp.build_artifact(
        _exp2898_payload(seeds=[42], sample_counts=[100], n_spins=63),
        _exp2912_payload(seeds=[137], sample_counts=[100], n_spins=64),
        duration_s=0.0,
    )
    assert "n_spins_mismatch" in mismatch["comparison_notes"]
    assert "seeds_mismatch" in mismatch["comparison_notes"]
    assert exp._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert exp._read_json(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[1]", encoding="utf-8")
    assert exp._read_json(list_json) == {}

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root_path: {"honest_verdict": "complete: cli-ok"},
    )
    assert exp.main(["--root", str(tmp_path)]) == 0
    assert "complete: cli-ok" in capsys.readouterr().out
