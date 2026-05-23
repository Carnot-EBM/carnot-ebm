"""Tests for Exp 2930 KV260 p-bit/SSQA scaling projection.

Spec refs: REQ-HW-070, SCENARIO-HW-070.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import kv260_pbit_ssqa_scaling_projection as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def _upload(*, n_spins: int = 64, fanout: int = 16) -> dict[str, Any]:
    return {
        "layout": "ising_sampler_v2_n64_sparse_axi_q8_8",
        "max_degree": fanout,
        "adjacency": [
            [int((row + col + 1) % n_spins) for col in range(fanout)]
            for row in range(n_spins)
        ],
        "couplings_q88": [[64 for _ in range(fanout)] for _ in range(n_spins)],
        "h_q88": [0 for _ in range(n_spins)],
    }


def _exp2898_payload() -> dict[str, Any]:
    seeds = [42, 137, 271]
    sample_counts = [100, 1000, 10000]
    latency_by_count = {
        100: (24.0605, 24.78),
        1000: (24.06, 24.431),
        10000: (24.05, 24.38),
    }
    return {
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "random_seeds_used": seeds,
        "problem_payload": {
            "n_spins": 64,
            "max_degree_uploaded": 16,
            "random_seeds_used": seeds,
            "n_sample_counts": sample_counts,
            "problems": [
                {"random_seed": seed, "n_spins": 64, "upload": _upload()}
                for seed in seeds
            ],
        },
        "sample_count_sweep_results": [
            {
                "seed": seed,
                "n_samples": count,
                "per_sample_wall_clock_us_median": latency_by_count[count][0],
                "per_sample_wall_clock_us_p95": latency_by_count[count][1],
            }
            for seed in seeds
            for count in sample_counts
        ],
    }


def _exp2912_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
        "same_basis_cpu_baseline_ready": True,
        "n_spins": 64,
        "random_seeds_used": [42, 137, 271],
        "sample_count_sweep": [100, 1000, 10000],
        "cpu_latency_us_median_by_sample_count": {
            "100": 459.2165,
            "1000": 461.581,
            "10000": 448.4065,
        },
        "cpu_latency_us_p95_by_sample_count": {
            "100": 635.312,
            "1000": 741.561,
            "10000": 489.238,
        },
    }


def _exp2913_payload(*, eligible: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_same_basis_hardware_cpu_speedup_claim_eligible"
        if eligible
        else "complete: kv260_claim_boundary_ready_no_speedup_claim",
        "hardware_speedup_claim_eligible": eligible,
        "kv260_claim_boundary_ready": True,
        "same_basis_verified": eligible,
        "speedup_claim_made": eligible,
        "speedup_ratio_median_by_sample_count": {"100": 19.134021} if eligible else {},
        "speedup_ratio_p95_by_sample_count": {"100": 25.638095} if eligible else {},
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "run_date": "20260523",
    }


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_ready_sources(root: Path) -> None:
    _write_json(root, exp.EXP2913_REL_PATH, _exp2913_payload())
    _write_json(root, exp.EXP2898_REL_PATH, _exp2898_payload())
    _write_json(root, exp.EXP2912_REL_PATH, _exp2912_payload())


def test_req_hw_070_spec_anchor_exists() -> None:
    """REQ-HW-070: OpenSpec anchors the projection-only artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-070" in spec
    assert "SCENARIO-HW-070" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert exp.INFERENCE_SUBSTRATE in spec


def test_req_hw_070_blocks_without_clean_exp2913(tmp_path: Path) -> None:
    """REQ-HW-070: absent Exp 2913 blocks before any scaling estimate."""

    artifact = exp.run_experiment(tmp_path, started_s=10.0, now_s=11.25)

    assert artifact["honest_verdict"] == "blocked_clean_kv260_basis_missing"
    assert artifact["kv260_scaling_projection_ready"] is False
    assert artifact["projection_only"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["not_a_speedup_claim"] is True
    assert artifact["n128_projection"] == {}
    assert artifact["n256_projection"] == {}
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()


def test_req_hw_070_blocks_when_exp2913_is_not_speedup_eligible(tmp_path: Path) -> None:
    """REQ-HW-070: ineligible Exp 2913 evidence stops without fake projections."""

    _write_json(tmp_path, exp.EXP2913_REL_PATH, _exp2913_payload(eligible=False))

    artifact = exp.run_experiment(tmp_path, started_s=20.0, now_s=20.5)

    assert artifact["honest_verdict"] == "blocked_clean_kv260_basis_missing"
    assert artifact["kv260_scaling_projection_ready"] is False
    assert any(
        "hardware_speedup_claim_eligible was not true" in item
        for item in artifact["assumptions"]
    )
    assert artifact["projection_models"] == {}


def test_req_hw_070_memory_models_are_deterministic() -> None:
    """REQ-HW-070: memory accounting uses explicit spreadsheet-style formulae."""

    dense = exp.dense_memory_projection(128)
    sparse = exp.sparse_memory_projection(128, fanout=16)
    dual = exp.dual_bram_projection(128, fanout=16)

    assert dense["total_bits"] == 264320
    assert dense["total_bytes"] == 33040
    assert dense["bram36_blocks_min"] == 8
    assert sparse["neighbor_index_bits_per_entry"] == 7
    assert sparse["total_bits"] == 49280
    assert sparse["bram36_blocks_min"] == 2
    assert dual["bank_a_bram36_blocks_min"] == 2
    assert dual["bank_b_bram36_blocks_min"] == 1
    assert dual["bram36_blocks_min"] == 3


def test_scenario_hw_070_ready_artifact_contains_real_n64_evidence(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-070: clean n=64 evidence feeds projection-only summaries."""

    _write_ready_sources(tmp_path)

    artifact = exp.run_experiment(tmp_path, started_s=30.0, now_s=31.0)

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["kv260_scaling_projection_ready"] is True
    assert artifact["projection_only"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["not_a_speedup_claim"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["run_date"] == "20260523"

    n64 = artifact["n64_real_evidence_summary"]
    assert n64["n_spins"] == 64
    assert n64["sparse_fanout_k"] == 16
    assert n64["random_seeds_used"] == [42, 137, 271]
    assert n64["sample_count_sweep"] == [100, 1000, 10000]
    assert n64["kv260_latency_us_median_by_sample_count"]["100"] == pytest.approx(24.0605)
    assert n64["resource_fields_present_in_exp2913_upstreams"] is False
    assert "resource_unknowns" in n64

    assert "dense_q8_8" in artifact["projection_models"]
    assert "sparse_q8_8_k16" in artifact["projection_models"]
    assert "dual_bram_ssqa_delay_k16" in artifact["projection_models"]


def test_scenario_hw_070_n128_and_n256_resource_pressure(tmp_path: Path) -> None:
    """REQ-HW-070: n=128/n=256 projections mark fit pressure and unknown FFs."""

    _write_ready_sources(tmp_path)

    artifact = exp.run_experiment(tmp_path)

    n128 = artifact["n128_projection"]
    assert n128["dense_q8_8"]["memory"]["bram36_blocks_min"] == 8
    assert n128["dense_q8_8"]["lut_pressure"]["estimated_lut"] == 290000
    assert n128["dense_q8_8"]["lut_pressure"]["fits_kv260_lut_budget"] is False
    assert n128["sparse_q8_8_k16"]["memory"]["bram36_blocks_min"] == 2
    assert n128["sparse_q8_8_k16"]["lut_pressure"]["estimated_lut"] == 35872
    assert n128["sparse_q8_8_k16"]["lut_pressure"]["fits_kv260_lut_budget"] is True
    assert n128["dual_bram_ssqa_delay_k16"]["memory"]["bram36_blocks_min"] == 3
    assert n128["dual_bram_ssqa_delay_k16"]["ff_pressure"]["total_ff_estimate"] == "unknown"

    n256 = artifact["n256_projection"]
    assert n256["dense_q8_8"]["memory"]["total_bytes"] == 131616
    assert n256["dense_q8_8"]["memory"]["bram36_blocks_min"] == 29
    assert n256["dense_q8_8"]["lut_pressure"]["estimated_lut"] == 1160000
    assert n256["dense_q8_8"]["lut_pressure"]["fits_kv260_lut_budget"] is False
    assert n256["sparse_q8_8_k16"]["memory"]["total_bytes"] == 12832
    assert n256["sparse_q8_8_k16"]["lut_pressure"]["estimated_lut"] == 67744
    assert n256["sparse_q8_8_k16"]["lut_pressure"]["fits_kv260_lut_budget"] is True
    assert n256["dual_bram_ssqa_delay_k16"]["memory"]["bram36_blocks_min"] == 4


def test_scenario_hw_070_writes_stable_json_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-070: the saved deliverable exactly matches the returned payload."""

    _write_ready_sources(tmp_path)

    artifact = exp.run_experiment(tmp_path, started_s=40.0, now_s=42.0)
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["source_artifacts"] == [
        exp.EXP2913_REL_PATH.as_posix(),
        exp.EXP2898_REL_PATH.as_posix(),
        exp.EXP2912_REL_PATH.as_posix(),
        "results/experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json",
        "results/experiment_1320_pbit_sampler_portability_packet.json",
        "research-references.md",
        "research-hardware-wishlist.md",
        "hardware/kv260/ising_sampler_v4_spec.md",
        "hardware/kv260/ising_sampler_v3.v",
    ]


def test_req_hw_070_malformed_exp2898_uses_unknown_fallbacks() -> None:
    """REQ-HW-070: malformed upstream topology does not fabricate n=64 details."""

    artifact = exp.build_artifact(
        _exp2913_payload(),
        {"problem_payload": [], "sample_count_sweep_results": "not rows"},
        {"n_spins": 64, "random_seeds_used": [42], "sample_count_sweep": [100]},
        duration_s=0.0,
    )

    n64 = artifact["n64_real_evidence_summary"]
    assert n64["n_spins"] == 64
    assert n64["sparse_fanout_k"] == 16
    assert n64["random_seeds_used"] == [42]
    assert n64["sample_count_sweep"] == [100]
    assert n64["kv260_latency_us_median_by_sample_count"] == {}


def test_scenario_hw_070_cli_writes_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-HW-070: the module CLI emits the written artifact path."""

    _write_ready_sources(tmp_path)

    assert exp.main(["--root", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["honest_verdict"].startswith("complete:")
    assert output["result"] == str(tmp_path / exp.OUTPUT_REL_PATH)


def test_req_hw_070_validate_rejects_claim_invariant_break() -> None:
    """REQ-HW-070: schema validation refuses accidental hardware/speedup claims."""

    artifact = exp.blocked_artifact(
        duration_s=0.0,
        assumptions=["test"],
    )
    artifact["projection_only"] = False

    with pytest.raises(ValueError, match="projection_only"):
        exp.validate_artifact(artifact)
