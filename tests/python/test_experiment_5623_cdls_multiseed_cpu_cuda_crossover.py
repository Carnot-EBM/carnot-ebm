"""Tests for Exp 5623 corrected-cDLS multi-seed CPU/CUDA crossover.

Spec refs: REQ-SAMPLE-5623, SCENARIO-SAMPLE-5623.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622
from carnot import experiment_5623_cdls_multiseed_cpu_cuda_crossover as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5623_cdls_multiseed_cpu_cuda_crossover.py")


class StepClock:
    """Deterministic clock for stable timing and duration receipts."""

    def __init__(self) -> None:
        self.value = 5623.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


class FakeCuda:
    """CUDA-shaped receipt facade that never runs kernels in tests."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.sync_calls = 0

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Fake RTX 4090"

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (20 * 1024**3, 24 * 1024**3)

    def memory_reserved(self, index: int) -> int:
        assert index == 0
        return 512 * 1024**2

    def synchronize(self) -> None:
        self.sync_calls += 1


class FakeTensorRuntime:
    """Small tensor-runtime facade exposing only metadata used by receipts."""

    __version__ = "unit-runtime-1.0"

    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = FakeCuda(available=cuda_available)
        self.version = type("Version", (), {"cuda": "12.8"})()


def _write_ready_upstream(root: Path) -> None:
    artifact = exp5622.build_artifact(retained_samples=512, burn_in_steps=64)
    exp5622.write_output(root, artifact)


def _trace(base: float, samples: int, seed: int) -> list[float]:
    return [round(base + ((index + seed) % 11 - 5) * 0.002, 6) for index in range(samples)]


def _constraint_trace(samples: int, seed: int) -> list[float]:
    return [1.0 if (index + seed) % 17 else 0.0 for index in range(samples)]


def _fake_sampler_rows(
    instances: list[mod.IsingInstance],
    seeds: tuple[int, ...],
    samples_per_pair: int,
    matched_schedule: dict[str, object],
    tensor_runtime: object,
    clock: object,
    row_timeout_s: float | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    del matched_schedule, tensor_runtime, clock, row_timeout_s
    rows: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    for instance in instances:
        for seed in seeds:
            for device in ("cpu", "cuda"):
                for method_id in ("discrete_dls_heat_bath", "corrected_cdls_projection_mh"):
                    row_id = mod.row_id(instance.instance_id, seed, device, method_id)
                    base = -0.1 * float(instance.size) - float(seed % 13)
                    energies = _trace(base, samples_per_pair, seed)
                    constraints = _constraint_trace(samples_per_pair, seed)
                    wall = 5.0 if device == "cpu" else 2.5
                    if method_id == "corrected_cdls_projection_mh":
                        wall *= 0.8
                    metrics = mod.metrics_from_traces(energies, constraints, acceptance_rate=0.84)
                    rows.append(
                        {
                            "status": "success",
                            "row_id": row_id,
                            "pair_id": f"{instance.instance_id}:seed{seed}",
                            "method_id": method_id,
                            "device": device,
                            "backend": device,
                            "instance_id": instance.instance_id,
                            "size": instance.size,
                            "seed": seed,
                            "samples": samples_per_pair,
                            "temperature": mod.DEFAULT_TEMPERATURE,
                            "warmup_steps": mod.DEFAULT_WARMUP_STEPS,
                            "thinning": mod.DEFAULT_THINNING,
                            "precision": mod.DEFAULT_PRECISION,
                            "acceptance_rate": 1.0 if method_id == "discrete_dls_heat_bath" else 0.84,
                            "best_energy": metrics["best_energy"],
                            "energy_mean": metrics["mean_energy"],
                            "energy_std": metrics["energy_std"],
                            "energy_min": metrics["energy_min"],
                            "energy_max": metrics["energy_max"],
                            "energy_quantiles": metrics["energy_quantiles"],
                            "exact_constraint_satisfaction_rate": metrics[
                                "exact_constraint_satisfaction_rate"
                            ],
                            "autocorrelation_time": metrics["integrated_autocorrelation_time"],
                            "effective_sample_size": metrics["effective_sample_size"],
                            "compile_time_s": 0.05,
                            "warmup_time_s": wall * 0.1,
                            "sample_time_s": wall * 0.9,
                            "wall_time_s": wall,
                            "end_to_end_wall_time_s": wall + 0.05,
                            "memory_before": {"status": "reachable", "free_mib": 20480},
                            "memory_after": {"status": "reachable", "free_mib": 20470},
                            "kernel_device_path": f"tensor_{device}_{method_id}",
                            "result_hash": mod.sha256_json(
                                {"row_id": row_id, "energies": energies[:16], "samples": samples_per_pair}
                            ),
                        }
                    )
                    stats.append(
                        {
                            "row_id": row_id,
                            "pair_id": f"{instance.instance_id}:seed{seed}",
                            "method_id": method_id,
                            "device": device,
                            "instance_id": instance.instance_id,
                            "size": instance.size,
                            "seed": seed,
                            "samples": samples_per_pair,
                            "energy_trace": energies,
                            "constraint_trace": constraints,
                            "timing": {
                                "compile_time_s": 0.05,
                                "warmup_time_s": wall * 0.1,
                                "sample_time_s": wall * 0.9,
                                "end_to_end_wall_time_s": wall + 0.05,
                            },
                        }
                    )
    return rows, stats


def test_req_sample_5623_spec_declares_multiseed_crossover_contract() -> None:
    """REQ-SAMPLE-5623: OpenSpec anchors upstream, quality, timing, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5623") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-SAMPLE-5623",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.UPSTREAM_GATE_RELATIVE_PATH),
        "corrected_cdls_projection_mh",
        "SHALL NOT substitute",
        "n=128,256,512,1024",
        "At least five paired seeds",
        "at least 10000 post-warmup samples",
        "sufficient-statistic",
        mod.INFERENCE_SUBSTRATE,
        "`board_speedup_claimed` SHALL be false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5623_uses_exp5622_corrected_kernel_parameters() -> None:
    """REQ-SAMPLE-5623: corrected cDLS parameters are inherited from Exp5622."""

    params = mod.corrected_cdls_kernel_parameters()

    assert params["final_kernel"] == "corrected_cdls_projection_mh"
    assert params["proposal_std"] == exp5622.CDLS_PROPOSAL_STD
    assert params["drift_scale"] == exp5622.CDLS_DRIFT_SCALE
    assert params["continuous_bound"] == exp5622.CDLS_CONTINUOUS_BOUND
    assert params["biased_control_kernel_used"] is False
    assert {row["model_id"] for row in mod.models_tested()} == {
        "discrete_dls_heat_bath",
        "corrected_cdls_projection_mh",
    }


def test_scenario_sample_5623_builds_artifact_with_gated_speedups(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5623: fake matched rows emit valid quality-gated evidence."""

    _write_ready_upstream(tmp_path)
    seeds = (5623, 5624, 5625, 5626, 5627)
    artifact = mod.build_artifact(
        root=tmp_path,
        tensor_runtime=FakeTensorRuntime(cuda_available=True),
        sampler_runner=_fake_sampler_rows,
        clock=StepClock(),
        instance_sizes=(128,),
        seeds=seeds,
        samples_per_pair=10_000,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["upstream_gate_receipt"]["ready"] is True
    assert saved["upstream_gate_receipt"]["final_kernel"] == "corrected_cdls_projection_mh"
    assert saved["instance_sizes"] == [128]
    assert saved["seeds"] == list(seeds)
    assert saved["random_seeds"] == list(seeds)
    assert saved["samples_per_pair"] == 10_000
    assert saved["cpu_device_receipt"]["status"] == "reachable"
    assert saved["cuda_device_receipt"]["status"] == "reachable"
    assert len(saved["timing_rows"]) == 20
    assert len(saved["quality_gate_results_by_pair"]) == 5
    assert len(saved["successful_matched_pairs"]) == 5
    assert len(saved["speedup_by_pair"]) == 5
    assert all(row["included_in_speedups"] is True for row in saved["quality_gate_results_by_pair"])
    assert saved["crossover_size"] == 128
    assert saved["crossover_claim_allowed"] is True
    assert saved["board_speedup_claimed"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["sufficient_statistics_sha256"] == mod.file_sha256(
        tmp_path / saved["sufficient_statistics_path"]
    )
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)

    recomputed = mod.recompute_metrics_from_sufficient_statistics(
        tmp_path / saved["sufficient_statistics_path"]
    )
    assert recomputed["row_count"] == 20
    assert recomputed["successful_row_count"] == 20
    assert recomputed["pair_count"] == 5
    assert recomputed["samples_min"] == 10_000
    mod.validate_artifact(saved)


def test_scenario_sample_5623_blocks_without_upstream_gate_before_sampler(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5623: missing Exp5622 gate blocks before GPU work."""

    def forbidden_sampler(*args: Any, **kwargs: Any) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        raise AssertionError("sampler must not run without upstream readiness")

    artifact = mod.build_artifact(
        root=tmp_path,
        tensor_runtime=FakeTensorRuntime(cuda_available=True),
        sampler_runner=forbidden_sampler,
        clock=StepClock(),
        instance_sizes=(128,),
        seeds=mod.DEFAULT_SEEDS,
        samples_per_pair=10_000,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )

    assert artifact["upstream_gate_receipt"]["ready"] is False
    assert artifact["upstream_gate_receipt"]["blocked_reason"] == "upstream_gate_missing"
    assert len(artifact["timing_rows"]) == 4 * len(mod.DEFAULT_SEEDS)
    assert {row["status"] for row in artifact["timing_rows"]} == {"blocked"}
    assert artifact["successful_matched_pairs"] == []
    assert artifact["speedup_by_pair"] == []
    assert artifact["crossover_size"] is None
    assert artifact["crossover_claim_allowed"] is False
    assert artifact["board_speedup_claimed"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_sample_5623_validation_rejects_overclaims_and_bad_methodology(tmp_path: Path) -> None:
    """REQ-SAMPLE-5623: validation fails closed on unsafe crossover edits."""

    _write_ready_upstream(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        tensor_runtime=FakeTensorRuntime(cuda_available=True),
        sampler_runner=_fake_sampler_rows,
        clock=StepClock(),
        instance_sizes=(128,),
        seeds=mod.DEFAULT_SEEDS,
        samples_per_pair=10_000,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )

    mutations = [
        ("board_speedup_claimed", lambda data: data.__setitem__("board_speedup_claimed", True)),
        ("samples_per_pair", lambda data: data.__setitem__("samples_per_pair", 9999)),
        ("seeds", lambda data: data.__setitem__("seeds", data["seeds"][:4])),
        (
            "biased_control_kernel_used",
            lambda data: data["models_tested"][1].__setitem__("biased_control_kernel_used", True),
        ),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "llm")),
        ("crossover_claim_allowed", lambda data: data.__setitem__("crossover_claim_allowed", True)),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        if expected == "crossover_claim_allowed":
            bad["timing_intervals_by_size"] = [
                {
                    "size": 128,
                    "n_seed_pairs": 5,
                    "corrected_cdls_cuda_vs_cpu_speedup_interval_95": [0.9, 1.1],
                    "excludes_1_favorable": False,
                }
            ]
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing_stats = deepcopy(artifact)
    missing_stats["sufficient_statistics_sha256"] = "0" * 64
    missing_stats["reproducibility_checksum"] = mod.payload_checksum(missing_stats)
    with pytest.raises(ValueError, match="sufficient_statistics"):
        mod.validate_artifact(missing_stats)
