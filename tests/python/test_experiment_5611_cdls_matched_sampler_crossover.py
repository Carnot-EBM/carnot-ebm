"""Tests for Exp 5611 bounded cDLS matched sampler crossover.

Spec refs: REQ-SAMPLE-5611, SCENARIO-SAMPLE-5611.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_5611_cdls_matched_sampler_crossover as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5611_cdls_matched_sampler_crossover.py")


class StepClock:
    """Deterministic clock that makes sampler timing stable in tests."""

    def __init__(self) -> None:
        self.value = 5611.0

    def __call__(self) -> float:
        self.value += 0.125
        return self.value


class FakeCuda:
    """Small CUDA facade used for receipt and blocked-path tests."""

    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.sync_calls = 0

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Fake RTX 3090"

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (16 * 1024**3, 24 * 1024**3)

    def memory_reserved(self, index: int) -> int:
        assert index == 0
        return 256 * 1024**2

    def synchronize(self) -> None:
        self.sync_calls += 1


class FakeTorch:
    """Torch-shaped object exposing CUDA metadata without running CUDA kernels."""

    __version__ = "2.11.0+cu128"

    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = FakeCuda(available=cuda_available)
        self.version = type("Version", (), {"cuda": "12.8"})()


class RaisingCuda(FakeCuda):
    """CUDA facade whose memory probe fails."""

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        raise RuntimeError("memory probe failed")


class RaisingTorch(FakeTorch):
    """Torch facade with reachable CUDA but failing memory APIs."""

    def __init__(self) -> None:
        self.cuda = RaisingCuda(available=True)
        self.version = type("Version", (), {"cuda": "12.8"})()


def _descriptor_payload() -> dict[str, object]:
    descriptors: list[dict[str, object]] = []
    for row_index, row_id in enumerate(("row_a", "row_b", "row_c", "row_d")):
        variables = [f"fact:{row_id}:v{i}" for i in range(12)]
        target = {
            variables[row_index]: "present",
            variables[(row_index + 5) % len(variables)]: "absent",
        }
        descriptors.append(
            {
                "descriptor_id": f"desc:{row_id}",
                "row_id": row_id,
                "all_repair_variables": variables,
                "repair_block_variables": list(target),
                "target_repair_assignment": target,
                "active_constraints": [
                    {"kind": "unit", "repair_variable": variable, "target": value}
                    for variable, value in target.items()
                ],
            }
        )
    return {"schema": "unit.descriptor.bundle", "asp_repair_descriptors": descriptors}


def _write_descriptor(root: Path, payload: dict[str, object] | None = None) -> None:
    path = root / mod.DESCRIPTOR_SOURCE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _descriptor_payload(), sort_keys=True), encoding="utf-8")


def _small_instance() -> mod.IsingInstance:
    couplings = np.array(
        [
            [0.0, 0.12, -0.04, 0.03],
            [0.12, 0.0, 0.08, -0.02],
            [-0.04, 0.08, 0.0, 0.10],
            [0.03, -0.02, 0.10, 0.0],
        ],
        dtype=np.float32,
    )
    biases = np.array([0.15, -0.10, 0.05, 0.20], dtype=np.float32)
    target = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float32)
    return mod.IsingInstance(
        instance_id="unit_n4_cdls",
        size=4,
        descriptor_ids=("unit",),
        couplings=couplings,
        biases=biases,
        target_spins=target,
        constraint_indices=(0, 1, 2, 3),
        checksum="unit-checksum",
    )


def _fake_sampler_rows(
    instances: list[mod.IsingInstance],
    seeds: tuple[int, ...],
    samples_per_pair: int,
    matched_schedule: dict[str, object],
    torch_module: object,
    clock: object,
) -> list[dict[str, object]]:
    del torch_module, clock
    rows: list[dict[str, object]] = []
    for instance in instances:
        for seed in seeds:
            for device in ("cpu", "cuda"):
                for method in ("discrete_dls", "bounded_cdls"):
                    method_offset = 0.0 if method == "discrete_dls" else -0.02
                    base = -0.25 * float(instance.size) - float(seed % 5)
                    wall_base = 6.0 + float(instance.size) / (64.0 if device == "cpu" else 140.0)
                    wall_scale = 1.0 if method == "discrete_dls" else 0.55
                    wall_time = wall_base * wall_scale
                    rows.append(
                        {
                            "status": "success",
                            "pair_id": f"{instance.instance_id}:seed{seed}",
                            "method": method,
                            "backend": device,
                            "device": device,
                            "instance_id": instance.instance_id,
                            "size": instance.size,
                            "seed": seed,
                            "samples": samples_per_pair,
                            "temperature": matched_schedule["temperature"],
                            "warmup_steps": matched_schedule["warmup_steps"],
                            "thinning": matched_schedule["thinning"],
                            "precision": matched_schedule["precision"],
                            "best_energy": base - 1.0 + method_offset,
                            "energy_mean": base + method_offset,
                            "energy_std": 0.75,
                            "energy_min": base - 1.0 + method_offset,
                            "energy_max": base + 2.0 + method_offset,
                            "energy_quantiles": {
                                "p05": base - 0.5 + method_offset,
                                "p50": base + method_offset,
                                "p95": base + 0.5 + method_offset,
                            },
                            "constraint_satisfaction_rate": 0.93,
                            "exact_constraint_satisfaction_rate": 0.93,
                            "acceptance_rate": 1.0 if method == "discrete_dls" else 0.82,
                            "autocorrelation_time": 2.0,
                            "effective_sample_size": samples_per_pair / 2.0,
                            "compile_time_s": 0.0,
                            "warmup_time_s": wall_time * 0.10,
                            "sample_time_s": wall_time * 0.90,
                            "wall_time_s": wall_time,
                            "end_to_end_wall_time_s": wall_time,
                            "memory_before": {"free_mib": 1000},
                            "memory_after": {"free_mib": 990},
                            "kernel_device_path": f"torch_{device}_{method}",
                            "projection_correction": (
                                "not_applicable_discrete_baseline"
                                if method == "discrete_dls"
                                else "metropolis_hastings_exact_discrete_target"
                            ),
                            "result_hash": f"{method}-{device}-{instance.instance_id}-{seed}",
                        }
                    )
    return rows


def test_req_sample_5611_spec_declares_cdls_artifact_contract() -> None:
    """REQ-SAMPLE-5611: OpenSpec anchors cDLS, quality, and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5611") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-SAMPLE-5611",
        str(mod.RESULT_RELATIVE_PATH),
        "Metropolis-Hastings correction",
        "exact discrete Ising energy",
        mod.INFERENCE_SUBSTRATE,
        "n=128,256,512,1024",
        "at least 10000 post-warmup samples",
        "`crossover_claim_allowed=true`",
        "`board_speedup_claimed` SHALL be false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


@pytest.mark.memory_watchdog_skip
def test_req_sample_5611_cdls_projection_and_mh_correction_are_deterministic() -> None:
    """REQ-SAMPLE-5611: cDLS projects through a bounded exact-target correction."""

    import torch

    instance = _small_instance()
    schedule = mod.matched_schedule_for_instances([instance])
    kwargs = {
        "instance": instance,
        "backend": "cpu",
        "seed": 5611,
        "samples_per_pair": 32,
        "matched_schedule": schedule,
        "torch_module": torch,
    }

    first = mod.run_cdls_sampler_row(clock=StepClock(), **kwargs)
    second = mod.run_cdls_sampler_row(clock=StepClock(), **kwargs)

    comparable_keys = set(first) - {"memory_before", "memory_after"}
    assert {key: first[key] for key in comparable_keys} == {key: second[key] for key in comparable_keys}
    assert first["status"] == "success"
    assert first["method"] == "bounded_cdls"
    assert first["backend"] == "cpu"
    assert first["samples"] == 32
    assert first["projection_correction"] == "metropolis_hastings_exact_discrete_target"
    assert first["continuous_bound_observed_abs_max"] <= first["continuous_bound"]
    assert 0.0 <= first["acceptance_rate"] <= 1.0
    assert first["mh_accept_count"] + first["mh_reject_count"] == first["proposal_count"]
    assert np.isfinite(first["energy_mean"])
    assert first["result_hash"] == second["result_hash"]


def test_scenario_sample_5611_builds_matched_artifact_and_crossover(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5611: matched fake rows summarize only gated pairs."""

    _write_descriptor(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        torch_module=FakeTorch(cuda_available=True),
        sampler_runner=_fake_sampler_rows,
        clock=StepClock(),
        instance_sizes=(128, 256),
        seeds=(5611, 5612, 5613),
        samples_per_pair=10_000,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["target_descriptors"]
    assert saved["instance_sizes"] == [128, 256]
    assert saved["methods"][0]["method_id"] == "discrete_dls"
    assert saved["methods"][1]["method_id"] == "bounded_cdls"
    assert saved["seeds"] == [5611, 5612, 5613]
    assert saved["samples_per_pair"] == 10_000
    assert saved["cpu_device_receipt"]["status"] == "reachable"
    assert saved["cuda_device_receipt"]["status"] == "reachable"
    assert len(saved["timing_rows"]) == 24
    assert saved["successful_matched_pairs"] == 6
    assert len(saved["speedup_by_pair"]) == 6
    assert all(row["quality_noninferior"] is True for row in saved["speedup_by_pair"])
    assert saved["crossover_size"] == 128
    assert saved["crossover_claim_allowed"] is True
    assert saved["board_speedup_claimed"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_scenario_sample_5611_blocks_cuda_absence_without_cpu_fallback(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5611: absent CUDA blocks comparison instead of fabricating rows."""

    _write_descriptor(tmp_path)

    def forbidden_sampler(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        raise AssertionError("sampler must not run when CUDA is unavailable")

    artifact = mod.build_artifact(
        root=tmp_path,
        torch_module=FakeTorch(cuda_available=False),
        sampler_runner=forbidden_sampler,
        clock=StepClock(),
        instance_sizes=(128,),
        seeds=(5611,),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )

    assert artifact["preconditions"]["cuda_available"] is False
    assert artifact["cuda_device_receipt"]["status"] == "blocked"
    assert artifact["timing_rows"] == []
    assert artifact["successful_matched_pairs"] == 0
    assert artifact["speedup_by_pair"] == []
    assert artifact["crossover_claim_allowed"] is False
    assert artifact["crossover_size"] is None
    assert artifact["board_speedup_claimed"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_sample_5611_validation_rejects_overclaims_and_bad_rows(tmp_path: Path) -> None:
    """REQ-SAMPLE-5611: validation fails closed on unsupported crossover claims."""

    _write_descriptor(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        torch_module=FakeTorch(cuda_available=True),
        sampler_runner=_fake_sampler_rows,
        clock=StepClock(),
        instance_sizes=(128,),
        seeds=(5611, 5612, 5613),
        samples_per_pair=10_000,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )

    bad_board = deepcopy(artifact)
    bad_board["board_speedup_claimed"] = True
    bad_board["reproducibility_checksum"] = mod.payload_checksum(bad_board)
    try:
        mod.validate_artifact(bad_board)
    except ValueError as exc:
        assert "board_speedup_claimed" in str(exc)
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("board_speedup_claimed overclaim accepted")

    bad_samples = deepcopy(artifact)
    bad_samples["samples_per_pair"] = 9999
    bad_samples["reproducibility_checksum"] = mod.payload_checksum(bad_samples)
    try:
        mod.validate_artifact(bad_samples)
    except ValueError as exc:
        assert "samples_per_pair" in str(exc)
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("undersized samples accepted")

    bad_crossover = deepcopy(artifact)
    bad_crossover["speedup_by_pair"][0]["quality_noninferior"] = False
    bad_crossover["crossover_claim_allowed"] = True
    bad_crossover["reproducibility_checksum"] = mod.payload_checksum(bad_crossover)
    try:
        mod.validate_artifact(bad_crossover)
    except ValueError as exc:
        assert "crossover_claim_allowed" in str(exc)
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("unsupported crossover accepted")

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    try:
        mod.validate_artifact(bad_checksum)
    except ValueError as exc:
        assert "reproducibility_checksum" in str(exc)
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("bad checksum accepted")


def test_req_sample_5611_helper_blockers_and_no_crossover_paths(tmp_path: Path) -> None:
    """REQ-SAMPLE-5611: blocker helpers keep failed evidence out of speedups."""

    instance = _small_instance()
    assert mod.memory_snapshot(FakeTorch(cuda_available=False), "cuda")["status"] == "blocked"
    assert mod.memory_snapshot(FakeTorch(cuda_available=True), "cuda")["free_mib"] == 16384
    assert mod.memory_snapshot(RaisingTorch(), "cuda")["blocked_reason"] == "RuntimeError"
    mod._sync_if_cuda(FakeTorch(cuda_available=True), "cuda")
    assert mod._is_oom_error(RuntimeError("CUDA error: out of memory"))
    assert mod._ratio({"wall_time_s": 1.0}, {"wall_time_s": 0.0}) is None
    assert mod.crossover_from_speedups(
        [{"quality_noninferior": True, "size": 128, "bounded_cdls_cuda_vs_cpu_speedup": None}]
    ) == (None, False, [])
    assert mod.crossover_from_speedups(
        [{"quality_noninferior": False, "size": 128, "bounded_cdls_cuda_vs_cpu_speedup": 2.0}]
    ) == (None, False, [])
    assert mod._normalize_tests(None)[0].endswith("test_experiment_5611_cdls_matched_sampler_crossover.py")

    blocked = mod.blocked_timing_row(instance, 5611, "cuda", "bounded_cdls", "timeout")
    energy, mixing, timing, speedups = mod.summarize_rows([blocked])
    assert energy == []
    assert mixing == []
    assert timing[0]["blocked_reason"] == "timeout"
    assert speedups == []

    schedule = mod.matched_schedule_for_instances([instance])
    rows = _fake_sampler_rows([instance], (5611,), 10_000, schedule, object(), object())
    blocked_rows = deepcopy(rows)
    blocked_rows[0]["status"] = "blocked"
    assert mod.summarize_rows(blocked_rows)[3] == []
    mismatched_rows = deepcopy(rows)
    mismatched_rows[0]["samples"] = 9999
    assert mod.summarize_rows(mismatched_rows)[3] == []

    missing_descriptor = mod.build_artifact(
        root=tmp_path,
        torch_module=FakeTorch(cuda_available=False),
        sampler_runner=_fake_sampler_rows,
        instance_sizes=(128,),
        seeds=(5611,),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    assert missing_descriptor["descriptor_source"]["available"] is False
    assert missing_descriptor["honest_verdict"].startswith("blocked:")

    original_import = mod.exp5573._import_torch
    mod.exp5573._import_torch = lambda: (_ for _ in ()).throw(RuntimeError("torch unavailable"))
    try:
        import_blocked = mod.build_artifact(
            root=tmp_path,
            torch_module=None,
            sampler_runner=_fake_sampler_rows,
            instance_sizes=(128,),
            seeds=(5611,),
            tests_added_or_reused=[TEST_PATH.as_posix()],
        )
    finally:
        mod.exp5573._import_torch = original_import
    assert import_blocked["cuda_device_receipt"]["status"] == "blocked"

    _write_descriptor(tmp_path)
    descriptor_build_blocked = mod.build_artifact(
        root=tmp_path,
        torch_module=FakeTorch(cuda_available=False),
        sampler_runner=_fake_sampler_rows,
        instance_sizes=(8,),
        seeds=(5611,),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    assert "descriptor_instance_build_failed:ValueError" in descriptor_build_blocked["preconditions"][
        "blocked_reasons"
    ]

    def raising_sampler(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        raise RuntimeError("synthetic sampler failure")

    sampler_blocked = mod.build_artifact(
        root=tmp_path,
        torch_module=FakeTorch(cuda_available=True),
        sampler_runner=raising_sampler,
        instance_sizes=(128,),
        seeds=(5611,),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    assert sampler_blocked["timing_rows"] == []
    assert any("sampler_run_failed:RuntimeError" in item for item in sampler_blocked["preconditions"]["blocked_reasons"])

    no_crossover = mod.build_artifact(
        root=tmp_path,
        torch_module=FakeTorch(cuda_available=True),
        sampler_runner=_fake_sampler_rows,
        instance_sizes=(128,),
        seeds=(5611,),
        samples_per_pair=10_000,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    assert no_crossover["successful_matched_pairs"] == 1
    assert no_crossover["crossover_claim_allowed"] is False
    assert "no gated crossover" in no_crossover["honest_verdict"]

    bad_missing_size = deepcopy(no_crossover)
    bad_missing_size["crossover_claim_allowed"] = True
    bad_missing_size["crossover_size"] = None
    bad_missing_size["reproducibility_checksum"] = mod.payload_checksum(bad_missing_size)
    with pytest.raises(ValueError, match="crossover_size"):
        mod.validate_artifact(bad_missing_size)

    bad_interval = deepcopy(no_crossover)
    bad_interval["crossover_claim_allowed"] = True
    bad_interval["crossover_size"] = 128
    bad_interval["timing_intervals_by_size"] = [{"size": 128, "excludes_1_favorable": False}]
    bad_interval["reproducibility_checksum"] = mod.payload_checksum(bad_interval)
    with pytest.raises(ValueError, match="favorable timing interval"):
        mod.validate_artifact(bad_interval)
