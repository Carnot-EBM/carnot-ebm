"""Tests for Exp5724 one-axis Rust/Python matched crossover.

Spec refs: REQ-SAMPLE-5724, SCENARIO-SAMPLE-5724.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5724_one_axis_rust_python_matched_crossover as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5724_one_axis_rust_python_matched_crossover.py")


def _fake_benchmark_runner(
    *,
    protocol: dict[str, Any],
    workloads: list[dict[str, Any]],
    **_: Any,
) -> dict[str, Any]:
    quality: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    ratios: list[dict[str, Any]] = []
    for workload in workloads:
        counters = {
            "replicas": len(protocol["beta_ladder"]),
            "corrected_transitions": 9,
            "swap_attempts": 6,
            "energy_evaluations": 30,
            "checkpoints": 2,
            "restarts": 2,
            "stopping_rule": protocol["stopping_rule"],
        }
        work.append(
            {
                "workload_id": workload["workload_id"],
                "size": workload["size"],
                "family": workload["family"],
                "matched": True,
                "rust": dict(counters),
                "python": dict(counters),
                "initial_state_hashes_match": True,
                "checkpoint_schema_match": True,
            }
        )
        for seed in protocol["random_seeds"]:
            quality.append(
                {
                    "pair_id": f"{workload['workload_id']}:seed{seed}",
                    "workload_id": workload["workload_id"],
                    "size": workload["size"],
                    "family": workload["family"],
                    "seed": seed,
                    "quality_matched": True,
                    "excluded_reason": None,
                    "feasibility_delta": 0.0,
                    "best_energy_delta_abs": 0.0,
                    "mean_energy_delta_abs": 0.0,
                    "acceptance_rate_delta_abs": 0.0,
                    "swap_acceptance_rate_delta_abs": 0.0,
                    "target_distribution_tv_delta": 0.0,
                    "restart_match": True,
                    "work_counters_match": True,
                    "rust_active_backend": "rust_pyo3",
                    "python_active_backend": "python_exact_fallback",
                }
            )
        ratios.append(
            {
                "workload_id": workload["workload_id"],
                "size": workload["size"],
                "family": workload["family"],
                "repetition_count": protocol["measured_repetition_count"],
                "ratio_samples": [1.35 for _ in range(protocol["measured_repetition_count"])],
                "ratio_mean": 1.35,
                "rust_faster_fraction": 1.0,
            }
        )

    intervals = [
        {
            "size": size,
            "family_count": len(protocol["topology_families"]),
            "repetition_count": len(protocol["topology_families"])
            * protocol["measured_repetition_count"],
            "rust_end_to_end_speedup_interval_95": [1.22, 1.47],
            "interval_entirely_above_one": True,
            "quality_matched": True,
        }
        for size in protocol["problem_sizes"]
    ]
    component_rows = [
        {
            "workload_id": workload["workload_id"],
            "size": workload["size"],
            "family": workload["family"],
            "arm": arm,
            "n": protocol["measured_repetition_count"],
            "mean_s": 0.002 if arm == "rust_pyo3" else 0.003,
            "median_s": 0.002 if arm == "rust_pyo3" else 0.003,
            "min_s": 0.001,
            "max_s": 0.004,
            "stdev_s": 0.0001,
            "mad_s": 0.00005,
            "coefficient_of_variation": 0.05,
        }
        for workload in workloads
        for arm in ("rust_pyo3", "python_exact_fallback")
    ]
    return {
        "matched_work_receipts": work,
        "quality_metrics_by_pair": quality,
        "excluded_pair_reasons": [],
        "kernel_times": list(component_rows),
        "pyo3_overhead_times": [
            {
                "arm": "rust_pyo3",
                "n": protocol["measured_repetition_count"],
                "mean_s": 0.0001,
                "median_s": 0.0001,
                "min_s": 0.00009,
                "max_s": 0.00011,
                "stdev_s": 0.00001,
                "mad_s": 0.000005,
                "coefficient_of_variation": 0.1,
                "not_subtracted_from_end_to_end": True,
            },
            {
                "arm": "python_exact_fallback",
                "n": protocol["measured_repetition_count"],
                "mean_s": 0.0,
                "median_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0,
                "stdev_s": 0.0,
                "mad_s": 0.0,
                "coefficient_of_variation": 0.0,
                "not_applicable": True,
                "not_subtracted_from_end_to_end": True,
            },
        ],
        "serialization_times": list(component_rows),
        "validation_times": list(component_rows),
        "end_to_end_times": list(component_rows),
        "peak_rss_by_arm": {
            "rust_pyo3": {"peak_kib": 1000},
            "python_exact_fallback": {"peak_kib": 990},
        },
        "paired_speedup_ratios": ratios,
        "paired_speedup_intervals": intervals,
    }


def _fake_null_benchmark_runner(
    *,
    protocol: dict[str, Any],
    workloads: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    evidence = _fake_benchmark_runner(protocol=protocol, workloads=workloads, **kwargs)
    midpoint = len(protocol["problem_sizes"]) // 2
    for index, row in enumerate(evidence["paired_speedup_intervals"]):
        if index >= midpoint:
            row["rust_end_to_end_speedup_interval_95"] = [0.71, 0.93]
            row["interval_entirely_above_one"] = False
    for row in evidence["paired_speedup_ratios"]:
        if protocol["problem_sizes"].index(row["size"]) >= midpoint:
            row["ratio_samples"] = [0.82 for _ in range(protocol["measured_repetition_count"])]
            row["ratio_mean"] = 0.82
            row["rust_faster_fraction"] = 0.0
    return evidence


def test_req_sample_5724_spec_declares_matched_crossover_contract() -> None:
    """REQ-SAMPLE-5724: OpenSpec anchors design, fields, and no-hardware scope."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5724") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "same production `SamplerBackend` API",
        "at least six size strata",
        "at least three topology/hardness families",
        "at least ten paired seeds",
        "at least thirty measured paired repetitions",
        "python_end_to_end_time / rust_end_to_end_time",
        "`timing_claimed` SHALL be true",
        "`hardware_speedup_claimed`, `gpu_speedup_claimed`, and",
        mod.INFERENCE_SUBSTRATE,
        "SCENARIO-SAMPLE-5724",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_sample_5724_builds_speedup_artifact_from_quality_matched_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-5724: matched fake evidence emits a valid crossover artifact."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_benchmark_runner,
        problem_sizes=mod.DEFAULT_PROBLEM_SIZES,
        topology_families=mod.DEFAULT_TOPOLOGY_FAMILIES,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        warmup_count=1,
        measured_repetition_count=30,
        freeze_affinity=False,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["upstream_gate_receipts"]["exp5723"]["ready"] is True
    assert saved["problem_sizes"] == list(mod.DEFAULT_PROBLEM_SIZES)
    assert saved["topology_families"] == list(mod.DEFAULT_TOPOLOGY_FAMILIES)
    assert len(saved["random_seeds"]) == 10
    assert saved["measured_repetition_count"] == 30
    assert saved["quality_matched_pair_count"] == 180
    assert saved["excluded_pair_reasons"] == []
    assert saved["qualified_crossover_n"] == min(mod.DEFAULT_PROBLEM_SIZES)
    assert saved["rust_crossover_ready_score"] == 1.0
    assert saved["software_speedup_claimed"] is True
    assert saved["timing_claimed"] is True
    assert saved["hardware_speedup_claimed"] is False
    assert saved["gpu_speedup_claimed"] is False
    assert saved["fpga_or_tsu_used"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["tests_added_or_reused"] == [TEST_PATH.as_posix()]
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5724_terminal_null_when_large_size_suffix_does_not_win() -> None:
    """REQ-SAMPLE-5724: isolated small-size wins do not become a crossover claim."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_null_benchmark_runner,
        problem_sizes=mod.DEFAULT_PROBLEM_SIZES,
        topology_families=mod.DEFAULT_TOPOLOGY_FAMILIES,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        measured_repetition_count=30,
        freeze_affinity=False,
    )

    assert artifact["qualified_crossover_n"] is None
    assert artifact["rust_crossover_ready_score"] == 0.0
    assert artifact["software_speedup_claimed"] is False
    assert artifact["honest_verdict"].startswith("complete: terminal null")
    mod.validate_artifact(artifact)


def test_req_sample_5724_actual_samplerbackend_runner_matches_work_and_quality() -> None:
    """REQ-SAMPLE-5724: real Rust/Python SamplerBackend arms match work and quality."""

    protocol = mod.preregistered_protocol(
        problem_sizes=(3,),
        topology_families=("ferromagnetic_ring_easy",),
        random_seeds=(5724,),
        warmup_count=1,
        measured_repetition_count=2,
        allow_underpowered=True,
    )
    workloads = mod.build_workload_manifest(
        problem_sizes=(3,),
        topology_families=("ferromagnetic_ring_easy",),
    )
    evidence = mod.run_matched_crossover_study(
        protocol=protocol,
        workloads=workloads,
        clock=mod.Clock(),
    )

    assert len(evidence["quality_metrics_by_pair"]) == 1
    assert evidence["quality_metrics_by_pair"][0]["quality_matched"] is True
    assert evidence["quality_metrics_by_pair"][0]["rust_active_backend"] == "rust_pyo3"
    assert evidence["quality_metrics_by_pair"][0]["python_active_backend"] == (
        "python_exact_fallback"
    )
    assert evidence["matched_work_receipts"][0]["matched"] is True
    assert evidence["end_to_end_times"][0]["n"] == 2
    assert evidence["paired_speedup_ratios"][0]["repetition_count"] == 2


def test_req_sample_5724_validation_rejects_overclaims_and_bad_design() -> None:
    """REQ-SAMPLE-5724: schema validation fails closed on unsafe edits."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_null_benchmark_runner,
        problem_sizes=mod.DEFAULT_PROBLEM_SIZES,
        topology_families=mod.DEFAULT_TOPOLOGY_FAMILIES,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        measured_repetition_count=30,
        freeze_affinity=False,
    )
    mutations = [
        ("field_principles", lambda data: data["field_principles"].__setitem__("bad", "bad")),
        (
            "problem_sizes",
            lambda data: data.__setitem__("problem_sizes", data["problem_sizes"][:5]),
        ),
        (
            "topology_families",
            lambda data: data.__setitem__("topology_families", data["topology_families"][:2]),
        ),
        ("random_seeds", lambda data: data.__setitem__("random_seeds", data["random_seeds"][:9])),
        (
            "measured_repetition_count",
            lambda data: data.__setitem__("measured_repetition_count", 29),
        ),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", False)),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        ("gpu_speedup_claimed", lambda data: data.__setitem__("gpu_speedup_claimed", True)),
        ("fpga_or_tsu_used", lambda data: data.__setitem__("fpga_or_tsu_used", True)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "bad")),
        (
            "software_speedup_claimed",
            lambda data: data.__setitem__("software_speedup_claimed", True),
        ),
        (
            "rust_crossover_ready_score",
            lambda data: data.__setitem__("rust_crossover_ready_score", 1.0),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "bad"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_sample_5724_helper_edges_fail_closed() -> None:
    """REQ-SAMPLE-5724: helpers reject underspecified protocols and bad checksums."""

    assert (
        mod.qualified_crossover_from_intervals(
            [
                {"size": 4, "quality_matched": True, "interval_entirely_above_one": True},
                {"size": 8, "quality_matched": True, "interval_entirely_above_one": False},
            ],
            problem_sizes=(4, 8),
        )
        is None
    )
    assert (
        mod.qualified_crossover_from_intervals(
            [
                {"size": 4, "quality_matched": True, "interval_entirely_above_one": True},
                {"size": 8, "quality_matched": True, "interval_entirely_above_one": True},
            ],
            problem_sizes=(4, 8),
        )
        == 4
    )
    with pytest.raises(ValueError, match="problem_sizes"):
        mod.preregistered_protocol(problem_sizes=(3, 6, 12, 24, 48), random_seeds=range(10))
    with pytest.raises(ValueError, match="random_seeds"):
        mod.preregistered_protocol(problem_sizes=mod.DEFAULT_PROBLEM_SIZES, random_seeds=range(9))
    with pytest.raises(ValueError, match="topology_families"):
        mod.preregistered_protocol(
            problem_sizes=mod.DEFAULT_PROBLEM_SIZES,
            topology_families=("a", "b"),
            random_seeds=range(10),
        )
    with pytest.raises(ValueError, match="measured_repetition_count"):
        mod.preregistered_protocol(
            problem_sizes=mod.DEFAULT_PROBLEM_SIZES,
            random_seeds=range(10),
            measured_repetition_count=29,
        )


def test_scenario_sample_5724_main_delegates_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-5724: CLI entrypoint delegates build and write steps."""

    calls: list[tuple[str, object]] = []

    def fake_build(**kwargs: Any) -> dict[str, bool]:
        calls.append(("build", kwargs))
        return {"ok": True}

    def fake_write(root: Path, artifact: dict[str, bool]) -> Path:
        calls.append(("write", (root, artifact)))
        return Path("results/fake.json")

    monkeypatch.setattr(mod, "build_artifact", fake_build)
    monkeypatch.setattr(mod, "write_output", fake_write)

    mod.main()

    assert calls == [
        ("build", {"root": mod.REPO_ROOT}),
        ("write", (mod.REPO_ROOT, {"ok": True})),
    ]
