"""Tests for Exp5739 one-axis batched Rust/Python 10x crossover.

Spec refs: REQ-SAMPLE-5739, SCENARIO-SAMPLE-5739.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5739_one_axis_batched_10x_crossover as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5739_one_axis_batched_10x_crossover.py")


def _fake_benchmark_evidence(
    *,
    protocol: dict[str, Any],
    workloads: list[dict[str, Any]],
    thread_regimes: list[dict[str, Any]],
    pass_10x: bool,
    **_: Any,
) -> dict[str, Any]:
    quality: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    throughput_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    peak_rss: dict[str, dict[str, dict[str, int]]] = {}
    for regime in thread_regimes:
        regime_id = regime["regime_id"]
        peak_rss[regime_id] = {
            mod.RUST_ARM: {"peak_kib": 1500, "source": "fake"},
            mod.PYTHON_ARM: {"peak_kib": 1400, "source": "fake"},
        }
        for workload in workloads:
            for batch_size in protocol["batch_sizes"]:
                cell_id = mod.cell_id(regime_id, workload, batch_size)
                counters = {
                    "batch_size": batch_size,
                    "samples_per_item": protocol["sample_sweeps"],
                    "cold_target_samples": batch_size * protocol["sample_sweeps"],
                    "corrected_transitions": batch_size
                    * (protocol["burn_in_sweeps"] + protocol["sample_sweeps"])
                    * len(protocol["beta_ladder"]),
                    "swap_attempts": batch_size
                    * (protocol["burn_in_sweeps"] + protocol["sample_sweeps"])
                    * (len(protocol["beta_ladder"]) - 1),
                    "checkpoint_restarts": batch_size,
                    "stopping_rule": protocol["stopping_rule"],
                }
                work.append(
                    {
                        "cell_id": cell_id,
                        "thread_regime": regime_id,
                        "size": workload["size"],
                        "family": workload["family"],
                        "batch_size": batch_size,
                        "measured_batch_count": protocol["measured_batch_count"],
                        "matched": True,
                        "rust": dict(counters),
                        "python": dict(counters),
                    }
                )
                ratios = []
                for batch_index in range(protocol["measured_batch_count"]):
                    pair_id = f"{cell_id}:batch{batch_index}"
                    quality.append(
                        {
                            "pair_id": pair_id,
                            "cell_id": cell_id,
                            "thread_regime": regime_id,
                            "size": workload["size"],
                            "family": workload["family"],
                            "batch_size": batch_size,
                            "batch_index": batch_index,
                            "quality_matched": True,
                            "excluded_reason": None,
                            "sample_count_match": True,
                            "work_counters_match": True,
                            "restart_match": True,
                            "result_order_match": True,
                            "energy_histogram_tv": 0.0,
                            "best_energy_delta_abs": 0.0,
                            "mean_energy_delta_abs": 0.0,
                        }
                    )
                    ratio = (
                        12.5
                        if pass_10x
                        and regime_id == "fixed_recorded_cores"
                        and workload["size"] in {96, 192}
                        else 1.2
                    )
                    ratios.append(ratio)
                    rust_s = 0.001
                    python_s = rust_s * ratio
                    for arm, elapsed in (
                        (mod.RUST_ARM, rust_s),
                        (mod.PYTHON_ARM, python_s),
                    ):
                        timing_rows.append(
                            {
                                "cell_id": cell_id,
                                "thread_regime": regime_id,
                                "size": workload["size"],
                                "family": workload["family"],
                                "batch_size": batch_size,
                                "arm": arm,
                                "n": protocol["measured_batch_count"],
                                "samples_s": [elapsed for _ in range(protocol["measured_batch_count"])],
                                "mean_s": elapsed,
                                "median_s": elapsed,
                                "min_s": elapsed,
                                "max_s": elapsed,
                                "stdev_s": 0.0,
                                "mad_s": 0.0,
                                "coefficient_of_variation": 0.0,
                            }
                        )
                        throughput = batch_size * protocol["sample_sweeps"] / elapsed
                        throughput_rows.append(
                            {
                                "cell_id": cell_id,
                                "thread_regime": regime_id,
                                "size": workload["size"],
                                "family": workload["family"],
                                "batch_size": batch_size,
                                "arm": arm,
                                "samples_per_batch": batch_size * protocol["sample_sweeps"],
                                "throughput_samples_per_s": [
                                    throughput for _ in range(protocol["measured_batch_count"])
                                ],
                                "mean_samples_per_s": throughput,
                            }
                        )
                    for phase in ("setup", "sample_batch", "serialization", "validation", "restart"):
                        phase_rows.append(
                            {
                                "cell_id": cell_id,
                                "thread_regime": regime_id,
                                "size": workload["size"],
                                "family": workload["family"],
                                "batch_size": batch_size,
                                "arm": mod.RUST_ARM,
                                "phase": phase,
                                "samples_s": [0.0001 for _ in range(protocol["measured_batch_count"])],
                                "mean_s": 0.0001,
                                "included_in_end_to_end": True,
                            }
                        )
                ratio_rows.append(
                    {
                        "cell_id": cell_id,
                        "thread_regime": regime_id,
                        "size": workload["size"],
                        "family": workload["family"],
                        "batch_size": batch_size,
                        "repetition_count": protocol["measured_batch_count"],
                        "ratio_samples": ratios,
                        "ratio_mean": sum(ratios) / len(ratios),
                    }
                )
    intervals = mod.paired_speedup_intervals_from_ratios(
        ratio_rows,
        quality_rows=quality,
        thread_regimes=[row["regime_id"] for row in thread_regimes],
        problem_sizes=protocol["problem_sizes"],
        measured_batch_count=protocol["measured_batch_count"],
    )
    return {
        "matched_work_receipts": work,
        "quality_metrics_by_pair": quality,
        "excluded_pair_reasons": [],
        "end_to_end_times": timing_rows,
        "throughput_distributions": throughput_rows,
        "phase_times": phase_rows,
        "peak_rss_by_arm": peak_rss,
        "paired_speedup_ratios": ratio_rows,
        "paired_speedup_intervals": intervals,
    }


def _fake_10x_runner(**kwargs: Any) -> dict[str, Any]:
    return _fake_benchmark_evidence(pass_10x=True, **kwargs)


def _fake_null_runner(**kwargs: Any) -> dict[str, Any]:
    return _fake_benchmark_evidence(pass_10x=False, **kwargs)


def test_req_sample_5739_spec_declares_strict_batched_10x_contract() -> None:
    """REQ-SAMPLE-5739: OpenSpec lists the 10x rule, fields, and no-hardware scope."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5739") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "at least thirty independent measured batches per arm",
        "one physical-core placement",
        "fixed recorded physical-core allocation",
        "adjusted lower confidence bound is at least `10.0`",
        "`gpu_speedup_claimed`, `hardware_speedup_claimed`, and",
        mod.INFERENCE_SUBSTRATE,
        "SCENARIO-SAMPLE-5739",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_sample_5739_builds_strict_10x_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5739: fake matched evidence can pass only the strict 10x rule."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_10x_runner,
        freeze_affinity=False,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(saved["field_principles"]) == set(saved)
    assert saved["upstream_gate_receipts"]["exp5738"]["ready"] is True
    assert saved["problem_sizes"] == [48, 96, 192]
    assert saved["measured_batch_count"] == 30
    assert saved["qualified_10x_sizes"] == [96, 192]
    assert saved["qualified_10x_thread_regime"] == "fixed_recorded_cores"
    assert saved["rust_batched_10x_ready_score"] == 1.0
    assert saved["software_speedup_claimed"] is True
    assert saved["timing_claimed"] is True
    assert saved["gpu_speedup_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["fpga_or_tsu_used"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5739_terminal_null_when_adjusted_10x_rule_fails() -> None:
    """REQ-SAMPLE-5739: matched quality without two larger 10x intervals is a terminal null."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_null_runner,
        freeze_affinity=False,
    )

    assert artifact["qualified_10x_sizes"] == []
    assert artifact["qualified_10x_thread_regime"] is None
    assert artifact["rust_batched_10x_ready_score"] == 0.0
    assert artifact["software_speedup_claimed"] is False
    assert artifact["honest_verdict"].startswith("complete: terminal null")
    mod.validate_artifact(artifact)


def test_req_sample_5739_actual_small_batched_runner_matches_work_and_quality() -> None:
    """REQ-SAMPLE-5739: real small Rust/Python batches match work before ratios count."""

    protocol = mod.preregistered_protocol(
        problem_sizes=(3,),
        topology_families=("ferromagnetic_ring_easy",),
        batch_sizes=(1,),
        random_seeds=(5739, 5740),
        warmup_count=1,
        measured_batch_count=2,
        allow_underpowered=True,
    )
    workloads = mod.build_workload_manifest(
        problem_sizes=(3,),
        topology_families=("ferromagnetic_ring_easy",),
    )
    evidence = mod.run_matched_batched_benchmark(
        protocol=protocol,
        workloads=workloads,
        thread_regimes=[{"regime_id": "smoke", "cpus": [], "affinity_enforced": False}],
    )

    assert len(evidence["quality_metrics_by_pair"]) == 2
    assert all(row["quality_matched"] is True for row in evidence["quality_metrics_by_pair"])
    assert evidence["matched_work_receipts"][0]["matched"] is True
    assert evidence["end_to_end_times"][0]["n"] == 2
    assert evidence["paired_speedup_ratios"][0]["repetition_count"] == 2


def test_req_sample_5739_validation_rejects_overclaims_and_bad_schema() -> None:
    """REQ-SAMPLE-5739: unsafe manual edits fail closed."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_null_runner,
        freeze_affinity=False,
    )
    mutations = [
        ("field_principles", lambda data: data["field_principles"].__setitem__("bad", "bad")),
        ("problem_sizes", lambda data: data.__setitem__("problem_sizes", [48, 96])),
        ("random_seeds", lambda data: data.__setitem__("random_seeds", data["random_seeds"][:29])),
        ("measured_batch_count", lambda data: data.__setitem__("measured_batch_count", 29)),
        (
            "quality_matched_pair_count",
            lambda data: data.__setitem__("quality_matched_pair_count", 0),
        ),
        (
            "software_speedup_claimed",
            lambda data: data.__setitem__("software_speedup_claimed", True),
        ),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", False)),
        ("gpu_speedup_claimed", lambda data: data.__setitem__("gpu_speedup_claimed", True)),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        ("fpga_or_tsu_used", lambda data: data.__setitem__("fpga_or_tsu_used", True)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "bad")),
        (
            "rust_batched_10x_ready_score",
            lambda data: data.__setitem__("rust_batched_10x_ready_score", 1.0),
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


def test_scenario_sample_5739_main_delegates_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-5739: CLI entrypoint delegates build and write steps."""

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
