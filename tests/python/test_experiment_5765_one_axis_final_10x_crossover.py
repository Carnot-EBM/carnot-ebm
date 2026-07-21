"""Tests for Exp5765 final one-axis Rust/Python 10x crossover.

Spec refs: REQ-SAMPLE-5765, SCENARIO-SAMPLE-5765.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5765_one_axis_final_10x_crossover as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5765_one_axis_final_10x_crossover.py")


def _fake_evidence(
    *,
    manifest: dict[str, Any],
    pass_10x: bool,
    quality_passed: bool = True,
    **_: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    semantic: list[dict[str, Any]] = []
    restart: list[dict[str, Any]] = []
    distributional: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for size in manifest["cell_sizes"]:
        ratios = []
        for batch_index in range(manifest["paired_batches_per_cell"]):
            if pass_10x and size in {96, 192}:
                ratio = 12.5
            else:
                ratio = 1.25
            rust_s = 0.001
            python_s = rust_s * ratio
            ratios.append(ratio)
            pair_id = f"n{size}:batch{batch_index}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "size": size,
                    "batch_index": batch_index,
                    "path_order": [mod.RUST_ARM, mod.PYTHON_ARM]
                    if batch_index % 2 == 0
                    else [mod.PYTHON_ARM, mod.RUST_ARM],
                    "rust_end_to_end_s": rust_s,
                    "python_end_to_end_s": python_s,
                    "speedup_ratio": ratio,
                    "phase_receipts": {
                        mod.RUST_ARM: {
                            "setup_s": 0.0001,
                            "sample_batch_s": 0.0007,
                            "validation_s": 0.0001,
                            "restart_s": 0.0001,
                        },
                        mod.PYTHON_ARM: {
                            "setup_s": 0.0001,
                            "sample_batch_s": python_s - 0.0003,
                            "validation_s": 0.0001,
                            "restart_s": 0.0001,
                        },
                    },
                    "included": quality_passed,
                    "exclusion_reason": None if quality_passed else "semantic_parity_failed",
                }
            )
            if not quality_passed:
                exclusions.append(
                    {
                        "pair_id": pair_id,
                        "size": size,
                        "batch_index": batch_index,
                        "reason": "semantic_parity_failed",
                    }
                )
        quality.append(
            {
                "size": size,
                "pair_count": manifest["paired_batches_per_cell"],
                "quality_matched": quality_passed,
                "energy_delta_abs_max": 0.0 if quality_passed else 1.0,
                "feasibility_match": quality_passed,
                "acceptance_delta_abs": 0.0,
                "retained_sample_count": manifest["paired_batches_per_cell"]
                * manifest["retained_samples_per_batch"],
                "ess_min": float(manifest["paired_batches_per_cell"]),
                "autocorrelation_abs_max": 0.0,
                "median_speedup_ratio": ratios[len(ratios) // 2],
            }
        )
        semantic.append(
            {"size": size, "passed": quality_passed, "sample_hash_match": quality_passed}
        )
        restart.append(
            {
                "size": size,
                "passed": quality_passed,
                "checkpoint_hash_match": quality_passed,
                "restart_suffix_hash_match": quality_passed,
            }
        )
        distributional.append(
            {
                "size": size,
                "passed": quality_passed,
                "energy_histogram_tv": 0.0 if quality_passed else 1.0,
                "mean_energy_delta_abs": 0.0 if quality_passed else 1.0,
                "best_energy_delta_abs": 0.0 if quality_passed else 1.0,
            }
        )
    return {
        "raw_timing_receipts": rows,
        "warmup_receipts": [
            {
                "size": size,
                "warmup_batches": manifest["warmup_batches"],
                "stable": True,
                "rust_active_backend": mod.ACTIVE_RUST_BACKEND,
                "python_active_backend": mod.ACTIVE_PYTHON_FALLBACK,
            }
            for size in manifest["cell_sizes"]
        ],
        "quality_metrics_by_cell": quality,
        "semantic_parity_by_cell": semantic,
        "restart_parity_by_cell": restart,
        "distributional_parity_by_cell": distributional,
        "fallback_equivalence": {
            "passed": quality_passed,
            "exact_fallback_equivalence": quality_passed,
        },
        "production_backend_reachable": {
            "passed": quality_passed,
            "active_backend": mod.ACTIVE_RUST_BACKEND,
            "optimized_hot_path_used": quality_passed,
        },
        "exclusion_manifest": {
            "preregistered_reasons": list(mod.PREREGISTERED_EXCLUSION_REASONS),
            "exclusions": exclusions,
        },
    }


def _fake_10x_runner(**kwargs: Any) -> dict[str, Any]:
    return _fake_evidence(pass_10x=True, **kwargs)


def _fake_null_runner(**kwargs: Any) -> dict[str, Any]:
    return _fake_evidence(pass_10x=False, **kwargs)


def _fake_quality_fail_runner(**kwargs: Any) -> dict[str, Any]:
    return _fake_evidence(pass_10x=True, quality_passed=False, **kwargs)


def test_req_sample_5765_spec_declares_final_rule_and_artifact_fields() -> None:
    """REQ-SAMPLE-5765: OpenSpec lists the final 10x rule and retirement fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5765") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "at least thirty paired measured batches",
        "paired lower confidence bound is at least `10.0` at two consecutive larger sizes",
        "`rust_10x_claimed=false`",
        "`rust_10x_retired=true`",
        "`hardware_speedup_claimed=false`",
        "`two_axis_exchange_reopened=false`",
        mod.INFERENCE_SUBSTRATE,
        "SCENARIO-SAMPLE-5765",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_sample_5765_builds_claim_artifact_under_strict_rule(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-5765: fake matched evidence can claim only the final 10x rule."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_10x_runner,
        freeze_affinity=False,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output = mod.write_output(tmp_path, artifact)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == artifact
    assert tuple(saved) == mod.REQUIRED_ARTIFACT_FIELDS
    assert set(saved["field_principles"]) == set(saved)
    assert saved["status"] == "complete"
    assert saved["cell_sizes"] == [48, 96, 192, 256]
    assert saved["paired_batches_per_cell"] == 30
    assert saved["benchmark_manifest_hash"] == mod.sha256_json(saved["benchmark_manifest"])
    assert saved["speedup_lcb_by_size"]["96"] >= 10.0
    assert saved["speedup_lcb_by_size"]["192"] >= 10.0
    assert saved["consecutive_larger_size_rule_passed"] is True
    assert saved["matched_quality_gate_passed"] is True
    assert saved["rust_10x_claimed"] is True
    assert saved["rust_10x_retired"] is False
    assert saved["nfr01_status"] == "qualified_for_this_one_axis_pyo3_technique"
    assert saved["hardware_speedup_claimed"] is False
    assert saved["two_axis_exchange_reopened"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_req_sample_5765_terminal_retirement_when_final_rule_fails() -> None:
    """REQ-SAMPLE-5765: a repeated matched-quality null retires only this technique."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_null_runner,
        freeze_affinity=False,
    )

    assert artifact["consecutive_larger_size_rule_passed"] is False
    assert artifact["matched_quality_gate_passed"] is True
    assert artifact["rust_10x_claimed"] is False
    assert artifact["rust_10x_retired"] is True
    assert artifact["remaining_bottleneck"]
    assert artifact["nfr01_status"] == "retired_allocation_free_one_axis_pyo3_technique"
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_sample_5765_quality_failure_blocks_without_retirement() -> None:
    """REQ-SAMPLE-5765: quality mismatch blocks timing interpretation without a claim."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_quality_fail_runner,
        freeze_affinity=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["matched_quality_gate_passed"] is False
    assert artifact["rust_10x_claimed"] is False
    assert artifact["rust_10x_retired"] is False
    assert artifact["nfr01_status"] == "blocked_quality_or_precondition_gate"
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_sample_5765_actual_small_runner_collects_matched_timing() -> None:
    """REQ-SAMPLE-5765: tiny real cells preserve parity before speed ratios count."""

    manifest = mod.benchmark_manifest(
        cell_sizes=(3,),
        random_seeds=(57_650, 57_651),
        paired_batches_per_cell=2,
        allow_underpowered=True,
    )
    evidence = mod.run_matched_release_benchmark(
        manifest=manifest,
        cell_sizes=(3,),
        random_seeds=(57_650, 57_651),
    )

    assert len(evidence["raw_timing_receipts"]) == 2
    assert all(row["included"] is True for row in evidence["raw_timing_receipts"])
    assert evidence["quality_metrics_by_cell"][0]["quality_matched"] is True
    assert evidence["semantic_parity_by_cell"][0]["passed"] is True
    assert evidence["restart_parity_by_cell"][0]["passed"] is True
    assert evidence["distributional_parity_by_cell"][0]["passed"] is True
    assert evidence["fallback_equivalence"]["passed"] is True
    assert evidence["production_backend_reachable"]["optimized_hot_path_used"] is True
    assert (
        mod.speedup_intervals_from_raw(evidence["raw_timing_receipts"], cell_sizes=(3,))["median"][
            "aggregate"
        ]
        > 0.0
    )


def test_req_sample_5765_validation_rejects_overclaims_and_bad_schema() -> None:
    """REQ-SAMPLE-5765: unsafe manual edits fail closed."""

    artifact = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_null_runner,
        freeze_affinity=False,
    )
    mutations = [
        ("artifact fields", lambda data: data.pop("status")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("bad", "bad")),
        ("spec_refs", lambda data: data.__setitem__("spec_refs", ["bad"])),
        ("paired_batches_per_cell", lambda data: data.__setitem__("paired_batches_per_cell", 29)),
        ("cell_sizes", lambda data: data.__setitem__("cell_sizes", [48, 96])),
        ("one larger", lambda data: data.__setitem__("cell_sizes", [48, 96, 192])),
        (
            "benchmark_manifest_hash",
            lambda data: data.__setitem__("benchmark_manifest_hash", "bad"),
        ),
        (
            "matched_quality_gate_passed",
            lambda data: data.__setitem__("matched_quality_gate_passed", {"value": True}),
        ),
        ("rust_10x_claimed", lambda data: data.__setitem__("rust_10x_claimed", True)),
        ("rust_10x_retired", lambda data: data.__setitem__("rust_10x_retired", False)),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        (
            "two_axis_exchange_reopened",
            lambda data: data.__setitem__("two_axis_exchange_reopened", True),
        ),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "bad")),
        (
            "nfr01_status",
            lambda data: data.__setitem__(
                "nfr01_status", "qualified_for_this_one_axis_pyo3_technique"
            ),
        ),
        ("status", lambda data: data.__setitem__("status", "blocked")),
        ("status", lambda data: data.__setitem__("status", "bad")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "complete: bad")),
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


def test_req_sample_5765_defensive_branches_and_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-5765: edge guards preserve blocked evidence instead of overclaiming."""

    manifest_error_cases = [
        ({"cell_sizes": (48, 96), "random_seeds": tuple(range(30))}, "48, 96, and 192"),
        ({"cell_sizes": (48, 96, 192), "random_seeds": tuple(range(30))}, "larger feasible"),
        ({"paired_batches_per_cell": 29}, "at least thirty"),
        ({"random_seeds": tuple(range(29))}, "cover every paired batch"),
        ({"cell_sizes": (), "allow_underpowered": True}, "positive"),
        ({"cell_sizes": (48, 48), "allow_underpowered": True}, "unique"),
        ({"random_seeds": (), "allow_underpowered": True}, "must not be empty"),
        ({"warmup_batches": -1, "allow_underpowered": True}, "nonnegative"),
        ({"retained_samples_per_batch": 0, "allow_underpowered": True}, "positive"),
        ({"burn_in_sweeps": -1, "allow_underpowered": True}, "nonnegative"),
    ]
    for kwargs, message in manifest_error_cases:
        with pytest.raises(ValueError, match=message):
            mod.benchmark_manifest(**kwargs)

    monkeypatch.setattr(mod, "all_preconditions_passed", lambda _: False)
    blocked = mod.build_artifact(
        root=REPO,
        benchmark_runner=_fake_10x_runner,
        freeze_affinity=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["raw_timing_receipts"] == []
    assert blocked["remaining_bottleneck"] == "quality_or_precondition_gate_blocked_timing"
    mod.validate_artifact(blocked)

    zero_or_single = mod.speedup_intervals_from_raw(
        [
            {"size": 48, "included": True, "speedup_ratio": 0.0},
            {"size": 48, "included": True, "speedup_ratio": 2.0},
        ],
        cell_sizes=(48,),
    )
    assert zero_or_single["median"]["48"] == 2.0
    assert zero_or_single["lcb"]["48"] == 2.0
    assert mod.bootstrap_interval([], seed=1) == [None, None]
    assert mod.remaining_bottleneck({"raw_timing_receipts": []}, claimed=False).startswith(
        "quality_or_precondition"
    )
    assert mod.remaining_bottleneck(
        {"raw_timing_receipts": [{"rust_end_to_end_s": 0.0, "phase_receipts": {}}]},
        claimed=False,
    ).startswith("quality_or_precondition")
    assert mod.remaining_bottleneck({"raw_timing_receipts": []}, claimed=True).startswith(
        "none_10x"
    )
    assert (
        mod.nfr01_status(claimed=False, retired=False, status="complete")
        == "not_qualified_without_retirement"
    )

    quality = {
        "sample_hash_match": True,
        "decision_log_hash_match": True,
        "restart_suffix_hash_match": True,
        "energy_histogram_tv": 0.0,
        "acceptance_delta_abs": 0.0,
    }
    measured = {
        mod.RUST_ARM: {
            "active_backend": mod.ACTIVE_RUST_BACKEND,
            "optimized_hot_path_used": True,
            "end_to_end_s": 1.0,
        },
        mod.PYTHON_ARM: {"active_backend": mod.ACTIVE_PYTHON_FALLBACK},
    }
    assert mod._pair_exclusion_reason(quality, measured) is None  # noqa: SLF001
    rust_bad = deepcopy(measured)
    rust_bad[mod.RUST_ARM]["active_backend"] = "fallback"
    assert mod._pair_exclusion_reason(quality, rust_bad) == "production_backend_not_reachable"  # noqa: SLF001
    python_bad = deepcopy(measured)
    python_bad[mod.PYTHON_ARM]["active_backend"] = "rust"
    assert mod._pair_exclusion_reason(quality, python_bad) == "fallback_equivalence_failed"  # noqa: SLF001
    optimized_bad = deepcopy(measured)
    optimized_bad[mod.RUST_ARM]["optimized_hot_path_used"] = False
    assert mod._pair_exclusion_reason(quality, optimized_bad) == "production_backend_not_reachable"  # noqa: SLF001
    semantic_bad = {**quality, "sample_hash_match": False}
    assert mod._pair_exclusion_reason(semantic_bad, measured) == "semantic_parity_failed"  # noqa: SLF001
    acceptance_bad = {**quality, "acceptance_delta_abs": 1.0}
    assert mod._pair_exclusion_reason(acceptance_bad, measured) == "semantic_parity_failed"  # noqa: SLF001
    restart_bad = {**quality, "restart_suffix_hash_match": False}
    assert mod._pair_exclusion_reason(restart_bad, measured) == "restart_parity_failed"  # noqa: SLF001
    distribution_bad = {**quality, "energy_histogram_tv": 1.0}
    assert mod._pair_exclusion_reason(distribution_bad, measured) == "distributional_parity_failed"  # noqa: SLF001
    timing_bad = deepcopy(measured)
    timing_bad[mod.RUST_ARM]["end_to_end_s"] = 0.0
    assert mod._pair_exclusion_reason(quality, timing_bad) == "timing_nonpositive"  # noqa: SLF001

    monkeypatch.setattr(mod.exp5764, "_command_output", lambda _: {"exit_code": 1, "lines": []})
    assert mod._competing_processes() == []  # noqa: SLF001
    monkeypatch.setattr(
        mod.exp5764,
        "_command_output",
        lambda _: {
            "exit_code": 0,
            "lines": [
                "999 python python experiment_5765_one_axis_final_10x_crossover.py",
                "1000 pytest pytest test_experiment_5765_one_axis_final_10x_crossover.py",
                "1001 bash bash -lc python -m carnot.experiment_5765_one_axis_final_10x_crossover",
            ],
        },
    )
    assert mod._competing_processes() == [  # noqa: SLF001
        {
            "pid": 999,
            "command": "python",
            "args": "python experiment_5765_one_axis_final_10x_crossover.py",
        }
    ]

    with mod._temporary_affinity([0], enabled=False):  # noqa: SLF001
        assert True
    affinity_calls: list[set[int]] = []
    monkeypatch.setattr(mod.os, "sched_getaffinity", lambda _: {2, 3})
    monkeypatch.setattr(
        mod.os,
        "sched_setaffinity",
        lambda _pid, cpus: affinity_calls.append(set(cpus)),
    )
    with mod._temporary_affinity([], enabled=True):  # noqa: SLF001
        assert affinity_calls == []
    with mod._temporary_affinity([2], enabled=True):  # noqa: SLF001
        assert affinity_calls == [{2}]
    assert affinity_calls == [{2}, {2, 3}]


def test_scenario_sample_5765_main_delegates_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-5765: CLI entrypoint delegates build and write steps."""

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
