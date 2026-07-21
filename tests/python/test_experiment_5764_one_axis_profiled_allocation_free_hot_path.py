"""Tests for Exp5764 profiled one-axis allocation-free hot path.

Spec refs: REQ-SAMPLE-5764, SCENARIO-SAMPLE-5764.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5764_one_axis_profiled_allocation_free_hot_path as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5764_one_axis_profiled_allocation_free_hot_path.py")


def _fake_ready_evidence(**_: Any) -> dict[str, Any]:
    phase_rows: list[dict[str, Any]] = []
    phase_share: list[dict[str, Any]] = []
    for size in mod.DEFAULT_PROBLEM_SIZES:
        total = 0.01
        for batch_kind in ("steady_state", "restart_containing"):
            for phase in mod.PHASE_DEFINITIONS:
                median = 0.006 if phase == "result_conversion" else 0.0005
                phase_rows.append(
                    {
                        "size": size,
                        "batch_kind": batch_kind,
                        "phase": phase,
                        "samples_s": [median, median * 1.01, median * 0.99],
                        "median_s": median,
                        "included_in_end_to_end": True,
                    }
                )
                phase_share.append(
                    {
                        "size": size,
                        "batch_kind": batch_kind,
                        "phase": phase,
                        "median_share": median / total,
                        "ci95": [median / total, median / total],
                    }
                )
    return {
        "phase_timing_receipts": phase_rows,
        "phase_share_by_size": phase_share,
        "dominant_phase": {
            "phase": "result_conversion",
            "median_phase_share": 0.6,
            "ci95": [0.6, 0.6],
            "optimized": True,
        },
        "dominant_phase_selection_receipt": {
            "preregistered_statistic": "median_phase_share",
            "selected_phase": "result_conversion",
            "optimized_only_selected_phase": True,
            "confidence_interval_excludes_tie": True,
        },
        "allocation_counts_before": {
            "rust_per_sample_heap_allocations": 2,
            "python_per_sample_heap_allocations": 1,
            "documented_unavoidable_boundaries": ["diagnostic_decision_log_dicts"],
        },
        "allocation_counts_after": {
            "rust_per_sample_heap_allocations": 0,
            "python_per_sample_heap_allocations": 0,
            "documented_unavoidable_boundaries": ["samples_numpy_array", "checkpoint_dict"],
        },
        "buffer_reuse_receipts": [
            {
                "path": "run_sweeps_compact",
                "contiguous_samples": True,
                "workspace_reused": True,
                "per_sample_heap_buffers": 0,
            }
        ],
        "worker_pool_receipts": [
            {
                "path": "run_sweeps_compact",
                "fixed_worker_count": 1,
                "dynamic_per_sample_workers": 0,
            }
        ],
        "checkpoint_compatibility": {
            "schema_version_preserved": True,
            "checkpoint_schema": "carnot.one_axis_samplerbackend.checkpoint.v1",
        },
        "restart_parity_receipts": [
            {"size": size, "restart_match": True, "suffix_hash_match": True}
            for size in mod.DEFAULT_PROBLEM_SIZES
        ],
        "fallback_equivalence_receipts": [
            {"size": size, "fallback_equivalent": True} for size in mod.DEFAULT_PROBLEM_SIZES
        ],
        "semantic_parity_score": 1.0,
        "distributional_parity_score": 1.0,
        "production_backend_reachable_score": 1.0,
    }


def test_req_sample_5764_spec_declares_profiled_hot_path_contract() -> None:
    """REQ-SAMPLE-5764: OpenSpec lists every required field and no-speed gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5764") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "`n=48`",
        "`n=96`",
        "`n=192`",
        "one larger feasible size",
        "producer_gate_fields",
        "timing_promotion_claimed=false",
        "hardware_speedup_claimed=false",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section or marker in normalized


def test_scenario_sample_5764_builds_ready_artifact_with_bare_gate_scores(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-5764: fake profiled evidence emits a valid ready artifact."""

    artifact = mod.build_artifact(
        root=REPO,
        evidence_runner=_fake_ready_evidence,
        freeze_affinity=False,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output = mod.write_output(tmp_path, artifact)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == artifact
    assert tuple(saved) == mod.REQUIRED_ARTIFACT_FIELDS
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["status"] == "complete"
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert saved["dominant_phase"]["phase"] == "result_conversion"
    assert saved["allocation_counts_after"]["rust_per_sample_heap_allocations"] == 0
    assert saved["allocation_counts_after"]["python_per_sample_heap_allocations"] == 0
    assert saved["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    for field in mod.PRODUCER_GATE_FIELDS:
        assert saved[field] == pytest.approx(1.0)
        assert not isinstance(saved[field], dict)
    assert saved["timing_promotion_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["two_axis_exchange_reopened"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_req_sample_5764_actual_small_compact_parity_receipt() -> None:
    """REQ-SAMPLE-5764: real compact path matches diagnostic and fallback semantics."""

    receipt = mod.run_compact_semantic_parity(
        problem_sizes=(3,),
        random_seeds=tuple(range(57_640, 57_650)),
        n_samples=2,
    )

    assert receipt["semantic_parity_score"] == 1.0
    assert receipt["distributional_parity_score"] == 1.0
    assert receipt["production_backend_reachable_score"] == 1.0
    assert all(row["restart_match"] is True for row in receipt["restart_parity_receipts"])
    assert all(
        row["fallback_equivalent"] is True for row in receipt["fallback_equivalence_receipts"]
    )


def test_req_sample_5764_real_phase_helpers_on_tiny_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAMPLE-5764: profiling helpers measure all preregistered phases."""

    monkeypatch.setattr(mod, "PHASE_REPETITIONS", 1)
    evidence = mod.run_profiled_evidence(
        root=REPO,
        problem_sizes=(3,),
        random_seeds=(57_660,),
    )

    assert {row["phase"] for row in evidence["phase_timing_receipts"]} == set(mod.PHASE_DEFINITIONS)
    assert {row["batch_kind"] for row in evidence["phase_timing_receipts"]} == {
        "steady_state",
        "restart_containing",
    }
    assert evidence["semantic_parity_score"] == 1.0
    assert evidence["allocation_counts_after"]["rust_per_sample_heap_allocations"] == 0
    assert (
        mod.dominant_phase_from(
            [
                {"phase": "result_conversion", "median_share": 0.7},
                {"phase": "validation", "median_share": 0.3},
            ]
        )["phase"]
        == "result_conversion"
    )

    item = mod._batch_item(mod._workload_for_size(3), 57_661, 1, compact=True)  # noqa: SLF001
    with pytest.raises(ValueError, match="unknown phase"):
        mod._time_phase("bad", item, include_restart=False)  # noqa: SLF001
    assert (
        mod.phase_share_by_size(
            [
                {
                    "size": 3,
                    "batch_kind": "steady_state",
                    "phase": "zero",
                    "samples_s": [0.0],
                    "median_s": 0.0,
                }
            ]
        )[0]["median_share"]
        == 0.0
    )
    assert mod._stable_float(-0.0) == 0.0  # noqa: SLF001


def test_req_sample_5764_validation_rejects_wrapped_gates_and_overclaims() -> None:
    """REQ-SAMPLE-5764: schema validation fails closed on unsafe manual edits."""

    artifact = mod.build_artifact(
        root=REPO,
        evidence_runner=_fake_ready_evidence,
        freeze_affinity=False,
    )
    blocked = deepcopy(artifact)
    blocked["optimized_path_ready_score"] = 0.0
    blocked["status"] = "blocked"
    assert mod.honest_verdict(blocked).startswith("blocked:")
    assert mod._gate_score(True) == 0.0  # noqa: SLF001

    mutations = [
        ("artifact fields", lambda data: data.pop("status")),
        ("field_principles", lambda data: data.__setitem__("field_principles", {})),
        ("spec_refs", lambda data: data.__setitem__("spec_refs", ["bad"])),
        (
            "semantic_parity_score",
            lambda data: data.__setitem__("semantic_parity_score", {"value": 1.0}),
        ),
        ("producer_gate_fields", lambda data: data.__setitem__("producer_gate_fields", [])),
        (
            "dominant_phase",
            lambda data: data["dominant_phase"].__setitem__("phase", "worker_scheduling"),
        ),
        (
            "result_conversion",
            lambda data: (
                data["dominant_phase"].__setitem__("phase", "worker_scheduling"),
                data["dominant_phase_selection_receipt"].__setitem__(
                    "selected_phase",
                    "worker_scheduling",
                ),
            ),
        ),
        (
            "allocation_counts_after rust",
            lambda data: data["allocation_counts_after"].__setitem__(
                "rust_per_sample_heap_allocations",
                1,
            ),
        ),
        (
            "allocation_counts_after python",
            lambda data: data["allocation_counts_after"].__setitem__(
                "python_per_sample_heap_allocations",
                1,
            ),
        ),
        (
            "timing_promotion_claimed",
            lambda data: data.__setitem__("timing_promotion_claimed", True),
        ),
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
            "optimized_path_ready_score",
            lambda data: data.__setitem__("optimized_path_ready_score", 0.0),
        ),
        ("status", lambda data: data.__setitem__("status", "blocked")),
        ("honest_verdict prefix", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "honest_verdict mismatch",
            lambda data: data.__setitem__("honest_verdict", "complete: wrong"),
        ),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "bad"),
        ),
    ]

    for label, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if label != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(bad)


def test_req_sample_5764_environment_receipt_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-5764: host and build receipt edge cases are explicit."""

    assert mod.source_hashes_before(tmp_path) == {"source": "missing_exp5758"}

    class Completed:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "bad\n"

    class Competing:
        returncode = 0
        stdout = "999 cargo experiment_5739_one_axis --bench\n"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Completed())
    assert mod.release_build_receipt(REPO, run_build=True)["completed"] is True
    assert mod._competing_processes() == []  # noqa: SLF001

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Failed())
    assert mod._competing_processes() == []  # noqa: SLF001
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Competing())
    assert mod._competing_processes()[0]["pid"] == 999  # noqa: SLF001

    if hasattr(mod.os, "sched_getaffinity"):
        monkeypatch.delattr(mod.os, "sched_getaffinity")
        assert mod.affinity_receipt(freeze_affinity=True)["observable"] is False
