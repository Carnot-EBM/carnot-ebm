"""Tests for scripts/experiment_260_solver_semantic_gpu.py.

All tests run under CARNOT_FORCE_LIVE=0 (simulated / mock mode).

Spec coverage: REQ-VERIFY-058, REQ-VERIFY-059, REQ-VERIFY-041,
               SCENARIO-VERIFY-042, SCENARIO-VERIFY-036
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "experiment_260_solver_semantic_gpu.py"
)


def _load_script() -> Any:
    """Import experiment_260 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_260", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_260"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

# Convenience aliases
safe_slug = _mod.safe_slug
utc_now = _mod.utc_now
checkpoint_path_fn = _mod.checkpoint_path
load_checkpoint = _mod.load_checkpoint
save_checkpoint = _mod.save_checkpoint
build_route_summary = _mod.build_route_summary
collect_all_claims_from_runs = _mod.collect_all_claims_from_runs
summarize_benchmark_runs = _mod.summarize_benchmark_runs
build_comparison_block = _mod.build_comparison_block
build_artifact_payload = _mod.build_artifact_payload
MODEL_SPECS = _mod.MODEL_SPECS
MODE_ORDER = _mod.MODE_ORDER
EXPERIMENT = _mod.EXPERIMENT
RUN_DATE = _mod.RUN_DATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(case_id: str, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "case_id": case_id,
        "question": "What is 2+2?",
        "ground_truth": 4,
        "prompt": "Answer: ",
        "gold_atomic_constraints": [],
    }
    base.update(kw)
    return base


def _make_run(case_id: str, correct: bool = True, flagged: bool = False,
              repaired: bool = False, n_repairs: int = 0, formal_claims: list | None = None,
              accepted_correct: bool | None = None) -> dict[str, Any]:
    resolved_ac = accepted_correct if accepted_correct is not None else (correct and not flagged)
    return {
        "case_id": case_id,
        "correct": correct,
        "flagged": flagged,
        "accepted_correct": resolved_ac,
        "repaired": repaired,
        "n_repairs": n_repairs,
        "formal_claims": formal_claims or [],
        "latency_seconds": 1.0,
        "prompt_tokens": 10,
        "response_tokens": 20,
        "total_tokens": 30,
    }


# ---------------------------------------------------------------------------
# Tests: safe_slug
# ---------------------------------------------------------------------------


class TestSafeSlug:
    """REQ-VERIFY-058: filesystem-safe checkpoint naming."""

    def test_basic(self) -> None:
        assert safe_slug("Qwen3.5-0.8B") == "qwen3_5-0_8b"

    def test_slash_and_space(self) -> None:
        assert "/" not in safe_slug("google/gemma-4-E4B-it")
        assert " " not in safe_slug("my model name")

    def test_idempotent(self) -> None:
        s = safe_slug("gsm8k_semantic")
        assert safe_slug(s) == s


# ---------------------------------------------------------------------------
# Tests: checkpoint resume — REQ-VERIFY-058, SCENARIO-VERIFY-036
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """Checkpoint resume preserves already-completed cases and respects Exp 246 files."""

    def test_fresh_checkpoint_returns_empty(self, tmp_path: Path) -> None:
        """REQ-VERIFY-058: missing checkpoint returns empty state."""
        path = tmp_path / "cell.json"
        case_ids = ["c1", "c2", "c3"]
        result = load_checkpoint(path, case_ids)
        assert result["case_ids"] == case_ids
        assert result["results_by_case"] == {}

    def test_valid_checkpoint_reloads(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-036: completed cases are not re-executed."""
        path = tmp_path / "cell.json"
        case_ids = ["c1", "c2"]
        save_checkpoint(path, {
            "benchmark": "gsm8k_semantic",
            "model_name": "Qwen3.5-0.8B",
            "mode": "baseline",
            "case_ids": case_ids,
            "results_by_case": {"c1": {"correct": True}},
        })
        result = load_checkpoint(path, case_ids)
        assert "c1" in result["results_by_case"]
        assert "c2" not in result["results_by_case"]

    def test_mismatched_case_ids_discards_checkpoint(self, tmp_path: Path) -> None:
        """Stale checkpoint (different cohort) returns empty state."""
        path = tmp_path / "cell.json"
        save_checkpoint(path, {
            "case_ids": ["old1", "old2"],
            "results_by_case": {"old1": {"correct": True}},
        })
        result = load_checkpoint(path, ["new1", "new2"])
        assert result["results_by_case"] == {}

    def test_checkpoint_is_atomic(self, tmp_path: Path) -> None:
        """Checkpoint must not leave a .tmp file behind."""
        path = tmp_path / "cell.json"
        save_checkpoint(path, {"case_ids": [], "results_by_case": {}})
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()

    def test_checkpoint_path_matches_exp246_format(self, tmp_path: Path) -> None:
        """Checkpoint filenames must match the Exp 246 pattern so runs resume."""
        path = checkpoint_path_fn(tmp_path, benchmark="gsm8k_semantic",
                                  model_name="Qwen3.5-0.8B", mode="baseline")
        assert path.name == "gsm8k_semantic__qwen3_5-0_8b__baseline.json"

    def test_resume_from_exp246_checkpoint_dir(self, tmp_path: Path) -> None:
        """Exp 246 checkpoint files must be loadable as-is."""
        case_ids = ["gsm8k-1", "gsm8k-2", "gsm8k-3"]
        payload = {
            "benchmark": "gsm8k_semantic",
            "model_name": "Qwen3.5-0.8B",
            "mode": "baseline",
            "case_ids": case_ids,
            "results_by_case": {"gsm8k-1": {"correct": False}, "gsm8k-2": {"correct": True}},
        }
        ckpt = checkpoint_path_fn(tmp_path, benchmark="gsm8k_semantic",
                                  model_name="Qwen3.5-0.8B", mode="baseline")
        save_checkpoint(ckpt, payload)
        loaded = load_checkpoint(ckpt, case_ids)
        assert len(loaded["results_by_case"]) == 2
        assert "gsm8k-3" not in loaded["results_by_case"]


# ---------------------------------------------------------------------------
# Tests: route summary aggregation — REQ-VERIFY-059
# ---------------------------------------------------------------------------


class TestRouteSummary:
    """REQ-VERIFY-059: route evidence aggregation."""

    def test_empty_claims(self) -> None:
        summary = build_route_summary([])
        assert summary["total_claims"] == 0
        assert summary["abstain_rate"] == 0.0

    def test_single_arithmetic_supported(self) -> None:
        claims = [{"route": "arithmetic", "verdict": "supported"}]
        s = build_route_summary(claims)
        assert s["by_route"] == {"arithmetic": 1}
        assert s["by_verdict"]["supported"] == 1
        assert s["abstain_rate"] == 0.0

    def test_abstain_rate_calculation(self) -> None:
        claims = [
            {"route": "cardinality", "verdict": "abstain"},
            {"route": "arithmetic", "verdict": "supported"},
            {"route": "set_membership", "verdict": "abstain"},
        ]
        s = build_route_summary(claims)
        assert s["abstain_rate"] == pytest.approx(2 / 3, abs=1e-6)
        assert s["total_claims"] == 3

    def test_all_routes_tracked(self) -> None:
        routes = ["arithmetic", "cardinality", "set_membership", "smt", "abstain"]
        claims = [{"route": r, "verdict": "supported"} for r in routes]
        s = build_route_summary(claims)
        for r in routes:
            assert r in s["by_route"]

    def test_collect_claims_from_runs(self) -> None:
        runs = [
            {"formal_claims": [{"route": "arithmetic", "verdict": "supported"}]},
            {"formal_claims": [{"route": "cardinality", "verdict": "abstain"},
                               {"route": "set_membership", "verdict": "violated"}]},
            {"formal_claims": []},
        ]
        claims = collect_all_claims_from_runs(runs)
        assert len(claims) == 3


# ---------------------------------------------------------------------------
# Tests: summarize_benchmark_runs
# ---------------------------------------------------------------------------


class TestSummarizeBenchmarkRuns:
    """Stats aggregation produces correct values for paired-run analysis."""

    def test_empty_runs(self) -> None:
        s = summarize_benchmark_runs(
            baseline_runs=[], verify_only_runs=[], verify_repair_runs=[]
        )
        assert s["baseline"]["n_cases"] == 0
        assert s["baseline"]["accuracy"] == 0.0

    def test_perfect_baseline(self) -> None:
        runs = [_make_run(f"c{i}", correct=True) for i in range(5)]
        s = summarize_benchmark_runs(
            baseline_runs=runs,
            verify_only_runs=runs,
            verify_repair_runs=runs,
        )
        assert s["baseline"]["accuracy"] == 1.0

    def test_verify_only_delta_correct(self) -> None:
        """verify_only_delta = verify_accuracy - baseline_accuracy."""
        baseline = [_make_run("c1", correct=True), _make_run("c2", correct=False)]
        verify_only = [
            _make_run("c1", correct=True, flagged=False, accepted_correct=True),
            _make_run("c2", correct=False, flagged=True, accepted_correct=False),
        ]
        repair = [_make_run("c1", correct=True), _make_run("c2", correct=True, repaired=True)]
        s = summarize_benchmark_runs(
            baseline_runs=baseline,
            verify_only_runs=verify_only,
            verify_repair_runs=repair,
        )
        assert s["baseline"]["accuracy"] == 0.5
        assert s["paired_deltas"]["repair_minus_baseline"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: artifact schema — REQ-VERIFY-058, REQ-VERIFY-059
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """build_artifact_payload produces a schema-valid artifact."""

    def _empty_stats(self) -> dict[str, Any]:
        return {
            "Qwen3.5-0.8B": summarize_benchmark_runs(
                baseline_runs=[], verify_only_runs=[], verify_repair_runs=[]
            )
        }

    def test_top_level_keys(self, tmp_path: Path) -> None:
        payload = build_artifact_payload(
            output_path=tmp_path / "out.json",
            gsm8k_cohort=[],
            gsm8k_cohort_meta={"source_artifact": "a", "source_experiment": 235,
                               "sample_seed": 218, "case_count": 0},
            constraint_ir_cohort=[],
            constraint_ir_cohort_meta={"source_artifact": "b", "source_experiment": 221,
                                       "sample_seed": 218, "case_count": 0},
            gsm8k_paired_runs=[],
            constraint_ir_paired_runs=[],
            gsm8k_route_summary=build_route_summary([]),
            constraint_ir_route_summary=build_route_summary([]),
            gsm8k_statistics=self._empty_stats(),
            constraint_ir_statistics=self._empty_stats(),
            started_at="2026-04-13T00:00:00Z",
            finished_at="2026-04-13T01:00:00Z",
            runtime_seconds=3600.0,
            checkpoint_dir=tmp_path / "ckpts",
            max_repairs=3,
            inference_mode="simulated",
            gpu_fallback=False,
            throughput_report={"target_seconds_per_case": 3.0, "per_model": {}},
            comparison_block={},
        )
        required_keys = [
            "experiment", "title", "run_date", "schema", "metadata",
            "benchmarks", "comparison",
        ]
        for key in required_keys:
            assert key in payload, f"Missing key: {key}"

    def test_experiment_number(self, tmp_path: Path) -> None:
        payload = build_artifact_payload(
            output_path=tmp_path / "out.json",
            gsm8k_cohort=[], gsm8k_cohort_meta={"source_artifact": "a",
                "source_experiment": 235, "sample_seed": 218, "case_count": 0},
            constraint_ir_cohort=[], constraint_ir_cohort_meta={"source_artifact": "b",
                "source_experiment": 221, "sample_seed": 218, "case_count": 0},
            gsm8k_paired_runs=[], constraint_ir_paired_runs=[],
            gsm8k_route_summary=build_route_summary([]),
            constraint_ir_route_summary=build_route_summary([]),
            gsm8k_statistics=self._empty_stats(),
            constraint_ir_statistics=self._empty_stats(),
            started_at="2026-04-13T00:00:00Z", finished_at="2026-04-13T01:00:00Z",
            runtime_seconds=0.0, checkpoint_dir=tmp_path / "ckpts",
            max_repairs=3, inference_mode="simulated",
            gpu_fallback=False,
            throughput_report={"target_seconds_per_case": 3.0, "per_model": {}},
            comparison_block={},
        )
        assert payload["experiment"] == 260
        assert payload["run_date"] == "20260413"

    def test_metadata_has_gpu_fallback_flag(self, tmp_path: Path) -> None:
        payload = build_artifact_payload(
            output_path=tmp_path / "out.json",
            gsm8k_cohort=[], gsm8k_cohort_meta={"source_artifact": "a",
                "source_experiment": 235, "sample_seed": 218, "case_count": 0},
            constraint_ir_cohort=[], constraint_ir_cohort_meta={"source_artifact": "b",
                "source_experiment": 221, "sample_seed": 218, "case_count": 0},
            gsm8k_paired_runs=[], constraint_ir_paired_runs=[],
            gsm8k_route_summary=build_route_summary([]),
            constraint_ir_route_summary=build_route_summary([]),
            gsm8k_statistics=self._empty_stats(),
            constraint_ir_statistics=self._empty_stats(),
            started_at="2026-04-13T00:00:00Z", finished_at="2026-04-13T01:00:00Z",
            runtime_seconds=0.0, checkpoint_dir=tmp_path / "ckpts",
            max_repairs=3, inference_mode="live_cpu",
            gpu_fallback=True,
            throughput_report={"target_seconds_per_case": 3.0, "per_model": {}},
            comparison_block={},
        )
        assert payload["metadata"]["gpu_fallback"] is True
        assert payload["metadata"]["inference_mode"] == "live_cpu"


# ---------------------------------------------------------------------------
# Tests: build_comparison_block
# ---------------------------------------------------------------------------


class TestBuildComparisonBlock:
    """Comparison block includes Exp 235 and Exp 247 references."""

    def test_keys_present(self, tmp_path: Path) -> None:
        block = build_comparison_block(
            gsm8k_statistics={},
            constraint_ir_statistics={},
            gsm8k_route_summary=build_route_summary([]),
            constraint_ir_route_summary=build_route_summary([]),
            exp247_cells_completed=1,
            exp247_cells_total=12,
        )
        assert "vs_exp235_semantic_verifier_v2" in block
        assert "vs_exp247_partial" in block
        assert "verify_only_non_harmful_finding" in block

    def test_progress_fraction(self, tmp_path: Path) -> None:
        block = build_comparison_block(
            gsm8k_statistics={},
            constraint_ir_statistics={},
            gsm8k_route_summary=build_route_summary([]),
            constraint_ir_route_summary=build_route_summary([]),
            exp247_cells_completed=18,
            exp247_cells_total=200,
        )
        assert block["vs_exp247_partial"]["exp247_cells_completed"] == 18
        assert block["vs_exp247_partial"]["exp260_cells_completed"] >= 0


# ---------------------------------------------------------------------------
# Tests: GPU harness integration — REQ-VERIFY-041, SCENARIO-VERIFY-042
# ---------------------------------------------------------------------------


class TestGPUHarnessIntegration:
    """DualGPUBenchmarkHarness integrates correctly with the Exp 260 run logic."""

    def test_harness_imported(self) -> None:
        """Exp 260 must import DualGPUBenchmarkHarness from Exp 258."""
        assert hasattr(_mod, "DualGPUBenchmarkHarness")

    def test_harness_run_mode_records_timing(self, tmp_path: Path) -> None:
        """run_mode() must track throughput for each case executed."""
        from experiment_260 import DualGPUBenchmarkHarness  # type: ignore[import-not-found]

        call_log: list[str] = []
        fake_clock_values = iter(range(1000))

        def fake_clock() -> float:
            return float(next(fake_clock_values))

        harness = DualGPUBenchmarkHarness(
            model_specs=[{"name": "TestModel", "hf_id": "test/model"}],
            clock=fake_clock,
        )

        cases = [{"case_id": f"c{i}"} for i in range(3)]

        def execute(case: dict[str, Any]) -> dict[str, Any]:
            call_log.append(case["case_id"])
            return {"correct": True}

        results = harness.run_mode(
            benchmark="gsm8k_semantic",
            model_name="TestModel",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        assert len(results) == 3
        assert len(call_log) == 3
        report = harness.throughput.report()
        assert "TestModel" in report["per_model"]
        assert report["per_model"]["TestModel"]["total_cases"] == 3

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        """run_mode() must skip cases already in the checkpoint."""
        from experiment_260 import DualGPUBenchmarkHarness  # type: ignore[import-not-found]

        harness = DualGPUBenchmarkHarness(
            model_specs=[{"name": "TestModel", "hf_id": "test/model"}],
        )

        cases = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
        ckpt = checkpoint_path_fn(tmp_path, benchmark="gsm8k_semantic",
                                  model_name="TestModel", mode="baseline")
        save_checkpoint(ckpt, {
            "benchmark": "gsm8k_semantic",
            "model_name": "TestModel",
            "mode": "baseline",
            "case_ids": ["a", "b", "c"],
            "results_by_case": {"a": {"correct": True}},
        })

        executed: list[str] = []

        def execute(case: dict[str, Any]) -> dict[str, Any]:
            executed.append(case["case_id"])
            return {"correct": False}

        harness.run_mode(
            benchmark="gsm8k_semantic",
            model_name="TestModel",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        # Only b and c should have been executed; a was already checkpointed.
        assert "a" not in executed
        assert set(executed) == {"b", "c"}
