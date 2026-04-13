"""Tests for scripts/experiment_261_humaneval_qwen_gpu.py.

Covers:
  - GPU harness integration: run_mode checkpointing, resume, throughput tracking
  - Artifact schema: required top-level keys, cohort block, statistics block
  - Cross-model comparison block structure: Exp 261 Qwen vs Exp 226 Gemma schema mapping

All tests run under CARNOT_FORCE_LIVE=0 (no live GPU required).

Spec: REQ-CODE-028, REQ-CODE-029, REQ-CODE-030,
      REQ-VERIFY-061, REQ-VERIFY-062, REQ-VERIFY-041
SCENARIO-CODE-026, SCENARIO-CODE-027, SCENARIO-CODE-028,
SCENARIO-VERIFY-065, SCENARIO-VERIFY-066, SCENARIO-VERIFY-069
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_261_humaneval_qwen_gpu.py"


# ---------------------------------------------------------------------------
# Module loader (no live GPU, no real model loading)
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_261 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_261", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_261"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    python_dir = str(REPO_ROOT / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()


# ---------------------------------------------------------------------------
# Fixtures: minimal case and result builders
# ---------------------------------------------------------------------------


def _make_case(idx: int) -> dict[str, Any]:
    """Return a minimal HumanEval-style case compatible with Exp 261 schema."""
    seed = 100000 + idx * 37
    return {
        "case_id": f"humaneval-{idx}",
        "dataset_idx": idx,
        "task_id": f"HumanEval/{idx}",
        "prompt": f"def fn_{idx}(x: int) -> int:\n    pass\n",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        "entry_point": f"fn_{idx}",
        "sample_position": idx + 1,
        "prompt_seeds": {"baseline": seed, "verify_only": seed, "verify_repair": seed},
    }


def _make_process_flags(
    *,
    process_valid: bool = True,
    right_for_wrong_reasons: bool = False,
) -> dict[str, Any]:
    return {
        "process_valid": process_valid,
        "outcome_correct": True,
        "right_for_wrong_reasons": right_for_wrong_reasons,
        "defects": (
            [{"kind": "outcome_correct_process_invalid", "detail": "rfwr", "step_id": None, "evidence": {}}]
            if right_for_wrong_reasons
            else []
        ),
        "process_label": "clean" if process_valid else "right_answer_wrong_process",
        "run_date": "20260413",
    }


def _make_case_result(
    idx: int,
    *,
    official_passed: bool = True,
    pbt_accepted: bool = True,
    spec_accepted: bool = True,
    process_accepted: bool = True,
    right_for_wrong_reasons: bool = False,
    repair_accepted: bool = True,
) -> dict[str, Any]:
    """Build a minimal per-case result matching the Exp 261 / Exp 250 schema."""
    pf = _make_process_flags(
        process_valid=process_accepted,
        right_for_wrong_reasons=right_for_wrong_reasons,
    )
    return {
        "case_id": f"humaneval-{idx}",
        "dataset_idx": idx,
        "task_id": f"HumanEval/{idx}",
        "entry_point": f"fn_{idx}",
        "baseline": {"official_passed": official_passed, "body": "    pass", "candidate_code": "def fn(x): pass"},
        "official_tests_verify_only": {"accepted": official_passed},
        "pbt_verify_only": {
            "accepted": pbt_accepted,
            "harness_passing_rejected_by_pbt": official_passed and not pbt_accepted,
        },
        "spec_aware_verify_only": {
            "accepted": spec_accepted,
            "harness_passing_rejected_by_specs": pbt_accepted and not spec_accepted,
        },
        "process_aware_verify_only": {
            "accepted": process_accepted,
            "right_for_wrong_reasons": right_for_wrong_reasons,
        },
        "verify_repair": {
            "accepted": repair_accepted,
            "official_passed": official_passed,
            "repaired": repair_accepted and not process_accepted,
            "n_repairs": 0 if process_accepted else 1,
            "final_body": "    return x + 1",
            "final_code": "def fn(x): return x + 1",
        },
        "process_flags": {
            "baseline": dict(pf),
            "history": [dict(pf)],
            "final": dict(pf),
        },
        "history": [],
    }


# ---------------------------------------------------------------------------
# 1. Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    """Verify that path helpers return sensible defaults."""

    # REQ-CODE-030 (path discipline)

    def test_default_output_path_is_exp261(self) -> None:
        path = _mod.default_output_path()
        assert path.name == "experiment_261_results.json"

    def test_default_checkpoint_dir_is_exp261(self) -> None:
        path = _mod.default_checkpoint_dir()
        assert "experiment_261" in str(path)

    def test_resolve_path_absolute_passthrough(self, tmp_path: Path) -> None:
        result = _mod.resolve_path(tmp_path)
        assert result == tmp_path

    def test_utc_now_format(self) -> None:
        ts = _mod.utc_now()
        assert len(ts) == 20
        assert ts.endswith("Z")
        assert "T" in ts


# ---------------------------------------------------------------------------
# 2. Cohort loading from Exp 226 reference artifact
# ---------------------------------------------------------------------------


class TestCohortLoading:
    """Verify cohort extraction from the real Exp 226 artifact."""

    # REQ-CODE-028, SCENARIO-CODE-026

    def test_load_cohort_from_exp226(self) -> None:
        ref = REPO_ROOT / "results" / "experiment_226_results.json"
        if not ref.exists():
            pytest.skip("Exp 226 artifact not present")
        cases, meta = _mod.load_full_cohort(ref)
        assert len(cases) == 164
        assert meta["source_experiment"] == 226
        assert meta["case_count"] == 164

    def test_load_cohort_cases_have_required_fields(self) -> None:
        ref = REPO_ROOT / "results" / "experiment_226_results.json"
        if not ref.exists():
            pytest.skip("Exp 226 artifact not present")
        cases, _ = _mod.load_full_cohort(ref)
        for case in cases:
            assert "case_id" in case
            assert "prompt" in case
            assert "test" in case
            assert "entry_point" in case
            assert "prompt_seeds" in case
            seeds = case["prompt_seeds"]
            assert seeds["baseline"] == seeds["verify_only"] == seeds["verify_repair"]

    def test_load_cohort_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _mod.load_full_cohort("/nonexistent/path.json")


# ---------------------------------------------------------------------------
# 3. Checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpointHelpers:
    """Verify load/save checkpoint round-trips (SCENARIO-CODE-028)."""

    # REQ-CODE-030, SCENARIO-CODE-028

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "ckpt.json"
        case_ids = ["humaneval-0", "humaneval-1", "humaneval-2"]
        payload = {
            "case_ids": case_ids,
            "results_by_case": {"humaneval-0": {"case_id": "humaneval-0"}},
        }
        _mod.save_checkpoint(ckpt_path, payload)
        loaded = _mod.load_checkpoint(ckpt_path, case_ids)
        assert loaded["case_ids"] == case_ids
        assert "humaneval-0" in loaded["results_by_case"]

    def test_load_nonexistent_checkpoint_returns_fresh(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "missing.json"
        case_ids = ["humaneval-0"]
        result = _mod.load_checkpoint(ckpt_path, case_ids)
        assert result["results_by_case"] == {}
        assert result["case_ids"] == case_ids

    def test_load_checkpoint_rejects_cohort_mismatch(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "ckpt.json"
        _mod.save_checkpoint(
            ckpt_path,
            {"case_ids": ["humaneval-0"], "results_by_case": {}},
        )
        fresh = _mod.load_checkpoint(ckpt_path, ["humaneval-99"])
        assert fresh["results_by_case"] == {}

    def test_checkpoint_written_atomically(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "atomic.json"
        _mod.save_checkpoint(
            ckpt_path,
            {"case_ids": ["humaneval-0"], "results_by_case": {}},
        )
        assert ckpt_path.exists()
        assert not ckpt_path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# 4. GPU harness integration (run_mode checkpointing and throughput)
# ---------------------------------------------------------------------------


class TestGPUHarnessIntegration:
    """Verify that DualGPUBenchmarkHarness.run_mode is wired correctly.

    Spec: REQ-VERIFY-041, SCENARIO-VERIFY-042
    """

    def test_run_mode_produces_one_result_per_case(self, tmp_path: Path) -> None:
        from scripts.experiment_258_dual_gpu_harness import DualGPUBenchmarkHarness  # type: ignore

        harness = DualGPUBenchmarkHarness()
        cases = [_make_case(i) for i in range(5)]

        def execute(case: dict[str, Any]) -> dict[str, Any]:
            return {"case_id": case["case_id"], "dummy": True}

        results = harness.run_mode(
            benchmark="humaneval_qwen_full",
            model_name="Qwen3.5-0.8B",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        assert len(results) == 5
        for r in results:
            assert "case_id" in r
            assert r["dummy"] is True

    def test_run_mode_resumes_from_checkpoint(self, tmp_path: Path) -> None:
        from scripts.experiment_258_dual_gpu_harness import DualGPUBenchmarkHarness  # type: ignore

        harness = DualGPUBenchmarkHarness()
        cases = [_make_case(i) for i in range(3)]

        call_count = {"n": 0}

        def execute(case: dict[str, Any]) -> dict[str, Any]:
            call_count["n"] += 1
            return {"case_id": case["case_id"], "done": True}

        # First run: processes all 3.
        harness.run_mode(
            benchmark="humaneval_qwen_full",
            model_name="Qwen3.5-0.8B",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        first_count = call_count["n"]

        # Second run on same checkpoint_dir: should resume, no extra calls.
        harness2 = DualGPUBenchmarkHarness()
        harness2.run_mode(
            benchmark="humaneval_qwen_full",
            model_name="Qwen3.5-0.8B",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        assert call_count["n"] == first_count  # No re-execution.

    def test_run_mode_tracks_throughput(self, tmp_path: Path) -> None:
        from scripts.experiment_258_dual_gpu_harness import DualGPUBenchmarkHarness  # type: ignore

        harness = DualGPUBenchmarkHarness()
        cases = [_make_case(i) for i in range(4)]

        def execute(case: dict[str, Any]) -> dict[str, Any]:
            return {"case_id": case["case_id"]}

        harness.run_mode(
            benchmark="humaneval_qwen_full",
            model_name="Qwen3.5-0.8B",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        report = harness.throughput.report()
        assert "Qwen3.5-0.8B" in report["per_model"]
        assert report["per_model"]["Qwen3.5-0.8B"]["total_cases"] == 4


# ---------------------------------------------------------------------------
# 5. Stage summary and process integrity stats
# ---------------------------------------------------------------------------


class TestStageSummary:
    """Verify _stage_flags, summarize_model_results, _process_integrity_stats."""

    # REQ-CODE-029, REQ-VERIFY-062, SCENARIO-CODE-027, SCENARIO-VERIFY-066

    def test_stage_flags_all_accepted(self) -> None:
        case = _make_case_result(0)
        flags = _mod._stage_flags(case)
        assert flags["baseline"] is True
        assert flags["official_tests_verify_only"] is True
        assert flags["pbt_verify_only"] is True
        assert flags["spec_aware_verify_only"] is True
        assert flags["process_aware_verify_only"] is True
        assert flags["verify_repair"] is True

    def test_stage_flags_all_rejected(self) -> None:
        case = _make_case_result(
            0,
            official_passed=False,
            pbt_accepted=False,
            spec_accepted=False,
            process_accepted=False,
            repair_accepted=False,
        )
        flags = _mod._stage_flags(case)
        for flag in flags.values():
            assert flag is False

    def test_summarize_model_results_empty(self) -> None:
        result = _mod.summarize_model_results([], n_bootstrap=100, seed=0)
        assert "stages" in result
        assert "process_integrity" in result
        for stage in ("baseline", "official_tests_verify_only", "pbt_verify_only",
                      "spec_aware_verify_only", "process_aware_verify_only", "verify_repair"):
            assert stage in result["stages"]

    def test_summarize_model_results_pass_at_1(self) -> None:
        cases = [_make_case_result(i, official_passed=True) for i in range(4)]
        result = _mod.summarize_model_results(cases, n_bootstrap=100, seed=42)
        assert result["stages"]["baseline"]["accepted_pass_at_1"] == pytest.approx(1.0)

    def test_process_integrity_stats_rfwr_count(self) -> None:
        cases = [
            _make_case_result(0, right_for_wrong_reasons=True),
            _make_case_result(1, right_for_wrong_reasons=False),
            _make_case_result(2, right_for_wrong_reasons=True),
        ]
        stats = _mod._process_integrity_stats(cases)
        assert stats["right_for_wrong_reasons_count"] == 2
        assert stats["total_cases"] == 3


# ---------------------------------------------------------------------------
# 6. Cross-model comparison block (Exp 261 Qwen vs Exp 226 Gemma)
# ---------------------------------------------------------------------------


class TestCrossModelComparisonBlock:
    """Verify the Exp 226 Gemma vs Exp 261 Qwen comparison block structure.

    Spec: REQ-CODE-029, SCENARIO-CODE-027
    """

    def _make_exp226_case(
        self,
        idx: int,
        *,
        baseline_passed: bool = True,
        verify_only_accepted: bool = True,
        verify_repair_passed: bool = True,
    ) -> dict[str, Any]:
        """Minimal Exp 226 per-problem-results row."""
        return {
            "case_id": f"humaneval-{idx}",
            "dataset_idx": idx,
            "task_id": f"HumanEval/{idx}",
            "baseline": {"passed": baseline_passed},
            "verify_only": {"accepted": verify_only_accepted},
            "verify_repair": {"passed": verify_repair_passed},
        }

    def test_build_cross_model_comparison_has_required_keys(self) -> None:
        qwen_cases = [_make_case_result(i) for i in range(5)]
        gemma_cases = [self._make_exp226_case(i) for i in range(5)]
        block = _mod.build_cross_model_comparison(
            qwen_cases=qwen_cases,
            gemma_exp226_cases=gemma_cases,
            n_bootstrap=100,
            seed=0,
            repair_budget=3,
        )
        assert "paired_case_count" in block
        assert "stage_deltas" in block
        assert "stage_outcomes" in block
        assert "schema_mapping_note" in block
        assert block["paired_case_count"] == 5

    def test_build_cross_model_comparison_stage_outcome_sums(self) -> None:
        qwen_cases = [_make_case_result(i, official_passed=(i % 2 == 0)) for i in range(4)]
        gemma_cases = [self._make_exp226_case(i, baseline_passed=(i % 2 == 0)) for i in range(4)]
        block = _mod.build_cross_model_comparison(
            qwen_cases=qwen_cases,
            gemma_exp226_cases=gemma_cases,
            n_bootstrap=100,
            seed=0,
            repair_budget=3,
        )
        outcomes = block["stage_outcomes"]["baseline"]
        total = outcomes["gemma_only"] + outcomes["qwen_only"] + outcomes["both"] + outcomes["neither"]
        assert total == 4

    def test_build_cross_model_comparison_empty_returns_zero_block(self) -> None:
        block = _mod.build_cross_model_comparison(
            qwen_cases=[],
            gemma_exp226_cases=[],
            n_bootstrap=100,
            seed=0,
            repair_budget=3,
        )
        assert block["paired_case_count"] == 0

    def test_build_cross_model_comparison_schema_mapping_note_present(self) -> None:
        qwen_cases = [_make_case_result(i) for i in range(3)]
        gemma_cases = [self._make_exp226_case(i) for i in range(3)]
        block = _mod.build_cross_model_comparison(
            qwen_cases=qwen_cases,
            gemma_exp226_cases=gemma_cases,
            n_bootstrap=100,
            seed=0,
            repair_budget=3,
        )
        note = block["schema_mapping_note"]
        assert "226" in note or "Gemma" in note or "schema" in note.lower()

    def test_build_cross_model_comparison_paired_only_on_matching_case_ids(self) -> None:
        qwen_cases = [_make_case_result(i) for i in range(5)]
        # Gemma only has 3 of the 5 case IDs.
        gemma_cases = [self._make_exp226_case(i) for i in range(3)]
        block = _mod.build_cross_model_comparison(
            qwen_cases=qwen_cases,
            gemma_exp226_cases=gemma_cases,
            n_bootstrap=100,
            seed=0,
            repair_budget=3,
        )
        assert block["paired_case_count"] == 3


# ---------------------------------------------------------------------------
# 7. Artifact schema validation
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Validate the final artifact payload structure.

    Spec: REQ-CODE-029, SCENARIO-CODE-027
    """

    def _minimal_artifact(self) -> dict[str, Any]:
        cases = [_make_case(i) for i in range(5)]
        case_results = [_make_case_result(i) for i in range(5)]
        model_run = {
            "model_name": "Qwen3.5-0.8B",
            "model_hf_id": "Qwen/Qwen3.5-0.8B",
            "device": "cuda:0",
            "run_status": "complete",
            "completed_case_count": 5,
            "pending_case_count": 0,
            "blockers": [],
            "checkpoint_path": "results/checkpoints/experiment_261/qwen.json",
            "statistics": _mod.summarize_model_results(case_results, n_bootstrap=100, seed=7),
            "per_problem_results": case_results,
        }
        gemma_cases = [
            {
                "case_id": f"humaneval-{i}",
                "dataset_idx": i,
                "task_id": f"HumanEval/{i}",
                "baseline": {"passed": True},
                "verify_only": {"accepted": True},
                "verify_repair": {"passed": True},
            }
            for i in range(5)
        ]
        comparison = _mod.build_cross_model_comparison(
            qwen_cases=case_results,
            gemma_exp226_cases=gemma_cases,
            n_bootstrap=100,
            seed=13,
            repair_budget=3,
        )
        return _mod.build_artifact_payload(
            output_path=Path("results/experiment_261_results.json"),
            cohort=cases,
            cohort_meta={
                "source_artifact": "results/experiment_226_results.json",
                "source_experiment": 226,
                "case_count": 5,
            },
            model_run=model_run,
            comparison=comparison,
            blockers=[],
            started_at="2026-04-13T22:00:00Z",
            finished_at="2026-04-13T23:00:00Z",
            runtime_seconds=3600.0,
            checkpoint_dir=Path("results/checkpoints/experiment_261"),
            max_repairs=3,
            pbt_max_examples=64,
            bootstrap_samples=100,
            run_status="complete",
        )

    def test_artifact_top_level_keys(self) -> None:
        artifact = self._minimal_artifact()
        for key in ("experiment", "benchmark", "run_date", "schema", "metadata",
                    "cohort", "model_run", "comparison", "blockers", "run_status"):
            assert key in artifact, f"Missing key: {key}"

    def test_artifact_experiment_number(self) -> None:
        artifact = self._minimal_artifact()
        assert artifact["experiment"] == 261

    def test_artifact_benchmark_name(self) -> None:
        artifact = self._minimal_artifact()
        assert artifact["benchmark"] == "humaneval_qwen_full_process"

    def test_artifact_schema_field(self) -> None:
        artifact = self._minimal_artifact()
        assert "artifact" in artifact["schema"]
        assert "261" in artifact["schema"]["artifact"] or "qwen" in artifact["schema"]["artifact"].lower()

    def test_artifact_cohort_case_count(self) -> None:
        artifact = self._minimal_artifact()
        assert artifact["cohort"]["case_count"] == 5

    def test_artifact_model_run_has_statistics(self) -> None:
        artifact = self._minimal_artifact()
        mr = artifact["model_run"]
        assert "statistics" in mr
        stats = mr["statistics"]
        assert "stages" in stats
        assert "process_integrity" in stats

    def test_artifact_comparison_has_paired_case_count(self) -> None:
        artifact = self._minimal_artifact()
        assert "paired_case_count" in artifact["comparison"]

    def test_artifact_run_status_valid(self) -> None:
        artifact = self._minimal_artifact()
        assert artifact["run_status"] in ("complete", "partial", "blocked")

    def test_artifact_is_json_serializable(self) -> None:
        artifact = self._minimal_artifact()
        dumped = json.dumps(artifact)
        loaded = json.loads(dumped)
        assert loaded["experiment"] == 261

    def test_artifact_metadata_has_timing(self) -> None:
        artifact = self._minimal_artifact()
        meta = artifact["metadata"]
        assert "started_at" in meta
        assert "finished_at" in meta
        assert "runtime_seconds" in meta

    def test_artifact_model_run_process_integrity_stats(self) -> None:
        artifact = self._minimal_artifact()
        pi = artifact["model_run"]["statistics"]["process_integrity"]
        assert "right_for_wrong_reasons_count" in pi
        assert "defect_kind_counts" in pi
        assert "total_cases" in pi


# ---------------------------------------------------------------------------
# 8. Checkpoint interval enforcement (10-problem granularity)
# ---------------------------------------------------------------------------


class TestCheckpointInterval:
    """Verify that a checkpoint is written at 10-problem granularity."""

    # REQ-CODE-030, SCENARIO-CODE-028

    def test_checkpoint_written_every_10_cases(self, tmp_path: Path) -> None:
        case_ids = [f"humaneval-{i}" for i in range(20)]
        cases = [_make_case(i) for i in range(20)]
        results_by_case: dict[str, Any] = {}
        checkpoint_count = [0]
        ckpt_path = tmp_path / "test.json"
        original_save = _mod.save_checkpoint

        def counting_save(path: Any, payload: Any) -> None:
            checkpoint_count[0] += 1
            original_save(path, payload)

        # Run the benchmark loop manually with patched save.
        _mod.save_checkpoint = counting_save
        try:
            _mod.run_benchmark(
                cases,
                model=None,
                tokenizer=None,
                device_str="cpu",
                checkpoint_path=ckpt_path,
                checkpoint_interval=10,
                max_repairs=0,
                pbt_max_examples=1,
                max_new_tokens=16,
                _execute_case_override=lambda case, **kw: _make_case_result(case["dataset_idx"]),
            )
        except Exception:
            pass
        finally:
            _mod.save_checkpoint = original_save

        # At minimum 2 checkpoints should be written (one at 10, one at 20 or end).
        assert checkpoint_count[0] >= 2
