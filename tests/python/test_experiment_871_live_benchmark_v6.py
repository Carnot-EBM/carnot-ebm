"""Tests for Experiment 871 — Live Benchmark v6: Single-Model Cascade.

Traces to:
    REQ-BENCH-015  (live cascade benchmark)
    SCENARIO-BENCH-034 (DualGPU cascade with single inference model)

Why these tests exist:
    The experiment has four critical paths that must be covered by CPU-only,
    mock-backed tests so CI never requires real GPU hardware:

    1. Gate check — Exp 856 artifact absent or dual_gpu_deployed!=True yields blocked.
    2. GPU-unhealthy path — setup_gpu() reporting unhealthy yields blocked artifact.
    3. _run_cascade() — correct tier_exited_at and repaired logic for all branches.
    4. _compute_metrics() — signed_improvement and honest_verdict computed correctly.
    5. Problem corpus — 50 GSM8K problems with correct schema.
    6. Happy-path main() — all REQUIRED_RESULT_FIELDS present in written artifact.

All GPU calls, ThreeTierPipeline, and VerifyRepairPipeline are mocked so the
entire suite runs in < 5 s on any CPU-only CI machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "scripts"))

import scripts.experiment_871_live_benchmark_v6 as exp871  # noqa: E402
from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_gate_artifact(results_dir: Path, *, dual_gpu_deployed: bool) -> None:
    """Write a minimal Exp 856 gate artifact into results_dir."""
    p = results_dir / "experiment_856_dualgpu_production.json"
    p.write_text(json.dumps({"dual_gpu_deployed": dual_gpu_deployed, "status": "success"}))


def _mock_tmpl_factory(tmp_path: Path, deliverable_name: str) -> MagicMock:
    """Return a MagicMock ExperimentTemplate that writes the deliverable when build_result is called.

    Why set assert_deliverable_written explicitly: MagicMock intercepts calls to
    any attribute starting with 'assert_' and treats them as built-in mock
    assertion helpers, raising AttributeError if the call signature doesn't match.
    Setting the attribute explicitly as a plain MagicMock bypasses that interception.
    """

    def _build_result(payload: dict, **kwargs: Any) -> dict:
        base = {
            "status": kwargs.get("status", "success"),
            "experiment": 871,
            "title": "Live Benchmark v6",
            "schema": [],
            "run_date": "20260425",
            "started_at": "2026-04-25T00:00:00Z",
            "finished_at": "2026-04-25T00:00:01Z",
            "duration_s": 1.0,
        }
        base.update(payload)
        return base

    mock = MagicMock()
    mock.build_result.side_effect = _build_result
    mock.setup.return_value = None
    mock.setup_gpu.return_value = {"all_healthy": True, "models": [], "cpu_fallback": False}
    mock.apply_env_autofix.return_value = None
    # Explicitly set to bypass MagicMock's assert_* interception.
    mock.assert_deliverable_written = MagicMock(return_value=None)
    return mock


# ---------------------------------------------------------------------------
# Gate check — missing file
# ---------------------------------------------------------------------------


class TestGateCheckMissingFile:
    """REQ-BENCH-015: gate must block when Exp 856 artifact is absent."""

    def test_blocked_when_gate_file_missing(self, tmp_path: Path) -> None:
        """If results/experiment_856_dualgpu_production.json does not exist, write blocked."""
        (tmp_path / "results").mkdir()
        deliverable = tmp_path / "results" / "experiment_871_live_benchmark_v6.json"

        mock_tmpl = _mock_tmpl_factory(tmp_path, "experiment_871_live_benchmark_v6.json")

        with (
            patch.object(exp871, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_871_live_benchmark_v6.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
        ):
            exp871.main()

        assert deliverable.exists(), "Blocked artifact must be written when gate file absent"
        data = json.loads(deliverable.read_text())
        assert data["status"] == "blocked"
        assert data["honest_verdict"] == "blocked"


# ---------------------------------------------------------------------------
# Gate check — dual_gpu_deployed False
# ---------------------------------------------------------------------------


class TestGateCheckFlagFalse:
    """REQ-BENCH-015: gate must block when dual_gpu_deployed is False."""

    def test_blocked_when_dual_gpu_not_deployed(self, tmp_path: Path) -> None:
        """dual_gpu_deployed=False must yield a blocked artifact."""
        (tmp_path / "results").mkdir()
        _write_gate_artifact(tmp_path / "results", dual_gpu_deployed=False)
        deliverable = tmp_path / "results" / "experiment_871_live_benchmark_v6.json"

        mock_tmpl = _mock_tmpl_factory(tmp_path, "experiment_871_live_benchmark_v6.json")

        with (
            patch.object(exp871, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_871_live_benchmark_v6.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
        ):
            exp871.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert data["status"] == "blocked"


# ---------------------------------------------------------------------------
# GPU unhealthy path
# ---------------------------------------------------------------------------


class TestGPUUnhealthyBlocked:
    """SCENARIO-BENCH-034: GPU unhealthy must yield blocked artifact."""

    def test_blocked_when_gpu_unhealthy(self, tmp_path: Path) -> None:
        """setup_gpu() reporting all_healthy=False must write blocked artifact."""
        (tmp_path / "results").mkdir()
        _write_gate_artifact(tmp_path / "results", dual_gpu_deployed=True)
        deliverable = tmp_path / "results" / "experiment_871_live_benchmark_v6.json"

        mock_tmpl = _mock_tmpl_factory(tmp_path, "experiment_871_live_benchmark_v6.json")
        mock_tmpl.setup_gpu.return_value = {
            "all_healthy": False,
            "models": [],
            "cpu_fallback": True,
            "error": "no GPU",
        }

        with (
            patch.object(exp871, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_871_live_benchmark_v6.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1", "CARNOT_DUAL_GPU": "1"}),
        ):
            exp871.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert data["status"] == "blocked"
        assert data["honest_verdict"] == "blocked"


# ---------------------------------------------------------------------------
# _baseline_answer
# ---------------------------------------------------------------------------


class TestBaselineAnswer:
    """_baseline_answer() must be deterministic."""

    def test_degraded_index_returns_incorrect(self) -> None:
        """REQ-BENCH-015: index 0 (0%10 < 3) returns INCORRECT."""
        p = {"id": "gsm8k_0", "question": "Q", "answer": "8"}
        assert exp871._baseline_answer(p) == "INCORRECT"

    def test_non_degraded_index_returns_reference(self) -> None:
        """Index 5 (5%10 >= 3) returns the reference answer."""
        p = {"id": "gsm8k_5", "question": "Q", "answer": "40"}
        assert exp871._baseline_answer(p) == "40"

    def test_index_2_degraded(self) -> None:
        """Index 2 is degraded (2%10 < 3) → INCORRECT."""
        p = {"id": "gsm8k_2", "question": "Q", "answer": "$12"}
        assert exp871._baseline_answer(p) == "INCORRECT"

    def test_index_3_not_degraded(self) -> None:
        """Index 3 is NOT degraded (3%10 == 3, not < 3) → reference."""
        p = {"id": "gsm8k_3", "question": "Q", "answer": "4"}
        assert exp871._baseline_answer(p) == "4"


# ---------------------------------------------------------------------------
# _run_cascade — simulation path
# ---------------------------------------------------------------------------


class TestRunCascadeSimulation:
    """_run_cascade() simulation path (inference_mode != live_gpu)."""

    def test_returns_required_keys(self) -> None:
        """SCENARIO-BENCH-034: result must contain all documented keys."""
        p = {"id": "gsm8k_5", "question": "Q", "answer": "40"}
        result = exp871._run_cascade(p, None, None, "simulation_fallback")
        for key in (
            "id",
            "tier_exited_at",
            "was_correct_baseline",
            "was_correct_repaired",
            "repaired",
            "latency_ms",
        ):
            assert key in result, f"Missing key: {key}"

    def test_no_crash_on_none_pipelines(self) -> None:
        """Must not raise when both pipelines are None."""
        p = {"id": "gsm8k_5", "question": "Q", "answer": "40"}
        exp871._run_cascade(p, None, None, "simulation_fallback")

    def test_tier_exited_at_none_in_simulation(self) -> None:
        """Simulation path never sets tier_exited_at (no real tiers)."""
        p = {"id": "gsm8k_5", "question": "Q", "answer": "40"}
        result = exp871._run_cascade(p, None, None, "simulation_fallback")
        assert result["tier_exited_at"] is None


# ---------------------------------------------------------------------------
# _run_cascade — live GPU path with mocked ThreeTierPipeline
# ---------------------------------------------------------------------------


class TestRunCascadeLiveGPU:
    """SCENARIO-BENCH-034: live_gpu path with mocked pipelines."""

    def test_tier_exited_at_set_when_cascade_clears(self) -> None:
        """When ThreeTierPipeline returns tier_cleared=1, tier_exited_at must be 1."""
        p = {"id": "gsm8k_5", "question": "Q", "answer": "40"}
        mock_three_tier = MagicMock()
        mock_result = MagicMock()
        mock_result.tier_cleared = 1
        mock_result.verified = True
        mock_three_tier.verify.return_value = mock_result

        result = exp871._run_cascade(p, mock_three_tier, None, "live_gpu")
        assert result["tier_exited_at"] == 1
        assert result["was_correct_repaired"] is True
        assert result["repaired"] is False

    def test_tier3_invoked_when_cascade_does_not_clear(self) -> None:
        """When tier_cleared is not set, VerifyRepairPipeline must be invoked.

        Use gsm8k_0 (idx=0, baseline=INCORRECT) so repaired_ans ("8") differs
        from baseline_ans ("INCORRECT"), triggering repaired=True.
        """
        p = {"id": "gsm8k_0", "question": "Janet has 3 apples.", "answer": "8"}

        mock_three_tier = MagicMock()
        mock_tt_result = MagicMock()
        mock_tt_result.tier_cleared = None  # not an int 0-2 → Tier 3 needed
        mock_three_tier.verify.return_value = mock_tt_result

        mock_vr = MagicMock()
        mock_repair = MagicMock()
        # "8" != "INCORRECT" (the baseline for idx=0), so repaired=True fires.
        mock_repair.repaired_response = "8"
        mock_vr.verify_and_repair.return_value = mock_repair

        result = exp871._run_cascade(p, mock_three_tier, mock_vr, "live_gpu")
        assert result["repaired"] is True
        assert result["tier_exited_at"] is None

    def test_cascade_exception_does_not_crash(self) -> None:
        """If ThreeTierPipeline.verify() raises, result is returned without crash."""
        p = {"id": "gsm8k_5", "question": "Q", "answer": "40"}
        mock_three_tier = MagicMock()
        mock_three_tier.verify.side_effect = RuntimeError("GPU OOM")

        result = exp871._run_cascade(p, mock_three_tier, None, "live_gpu")
        assert isinstance(result, dict)
        assert "was_correct_repaired" in result

    def test_repair_exception_does_not_crash(self) -> None:
        """If VerifyRepairPipeline.verify_and_repair() raises, result is still returned."""
        p = {"id": "gsm8k_5", "question": "Q", "answer": "40"}
        mock_three_tier = MagicMock()
        mock_result = MagicMock()
        mock_result.tier_cleared = None
        mock_three_tier.verify.return_value = mock_result

        mock_vr = MagicMock()
        mock_vr.verify_and_repair.side_effect = RuntimeError("repair failed")

        result = exp871._run_cascade(p, mock_three_tier, mock_vr, "live_gpu")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _compute_metrics — signed_improvement and honest_verdict
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """REQ-BENCH-015: metrics must be computed from per_question results correctly."""

    def _make_row(
        self,
        *,
        baseline_correct: bool,
        repaired_correct: bool,
        tier_exited_at: int | None = None,
        repaired: bool = False,
    ) -> dict:
        return {
            "id": "gsm8k_0",
            "tier_exited_at": tier_exited_at,
            "was_correct_baseline": baseline_correct,
            "was_correct_repaired": repaired_correct,
            "repaired": repaired,
            "latency_ms": 10.0,
        }

    def test_signed_improvement_positive(self) -> None:
        """SCENARIO-BENCH-034: repair fixing answers → positive signed_improvement."""
        rows = [
            self._make_row(baseline_correct=False, repaired_correct=True, repaired=True),
            self._make_row(baseline_correct=True, repaired_correct=True),
        ]
        metrics = exp871._compute_metrics(rows, "simulation_fallback")
        assert metrics["signed_improvement"] > 0

    def test_signed_improvement_zero_when_no_repair(self) -> None:
        """No repairs → signed_improvement == 0."""
        rows = [
            self._make_row(baseline_correct=True, repaired_correct=True),
            self._make_row(baseline_correct=False, repaired_correct=False),
        ]
        metrics = exp871._compute_metrics(rows, "simulation_fallback")
        assert metrics["signed_improvement"] == 0.0

    def test_honest_verdict_simulation_fallback(self) -> None:
        """inference_mode != live_gpu always → simulation_fallback verdict."""
        rows = [self._make_row(baseline_correct=True, repaired_correct=True)]
        metrics = exp871._compute_metrics(rows, "simulation_fallback")
        assert metrics["honest_verdict"] == "simulation_fallback"

    def test_honest_verdict_positive_improvement(self) -> None:
        """live_gpu + signed_improvement > 0 → positive_improvement."""
        rows = [
            self._make_row(baseline_correct=False, repaired_correct=True, repaired=True),
        ]
        metrics = exp871._compute_metrics(rows, "live_gpu")
        assert metrics["honest_verdict"] == "positive_improvement"

    def test_honest_verdict_live_no_improvement(self) -> None:
        """live_gpu + signed_improvement <= 0 → live_no_improvement."""
        rows = [self._make_row(baseline_correct=True, repaired_correct=True)]
        metrics = exp871._compute_metrics(rows, "live_gpu")
        assert metrics["honest_verdict"] == "live_no_improvement"

    def test_honest_verdict_cascade_running_when_4_tiers(self) -> None:
        """SCENARIO-BENCH-034: cascade_tiers_active >= 4 → cascade_running."""
        rows = [
            self._make_row(baseline_correct=True, repaired_correct=True, tier_exited_at=0),
            self._make_row(baseline_correct=True, repaired_correct=True, tier_exited_at=1),
            self._make_row(baseline_correct=True, repaired_correct=True, tier_exited_at=2),
            self._make_row(baseline_correct=False, repaired_correct=True, repaired=True),  # Tier 3
        ]
        metrics = exp871._compute_metrics(rows, "live_gpu")
        assert metrics["cascade_tiers_active"] == 4
        assert metrics["honest_verdict"] == "cascade_running"

    def test_cascade_skip_rate(self) -> None:
        """Fraction with tier_exited_at not None = cascade_skip_rate."""
        rows = [
            self._make_row(baseline_correct=True, repaired_correct=True, tier_exited_at=0),
            self._make_row(baseline_correct=True, repaired_correct=True, tier_exited_at=1),
            self._make_row(
                baseline_correct=False, repaired_correct=True, repaired=True
            ),  # Tier 3: tier_exited_at is None
            self._make_row(baseline_correct=False, repaired_correct=False),
        ]
        metrics = exp871._compute_metrics(rows, "simulation_fallback")
        # 2 out of 4 exited early
        assert metrics["cascade_skip_rate"] == 0.5

    def test_empty_rows_returns_blocked(self) -> None:
        """Empty per_question list → honest_verdict='blocked'."""
        metrics = exp871._compute_metrics([], "live_gpu")
        assert metrics["honest_verdict"] == "blocked"


# ---------------------------------------------------------------------------
# Problem corpus
# ---------------------------------------------------------------------------


class TestProblemCorpus:
    """Problem list must meet schema and count requirements."""

    def test_gsm8k_count_exactly_50(self) -> None:
        """REQ-BENCH-015: exactly 50 GSM8K problems must be defined."""
        assert len(exp871._GSM8K_PROBLEMS) == exp871.N_GSM8K

    def test_problem_schema(self) -> None:
        """Each problem must have id, question, answer keys."""
        for p in exp871._GSM8K_PROBLEMS:
            assert "id" in p, f"Missing 'id' in {p}"
            assert "question" in p, f"Missing 'question' in {p}"
            assert "answer" in p, f"Missing 'answer' in {p}"

    def test_problem_ids_unique(self) -> None:
        """All problem IDs must be unique."""
        ids = [p["id"] for p in exp871._GSM8K_PROBLEMS]
        assert len(ids) == len(set(ids)), "Duplicate problem IDs found"


# ---------------------------------------------------------------------------
# Happy-path main() with mocked GPU
# ---------------------------------------------------------------------------


class TestMainHappyPath:
    """SCENARIO-BENCH-034: successful run writes artifact with all required fields."""

    def test_artifact_has_required_fields(self, tmp_path: Path) -> None:
        """All REQUIRED_RESULT_FIELDS must be present in the written artifact."""
        (tmp_path / "results").mkdir()
        _write_gate_artifact(tmp_path / "results", dual_gpu_deployed=True)
        deliverable = tmp_path / "results" / "experiment_871_live_benchmark_v6.json"

        mock_tmpl = _mock_tmpl_factory(tmp_path, "experiment_871_live_benchmark_v6.json")

        with (
            patch.object(exp871, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_871_live_benchmark_v6.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0", "CARNOT_DUAL_GPU": "1"}),
        ):
            exp871.main()

        assert deliverable.exists(), "Deliverable must be written on happy path"
        data = json.loads(deliverable.read_text())
        missing = [f for f in REQUIRED_RESULT_FIELDS if f not in data]
        assert not missing, f"Missing required fields: {missing}"

    def test_artifact_honest_verdict_present(self, tmp_path: Path) -> None:
        """honest_verdict must be one of the defined verdict strings."""
        (tmp_path / "results").mkdir()
        _write_gate_artifact(tmp_path / "results", dual_gpu_deployed=True)
        deliverable = tmp_path / "results" / "experiment_871_live_benchmark_v6.json"

        mock_tmpl = _mock_tmpl_factory(tmp_path, "experiment_871_live_benchmark_v6.json")

        with (
            patch.object(exp871, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_871_live_benchmark_v6.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0", "CARNOT_DUAL_GPU": "1"}),
        ):
            exp871.main()

        data = json.loads(deliverable.read_text())
        valid_verdicts = {
            "positive_improvement",
            "live_no_improvement",
            "cascade_running",
            "simulation_fallback",
            "blocked",
        }
        assert data.get("honest_verdict") in valid_verdicts, (
            f"unexpected verdict: {data.get('honest_verdict')}"
        )

    def test_artifact_experiment_id(self, tmp_path: Path) -> None:
        """experiment field must be 871."""
        (tmp_path / "results").mkdir()
        _write_gate_artifact(tmp_path / "results", dual_gpu_deployed=True)
        deliverable = tmp_path / "results" / "experiment_871_live_benchmark_v6.json"

        mock_tmpl = _mock_tmpl_factory(tmp_path, "experiment_871_live_benchmark_v6.json")

        with (
            patch.object(exp871, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_871_live_benchmark_v6.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0", "CARNOT_DUAL_GPU": "1"}),
        ):
            exp871.main()

        data = json.loads(deliverable.read_text())
        assert data["experiment"] == 871

    def test_deliverable_has_cascade_metrics(self, tmp_path: Path) -> None:
        """Artifact must contain cascade-specific fields."""
        (tmp_path / "results").mkdir()
        _write_gate_artifact(tmp_path / "results", dual_gpu_deployed=True)
        deliverable = tmp_path / "results" / "experiment_871_live_benchmark_v6.json"

        mock_tmpl = _mock_tmpl_factory(tmp_path, "experiment_871_live_benchmark_v6.json")

        with (
            patch.object(exp871, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_871_live_benchmark_v6.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0", "CARNOT_DUAL_GPU": "1"}),
        ):
            exp871.main()

        data = json.loads(deliverable.read_text())
        for field in (
            "cascade_skip_rate",
            "cascade_tiers_active",
            "signed_improvement",
            "baseline_accuracy",
            "carnot_accuracy",
            "inference_mode",
        ):
            assert field in data, f"Missing cascade field: {field}"


# ---------------------------------------------------------------------------
# Existing deliverable schema check
# ---------------------------------------------------------------------------


class TestExistingDeliverable:
    """If the deliverable exists on disk, it must pass schema validation."""

    def test_deliverable_required_fields(self) -> None:
        """REQ-BENCH-015: on-disk artifact must contain all REQUIRED_RESULT_FIELDS."""
        deliverable = _REPO / "results" / "experiment_871_live_benchmark_v6.json"
        if not deliverable.exists():
            pytest.skip("Deliverable not yet written")
        data = json.loads(deliverable.read_text())
        missing = [f for f in REQUIRED_RESULT_FIELDS if f not in data]
        assert not missing, f"Missing required fields: {missing}"

    def test_deliverable_honest_verdict(self) -> None:
        """honest_verdict must be one of the defined states."""
        deliverable = _REPO / "results" / "experiment_871_live_benchmark_v6.json"
        if not deliverable.exists():
            pytest.skip("Deliverable not yet written")
        data = json.loads(deliverable.read_text())
        valid = {
            "positive_improvement",
            "live_no_improvement",
            "cascade_running",
            "simulation_fallback",
            "blocked",
        }
        assert data.get("honest_verdict") in valid
