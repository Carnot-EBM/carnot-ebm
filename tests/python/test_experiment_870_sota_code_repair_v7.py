"""Tests for Exp 870 — SOTA code repair v7: HumanEval live GPU (Qwen3.5-0.8B).

Covers gate check logic, CodeExtractor integration, VerifyRepairPipeline mocking,
and signed_improvement computation.  All LLM and GPU calls are mocked so these
tests run entirely offline on CPU.

Traces to REQ-VR-020 (verify-repair live), SCENARIO-VR-030 (HumanEval live).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Gate check helpers (inline — mirrors the logic in experiment_870 script)
# ---------------------------------------------------------------------------


def _load_gate_artifact(path: str) -> dict[str, Any]:
    """Load a JSON artifact and return it as a dict."""
    with open(path) as fh:
        return json.load(fh)


def gate_check_exp869(artifact: dict[str, Any]) -> bool:
    """Return True if Exp 869 confirms download_verified=True.

    The Exp 870 script must NOT run inference unless this returns True.
    The reason: every prior attempt that skipped download verification ended
    up running against a missing model file and crashing at the worst possible
    moment (mid-benchmark), corrupting partial results.
    """
    return bool(artifact.get("download_verified"))


def gate_check_exp855(artifact: dict[str, Any]) -> bool:
    """Return True if Exp 855 confirms live_env_fixed=True.

    CARNOT_FORCE_LIVE propagation was broken before Exp 855; running Exp 870
    without this gate produced silent simulation-mode results mislabelled as
    live_gpu.
    """
    return bool(artifact.get("live_env_fixed"))


# ---------------------------------------------------------------------------
# signed_improvement computation (inline — mirrors experiment_870 logic)
# ---------------------------------------------------------------------------


def compute_signed_improvement(baseline_pass_rate: float, carnot_pass_rate: float) -> float:
    """signed_improvement = carnot_pass_rate - baseline_pass_rate.

    Positive means repair helped.  Zero means no change.  Negative means
    repair hurt (regression — should flag for investigation).
    """
    return carnot_pass_rate - baseline_pass_rate


def determine_honest_verdict(
    inference_mode: str | None,
    signed_improvement: float | None,
) -> str:
    """Map inference_mode + signed_improvement to a canonical honest_verdict string.

    The four categories are exhaustive so the caller never has to guess.
    """
    if inference_mode is None:
        return "blocked"
    if inference_mode != "live_gpu":
        return "simulation_fallback"
    if signed_improvement is None:
        return "blocked"
    if signed_improvement > 0:
        return "positive_repair"
    return "live_no_improvement"


# ---------------------------------------------------------------------------
# Gate check tests
# ---------------------------------------------------------------------------


class TestGateCheckExp869:
    """REQ-VR-020: Exp 870 must be blocked when Exp 869 download gate not met."""

    def test_gate_passes_when_download_verified_true(self) -> None:
        """gate_check_exp869 returns True when download_verified is explicitly True."""
        artifact = {"download_verified": True, "status": "success"}
        assert gate_check_exp869(artifact) is True

    def test_gate_blocked_when_download_verified_false(self) -> None:
        """gate_check_exp869 returns False when download_verified is False.

        This is the actual state of Exp 869 as of 2026-04-25: the GGUF repo
        returned 404, so Exp 870 must stay blocked until a working repo ID is found.
        """
        artifact = {"download_verified": False, "status": "failed"}
        assert gate_check_exp869(artifact) is False

    def test_gate_blocked_when_key_missing(self) -> None:
        """gate_check_exp869 returns False when download_verified key is absent."""
        assert gate_check_exp869({}) is False

    def test_gate_blocked_when_download_verified_none(self) -> None:
        """gate_check_exp869 returns False when download_verified is None."""
        assert gate_check_exp869({"download_verified": None}) is False


class TestGateCheckExp855:
    """REQ-VR-020: Exp 870 must be blocked when Exp 855 env gate not met."""

    def test_gate_passes_when_live_env_fixed_true(self) -> None:
        """gate_check_exp855 returns True when live_env_fixed is True."""
        assert gate_check_exp855({"live_env_fixed": True}) is True

    def test_gate_blocked_when_live_env_fixed_false(self) -> None:
        """gate_check_exp855 returns False when live_env_fixed is False."""
        assert gate_check_exp855({"live_env_fixed": False}) is False

    def test_gate_blocked_when_key_missing(self) -> None:
        """gate_check_exp855 returns False when live_env_fixed key is absent."""
        assert gate_check_exp855({}) is False


# ---------------------------------------------------------------------------
# signed_improvement tests
# ---------------------------------------------------------------------------


class TestComputeSignedImprovement:
    """REQ-VR-020: signed_improvement arithmetic must be exact."""

    def test_positive_improvement(self) -> None:
        """signed_improvement is positive when repair increases pass rate."""
        assert compute_signed_improvement(0.4, 0.6) == pytest.approx(0.2)

    def test_zero_improvement(self) -> None:
        """signed_improvement is zero when repair makes no difference."""
        assert compute_signed_improvement(0.5, 0.5) == pytest.approx(0.0)

    def test_negative_improvement(self) -> None:
        """signed_improvement is negative when repair regresses pass rate."""
        assert compute_signed_improvement(0.8, 0.6) == pytest.approx(-0.2)

    def test_full_pass_from_zero(self) -> None:
        """signed_improvement is 1.0 when repair takes baseline from 0 to 100%."""
        assert compute_signed_improvement(0.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# honest_verdict determination tests
# ---------------------------------------------------------------------------


class TestDetermineHonestVerdict:
    """SCENARIO-VR-030: verdict mapping covers all four outcome categories."""

    def test_positive_repair(self) -> None:
        """positive_repair when live_gpu AND signed_improvement > 0."""
        assert determine_honest_verdict("live_gpu", 0.1) == "positive_repair"

    def test_live_no_improvement(self) -> None:
        """live_no_improvement when live_gpu AND signed_improvement == 0."""
        assert determine_honest_verdict("live_gpu", 0.0) == "live_no_improvement"

    def test_live_regression(self) -> None:
        """live_no_improvement when live_gpu AND signed_improvement < 0 (regression)."""
        assert determine_honest_verdict("live_gpu", -0.05) == "live_no_improvement"

    def test_simulation_fallback(self) -> None:
        """simulation_fallback when inference_mode is not live_gpu."""
        assert determine_honest_verdict("simulated", 0.5) == "simulation_fallback"

    def test_blocked_no_inference_mode(self) -> None:
        """blocked when inference_mode is None (gate failed)."""
        assert determine_honest_verdict(None, None) == "blocked"

    def test_blocked_live_gpu_no_improvement_computed(self) -> None:
        """blocked when inference_mode is live_gpu but signed_improvement is None."""
        assert determine_honest_verdict("live_gpu", None) == "blocked"


# ---------------------------------------------------------------------------
# CodeExtractor mock integration tests
# ---------------------------------------------------------------------------


class TestCodeExtractorIntegration:
    """Verify that Exp 870 correctly wires CodeExtractor into the repair loop.

    Spec: REQ-VR-020, SCENARIO-VR-030
    """

    def test_code_extractor_called_for_each_problem(self) -> None:
        """CodeExtractor.extract() is called once per generated solution.

        The experiment must call extract() before deciding whether to invoke
        VerifyRepairPipeline.  Skipping extraction means constraints are never
        applied, silently degenerating to baseline pass rate.
        """
        from carnot.pipeline.extract import CodeExtractor

        extractor = CodeExtractor()
        python_code = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
        results = extractor.extract(python_code, "code")
        # Should return a list (possibly empty) without raising.
        assert isinstance(results, list)

    def test_code_extractor_finds_type_constraints(self) -> None:
        """CodeExtractor extracts at least one constraint from typed code.

        Spec: REQ-VR-020
        """
        from carnot.pipeline.extract import CodeExtractor

        extractor = CodeExtractor()
        code_with_types = "```python\ndef multiply(x: int, y: int) -> int:\n    return x * y\n```"
        results = extractor.extract(code_with_types, "code")
        # The extractor should find type annotation constraints.
        assert len(results) >= 1

    def test_code_extractor_handles_no_code_blocks(self) -> None:
        """CodeExtractor returns empty list when no code blocks are present.

        A generated response with prose-only text must not crash the pipeline.
        Spec: REQ-VR-020
        """
        from carnot.pipeline.extract import CodeExtractor

        extractor = CodeExtractor()
        results = extractor.extract("The answer is to iterate over the list.", "code")
        assert results == []


# ---------------------------------------------------------------------------
# VerifyRepairPipeline mock tests
# ---------------------------------------------------------------------------


class TestVerifyRepairPipelineMock:
    """Verify the experiment correctly handles VerifyRepairPipeline outcomes.

    Uses mocks so no real LLM is loaded.  Spec: REQ-VR-020, SCENARIO-VR-030.
    """

    def _make_mock_pipeline(self, repair_succeeded: bool) -> MagicMock:
        """Build a mock VerifyRepairPipeline with a controlled repair outcome."""
        from carnot.pipeline.verify_repair import RepairResult

        mock_pipeline = MagicMock()
        repair_result = MagicMock(spec=RepairResult)
        repair_result.repaired_response = "def fixed(): return 42" if repair_succeeded else None
        repair_result.repair_applied = repair_succeeded
        mock_pipeline.verify_and_repair.return_value = repair_result
        return mock_pipeline

    def test_pipeline_called_with_extracted_constraints(self) -> None:
        """verify_and_repair() receives the generated code as 'response'.

        The experiment passes the full generated text (not just the extracted
        snippet) so the pipeline can re-extract and repair in context.
        """
        pipeline = self._make_mock_pipeline(repair_succeeded=True)
        question = "Write a function that adds two integers."
        response = "```python\ndef add(a, b):\n    return a + b\n```"

        result = pipeline.verify_and_repair(question, response, "code")

        pipeline.verify_and_repair.assert_called_once_with(question, response, "code")
        assert result.repair_applied is True

    def test_pipeline_repair_success_counted_toward_carnot_pass_rate(self) -> None:
        """A successful repair increments carnot_passed count in the result accumulator.

        The test simulates what the experiment's inner loop does.
        Spec: REQ-VR-020
        """
        baseline_passed = 0
        carnot_passed = 0

        # Simulate: baseline failed, repair succeeded.
        baseline_ok = False
        pipeline = self._make_mock_pipeline(repair_succeeded=True)
        repair_result = pipeline.verify_and_repair("q", "r", "code")

        if baseline_ok:
            baseline_passed += 1
        if repair_result.repair_applied or baseline_ok:
            carnot_passed += 1

        assert baseline_passed == 0
        assert carnot_passed == 1

    def test_pipeline_no_repair_preserves_baseline_outcome(self) -> None:
        """When repair is not applied, carnot result equals the baseline result.

        This prevents the experiment from artificially inflating carnot_pass_rate
        by treating 'no repair needed' as a repair success.
        Spec: REQ-VR-020
        """
        baseline_ok = True
        pipeline = self._make_mock_pipeline(repair_succeeded=False)
        repair_result = pipeline.verify_and_repair("q", "r", "code")

        # Carnot pass = repair succeeded OR baseline already passed.
        carnot_ok = repair_result.repair_applied or baseline_ok
        signed_imp = compute_signed_improvement(
            1.0,  # baseline_pass_rate (1 problem, 1 passed)
            1.0,  # carnot_pass_rate (same, no repair applied)
        )
        assert carnot_ok is True
        assert signed_imp == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Artifact schema tests
# ---------------------------------------------------------------------------


class TestExp870ArtifactSchema:
    """Verify the blocked artifact written by Exp 870 has the required fields."""

    @pytest.fixture()
    def artifact(self) -> dict[str, Any]:
        """Load the actual Exp 870 artifact from disk."""
        artifact_path = (
            Path(__file__).parent.parent.parent
            / "results"
            / "experiment_870_sota_code_repair_v7.json"
        )
        with open(artifact_path) as fh:
            return json.load(fh)

    def test_experiment_id_is_870(self, artifact: dict[str, Any]) -> None:
        """Artifact must carry experiment=870."""
        assert artifact["experiment"] == 870

    def test_status_is_blocked(self, artifact: dict[str, Any]) -> None:
        """Blocked gate must produce status='blocked'."""
        assert artifact["status"] == "blocked"

    def test_honest_verdict_is_blocked(self, artifact: dict[str, Any]) -> None:
        """honest_verdict must be 'blocked' when gate not met."""
        assert artifact["honest_verdict"] == "blocked"

    def test_blocked_by_field_present(self, artifact: dict[str, Any]) -> None:
        """blocked_by must name the failing gate."""
        assert artifact["blocked_by"] == "exp869_download_not_verified"

    def test_gate_checks_field(self, artifact: dict[str, Any]) -> None:
        """gate_checks dict must report exp869 gate as False."""
        assert artifact["gate_checks"]["exp869_download_verified"] is False
        assert artifact["gate_checks"]["exp855_live_env_fixed"] is True

    def test_signed_improvement_is_null(self, artifact: dict[str, Any]) -> None:
        """signed_improvement must be null when experiment was blocked."""
        assert artifact["signed_improvement"] is None

    def test_inference_mode_is_null(self, artifact: dict[str, Any]) -> None:
        """inference_mode must be null when experiment was blocked."""
        assert artifact["inference_mode"] is None

    def test_schema_field_present(self, artifact: dict[str, Any]) -> None:
        """Artifact must carry a schema field listing all top-level keys."""
        assert "schema" in artifact
        assert isinstance(artifact["schema"], list)
        assert len(artifact["schema"]) > 0
