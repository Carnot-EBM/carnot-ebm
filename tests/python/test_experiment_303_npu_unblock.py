"""Tests for Experiment 303: AMD XDNA NPU unblock — install prereqs, source build, benchmark.

Tests cover:
  - Artifact schema: execution_path, prereq_check, build_outcome, inference_result,
    honest_verdict all present with correct types.
  - prereq_check: ninja_installed and openblas_installed are booleans; if False, an
    install_command string must explain how to fix it.
  - build_outcome: present when prereqs were met; has success (bool), duration_seconds
    (float), and, if failed, error_summary (str) and build_log_tail (list of str).
  - inference_result: None when any upstream step blocked; when present contains
    npu_latency_us, cpu_latency_us, speedup_factor, provider_used.
  - honest_verdict: exactly one of "npu_working" / "blocked_build" / "blocked_prereq"
    / "blocked_abi"; no fabricated latency on blocked paths.

Spec: REQ-PRED-003
SCENARIO-EXP303-A (prereq check — ninja and openblas detection with install_command)
SCENARIO-EXP303-B (source build path — attempt with 45-min timeout, log tail on failure)
SCENARIO-EXP303-C (inference benchmark — npu_latency_us vs cpu_latency_us when working)
SCENARIO-EXP303-D (honest labeling — null inference_result on all blocked paths)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "description",
    "run_date",
    "execution_path",
    "prereq_check",
    "build_outcome",
    "inference_result",
    "honest_verdict",
}

# Valid execution_path values (REQ-PRED-003 honest labeling)
VALID_EXECUTION_PATHS = {
    "npu_working",
    "blocked_build",
    "blocked_prereq",
    "blocked_abi",
}

# Fields required inside prereq_check (SCENARIO-EXP303-A)
REQUIRED_PREREQ_KEYS = {
    "ninja_installed",
    "openblas_installed",
    "cmake_version",
    "cmake_sufficient",
    "ryzen_ai_sw_present",
    "vitisai_so_present",
}

# Fields required inside build_outcome when present (SCENARIO-EXP303-B)
REQUIRED_BUILD_OUTCOME_KEYS = {
    "success",
    "duration_seconds",
}

# Fields required inside inference_result when present (SCENARIO-EXP303-C)
REQUIRED_INFERENCE_RESULT_KEYS = {
    "npu_latency_us",
    "cpu_latency_us",
    "speedup_factor",
    "provider_used",
    "timed_calls",
}

# Blocked paths that must NOT have a real inference_result (SCENARIO-EXP303-D)
BLOCKED_EXECUTION_PATHS = {
    "blocked_build",
    "blocked_prereq",
    "blocked_abi",
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_json() -> dict[str, Any]:
    """Load the Exp 303 results JSON artifact."""
    path = (
        Path(__file__).parent.parent.parent / "results" / "experiment_303_npu_results.json"
    )
    if not path.exists():
        pytest.skip(f"Results artifact not yet generated: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Top-level schema (all execution paths)
# ---------------------------------------------------------------------------


class TestExp303Schema:
    """Top-level schema validation for the Exp 303 results artifact.

    Spec: REQ-PRED-003 — artifacts must be schema-valid for downstream consumers.
    """

    def test_top_level_keys_present(self, results_json: dict[str, Any]) -> None:
        """All required top-level keys must be present.

        Spec: REQ-PRED-003, SCENARIO-EXP303-A
        """
        missing = REQUIRED_TOP_LEVEL_KEYS - set(results_json.keys())
        assert not missing, f"Top-level keys missing: {missing}"

    def test_experiment_number(self, results_json: dict[str, Any]) -> None:
        """experiment must be 303.

        Spec: REQ-PRED-003
        """
        assert results_json["experiment"] == 303

    def test_run_date_format(self, results_json: dict[str, Any]) -> None:
        """run_date must be an 8-digit YYYYMMDD string.

        Spec: REQ-PRED-003
        """
        run_date = results_json["run_date"]
        assert isinstance(run_date, str), "run_date must be a string"
        assert re.fullmatch(r"\d{8}", run_date), (
            f"run_date must be YYYYMMDD, got: {run_date!r}"
        )

    def test_execution_path_is_valid(self, results_json: dict[str, Any]) -> None:
        """execution_path must be one of the four honest verdict values.

        Spec: REQ-PRED-003, SCENARIO-EXP303-A
        """
        path = results_json["execution_path"]
        assert path in VALID_EXECUTION_PATHS, (
            f"execution_path must be one of {VALID_EXECUTION_PATHS}, got: {path!r}"
        )

    def test_execution_path_matches_honest_verdict(
        self, results_json: dict[str, Any]
    ) -> None:
        """execution_path must equal honest_verdict (they are the same classification).

        Spec: REQ-PRED-003 — single source of truth for outcome label.
        """
        assert results_json["execution_path"] == results_json["honest_verdict"], (
            f"execution_path {results_json['execution_path']!r} must match "
            f"honest_verdict {results_json['honest_verdict']!r}"
        )

    def test_prereq_check_is_dict(self, results_json: dict[str, Any]) -> None:
        """prereq_check must be a dict.

        Spec: SCENARIO-EXP303-A
        """
        assert isinstance(results_json["prereq_check"], dict), (
            "prereq_check must be a dict"
        )

    def test_inference_result_type(self, results_json: dict[str, Any]) -> None:
        """inference_result must be None or a dict.

        Spec: SCENARIO-EXP303-C, SCENARIO-EXP303-D
        """
        ir = results_json["inference_result"]
        assert ir is None or isinstance(ir, dict), (
            f"inference_result must be None or dict, got: {type(ir)}"
        )

    def test_build_outcome_type(self, results_json: dict[str, Any]) -> None:
        """build_outcome must be None or a dict.

        Spec: SCENARIO-EXP303-B
        """
        bo = results_json["build_outcome"]
        assert bo is None or isinstance(bo, dict), (
            f"build_outcome must be None or dict, got: {type(bo)}"
        )


# ---------------------------------------------------------------------------
# prereq_check section (SCENARIO-EXP303-A)
# ---------------------------------------------------------------------------


class TestPrereqCheck:
    """Validate prereq_check section for all execution paths.

    Spec: SCENARIO-EXP303-A
    """

    def test_prereq_check_has_required_keys(
        self, results_json: dict[str, Any]
    ) -> None:
        """prereq_check must contain all required detection fields.

        Spec: SCENARIO-EXP303-A
        """
        pc = results_json["prereq_check"]
        missing = REQUIRED_PREREQ_KEYS - set(pc.keys())
        assert not missing, f"prereq_check missing keys: {missing}"

    def test_ninja_installed_is_bool(self, results_json: dict[str, Any]) -> None:
        """ninja_installed must be a boolean.

        Spec: SCENARIO-EXP303-A
        """
        pc = results_json["prereq_check"]
        assert isinstance(pc["ninja_installed"], bool), (
            f"ninja_installed must be bool, got: {type(pc['ninja_installed'])}"
        )

    def test_openblas_installed_is_bool(self, results_json: dict[str, Any]) -> None:
        """openblas_installed must be a boolean.

        Spec: SCENARIO-EXP303-A
        """
        pc = results_json["prereq_check"]
        assert isinstance(pc["openblas_installed"], bool), (
            f"openblas_installed must be bool, got: {type(pc['openblas_installed'])}"
        )

    def test_cmake_sufficient_is_bool(self, results_json: dict[str, Any]) -> None:
        """cmake_sufficient must be a boolean.

        Spec: SCENARIO-EXP303-A
        """
        pc = results_json["prereq_check"]
        assert isinstance(pc["cmake_sufficient"], bool), (
            f"cmake_sufficient must be bool, got: {type(pc['cmake_sufficient'])}"
        )

    def test_install_command_present_when_ninja_missing(
        self, results_json: dict[str, Any]
    ) -> None:
        """When ninja_installed is False, ninja_install_command must be present.

        Spec: SCENARIO-EXP303-A — blocked artifact names install command.
        """
        pc = results_json["prereq_check"]
        if not pc["ninja_installed"]:
            cmd = pc.get("ninja_install_command", "")
            assert isinstance(cmd, str) and cmd.strip(), (
                "ninja_install_command must be a non-empty string when ninja is missing"
            )

    def test_install_command_present_when_openblas_missing(
        self, results_json: dict[str, Any]
    ) -> None:
        """When openblas_installed is False, openblas_install_command must be present.

        Spec: SCENARIO-EXP303-A — blocked artifact names install command.
        """
        pc = results_json["prereq_check"]
        if not pc["openblas_installed"]:
            cmd = pc.get("openblas_install_command", "")
            assert isinstance(cmd, str) and cmd.strip(), (
                "openblas_install_command must be non-empty string when openblas missing"
            )

    def test_blocked_prereq_when_prereqs_missing(
        self, results_json: dict[str, Any]
    ) -> None:
        """When ninja or openblas is False, execution_path must be blocked_prereq.

        Spec: SCENARIO-EXP303-A — prereq failure determines path immediately.
        """
        pc = results_json["prereq_check"]
        if not pc["ninja_installed"] or not pc["openblas_installed"]:
            assert results_json["execution_path"] == "blocked_prereq", (
                f"Missing prereq must yield blocked_prereq, "
                f"got: {results_json['execution_path']!r}"
            )


# ---------------------------------------------------------------------------
# build_outcome section (SCENARIO-EXP303-B)
# ---------------------------------------------------------------------------


class TestBuildOutcome:
    """Validate build_outcome section when build was attempted.

    Spec: SCENARIO-EXP303-B
    """

    @pytest.fixture(autouse=True)
    def _require_build_attempted(self, results_json: dict[str, Any]) -> None:
        """Skip if build was never attempted (prereqs were missing)."""
        if results_json["build_outcome"] is None:
            pytest.skip("build_outcome is None — build was not attempted")

    def test_build_outcome_has_required_keys(
        self, results_json: dict[str, Any]
    ) -> None:
        """build_outcome must have success and duration_seconds.

        Spec: SCENARIO-EXP303-B
        """
        bo = results_json["build_outcome"]
        missing = REQUIRED_BUILD_OUTCOME_KEYS - set(bo.keys())
        assert not missing, f"build_outcome missing keys: {missing}"

    def test_success_is_bool(self, results_json: dict[str, Any]) -> None:
        """build_outcome.success must be a boolean.

        Spec: SCENARIO-EXP303-B
        """
        bo = results_json["build_outcome"]
        assert isinstance(bo["success"], bool), (
            f"build_outcome.success must be bool, got: {type(bo['success'])}"
        )

    def test_duration_seconds_is_non_negative(
        self, results_json: dict[str, Any]
    ) -> None:
        """build_outcome.duration_seconds must be a non-negative number.

        Spec: SCENARIO-EXP303-B
        """
        bo = results_json["build_outcome"]
        dur = bo["duration_seconds"]
        assert isinstance(dur, (int, float)) and dur >= 0, (
            f"build_outcome.duration_seconds must be >= 0, got: {dur}"
        )

    def test_error_summary_present_on_failure(
        self, results_json: dict[str, Any]
    ) -> None:
        """When build failed, error_summary must be a non-empty string.

        Spec: SCENARIO-EXP303-B — exact failure reason required for diagnosis.
        """
        bo = results_json["build_outcome"]
        if not bo["success"]:
            error = bo.get("error_summary", "")
            assert isinstance(error, str) and error.strip(), (
                "build_outcome.error_summary must be non-empty when build failed"
            )

    def test_build_log_tail_on_failure(self, results_json: dict[str, Any]) -> None:
        """When build failed, build_log_tail must be a list of 1-50 strings.

        Spec: SCENARIO-EXP303-B — last N lines of build log required.
        """
        bo = results_json["build_outcome"]
        if not bo["success"]:
            tail = bo.get("build_log_tail", [])
            assert isinstance(tail, list) and 1 <= len(tail) <= 50, (
                f"build_log_tail must be a list of 1-50 lines, got: {len(tail)}"
            )
            for line in tail:
                assert isinstance(line, str), (
                    "Each build_log_tail entry must be a string"
                )

    def test_timeout_flag_when_timed_out(self, results_json: dict[str, Any]) -> None:
        """When build timed out, build_outcome.timeout_exceeded must be True.

        Spec: SCENARIO-EXP303-B — timeout vs compile-error distinction required.
        """
        bo = results_json["build_outcome"]
        if bo.get("timeout_exceeded"):
            assert bo["timeout_exceeded"] is True, (
                "timeout_exceeded must be boolean True"
            )

    def test_blocked_build_when_build_failed(
        self, results_json: dict[str, Any]
    ) -> None:
        """When build_outcome.success is False, execution_path must be blocked_build.

        Spec: SCENARIO-EXP303-B
        """
        bo = results_json["build_outcome"]
        if not bo["success"]:
            assert results_json["execution_path"] == "blocked_build", (
                f"Failed build must yield blocked_build path, "
                f"got: {results_json['execution_path']!r}"
            )


# ---------------------------------------------------------------------------
# inference_result section (SCENARIO-EXP303-C and SCENARIO-EXP303-D)
# ---------------------------------------------------------------------------


class TestInferenceResult:
    """Validate inference_result when NPU benchmark succeeded.

    Spec: SCENARIO-EXP303-C
    """

    @pytest.fixture(autouse=True)
    def _require_npu_working(self, results_json: dict[str, Any]) -> None:
        """Skip if NPU did not run."""
        if results_json.get("execution_path") != "npu_working":
            pytest.skip("execution_path is not 'npu_working' — skipping inference tests")

    def test_inference_result_is_dict(self, results_json: dict[str, Any]) -> None:
        """When npu_working, inference_result must be a dict.

        Spec: SCENARIO-EXP303-C
        """
        assert isinstance(results_json["inference_result"], dict), (
            "inference_result must be a dict when execution_path='npu_working'"
        )

    def test_inference_result_has_required_keys(
        self, results_json: dict[str, Any]
    ) -> None:
        """inference_result must have all required benchmark fields.

        Spec: SCENARIO-EXP303-C
        """
        ir = results_json["inference_result"]
        missing = REQUIRED_INFERENCE_RESULT_KEYS - set(ir.keys())
        assert not missing, f"inference_result missing keys: {missing}"

    def test_npu_latency_us_is_positive(self, results_json: dict[str, Any]) -> None:
        """npu_latency_us must be a positive float (real measurement).

        Spec: SCENARIO-EXP303-C — no fabricated latency.
        """
        lat = results_json["inference_result"]["npu_latency_us"]
        assert isinstance(lat, (int, float)) and lat > 0, (
            f"npu_latency_us must be > 0, got: {lat}"
        )

    def test_cpu_latency_us_is_positive(self, results_json: dict[str, Any]) -> None:
        """cpu_latency_us must be a positive float (measured in same run).

        Spec: SCENARIO-EXP303-C — CPU baseline measured alongside NPU.
        """
        lat = results_json["inference_result"]["cpu_latency_us"]
        assert isinstance(lat, (int, float)) and lat > 0, (
            f"cpu_latency_us must be > 0, got: {lat}"
        )

    def test_speedup_factor_consistent(self, results_json: dict[str, Any]) -> None:
        """speedup_factor must equal cpu_latency_us / npu_latency_us.

        Spec: SCENARIO-EXP303-C — derived field must be internally consistent.
        """
        ir = results_json["inference_result"]
        npu_us = ir["npu_latency_us"]
        cpu_us = ir["cpu_latency_us"]
        reported = ir["speedup_factor"]
        expected = cpu_us / npu_us
        assert abs(reported - expected) < 0.05, (
            f"speedup_factor {reported:.4f} inconsistent with "
            f"{cpu_us}/{npu_us} = {expected:.4f}"
        )

    def test_timed_calls_at_least_100(self, results_json: dict[str, Any]) -> None:
        """Benchmark must time at least 100 calls for stable latency.

        Spec: SCENARIO-EXP303-C
        """
        timed = results_json["inference_result"]["timed_calls"]
        assert isinstance(timed, int) and timed >= 100, (
            f"timed_calls must be >= 100, got: {timed}"
        )

    def test_provider_used_contains_vitisai(self, results_json: dict[str, Any]) -> None:
        """provider_used must reference VitisAI when NPU is working.

        Spec: SCENARIO-EXP303-C
        """
        provider = results_json["inference_result"]["provider_used"]
        assert isinstance(provider, str) and "VitisAI" in provider, (
            f"provider_used must contain 'VitisAI', got: {provider!r}"
        )


class TestNoFabricatedLatency:
    """Blocked paths must never have real-looking inference_result data.

    Spec: REQ-PRED-003, SCENARIO-EXP303-D
    """

    @pytest.fixture(autouse=True)
    def _require_blocked(self, results_json: dict[str, Any]) -> None:
        """Only run on blocked paths."""
        if results_json.get("execution_path") not in BLOCKED_EXECUTION_PATHS:
            pytest.skip("execution_path is not a blocked path")

    def test_inference_result_is_none_on_blocked_path(
        self, results_json: dict[str, Any]
    ) -> None:
        """inference_result must be None on any blocked execution path.

        Spec: REQ-PRED-003 (honest labeling invariant), SCENARIO-EXP303-D
        """
        ir = results_json["inference_result"]
        assert ir is None, (
            f"Blocked path {results_json['execution_path']!r} must not have "
            f"fabricated inference_result, got: {ir!r}"
        )
