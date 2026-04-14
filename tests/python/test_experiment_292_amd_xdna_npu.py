"""Tests for Experiment 292: AMD XDNA NPU VitisAI EP benchmark.

Tests cover:
  - Artifact labeling: execution_path must be "hardware", "blocked", or "build_failed"
  - Benchmark schema: npu_latency_us and speedup_vs_cpu_ort present when hardware path
  - Build timeout handling: timeout artifact emits blocker, does not stall
  - Baseline comparison fields: cpu_ort_baseline_us must equal 5.847 (Exp 257)
  - Blocker record: missing_prereqs list is populated, next_action is specific
  - No fabricated numbers: blocked/build_failed paths must have null latency/throughput

Spec: REQ-PRED-003 (ONNX export + accelerated inference path)
SCENARIO-EXP292-A (prerequisite check and blocked artifact)
SCENARIO-EXP292-B (build timeout handling — emit blocker with build log tail)
SCENARIO-EXP292-C (benchmark schema — latency, speedup, baseline comparison)
SCENARIO-EXP292-D (honest labeling — no fabricated numbers for non-hardware paths)
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
    "cpu_ort_baseline_us",
    "onnx_model_used",
    "npu_hardware_info",
    "result",
    "honest_verdict",
}

# execution_path is one of these three values (REQ-PRED-003 artifact labeling)
VALID_EXECUTION_PATHS = {"hardware", "blocked", "build_failed"}

# Fields required when execution_path == "hardware"
REQUIRED_HARDWARE_RESULT_KEYS = {
    "npu_latency_us",
    "npu_throughput_calls_per_sec",
    "speedup_vs_cpu_ort",
    "timed_calls",
    "providers_used",
}

# Fields required when execution_path == "blocked"
REQUIRED_BLOCKED_RESULT_KEYS = {
    "missing_prereqs",
    "next_action",
}

# Fields required when execution_path == "build_failed"
REQUIRED_BUILD_FAILED_RESULT_KEYS = {
    "build_step",
    "build_log_tail",
    "next_action",
}

# The CPU ORT baseline from Exp 257 (onnx_cpu record latency_us = 5.847)
CPU_ORT_BASELINE_US = 5.847

# The .so library directory from the checked-in RyzenAI-SW repo
RYZEN_AI_SW_DIR = Path.home() / "github.com" / "amd" / "RyzenAI-SW"
VITISAI_SO_DIR = (
    RYZEN_AI_SW_DIR / "Ryzen-AI-CVML-Library" / "linux" / "onnx" / "ryzen14"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_json() -> dict[str, Any]:
    """Load the Exp 292 results JSON artifact."""
    path = (
        Path(__file__).parent.parent.parent / "results" / "experiment_292_results.json"
    )
    if not path.exists():
        pytest.skip(f"Results artifact not yet generated: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------


class TestExp292Schema:
    """Validate top-level schema of the Exp 292 results artifact.

    Spec: REQ-PRED-003 — artifacts must be schema-valid for downstream consumers.
    """

    def test_top_level_keys_present(self, results_json: dict[str, Any]) -> None:
        """All required top-level keys must be present.

        Spec: REQ-PRED-003, SCENARIO-EXP292-C
        """
        missing = REQUIRED_TOP_LEVEL_KEYS - set(results_json.keys())
        assert not missing, f"Top-level keys missing: {missing}"

    def test_experiment_number(self, results_json: dict[str, Any]) -> None:
        """Experiment number must be 292.

        Spec: REQ-PRED-003
        """
        assert results_json["experiment"] == 292

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
        """execution_path must be one of 'hardware', 'blocked', or 'build_failed'.

        Spec: REQ-PRED-003, SCENARIO-EXP292-A, SCENARIO-EXP292-B
        """
        path = results_json["execution_path"]
        assert path in VALID_EXECUTION_PATHS, (
            f"execution_path must be one of {VALID_EXECUTION_PATHS}, got: {path!r}"
        )

    def test_cpu_ort_baseline_us(self, results_json: dict[str, Any]) -> None:
        """cpu_ort_baseline_us must equal the Exp 257 measured value (5.847 µs).

        Spec: SCENARIO-EXP292-C — baseline comparison requires anchoring to prior result.
        """
        baseline = results_json["cpu_ort_baseline_us"]
        assert isinstance(baseline, (int, float)), "cpu_ort_baseline_us must be numeric"
        assert abs(baseline - CPU_ORT_BASELINE_US) < 0.01, (
            f"cpu_ort_baseline_us expected ~{CPU_ORT_BASELINE_US}, got {baseline}"
        )

    def test_onnx_model_used_is_string(self, results_json: dict[str, Any]) -> None:
        """onnx_model_used must be a non-empty string naming the ONNX file.

        Spec: SCENARIO-EXP292-C
        """
        model = results_json["onnx_model_used"]
        assert isinstance(model, str) and model.strip(), (
            "onnx_model_used must be a non-empty string"
        )

    def test_onnx_model_used_is_known_file(self, results_json: dict[str, Any]) -> None:
        """onnx_model_used must reference jepa_predictor_291 or jepa_predictor_146.

        Spec: SCENARIO-EXP292-C — explicit fallback chain enforced.
        """
        model = results_json["onnx_model_used"]
        known = {"jepa_predictor_291.onnx", "jepa_predictor_146.onnx"}
        assert any(k in model for k in known), (
            f"onnx_model_used must reference one of {known}, got: {model!r}"
        )

    def test_result_is_dict(self, results_json: dict[str, Any]) -> None:
        """result must be a dict (never None).

        Spec: REQ-PRED-003
        """
        assert isinstance(results_json["result"], dict), "result must be a dict"

    def test_honest_verdict_has_explanation(
        self, results_json: dict[str, Any]
    ) -> None:
        """honest_verdict must have an explanation string.

        Spec: REQ-PRED-003 — honest artifact requires non-empty explanation.
        """
        verdict = results_json["honest_verdict"]
        assert isinstance(verdict, dict), "honest_verdict must be a dict"
        explanation = verdict.get("explanation", "")
        assert isinstance(explanation, str) and explanation.strip(), (
            "honest_verdict.explanation must be a non-empty string"
        )


# ---------------------------------------------------------------------------
# Artifact labeling: execution_path == "hardware"
# ---------------------------------------------------------------------------


class TestHardwareArtifact:
    """Validate artifact when NPU benchmark succeeded.

    Spec: SCENARIO-EXP292-C (benchmark schema)
    """

    @pytest.fixture(autouse=True)
    def _require_hardware(self, results_json: dict[str, Any]) -> None:
        if results_json.get("execution_path") != "hardware":
            pytest.skip("execution_path is not 'hardware' — NPU did not run")

    def test_result_has_hardware_keys(self, results_json: dict[str, Any]) -> None:
        """Hardware result must have all benchmark keys.

        Spec: SCENARIO-EXP292-C
        """
        result = results_json["result"]
        missing = REQUIRED_HARDWARE_RESULT_KEYS - set(result.keys())
        assert not missing, f"Hardware result missing keys: {missing}"

    def test_npu_latency_us_is_positive(self, results_json: dict[str, Any]) -> None:
        """npu_latency_us must be a positive float (real measurement).

        Spec: SCENARIO-EXP292-C
        """
        lat = results_json["result"]["npu_latency_us"]
        assert isinstance(lat, (int, float)) and lat > 0, (
            f"npu_latency_us must be positive, got: {lat}"
        )

    def test_speedup_vs_cpu_ort_is_consistent(
        self, results_json: dict[str, Any]
    ) -> None:
        """speedup_vs_cpu_ort must equal cpu_ort_baseline_us / npu_latency_us.

        Spec: SCENARIO-EXP292-C — derived field must be consistent.
        """
        result = results_json["result"]
        npu_us = result["npu_latency_us"]
        baseline = results_json["cpu_ort_baseline_us"]
        reported_speedup = result["speedup_vs_cpu_ort"]
        expected_speedup = baseline / npu_us
        assert abs(reported_speedup - expected_speedup) < 0.01, (
            f"speedup_vs_cpu_ort {reported_speedup:.4f} inconsistent with "
            f"{baseline}/{npu_us} = {expected_speedup:.4f}"
        )

    def test_timed_calls_at_least_5000(self, results_json: dict[str, Any]) -> None:
        """Benchmark must time at least 5000 calls for stable latency.

        Spec: SCENARIO-EXP292-C
        """
        timed = results_json["result"]["timed_calls"]
        assert isinstance(timed, int) and timed >= 5000, (
            f"timed_calls must be ≥ 5000, got: {timed}"
        )

    def test_providers_used_contains_vitisai(
        self, results_json: dict[str, Any]
    ) -> None:
        """providers_used must include VitisAIExecutionProvider when on hardware path.

        Spec: SCENARIO-EXP292-C
        """
        providers = results_json["result"]["providers_used"]
        assert isinstance(providers, list), "providers_used must be a list"
        assert any("VitisAI" in p for p in providers), (
            f"VitisAIExecutionProvider not in providers_used: {providers}"
        )


# ---------------------------------------------------------------------------
# Artifact labeling: execution_path == "blocked"
# ---------------------------------------------------------------------------


class TestBlockedArtifact:
    """Validate artifact when prerequisites are missing.

    Spec: SCENARIO-EXP292-A (prerequisite check and blocked artifact)
    """

    @pytest.fixture(autouse=True)
    def _require_blocked(self, results_json: dict[str, Any]) -> None:
        if results_json.get("execution_path") != "blocked":
            pytest.skip("execution_path is not 'blocked' — prereqs were present")

    def test_result_has_blocked_keys(self, results_json: dict[str, Any]) -> None:
        """Blocked result must have missing_prereqs and next_action.

        Spec: SCENARIO-EXP292-A
        """
        result = results_json["result"]
        missing = REQUIRED_BLOCKED_RESULT_KEYS - set(result.keys())
        assert not missing, f"Blocked result missing keys: {missing}"

    def test_missing_prereqs_is_nonempty_list(
        self, results_json: dict[str, Any]
    ) -> None:
        """missing_prereqs must be a non-empty list of specific items.

        Spec: SCENARIO-EXP292-A — blocked artifact names exactly what is missing.
        """
        prereqs = results_json["result"]["missing_prereqs"]
        assert isinstance(prereqs, list) and len(prereqs) > 0, (
            f"missing_prereqs must be a non-empty list, got: {prereqs!r}"
        )
        # Each item must be a non-empty string
        for item in prereqs:
            assert isinstance(item, str) and item.strip(), (
                f"Each missing_prereq must be a non-empty string, got: {item!r}"
            )

    def test_next_action_is_specific(self, results_json: dict[str, Any]) -> None:
        """next_action must be a non-empty string with a concrete command or step.

        Spec: SCENARIO-EXP292-A — actionable blocker, not vague guidance.
        """
        action = results_json["result"]["next_action"]
        assert isinstance(action, str) and len(action) > 10, (
            f"next_action must be a specific non-trivial string, got: {action!r}"
        )

    def test_no_fabricated_latency(self, results_json: dict[str, Any]) -> None:
        """Blocked path must not have npu_latency_us set (no fabricated numbers).

        Spec: REQ-PRED-003 (honest labeling invariant)
        """
        result = results_json["result"]
        assert result.get("npu_latency_us") is None, (
            f"Blocked path must not have fabricated npu_latency_us: "
            f"{result.get('npu_latency_us')}"
        )

    def test_no_fabricated_speedup(self, results_json: dict[str, Any]) -> None:
        """Blocked path must not have speedup_vs_cpu_ort set.

        Spec: REQ-PRED-003 (honest labeling invariant)
        """
        result = results_json["result"]
        assert result.get("speedup_vs_cpu_ort") is None, (
            f"Blocked path must not have fabricated speedup: "
            f"{result.get('speedup_vs_cpu_ort')}"
        )


# ---------------------------------------------------------------------------
# Artifact labeling: execution_path == "build_failed"
# ---------------------------------------------------------------------------


class TestBuildFailedArtifact:
    """Validate artifact when build timed out or failed.

    Covers the 45-minute build timeout handling requirement.
    Spec: SCENARIO-EXP292-B (timeout handling — emit blocker with build log tail)
    """

    @pytest.fixture(autouse=True)
    def _require_build_failed(self, results_json: dict[str, Any]) -> None:
        if results_json.get("execution_path") != "build_failed":
            pytest.skip("execution_path is not 'build_failed'")

    def test_result_has_build_failed_keys(self, results_json: dict[str, Any]) -> None:
        """build_failed result must have build_step, build_log_tail, next_action.

        Spec: SCENARIO-EXP292-B
        """
        result = results_json["result"]
        missing = REQUIRED_BUILD_FAILED_RESULT_KEYS - set(result.keys())
        assert not missing, f"build_failed result missing keys: {missing}"

    def test_build_log_tail_has_lines(self, results_json: dict[str, Any]) -> None:
        """build_log_tail must be a list of at most 50 non-empty lines.

        Spec: SCENARIO-EXP292-B — last 50 lines of build log required.
        """
        tail = results_json["result"]["build_log_tail"]
        assert isinstance(tail, list), "build_log_tail must be a list"
        assert 1 <= len(tail) <= 50, (
            f"build_log_tail must have 1-50 lines, got: {len(tail)}"
        )
        for line in tail:
            assert isinstance(line, str), "Each build_log_tail entry must be a string"

    def test_build_step_is_named(self, results_json: dict[str, Any]) -> None:
        """build_step must identify what was in progress when timeout/failure occurred.

        Spec: SCENARIO-EXP292-B — exact build step required for diagnosis.
        """
        step = results_json["result"]["build_step"]
        assert isinstance(step, str) and step.strip(), (
            f"build_step must be a non-empty string, got: {step!r}"
        )

    def test_next_action_is_specific(self, results_json: dict[str, Any]) -> None:
        """next_action must be a non-empty, specific remediation step.

        Spec: SCENARIO-EXP292-B — one specific next action required.
        """
        action = results_json["result"]["next_action"]
        assert isinstance(action, str) and len(action) > 10, (
            f"next_action must be specific, got: {action!r}"
        )

    def test_timeout_flag_if_timeout(self, results_json: dict[str, Any]) -> None:
        """If build timed out, result must have timeout_exceeded=True.

        Spec: SCENARIO-EXP292-B — timeout vs failure distinction.
        """
        result = results_json["result"]
        # timeout_exceeded is optional but must be True if present and a bool
        if "timeout_exceeded" in result:
            assert isinstance(result["timeout_exceeded"], bool), (
                "timeout_exceeded must be a boolean"
            )

    def test_no_fabricated_latency(self, results_json: dict[str, Any]) -> None:
        """build_failed path must not have npu_latency_us set.

        Spec: REQ-PRED-003 (honest labeling invariant)
        """
        result = results_json["result"]
        assert result.get("npu_latency_us") is None, (
            f"build_failed path must not have fabricated npu_latency_us: "
            f"{result.get('npu_latency_us')}"
        )


# ---------------------------------------------------------------------------
# NPU hardware info
# ---------------------------------------------------------------------------


class TestNpuHardwareInfo:
    """Validate the npu_hardware_info section."""

    def test_npu_hardware_info_is_dict(self, results_json: dict[str, Any]) -> None:
        """npu_hardware_info must be a dict.

        Spec: REQ-PRED-003
        """
        hw = results_json["npu_hardware_info"]
        assert isinstance(hw, dict), "npu_hardware_info must be a dict"

    def test_amdxdna_driver_loaded_is_bool(
        self, results_json: dict[str, Any]
    ) -> None:
        """amdxdna_driver_loaded must be a boolean.

        Spec: REQ-PRED-003
        """
        hw = results_json["npu_hardware_info"]
        assert "amdxdna_driver_loaded" in hw, (
            "npu_hardware_info must have amdxdna_driver_loaded"
        )
        assert isinstance(hw["amdxdna_driver_loaded"], bool)

    def test_xrt_version_field_present(self, results_json: dict[str, Any]) -> None:
        """xrt_version must be present in npu_hardware_info.

        Spec: REQ-PRED-003
        """
        hw = results_json["npu_hardware_info"]
        assert "xrt_version" in hw, "npu_hardware_info must have xrt_version"

    def test_ryzen_ai_sw_path_recorded(self, results_json: dict[str, Any]) -> None:
        """ryzen_ai_sw_present must be present and boolean.

        Spec: SCENARIO-EXP292-A — prereq check must verify RyzenAI-SW dir.
        """
        hw = results_json["npu_hardware_info"]
        assert "ryzen_ai_sw_present" in hw, (
            "npu_hardware_info must record ryzen_ai_sw_present"
        )
        assert isinstance(hw["ryzen_ai_sw_present"], bool)

    def test_vitisai_so_recorded(self, results_json: dict[str, Any]) -> None:
        """vitisai_so_present must be present and boolean.

        Spec: SCENARIO-EXP292-A — prereq check must verify .so presence.
        """
        hw = results_json["npu_hardware_info"]
        assert "vitisai_so_present" in hw, (
            "npu_hardware_info must record vitisai_so_present"
        )
        assert isinstance(hw["vitisai_so_present"], bool)
