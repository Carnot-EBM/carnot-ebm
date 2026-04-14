"""Tests for Experiment 314: AMD XDNA NPU prereq retry — check if ninja/openblas installed.

This experiment re-runs the Exp 303 NPU unblock workflow AFTER checking whether the
prerequisites (ninja, openblas) are now installed on the system.  If they are, it
attempts the ORT source build (45-min timeout).  If still missing, it emits a
blocked_prereq artifact with the same install commands as Exp 303.

Key additions over Exp 303:
  - prereq_changes: compares current prereq state to Exp 303's blocked state,
    reporting each package as "now_available" or "still_missing".
  - If honest_verdict="npu_working": npu_latency_us, cpu_latency_us, speedup_factor
    must all be present and non-null.
  - If honest_verdict != "npu_working": npu_latency_us must be null (no fabrication).

Spec:
  REQ-PRED-003 (honest labeling — no fabricated latency on blocked paths)
  SCENARIO-EXP303-A (prereq check — ninja and openblas detection with install_command)
  SCENARIO-EXP303-B (source build path — attempt with 45-min timeout, log tail on failure)
  SCENARIO-EXP303-C (inference benchmark — npu_latency_us vs cpu_latency_us when working)
  SCENARIO-EXP303-D (honest labeling — null inference_result on all blocked paths)

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_experiment_314_npu_prereq_install.py -v
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
    "prereq_changes",
    "build_outcome",
    "inference_result",
    "honest_verdict",
    "onnx_model_considered",
    "next_steps",
}

# Valid honest_verdict / execution_path values (REQ-PRED-003)
VALID_VERDICTS = {
    "npu_working",
    "blocked_build",
    "timeout",
    "blocked_prereq",
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

# Required keys inside prereq_changes
REQUIRED_PREREQ_CHANGE_KEYS = {"ninja", "openblas"}

# Valid prereq_change values for each package
VALID_PREREQ_CHANGE_VALUES = {"now_available", "still_missing"}

# Fields required inside build_outcome when present (SCENARIO-EXP303-B)
REQUIRED_BUILD_OUTCOME_KEYS = {
    "success",
    "duration_seconds",
}

# Fields required inside inference_result when npu_working (SCENARIO-EXP303-C)
REQUIRED_INFERENCE_RESULT_KEYS = {
    "npu_latency_us",
    "cpu_latency_us",
    "speedup_factor",
    "provider_used",
    "timed_calls",
}

# Blocked paths that must NOT have a real inference_result (SCENARIO-EXP303-D)
BLOCKED_VERDICTS = {
    "blocked_build",
    "blocked_prereq",
    "timeout",
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_json() -> dict[str, Any]:
    """Load the Exp 314 results JSON artifact."""
    path = (
        Path(__file__).parent.parent.parent
        / "results"
        / "experiment_314_npu_prereq_install.json"
    )
    if not path.exists():
        pytest.skip(f"Results artifact not yet generated: {path}")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def exp303_json() -> dict[str, Any]:
    """Load the Exp 303 reference artifact for comparison."""
    path = (
        Path(__file__).parent.parent.parent
        / "results"
        / "experiment_303_npu_results.json"
    )
    if not path.exists():
        pytest.skip("Exp 303 artifact not found — cannot compare prereq states")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Top-level schema (all execution paths)
# ---------------------------------------------------------------------------


class TestExp314Schema:
    """Top-level schema validation for the Exp 314 results artifact.

    Every execution path (blocked or working) must emit all required keys
    so downstream consumers can deserialize the artifact without branching.

    Spec: REQ-PRED-003 — artifacts must be schema-valid for downstream consumers.
    """

    def test_top_level_keys_present(self, results_json: dict[str, Any]) -> None:
        """All required top-level keys must be present.

        Spec: REQ-PRED-003
        """
        missing = REQUIRED_TOP_LEVEL_KEYS - set(results_json.keys())
        assert not missing, f"Top-level keys missing: {missing}"

    def test_experiment_number(self, results_json: dict[str, Any]) -> None:
        """experiment must be 314.

        Spec: REQ-PRED-003 — experiment ID identifies the artifact.
        """
        assert results_json["experiment"] == 314

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
        assert path in VALID_VERDICTS, (
            f"execution_path must be one of {VALID_VERDICTS}, got: {path!r}"
        )

    def test_execution_path_matches_honest_verdict(
        self, results_json: dict[str, Any]
    ) -> None:
        """execution_path must equal honest_verdict (single source of truth).

        Spec: REQ-PRED-003
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

    def test_prereq_changes_is_dict(self, results_json: dict[str, Any]) -> None:
        """prereq_changes must be a dict (new field in Exp 314).

        prereq_changes captures which packages changed state since Exp 303,
        giving the researcher an at-a-glance summary without diffing two files.
        """
        assert isinstance(results_json["prereq_changes"], dict), (
            "prereq_changes must be a dict"
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

    def test_next_steps_is_list(self, results_json: dict[str, Any]) -> None:
        """next_steps must be a list.

        Helps human operator know what to do next regardless of verdict.
        """
        ns = results_json["next_steps"]
        assert isinstance(ns, list), f"next_steps must be a list, got: {type(ns)}"

    def test_onnx_model_considered_is_str_or_none(
        self, results_json: dict[str, Any]
    ) -> None:
        """onnx_model_considered must be a string path or null.

        Spec: REQ-PRED-003 — provenance of the model file used.
        """
        v = results_json["onnx_model_considered"]
        assert v is None or isinstance(v, str), (
            f"onnx_model_considered must be str or None, got: {type(v)}"
        )


# ---------------------------------------------------------------------------
# prereq_check section (SCENARIO-EXP303-A)
# ---------------------------------------------------------------------------


class TestPrereqCheck314:
    """Validate prereq_check section — same detection logic as Exp 303.

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

    def test_install_command_when_ninja_missing(
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

    def test_install_command_when_openblas_missing(
        self, results_json: dict[str, Any]
    ) -> None:
        """When openblas_installed is False, openblas_install_command must be present.

        Spec: SCENARIO-EXP303-A
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

        Spec: SCENARIO-EXP303-A
        """
        pc = results_json["prereq_check"]
        if not pc["ninja_installed"] or not pc["openblas_installed"]:
            assert results_json["execution_path"] == "blocked_prereq", (
                f"Missing prereq must yield blocked_prereq, "
                f"got: {results_json['execution_path']!r}"
            )


# ---------------------------------------------------------------------------
# prereq_changes section (new in Exp 314)
# ---------------------------------------------------------------------------


class TestPrereqChanges:
    """Validate prereq_changes — delta vs Exp 303 blocked state.

    This is the key new section in Exp 314: it tells the researcher whether
    installing the packages since Exp 303 was blocked has actually taken effect.
    Without this section, comparing the two JSON artifacts manually is needed.
    """

    def test_prereq_changes_has_required_keys(
        self, results_json: dict[str, Any]
    ) -> None:
        """prereq_changes must have ninja and openblas keys.

        Both packages were blocked in Exp 303, so both must be reported here.
        """
        pc = results_json["prereq_changes"]
        missing = REQUIRED_PREREQ_CHANGE_KEYS - set(pc.keys())
        assert not missing, f"prereq_changes missing keys: {missing}"

    def test_ninja_change_is_valid_value(self, results_json: dict[str, Any]) -> None:
        """prereq_changes.ninja must be 'now_available' or 'still_missing'.

        The controlled vocabulary prevents ambiguous string values that would
        complicate automated parsing downstream.
        """
        v = results_json["prereq_changes"]["ninja"]
        assert v in VALID_PREREQ_CHANGE_VALUES, (
            f"prereq_changes.ninja must be one of {VALID_PREREQ_CHANGE_VALUES}, got: {v!r}"
        )

    def test_openblas_change_is_valid_value(self, results_json: dict[str, Any]) -> None:
        """prereq_changes.openblas must be 'now_available' or 'still_missing'.

        Spec: new field introduced in Exp 314.
        """
        v = results_json["prereq_changes"]["openblas"]
        assert v in VALID_PREREQ_CHANGE_VALUES, (
            f"prereq_changes.openblas must be one of {VALID_PREREQ_CHANGE_VALUES}, "
            f"got: {v!r}"
        )

    def test_ninja_change_consistent_with_prereq_check(
        self, results_json: dict[str, Any]
    ) -> None:
        """prereq_changes.ninja must match prereq_check.ninja_installed.

        If ninja is now installed, change must say 'now_available'; if still
        missing, 'still_missing'.  Inconsistency would mean the delta reporting
        is wrong.
        """
        pc = results_json["prereq_check"]
        chg = results_json["prereq_changes"]
        if pc["ninja_installed"]:
            assert chg["ninja"] == "now_available", (
                "ninja is installed but prereq_changes.ninja is not 'now_available'"
            )
        else:
            assert chg["ninja"] == "still_missing", (
                "ninja is not installed but prereq_changes.ninja is not 'still_missing'"
            )

    def test_openblas_change_consistent_with_prereq_check(
        self, results_json: dict[str, Any]
    ) -> None:
        """prereq_changes.openblas must match prereq_check.openblas_installed.

        Same consistency rule as ninja — the change summary must not lie.
        """
        pc = results_json["prereq_check"]
        chg = results_json["prereq_changes"]
        if pc["openblas_installed"]:
            assert chg["openblas"] == "now_available", (
                "openblas is installed but prereq_changes.openblas is not 'now_available'"
            )
        else:
            assert chg["openblas"] == "still_missing", (
                "openblas not installed but prereq_changes.openblas is not 'still_missing'"
            )

    def test_prereq_changes_consistent_with_exp303(
        self,
        results_json: dict[str, Any],
        exp303_json: dict[str, Any],
    ) -> None:
        """prereq_changes must correctly reflect the delta from Exp 303.

        Exp 303 had both ninja_installed=False and openblas_installed=False.
        If Exp 314 still shows them as False, changes must both be 'still_missing'.
        If now True (user installed them), changes must be 'now_available'.
        This test catches an artifact that claims 'now_available' while the
        package is still absent.
        """
        exp303_pc = exp303_json["prereq_check"]
        exp314_pc = results_json["prereq_check"]
        exp314_chg = results_json["prereq_changes"]

        # Ninja
        was_missing_303 = not exp303_pc["ninja_installed"]
        now_available_314 = exp314_pc["ninja_installed"]
        expected_ninja_change = "now_available" if now_available_314 else "still_missing"
        assert exp314_chg["ninja"] == expected_ninja_change, (
            f"Exp 303 ninja_installed={not was_missing_303}, "
            f"Exp 314 ninja_installed={now_available_314}, "
            f"expected prereq_changes.ninja={expected_ninja_change!r}, "
            f"got: {exp314_chg['ninja']!r}"
        )

        # OpenBLAS
        now_available_openblas = exp314_pc["openblas_installed"]
        expected_openblas_change = (
            "now_available" if now_available_openblas else "still_missing"
        )
        assert exp314_chg["openblas"] == expected_openblas_change, (
            f"Exp 303 openblas_installed={exp303_pc['openblas_installed']}, "
            f"Exp 314 openblas_installed={now_available_openblas}, "
            f"expected prereq_changes.openblas={expected_openblas_change!r}, "
            f"got: {exp314_chg['openblas']!r}"
        )


# ---------------------------------------------------------------------------
# build_outcome section (SCENARIO-EXP303-B)
# ---------------------------------------------------------------------------


class TestBuildOutcome314:
    """Validate build_outcome section when build was attempted.

    Spec: SCENARIO-EXP303-B
    """

    @pytest.fixture(autouse=True)
    def _require_build_attempted(self, results_json: dict[str, Any]) -> None:
        """Skip if build was never attempted (prereqs were still missing)."""
        if results_json["build_outcome"] is None:
            pytest.skip("build_outcome is None — build was not attempted (prereqs missing)")

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
                assert isinstance(line, str), "Each build_log_tail entry must be a string"

    def test_timeout_flag_when_timed_out(self, results_json: dict[str, Any]) -> None:
        """When build timed out, build_outcome.timeout_exceeded must be True.

        Spec: SCENARIO-EXP303-B — timeout vs compile-error distinction required.
        """
        bo = results_json["build_outcome"]
        if bo.get("timeout_exceeded"):
            assert bo["timeout_exceeded"] is True

    def test_blocked_build_when_build_failed(
        self, results_json: dict[str, Any]
    ) -> None:
        """When build_outcome.success is False and no timeout, execution_path blocked_build.

        Spec: SCENARIO-EXP303-B
        """
        bo = results_json["build_outcome"]
        if not bo["success"] and not bo.get("timeout_exceeded"):
            assert results_json["execution_path"] == "blocked_build", (
                f"Failed build (non-timeout) must yield blocked_build, "
                f"got: {results_json['execution_path']!r}"
            )

    def test_timeout_yields_timeout_verdict(
        self, results_json: dict[str, Any]
    ) -> None:
        """When build timed out, execution_path must be 'timeout'.

        Distinguishing 'timeout' from 'blocked_build' helps the researcher
        know whether to increase the build timeout vs fix a compile error.
        """
        bo = results_json["build_outcome"]
        if bo.get("timeout_exceeded"):
            assert results_json["execution_path"] == "timeout", (
                f"Timed-out build must yield 'timeout', got: {results_json['execution_path']!r}"
            )


# ---------------------------------------------------------------------------
# inference_result section (SCENARIO-EXP303-C and SCENARIO-EXP303-D)
# ---------------------------------------------------------------------------


class TestInferenceResult314:
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
        """npu_latency_us must be a positive float (real measurement, not null).

        Spec: SCENARIO-EXP303-C — no fabricated latency.
        If NPU not tested, npu_latency_us must be null (checked in TestNoFabricatedLatency).
        """
        lat = results_json["inference_result"]["npu_latency_us"]
        assert isinstance(lat, (int, float)) and lat > 0, (
            f"npu_latency_us must be > 0, got: {lat}"
        )

    def test_cpu_latency_us_is_positive(self, results_json: dict[str, Any]) -> None:
        """cpu_latency_us must be a positive float.

        Spec: SCENARIO-EXP303-C
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


class TestNoFabricatedLatency314:
    """Blocked paths must never have real-looking inference_result data.

    The invariant: if NPU was not exercised, npu_latency_us must be null.
    Fabricating a plausible-looking latency would corrupt research records.

    Spec: REQ-PRED-003 (honest labeling), SCENARIO-EXP303-D
    """

    @pytest.fixture(autouse=True)
    def _require_blocked(self, results_json: dict[str, Any]) -> None:
        """Only run on blocked paths."""
        if results_json.get("execution_path") not in BLOCKED_VERDICTS:
            pytest.skip("execution_path is not a blocked path")

    def test_inference_result_is_none_on_blocked_path(
        self, results_json: dict[str, Any]
    ) -> None:
        """inference_result must be None on any blocked execution path.

        Spec: REQ-PRED-003, SCENARIO-EXP303-D
        """
        ir = results_json["inference_result"]
        assert ir is None, (
            f"Blocked path {results_json['execution_path']!r} must not have "
            f"fabricated inference_result, got: {ir!r}"
        )

    def test_build_outcome_is_none_on_prereq_blocked(
        self, results_json: dict[str, Any]
    ) -> None:
        """build_outcome must be None when prereqs are still missing.

        If we never tried the build, build_outcome must not be present as a dict.
        """
        if results_json["execution_path"] == "blocked_prereq":
            bo = results_json["build_outcome"]
            assert bo is None, (
                f"blocked_prereq path must not have build_outcome, got: {bo!r}"
            )
