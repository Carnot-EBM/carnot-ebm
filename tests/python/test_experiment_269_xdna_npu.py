"""Tests for Experiment 269: AMD XDNA NPU VitisAI EP diagnostic.

Tests cover:
  - Diagnostic artifact schema validation (all required fields present)
  - Honest blocker handling (no fabricated throughput when EP absent)
  - Provider enumeration completeness
  - Blocker record has actionable ``next_command`` field

Spec: REQ-PRED-003 (ONNX export + accelerated inference path)
Run date: 20260413
"""

from __future__ import annotations

import json
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
    "hardware_paths",
    "honest_verdict",
    "npu_hardware_info",
}

REQUIRED_PROVIDER_ENUM_KEYS = {
    "hardware_path",
    "status",
    "run_date",
    "ort_version",
    "available_providers",
}

REQUIRED_BLOCKER_KEYS = {
    "hardware_path",
    "status",
    "run_date",
    "missing_component",
    "exact_error",
    "next_command",
    "latency_ms",
    "throughput_calls_per_sec",
}

REQUIRED_OK_KEYS = {
    "hardware_path",
    "status",
    "run_date",
    "latency_us",
    "latency_ms",
    "throughput_calls_per_sec",
    "providers_used",
}

REQUIRED_VERDICT_KEYS = {
    "npu_ep_loaded",
    "explanation",
    "recommended_next_steps",
}

REQUIRED_NPU_HW_KEYS = {
    "amdxdna_driver_loaded",
    "pci_id",
    "xrt_version",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_json() -> dict[str, Any]:
    """Load the experiment 269 results JSON artifact."""
    path = Path(__file__).parent.parent.parent / "results" / "experiment_269_results.json"
    if not path.exists():
        pytest.skip(f"Results artifact not yet generated: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestExp269Schema:
    """Validate top-level schema of the experiment 269 results artifact."""

    def test_top_level_keys_present(self, results_json: dict[str, Any]) -> None:
        """All required top-level keys must be present.

        Spec: REQ-PRED-003 — artifacts must be schema-valid for downstream consumers.
        """
        missing = REQUIRED_TOP_LEVEL_KEYS - set(results_json.keys())
        assert not missing, f"Top-level keys missing: {missing}"

    def test_experiment_number(self, results_json: dict[str, Any]) -> None:
        """Experiment number must be 269.

        Spec: REQ-PRED-003
        """
        assert results_json["experiment"] == 269

    def test_run_date(self, results_json: dict[str, Any]) -> None:
        """Run date must be 20260413.

        Spec: REQ-PRED-003
        """
        assert results_json["run_date"] == "20260413"

    def test_hardware_paths_is_list(self, results_json: dict[str, Any]) -> None:
        """hardware_paths must be a non-empty list.

        Spec: REQ-PRED-003
        """
        assert isinstance(results_json["hardware_paths"], list)
        assert len(results_json["hardware_paths"]) >= 1

    def test_honest_verdict_keys(self, results_json: dict[str, Any]) -> None:
        """honest_verdict must have all required fields.

        Spec: REQ-PRED-003
        """
        verdict = results_json["honest_verdict"]
        missing = REQUIRED_VERDICT_KEYS - set(verdict.keys())
        assert not missing, f"honest_verdict missing keys: {missing}"

    def test_npu_hardware_info_keys(self, results_json: dict[str, Any]) -> None:
        """npu_hardware_info must report amdxdna driver status and XRT version.

        Spec: REQ-PRED-003
        """
        hw = results_json["npu_hardware_info"]
        missing = REQUIRED_NPU_HW_KEYS - set(hw.keys())
        assert not missing, f"npu_hardware_info missing keys: {missing}"


# ---------------------------------------------------------------------------
# Provider enumeration record
# ---------------------------------------------------------------------------


class TestProviderEnumeration:
    """Validate the provider enumeration record within hardware_paths."""

    def _get_provider_record(self, results_json: dict[str, Any]) -> dict[str, Any]:
        records = [
            r
            for r in results_json["hardware_paths"]
            if r.get("hardware_path") == "provider_enumeration"
        ]
        assert records, "No 'provider_enumeration' hardware_path record found"
        return records[0]

    def test_provider_enum_record_present(self, results_json: dict[str, Any]) -> None:
        """A provider_enumeration record must exist in hardware_paths.

        Spec: REQ-PRED-003
        """
        self._get_provider_record(results_json)  # asserts internally

    def test_provider_enum_required_keys(self, results_json: dict[str, Any]) -> None:
        """provider_enumeration record must have all required keys.

        Spec: REQ-PRED-003
        """
        rec = self._get_provider_record(results_json)
        missing = REQUIRED_PROVIDER_ENUM_KEYS - set(rec.keys())
        assert not missing, f"provider_enumeration missing keys: {missing}"

    def test_available_providers_is_list(self, results_json: dict[str, Any]) -> None:
        """available_providers must be a list of strings.

        Spec: REQ-PRED-003
        """
        rec = self._get_provider_record(results_json)
        providers = rec["available_providers"]
        assert isinstance(providers, list), "available_providers must be a list"
        assert all(isinstance(p, str) for p in providers), "all providers must be strings"

    def test_cpu_provider_always_present(self, results_json: dict[str, Any]) -> None:
        """CPUExecutionProvider must always appear in the enumeration.

        Spec: REQ-PRED-003
        """
        rec = self._get_provider_record(results_json)
        assert "CPUExecutionProvider" in rec["available_providers"], (
            "CPUExecutionProvider must always be listed"
        )


# ---------------------------------------------------------------------------
# Blocker honesty
# ---------------------------------------------------------------------------


class TestBlockerHonesty:
    """Validate that blocked paths have no fabricated throughput numbers."""

    def test_blocked_vitisai_has_null_throughput(self, results_json: dict[str, Any]) -> None:
        """If VitisAI EP is blocked, throughput_calls_per_sec must be null.

        This enforces the 'no fabricated numbers' invariant.

        Spec: REQ-PRED-003
        """
        for rec in results_json["hardware_paths"]:
            if rec.get("status") == "blocker":
                assert rec.get("throughput_calls_per_sec") is None, (
                    f"Blocked path '{rec.get('hardware_path')}' must not have "
                    f"fabricated throughput: {rec.get('throughput_calls_per_sec')}"
                )
                assert rec.get("latency_ms") is None, (
                    f"Blocked path '{rec.get('hardware_path')}' must not have "
                    f"fabricated latency: {rec.get('latency_ms')}"
                )

    def test_blocked_path_has_exact_error(self, results_json: dict[str, Any]) -> None:
        """Every blocked path must name the exact error string.

        Spec: REQ-PRED-003
        """
        for rec in results_json["hardware_paths"]:
            if rec.get("status") == "blocker":
                assert rec.get("exact_error"), (
                    f"Blocked path '{rec.get('hardware_path')}' must have an 'exact_error' field"
                )

    def test_blocked_path_has_next_command(self, results_json: dict[str, Any]) -> None:
        """Every blocked path must provide a concrete next_command to unblock it.

        Spec: REQ-PRED-003
        """
        for rec in results_json["hardware_paths"]:
            if rec.get("status") == "blocker":
                assert rec.get("next_command"), (
                    f"Blocked path '{rec.get('hardware_path')}' must have a 'next_command' field"
                )


# ---------------------------------------------------------------------------
# NPU hardware info
# ---------------------------------------------------------------------------


class TestNpuHardwareInfo:
    """Validate the NPU hardware info section."""

    def test_amdxdna_driver_status_is_bool(self, results_json: dict[str, Any]) -> None:
        """amdxdna_driver_loaded must be a boolean.

        Spec: REQ-PRED-003
        """
        hw = results_json["npu_hardware_info"]
        assert isinstance(hw["amdxdna_driver_loaded"], bool)

    def test_xrt_version_is_string(self, results_json: dict[str, Any]) -> None:
        """xrt_version must be a non-empty string when XRT is present.

        Spec: REQ-PRED-003
        """
        hw = results_json["npu_hardware_info"]
        xrt = hw.get("xrt_version")
        if xrt is not None:
            assert isinstance(xrt, str) and xrt.strip(), "xrt_version must be a non-empty string"

    def test_pci_id_format(self, results_json: dict[str, Any]) -> None:
        """pci_id must be a string in XXXX:XXXX format (or 'unknown').

        Spec: REQ-PRED-003
        """
        hw = results_json["npu_hardware_info"]
        pci_id = hw.get("pci_id", "")
        # Accept 'unknown', or XXXX:XXXX hex format
        import re
        assert pci_id == "unknown" or re.match(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{4}", pci_id), (
            f"pci_id must be 'unknown' or XXXX:XXXX format, got: {pci_id!r}"
        )
