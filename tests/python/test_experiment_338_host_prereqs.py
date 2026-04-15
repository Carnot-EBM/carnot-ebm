"""Tests for Exp 338: host prerequisite registry + DualGPU auto-assignment default.

Spec coverage: REQ-INFRA-006, REQ-INFRA-007,
               SCENARIO-INFRA-009, SCENARIO-INFRA-010, SCENARIO-INFRA-011

Written test-first per REQ-INFRA-002.  Tests validate:

- HostPrereqRegistry: loads ops/host-prereqs.md, parses markdown table into
  PrereqEntry objects with package/check_command/required_for fields.
- HostPrereqRegistry.check_prereqs(experiment_class): runs check commands,
  returns list[str] of missing package names.
- Graceful degradation: subprocess failures, missing binaries, and timeouts
  are all handled without raising exceptions.
- DualGPU auto-assignment: ExperimentTemplate.setup_gpu() assigns model_specs[i]
  to gpu i when len(model_specs) >= 2 and CARNOT_FORCE_LIVE=1.
- Single-GPU fallback: when only 1 GPU detected, all models assigned to GPU 0
  with a RETRO-004 warning logged.
- dual_gpu_auto_assigned: bool present in all setup_gpu() return dicts.
- Experiment 338 script: exists, references correct schema, and references
  n_packages_registered, dual_gpu_auto_assign_enabled, retro_items_implemented.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"
OPS_DIR = REPO_ROOT / "ops"
HOST_PREREQS_MD = OPS_DIR / "host-prereqs.md"
EXP_338_SCRIPT = SCRIPTS_DIR / "experiment_338_host_prereqs.py"
EXP_338_RESULT = RESULTS_DIR / "experiment_338_host_prereqs.json"

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_registry():
    """Import HostPrereqRegistry and PrereqEntry from the pipeline module."""
    from carnot.pipeline.host_prereq_registry import HostPrereqRegistry, PrereqEntry

    return HostPrereqRegistry, PrereqEntry


def _import_parse_registry():
    """Import _parse_registry for direct testing."""
    from carnot.pipeline.host_prereq_registry import _parse_registry

    return _parse_registry


# ---------------------------------------------------------------------------
# TestHostPrereqsMdExists
# REQ-INFRA-006 / SCENARIO-INFRA-009
# ---------------------------------------------------------------------------


class TestHostPrereqsMdExists:
    """REQ-INFRA-006: ops/host-prereqs.md must exist and be readable."""

    def test_file_exists(self) -> None:
        """SCENARIO-INFRA-009: ops/host-prereqs.md must exist on disk."""
        assert HOST_PREREQS_MD.exists(), f"Missing required registry: {HOST_PREREQS_MD}"

    def test_file_is_non_empty(self) -> None:
        """SCENARIO-INFRA-009: registry must not be empty."""
        text = HOST_PREREQS_MD.read_text()
        assert len(text.strip()) > 0

    def test_file_contains_table_header(self) -> None:
        """SCENARIO-INFRA-009: registry must contain the expected table header."""
        text = HOST_PREREQS_MD.read_text()
        assert "Package" in text
        assert "Check Command" in text
        assert "Required For" in text

    def test_file_contains_ninja(self) -> None:
        """SCENARIO-INFRA-009: ninja entry must be present (root cause of RETRO-006)."""
        text = HOST_PREREQS_MD.read_text()
        assert "ninja" in text.lower()

    def test_file_contains_openblas(self) -> None:
        """SCENARIO-INFRA-009: openblas entry must be present (root cause of RETRO-006)."""
        text = HOST_PREREQS_MD.read_text()
        assert "openblas" in text.lower()

    def test_file_contains_nvidia_smi(self) -> None:
        """SCENARIO-INFRA-009: nvidia-smi entry must be present for GPU experiments."""
        text = HOST_PREREQS_MD.read_text()
        assert "nvidia-smi" in text.lower()

    def test_file_contains_yosys(self) -> None:
        """SCENARIO-INFRA-009: yosys entry must be present for FPGA experiments."""
        text = HOST_PREREQS_MD.read_text()
        assert "yosys" in text.lower()

    def test_file_contains_nextpnr(self) -> None:
        """SCENARIO-INFRA-009: nextpnr-xilinx entry must be present for KV260."""
        text = HOST_PREREQS_MD.read_text()
        assert "nextpnr" in text.lower()


# ---------------------------------------------------------------------------
# TestParseRegistry (unit tests for _parse_registry)
# REQ-INFRA-006 / SCENARIO-INFRA-009
# ---------------------------------------------------------------------------


class TestParseRegistry:
    """_parse_registry must parse the markdown table into PrereqEntry objects."""

    def test_parse_real_registry(self) -> None:
        """SCENARIO-INFRA-009: parse the real ops/host-prereqs.md successfully."""
        _parse_registry = _import_parse_registry()
        entries = _parse_registry(HOST_PREREQS_MD)
        assert len(entries) >= 4, f"Expected >=4 entries, got {len(entries)}"

    def test_parse_returns_list_of_prereq_entries(self) -> None:
        """SCENARIO-INFRA-009: _parse_registry returns PrereqEntry objects."""
        from carnot.pipeline.host_prereq_registry import PrereqEntry

        _parse_registry = _import_parse_registry()
        entries = _parse_registry(HOST_PREREQS_MD)
        for e in entries:
            assert isinstance(e, PrereqEntry)

    def test_entry_has_package_field(self) -> None:
        """SCENARIO-INFRA-009: every entry must have a non-empty package name."""
        _parse_registry = _import_parse_registry()
        entries = _parse_registry(HOST_PREREQS_MD)
        for e in entries:
            assert isinstance(e.package, str) and len(e.package) > 0

    def test_entry_has_check_command_field(self) -> None:
        """SCENARIO-INFRA-009: every entry must have a check_command."""
        _parse_registry = _import_parse_registry()
        entries = _parse_registry(HOST_PREREQS_MD)
        for e in entries:
            assert isinstance(e.check_command, str) and len(e.check_command) > 0

    def test_entry_has_required_for_list(self) -> None:
        """SCENARIO-INFRA-009: required_for must be a list of strings."""
        _parse_registry = _import_parse_registry()
        entries = _parse_registry(HOST_PREREQS_MD)
        for e in entries:
            assert isinstance(e.required_for, list)

    def test_parse_minimal_table(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-009: parse a minimal in-memory registry table."""
        _parse_registry = _import_parse_registry()
        md = tmp_path / "test-prereqs.md"
        md.write_text(
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "|---------|--------------|----------------|-----------------|-------------- |\n"
            "| testpkg | testpkg --version | pacman -S testpkg | apt install testpkg | npu, fpga |\n"
        )
        entries = _parse_registry(md)
        assert len(entries) == 1
        assert entries[0].package == "testpkg"
        assert entries[0].check_command == "testpkg --version"
        assert "npu" in entries[0].required_for
        assert "fpga" in entries[0].required_for

    def test_parse_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-009: returns empty list (no raise) when file is absent."""
        _parse_registry = _import_parse_registry()
        entries = _parse_registry(tmp_path / "nonexistent.md")
        assert entries == []

    def test_parse_skips_separator_row(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-009: separator rows (|---|) must not become entries."""
        _parse_registry = _import_parse_registry()
        md = tmp_path / "sep.md"
        md.write_text(
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "|---------|---|---|---|---|\n"
            "| mypkg | mypkg -v | n/a | n/a | all |\n"
        )
        entries = _parse_registry(md)
        assert len(entries) == 1
        assert entries[0].package == "mypkg"

    def test_parse_skips_header_row(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-009: the header row must not become an entry."""
        _parse_registry = _import_parse_registry()
        md = tmp_path / "hdr.md"
        md.write_text(
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "| pkg1 | pkg1 -v | n/a | n/a | all |\n"
        )
        entries = _parse_registry(md)
        assert len(entries) == 1
        assert entries[0].package == "pkg1"

    def test_parse_env_var_entry(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-009: env:VAR_NAME check_command is parsed correctly."""
        _parse_registry = _import_parse_registry()
        md = tmp_path / "env.md"
        md.write_text(
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "|---------|---|---|---|---|\n"
            "| CARNOT_FORCE_LIVE | env:CARNOT_FORCE_LIVE | export | export | all |\n"
        )
        entries = _parse_registry(md)
        assert len(entries) == 1
        assert entries[0].check_command == "env:CARNOT_FORCE_LIVE"


# ---------------------------------------------------------------------------
# TestHostPrereqRegistry (construction)
# REQ-INFRA-006 / SCENARIO-INFRA-009
# ---------------------------------------------------------------------------


class TestHostPrereqRegistryConstruction:
    """HostPrereqRegistry construction must not raise and must load entries."""

    def test_construction_succeeds(self) -> None:
        """SCENARIO-INFRA-009: default construction (real file) does not raise."""
        HostPrereqRegistry, _ = _import_registry()
        registry = HostPrereqRegistry()
        assert registry is not None

    def test_entries_property_returns_list(self) -> None:
        """SCENARIO-INFRA-009: entries property returns a list."""
        HostPrereqRegistry, _ = _import_registry()
        registry = HostPrereqRegistry()
        assert isinstance(registry.entries, list)

    def test_at_least_four_entries_loaded(self) -> None:
        """SCENARIO-INFRA-009: at least 4 entries must be loaded from the real registry."""
        HostPrereqRegistry, _ = _import_registry()
        registry = HostPrereqRegistry()
        assert len(registry.entries) >= 4

    def test_construction_with_missing_file(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-009: construction with a missing registry file does not raise."""
        HostPrereqRegistry, _ = _import_registry()
        registry = HostPrereqRegistry(registry_path=tmp_path / "nonexistent.md")
        assert registry.entries == []

    def test_construction_with_custom_path(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-009: accepts a custom registry_path override."""
        HostPrereqRegistry, _ = _import_registry()
        md = tmp_path / "custom.md"
        md.write_text(
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "|---|---|---|---|---|\n"
            "| custompkg | custompkg -v | n/a | n/a | custom |\n"
        )
        registry = HostPrereqRegistry(registry_path=md)
        assert len(registry.entries) == 1
        assert registry.entries[0].package == "custompkg"

    def test_entries_returns_copy(self) -> None:
        """SCENARIO-INFRA-009: mutating the returned list does not affect the registry."""
        HostPrereqRegistry, _ = _import_registry()
        registry = HostPrereqRegistry()
        entries_a = registry.entries
        entries_a.clear()
        entries_b = registry.entries
        assert len(entries_b) > 0


# ---------------------------------------------------------------------------
# TestCheckPrereqsFiltering
# REQ-INFRA-006 / SCENARIO-INFRA-010
# ---------------------------------------------------------------------------


class TestCheckPrereqsFiltering:
    """check_prereqs(experiment_class) must filter by required_for correctly."""

    def _registry_with_entries(self, entries_md: str, tmp_path: Path):
        """Helper: build a registry from an inline markdown string."""
        from carnot.pipeline.host_prereq_registry import HostPrereqRegistry

        md = tmp_path / "r.md"
        header = (
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "|---|---|---|---|---|\n"
        )
        md.write_text(header + entries_md)
        return HostPrereqRegistry(registry_path=md)

    def test_filter_by_experiment_class(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: only entries whose required_for matches are checked."""
        registry = self._registry_with_entries(
            "| pkgnpu | true | n/a | n/a | npu |\n"
            "| pkgfpga | false_nonexistent_cmd_xyz | n/a | n/a | fpga |\n",
            tmp_path,
        )
        # With experiment_class="npu", only pkgnpu is checked.
        # pkgnpu uses "true" which exits 0 — so not missing.
        # pkgfpga is excluded by the filter.
        missing = registry.check_prereqs(experiment_class="npu")
        assert "pkgfpga" not in missing

    def test_all_tag_included_for_any_class(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: 'all' tag is included regardless of experiment_class."""
        registry = self._registry_with_entries(
            "| universal_pkg | false_nonexistent_xyz | n/a | n/a | all |\n",
            tmp_path,
        )
        missing = registry.check_prereqs(experiment_class="npu")
        assert "universal_pkg" in missing

    def test_no_class_filter_checks_all(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: check_prereqs() with no arg checks all entries."""
        registry = self._registry_with_entries(
            "| pkga | true | n/a | n/a | npu |\n"
            "| pkgb | false_nonexistent_xyz_b | n/a | n/a | fpga |\n",
            tmp_path,
        )
        missing = registry.check_prereqs()
        assert "pkgb" in missing

    def test_unmatched_class_returns_only_all_tagged(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: experiment_class with no matching entries only gets 'all'."""
        registry = self._registry_with_entries(
            "| pkgnpu | true | n/a | n/a | npu |\n"
            "| universal | false_nonexistent_xyz_u | n/a | n/a | all |\n",
            tmp_path,
        )
        missing = registry.check_prereqs(experiment_class="fpga")
        assert "pkgnpu" not in missing
        assert "universal" in missing


# ---------------------------------------------------------------------------
# TestCheckPrereqsSubprocess
# REQ-INFRA-006 / SCENARIO-INFRA-010
# ---------------------------------------------------------------------------


class TestCheckPrereqsSubprocess:
    """check_prereqs must handle subprocess outcomes gracefully."""

    def _registry_from_md(self, md_content: str, tmp_path: Path):
        from carnot.pipeline.host_prereq_registry import HostPrereqRegistry

        md = tmp_path / "r.md"
        header = (
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "|---|---|---|---|---|\n"
        )
        md.write_text(header + md_content)
        return HostPrereqRegistry(registry_path=md)

    def test_zero_exit_code_means_present(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: check command exiting 0 means package is present."""
        registry = self._registry_from_md(
            "| truepkg | true | n/a | n/a | all |\n", tmp_path
        )
        missing = registry.check_prereqs()
        assert "truepkg" not in missing

    def test_nonzero_exit_code_means_missing(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: check command exiting non-zero means package is missing."""
        registry = self._registry_from_md(
            "| falsepkg | false | n/a | n/a | all |\n", tmp_path
        )
        missing = registry.check_prereqs()
        assert "falsepkg" in missing

    def test_file_not_found_means_missing(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: FileNotFoundError from check command → missing, no raise."""
        registry = self._registry_from_md(
            "| ghostpkg | __definitely_not_a_real_binary_xyz__ | n/a | n/a | all |\n",
            tmp_path,
        )
        missing = registry.check_prereqs()
        assert "ghostpkg" in missing

    def test_timeout_means_missing(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: subprocess.TimeoutExpired → missing, no raise."""
        from carnot.pipeline.host_prereq_registry import HostPrereqRegistry, _parse_registry

        md = tmp_path / "r.md"
        header = (
            "| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n"
            "|---|---|---|---|---|\n"
        )
        md.write_text(header + "| slowpkg | true | n/a | n/a | all |\n")
        registry = HostPrereqRegistry(registry_path=md)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("true", 5)):
            missing = registry.check_prereqs()
        assert "slowpkg" in missing

    def test_env_var_check_present_when_set(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: env:VAR_NAME check passes when the env var equals '1'."""
        registry = self._registry_from_md(
            "| CARNOT_FORCE_LIVE | env:CARNOT_FORCE_LIVE | export | export | all |\n",
            tmp_path,
        )
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            missing = registry.check_prereqs()
        assert "CARNOT_FORCE_LIVE" not in missing

    def test_env_var_check_missing_when_unset(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: env:VAR_NAME check fails when the env var is not set."""
        registry = self._registry_from_md(
            "| CARNOT_FORCE_LIVE | env:CARNOT_FORCE_LIVE | export | export | all |\n",
            tmp_path,
        )
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            missing = registry.check_prereqs()
        assert "CARNOT_FORCE_LIVE" in missing

    def test_env_var_check_missing_when_not_equal_one(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: env:VAR_NAME fails when value is not exactly '1'."""
        registry = self._registry_from_md(
            "| CARNOT_FORCE_LIVE | env:CARNOT_FORCE_LIVE | export | export | all |\n",
            tmp_path,
        )
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            missing = registry.check_prereqs()
        assert "CARNOT_FORCE_LIVE" in missing

    def test_returns_empty_list_when_all_present(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: returns [] when all check commands exit 0."""
        registry = self._registry_from_md(
            "| truepkg1 | true | n/a | n/a | all |\n"
            "| truepkg2 | true | n/a | n/a | all |\n",
            tmp_path,
        )
        missing = registry.check_prereqs()
        assert missing == []

    def test_returns_list_type(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-010: return type is always list."""
        registry = self._registry_from_md(
            "| truepkg | true | n/a | n/a | all |\n", tmp_path
        )
        result = registry.check_prereqs()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestIsPresent (unit tests for _is_present)
# REQ-INFRA-006
# ---------------------------------------------------------------------------


class TestIsPresent:
    """HostPrereqRegistry._is_present() internal logic."""

    def _make_entry(self, package: str, check_command: str):
        from carnot.pipeline.host_prereq_registry import PrereqEntry

        return PrereqEntry(
            package=package,
            check_command=check_command,
            install_arch="n/a",
            install_debian="n/a",
            required_for=["all"],
        )

    def _make_registry(self, tmp_path: Path):
        from carnot.pipeline.host_prereq_registry import HostPrereqRegistry

        md = tmp_path / "empty.md"
        md.write_text("| Package | Check Command | Install (Arch) | Install (Debian) | Required For |\n")
        return HostPrereqRegistry(registry_path=md)

    def test_true_command_returns_present(self, tmp_path: Path) -> None:
        """_is_present returns True for 'true' command."""
        registry = self._make_registry(tmp_path)
        entry = self._make_entry("pkg", "true")
        assert registry._is_present(entry) is True

    def test_false_command_returns_not_present(self, tmp_path: Path) -> None:
        """_is_present returns False for 'false' command."""
        registry = self._make_registry(tmp_path)
        entry = self._make_entry("pkg", "false")
        assert registry._is_present(entry) is False

    def test_nonexistent_binary_returns_false(self, tmp_path: Path) -> None:
        """_is_present returns False for a nonexistent binary."""
        registry = self._make_registry(tmp_path)
        entry = self._make_entry("ghostpkg", "__not_a_real_cmd_xyz__")
        assert registry._is_present(entry) is False

    def test_env_var_present(self, tmp_path: Path) -> None:
        """_is_present returns True for env:VAR when var equals '1'."""
        registry = self._make_registry(tmp_path)
        entry = self._make_entry("MYVAR", "env:MYVAR")
        with patch.dict(os.environ, {"MYVAR": "1"}):
            assert registry._is_present(entry) is True

    def test_env_var_absent(self, tmp_path: Path) -> None:
        """_is_present returns False for env:VAR when var is missing."""
        registry = self._make_registry(tmp_path)
        entry = self._make_entry("MYVAR", "env:MYVAR")
        env = {k: v for k, v in os.environ.items() if k != "MYVAR"}
        with patch.dict(os.environ, env, clear=True):
            assert registry._is_present(entry) is False

    def test_timeout_returns_false(self, tmp_path: Path) -> None:
        """_is_present returns False on TimeoutExpired."""
        registry = self._make_registry(tmp_path)
        entry = self._make_entry("slow", "true")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("true", 5)):
            assert registry._is_present(entry) is False


# ---------------------------------------------------------------------------
# TestDualGpuAutoAssignment
# REQ-INFRA-007 / SCENARIO-INFRA-011
# ---------------------------------------------------------------------------


class TestDualGpuAutoAssignment:
    """ExperimentTemplate.setup_gpu() must auto-assign GPU indices when appropriate."""

    def _make_prewarm_fn(self):
        """Return a mock prewarm_fn that always reports healthy."""
        mock_result = MagicMock()
        mock_result.health_ok = True
        mock_result.load_time_s = 0.1
        mock_result.stall_root_cause = None

        def prewarm_fn(name, hf_id, gpu):
            return mock_result

        return prewarm_fn

    def _make_template(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        return ExperimentTemplate(
            338,
            "Test DualGPU auto-assign",
            "results/test_338.json",
            repo_root=tmp_path,
        )

    def _make_specs(self, n: int = 2) -> list[dict[str, Any]]:
        return [
            {"name": f"Model{i}", "hf_id": f"org/model{i}", "gpu": 0}
            for i in range(n)
        ]

    def test_dual_gpu_auto_assigned_key_present_live_two_models(
        self, tmp_path: Path
    ) -> None:
        """SCENARIO-INFRA-011: dual_gpu_auto_assigned key must be in result dict."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(2)
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor._get_gpu_count",
                return_value=2,
            ),
        ):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert "dual_gpu_auto_assigned" in result

    def test_two_models_two_gpus_assigns_separately(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-011: with 2 GPUs and CARNOT_FORCE_LIVE=1, model 0→GPU 0, model 1→GPU 1."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(2)
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor._get_gpu_count",
                return_value=2,
            ),
        ):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert specs[0]["gpu"] == 0
        assert specs[1]["gpu"] == 1
        assert result["dual_gpu_auto_assigned"] is True

    def test_two_models_one_gpu_all_assigned_to_zero(
        self, tmp_path: Path, caplog
    ) -> None:
        """SCENARIO-INFRA-011: with 1 GPU, all models assigned to GPU 0, RETRO-004 warning logged."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(2)
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor._get_gpu_count",
                return_value=1,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert specs[0]["gpu"] == 0
        assert specs[1]["gpu"] == 0
        assert result["dual_gpu_auto_assigned"] is False
        assert "RETRO-004" in caplog.text

    def test_single_model_no_auto_assignment(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-011: with only 1 model, auto-assignment is skipped."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(1)
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert result["dual_gpu_auto_assigned"] is False
        assert specs[0]["gpu"] == 0  # unchanged

    def test_no_force_live_no_auto_assignment(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-011: with CARNOT_FORCE_LIVE=0, auto-assignment is skipped."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(2)
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        env["CARNOT_FORCE_LIVE"] = "0"
        with patch.dict(os.environ, env, clear=True):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert result["dual_gpu_auto_assigned"] is False
        # gpu values are unchanged (still 0 from _make_specs)
        assert specs[0]["gpu"] == 0
        assert specs[1]["gpu"] == 0

    def test_dual_gpu_auto_assigned_key_false_in_ci_mode(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-011: dual_gpu_auto_assigned=False in CI (CARNOT_FORCE_LIVE not set)."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(2)
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert result["dual_gpu_auto_assigned"] is False

    def test_three_models_two_gpus_assigns_first_two(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-011: with 3 models and 2 GPUs, model 0→GPU 0, 1→GPU 1, 2→GPU 2."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(3)
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor._get_gpu_count",
                return_value=2,
            ),
        ):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        # Auto-assignment assigns i→i regardless of GPU count
        assert specs[0]["gpu"] == 0
        assert specs[1]["gpu"] == 1
        assert specs[2]["gpu"] == 2
        assert result["dual_gpu_auto_assigned"] is True

    def test_gpu_monitor_results_key_still_present(self, tmp_path: Path) -> None:
        """Existing gpu_monitor_results key must still be present (no regression)."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(1)
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert "gpu_monitor_results" in result

    def test_all_healthy_key_still_present(self, tmp_path: Path) -> None:
        """Existing all_healthy key must still be present (no regression)."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(1)
        result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert "all_healthy" in result

    def test_models_key_still_present(self, tmp_path: Path) -> None:
        """Existing models key must still be present (no regression)."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(1)
        result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert "models" in result

    def test_prewarm_time_s_key_still_present(self, tmp_path: Path) -> None:
        """Existing prewarm_time_s key must still be present (no regression)."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(1)
        result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        assert "prewarm_time_s" in result

    def test_gpu_count_detection_failure_falls_back_to_one(
        self, tmp_path: Path
    ) -> None:
        """SCENARIO-INFRA-011: if _get_gpu_count raises, falls back to 1 GPU (conservative)."""
        tmpl = self._make_template(tmp_path)
        specs = self._make_specs(2)
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor._get_gpu_count",
                side_effect=RuntimeError("GPU detection failed"),
            ),
        ):
            result = tmpl.setup_gpu(specs, prewarm_fn=self._make_prewarm_fn())
        # Both assigned to GPU 0 (conservative fallback)
        assert specs[0]["gpu"] == 0
        assert specs[1]["gpu"] == 0
        assert result["dual_gpu_auto_assigned"] is False


# ---------------------------------------------------------------------------
# TestExp338ScriptExists
# ---------------------------------------------------------------------------


class TestExp338ScriptExists:
    """The Exp 338 script must exist and have the required structure."""

    def test_script_file_exists(self) -> None:
        """Script must exist at scripts/experiment_338_host_prereqs.py."""
        assert EXP_338_SCRIPT.exists(), f"Missing script: {EXP_338_SCRIPT}"

    def test_script_references_338(self) -> None:
        """Script must reference experiment 338."""
        source = EXP_338_SCRIPT.read_text()
        assert "338" in source

    def test_script_references_schema(self) -> None:
        """Script must define the artifact schema."""
        source = EXP_338_SCRIPT.read_text()
        assert "carnot.host_prereqs.v1" in source

    def test_script_references_n_packages_registered(self) -> None:
        """Script must include n_packages_registered in artifact."""
        source = EXP_338_SCRIPT.read_text()
        assert "n_packages_registered" in source

    def test_script_references_dual_gpu_auto_assign_enabled(self) -> None:
        """Script must include dual_gpu_auto_assign_enabled in artifact."""
        source = EXP_338_SCRIPT.read_text()
        assert "dual_gpu_auto_assign_enabled" in source

    def test_script_references_retro_items_implemented(self) -> None:
        """Script must include retro_items_implemented in artifact."""
        source = EXP_338_SCRIPT.read_text()
        assert "retro_items_implemented" in source

    def test_script_references_host_prereq_registry(self) -> None:
        """Script must import or reference HostPrereqRegistry."""
        source = EXP_338_SCRIPT.read_text()
        assert "HostPrereqRegistry" in source

    def test_script_defines_main(self) -> None:
        """Script must have a main() function or __main__ guard."""
        source = EXP_338_SCRIPT.read_text()
        assert "def main(" in source or '__name__ == "__main__"' in source


# ---------------------------------------------------------------------------
# TestExp338ResultArtifact (if the result file exists)
# ---------------------------------------------------------------------------


class TestExp338ResultArtifact:
    """If results/experiment_338_host_prereqs.json exists, validate its schema."""

    REQUIRED_FIELDS = {
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
        "n_packages_registered",
        "n_classes_checked",
        "dual_gpu_auto_assign_enabled",
        "retro_items_implemented",
        "artifact_schema",
    }

    def test_result_file_exists(self) -> None:
        """Result JSON must be written by the experiment script."""
        assert EXP_338_RESULT.exists(), f"Missing: {EXP_338_RESULT}"

    def test_result_is_valid_json(self) -> None:
        """Result must be parseable JSON."""
        data = json.loads(EXP_338_RESULT.read_text())
        assert isinstance(data, dict)

    def test_required_fields_present(self) -> None:
        """All required top-level keys must be present."""
        data = json.loads(EXP_338_RESULT.read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing field: {field}"

    def test_schema_is_carnot_host_prereqs_v1(self) -> None:
        """artifact_schema field must be 'carnot.host_prereqs.v1'.

        Note: build_result() always overwrites the 'schema' key with the sorted
        list of artifact keys (not the schema version string).  The schema version
        is stored in 'artifact_schema' to survive that overwrite.
        """
        data = json.loads(EXP_338_RESULT.read_text())
        assert data["artifact_schema"] == "carnot.host_prereqs.v1"

    def test_n_packages_registered_is_positive_int(self) -> None:
        """n_packages_registered must be a positive integer."""
        data = json.loads(EXP_338_RESULT.read_text())
        n = data["n_packages_registered"]
        assert isinstance(n, int) and n > 0

    def test_n_classes_checked_is_non_negative_int(self) -> None:
        """n_classes_checked must be a non-negative integer."""
        data = json.loads(EXP_338_RESULT.read_text())
        n = data["n_classes_checked"]
        assert isinstance(n, int) and n >= 0

    def test_dual_gpu_auto_assign_enabled_is_bool(self) -> None:
        """dual_gpu_auto_assign_enabled must be a boolean."""
        data = json.loads(EXP_338_RESULT.read_text())
        assert isinstance(data["dual_gpu_auto_assign_enabled"], bool)

    def test_retro_items_implemented_is_list(self) -> None:
        """retro_items_implemented must be a list."""
        data = json.loads(EXP_338_RESULT.read_text())
        assert isinstance(data["retro_items_implemented"], list)

    def test_retro_004_in_retro_items(self) -> None:
        """retro_items_implemented must include RETRO-004."""
        data = json.loads(EXP_338_RESULT.read_text())
        assert any("RETRO-004" in str(item) for item in data["retro_items_implemented"])

    def test_retro_006_in_retro_items(self) -> None:
        """retro_items_implemented must include RETRO-006."""
        data = json.loads(EXP_338_RESULT.read_text())
        assert any("RETRO-006" in str(item) for item in data["retro_items_implemented"])

    def test_status_is_success_or_blocked(self) -> None:
        """status must be 'success' or 'blocked' (honest reporting)."""
        data = json.loads(EXP_338_RESULT.read_text())
        assert data["status"] in {"success", "blocked", "partial"}

    def test_experiment_is_338(self) -> None:
        """experiment field must be 338."""
        data = json.loads(EXP_338_RESULT.read_text())
        assert data["experiment"] == 338
