"""Tests for Exp 807: OSS-CAD-Suite Installation and Minimal Ising Synthesis.

Spec traces: REQ-HW-036, REQ-HW-037, SCENARIO-HW-034

All subprocess calls and network I/O are mocked so the suite runs on any machine
without FPGA tools or internet access.
"""

from __future__ import annotations

import json
import subprocess
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts.experiment_807_oss_cad_suite_install import (
    GITHUB_API_URL,
    INSTALL_DIR,
    TOOLS,
    _check_tool,
    _count_luts_from_netlist,
    download_tarball,
    extract_tarball,
    fetch_download_url,
    verify_tools,
    run_synthesis,
)


# ---------------------------------------------------------------------------
# fetch_download_url — REQ-HW-036-1
# ---------------------------------------------------------------------------


class TestFetchDownloadUrl:
    """REQ-HW-036-1: GitHub API asset URL resolution."""

    def _make_response(self, assets: list[dict]) -> MagicMock:
        body = json.dumps({"assets": assets}).encode()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read = MagicMock(return_value=body)
        return cm

    def test_finds_linux_x64_tgz_asset(self):
        """Correct linux-x64 .tgz asset URL is extracted from release payload (REQ-HW-036-1)."""
        assets = [
            {
                "name": "oss-cad-suite-linux-x64-20260424.tgz",
                "browser_download_url": "https://example.com/linux-x64.tgz",
            },
            {
                "name": "oss-cad-suite-darwin-arm64-20260424.tgz",
                "browser_download_url": "https://example.com/darwin.tgz",
            },
        ]
        with patch("urllib.request.urlopen", return_value=self._make_response(assets)):
            url = fetch_download_url()
        assert url == "https://example.com/linux-x64.tgz"

    def test_returns_none_when_no_matching_asset(self):
        """Returns None when release has no linux-x64 .tgz asset."""
        assets = [
            {
                "name": "oss-cad-suite-windows-x64-20260424.zip",
                "browser_download_url": "https://example.com/win.zip",
            }
        ]
        with patch("urllib.request.urlopen", return_value=self._make_response(assets)):
            url = fetch_download_url()
        assert url is None

    def test_returns_none_on_network_error(self):
        """Network failure (OSError) returns None, not an exception (REQ-HW-036-1)."""
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            url = fetch_download_url()
        assert url is None

    def test_skips_non_tgz_linux_x64_asset(self):
        """linux-x64 .zip asset is ignored — only .tgz matches (REQ-HW-036-1)."""
        assets = [
            {
                "name": "oss-cad-suite-linux-x64-20260424.zip",
                "browser_download_url": "https://example.com/linux.zip",
            }
        ]
        with patch("urllib.request.urlopen", return_value=self._make_response(assets)):
            url = fetch_download_url()
        assert url is None


# ---------------------------------------------------------------------------
# _check_tool — REQ-HW-037-1, REQ-HW-037-2
# ---------------------------------------------------------------------------


class TestCheckTool:
    """REQ-HW-037-1, REQ-HW-037-2: tool presence detection via subprocess."""

    def test_tool_present_stdout(self):
        """Tool that exits 0 with stdout is reported as present."""
        mock_result = MagicMock(returncode=0, stdout="yosys 0.45\n", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            present, ver = _check_tool(["yosys", "--version"])
        assert present is True
        assert "yosys" in ver

    def test_tool_present_stderr_only(self):
        """icepack writes usage to stderr and exits 1 — still present (REQ-HW-037-1)."""
        mock_result = MagicMock(returncode=1, stdout="", stderr="Usage: icepack [input]\n")
        with patch("subprocess.run", return_value=mock_result):
            present, ver = _check_tool(["icepack", "--help"])
        assert present is True

    def test_tool_missing_file_not_found(self):
        """FileNotFoundError → tool absent (REQ-HW-037-2)."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            present, ver = _check_tool(["yosys", "--version"])
        assert present is False
        assert "not found" in ver

    def test_tool_timeout(self):
        """TimeoutExpired → tool absent (REQ-HW-037-2)."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["t"], 15)):
            present, ver = _check_tool(["t", "--version"])
        assert present is False
        assert "timeout" in ver


# ---------------------------------------------------------------------------
# verify_tools — REQ-HW-037-3
# ---------------------------------------------------------------------------


class TestVerifyTools:
    """REQ-HW-037-3: version strings recorded for all three tools."""

    def test_all_tools_present(self):
        """All three tools respond → all present=True (SCENARIO-HW-034)."""
        ok = MagicMock(returncode=0, stdout="version 1.0", stderr="")
        with patch("subprocess.run", return_value=ok):
            result = verify_tools()
        for t in TOOLS:
            assert result[t]["present"] is True

    def test_all_tools_missing(self):
        """All FileNotFoundError → all present=False."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = verify_tools()
        for t in TOOLS:
            assert result[t]["present"] is False

    def test_version_string_captured(self):
        """Version output is stored in the 'version' field (REQ-HW-037-3)."""
        ok = MagicMock(returncode=0, stdout="yosys 0.45 (git sha1 abcdef)", stderr="")
        with patch("subprocess.run", return_value=ok):
            result = verify_tools()
        assert "yosys" in result["yosys"]["version"]


# ---------------------------------------------------------------------------
# download_tarball — REQ-HW-036-1
# ---------------------------------------------------------------------------


class TestDownloadTarball:
    """REQ-HW-036-1: curl download."""

    def test_curl_success(self):
        """curl exits 0 → (True, dest_path)."""
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=ok):
            success, msg = download_tarball("https://example.com/oss.tgz")
        assert success is True

    def test_curl_nonzero(self):
        """curl exits non-zero → (False, message)."""
        fail = MagicMock(returncode=1, stdout="", stderr="Connection refused")
        with patch("subprocess.run", return_value=fail):
            success, msg = download_tarball("https://example.com/oss.tgz")
        assert success is False
        assert "curl exited" in msg

    def test_curl_not_found(self):
        """curl missing → (False, message) without raising."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            success, msg = download_tarball("https://example.com/oss.tgz")
        assert success is False
        assert "not found" in msg

    def test_curl_timeout(self):
        """curl timeout → (False, message)."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["curl"], 600)):
            success, msg = download_tarball("https://example.com/oss.tgz")
        assert success is False
        assert "timed out" in msg


# ---------------------------------------------------------------------------
# extract_tarball — REQ-HW-036-2
# ---------------------------------------------------------------------------


class TestExtractTarball:
    """REQ-HW-036-2: tar extraction into target parent dir."""

    def test_tar_success(self, tmp_path):
        """tar exits 0 → (True, install_dir_path)."""
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=ok):
            success, msg = extract_tarball("/tmp/oss-cad.tgz", str(tmp_path))
        assert success is True

    def test_tar_nonzero(self, tmp_path):
        """tar non-zero exit → (False, message)."""
        fail = MagicMock(returncode=1, stdout="", stderr="error: corrupt archive")
        with patch("subprocess.run", return_value=fail):
            success, msg = extract_tarball("/tmp/oss-cad.tgz", str(tmp_path))
        assert success is False
        assert "tar exited" in msg

    def test_tar_timeout(self, tmp_path):
        """tar timeout → (False, message)."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["tar"], 300)):
            success, msg = extract_tarball("/tmp/oss-cad.tgz", str(tmp_path))
        assert success is False
        assert "timed out" in msg


# ---------------------------------------------------------------------------
# _count_luts_from_netlist — REQ-HW-037
# ---------------------------------------------------------------------------


class TestCountLutsFromNetlist:
    """REQ-HW-037: LUT count parsed from yosys JSON netlist."""

    def test_counts_sb_lut4_cells(self, tmp_path):
        """SB_LUT4 cells are counted correctly (SCENARIO-HW-034)."""
        netlist = {
            "modules": {
                "ising2": {
                    "cells": {
                        "c1": {"type": "SB_LUT4"},
                        "c2": {"type": "SB_LUT4"},
                        "c3": {"type": "SB_DFF"},
                    }
                }
            }
        }
        p = tmp_path / "netlist.json"
        p.write_text(json.dumps(netlist))
        assert _count_luts_from_netlist(p) == 2

    def test_zero_luts_if_no_sb_lut4(self, tmp_path):
        """Netlist with no SB_LUT4 cells returns 0."""
        netlist = {"modules": {"ising2": {"cells": {"c1": {"type": "SB_DFF"}}}}}
        p = tmp_path / "netlist.json"
        p.write_text(json.dumps(netlist))
        assert _count_luts_from_netlist(p) == 0

    def test_zero_on_missing_file(self, tmp_path):
        """Missing netlist file returns 0 (not an exception)."""
        assert _count_luts_from_netlist(tmp_path / "missing.json") == 0

    def test_zero_on_malformed_json(self, tmp_path):
        """Malformed JSON returns 0 (not an exception)."""
        p = tmp_path / "bad.json"
        p.write_text("not json {{")
        assert _count_luts_from_netlist(p) == 0


# ---------------------------------------------------------------------------
# run_synthesis — SCENARIO-HW-034
# ---------------------------------------------------------------------------


class TestRunSynthesis:
    """SCENARIO-HW-034: minimal synthesis execution and result parsing."""

    def test_synthesis_clean(self, tmp_path):
        """yosys exits 0 without errors → success=True, lut_count extracted."""
        netlist = {
            "modules": {
                "ising2": {
                    "cells": {
                        "c1": {"type": "SB_LUT4"},
                        "c2": {"type": "SB_LUT4"},
                    }
                }
            }
        }

        def fake_run(cmd, **kwargs):
            # Write netlist JSON to the path specified in the -p script arg.
            for arg in cmd:
                if "write_json" in str(arg):
                    json_path = str(arg).split("write_json")[-1].strip()
                    Path(json_path).write_text(json.dumps(netlist))
            return MagicMock(returncode=0, stdout="End of script.\n", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            result = run_synthesis()
        assert result["success"] is True
        assert result["lut_count"] == 2

    def test_synthesis_error_in_stderr(self):
        """ERROR in stderr + nonzero exit → success=False (SCENARIO-HW-034)."""
        mock_result = MagicMock(returncode=1, stdout="", stderr="ERROR: syntax error")
        with patch("subprocess.run", return_value=mock_result):
            result = run_synthesis()
        assert result["success"] is False
        assert result["lut_count"] is None

    def test_synthesis_yosys_not_found(self):
        """FileNotFoundError from yosys → success=False without raising."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_synthesis()
        assert result["success"] is False
        assert "not found" in result["stderr_snippet"]

    def test_synthesis_timeout(self):
        """Timeout → success=False, stderr_snippet mentions timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["yosys"], 120)):
            result = run_synthesis()
        assert result["success"] is False
        assert "timed out" in result["stderr_snippet"]
