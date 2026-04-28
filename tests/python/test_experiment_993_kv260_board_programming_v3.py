"""Tests for experiment_993_kv260_board_programming_v3.

Spec refs: REQ-HW-040, SCENARIO-HW-040

**Why these tests exist:**
    Exp 993 is the first run where the KV260 board is actually reachable (at
    192.168.51.98 via kv260.local mDNS), but SSH auth is not set up yet, so
    board_programmed=False and human_action_required=True.  These tests verify:
      1. Discovery functions work correctly when board is reachable vs not.
      2. CPU baseline always returns a positive value.
      3. Result JSON is always written with all required schema fields.
      4. The honest_verdict progression is correct for each partial state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import experiment_993_kv260_board_programming_v3 as exp993  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests: board discovery helpers
# ---------------------------------------------------------------------------


class TestTcpSshProbe:
    """REQ-HW-040: Board must be discoverable via TCP SSH probe."""

    def test_returns_ip_when_port_open(self) -> None:
        """Returns the resolved IP string when TCP port 22 is reachable."""
        with (
            mock.patch("socket.getaddrinfo") as mock_gai,
            mock.patch("socket.create_connection") as mock_cc,
        ):
            mock_gai.return_value = [(None, None, None, None, ("192.168.51.98", 22))]
            mock_cc.return_value.__enter__ = mock.Mock(return_value=None)
            mock_cc.return_value.__exit__ = mock.Mock(return_value=False)
            result = exp993._tcp_ssh_probe("kv260.local")
        assert result == "192.168.51.98"

    def test_returns_none_when_connection_refused(self) -> None:
        """Returns None when the TCP connection is refused."""
        with (
            mock.patch("socket.getaddrinfo") as mock_gai,
            mock.patch("socket.create_connection", side_effect=OSError("refused")),
        ):
            mock_gai.return_value = [(None, None, None, None, ("192.168.51.98", 22))]
            result = exp993._tcp_ssh_probe("kv260.local")
        assert result is None

    def test_returns_none_when_dns_fails(self) -> None:
        """Returns None when the hostname cannot be resolved."""
        import socket as _socket

        with mock.patch("socket.getaddrinfo", side_effect=_socket.gaierror("no name")):
            result = exp993._tcp_ssh_probe("kv260.nonexistent")
        assert result is None


class TestArpLookup:
    """REQ-HW-040: ARP table fallback discovery."""

    def test_finds_ip_from_kria_line(self) -> None:
        """Extracts IP from an ARP line containing 'kria'."""
        arp_output = "kria.local (192.168.51.98) at aa:bb:cc:dd:ee:ff [ether] on eth0\n"
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=arp_output)
            result = exp993._arp_lookup()
        assert result == "192.168.51.98"

    def test_returns_none_when_no_kria_in_arp(self) -> None:
        """Returns None when no Kria/KV260/Xilinx entries in ARP table."""
        arp_output = "router.local (192.168.1.1) at aa:bb:cc:dd:ee:ff [ether] on eth0\n"
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=arp_output)
            result = exp993._arp_lookup()
        assert result is None


class TestDiscoverBoard:
    """REQ-HW-040: Multi-method board discovery must find board when any method works."""

    def test_finds_board_via_direct_probe(self) -> None:
        """board_discovered=True and board_ip returned when TCP probe succeeds."""
        with mock.patch.object(exp993, "_tcp_ssh_probe", return_value="192.168.51.98"):
            discovered, ip = exp993._discover_board()
        assert discovered is True
        assert ip == "192.168.51.98"

    def test_falls_back_to_arp_when_direct_probe_fails(self) -> None:
        """Falls back to ARP lookup when direct TCP probe fails."""
        with (
            mock.patch.object(exp993, "_tcp_ssh_probe", return_value=None),
            mock.patch.object(exp993, "_arp_lookup", return_value="10.0.0.5"),
            mock.patch.object(exp993, "_avahi_lookup", return_value=None),
        ):
            discovered, ip = exp993._discover_board()
        assert discovered is True
        assert ip == "10.0.0.5"

    def test_falls_back_to_avahi_when_arp_fails(self) -> None:
        """Falls back to avahi-browse when both direct probe and ARP fail."""
        with (
            mock.patch.object(exp993, "_tcp_ssh_probe", return_value=None),
            mock.patch.object(exp993, "_arp_lookup", return_value=None),
            mock.patch.object(exp993, "_avahi_lookup", return_value="10.0.0.7"),
        ):
            discovered, ip = exp993._discover_board()
        assert discovered is True
        assert ip == "10.0.0.7"

    def test_returns_not_found_when_all_methods_fail(self) -> None:
        """Returns (False, None) when all discovery methods fail."""
        with (
            mock.patch.object(exp993, "_tcp_ssh_probe", return_value=None),
            mock.patch.object(exp993, "_arp_lookup", return_value=None),
            mock.patch.object(exp993, "_avahi_lookup", return_value=None),
        ):
            discovered, ip = exp993._discover_board()
        assert discovered is False
        assert ip is None


# ---------------------------------------------------------------------------
# Unit tests: CPU baseline
# ---------------------------------------------------------------------------


class TestCpuBaseline:
    """REQ-HW-040: CPU baseline must produce a positive timing value."""

    def test_returns_positive_float(self) -> None:
        """The E-MVL EMA Ising sweep returns a positive microseconds value."""
        us = exp993._cpu_baseline_latency_us()
        assert isinstance(us, float)
        assert us > 0.0

    def test_two_calls_in_same_ballpark(self) -> None:
        """Two back-to-back calls return similar values (no more than 10x apart)."""
        us1 = exp993._cpu_baseline_latency_us()
        us2 = exp993._cpu_baseline_latency_us()
        assert us1 > 0.0 and us2 > 0.0
        ratio = max(us1, us2) / min(us1, us2)
        assert ratio < 10.0, f"CPU baseline too variable: {us1:.1f} vs {us2:.1f} us"


# ---------------------------------------------------------------------------
# Unit tests: result JSON schema
# ---------------------------------------------------------------------------


class TestResultSchema:
    """SCENARIO-HW-040: Result JSON must contain all required v3 schema fields."""

    REQUIRED_FIELDS = [
        "experiment",
        "title",
        "run_date",
        "schema",
        "duration_s",
        "bitstream_path",
        "board_discovered",
        "board_ip",
        "board_programmed",
        "hardware_latency_us",
        "cpu_baseline_latency_us",
        "speedup_vs_cpu",
        "human_action_required",
        "honest_verdict",
    ]

    VALID_VERDICTS = {
        "hardware_working",
        "board_programmed_latency_pending",
        "board_unreachable_human_required",
    }

    def _run_board_unreachable(self, result_path: Path) -> dict:
        """Run main() in a context where the board cannot be discovered."""
        fake_bitstream = result_path.parent / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", fake_bitstream),
            mock.patch.object(exp993, "_cpu_baseline_latency_us", return_value=12.0),
            mock.patch.object(exp993, "_discover_board", return_value=(False, None)),
        ):
            exp993.main()
        return json.loads(result_path.read_text())

    def test_all_required_fields_present(self, tmp_path: Path) -> None:
        """All required schema fields must be in the output JSON."""
        result_path = tmp_path / "result.json"
        data = self._run_board_unreachable(result_path)
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing required field: {field}"

    def test_schema_name_is_v3(self, tmp_path: Path) -> None:
        """Schema version field must be kv260_board_programming_v3."""
        result_path = tmp_path / "result.json"
        data = self._run_board_unreachable(result_path)
        assert data["schema"] == "kv260_board_programming_v3"

    def test_honest_verdict_is_valid(self, tmp_path: Path) -> None:
        """honest_verdict must be one of the three documented values."""
        result_path = tmp_path / "result.json"
        data = self._run_board_unreachable(result_path)
        assert data["honest_verdict"] in self.VALID_VERDICTS

    def test_board_unreachable_verdict_and_human_action(self, tmp_path: Path) -> None:
        """When board not found: verdict=board_unreachable_human_required and human_action_required=True."""
        result_path = tmp_path / "result.json"
        data = self._run_board_unreachable(result_path)
        assert data["honest_verdict"] == "board_unreachable_human_required"
        assert data["human_action_required"] is True
        assert data["board_discovered"] is False
        assert data["board_ip"] is None


# ---------------------------------------------------------------------------
# Integration tests: result always written
# ---------------------------------------------------------------------------


class TestResultAlwaysWritten:
    """REQ-HW-040 critical invariant: result JSON written regardless of failure path."""

    def test_result_written_when_board_unreachable(self, tmp_path: Path) -> None:
        """Result JSON exists when board is unreachable (the Exp 971 regression guard)."""
        result_path = tmp_path / "result.json"
        fake_bitstream = tmp_path / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", fake_bitstream),
            mock.patch.object(exp993, "_cpu_baseline_latency_us", return_value=10.0),
            mock.patch.object(exp993, "_discover_board", return_value=(False, None)),
        ):
            exp993.main()
        assert result_path.exists()

    def test_result_written_when_bitstream_missing(self, tmp_path: Path) -> None:
        """Result JSON exists even when the bitstream file is not present."""
        result_path = tmp_path / "result.json"
        missing_bitstream = tmp_path / "no.bit"
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", missing_bitstream),
            mock.patch.object(exp993, "_cpu_baseline_latency_us", return_value=10.0),
        ):
            exp993.main()
        assert result_path.exists()

    def test_result_written_when_cpu_baseline_raises(self, tmp_path: Path) -> None:
        """Result JSON exists even when CPU baseline raises an exception."""
        result_path = tmp_path / "result.json"
        fake_bitstream = tmp_path / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", fake_bitstream),
            mock.patch.object(
                exp993, "_cpu_baseline_latency_us", side_effect=RuntimeError("numpy missing")
            ),
            mock.patch.object(exp993, "_discover_board", return_value=(False, None)),
        ):
            exp993.main()
        assert result_path.exists()

    def test_result_written_when_scp_fails(self, tmp_path: Path) -> None:
        """Result JSON exists when SCP to board fails."""
        result_path = tmp_path / "result.json"
        fake_bitstream = tmp_path / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", fake_bitstream),
            mock.patch.object(exp993, "_cpu_baseline_latency_us", return_value=10.0),
            mock.patch.object(exp993, "_discover_board", return_value=(True, "192.168.51.98")),
            mock.patch.object(exp993, "_scp_bitstream", return_value=False),
        ):
            exp993.main()
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["board_discovered"] is True
        assert data["board_programmed"] is False
        assert data["human_action_required"] is True

    def test_result_written_when_programming_fails(self, tmp_path: Path) -> None:
        """Result JSON exists and board_programmed=False when dfx-mgr and fpgautil both fail."""
        result_path = tmp_path / "result.json"
        fake_bitstream = tmp_path / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", fake_bitstream),
            mock.patch.object(exp993, "_cpu_baseline_latency_us", return_value=10.0),
            mock.patch.object(exp993, "_discover_board", return_value=(True, "192.168.51.98")),
            mock.patch.object(exp993, "_scp_bitstream", return_value=True),
            mock.patch.object(exp993, "_program_board", return_value=False),
        ):
            exp993.main()
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["board_programmed"] is False
        assert data["honest_verdict"] == "board_unreachable_human_required"


# ---------------------------------------------------------------------------
# Integration test: full success path (mocked hardware)
# ---------------------------------------------------------------------------


class TestSuccessPath:
    """SCENARIO-HW-040: Mocked end-to-end success produces hardware_working verdict."""

    def test_hardware_working_verdict_with_latency(self, tmp_path: Path) -> None:
        """When board is programmed and hardware_latency_us>0, verdict=hardware_working."""
        result_path = tmp_path / "result.json"
        fake_bitstream = tmp_path / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", fake_bitstream),
            mock.patch.object(exp993, "_cpu_baseline_latency_us", return_value=50.0),
            mock.patch.object(exp993, "_discover_board", return_value=(True, "192.168.51.98")),
            mock.patch.object(exp993, "_scp_bitstream", return_value=True),
            mock.patch.object(exp993, "_program_board", return_value=True),
            mock.patch.object(exp993, "_measure_hardware_latency", return_value=5.0),
        ):
            exp993.main()
        data = json.loads(result_path.read_text())
        assert data["honest_verdict"] == "hardware_working"
        assert data["board_discovered"] is True
        assert data["board_programmed"] is True
        assert data["hardware_latency_us"] == 5.0
        assert data["cpu_baseline_latency_us"] == 50.0
        # speedup = cpu / hardware = 50 / 5 = 10x
        assert abs(data["speedup_vs_cpu"] - 10.0) < 0.01

    def test_board_programmed_latency_pending_verdict(self, tmp_path: Path) -> None:
        """When programmed but hardware latency unavailable, verdict=board_programmed_latency_pending."""
        result_path = tmp_path / "result.json"
        fake_bitstream = tmp_path / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)
        with (
            mock.patch.object(exp993, "_RESULT_FILE", result_path),
            mock.patch.object(exp993, "_BITSTREAM_PATH", fake_bitstream),
            mock.patch.object(exp993, "_cpu_baseline_latency_us", return_value=50.0),
            mock.patch.object(exp993, "_discover_board", return_value=(True, "192.168.51.98")),
            mock.patch.object(exp993, "_scp_bitstream", return_value=True),
            mock.patch.object(exp993, "_program_board", return_value=True),
            mock.patch.object(exp993, "_measure_hardware_latency", return_value=None),
        ):
            exp993.main()
        data = json.loads(result_path.read_text())
        assert data["honest_verdict"] == "board_programmed_latency_pending"
        assert data["board_programmed"] is True
        assert data["hardware_latency_us"] is None
        assert data["speedup_vs_cpu"] is None
