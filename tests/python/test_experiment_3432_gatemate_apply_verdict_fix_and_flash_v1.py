"""Tests for Experiment 3432 — GateMate bootstrap verdict fix + single flash.

Covers REQ-HW-108 / SCENARIO-HW-108.

These tests exercise the PURE helpers only (the verdict classifier, the PATH
resolver, and the precondition builder). The subprocess/hardware paths in the
script are marked ``# pragma: no cover`` because they drive a physical GateMate
board over dirtyJtag and cannot run deterministically in CI. The load-bearing
logic — that ``unspecified`` is structurally impossible to emit — lives entirely
in the covered pure functions.
"""

import os

import pytest

from scripts import experiment_3432_gatemate_apply_verdict_fix_and_flash_v1 as exp


# ---------------------------------------------------------------------------
# is_terminal_verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    [
        "complete: blocked_gatemate_toolchain_missing",
        "complete_gatemate_n16_flow_ran_not_flashed",
        "success: gatemate_n16_ising_tile_flashed",
        "success_x",
        "passed: x",
        "passed_x",
        "shipped: x",
        "shipped_x",
    ],
)
def test_is_terminal_verdict_accepts_terminal_prefixes(verdict):
    """SCENARIO-HW-108: recognised terminal prefixes are accepted."""
    assert exp.is_terminal_verdict(verdict) is True


@pytest.mark.parametrize("verdict", ["", "unspecified", "Unspecified", "  unspecified  ", "blocked_x", "marginal"])
def test_is_terminal_verdict_rejects_empty_and_unspecified(verdict):
    """REQ-HW-108: empty / 'unspecified' / bare partial tokens are NOT terminal."""
    assert exp.is_terminal_verdict(verdict) is False


# ---------------------------------------------------------------------------
# classify_verdict — the fix
# ---------------------------------------------------------------------------


def test_classify_verdict_toolchain_missing_takes_precedence():
    """Toolchain absence yields a blocked-but-terminal verdict, regardless of others."""
    v = exp.classify_verdict(toolchain_ok=False, board_reachable=True, flash_success=True)
    assert v.startswith("complete: blocked_gatemate_toolchain_missing")


def test_classify_verdict_board_unreachable():
    v = exp.classify_verdict(toolchain_ok=True, board_reachable=False, flash_success=False)
    assert v.startswith("complete: blocked_gatemate_board_unreachable")


def test_classify_verdict_flash_success():
    v = exp.classify_verdict(toolchain_ok=True, board_reachable=True, flash_success=True)
    assert v.startswith("success:")
    assert "flashed" in v


def test_classify_verdict_flow_ran_not_flashed():
    v = exp.classify_verdict(toolchain_ok=True, board_reachable=True, flash_success=False)
    assert v.startswith("complete:")
    assert "not_flashed" in v


def test_classify_verdict_is_total_and_never_unspecified():
    """REQ-HW-108 core guarantee: no input combination yields 'unspecified'/empty."""
    for tc in (False, True):
        for br in (False, True):
            for fs in (False, True):
                v = exp.classify_verdict(tc, br, fs)
                assert v, "verdict must be non-empty"
                assert v.strip().lower() != "unspecified"
                assert exp.is_terminal_verdict(v)


# ---------------------------------------------------------------------------
# verdict_fix_self_check
# ---------------------------------------------------------------------------


def test_verdict_fix_self_check_passes():
    """The self-check proves the classifier is total -> verdict_fix_applied=True."""
    assert exp.verdict_fix_self_check() is True


# ---------------------------------------------------------------------------
# find_oss_cad_suite_bin / resolve_toolchain_path
# ---------------------------------------------------------------------------


def test_find_oss_cad_suite_bin_returns_first_with_yosys(tmp_path):
    """A candidate dir qualifies only when it actually contains a yosys binary."""
    empty = tmp_path / "empty" / "bin"
    real = tmp_path / "real" / "bin"
    empty.mkdir(parents=True)
    real.mkdir(parents=True)
    (real / "yosys").write_text("#!/bin/sh\n")
    found = exp.find_oss_cad_suite_bin((str(empty), str(real)))
    assert found == str(real)


def test_find_oss_cad_suite_bin_returns_none_when_absent(tmp_path):
    missing = tmp_path / "nope" / "bin"
    assert exp.find_oss_cad_suite_bin((str(missing),)) is None


def test_resolve_toolchain_path_prepends_and_is_idempotent(tmp_path, monkeypatch):
    """resolve_toolchain_path prepends the bin dir once and does not duplicate it."""
    bin_dir = tmp_path / "oss" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "yosys").write_text("#!/bin/sh\n")
    monkeypatch.setattr(exp, "OSS_CAD_SUITE_BIN_CANDIDATES", (str(bin_dir),))
    monkeypatch.setattr(exp, "find_oss_cad_suite_bin", lambda candidates=(str(bin_dir),): str(bin_dir))
    monkeypatch.setenv("PATH", "/usr/bin")

    first = exp.resolve_toolchain_path()
    assert first == str(bin_dir)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)

    # Idempotent: a second call must not add a duplicate entry.
    exp.resolve_toolchain_path()
    assert os.environ["PATH"].split(os.pathsep).count(str(bin_dir)) == 1


def test_resolve_toolchain_path_returns_none_when_no_install(monkeypatch):
    monkeypatch.setattr(exp, "find_oss_cad_suite_bin", lambda candidates=None: None)
    monkeypatch.setenv("PATH", "/usr/bin")
    assert exp.resolve_toolchain_path() is None
    assert os.environ["PATH"] == "/usr/bin"  # unchanged


# ---------------------------------------------------------------------------
# build_preconditions
# ---------------------------------------------------------------------------


def test_build_preconditions_shape_all_present():
    tool_paths = {
        "yosys": "/opt/oss-cad-suite/bin/yosys",
        "nextpnr-himbaechel": "/opt/oss-cad-suite/bin/nextpnr-himbaechel",
        "openFPGALoader": "/opt/oss-cad-suite/bin/openFPGALoader",
    }
    checks = exp.build_preconditions(tool_paths, (True, "GateMate Series GM1Ax"))
    resources = {c["resource"]: c for c in checks}
    assert resources["toolchain:yosys"]["available"] is True
    assert resources["gatemate_board_detect"]["available"] is True
    assert all("available" in c and "detail" in c for c in checks)
    # one entry per tool plus the board detect
    assert len(checks) == len(exp.REQUIRED_TOOLS) + 1


def test_build_preconditions_marks_missing_tools():
    tool_paths = {"yosys": "/usr/bin/yosys", "nextpnr-himbaechel": None, "openFPGALoader": None}
    checks = exp.build_preconditions(tool_paths, (False, "exit=127"))
    resources = {c["resource"]: c for c in checks}
    assert resources["toolchain:nextpnr-himbaechel"]["available"] is False
    assert resources["toolchain:nextpnr-himbaechel"]["detail"] == "not on PATH"
    assert resources["gatemate_board_detect"]["available"] is False
