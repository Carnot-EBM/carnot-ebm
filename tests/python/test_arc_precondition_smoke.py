"""Tests for the shared ARC focused pytest smoke helper.

Spec refs: REQ-REPORT-4475, SCENARIO-REPORT-4475-SMOKE.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.agentic.arc_precondition_smoke import arc_precondition_smoke, build_pytest_command


def test_scenario_report_4475_smoke_helper_runs_pytest_k_with_no_cov(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-4475-SMOKE: focused smoke pytest bypasses coverage addopts."""

    calls: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": cmd, **kwargs})
        return subprocess.CompletedProcess(cmd, 0, stdout="2 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    green, summary = arc_precondition_smoke("config_rule or arc_solver_kit", root=tmp_path)

    assert green is True
    assert summary == "2 passed\n"
    assert calls[0]["cmd"] == [
        str(tmp_path / ".venv" / "bin" / "pytest"),
        "-k",
        "config_rule or arc_solver_kit",
        "-q",
        "--no-cov",
    ]
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["check"] is False


def test_req_report_4475_smoke_helper_returns_bounded_failure_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-4475: smoke helper returns bare bool plus bounded text summary."""

    long_output = "x" * 3000

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout=long_output)

    monkeypatch.setattr(subprocess, "run", fake_run)

    green, summary = arc_precondition_smoke("arc_solver_kit", root=tmp_path, summary_chars=64)

    assert green is False
    assert summary == "x" * 64
    assert build_pytest_command("arc_solver_kit", root=tmp_path)[-1] == "--no-cov"


def test_req_report_4475_smoke_helper_reports_subprocess_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-4475: failed smoke process startup is an explicit red smoke gate."""

    def fake_run(_cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise TimeoutError("too slow")

    monkeypatch.setattr(subprocess, "run", fake_run)

    green, summary = arc_precondition_smoke("first_contact", root=tmp_path)

    assert green is False
    assert "TimeoutError: too slow" in summary
