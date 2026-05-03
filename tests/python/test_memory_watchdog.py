"""Pytest memory watchdog tests.

Spec: REQ-INFRA-076, SCENARIO-INFRA-088, SCENARIO-INFRA-089
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from carnot.testing.pytest_memory_watchdog import MemoryLeakDetected, PytestMemoryWatchdog


class FakeItem:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid


def rss_sequence(*values_mb: int):
    values = iter(value * 1024 for value in values_mb)

    def next_rss_kb() -> int:
        return next(values)

    return next_rss_kb


def test_watchdog_records_baseline() -> None:
    """REQ-INFRA-076: pytest_runtest_setup captures the per-test RSS baseline."""
    item = FakeItem("tests/python/test_example.py::test_baseline")
    watchdog = PytestMemoryWatchdog(get_rss_kb=rss_sequence(128))

    watchdog.record_setup(item)

    assert watchdog.baseline_for(item) == 128
    assert watchdog.per_test_leak_threshold_mb == 500
    assert watchdog.session_cumulative_limit_mb == 8192
    assert watchdog.finish_session(Path.cwd()) is None


def test_watchdog_detects_leak(tmp_path: Path) -> None:
    """SCENARIO-INFRA-088: a retained >500 MB allocation fails the owning test."""
    item = FakeItem("tests/python/test_example.py::test_leak")
    watchdog = PytestMemoryWatchdog(get_rss_kb=rss_sequence(64, 565))
    watchdog.record_setup(item)

    with pytest.raises(MemoryLeakDetected, match=r"Memory leak: \+501MB"):
        watchdog.record_teardown(item)

    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import sys

            sys.path.insert(0, {str(repo_root / "python")!r})

            import pytest
            from carnot.testing.pytest_memory_watchdog import (
                MemoryLeakDetected,
                PytestMemoryWatchdog,
            )


            def pytest_configure(config):
                config._carnot_memory_watchdog = PytestMemoryWatchdog()


            def pytest_runtest_setup(item):
                item.config._carnot_memory_watchdog.record_setup(item)


            def pytest_runtest_teardown(item, nextitem):
                try:
                    item.config._carnot_memory_watchdog.record_teardown(item)
                except MemoryLeakDetected as exc:
                    pytest.fail(str(exc), pytrace=False)
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_retained_allocation.py").write_text(
        textwrap.dedent(
            """
            _LEAK = []


            def test_retains_oversized_block():
                block = bytearray(530 * 1024 * 1024)
                for index in range(0, len(block), 4096):
                    block[index] = 1
                _LEAK.append(block)
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path / "test_retained_allocation.py"),
            "-q",
            "-p",
            "no:cov",
            "-o",
            "addopts=",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    nested_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Memory leak: +" in nested_output
    assert "test_retains_oversized_block" in nested_output


def test_watchdog_cleanup(tmp_path: Path) -> None:
    """REQ-INFRA-076, SCENARIO-INFRA-089: teardown cleanup and session log writing."""
    first = FakeItem("tests/python/test_example.py::test_first")
    second = FakeItem("tests/python/test_example.py::test_second")
    watchdog = PytestMemoryWatchdog(
        get_rss_kb=rss_sequence(100, 140, 140, 220),
        session_cumulative_limit_mb=50,
    )

    watchdog.record_setup(first)
    watchdog.record_teardown(first)
    watchdog.record_setup(second)
    watchdog.record_teardown(second)

    assert watchdog.baseline_for(first) is None
    assert watchdog.baseline_for(second) is None

    report = watchdog.finish_session(tmp_path, timestamp="20260502T000000Z")

    assert report is not None
    assert report.log_path == tmp_path / "results" / "pytest_memory_20260502T000000Z.log"
    assert report.log_path.exists()
    assert "test_second +80MB" in report.warning
    assert "test_first +40MB" in report.log_path.read_text(encoding="utf-8")
