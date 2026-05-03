"""Pytest address-space cap tests.

Spec: REQ-INFRA-077, SCENARIO-INFRA-090, SCENARIO-INFRA-091
"""

from __future__ import annotations

import importlib
import resource
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.testing.pytest_memory_watchdog import MemoryLeakDetected, SessionMemoryReport


LIMIT_BYTES = 8 * 1024**3


def test_rlimit_as_is_set() -> None:
    """REQ-INFRA-077, SCENARIO-INFRA-090: pytest caps virtual address space at 8 GB."""
    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)

    assert soft != resource.RLIM_INFINITY
    assert soft <= LIMIT_BYTES


def test_rlimit_does_not_break_existing_imports() -> None:
    """SCENARIO-INFRA-091: the cap keeps normal packaged imports working."""
    assert importlib.import_module("carnot") is not None


def test_rlimit_helper_warns_when_kernel_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-077: unsupported RLIMIT_AS kernels warn and let pytest continue."""
    conftest = sys.modules["conftest"]

    class RejectingResource:
        RLIMIT_AS = 9
        RLIM_INFINITY = -1
        error = OSError

        def getrlimit(self, _resource_id: int) -> tuple[int, int]:
            return (self.RLIM_INFINITY, self.RLIM_INFINITY)

        def setrlimit(self, _resource_id: int, _limits: tuple[int, int]) -> None:
            raise self.error("not supported")

    monkeypatch.setattr(conftest, "resource", RejectingResource())

    with pytest.warns(RuntimeWarning, match="Could not set RLIMIT_AS"):
        assert conftest._set_process_address_space_limit() is False


def test_rlimit_helper_caps_at_finite_hard_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-077: the helper never sets the soft limit above the hard limit."""
    conftest = sys.modules["conftest"]
    calls: list[tuple[int, tuple[int, int]]] = []

    class FiniteHardLimitResource:
        RLIMIT_AS = 9
        RLIM_INFINITY = -1
        error = OSError

        def getrlimit(self, _resource_id: int) -> tuple[int, int]:
            return (self.RLIM_INFINITY, 4 * 1024**3)

        def setrlimit(self, resource_id: int, limits: tuple[int, int]) -> None:
            calls.append((resource_id, limits))

    monkeypatch.setattr(conftest, "resource", FiniteHardLimitResource())

    assert conftest._set_process_address_space_limit() is True
    assert calls == [(9, (4 * 1024**3, 4 * 1024**3))]


def test_conftest_lazy_watchdog_initializes() -> None:
    """REQ-INFRA-076: conftest lazily installs the RSS watchdog if missing."""
    conftest = sys.modules["conftest"]
    config = SimpleNamespace()

    watchdog = conftest._get_memory_watchdog(config)

    assert config._carnot_memory_watchdog is watchdog


def test_conftest_watchdog_teardown_failure_is_pytest_failure() -> None:
    """REQ-INFRA-076: conftest converts watchdog leak exceptions to pytest failures."""
    conftest = sys.modules["conftest"]

    class FailingWatchdog:
        def record_teardown(self, _item: object) -> None:
            raise MemoryLeakDetected("Memory leak: +501MB")

    item = SimpleNamespace(config=SimpleNamespace(_carnot_memory_watchdog=FailingWatchdog()))

    with pytest.raises(pytest.fail.Exception, match="Memory leak: \\+501MB"):
        conftest.pytest_runtest_teardown(item, None)


def test_conftest_sessionfinish_warns_on_watchdog_report(tmp_path: Path) -> None:
    """SCENARIO-INFRA-089: conftest emits the watchdog session warning."""
    conftest = sys.modules["conftest"]

    class ReportingWatchdog:
        def finish_session(self, _root_path: Path) -> SessionMemoryReport:
            return SessionMemoryReport(
                warning="Pytest memory watchdog: cumulative RSS growth exceeded",
                log_path=tmp_path / "pytest_memory.log",
            )

    session = SimpleNamespace(
        config=SimpleNamespace(rootpath=tmp_path, _carnot_memory_watchdog=ReportingWatchdog())
    )

    with pytest.warns(pytest.PytestWarning, match="cumulative RSS growth"):
        conftest.pytest_sessionfinish(session, 0)
