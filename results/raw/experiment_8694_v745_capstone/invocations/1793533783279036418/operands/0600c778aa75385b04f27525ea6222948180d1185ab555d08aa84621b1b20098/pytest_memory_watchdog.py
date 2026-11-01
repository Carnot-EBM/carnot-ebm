"""Pytest RSS watchdog for conductor-safe Python test runs.

Spec: REQ-INFRA-076, SCENARIO-INFRA-088, SCENARIO-INFRA-089
"""

from __future__ import annotations

import resource
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any


DEFAULT_PER_TEST_LEAK_THRESHOLD_MB = 500
DEFAULT_SESSION_CUMULATIVE_LIMIT_MB = 8192


@dataclass(frozen=True)
class RssDeltaSample:
    nodeid: str
    rss_before_mb: int
    rss_after_mb: int
    delta_mb: int

    def summary(self) -> str:
        return (
            f"{self.nodeid} +{self.delta_mb}MB "
            f"(before={self.rss_before_mb}MB after={self.rss_after_mb}MB)"
        )


@dataclass(frozen=True)
class SessionMemoryReport:
    warning: str
    log_path: Path


class MemoryLeakDetected(AssertionError):
    """Raised when one pytest item grows RSS beyond the per-test threshold."""


def current_ru_maxrss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def ru_maxrss_to_mb(ru_maxrss: int) -> int:
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return int(round(ru_maxrss / divisor))


class PytestMemoryWatchdog:
    def __init__(
        self,
        *,
        get_rss_kb: Callable[[], int] | None = None,
        per_test_leak_threshold_mb: int = DEFAULT_PER_TEST_LEAK_THRESHOLD_MB,
        session_cumulative_limit_mb: int = DEFAULT_SESSION_CUMULATIVE_LIMIT_MB,
    ) -> None:
        self.get_rss_kb = get_rss_kb or current_ru_maxrss_kb
        self.per_test_leak_threshold_mb = per_test_leak_threshold_mb
        self.session_cumulative_limit_mb = session_cumulative_limit_mb
        self.cumulative_rss_mb = 0
        self.samples: list[RssDeltaSample] = []
        self._baselines_mb: dict[str, int] = {}

    def record_setup(self, item: Any) -> None:
        nodeid = self._nodeid(item)
        rss_before_mb = self._current_rss_mb()
        self._baselines_mb[nodeid] = rss_before_mb
        item._carnot_rss_before_mb = rss_before_mb

    def record_teardown(self, item: Any) -> RssDeltaSample:
        nodeid = self._nodeid(item)
        rss_after_mb = self._current_rss_mb()
        rss_before_mb = self._baselines_mb.pop(nodeid, rss_after_mb)
        delta_mb = max(0, rss_after_mb - rss_before_mb)
        sample = RssDeltaSample(nodeid, rss_before_mb, rss_after_mb, delta_mb)
        self.samples.append(sample)
        self.cumulative_rss_mb += delta_mb
        item._carnot_rss_after_mb = rss_after_mb
        item._carnot_rss_delta_mb = delta_mb
        if delta_mb > self.per_test_leak_threshold_mb:
            raise MemoryLeakDetected(f"Memory leak: +{delta_mb}MB")
        return sample

    def baseline_for(self, item: Any) -> int | None:
        return self._baselines_mb.get(self._nodeid(item))

    def top_samples(self, limit: int = 5) -> list[RssDeltaSample]:
        return sorted(self.samples, key=lambda sample: sample.delta_mb, reverse=True)[:limit]

    def finish_session(
        self, root_path: Path, *, timestamp: str | None = None
    ) -> SessionMemoryReport | None:
        if self.cumulative_rss_mb <= self.session_cumulative_limit_mb:
            return None
        timestamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        results_dir = root_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / f"pytest_memory_{timestamp}.log"
        top_summaries = [sample.summary() for sample in self.top_samples()]
        warning = (
            f"Pytest memory watchdog: cumulative RSS growth {self.cumulative_rss_mb}MB "
            f"exceeded {self.session_cumulative_limit_mb}MB; top RSS deltas: "
            f"{'; '.join(top_summaries)}"
        )
        log_path.write_text(self._format_log(timestamp, top_summaries), encoding="utf-8")
        return SessionMemoryReport(warning=warning, log_path=log_path)

    def _current_rss_mb(self) -> int:
        return ru_maxrss_to_mb(self.get_rss_kb())

    def _format_log(self, timestamp: str, top_summaries: list[str]) -> str:
        all_summaries = [sample.summary() for sample in self.top_samples(limit=len(self.samples))]
        return "\n".join(
            [
                "Carnot pytest memory watchdog",
                f"timestamp: {timestamp}",
                f"cumulative_rss_delta_mb: {self.cumulative_rss_mb}",
                f"session_cumulative_limit_mb: {self.session_cumulative_limit_mb}",
                "top_5_rss_delta_tests:",
                *[f"{index}. {summary}" for index, summary in enumerate(top_summaries, start=1)],
                "all_rss_delta_tests:",
                *[f"{index}. {summary}" for index, summary in enumerate(all_summaries, start=1)],
                "",
            ]
        )

    @staticmethod
    def _nodeid(item: Any) -> str:
        return str(getattr(item, "nodeid", item))
