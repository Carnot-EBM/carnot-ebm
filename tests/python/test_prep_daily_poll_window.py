"""REQ-HARNESS-6054 regression tests for `poll_save_run` — the daily-prep save-run poll.

THE INCIDENT (recurring through 2026-09-02). The prep script polled the kernel
save-run for 24 x 15s = 6 minutes, but the save-run starts a vLLM server that
alone takes ~7 minutes to come up. The poll always expired first, recorded the
ambiguous `save_run: "?"`, and exited 1 — so `carnot-arc-daily-prep.service`
failed on every unattended run while the Kaggle side finished fine minutes
later. Hand-verified false alarms: kernel versions 9, 22, 27, 37, and 53 (the
`note` trail in `ops/arc-daily-prep-status.json`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "prep_daily_submission_poll",
    Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "prep_daily_submission.py",
)
assert _SPEC is not None and _SPEC.loader is not None
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


class _StatusFeed:
    """Fake `kaggle` CLI: returns each queued status once, then repeats the last."""

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = list(statuses)
        self.calls = 0

    def __call__(self, *args: str, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return SimpleNamespace(stdout=status)


class _FakeTime:
    """Deterministic clock: sleeping advances it; nothing else does."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_complete_inside_budget_returns_complete(monkeypatch) -> None:
    # SCENARIO-HARNESS-6054-1: two "running" polls, then the real CLI string.
    feed = _StatusFeed(["running", "running", 'has status "KernelWorkerStatus.COMPLETE"'])
    monkeypatch.setattr(prep, "kaggle", feed)
    ft = _FakeTime()
    result = prep.poll_save_run(timeout_s=3600, interval_s=30, sleep=ft.sleep, clock=ft.clock)
    assert result == "complete"
    assert feed.calls == 3
    assert ft.sleeps == [30, 30]


def test_error_status_reported_as_error(monkeypatch) -> None:
    # SCENARIO-HARNESS-6054-2: an error breaks out immediately.
    feed = _StatusFeed(['has status "KernelWorkerStatus.ERROR"'])
    monkeypatch.setattr(prep, "kaggle", feed)
    ft = _FakeTime()
    result = prep.poll_save_run(timeout_s=3600, interval_s=30, sleep=ft.sleep, clock=ft.clock)
    assert result == "error"
    assert ft.sleeps == []


def test_budget_expiry_is_honest_not_ambiguous(monkeypatch) -> None:
    # SCENARIO-HARNESS-6054-3: a never-finishing save-run yields an explicit
    # still-running verdict, never the old ambiguous "?".
    feed = _StatusFeed(["running"])
    monkeypatch.setattr(prep, "kaggle", feed)
    ft = _FakeTime()
    result = prep.poll_save_run(timeout_s=100, interval_s=30, sleep=ft.sleep, clock=ft.clock)
    assert result == "still_running_after_100s"
    assert "?" not in result
    # The poll kept trying until the clock passed the budget.
    assert feed.calls >= 3
    assert ft.now >= 100


def test_default_budget_outlasts_known_save_run_runtime() -> None:
    # SCENARIO-HARNESS-6054-4: vLLM startup alone is ~7 minutes; the old
    # 6-minute budget could never see the truth. Require at least 30 minutes.
    assert prep.SAVE_RUN_POLL_TIMEOUT_S >= 1800
