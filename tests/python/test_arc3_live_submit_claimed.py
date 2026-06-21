"""Regression test for the 2026-06-21 arc3_live_submit dynamic-replay-set refresh.

REQ: the live-submit driver must replay the CURRENT validated submission package (auto-tracking
conductor refreshes as `reproducible_total_levels` grows), not a hardcoded 11-game/13-level set
frozen at the 2026-06-17 first submission. The stale hardcoded set left ~26 reproduced levels
on the table at submit time.

SCENARIO-LIVESUBMIT-1: _build_claimed sources the game/level set from the latest package
                       manifest, keeping only env-matched games the metaharness can replay.
SCENARIO-LIVESUBMIT-2: when a game in the package has NO replayable banked trajectory, it is
                       dropped (not silently claimed with zero actions).
SCENARIO-LIVESUBMIT-3: with no package available, the driver falls back to the preserved
                       2026-06-17 hardcoded set (never an empty replay set).
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "arc3_live_submit", str(REPO / "scripts" / "arc3_live_submit.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _FakeMH:
    """Stand-in metaharness: maps a fixed set of game ids to non-empty action lists."""

    def __init__(self, loadable):
        self.GAME_ARTIFACTS = {g: f"artifact_{g}" for g in loadable}
        self.RESOLVED_ARTIFACTS = {}

    def load_actions(self, src):
        return [{"id": 6}] if src else []


def test_scenario_livesubmit_1_sources_from_package(monkeypatch) -> None:
    """SCENARIO-LIVESUBMIT-1: env-matched + loadable games come through with package levels."""
    drv = _load_driver()
    monkeypatch.setattr(drv, "_latest_package", lambda: {"path": "pkg.json", "manifest": [
        {"game": "lp85", "env_matched": True, "levels": 5},
        {"game": "tn36", "env_matched": True, "levels": 7},
    ]})
    claimed, source = drv._build_claimed(_FakeMH({"lp85", "tn36"}))
    assert claimed == {"lp85": 5, "tn36": 7}
    assert source == "package:pkg.json"


def test_scenario_livesubmit_2_drops_unmatched_and_unloadable(monkeypatch) -> None:
    """SCENARIO-LIVESUBMIT-2: not-env-matched OR no-banked-trajectory games are dropped."""
    drv = _load_driver()
    monkeypatch.setattr(drv, "_latest_package", lambda: {"path": "pkg.json", "manifest": [
        {"game": "lp85", "env_matched": True, "levels": 5},   # kept
        {"game": "vc33", "env_matched": True, "levels": 1},   # dropped: no banked trajectory
        {"game": "zz99", "env_matched": False, "levels": 9},  # dropped: not env-matched
    ]})
    claimed, source = drv._build_claimed(_FakeMH({"lp85"}))  # only lp85 loadable
    assert claimed == {"lp85": 5}
    assert "vc33" not in claimed and "zz99" not in claimed


def test_scenario_livesubmit_3_fallback_when_no_package(monkeypatch) -> None:
    """SCENARIO-LIVESUBMIT-3: absent package -> preserved 2026-06-17 hardcoded set, never empty."""
    drv = _load_driver()
    monkeypatch.setattr(drv, "_latest_package", lambda: None)
    claimed, source = drv._build_claimed(_FakeMH(set()))
    assert claimed == drv.CLAIMED_FALLBACK
    assert sum(claimed.values()) == 13  # the 13 levels of the first submission
    assert source == "hardcoded_fallback_2026_06_17"
