"""Tests for scripts/root_clutter_sweep.py's protection of research-roadmap-next.yaml.

Origin: 2026-07-03. The outer-loop session diagnosed a real, previously-unconnected
interaction between two independent subsystems: milestone-activation stalls (a stuck
research-roadmap-next.yaml repeatedly REFUSED by scripts/exclusion_manifest_lint.py) and
this sweeper, which runs every 30 minutes via orphan-cleanup.sh and silently relocates
any untracked root file older than 120 minutes. research-roadmap-next.yaml is UNTRACKED
by nature exactly while it's stuck (it only becomes tracked once activation succeeds),
so a long-enough stall meant the sweeper would quietly move the draft to
.root-scratch-trash/ before the underlying HARD violation could ever be diagnosed and
fixed -- confirmed via /tmp/root-clutter-sweep.log showing "mv research-roadmap-
next.yaml" at least twice (.475, .476), each time discarding real planner compute.

Fix: added research-roadmap-next.yaml to ALLOWLIST alongside its already-protected
siblings (research-roadmap.yaml, research-complete.yaml). These tests exercise the real
sweep() function end-to-end (REPO monkeypatched to an isolated tmp_path, which is not a
git repo so _tracked_files() naturally falls back to an empty set -- exercising the
untracked-old-file code path the incident actually hit) rather than just asserting
membership in ALLOWLIST, so a future refactor of sweep()'s matching logic can't silently
re-break the protection while the membership check still (trivially) passes.

Spec refs: none (operational tooling, no OpenSpec capability).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest


def _load():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "root_clutter_sweep.py"
    spec = importlib.util.spec_from_file_location("root_clutter_sweep", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["root_clutter_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load()

_OLD_AGE_S = 200 * 60  # older than the 120-min default age guard


def _age_file(path: Path, age_s: float) -> None:
    mtime = time.time() - age_s
    os.utime(path, (mtime, mtime))


class TestResearchRoadmapNextProtection:
    def test_is_in_allowlist(self) -> None:
        assert "research-roadmap-next.yaml" in _MOD.ALLOWLIST

    def test_stuck_untracked_roadmap_draft_is_never_swept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end reproduction of the .475/.476 incident shape: an untracked,
        old (>120min) research-roadmap-next.yaml must survive a real --apply sweep."""
        monkeypatch.setattr(_MOD, "REPO", tmp_path)
        roadmap = tmp_path / "research-roadmap-next.yaml"
        roadmap.write_text("milestone: 2026.07.999\ntasks: []\n")
        _age_file(roadmap, _OLD_AGE_S)

        result = _MOD.sweep(apply=True, min_age_min=120)

        assert roadmap.exists(), "the stuck roadmap draft must not be moved out of place"
        assert "research-roadmap-next.yaml" not in result["moved_to_trash"]
        assert "research-roadmap-next.yaml" not in result["deleted_artifacts"]
        assert "research-roadmap-next.yaml" not in result["warn_tracked_nonallowlist"]

    def test_other_old_untracked_scratch_is_still_swept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: the allowlist addition must not accidentally protect
        genuine scratch -- an unrelated old untracked .py file must still be moved."""
        monkeypatch.setattr(_MOD, "REPO", tmp_path)
        scratch = tmp_path / "probe_debug_check.py"
        scratch.write_text("print('scratch')\n")
        _age_file(scratch, _OLD_AGE_S)

        result = _MOD.sweep(apply=True, min_age_min=120)

        assert not scratch.exists(), "genuine old untracked scratch should still be swept"
        assert "probe_debug_check.py" in result["moved_to_trash"]

    def test_young_stuck_roadmap_draft_is_skipped_as_in_flight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A freshly-written draft (well under the age guard) must be left alone
        regardless of allowlist status -- confirms the allowlist entry doesn't change
        the age-guard's own in-flight protection for a young draft."""
        monkeypatch.setattr(_MOD, "REPO", tmp_path)
        roadmap = tmp_path / "research-roadmap-next.yaml"
        roadmap.write_text("milestone: 2026.07.999\ntasks: []\n")
        # no _age_file call -- mtime is "now", well under the 120-min guard

        result = _MOD.sweep(apply=True, min_age_min=120)

        assert roadmap.exists()
        assert "research-roadmap-next.yaml" not in result["moved_to_trash"]
