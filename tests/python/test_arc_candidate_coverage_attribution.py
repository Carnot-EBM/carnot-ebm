"""Tests for the candidate-coverage attribution script's pure classification logic
(scripts/arc_candidate_coverage_attribution.py) -- the membership/tolerance matching that determines
bucket (a) vs (b)/(c) in the 2026-07-23 independent replication of REQ-ARC-FCP-5757.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "arc_candidate_coverage_attribution", REPO / "scripts" / "arc_candidate_coverage_attribution.py"
)
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)

_data_matches = _mod._data_matches
_find_match = _mod._find_match


class TestDataMatches:
    def test_both_none_matches(self):
        assert _data_matches(None, None, tol=0) is True

    def test_one_none_one_not_does_not_match(self):
        assert _data_matches(None, {"x": 1, "y": 1}, tol=0) is False
        assert _data_matches({"x": 1, "y": 1}, None, tol=0) is False

    def test_exact_match_within_zero_tolerance(self):
        assert _data_matches({"x": 5, "y": 7}, {"x": 5, "y": 7}, tol=0) is True

    def test_off_by_one_fails_zero_tolerance(self):
        assert _data_matches({"x": 6, "y": 7}, {"x": 5, "y": 7}, tol=0) is False

    def test_within_tolerance_radius_matches(self):
        assert _data_matches({"x": 6, "y": 8}, {"x": 5, "y": 7}, tol=2) is True

    def test_outside_tolerance_radius_does_not_match(self):
        assert _data_matches({"x": 9, "y": 7}, {"x": 5, "y": 7}, tol=2) is False

    def test_tolerance_is_chebyshev_not_euclidean(self):
        # dx=2, dy=2 both within tol=2 independently (Chebyshev), even though Euclidean dist > 2
        assert _data_matches({"x": 7, "y": 9}, {"x": 5, "y": 7}, tol=2) is True


class TestFindMatch:
    def _cand(self, action_id, x=None, y=None):
        data = {"x": x, "y": y} if x is not None else None
        return SimpleNamespace(action_id=action_id, data=data)

    def test_finds_exact_keyboard_action(self):
        candidates = [self._cand(1), self._cand(2), self._cand(3)]
        assert _find_match(candidates, 2, None, tol=0) == 1

    def test_returns_none_when_absent(self):
        candidates = [self._cand(1), self._cand(2)]
        assert _find_match(candidates, 6, {"x": 10, "y": 10}, tol=0) is None

    def test_finds_click_within_tolerance(self):
        candidates = [self._cand(6, x=10, y=10), self._cand(6, x=40, y=40)]
        assert _find_match(candidates, 6, {"x": 11, "y": 11}, tol=2) == 0

    def test_first_matching_index_returned_not_best(self):
        # two candidates both within tolerance -- _find_match returns the FIRST it iterates
        # (the returned candidate order IS the search-priority order in rich_action_candidates,
        # so "first match" is the correct semantic: the rank it would actually be tried at).
        candidates = [self._cand(6, x=10, y=10), self._cand(6, x=11, y=11)]
        assert _find_match(candidates, 6, {"x": 11, "y": 11}, tol=2) == 0

    def test_action_id_mismatch_never_matches_regardless_of_data(self):
        candidates = [self._cand(4, x=10, y=10)]
        assert _find_match(candidates, 6, {"x": 10, "y": 10}, tol=5) is None
