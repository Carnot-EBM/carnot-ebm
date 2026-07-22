"""Tests for the offline/live ACTION6 bounds-validation gap in arc_solver_kit.reproduce().

The OFFLINE arcade (arc_agi's LocalEnvironmentWrapper.step()) silently accepts and routes
any ACTION6 click, including coordinates outside the [0,63]x[0,63] range the LIVE
arcprize.org API enforces (arcengine.enums.ComplexAction's x/y fields are declared
Field(ge=0, le=63), but that validation is only wired into the live HTTP handler, never
into the local/offline path). A solve route can therefore reproduce cleanly offline while
being un-submittable live -- exactly what happened to lf52's original L9 route (22
out-of-bounds clicks, x up to 132) before its 2026-07-17 fix (commit 5ca2a999b). This test
suite covers the reusable-primitive hardening added to reproduce() so any FUTURE offline
solve is flagged for this gap immediately, rather than only being discovered at
live-submission time.

Spec: CLAUDE.md "ARC-AGI-3 Generalization-Testing Floor" (task class 2: reusable-primitive
hardening based on a genuine gap surfaced by a real measurement -- the 2026-07-17 live
re-validation that surfaced lf52's original OOB route).
"""

from __future__ import annotations

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_game_adapters import _json_action_label, get_adapter


class TestActionSixClickFromLabel:
    """_action6_click_from_label: best-effort extraction, never raises."""

    def test_extracts_action6_click(self):
        label = _json_action_label(6, {"x": 30, "y": 40})
        assert kit._action6_click_from_label(label) == (30, 40)

    def test_returns_none_for_non_action6_label(self):
        label = _json_action_label(4)
        assert kit._action6_click_from_label(label) is None

    def test_returns_none_for_action6_label_missing_xy(self):
        label = _json_action_label(6)
        assert kit._action6_click_from_label(label) is None

    def test_returns_none_for_unparseable_json(self):
        assert kit._action6_click_from_label("click:30,40") is None
        assert kit._action6_click_from_label("6:30:40") is None
        assert kit._action6_click_from_label("") is None

    def test_returns_none_for_json_non_mapping(self):
        assert kit._action6_click_from_label("[1, 2, 3]") is None
        assert kit._action6_click_from_label("42") is None


class TestActionSixOutOfLiveBounds:
    """_action6_out_of_live_bounds reuses arcengine's own declared [0,63] bound."""

    def test_in_bounds_values_pass(self):
        assert kit._action6_out_of_live_bounds(30, 40) is False
        assert kit._action6_out_of_live_bounds(0, 0) is False
        assert kit._action6_out_of_live_bounds(63, 63) is False

    def test_out_of_bounds_values_fail(self):
        assert kit._action6_out_of_live_bounds(132, 40) is True
        assert kit._action6_out_of_live_bounds(64, 0) is True
        assert kit._action6_out_of_live_bounds(0, -1) is True
        assert kit._action6_out_of_live_bounds(-8, 5) is True


class TestReproduceFlagsOobAction6Clicks:
    """Real, non-mocked end-to-end coverage: reproduce() against the actual lf52 offline env."""

    def test_reproduce_flags_a_synthetic_oob_click(self):
        adapter = get_adapter("lf52")
        assert adapter is not None
        solution = [
            _json_action_label(1),
            _json_action_label(1),
            _json_action_label(6, {"x": 999, "y": 5}),
        ]
        result = kit.reproduce("lf52", solution, adapter.apply, claimed_level=None)

        assert result["checked_action6_clicks"] == 1
        assert result["any_oob_action6_clicks"] is True
        assert result["oob_action6_clicks"] == [{"index": 2, "x": 999, "y": 5}]
        # the OOB flag is additive -- it must not change offline-reproduction semantics
        assert result["reproduced"] is True

    def test_reproduce_reports_clean_when_no_oob_clicks_present(self):
        adapter = get_adapter("lf52")
        assert adapter is not None
        solution = [
            _json_action_label(1),
            _json_action_label(6, {"x": 30, "y": 40}),
        ]
        result = kit.reproduce("lf52", solution, adapter.apply, claimed_level=None)

        assert result["checked_action6_clicks"] == 1
        assert result["any_oob_action6_clicks"] is False
        assert result["oob_action6_clicks"] == []

    def test_reproduce_with_no_action6_labels_reports_zero_checked(self):
        adapter = get_adapter("lf52")
        assert adapter is not None
        solution = [_json_action_label(1), _json_action_label(1)]
        result = kit.reproduce("lf52", solution, adapter.apply, claimed_level=None)

        assert result["checked_action6_clicks"] == 0
        assert result["any_oob_action6_clicks"] is False
        assert result["oob_action6_clicks"] == []
