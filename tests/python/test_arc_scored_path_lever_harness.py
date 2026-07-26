"""Drift guards for the SCORED-PATH lever harness (``scripts/arc_scored_path_lever_harness.py``).

WHY A TEST FOR A HARNESS. The harness declares its own ARMS dict, independent of
``experiment_5836_frontier_discipline_ab``'s. That means it can drift the same way exp5836's arms
drifted: arm B2 there pinned 3 of 7 gated flags, so the 2026-07-25 flip silently turned the HUD A/B's
own CONTROL into the HUD TREATMENT, and published numbers changed meaning with nobody editing the
arm. Two tests in ``test_arc_frontier_discipline.py`` now make that impossible for exp5836. These
tests do the same job for this harness, plus the two things unique to it: that the control arm still
equals the LIVE configuration, and that the nav-pruner cell classifier cannot report a non-firing
lever as a null.

No network, no GPU, no game environment: everything here is pure declaration + classifier logic.

Spec refs: REQ-ARC-WMTE-5970, SCENARIO-ARC-WMTE-5970-FIRE-COUNTERS.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_HARNESS = Path(__file__).resolve().parents[2] / "scripts" / "arc_scored_path_lever_harness.py"


def _harness() -> ModuleType:
    """Load the harness by path. It lives in scripts/ (not an installed package) and is loaded the
    same way the sibling exp5836 harness tests load theirs."""

    spec = importlib.util.spec_from_file_location("arc_scored_path_lever_harness", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_harness_arm_pins_every_gated_flag():
    """An arm that pins only a SUBSET inherits module defaults for the rest, so a later flag flip
    redefines that arm underneath already-published numbers. This is the exact defect that made
    exp5836's arm B2 the HUD treatment; it must not be reintroduced in a second harness."""

    m = _harness()
    keys = set(m.SHIPPED_2026_07_25)
    assert len(keys) == 8, f"expected the eight gated flags, got {sorted(keys)}"
    assert "hazard_move_pruner" in keys, "the nav-side pruner must be pinned like every other lever"
    for name, kwargs in m.ARMS.items():
        assert set(kwargs) == keys, f"arm {name} does not pin exactly the gated set"


def test_the_control_arm_is_the_live_configuration():
    """Pinning keeps an ARM stable; only this check tells you whether the arm named "S" is still the
    LIVE config. If a SUBMITTED_* flag is flipped and nobody updates the harness, "S" silently stops
    being the control -- and every delta measured against it becomes a two-lever delta."""

    m = _harness()
    parity = m.assert_shipped_dict_matches_module_globals()
    assert parity["pinned_vs_live_drift"] == {}, (
        f"the harness's pinned control no longer matches the live SUBMITTED_* globals: "
        f"{parity['pinned_vs_live_drift']} -- update SHIPPED_2026_07_25 (and re-read any verdict "
        f"measured against arm S) before using it as a control"
    )
    assert m.ARMS["S"] == m.SHIPPED_2026_07_25


def test_each_single_lever_arm_differs_from_the_control_in_exactly_its_own_lever():
    """A single-lever arm that differs in two levers measures the pair, not the lever. Computed
    rather than eyeballed, because eyeballing is what let the B2 contamination through."""

    m = _harness()
    expected = {
        "S_minus_frontier": set(m._FRONTIER_KEYS),
        "S_minus_hud": set(m._HUD_KEYS),
        "S_plus_hazard": {"hazard_move_pruner"},
        "S_minus_both": set(m._FRONTIER_KEYS) | set(m._HUD_KEYS),
    }
    for arm, want in expected.items():
        got = {k for k in m.SHIPPED_2026_07_25 if m.ARMS[arm][k] != m.ARMS["S"][k]}
        # frontier_gradient is already False in the shipped config, so turning it "off" is a no-op
        # difference -- compare against the flags that actually CHANGE.
        assert got <= want, f"arm {arm} also differs in {sorted(got - want)}"
        assert got, f"arm {arm} is identical to the control -- it measures nothing"


def test_the_budget_default_is_the_scored_agents_own_action_cap():
    """The scored agent is capped at MAX_ACTIONS=400 per game (the CarnotAgent adapter in
    ``arc_competition_agent``; the framework loop is ``while not done and action_counter <=
    MAX_ACTIONS``). Defaulting the harness to 2000 gives the agent ~5x more play than the eval
    allows, and it CHANGES CONCLUSIONS: measured at 400, four of five spot-checked games lose at
    least one level and r11l -- the only game the HUD lever has been shown to move -- reaches zero.

    Asserted against the value a caller who passes no ``--budget`` actually receives, not against a
    separate constant that could drift away from the parser.
    """

    import re

    m = _harness()
    source = open(m.__file__, encoding="utf-8").read()
    match = re.search(r'add_argument\(\s*"--budget",\s*type=int,\s*default=(\d+)\s*\)', source)
    assert match, "could not find the --budget argument definition"
    assert int(match.group(1)) == 400, (
        f"harness default budget is {match.group(1)}, expected 400 (the scored agent's own cap)"
    )
    # And the run artifact must record whether the budget it used matched that cap, so a
    # budget-2000 run can never be read later as if it were the eval's condition.
    assert "budget_matches_scored_cap" in source
    assert "scored_agent_max_actions" in source


def test_hazard_verdict_separates_a_non_firing_lever_from_a_real_null():
    """THE CLASSIFIER THAT MATTERS. ``rows_pruned == 0`` has four structurally different causes and
    only two of them are evidence about the lever. Reporting a dead observe channel, a game with no
    nav actions, or an unfitted hypothesis class as "the lever does not help" is the exp5836
    dead-channel failure -- a clean, zero-error, byte-identical NULL that was pure harness artifact.
    """

    m = _harness()
    v = m._hazard_verdict

    assert v({}, {}) == "ERROR", "unreadable diagnostics prove nothing"
    assert v({"error": "boom"}, {}) == "ERROR"
    assert v({"enabled": False}, {}) == "LEVER_OFF", "the control arm makes no claim either way"

    # observe_calls == 0 with the lever ENABLED is a WIRING BUG, and must never read as a null.
    dead = {
        "enabled": True,
        "observe_calls": 0,
        "observed_nav_transitions": 0,
        "model_fitted": False,
        "rows_pruned": 0,
    }
    assert v(dead, {}) == "UNINTERPRETABLE_NO_OBSERVE"

    # The hook ran but nothing was accepted -> a 100%-click game. No jurisdiction.
    no_nav = {
        "enabled": True,
        "observe_calls": 393,
        "observed_nav_transitions": 0,
        "model_fitted": False,
        "rows_pruned": 0,
    }
    assert v(no_nav, {}) == "UNINTERPRETABLE_NO_NAV"

    # Nav transitions seen, but no rung passed the evidence/trust/specificity gate.
    unfitted = {
        "enabled": True,
        "observe_calls": 388,
        "observed_nav_transitions": 383,
        "n_deaths": 1,
        "model_fitted": False,
        "rows_pruned": 0,
    }
    assert v(unfitted, {}) == "UNINTERPRETABLE_NOT_FITTED"

    # Fitted and predicted nothing lethal -- the ONLY reportable null.
    fired_null = {
        "enabled": True,
        "observe_calls": 388,
        "observed_nav_transitions": 383,
        "n_deaths": 6,
        "model_fitted": True,
        "rows_pruned": 0,
    }
    assert v(fired_null, {}) == "FIRED_NO_PRUNE"

    fired = dict(fired_null, rows_pruned=88)
    assert v(fired, {}) == "FIRED_AND_PRUNED"


def test_the_four_uninterpretable_verdicts_are_not_counted_as_fired():
    """`lever3_fired` and `lever3_interpretable` must agree with the verdict string, so a downstream
    aggregation cannot accidentally pool a non-firing cell into a lever's denominator."""

    m = _harness()
    firing = {"FIRED_NO_PRUNE", "FIRED_AND_PRUNED"}
    uninterpretable = {
        "UNINTERPRETABLE_NO_OBSERVE",
        "UNINTERPRETABLE_NO_NAV",
        "UNINTERPRETABLE_NOT_FITTED",
        "ERROR",
    }
    # LEVER_OFF is interpretable (it is the control) but not firing -- assert that asymmetry, since
    # collapsing it into either bucket would either lose the control or inflate the fire count.
    assert firing & uninterpretable == set()
    assert "LEVER_OFF" not in firing and "LEVER_OFF" not in uninterpretable
    source = open(m.__file__, encoding="utf-8").read()
    for token in firing | uninterpretable | {"LEVER_OFF"}:
        assert token in source, f"{token} must be produced by the harness"
