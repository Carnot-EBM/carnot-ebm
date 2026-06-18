"""Tests for the M2-v2 ObjectDeltaModel (object-level delta-DSL inducer).

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). Asserts the load-bearing upgrade over
the M2-v1a naive template: object-level rules GENERALIZE across positions. A keyboard-translate rule
learned at a few agent positions predicts an UNSEEN position; a click-recolor rule learned on a few
objects predicts an UNSEEN object. Every test asserts (no skips) per CLAUDE.md "Tests Must Run and
Assert".
"""

import numpy as np

from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel


def _agent(pos, agent=3, h=6, w=6):
    s = np.zeros((h, w), dtype=np.int16); s[pos] = agent
    return s


def test_translate_rule_generalizes_to_unseen_position():
    # ACTION1 moves the color-3 agent up by one row; learn at three positions...
    train = []
    for (y, x) in [(2, 1), (3, 2), (2, 3)]:
        train.append((_agent((y, x)), (1,), _agent((y - 1, x))))
    m = ObjectDeltaModel("t").fit(train)
    assert m.kbd_rules[1] == ("translate", 3, -1, 0)
    pred = m.predict(_agent((4, 4)), (1,))            # unseen start position
    assert np.array_equal(pred, _agent((3, 4)))       # agent moved up, generalized


def test_recolor_click_generalizes_to_unseen_object():
    # clicking a color-2 object recolors it to 5; learn on two objects...
    def obj(cells, color=2, h=6, w=6):
        s = np.zeros((h, w), dtype=np.int16)
        for (y, x) in cells:
            s[y, x] = color
        return s
    train = []
    for cells in [[(1, 1), (1, 2)], [(4, 4), (4, 5)]]:
        s = obj(cells)
        s2 = s.copy()
        for (y, x) in cells:
            s2[y, x] = 5
        train.append((s, (6, cells[0][1], cells[0][0]), s2))   # click first cell (x, y)
    m = ObjectDeltaModel("t").fit(train)
    assert m.click_rules[2] == ("recolor_clicked", 5)
    # unseen object at a new location
    s3 = obj([(2, 0), (3, 0)])                         # vertical 2-cell color-2 object
    pred = m.predict(s3, (6, 0, 2))                    # click (x=0, y=2)
    expect = s3.copy(); expect[2, 0] = 5; expect[3, 0] = 5
    assert np.array_equal(pred, expect)


def test_consistency_energy_low_on_generalized_dynamics():
    train = [(_agent((y, x)), (1,), _agent((y - 1, x))) for (y, x) in [(2, 1), (3, 2), (2, 3), (4, 1)]]
    m = ObjectDeltaModel("t").fit(train)
    held = [(_agent((5, 5)), (1,), _agent((4, 5)))]    # unseen position, same rule
    ce = m.consistency_energy(held)
    assert ce["energy"] == 0.0                          # generalized -> trustworthy
    assert ce["dynamics_accuracy"] == 1.0


def test_noop_rule_when_action_has_no_consistent_effect():
    # ACTION2 does nothing in training -> noop rule -> predicts unchanged
    train = [(_agent((2, 2)), (2,), _agent((2, 2)))]
    m = ObjectDeltaModel("t").fit(train)
    assert m.kbd_rules[2] == ("noop",)
    s = _agent((1, 1))
    assert np.array_equal(m.predict(s, (2,)), s)


def _agent_and_goal(apos, color=3, h=7, w=7):
    """A movable 2-cell agent AND a STATIC 1-cell goal at (0,0), BOTH color 3 -- the case where
    per-color-global translate fails (the union of color-3 cells does not rigidly translate) but
    per-OBJECT translate succeeds. The agent has a DISTINCT shape from the goal so the per-object
    rule identifies it by shape (v1 limitation: identical-shaped movable+static objects are
    ambiguous to shape-based identity)."""
    s = np.zeros((h, w), dtype=np.int16)
    s[0, 0] = color                                   # static 1-cell goal
    ay, ax = apos
    s[ay, ax] = color; s[ay, ax + 1] = color          # movable 2-cell horizontal agent
    return s


def test_per_object_translate_when_same_color_object_is_static():
    # ACTION1 moves ONLY the agent up; the same-colored goal at (0,0) stays. Per-color-global
    # translate cannot fit; per-object translate identifies the 2-cell agent by shape and moves it.
    train = [(_agent_and_goal((y, x)), (1,), _agent_and_goal((y - 1, x)))
             for (y, x) in [(3, 1), (4, 3), (3, 5)]]
    m = ObjectDeltaModel("t").fit(train)
    rule = m.kbd_rules[1]
    assert rule[0] == "translate_obj", f"expected per-object translate, got {rule}"
    assert rule[1] == 3 and (rule[3], rule[4]) == (-1, 0)        # color 3, moved up by one
    # generalizes to an UNSEEN agent position while the static goal is preserved
    pred = m.predict(_agent_and_goal((5, 2)), (1,))
    assert np.array_equal(pred, _agent_and_goal((4, 2)))         # agent moved up, goal at (0,0) intact


def test_per_object_translate_does_not_regress_single_object_translate():
    # the original single-agent case must still induce the simpler per-COLOR translate (back-compat)
    train = [(_agent((y, x)), (1,), _agent((y - 1, x))) for (y, x) in [(2, 1), (3, 2), (2, 3)]]
    m = ObjectDeltaModel("t").fit(train)
    assert m.kbd_rules[1] == ("translate", 3, -1, 0)            # simpler rule kept on tie


def _agent_with_hud(apos, hud_color, agent=3, h=6, w=6):
    """An agent (color 3) plus a single HUD/status cell at (0,0) whose color encodes step state."""
    s = np.zeros((h, w), dtype=np.int16)
    s[0, 0] = hud_color
    s[apos] = agent
    return s


def test_composite_rule_move_and_recolor_in_one_action():
    # ACTION1 moves the agent up AND flips the HUD cell from 7 to 8 -- ONE action, TWO effects.
    # A single rule can express only one; the greedy composer must induce a ('seq', [...]) of both.
    train = []
    for (y, x) in [(2, 1), (3, 2), (2, 3), (4, 2)]:
        train.append((_agent_with_hud((y, x), 7), (1,), _agent_with_hud((y - 1, x), 8)))
    m = ObjectDeltaModel("t").fit(train)
    rule = m.kbd_rules[1]
    assert rule[0] == "seq", f"expected composite seq, got {rule}"
    kinds = {r[0] for r in rule[1]}
    assert "recolor_all" in kinds and ("translate" in kinds or "translate_obj" in kinds)
    # generalizes to an unseen position: agent moves up AND HUD recolors 7->8
    pred = m.predict(_agent_with_hud((5, 4), 7), (1,))
    assert np.array_equal(pred, _agent_with_hud((4, 4), 8))
