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
