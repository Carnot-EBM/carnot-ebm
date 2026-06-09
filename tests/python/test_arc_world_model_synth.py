"""Tests for the M2 InducedWorldModel + grid-grounded consistency-energy verifier.

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). Asserts the load-bearing invariants:
the exact table reproduces seen transitions; the click template GENERALIZES a learned local effect to
an unseen click location; the consistency energy is 0 on a consistent (deterministic) held-out set and
> 0 on a contradictory (hidden-state-like) one — i.e. the verifier correctly flags an untrustworthy
model. Every test asserts (no skips) per CLAUDE.md "Tests Must Run and Assert".
"""

import numpy as np

from carnot.agentic.arc_world_model_synth import InducedWorldModel


def _toggle(pos, h=5, w=5, base=0, on=2, off=5):
    """A grid with `on` at pos, and the grid after a click there turns it `off` (a local toggle)."""
    s = np.full((h, w), base, dtype=np.int16); s[pos] = on
    s2 = s.copy(); s2[pos] = off
    return s, s2


def test_exact_table_predicts_seen_transition():
    s, s2 = _toggle((1, 1))
    m = InducedWorldModel("t").fit([(s, (6, 1, 1), s2)])
    pred = m.predict(s, (6, 1, 1))
    assert np.array_equal(pred, s2)


def test_click_template_generalizes_to_unseen_location():
    # learn "clicking a color-2 cell turns it to 5" from three positions...
    train = [(*_toggle(p), ) for p in [(0, 0), (1, 1), (2, 2)]]
    train = [(s, (6, p[1], p[0]), s2) for (s, s2), p in zip([_toggle(p) for p in [(0, 0), (1, 1), (2, 2)]],
                                                            [(0, 0), (1, 1), (2, 2)])]
    m = InducedWorldModel("t").fit(train)
    # ...then predict an UNSEEN click at (3,3): the color-2 cell there should toggle to 5
    s3, s3_expected = _toggle((3, 3))
    pred = m.predict(s3, (6, 3, 3))          # x=3, y=3
    assert pred[3, 3] == 5                    # generalized the local effect
    assert np.array_equal(pred, s3_expected)


def test_consistency_energy_zero_on_consistent_heldout():
    train = [(s, (6, p[1], p[0]), s2)
             for (s, s2), p in ((_toggle(p), p) for p in [(0, 0), (1, 1), (2, 2), (4, 4)])]
    m = InducedWorldModel("t").fit(train)
    s_h, s_h2 = _toggle((3, 3))               # same deterministic rule, unseen location
    ce = m.consistency_energy([(s_h, (6, 3, 3), s_h2)])
    assert ce["energy"] == 0.0                # model captures the dynamics -> trustworthy
    assert ce["dynamics_accuracy"] == 1.0


def test_consistency_energy_high_on_contradiction():
    # same (state, click) observed with TWO different outcomes = hidden-state / multivalued
    s = np.zeros((4, 4), dtype=np.int16); s[1, 1] = 2
    s_a = s.copy(); s_a[1, 1] = 5            # outcome A
    s_b = s.copy(); s_b[1, 1] = 7            # outcome B (contradicts A for the same input)
    # train mostly on A so the modal prediction is A...
    m = InducedWorldModel("t").fit([(s, (6, 1, 1), s_a)] * 3)
    ce = m.consistency_energy([(s, (6, 1, 1), s_b)])   # held-out is the minority outcome
    assert ce["energy"] is not None and ce["energy"] > 0.0   # verifier flags the model as wrong here
    assert ce["dynamics_accuracy"] < 1.0


def test_is_trustworthy_gate():
    train = [(s, (6, p[1], p[0]), s2)
             for (s, s2), p in ((_toggle(p), p) for p in [(0, 0), (1, 1), (2, 2)])]
    m = InducedWorldModel("t").fit(train)
    s_h, s_h2 = _toggle((3, 3))
    assert m.is_trustworthy([(s_h, (6, 3, 3), s_h2)], energy_threshold=0.2) is True
    # a contradictory held-out should NOT be trustworthy
    bad = s_h.copy(); bad[0, 0] = 9          # an effect the model never predicts
    assert m.is_trustworthy([(s_h, (6, 3, 3), bad)], energy_threshold=0.2) is False
