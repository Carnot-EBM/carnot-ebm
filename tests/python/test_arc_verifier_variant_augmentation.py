"""Regression tests for REFLECTION-based variant augmentation of the cross-game verifier corpus
(scripts/arc_cross_game_verifier_train.py).

Pins the HONEST constraint discovered + validated at build time (2026-06-19): color-permutation and the
v1 5-scalar features are augmentation-INVARIANT (the features are color-agnostic + symmetric by design),
so only REFLECTION + the v2 occupancy features produce genuinely new (non-duplicate) training points.
The whole value of the augmentation rests on that distinction, so it is the thing under test.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_value_learner import cross_game_features, cross_game_features_v2  # noqa: E402
from carnot.agentic.arc_variant_generator import reflect_grid  # noqa: E402


def _train_mod():
    spec = importlib.util.spec_from_file_location(
        "xgtrain", str(REPO / "scripts" / "arc_cross_game_verifier_train.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _one_frame():
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make("tr87", scorecard_id=arc.open_scorecard())
    return env.reset()


def test_reflect_grid_is_involution():
    g = np.arange(16).reshape(4, 4)
    assert np.array_equal(reflect_grid(reflect_grid(g, 1), 1), g)
    assert np.array_equal(reflect_grid(reflect_grid(g, 0), 0), g)


def test_steps_to_next_up_labels():
    # levels 0,0,1,1,2: level-ups at idx2 (0->1) and idx4 (1->2). Convention (preserved from the original
    # collector): dist=0 at the state JUST BEFORE a level-up (one action wins the level), incrementing
    # backward; the tail after the last level-up is unlabeled.
    nu = _train_mod()._steps_to_next_up([0, 0, 1, 1, 2])
    assert nu[1][0] == 0 and nu[0][0] == 1          # idx1 is one action from the idx2 up; idx0 is two
    assert nu[3][0] == 0 and nu[2][0] == 1          # idx3 is one action from the idx4 up; idx2 is two
    assert nu[4] is None or nu[4][0] is None        # no label past the last level-up


def test_v2_reflection_diversifies_but_v1_invariant():
    # THE load-bearing assertion: reflection adds signal ONLY with v2; v1 is invariant (so the augmentation
    # auto-selects v2 in main()). Without this, the augmentation silently produces duplicates.
    m = _train_mod()
    f = _one_frame()
    v1_base = np.array(cross_game_features(f))
    v2_base = np.array(cross_game_features_v2(f))
    v1_h = np.array(m._featurize_reflected(f, cross_game_features, 1))
    v2_h = np.array(m._featurize_reflected(f, cross_game_features_v2, 1))
    assert np.abs(v1_base - v1_h).sum() == pytest.approx(0.0, abs=1e-9)  # v1 reflection-invariant
    assert np.abs(v2_base - v2_h).sum() > 0.0                            # v2 genuinely diversified


def test_featurize_reflected_identity_is_noop():
    m = _train_mod()
    f = _one_frame()
    assert np.array_equal(
        np.array(m._featurize_reflected(f, cross_game_features_v2, None)),
        np.array(cross_game_features_v2(f)),
    )
