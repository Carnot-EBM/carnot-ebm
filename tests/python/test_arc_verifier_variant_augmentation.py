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
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_value_learner import (  # noqa: E402
    cross_game_feature_slices_v3,
    cross_game_features,
    cross_game_features_v2,
    cross_game_features_v3,
)
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


def test_auroc_helper():
    m = _train_mod()
    assert m._auroc([0.1, 0.2, 0.9, 0.8], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(1.0)  # perfect
    assert m._auroc([0.9, 0.8, 0.1, 0.2], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0)  # inverted
    assert m._auroc([0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.5)  # ties


def test_discriminative_verifier_separates_synthetic():
    # the win-reachability classifier must actually learn a separable boundary (the per-game
    # discrimination the steps-to-go regressor lacks). Identity featurize over raw vectors.
    from carnot.agentic.arc_value_learner import DiscriminativeVerifier

    rng = np.random.default_rng(0)
    pos = rng.normal(2.0, 0.5, size=(60, 4))
    neg = rng.normal(-2.0, 0.5, size=(60, 4))
    X = np.vstack([pos, neg]).tolist()
    y = [1.0] * 60 + [0.0] * 60
    clf = DiscriminativeVerifier(lambda v: v).fit(X, y)
    assert clf.proba(pos[0].tolist()) > 0.7   # confidently on-path
    assert clf.proba(neg[0].tolist()) < 0.3   # confidently off-path


def _synthetic_frame(grid, levels=0):
    return SimpleNamespace(frame=np.asarray(grid, dtype=int).tolist(), levels_completed=levels)


def _slice(values, name):
    start, stop = cross_game_feature_slices_v3()[name]
    return np.asarray(values[start:stop], dtype=float)


def test_v3_feature_slices_cover_relational_delta_action_and_predicate_context():
    # Spec: REQ-LEARN-4476, SCENARIO-LEARN-4476-FEATURES.
    prev = _synthetic_frame(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 0],
            [0, 1, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        levels=0,
    )
    cur = _synthetic_frame(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 2, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        levels=1,
    )
    goal = _synthetic_frame(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 2, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        levels=1,
    )

    values = cross_game_features_v3(cur, previous_frame=prev, action_id=4, goal_frame=goal)
    frame_only = cross_game_features_v3(cur)
    slices = cross_game_feature_slices_v3()

    assert len(values) > len(cross_game_features_v2(cur))
    assert set(slices) >= {
        "v2",
        "object_relational",
        "frame_delta",
        "action_conditioned",
        "predicate_distance",
    }
    assert np.any(_slice(values, "object_relational") != _slice(frame_only, "object_relational"))
    assert np.any(_slice(values, "frame_delta") != _slice(frame_only, "frame_delta"))
    assert np.any(_slice(values, "action_conditioned") != _slice(frame_only, "action_conditioned"))
    assert np.any(_slice(values, "predicate_distance") != _slice(frame_only, "predicate_distance"))


def test_v3_action_encoding_is_stable_and_game_agnostic():
    # Spec: REQ-LEARN-4476-3, SCENARIO-LEARN-4476-FEATURES.
    frame = _synthetic_frame([[0, 1], [0, 0]])
    action_2 = _slice(cross_game_features_v3(frame, action_id=2), "action_conditioned")
    action_6 = _slice(cross_game_features_v3(frame, action_id=6), "action_conditioned")

    assert action_2[0] == 1.0
    assert action_6[0] == 1.0
    assert action_2.sum() == pytest.approx(2.0)
    assert action_6.sum() == pytest.approx(2.0)
    assert not np.array_equal(action_2, action_6)


def test_v3_feature_names_and_edge_branches_are_stable():
    # Spec: REQ-LEARN-4476-1..4, SCENARIO-LEARN-4476-FEATURES.
    from carnot.agentic.arc_value_learner import cross_game_feature_names_v3

    frame = _synthetic_frame([[0, 1], [0, 2]])
    prev_1d = _synthetic_frame([0, 1, 0])
    empty = _synthetic_frame([[0, 0], [0, 0]])
    two_objects = _synthetic_frame([[0, 1, 0], [0, 0, 0], [2, 0, 3]])
    one_object = _synthetic_frame([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    tall_goal = _synthetic_frame([[0, 1], [0, 0], [2, 0]])

    names = cross_game_feature_names_v3()
    values = cross_game_features_v3(
        frame,
        previous_frame=prev_1d,
        action_id=(6, 1, 1),
        goal_frame=tall_goal,
    )
    assert len(names) == len(values)
    assert names[0] == "v2_0"
    assert names[-1].startswith("predicate_distance_")
    assert _slice(values, "frame_delta")[-1] == 1.0
    assert _slice(values, "predicate_distance")[1] == 1.0
    assert _slice(values, "action_conditioned")[6] == 1.0

    empty_values = cross_game_features_v3(empty, previous_frame=empty, goal_frame=empty)
    assert _slice(empty_values, "object_relational")[5] == 1.0
    assert _slice(empty_values, "frame_delta")[6] == 0.0
    assert _slice(empty_values, "predicate_distance")[5] == 0.0

    extra_current_values = cross_game_features_v3(two_objects, previous_frame=one_object)
    assert _slice(extra_current_values, "object_relational")[8] > 0.0


def test_v3_action_accepts_enum_string_and_invalid_forms():
    # Spec: REQ-LEARN-4476-3, SCENARIO-LEARN-4476-FEATURES.
    frame = _synthetic_frame([[0, 1], [0, 0]])

    enum_like = _slice(cross_game_features_v3(frame, action_id=SimpleNamespace(name="ACTION3")), "action_conditioned")
    string_like = _slice(cross_game_features_v3(frame, action_id="ACTION5"), "action_conditioned")
    invalid = _slice(cross_game_features_v3(frame, action_id="not-an-action"), "action_conditioned")

    assert enum_like[0] == 1.0 and enum_like[3] == 1.0
    assert string_like[0] == 1.0 and string_like[5] == 1.0
    assert invalid.sum() == pytest.approx(0.0)


def test_v3_empty_histogram_branch_is_defensive_zero():
    # Spec: REQ-LEARN-4476-2, REQ-LEARN-4476-4.
    from carnot.agentic.arc_value_learner import _color_hist_l1

    assert _color_hist_l1(np.array([]), np.array([])) == pytest.approx(0.0)


def test_exp4476_artifact_contract_has_required_terminal_fields():
    # Spec: REQ-LEARN-4476, SCENARIO-LEARN-4476-GATE.
    artifact = _train_mod()._build_exp4476_artifact(
        v2_metrics={"loo_auroc": 0.503, "in_sample_auroc": 0.726, "n_held_out_games": 5},
        v3_metrics={"loo_auroc": 0.612, "in_sample_auroc": 0.74, "n_held_out_games": 5},
        feature_class_loo_auroc={
            "object_relational": 0.58,
            "frame_delta": 0.61,
            "action_conditioned": 0.52,
            "predicate_distance": 0.57,
        },
        value_head_routing_measure={"ran": False, "artifact": "results/arc3_value_routing_v2.json"},
        tests_pass=True,
        preconditions_checked={"banked_trajectories": True, "offline_arcade": True},
        reproduced_levels=34,
    )

    for key in [
        "honest_verdict",
        "inference_substrate",
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
        "feature_class_loo_auroc",
        "feature_class_deltas",
        "value_head_routing_measure",
        "field_principles",
        "spec_refs",
        "reproducibility_checksum",
    ]:
        assert key in artifact
    assert artifact["honest_verdict"].startswith(("complete:", "complete_", "success:", "success_"))
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 34
    assert artifact["loo_gate_passed"] is True
    assert "SCENARIO-LEARN-4476-GATE" in artifact["spec_refs"]
