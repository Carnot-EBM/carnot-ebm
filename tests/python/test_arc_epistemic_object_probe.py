"""Tests for the generic ARC epistemic object-hypothesis probe planner.

Spec refs: REQ-ARC-WMTE-5630,
SCENARIO-ARC-WMTE-5630-INFORMATIVE-PROBE-POSITIVE,
SCENARIO-ARC-WMTE-5630-NEGATIVE-AND-UNSAFE-REJECTION.
"""

from __future__ import annotations

import math

import numpy as np

from carnot.agentic.arc_epistemic_object_probe import (
    EpistemicObjectProbePlanner,
    LiveProbeAction,
    ObjectProbeObservation,
    make_corrupted_effect_hypothesis,
    make_hallucinated_object_hypothesis,
)


def _two_object_grid(*, left_color: int = 5, right_color: int = 7) -> np.ndarray:
    grid = np.zeros((8, 12), dtype=np.int16)
    grid[2:4, 2:4] = left_color
    grid[2:4, 8:10] = right_color
    return grid


def _click_changes_object(
    *,
    trace: str,
    step: int,
    x: int,
    y: int,
    replacement: int = 9,
    left_color: int = 5,
    right_color: int = 7,
) -> ObjectProbeObservation:
    before = _two_object_grid(left_color=left_color, right_color=right_color)
    after = before.copy()
    if x < 6:
        after[2:4, 2:4] = replacement
    else:
        after[2:4, 8:10] = replacement
    return ObjectProbeObservation(
        trace_id=trace,
        step=step,
        state=before,
        action=6,
        data={"x": x, "y": y},
        successor=after,
        level_before=0,
        level_after=0,
    )


def test_scenario_5630_informative_probe_positive_updates_posterior() -> None:
    """SCENARIO-ARC-WMTE-5630-INFORMATIVE-PROBE-POSITIVE: an executable click that
    separates object hypotheses must beat random and reduce held-out entropy."""

    planner = EpistemicObjectProbePlanner(random_seed=5630)
    model = planner.build_trace_model(
        "toy",
        [_click_changes_object(trace="toy", step=0, x=3, y=3)],
    )
    legal = [
        LiveProbeAction(1, None),
        LiveProbeAction(6, {"x": 3, "y": 3}),
        LiveProbeAction(6, {"x": 9, "y": 3}),
    ]

    scores = planner.score_probes(model, _two_object_grid(), legal)
    chosen = planner.choose_probe(model, _two_object_grid(), legal)

    assert model.is_non_degenerate is True
    assert len(model.hypotheses) >= 2
    assert math.isclose(sum(model.weights.values()), 1.0)
    assert chosen is not None
    assert chosen.action == LiveProbeAction(6, {"x": 9, "y": 3})
    assert scores[0].expected_disagreement_reduction > 0.0
    assert scores[0].executable == 1.0
    assert scores[0].score > scores[-1].score

    control = planner.compare_controls(
        model,
        _two_object_grid(),
        legal,
        observed=_click_changes_object(trace="toy", step=1, x=9, y=3),
    )
    assert control["informative_control_delta"] > 0.0
    assert control["random_control_entropy_reduction"] < control["informative_entropy_reduction"]
    assert control["live_interface_replay_rate"] == 1.0
    assert model.weights["clicked_object_effect"] > model.weights["observed_object_anchor_effect"]


def test_scenario_5630_negative_controls_and_unsafe_models_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-5630-NEGATIVE-AND-UNSAFE-REJECTION: uninformative actions,
    corrupt effects, hallucinated objects, and no-object traces are not accepted."""

    planner = EpistemicObjectProbePlanner(random_seed=5630)
    model = planner.build_trace_model(
        "toy",
        [_click_changes_object(trace="toy", step=0, x=3, y=3)],
    )
    legal = [LiveProbeAction(1, None)]
    uninformative = planner.compare_controls(
        model,
        _two_object_grid(),
        legal,
        observed=ObjectProbeObservation(
            trace_id="toy",
            step=2,
            state=_two_object_grid(),
            action=1,
            data=None,
            successor=_two_object_grid(),
        ),
    )

    assert uninformative["uninformative_control_delta"] <= 0.0
    assert planner.reject_unsafe_models(
        model,
        [
            make_corrupted_effect_hypothesis(model.hypotheses[0]),
            make_hallucinated_object_hypothesis("ffffffffffffffff"),
        ],
        _two_object_grid(),
    ) == 0

    no_object = np.zeros((6, 6), dtype=np.int16)
    closed = planner.build_trace_model(
        "blank",
        [
            ObjectProbeObservation(
                trace_id="blank",
                step=0,
                state=no_object,
                action=6,
                data={"x": 1, "y": 1},
                successor=no_object.copy(),
            )
        ],
    )
    assert closed.is_non_degenerate is False
    assert planner.choose_probe(closed, no_object, [LiveProbeAction(6, {"x": 1, "y": 1})]) is None


def test_scenario_5630_object_permutation_preserves_hypothesis_weights() -> None:
    """SCENARIO-ARC-WMTE-5630-NEGATIVE-AND-UNSAFE-REJECTION: posterior weights depend
    on object hashes and effects, not absolute object positions."""

    planner = EpistemicObjectProbePlanner(random_seed=5630)
    original = planner.build_trace_model(
        "original",
        [_click_changes_object(trace="original", step=0, x=3, y=3)],
    )
    permuted = planner.build_trace_model(
        "permuted",
        [
            _click_changes_object(
                trace="permuted",
                step=0,
                x=9,
                y=3,
                left_color=7,
                right_color=5,
            )
        ],
    )

    assert original.hypothesis_names == permuted.hypothesis_names
    assert original.weights == permuted.weights
