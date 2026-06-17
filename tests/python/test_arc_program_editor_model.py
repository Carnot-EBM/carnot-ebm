"""Unit tests for the OFFLINE program-editor transition model
(python/carnot/agentic/arc_program_editor_model.py).

The model is the planning gradient the atomic-run frame stream withholds: it predicts the transition
(object_attrs, program) -> final_attrs, scores a candidate program by distance-to-target, and drives a
model-guided program search. These pin the per-code semantics (move/rotate/scale/property + wall
revert), the distance gradient, win prediction, and that the guided planner finds a solution — on
synthetic states, game-independent. (The tn36 win-bit agreement vs the real env — 105/105 — is the
end-to-end oracle check; these pin the algorithm.)

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the program-editor planning instrument).
"""

from carnot.agentic.arc_program_editor_model import (
    EditorGeometry,
    EditorState,
    apply_code,
    attribute_distance,
    plan_program,
    predict_win,
    simulate,
)

_OPEN = EditorGeometry(object_wh=(4, 4), walls=(), bounds=64)


def test_per_code_semantics():
    s = EditorState(x=20, y=20, scale=1, rotation=0, prop=11)
    assert apply_code(s, 3, _OPEN).y == 24 and apply_code(s, 33, _OPEN).y == 16  # down/up ±STEP
    assert apply_code(s, 1, _OPEN).x == 16 and apply_code(s, 2, _OPEN).x == 24  # left/right ±STEP
    assert apply_code(s, 10, _OPEN).x == 28  # right ×2
    assert apply_code(s, 5, _OPEN).rotation == 90  # rotate +90
    assert apply_code(EditorState(20, 20, 1, 270, 11), 6, _OPEN).rotation == 180  # -90 mod 360
    assert apply_code(s, 8, _OPEN).scale == 2  # scale +1
    assert apply_code(s, 9, _OPEN).scale == 1  # scale clamps to >=1
    assert apply_code(s, 63, _OPEN).prop == 15  # property absolute set
    assert apply_code(s, 0, _OPEN) == s  # settle = no-op


def test_move_reverts_on_wall_and_bounds():
    geom = EditorGeometry(object_wh=(4, 4), walls=((24, 20, 4, 4),), bounds=64)
    s = EditorState(x=20, y=20, scale=1, rotation=0, prop=11)
    assert apply_code(s, 2, geom) == s  # right into the wall reverts
    edge = EditorState(x=60, y=20, scale=1, rotation=0, prop=11)
    assert apply_code(edge, 2, geom) == edge  # right out of bounds reverts
    # the wall footprint scales with `scale`: a 2x object at x16 can't move right past x20 wall-adjacent
    big = EditorState(x=16, y=20, scale=2, rotation=0, prop=11)
    assert apply_code(big, 2, geom) == big


def test_distance_is_zero_at_target_and_strictly_decreases_toward_it():
    target = EditorState(x=40, y=20, scale=1, rotation=0, prop=11)
    assert attribute_distance(target, target) == 0
    s = EditorState(x=20, y=20, scale=1, rotation=0, prop=11)  # 5 steps left of target
    assert attribute_distance(apply_code(s, 2, _OPEN), target) < attribute_distance(s, target)


def test_predict_win_true_only_for_a_reaching_program():
    s = EditorState(x=20, y=20, scale=1, rotation=0, prop=11)
    target = EditorState(x=28, y=20, scale=1, rotation=0, prop=11)  # 2 rights
    assert predict_win(s, target, [2, 2], _OPEN) is True
    assert predict_win(s, target, [2, 0], _OPEN) is False  # only 1 right -> loses
    assert simulate(s, [2, 2], _OPEN).matches(target)


def test_guided_planner_finds_a_solution_the_model_confirms():
    s = EditorState(x=20, y=24, scale=1, rotation=90, prop=11)
    target = EditorState(
        x=28, y=16, scale=2, rotation=0, prop=15
    )  # move + rotate + scale + property
    prog = plan_program(s, target, _OPEN, n_slots=8)
    assert prog is not None and len(prog) == 8
    assert simulate(s, prog, _OPEN).matches(target)  # the plan actually reaches target


def test_guided_planner_returns_none_when_unreachable_in_budget():
    s = EditorState(x=20, y=20, scale=1, rotation=0, prop=11)
    target = EditorState(x=40, y=20, scale=1, rotation=0, prop=11)  # needs 5 rights
    assert plan_program(s, target, _OPEN, n_slots=2) is None  # only 2 slots -> impossible
