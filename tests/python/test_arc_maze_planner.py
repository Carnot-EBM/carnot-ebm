"""Unit tests for the reusable ARC maze planners (python/carnot/agentic/arc_maze_planner.py).

These pin the two solving strategies on SYNTHETIC models (game-independent, no env): the
checkpoint_multirun planner (stage a path across checkpoints when it exceeds one run's slot budget)
and the timed_trap planner (cross a blinking spike band only during its invisible window, clear by
the toggle). Correctness is checked by an INDEPENDENT re-walk of the produced plan through the model
dynamics — a plan must reach the target with no spike death. (The tn36 reproduction gate is the
end-to-end real-env check; these pin the algorithm itself.)

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the strategy-class solvers the router dispatches).
"""

from carnot.agentic import arc_maze_planner as planner
from carnot.agentic.arc_maze_planner import MazeModel

# command codes for the synthetic games (arbitrary, just distinct): up/down/left/right + settle
_MOVES = [((0, -4), 1), ((0, 4), 2), ((-4, 0), 3), ((4, 0), 4)]
_SETTLE = 0
_DELTA = {c: d for d, c in _MOVES}


def _walk(model, plan):
    """Replay a multi-leg plan through the model dynamics; return (final_pos, died). Mirrors the real
    'object resets to base each run unless it ends on a checkpoint (which advances the base)' rule and
    the spike-visibility schedule, as an INDEPENDENT check of the planner's output."""
    base = model.start
    pos = base
    for leg in plan:
        pos = base
        for idx, code in enumerate(leg):
            dx, dy = _DELTA.get(code, (0, 0))
            nx, ny = pos[0] + dx, pos[1] + dy
            if planner._collide(model, nx, ny):
                nx, ny = pos  # wall reverts
            visible = idx >= model.invisible_slots
            if planner._spike_death(model, nx, ny, visible):
                return (nx, ny), True
            if idx == model.invisible_slots - 1 and planner._spike_death(model, nx, ny, True):
                return (nx, ny), True
            pos = (nx, ny)
        if pos in model.checkpoints:  # ending on a checkpoint advances base
            base = pos
    return pos, False


def _model(**kw):
    base = dict(
        object_wh=(4, 4),
        walls=[],
        checkpoints=[],
        move_codes=_MOVES,
        settle_code=_SETTLE,
        n_slots=6,
        bounds=64,
    )
    base.update(kw)
    return MazeModel(**base)


def test_checkpoint_multirun_direct_leg():
    # target reachable within one run's slots, no checkpoint needed -> a single padded leg.
    m = _model(start=(8, 8), target=(8, 16))
    plan = planner.checkpoint_multirun_plan(m)
    assert plan is not None and len(plan) == 1
    assert all(len(leg) == m.n_slots for leg in plan)  # padded to the slot budget
    assert _walk(m, plan) == ((8, 16), False)


def test_checkpoint_multirun_stages_via_checkpoint():
    # target is 6 steps away but each run has only 3 slots -> must stage via the checkpoint.
    m = _model(start=(4, 4), target=(4, 28), checkpoints=[(4, 16)], n_slots=3)
    plan = planner.checkpoint_multirun_plan(m)
    assert plan is not None and len(plan) == 2  # two legs: start->cp, cp->target
    assert _walk(m, plan) == ((4, 28), False)


def test_checkpoint_multirun_unreachable_returns_none():
    # target fully walled in -> no plan.
    walls = [(4, 0, 4, 4), (4, 8, 4, 4), (0, 4, 4, 4), (8, 4, 4, 4)]
    m = _model(start=(40, 40), target=(4, 4), walls=walls)
    assert planner.checkpoint_multirun_plan(m) is None


def test_timed_trap_requires_spikes():
    # no spikes declared -> the timed planner declines (use the plain checkpoint planner).
    m = _model(start=(8, 24), target=(8, 8))
    assert planner.timed_trap_plan(m) is None


def test_timed_trap_crosses_during_invisible_window():
    # a full-row spike band at y[16,20); starting at y24, 3 ups clear the band by the slot-2 toggle
    # (out of band when it turns visible) -> a safe single-leg crossing the timed planner must find.
    band = [(0, 16, 64, 4)]
    m = _model(
        start=(8, 24), target=(8, 8), spikes_visible=band, spikes_hidden=[], invisible_slots=3
    )
    plan = planner.timed_trap_plan(m)
    assert plan is not None
    pos, died = _walk(m, plan)
    assert pos == (8, 8) and not died  # reached target, never on a visible spike


def test_timed_trap_none_when_band_cannot_be_cleared_in_time():
    # starting at y28 with a full-row visible band and NO checkpoint, the object cannot clear the band
    # by the slot-2 toggle (3 ups only reach y16, in-band) -> no safe crossing exists.
    band = [(0, 16, 64, 4)]
    m = _model(
        start=(8, 28), target=(8, 8), spikes_visible=band, spikes_hidden=[], invisible_slots=3
    )
    assert planner.timed_trap_plan(m) is None
