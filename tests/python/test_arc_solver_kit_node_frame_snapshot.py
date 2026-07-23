"""Regression test for a real bug found via direct diagnostic tracing (2026-07-23, investigating an
anomalously shallow lf52 search under REQ-ARC-WMTE-5828's large-budget tool-loop-lookahead run):
`OfflineSolver.solve_level()` (and its `_solve_level_deepcopy`/`_solve_level_fresh` siblings) captured
`node_frame = self.last_frame` ONCE before iterating a node's sibling candidates, then passed that same
`node_frame` reference to `move_pruner.observe()` for EVERY sibling. Confirmed via a live trace against
the real lf52 offline env: `env.step()` returns a DISTINCT Python object each call (`f1 is f0` is False),
but the underlying grid data is a SHARED, mutated-in-place buffer -- so `node_frame`, read at
`observe()`-call time (after later `apply()`/`env.step()` calls in the same sibling loop), silently
reflected the CURRENT env state rather than the state at capture time. This made `observe()`'s
before/after keys look identical for every genuinely state-changing action, corrupting the dead-end
pruner with false dead-ends. Fix: `copy.deepcopy(self.last_frame)` at each capture site, breaking the
aliasing. This test reproduces the exact aliasing shape (a frame wrapping a shared mutable list) on a
minimal toy env, independent of the real arcengine dependency.

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-5828 (the incident that
surfaced this bug); the fix itself is infrastructure-level (arc_solver_kit.py), not scoped to a REQ.
"""

from __future__ import annotations

from carnot.agentic.arc_solver_kit import OfflineSolver


class _MutatingFrame:
    """Wraps a SHARED mutable list -- mirrors the real arcengine.FrameDataRaw aliasing behavior found
    via direct trace: each step() returns a NEW wrapper object, but its underlying grid data is the
    SAME list, mutated in place. `.value` is read fresh each access (like hashing `.frame` at
    hash-time), not snapshotted at construction."""

    def __init__(self, shared_grid: list) -> None:
        self._shared_grid = shared_grid  # reference, NOT a copy

    @property
    def value(self):
        return tuple(self._shared_grid)


class _MutatingGame:
    __slots__ = ("pos",)

    def __init__(self, pos: int = 0) -> None:
        self.pos = pos


class _MutatingEnv:
    """A 1-D toy game whose step() mutates a SHARED buffer in place, exactly reproducing the real
    aliasing shape (distinct wrapper objects, shared underlying data) found in the real env."""

    def __init__(self) -> None:
        self._grid = [0]
        self._game = _MutatingGame(0)

    def reset(self):
        self._grid[0] = 0
        self._game = _MutatingGame(0)
        return _MutatingFrame(self._grid)

    def step(self, label, **kw):
        self._grid[0] = max(0, self._grid[0] + (1 if label == "R" else -1))
        self._game.pos = self._grid[0]
        return _MutatingFrame(self._grid)  # NEW object, SAME shared_grid reference


def _labels(env, frame=None, path=None):
    return ["R", "L"]  # two candidates per node -- required to expose the aliasing bug


def _apply(env, label, frame):
    return env.step(label)


def _state_key(game, frame=None):
    return game.pos


def _verifier(game, frame=None):
    return 0.0  # irrelevant to this test -- plain BFS ordering is fine


class _RecordingPruner:
    """Never actually prunes -- just records exactly what solve_level passes as `frame_before` to
    observe(), by VALUE (read immediately, not held as a live reference), so the test can assert on a
    true point-in-time snapshot rather than being fooled by the same aliasing bug it's testing for."""

    def __init__(self) -> None:
        self.observed: list[tuple] = []

    def should_prune(self, frame, label) -> bool:
        return False

    def observe(self, frame_before, label, frame_after, leveled_up) -> None:
        self.observed.append((frame_before.value, label, frame_after.value, leveled_up))


def test_node_frame_snapshot_is_not_corrupted_by_sibling_processing():
    env = _MutatingEnv()
    pruner = _RecordingPruner()
    solver = OfflineSolver(
        "toy_mutating",
        _labels,
        _apply,
        _state_key,
        verifier=_verifier,
        branch_mode="replay",
        move_pruner=pruner,
    )
    solver.solve_level(env, start_level=0, prefix=[], depth_cap=1)

    # The root's two siblings ("R" then "L") were both observed. BOTH observe() calls must report the
    # SAME before-value: the ROOT's actual starting position (0) -- NOT whatever the shared buffer
    # happened to hold at the moment observe() was called (which, pre-fix, was contaminated by
    # whichever sibling had already been applied).
    assert len(pruner.observed) == 2
    for before_value, label, after_value, leveled in pruner.observed:
        assert before_value == (0,), (
            f"observe() for label={label!r} saw before_value={before_value}, expected (0,) "
            "(the ROOT's real starting state) -- node_frame was corrupted by a later sibling's "
            "apply() call, the exact aliasing bug this test guards against."
        )
    # And each sibling's own after-value is genuinely distinct from the (correct) before-value --
    # confirming this isn't vacuously passing because nothing actually changed.
    after_values = {a for _, _, a, _ in pruner.observed}
    assert after_values == {(1,), (0,)}  # "R" -> pos 1, "L" clamped at pos 0
