"""Tests for the prior-warm-started, trust-GATED live engine wired into the agent (2026-06-21).

REQ: the live agent tries a per-game world model LEARNED from the played transitions (warm-started from
the cross-game prior) BEFORE the LLM induction, but only USES it when it reproduces a held-out split at
>= trust_threshold -- so a weak learned model can never displace the existing fallback path. The gate is
what makes wiring prior_state into the live agent SAFE.

SCENARIO-GATE-1: too few transitions -> the helper returns (None, None) and skips (no engine).
SCENARIO-GATE-2: a learnable mechanic (gravity) -> the gate PASSES (held-out accuracy >= threshold) and a
                 usable engine + is_level_complete are returned.
SCENARIO-GATE-3: an impossibly high threshold -> the gate FAILS and returns None (the fallback is preserved).
"""
import numpy as np
import torch  # noqa: F401  -- module-level so torch's ~600MB base alloc is charged to COLLECTION, not the
#                test (the pytest memory-watchdog flags a >500MB per-test RSS delta; lazy-importing torch
#                inside the gated helper would otherwise charge that base footprint to whichever test runs first)

from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_live_ttt import gated_engine_from_transitions


def _gravity_transitions(n: int = 80):
    rng = np.random.default_rng(0)

    def step(g):
        g2 = g.copy()
        ys, xs = np.where(g > 0)
        for y, x in zip(ys, xs):
            if y + 1 < g.shape[0]:
                g2[y, x] = 0
                g2[y + 1, x] = g[y, x]
        return g2

    out = []
    for _ in range(n):
        g = np.zeros((6, 6), dtype=np.int16)
        y, x = int(rng.integers(0, 5)), int(rng.integers(0, 6))
        g[y, x] = int(rng.integers(1, 6))
        out.append(Transition(grid=g, action=3, data=None, next_grid=step(g), level_before=0, level_after=0))
    return out


def test_scenario_gate_1_too_few_transitions_skips() -> None:
    """SCENARIO-GATE-1: below the minimum, no engine is built (skip), preserving the fallback."""
    eng, isdone, diag = gated_engine_from_transitions("x", [])
    assert eng is None and isdone is None
    assert diag.get("skip") == "too_few_transitions"


def test_scenario_gate_2_learnable_mechanic_passes() -> None:
    """SCENARIO-GATE-2: a learnable gravity rule passes the gate and yields a usable engine."""
    eng, isdone, diag = gated_engine_from_transitions("grav", _gravity_transitions(), trust_threshold=0.5)
    assert diag["gate"] == "PASS"
    assert diag["heldout_accuracy"] >= 0.5
    assert eng is not None and callable(eng) and callable(isdone)
    # the returned engine actually predicts the gravity rule on a fresh grid
    g = np.zeros((6, 6), dtype=np.int16)
    g[1, 2] = 4
    nxt = np.asarray(eng(g, 3, None))
    assert nxt[2, 2] == 4 and nxt[1, 2] == 0


def test_scenario_gate_3_unreachable_threshold_falls_back() -> None:
    """SCENARIO-GATE-3: an impossible threshold makes the gate fail -> None (fallback preserved)."""
    eng, isdone, diag = gated_engine_from_transitions("grav", _gravity_transitions(), trust_threshold=1.01)
    assert eng is None and isdone is None
    assert diag["gate"] == "FAIL"
