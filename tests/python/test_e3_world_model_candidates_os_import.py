"""Regression test: E3AgentPolicy._world_model_candidates must not crash with a
NameError on `os`.

Found 2026-07-12 while diagnosing why test_experiment_4821's live goal-energy
wiring test never reached `plan_in_model` (captured stayed empty ->
IndexError). Root cause: `_world_model_candidates` reads
`os.environ.get("CARNOT_ARC_POE_WORLD")` UNCONDITIONALLY as the first
executable line of the method, but the method has no local `import os` (this
file imports `os` only locally inside OTHER methods -- each function needs
its own import, a caller's local `import os` does not leak into a callee's
scope). Every single call to `_world_model_candidates` -- which is on the
CORE tier-3 world-model induction path (`_induce_and_plan` -> `load_engine`
-> `_world_model_candidates`) -- raised `NameError: name 'os' is not
defined`, silently caught by `_induce_and_plan`'s blanket
`except Exception: attempt["skipped"] = "exception"`. This has been live
since commit `4f3a4f1ef` (2026-06-28) -- roughly two weeks of the live
scored agent's tier-3 induction escalation silently no-op'ing on every
attempt, invisibly falling back to tier-1 exploration only.

Spec refs: REQ-ARC-WMTE-4491, REQ-ARC-WMTE-4494 (E3AgentPolicy's world-model
candidate collection, the code path this fix hardens).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_competition_agent import E3AgentPolicy, WorldModelCandidate


def _engine(grid: np.ndarray, _action: int, _data: object) -> np.ndarray:
    return np.asarray(grid)


def _is_done(grid: np.ndarray) -> bool:
    return not np.asarray(grid).any()


def test_world_model_candidates_does_not_raise_with_poe_world_unset(monkeypatch) -> None:
    """The bare, default case (CARNOT_ARC_POE_WORLD unset) must not crash."""

    monkeypatch.delenv("CARNOT_ARC_POE_WORLD", raising=False)
    policy = E3AgentPolicy("zz99", proposer=None, value_head=lambda _f: 0.0)

    candidates = policy._world_model_candidates(_engine, _is_done)

    assert isinstance(candidates, list)
    assert len(candidates) >= 1
    assert isinstance(candidates[0], WorldModelCandidate)


def test_world_model_candidates_does_not_raise_with_poe_world_explicitly_off(monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_POE_WORLD", "0")
    policy = E3AgentPolicy("zz99", proposer=None, value_head=lambda _f: 0.0)

    candidates = policy._world_model_candidates(_engine, _is_done)

    assert isinstance(candidates, list)
    assert len(candidates) >= 1
