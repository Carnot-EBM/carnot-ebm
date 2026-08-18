"""The refinement-round cap (3 -> 1, operator-approved 2026-08-17).

WHY THE CAP EXISTS. Refinement rounds past the first are measurably HARMFUL, not merely useless.
exp5760 measured mean_delta_heldout 0.0; exp5766 measured pooled_mean_delta_heldout -0.0598; 0 of
83 cells improved. They fired on 39 of 39 stalls at roughly 319 s each -- about 58% of all stall
wall clock -- and all 8 partial round-0 engines collapsed to exactly 0.0.

The cause is REQ-ARC-WMTE-6091: the refine prompt never contains the engine being repaired, so a
round is a blind re-induction from at most 5 mismatches, i.e. LESS evidence than round 0 had.

These tests pin the cap and its escape hatch. The escape hatch matters: fixing the underlying
defect (CARNOT_ARC_REFACTOR_SHOW_ENGINE) is a live separate lever, and whoever tests it needs to
restore multi-round behaviour without editing source.
"""

from __future__ import annotations

import importlib

import pytest


def _reload(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("CARNOT_ARC_MAX_REFINEMENT_ROUNDS", raising=False)
    else:
        monkeypatch.setenv("CARNOT_ARC_MAX_REFINEMENT_ROUNDS", value)
    import carnot.agentic.arc_llm_reinduction as mod

    return importlib.reload(mod)


def test_default_is_one_round(monkeypatch) -> None:
    """The shipped default. A regression here silently restores a pooled-negative code path."""
    assert _reload(monkeypatch, None).MAX_REFINEMENT_ROUNDS == 1


def test_env_can_restore_multi_round(monkeypatch) -> None:
    """The escape hatch, so the show-engine fix can be tested without a source edit."""
    assert _reload(monkeypatch, "3").MAX_REFINEMENT_ROUNDS == 3


@pytest.mark.parametrize("bad", ["", "not-a-number", "-2", "0"])
def test_malformed_or_disabling_values_floor_at_one(monkeypatch, bad: str) -> None:
    """A bad value must not disable refinement altogether -- that is a DIFFERENT change than this
    cap, and round 0 is the round that actually produces the engine. Floor at 1, never 0."""
    assert _reload(monkeypatch, bad).MAX_REFINEMENT_ROUNDS == 1


def test_the_executor_respects_the_cap(monkeypatch) -> None:
    """The constant is only meaningful if the loop bound reads it. `rounds_limit` is
    min(max_rounds, MAX_REFINEMENT_ROUNDS), so the module constant is a hard ceiling even when a
    caller passes a larger max_rounds."""
    mod = _reload(monkeypatch, None)
    import inspect

    src = inspect.getsource(mod)
    assert "min(int(max_rounds), MAX_REFINEMENT_ROUNDS)" in src, (
        "the executor no longer clamps to MAX_REFINEMENT_ROUNDS; the cap would be advisory only"
    )
