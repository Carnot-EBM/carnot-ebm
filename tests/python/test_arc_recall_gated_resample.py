"""REQ-ARC-WMTE-6410: recall-gated induction resample.

CPU-only. The proposer is a fake that writes engine files; no LLM, no GPU, no
network. `e3.E3_DIR` is monkeypatched to tmp_path in every store-touching test,
per that module's own documented monkeypatch contract, so the tracked evidence
store (`results/arc_e3`) is never written (SCENARIO-ARC-WMTE-6410-*).
"""

from __future__ import annotations

import numpy as np
import pytest

import carnot.agentic.arc_executable_world_model as e3
import carnot.agentic.arc_recall_gated_resample as rgr
from carnot.agentic.arc_competition_agent import E3AgentPolicy

# ---------------------------------------------------------------------------
# decision-function unit tests (pure; no files, no engines)
# ---------------------------------------------------------------------------

_CATASTROPHIC = dict(
    cell_recall=0.294,
    n_changing=6,
    downstream_rejects=True,
    resamples_used_this_game=0,
)


def test_disabled_by_default_never_fires(monkeypatch):
    """SCENARIO-ARC-WMTE-6410-1: with the env flag unset the scored path is unchanged --
    even the measured catastrophic cell (recall 0.294) draws no second generation."""
    monkeypatch.delenv("CARNOT_ARC_RECALL_RESAMPLE", raising=False)
    decision = rgr.decide_resample(**_CATASTROPHIC)
    assert decision.fire is False
    assert decision.reason == "disabled"


def test_never_fires_on_engine_the_trust_gate_accepts(monkeypatch):
    """SCENARIO-ARC-WMTE-6410-2: the structural memorization-trap defense. An engine the
    downstream gate would USE is never re-rolled, whatever its recall, so the gate
    applies no selection pressure among usable engines."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    decision = rgr.decide_resample(**{**_CATASTROPHIC, "downstream_rejects": False})
    assert decision.fire is False
    assert decision.reason == "downstream_accepted_engine"


def test_fires_on_catastrophic_rejected_draw(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    decision = rgr.decide_resample(**_CATASTROPHIC)
    assert decision.fire is True
    assert decision.reason == "catastrophic_recall_resample"


def test_seed_pinned_refuses(monkeypatch):
    """SCENARIO-ARC-WMTE-6410-3: with CARNOT_ARC_GENERATOR_SEED set, a re-call replays
    the identical sampler seed on the identical prompt, so a resample would buy a
    byte-identical answer for a full generation's cost. Refuse and say why."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_SEED", "42")
    decision = rgr.decide_resample(**_CATASTROPHIC)
    assert decision.fire is False
    assert decision.reason == "seed_pinned_resample_would_reproduce"


def test_malformed_seed_env_is_not_pinned(monkeypatch):
    """Mirrors sampling_seed's parse: a malformed value falls back to unseeded, where a
    fresh server-side random seed makes the resample a genuine second draw."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_SEED", "not-an-int")
    assert rgr.generator_seed_pinned() is False
    assert rgr.decide_resample(**_CATASTROPHIC).fire is True


def test_evidence_floor_blocks_quiet_windows(monkeypatch):
    """SCENARIO-ARC-WMTE-6410-4: cell_recall is 0.0 BY DEFINITION on a window with no
    changing transitions, and noise on 1-2. Those must not burn a generation: the
    window, not the draw, is the problem there."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    for n_changing in (0, 1, 2):
        decision = rgr.decide_resample(**{**_CATASTROPHIC, "n_changing": n_changing})
        assert decision.fire is False, n_changing
        assert decision.reason == "insufficient_changing_evidence"


def test_threshold_boundary(monkeypatch):
    """At-threshold recall does not fire; just below does. The default (0.6) sits far
    below ceiling on purpose: near-ceiling firing is the memorization-selection zone."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    at = rgr.decide_resample(**{**_CATASTROPHIC, "cell_recall": rgr.DEFAULT_THRESHOLD})
    assert at.fire is False
    assert at.reason == "recall_not_catastrophic"
    below = rgr.decide_resample(**{**_CATASTROPHIC, "cell_recall": rgr.DEFAULT_THRESHOLD - 1e-3})
    assert below.fire is True


def test_budgets_bound_the_cost(monkeypatch):
    """SCENARIO-ARC-WMTE-6410-5: one retry per call, a small per-game cap. Induction is
    the dominant live-eval cost, so both bounds are load-bearing, not hygiene."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    per_call = rgr.decide_resample(**{**_CATASTROPHIC, "retries_used_this_call": 1})
    assert per_call.fire is False
    assert per_call.reason == "per_call_retry_exhausted"
    per_game = rgr.decide_resample(
        **{**_CATASTROPHIC, "resamples_used_this_game": rgr.DEFAULT_MAX_PER_GAME}
    )
    assert per_game.fire is False
    assert per_game.reason == "per_game_budget_exhausted"


def test_malformed_tuning_envs_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE_THRESHOLD", "bogus")
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE_MIN_CHANGING", "")
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE_MAX_PER_GAME", "nan")
    assert rgr.resample_threshold() == rgr.DEFAULT_THRESHOLD
    assert rgr.resample_min_changing() == rgr.DEFAULT_MIN_CHANGING
    assert rgr.resample_max_per_game() == rgr.DEFAULT_MAX_PER_GAME


def test_keep_rule_is_satisficing_not_maximizing():
    """SCENARIO-ARC-WMTE-6410-6: a trust-passing re-draw is kept; between two REJECTS the
    better recall wins; a tie keeps the original. The rule never compares two usable
    engines by in-sample recall, which is where memorization selection would live."""
    assert rgr.keep_resample(new_passes_downstream=True, new_cell_recall=0.7, old_cell_recall=0.0)
    assert rgr.keep_resample(new_passes_downstream=False, new_cell_recall=0.4, old_cell_recall=0.29)
    assert not rgr.keep_resample(
        new_passes_downstream=False, new_cell_recall=0.29, old_cell_recall=0.29
    )
    assert not rgr.keep_resample(
        new_passes_downstream=False, new_cell_recall=0.1, old_cell_recall=0.29
    )


# ---------------------------------------------------------------------------
# agent-helper integration tests (fake proposer, E3_DIR -> tmp_path)
# ---------------------------------------------------------------------------

GAME = "gg01"

# Every transition changes every cell to 5. The identity engine (returns its input)
# scores recall 0.0 on this window; the good engine scores 1.0.
_OLD_ENGINE_CODE = """import numpy as np

def engine(grid, action, data):
    return np.asarray(grid)

def is_level_complete(grid):
    return False
"""

_GOOD_ENGINE_CODE = """import numpy as np

def engine(grid, action, data):
    out = np.asarray(grid).copy()
    out[:] = 5
    return out

def is_level_complete(grid):
    return False
"""

# Wrong in a different way (writes 7 where reality writes 5): still recall 0.0,
# and NOT strictly better than the original -- the keep rule must keep the original.
_WORSE_ENGINE_CODE = """import numpy as np

def engine(grid, action, data):
    out = np.asarray(grid).copy()
    out[:] = 7
    return out

def is_level_complete(grid):
    return False
"""


class _FakeProposer:
    """Writes a canned engine file where the real proposer would; counts calls so a
    test can assert the no-fire path makes zero LLM-shaped calls."""

    def __init__(self, code: str | None, ok: bool = True):
        self.code = code
        self.ok = ok
        self.calls = 0

    def induce(self, game, trans, cell, **kwargs):
        self.calls += 1
        if not self.ok:
            return False, "fake proposer refused"
        path = e3.E3_DIR / game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.code)
        return True, "fake proposer wrote world_model.py"


def _transitions(n: int = 4) -> list:
    rows = []
    for i in range(n):
        g0 = np.zeros((4, 4), dtype=int)
        g1 = np.full((4, 4), 5, dtype=int)
        rows.append(
            e3.Transition(
                grid=g0, action=1 + (i % 3), data=None, next_grid=g1, level_before=0, level_after=0
            )
        )
    return rows


def _policy(proposer) -> E3AgentPolicy:
    # __new__ + the four attributes the helper touches: constructing the full policy
    # needs a live session, which a CPU-only test must not require.
    policy = object.__new__(E3AgentPolicy)
    policy.short = GAME
    policy.cell = 1
    policy.proposer = proposer
    return policy


def _seed_store(tmp_path, monkeypatch, code: str):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    path = tmp_path / GAME / "world_model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return path


def _score(engine, trans):
    return e3.WorldModelVerifier(trans).score(engine)


def test_helper_recovers_catastrophic_draw(tmp_path, monkeypatch):
    """SCENARIO-ARC-WMTE-6410-7: identity draw (recall 0.0, rejected) + a good second
    draw -> the helper returns the second engine and its passing verdict, and the
    attempt row records both rounds."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    store = _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = _score(old_engine, trans)
    assert vr.cell_recall == 0.0 and vr.n_changing >= 3  # the fixture is catastrophic
    proposer = _FakeProposer(_GOOD_ENGINE_CODE)
    policy = _policy(proposer)
    attempt: dict = {}
    engine, is_done, kept_vr = policy._maybe_recall_gated_resample(
        attempt=attempt,
        transitions=trans,
        hud_mask=None,
        engine=old_engine,
        is_done=old_is_done,
        vr=vr,
        induce_rows=trans,
        induce_kwargs={},
    )
    assert proposer.calls == 1
    assert kept_vr.cell_recall == 1.0
    assert attempt["recall_resample"]["outcome"] == "kept_resample"
    assert attempt["recall_resample"]["original_cell_recall"] == 0.0
    assert attempt["recall_resample"]["resample_cell_recall"] == 1.0
    assert store.read_text() == _GOOD_ENGINE_CODE  # the store holds the kept engine
    # The returned engine really is the new one: it predicts the observed change.
    assert np.array_equal(engine(trans[0].grid.copy(), 1, None), trans[0].next_grid)


def test_helper_restores_store_when_second_draw_is_no_better(tmp_path, monkeypatch):
    """SCENARIO-ARC-WMTE-6410-8: a not-better second draw is discarded AND the store
    file is restored byte-identically. Keeping the old engine OBJECT while the FILE
    holds the re-draw would poison every later load (the ka59 overwrite lesson)."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    store = _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = _score(old_engine, trans)
    proposer = _FakeProposer(_WORSE_ENGINE_CODE)
    policy = _policy(proposer)
    attempt: dict = {}
    engine, is_done, kept_vr = policy._maybe_recall_gated_resample(
        attempt=attempt,
        transitions=trans,
        hud_mask=None,
        engine=old_engine,
        is_done=old_is_done,
        vr=vr,
        induce_rows=trans,
        induce_kwargs={},
    )
    assert attempt["recall_resample"]["outcome"] == "kept_original"
    assert attempt["recall_resample"]["engine_store_restored"] is True
    assert store.read_text() == _OLD_ENGINE_CODE  # byte-identical restore
    assert kept_vr is vr and engine is old_engine


def test_helper_flag_off_is_inert(tmp_path, monkeypatch):
    """SCENARIO-ARC-WMTE-6410-9: flag off -> zero proposer calls, inputs returned
    untouched, no record written. The scored path stays byte-identical."""
    monkeypatch.delenv("CARNOT_ARC_RECALL_RESAMPLE", raising=False)
    _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = _score(old_engine, trans)
    proposer = _FakeProposer(_GOOD_ENGINE_CODE)
    policy = _policy(proposer)
    attempt: dict = {}
    engine, is_done, kept_vr = policy._maybe_recall_gated_resample(
        attempt=attempt,
        transitions=trans,
        hud_mask=None,
        engine=old_engine,
        is_done=old_is_done,
        vr=vr,
        induce_rows=trans,
        induce_kwargs={},
    )
    assert proposer.calls == 0
    assert engine is old_engine and kept_vr is vr
    assert "recall_resample" not in attempt


def test_helper_respects_per_game_budget(tmp_path, monkeypatch):
    """The lazy per-policy counter bounds a game's total resamples even across many
    induce attempts in one episode."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = _score(old_engine, trans)
    proposer = _FakeProposer(_WORSE_ENGINE_CODE)
    policy = _policy(proposer)
    for expected_calls in (1, 2, 2):  # third attempt refused: budget is 2
        attempt: dict = {}
        policy._maybe_recall_gated_resample(
            attempt=attempt,
            transitions=trans,
            hud_mask=None,
            engine=old_engine,
            is_done=old_is_done,
            vr=vr,
            induce_rows=trans,
            induce_kwargs={},
        )
        assert proposer.calls == expected_calls
    assert attempt["recall_resample"]["reason"] == "per_game_budget_exhausted"


def test_helper_keeps_original_when_resample_induce_fails(tmp_path, monkeypatch):
    """A refused/failed second generation must degrade to today's behavior exactly:
    original engine, original verdict, store untouched."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    store = _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = _score(old_engine, trans)
    proposer = _FakeProposer(None, ok=False)
    policy = _policy(proposer)
    attempt: dict = {}
    engine, is_done, kept_vr = policy._maybe_recall_gated_resample(
        attempt=attempt,
        transitions=trans,
        hud_mask=None,
        engine=old_engine,
        is_done=old_is_done,
        vr=vr,
        induce_rows=trans,
        induce_kwargs={},
    )
    assert attempt["recall_resample"]["outcome"] == "resample_induce_failed"
    assert engine is old_engine and kept_vr is vr
    assert store.read_text() == _OLD_ENGINE_CODE


def test_wiring_order_on_plain_branch():
    """SCENARIO-ARC-WMTE-6410-7 wiring half. `_induce_and_plan` cannot be driven end to
    end in a CPU test (its own sibling tests inspect source for the same reason), so
    assert the call ORDER instead: the resample runs after the plain-branch score and
    before the trust-metric read, so the gate logic reads the KEPT engine's verdict."""
    import inspect

    src = inspect.getsource(E3AgentPolicy._induce_and_plan)
    score = src.index("WorldModelVerifier(active_transitions, hud_mask=_hud_mask).score(engine)")
    resample = src.index("_maybe_recall_gated_resample(")
    metric = src.index('os.environ.get("CARNOT_ARC_TRUST_METRIC", "exact")')
    assert score < resample < metric
    # The rebinding really is a rebinding: the kept triple replaces the gate's inputs.
    assert "engine, is_done, vr = self._maybe_recall_gated_resample(" in src


def test_record_survives_diagnostics_projection():
    """The attempt-row projection has twice before silently dropped a gate's evidence
    (REQ-ARC-WMTE-6017/-6019). Assert `recall_resample` is in the projected key tuple
    so this gate's record reaches the cell artifact from day one."""
    import inspect

    import carnot.agentic.arc_competition_agent as agent_mod

    src = inspect.getsource(agent_mod)
    projection_start = src.index("induction_attempt_gate_diagnostics")
    projection_end = src.index("for a in self.induction_attempts", projection_start)
    assert '"recall_resample",' in src[projection_start:projection_end]


def test_trust_rejects_mirror_matches_plain_gate():
    """SCENARIO-ARC-WMTE-6410-10: the helper's downstream-verdict mirror agrees with the
    inline plain-branch logic on both sides of the 0.5 default-metric threshold, so the
    gate cannot fire on an engine the real gate would accept."""
    trans = _transitions()
    identity = lambda grid, action, data: np.asarray(grid)  # noqa: E731
    good = lambda grid, action, data: np.full_like(np.asarray(grid), 5)  # noqa: E731
    assert E3AgentPolicy._plain_trust_rejects(_score(identity, trans)) is True
    assert E3AgentPolicy._plain_trust_rejects(_score(good, trans)) is False
