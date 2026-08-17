"""REQ-ARC-WMTE-6470: repair-mode tool loop (CARNOT_ARC_INDUCE_TOOL_LOOP=repair).

CPU-only, same discipline as the sibling suites: the loop is driven against a
scripted fake HTTP layer, the agent helper against a fake proposer, and
`e3.E3_DIR` is monkeypatched to tmp_path so tracked evidence is never written.
The load-bearing pins are the two inertness tests: env unset consults nothing,
and the `repair` value must NOT engage replace mode at induce() time.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import carnot.agentic.arc_executable_world_model as e3
import carnot.agentic.arc_induction_tool_loop as loop_mod
import carnot.agentic.arc_recall_gated_resample as rgr
from carnot.agentic.arc_competition_agent import E3AgentPolicy

# ---------------------------------------------------------------------------
# shared fixtures (mirrors test_arc_induction_tool_loop.py's fake HTTP layer)
# ---------------------------------------------------------------------------


class _T:
    def __init__(self, grid, next_grid, action=1):
        self.grid = np.asarray(grid)
        self.next_grid = np.asarray(next_grid)
        self.action = action
        self.data = None
        self.level_before = 0
        self.level_after = 0


def _mover_window(n=6):
    """Action 1 moves the single 2 one column right on a 6x6 board."""
    rows = []
    for x in range(n):
        a = np.zeros((6, 6), dtype=int)
        a[2, x % 5] = 2
        b = np.zeros((6, 6), dtype=int)
        b[2, x % 5 + 1] = 2
        rows.append(_T(a, b))
    return rows


GOOD_CODE = """import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    rs, cs = np.where(g == 2)
    if action == 1 and len(rs):
        g[rs[0], cs[0]] = 0
        g[rs[0], min(g.shape[1] - 1, cs[0] + 1)] = 2
    return g

def is_level_complete(grid):
    return False
"""

# One deliberate flaw: the mover stalls at column 2, so exactly one visible
# transition mismatches -- a seed the loop can measurably improve on.
FLAWED_SEED_CODE = GOOD_CODE.replace(
    "if action == 1 and len(rs):",
    "if action == 1 and len(rs) and cs[0] != 2:",
)

IDENTITY_CODE = """import numpy as np

def engine(grid, action, data):
    return grid

def is_level_complete(grid):
    return False
"""


def _chat_reply(message, finish="stop", tokens=100):
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"completion_tokens": tokens},
    }


def _tool_call(name, arguments, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class _FakeHTTP:
    def __init__(self, chat_replies=(), completion_replies=()):
        self.chat_replies = list(chat_replies)
        self.completion_replies = list(completion_replies)
        self.chat_payloads = []
        self.completion_payloads = []

    def __call__(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        body = json.loads(req.data.decode())
        if "/v1/chat/completions" in url:
            self.chat_payloads.append(body)
            reply = self.chat_replies[min(len(self.chat_payloads) - 1, len(self.chat_replies) - 1)]
        else:
            self.completion_payloads.append(body)
            reply = self.completion_replies[
                min(len(self.completion_payloads) - 1, len(self.completion_replies) - 1)
            ]

        class _R:
            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def read(self):
                return json.dumps(reply).encode()

        return _R()


@pytest.fixture(autouse=True)
def _json_load(monkeypatch):
    real = json.load
    monkeypatch.setattr(json, "load", lambda r, *a, **k: json.loads(r.read()))
    yield
    monkeypatch.setattr(json, "load", real)


@pytest.fixture
def proposer(monkeypatch, tmp_path):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    p = e3.LocalGGUFProposer()
    monkeypatch.setattr(type(p), "_ensure_server", lambda _self: True)
    monkeypatch.setattr(type(p), "_url", lambda _self: "http://127.0.0.1:1")
    return p


def _patch_http(monkeypatch, fake):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake)


# ---------------------------------------------------------------------------
# inertness pins (SCENARIO-ARC-WMTE-6470-1 / -2)
# ---------------------------------------------------------------------------


def test_env_unset_repair_is_inert(monkeypatch):
    """SCENARIO-6470-1: unset -> repair inactive and decide_resample unchanged."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP", raising=False)
    monkeypatch.delenv("CARNOT_ARC_RECALL_RESAMPLE", raising=False)
    assert rgr.tool_repair_active() is False
    decision = rgr.decide_resample(
        cell_recall=0.0, n_changing=6, downstream_rejects=True, resamples_used_this_game=0
    )
    assert decision.fire is False and decision.reason == "disabled"


def test_repair_value_does_not_engage_replace_mode_at_induce(monkeypatch, proposer, tmp_path):
    """SCENARIO-6470-2: `repair` != `1`, so induce() must run the shipped single-shot
    path and never consult the loop. The bomb proves the negative."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "repair")

    def _bomb(*_a, **_k):
        raise AssertionError("tool loop consulted at induce() time in repair mode")

    monkeypatch.setattr(loop_mod, "induce_with_tool_loop", _bomb)
    fake = _FakeHTTP(
        completion_replies=[{"content": f"```python\n{GOOD_CODE}```", "stop_type": "eos"}]
    )
    _patch_http(monkeypatch, fake)
    ok, _note = proposer.induce("repairgame", _mover_window(), 1)
    assert ok
    assert fake.chat_payloads == []  # no tool-loop chat requests
    assert (tmp_path / "repairgame" / "world_model.py").exists()


# ---------------------------------------------------------------------------
# decision matrix (SCENARIO-ARC-WMTE-6470-3 / -4)
# ---------------------------------------------------------------------------

_CATASTROPHIC = dict(
    cell_recall=0.294,
    n_changing=6,
    downstream_rejects=True,
    resamples_used_this_game=0,
)


def test_tool_repair_fires_without_resample_env(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_RECALL_RESAMPLE", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    decision = rgr.decide_resample(**_CATASTROPHIC, tool_repair=True)
    assert decision.fire is True
    assert decision.reason == "catastrophic_recall_tool_loop_repair"


def test_tool_repair_bypasses_seed_pin_blind_mode_keeps_it(monkeypatch):
    """SCENARIO-6470-4: the seed-pin refusal is about a blind re-draw replaying the
    same tokens; the loop's conversation is a different stream, so repair fires."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_SEED", "42")
    blind = rgr.decide_resample(**_CATASTROPHIC)
    assert blind.fire is False
    assert blind.reason == "seed_pinned_resample_would_reproduce"
    repair = rgr.decide_resample(**_CATASTROPHIC, tool_repair=True)
    assert repair.fire is True


def test_tool_repair_stays_trust_subordinate(monkeypatch):
    """SCENARIO-6470-3, refusal half: an engine the trust gate would USE is never
    repaired -- same structural memorization-trap defense as REQ-6410."""
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    decision = rgr.decide_resample(
        **{**_CATASTROPHIC, "downstream_rejects": False}, tool_repair=True
    )
    assert decision.fire is False
    assert decision.reason == "downstream_accepted_engine"


def test_tool_repair_respects_evidence_floor_and_budgets(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    floor = rgr.decide_resample(**{**_CATASTROPHIC, "n_changing": 2}, tool_repair=True)
    assert floor.fire is False and floor.reason == "insufficient_changing_evidence"
    budget = rgr.decide_resample(
        **{**_CATASTROPHIC, "resamples_used_this_game": rgr.DEFAULT_MAX_PER_GAME},
        tool_repair=True,
    )
    assert budget.fire is False and budget.reason == "per_game_budget_exhausted"


# ---------------------------------------------------------------------------
# seeded loop behaviour (SCENARIO-ARC-WMTE-6470-3 / -5)
# ---------------------------------------------------------------------------


def test_seeded_loop_prompt_carries_seed_and_report_and_repair_wins(
    monkeypatch, proposer, tmp_path
):
    """The opening message carries the seed code + its measured report, the seed is
    candidate zero, and a genuinely better tool-round candidate replaces it."""
    fake = _FakeHTTP(
        chat_replies=[
            _chat_reply(
                {
                    "role": "assistant",
                    "tool_calls": [
                        _tool_call(
                            "run_engine_on_transitions", json.dumps({"code": GOOD_CODE}), "c1"
                        )
                    ],
                },
                tokens=10,
            ),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, note = loop_mod.induce_with_tool_loop(
        proposer, "seedgame", _mover_window(), 1, seed_engine_code=FLAWED_SEED_CODE
    )
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["seeded"] is True
    assert stats["seed_scoreable"] is True
    assert stats["seed_visible_mismatches"] >= 1
    assert stats["terminated_by"] == "zero_mismatches"
    first_user = fake.chat_payloads[0]["messages"][0]["content"]
    assert "REPAIR MODE" in first_user
    assert "cs[0] != 2" in first_user  # the seed's own code is shown
    assert '"mismatches"' in first_user  # with its measured report
    written = (tmp_path / "seedgame" / "world_model.py").read_text()
    assert "cs[0] != 2" not in written  # the repaired engine won


def test_seeded_loop_floor_keeps_seed_when_candidates_worse(monkeypatch, proposer, tmp_path):
    """SCENARIO-6470-5: only-worse tool rounds -> early stop, and the SEED is what
    lands on disk. The monotone accept makes regression impossible."""
    worse = _chat_reply(
        {
            "role": "assistant",
            "tool_calls": [
                _tool_call("run_engine_on_transitions", json.dumps({"code": IDENTITY_CODE}), "cw")
            ],
        },
        tokens=10,
    )
    fake = _FakeHTTP(chat_replies=[worse, worse, worse])
    _patch_http(monkeypatch, fake)
    ok, note = loop_mod.induce_with_tool_loop(
        proposer, "floorgame", _mover_window(), 1, seed_engine_code=FLAWED_SEED_CODE
    )
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["terminated_by"] == "early_stop_non_improving"
    written = (tmp_path / "floorgame" / "world_model.py").read_text()
    assert "cs[0] != 2" in written  # the seed engine, not the identity regression


def test_seed_zero_mismatch_short_circuits_without_a_single_request(
    monkeypatch, proposer, tmp_path
):
    """A seed that already fits the visible split leaves nothing for a mismatch-driven
    loop to iterate on: accept it immediately, zero LLM requests."""
    fake = _FakeHTTP()
    _patch_http(monkeypatch, fake)
    ok, note = loop_mod.induce_with_tool_loop(
        proposer, "zsgame", _mover_window(), 1, seed_engine_code=GOOD_CODE
    )
    assert ok
    assert proposer.last_tool_loop_stats["terminated_by"] == "seed_zero_mismatches"
    assert fake.chat_payloads == []
    assert (tmp_path / "zsgame" / "world_model.py").exists()


# ---------------------------------------------------------------------------
# agent-helper integration (fake proposer; loop faked at the module seam)
# ---------------------------------------------------------------------------

GAME = "rg01"

_OLD_ENGINE_CODE = IDENTITY_CODE

_GOOD_ENGINE_CODE = """import numpy as np

def engine(grid, action, data):
    out = np.asarray(grid).copy()
    out[:] = 5
    return out

def is_level_complete(grid):
    return False
"""


class _FakeProposer:
    def __init__(self):
        self.calls = 0
        self.last_tool_loop_stats = {}

    def induce(self, game, trans, cell, **kwargs):
        self.calls += 1
        return False, "blind induce must not be called in repair mode"


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


def _helper_call(policy, attempt, trans, engine, is_done, vr):
    return policy._maybe_recall_gated_resample(
        attempt=attempt,
        transitions=trans,
        hud_mask=None,
        engine=engine,
        is_done=is_done,
        vr=vr,
        induce_rows=trans,
        induce_kwargs={},
    )


def test_helper_repair_routes_through_seeded_loop_and_recovers(tmp_path, monkeypatch):
    """SCENARIO-6470-3: catastrophic rejected draw + repair mode -> the helper calls the
    LOOP (not blind induce), seeds it with the failed engine, and keeps the recovery."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "repair")
    monkeypatch.delenv("CARNOT_ARC_RECALL_RESAMPLE", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    # Seed pinned ON PURPOSE: repair must fire anyway (SCENARIO-6470-4, wiring half).
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_SEED", "42")
    store = _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = e3.WorldModelVerifier(trans).score(old_engine)
    assert vr.cell_recall == 0.0

    seen: dict = {}

    def _fake_loop(prop, game, rows, cell, **kwargs):
        seen["seed_engine_code"] = kwargs.get("seed_engine_code")
        seen["game"] = game
        prop.last_tool_loop_stats = {
            "turns": 2,
            "terminated_by": "zero_mismatches",
            "candidates_scored": 2,
            "best_visible_mismatches": 0,
            "seed_visible_mismatches": 4,
            "decode_tokens_total": 500,
            "tool_call_parse_failures": 0,
        }
        path = e3.E3_DIR / game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_GOOD_ENGINE_CODE)
        return True, "tool loop: zero_mismatches"

    monkeypatch.setattr(loop_mod, "induce_with_tool_loop", _fake_loop)
    proposer = _FakeProposer()
    policy = _policy(proposer)
    attempt: dict = {}
    engine, is_done, kept_vr = _helper_call(policy, attempt, trans, old_engine, old_is_done, vr)
    assert proposer.calls == 0  # blind induce never ran
    assert seen["seed_engine_code"] == _OLD_ENGINE_CODE
    record = attempt["recall_resample"]
    assert record["reason"] == "catastrophic_recall_tool_loop_repair"
    assert record["via_tool_loop"] is True
    assert record["outcome"] == "kept_resample"
    assert record["tool_loop"]["terminated_by"] == "zero_mismatches"
    assert kept_vr.cell_recall == 1.0
    assert store.read_text() == _GOOD_ENGINE_CODE


def test_helper_repair_restores_store_when_loop_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "repair")
    monkeypatch.delenv("CARNOT_ARC_RECALL_RESAMPLE", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    store = _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = e3.WorldModelVerifier(trans).score(old_engine)

    def _failing_loop(prop, game, rows, cell, **kwargs):
        prop.last_tool_loop_stats = {"terminated_by": "transport_error"}
        return False, "tool loop: no scoreable engine (transport_error)"

    monkeypatch.setattr(loop_mod, "induce_with_tool_loop", _failing_loop)
    policy = _policy(_FakeProposer())
    attempt: dict = {}
    engine, is_done, kept_vr = _helper_call(policy, attempt, trans, old_engine, old_is_done, vr)
    assert attempt["recall_resample"]["outcome"] == "resample_induce_failed"
    assert engine is old_engine and kept_vr is vr
    assert store.read_text() == _OLD_ENGINE_CODE


def test_helper_blind_mode_still_uses_blind_induce(tmp_path, monkeypatch):
    """Regression pin on the pre-existing lever: CARNOT_ARC_RECALL_RESAMPLE=1 without
    repair mode must still call proposer.induce, not the loop."""
    monkeypatch.setenv("CARNOT_ARC_RECALL_RESAMPLE", "1")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    _seed_store(tmp_path, monkeypatch, _OLD_ENGINE_CODE)
    trans = _transitions()
    old_engine, old_is_done = e3.load_engine(GAME)
    vr = e3.WorldModelVerifier(trans).score(old_engine)

    def _bomb(*_a, **_k):
        raise AssertionError("loop consulted in blind-resample mode")

    monkeypatch.setattr(loop_mod, "induce_with_tool_loop", _bomb)
    proposer = _FakeProposer()
    policy = _policy(proposer)
    attempt: dict = {}
    _helper_call(policy, attempt, trans, old_engine, old_is_done, vr)
    assert proposer.calls == 1
    assert attempt["recall_resample"]["via_tool_loop"] is False
    assert attempt["recall_resample"]["reason"] == "catastrophic_recall_resample"
