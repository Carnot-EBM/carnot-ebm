"""Compacted carried state for the tool-calling induction loop (REQ-ARC-WMTE-6540).

Same discipline as test_arc_induction_tool_loop.py: drive the REAL loop against a
scripted fake HTTP layer (no GPU, no model) and assert the observable consequences.
The load-bearing pin is the first test: with `CARNOT_ARC_INDUCE_TOOL_COMPACT` unset,
the message stream must be byte-identical to today's append-only transcript and the
rebuild machinery must never be consulted. No test touches tracked state: the
world-model writer points at tmp_path (Test-Run Record Integrity).
"""

from __future__ import annotations

import json
import logging
import urllib.error

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_induction_compact_state as compact_mod
from carnot.agentic import arc_induction_tool_loop as loop_mod
from carnot.agentic import arc_induction_tools as tools_mod
from carnot.agentic.arc_induction_tools import CandidateRecord


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

IDENTITY_CODE = """import numpy as np

def engine(grid, action, data):
    return grid

def is_level_complete(grid):
    return False
"""


def _chat_reply(message, tokens=10, prompt_tokens=None):
    usage = {"completion_tokens": tokens}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage}


def _tool_call(name, arguments, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _diff_reply(t, call_id, prompt_tokens=None):
    return _chat_reply(
        {
            "role": "assistant",
            "tool_calls": [_tool_call("diff_grids", json.dumps({"t": t}), call_id)],
        },
        prompt_tokens=prompt_tokens,
    )


def _engine_reply(code, call_id, prompt_tokens=None):
    return _chat_reply(
        {
            "role": "assistant",
            "tool_calls": [
                _tool_call("run_engine_on_transitions", json.dumps({"code": code}), call_id)
            ],
        },
        prompt_tokens=prompt_tokens,
    )


class _FakeHTTP:
    """Scripted server for BOTH endpoints, same shape as the sibling test file."""

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


@pytest.fixture(autouse=True)
def _compact_env_clean(monkeypatch):
    # Every test states its own compaction env explicitly; leaked env must not steer.
    for var in (
        "CARNOT_ARC_INDUCE_TOOL_COMPACT",
        "CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH",
        "CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def proposer(monkeypatch, tmp_path):
    # E3_DIR -> tmp_path: a test must never write results/** (Test-Run Record Integrity).
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    p = e3.LocalGGUFProposer()
    monkeypatch.setattr(type(p), "_ensure_server", lambda _self: True)
    monkeypatch.setattr(type(p), "_url", lambda _self: "http://127.0.0.1:1")
    return p


def _patch_http(monkeypatch, fake):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def _carried_messages(payload):
    """The carried-state user messages inside one request payload."""
    return [
        m
        for m in payload["messages"]
        if m.get("role") == "user" and compact_mod.CARRIED_STATE_KIND in str(m.get("content"))
    ]


def _assert_tool_ids_resolve(payload):
    """SCENARIO-ARC-WMTE-6540-3: every tool message's tool_call_id resolves to a
    PRECEDING assistant turn -- no orphan survives a rebuild."""
    seen_ids: set[str] = set()
    for m in payload["messages"]:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                seen_ids.add(tc["id"])
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in seen_ids, f"orphan tool message {m['tool_call_id']}"


# ---------------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6540-1: default off, byte-identical
# ---------------------------------------------------------------------------------


def test_env_unset_message_stream_is_byte_identical_and_rebuild_never_consulted(
    monkeypatch, proposer
):
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-1. Two proofs at once. The bomb
    proves the rebuild machinery is never consulted with the env unset, even when
    the measured prompt size crosses any threshold. The exact-equality check pins
    the full message list to the hand-built append-only expectation -- byte-identical
    to today, not merely 'no compaction marker seen'."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_LEAN_PROMPT", raising=False)

    def _bomb(*_a, **_k):
        raise AssertionError("compaction machinery consulted with env unset")

    monkeypatch.setattr(loop_mod, "rebuild_messages", _bomb)
    monkeypatch.setattr(loop_mod, "build_carried_state", _bomb)
    args0 = json.dumps({"t": 0})
    args1 = json.dumps({"t": 1})
    fake = _FakeHTTP(
        chat_replies=[
            _diff_reply(0, "i0", prompt_tokens=1000),
            _diff_reply(1, "i1", prompt_tokens=50000),  # would cross the default growth
            _engine_reply(GOOD_CODE, "i2", prompt_tokens=51000),
        ]
    )
    _patch_http(monkeypatch, fake)
    window = _mover_window(6)
    ok, _ = proposer.induce("bytegame", window, 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["compactions"] == 0
    assert stats["compact_floor_hit"] is False
    # Telemetry records unconditionally -- it never alters a payload.
    assert stats["prompt_tokens_per_turn"] == [1000, 50000, 51000]
    # Hand-built append-only expectation, byte-identical to the shipped transcript.
    visible, _held = tools_mod.holdout_split(window)
    base = e3.induce_prompt(
        "bytegame",
        list(visible),
        1,
        k=e3._induce_transitions_k(),
        include_playbook_exemplars=getattr(proposer, "include_playbook_exemplars", False),
    )
    session2 = tools_mod.InductionToolSession(_mover_window(6), cell=1)
    diff0 = tools_mod.dispatch_tool(session2, "diff_grids", args0)
    diff1 = tools_mod.dispatch_tool(session2, "diff_grids", args1)
    expected = [
        {"role": "user", "content": base + "\n\n" + loop_mod._TOOL_INSTRUCTIONS},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("diff_grids", args0, "i0")]},
        {"role": "tool", "tool_call_id": "i0", "content": json.dumps(diff0)},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("diff_grids", args1, "i1")]},
        {"role": "tool", "tool_call_id": "i1", "content": json.dumps(diff1)},
    ]
    assert fake.chat_payloads[0]["messages"] == expected[:1]
    assert fake.chat_payloads[1]["messages"] == expected[:3]
    assert fake.chat_payloads[2]["messages"] == expected


def test_duplicate_candidate_counter_moves_with_compaction_off(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 rule 5. `duplicate_candidate_submissions` is arm-neutral
    telemetry: it counts a resubmitted code_sha8 even with compaction off, so the
    A/B has an OFF-arm baseline."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _FakeHTTP(
        chat_replies=[
            _engine_reply(IDENTITY_CODE, "c1"),
            _engine_reply(IDENTITY_CODE, "c2"),  # same sha8: a duplicate submission
            _engine_reply(GOOD_CODE, "c3"),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("dupgame", _mover_window(6), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["duplicate_candidate_submissions"] == 1
    assert stats["compactions"] == 0
    assert stats["terminated_by"] == "zero_mismatches"


# ---------------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6540-2 + -3: threshold trigger, stale-measurement no-refire,
# rebuild shape, conversation validity, refetch counter
# ---------------------------------------------------------------------------------


def test_trigger_threshold_rebuild_shape_and_no_stale_refire(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-2 and -3. The trigger fires exactly
    when the measured prompt crosses turn-0 + growth; the rebuild is [base verbatim,
    carried state, last complete round]; after the rebuild a MISSING measurement must
    not re-fire on the stale pre-rebuild value; a post-compaction re-fetch of an
    already-fetched key is counted."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "10")
    fake = _FakeHTTP(
        chat_replies=[
            _diff_reply(0, "i0", prompt_tokens=1000),  # baseline
            _diff_reply(1, "i1", prompt_tokens=1200),  # crosses 1000+100 -> fire next
            _diff_reply(0, "i2"),  # NO measurement; also re-fetches t=0
            _engine_reply(GOOD_CODE, "i3", prompt_tokens=1300),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("triggame", _mover_window(6), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["compactions"] == 1
    assert stats["prompt_tokens_per_turn"] == [1000, 1200, None, 1300]
    # Post-compaction repeat of the pre-compaction diff_grids(t=0) key.
    assert stats["refetch_tool_calls_post_compaction"] == 1
    # Request 2 is the first after the fire: [base, carried state, tail round].
    rebuilt = fake.chat_payloads[2]["messages"]
    assert rebuilt[0] == fake.chat_payloads[0]["messages"][0]  # base verbatim
    assert rebuilt[1]["role"] == "user"
    state = json.loads(rebuilt[1]["content"])
    assert state["kind"] == compact_mod.CARRIED_STATE_KIND
    assert state["v"] == 1
    assert state["note"] == compact_mod.CARRIED_STATE_NOTE
    assert {d["t"] for d in state["evidence"]["diffs_fetched"]} == {0, 1}
    # The tail is the LAST complete round (i1), not the older i0 round.
    assert rebuilt[2]["role"] == "assistant"
    assert rebuilt[2]["tool_calls"][0]["id"] == "i1"
    assert rebuilt[3] == {"role": "tool", "tool_call_id": "i1", "content": rebuilt[3]["content"]}
    assert len(rebuilt) == 4
    assert not any(
        tc["id"] == "i0" for m in rebuilt if m.get("role") == "assistant" for tc in m["tool_calls"]
    )
    # Turn 3 must NOT re-fire on the stale 1200: still exactly one carried state,
    # and the transcript after the event is append-only again.
    final = fake.chat_payloads[3]["messages"]
    assert len(_carried_messages(fake.chat_payloads[3])) == 1
    assert final[:4] == rebuilt
    assert final[4]["role"] == "assistant" and final[4]["tool_calls"][0]["id"] == "i2"
    for payload in fake.chat_payloads:
        _assert_tool_ids_resolve(payload)
        # Preserved protections: per-turn think budget on every request, and
        # reasoning_content never fed back.
        assert payload["thinking_budget_tokens"] == loop_mod.DEFAULT_THINK_BUDGET
        assert not any("reasoning_content" in m for m in payload["messages"])


def test_rebuild_messages_unit_tail_and_orphan_rules():
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-3, unit level. The tail starts at
    the last assistant turn WITH TOOL_CALLS and carries everything after it: the
    round's tool results, a trailing prose assistant turn, and the user nudge.
    Older rounds are dropped whole; with no tool round yet, no rebuild."""
    base = {"role": "user", "content": "base"}
    a1 = {"role": "assistant", "content": "", "tool_calls": [_tool_call("diff_grids", "{}", "a1")]}
    t1 = {"role": "tool", "tool_call_id": "a1", "content": "{}"}
    a2 = {"role": "assistant", "content": "", "tool_calls": [_tool_call("diff_grids", "{}", "a2")]}
    t2 = {"role": "tool", "tool_call_id": "a2", "content": "{}"}
    nudge = {"role": "user", "content": "nudge"}
    state = {"kind": compact_mod.CARRIED_STATE_KIND}
    carried = {"role": "user", "content": json.dumps(state)}
    rebuilt = compact_mod.rebuild_messages([base, a1, t1, a2, t2, nudge], state)
    assert rebuilt == [base, carried, a2, t2, nudge]
    # A prose turn after the last tool round: the tail STILL starts at the tool
    # round (its mismatch report stays verbatim) and the prose turn rides along.
    prose = {"role": "assistant", "content": "thinking about the rule"}
    rebuilt = compact_mod.rebuild_messages([base, a1, t1, a2, t2, prose, nudge], state)
    assert rebuilt == [base, carried, a2, t2, prose, nudge]
    # No tool round yet (prose-only history, or nothing): no rebuild.
    assert compact_mod.rebuild_messages([base, prose, nudge], state) is None
    assert compact_mod.rebuild_messages([base], state) is None


def test_rebuild_serialization_matches_the_sizing_estimate():
    """REQ-ARC-WMTE-6540 (2026-08-19 review, finding 9). `rebuild_messages` and
    `_estimate_tokens` must serialize identically (default=str), so a leaf that
    passed sizing can never raise at injection and break the loop's (bool, str)
    contract. np.int64 is the realistic non-JSON-native leaf."""
    base = {"role": "user", "content": "base"}
    a1 = {"role": "assistant", "content": "", "tool_calls": [_tool_call("diff_grids", "{}", "a1")]}
    t1 = {"role": "tool", "tool_call_id": "a1", "content": "{}"}
    state = {"kind": compact_mod.CARRIED_STATE_KIND, "x": np.int64(5)}
    assert compact_mod._estimate_tokens(state) > 0  # sizing accepts it
    rebuilt = compact_mod.rebuild_messages([base, a1, t1], state)
    assert rebuilt is not None  # injection must accept it too, not raise
    assert '"x": "5"' in rebuilt[1]["content"]


# ---------------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6540-4: round-trip fidelity (mechanical build, no LLM)
# ---------------------------------------------------------------------------------


def _fresh_stats():
    return {"refetch_tool_calls_post_compaction": 0, "duplicate_candidate_submissions": 0}


def _observe_engine(session, ledger, stats, code):
    result = session.run_engine_on_transitions(code)
    ledger.observe("run_engine_on_transitions", json.dumps({"code": code}), result, stats)
    return result


def test_round_trip_fidelity_best_code_verbatim_and_evidence_digests():
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-4. The carried state holds
    best.code VERBATIM, one ledger row per candidate with sha8 + one-line
    refutation, and bounded digests that distinguish 'never looked' from
    'looked, found X'."""
    session = tools_mod.InductionToolSession(_mover_window(14), cell=1)  # 11 visible, 3 held
    ledger = compact_mod.EvidenceLedger()
    stats = _fresh_stats()
    _observe_engine(session, ledger, stats, IDENTITY_CODE)  # mismatches everywhere
    _observe_engine(session, ledger, stats, GOOD_CODE)  # zero mismatches -> best
    ledger.observe("list_transitions", "{}", session.list_transitions(), stats)
    ledger.observe("diff_grids", '{"t": 0}', session.diff_grids(0), stats)
    ledger.observe(
        "query_region", '{"t": 0, ...}', session.query_region(0, 2, 3, 0, 3, "before"), stats
    )
    goal = "def is_level_complete(grid):\n    return False\n"
    ledger.observe(
        "run_goal_on_states", json.dumps({"code": goal}), session.run_goal_on_states(goal), stats
    )
    state, floor = compact_mod.build_carried_state(session, ledger, turn=7, budget_tokens=100000)
    assert floor is False
    assert state["best"]["code"] == GOOD_CODE  # verbatim, untruncated
    assert state["best"]["visible_mismatches"] == 0
    rows = state["candidates"]
    assert [r["idx"] for r in rows] == [0, 1]
    assert rows[0]["code_sha8"] == compact_mod.code_sha8(IDENTITY_CODE)
    assert rows[0]["first_mismatch"]  # the one-line refutation exists
    assert rows[1]["first_mismatch"] is None  # nothing to refute
    assert rows[0]["code_head"].startswith("def engine")
    assert state["session"] == {"n_visible": 11, "n_held_out": 3, "memorization_scan": False}
    ev = state["evidence"]
    assert len(ev["transitions_index"]) == 11
    assert ev["transitions_index"][0]["changed"] == 2
    assert ev["diffs_fetched"] == [{"t": 0, "n_changed": 2, "value_pairs": {"2->0": 1, "0->2": 1}}]
    assert ev["regions_fetched"] == [{"t": 0, "which": "before", "r": [2, 3], "c": [0, 3]}]
    assert ev["goal_probes"] == [{"idx": 0, "n_grids": 22, "n_true": 0, "constant": True}]
    assert state["budget"]["evicted"] == {
        "regions": 0,
        "diffs": 0,
        "transitions": 0,
        "goal_probes": 0,
        "candidates": 0,
    }
    assert state["budget"]["tokens_est"] > 0
    # The irreducible-core estimate is always recorded (review finding 5).
    assert 0 < state["budget"]["tokens_floor"] <= state["budget"]["tokens_est"]


# ---------------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6540-5: eviction order, keep-set, floor
# ---------------------------------------------------------------------------------


def _cand(i, mismatches):
    return CandidateRecord(
        code=f"import numpy as np\n\ndef engine(grid, action, data):  # variant {i}\n    return grid\n",
        visible_mismatches=mismatches,
        visible_accuracy=0.5,
        visible_cell_recall=0.5,
        holdout_accuracy=0.5,
        holdout_cell_recall=0.5,
        is_memorizing=False,
        has_goal=False,
    )


def _loaded_ledger():
    ledger = compact_mod.EvidenceLedger()
    ledger.regions = [{"t": t, "which": "before", "r": [0, 2], "c": [0, 2]} for t in range(3)]
    ledger.diffs = [{"t": t, "n_changed": 2, "value_pairs": {"0->2": 2}} for t in range(2)]
    ledger.transitions_index = [
        {"t": 0, "action": 1, "changed": 0, "bbox": None},  # inert -> evictable
        {"t": 1, "action": 1, "changed": 4, "bbox": [0, 0, 1, 1]},
        {"t": 2, "action": 1, "changed": 0, "bbox": None},  # inert -> evictable
        {"t": 3, "action": 1, "changed": 2, "bbox": [2, 2, 3, 3]},
    ]
    ledger.goal_probes = [
        {"idx": 0, "n_grids": 8, "n_true": 0, "constant": True},
        {"idx": 1, "n_grids": 8, "n_true": 2, "constant": False},
    ]
    return ledger


def test_eviction_order_regions_go_first(monkeypatch):
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-5. One token under the full size
    must evict exactly one region row -- the cheapest-to-refetch class -- and
    nothing else."""
    session = tools_mod.InductionToolSession(_mover_window(6), cell=1)
    session.candidates = [
        _cand(0, 5),
        _cand(1, 4),
        _cand(2, 0),
        _cand(3, 3),
        _cand(4, 2),
        _cand(5, 1),
    ]
    full, floor = compact_mod.build_carried_state(
        session, _loaded_ledger(), turn=3, budget_tokens=10**6
    )
    assert floor is False
    # The builder sizes the state with tokens_est/tokens_floor still at their 0
    # placeholders; zero them back so the probe matches the internal check.
    probe = json.loads(json.dumps(full))
    probe["budget"]["tokens_est"] = 0
    probe["budget"]["tokens_floor"] = 0
    e_full = compact_mod._estimate_tokens(probe)
    state, floor = compact_mod.build_carried_state(
        session, _loaded_ledger(), turn=3, budget_tokens=e_full - 1
    )
    assert floor is False
    assert state["budget"]["evicted"] == {
        "regions": 1,
        "diffs": 0,
        "transitions": 0,
        "goal_probes": 0,
        "candidates": 0,
    }
    # Oldest region first.
    assert [r["t"] for r in state["evidence"]["regions_fetched"]] == [1, 2]


_SIX_CANDIDATES = [(0, 5), (1, 4), (2, 0), (3, 3), (4, 2), (5, 1)]


def test_eviction_cascade_evicts_every_class_in_order_and_fits():
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-5 and -6. At a budget equal to
    the irreducible core's size, the cascade evicts every evictable row in every
    class -- regions, diffs, inert transitions, goal probes, middle candidates --
    keeps row 0 + best + last two, never touches best.code, and FITS (no floor)."""
    session = tools_mod.InductionToolSession(_mover_window(6), cell=1)
    session.candidates = [_cand(i, m) for i, m in _SIX_CANDIDATES]
    best_code = session.candidates[2].code
    full, _ = compact_mod.build_carried_state(
        session, _loaded_ledger(), turn=9, budget_tokens=10**6
    )
    floor_est = full["budget"]["tokens_floor"]
    state, floor = compact_mod.build_carried_state(
        session, _loaded_ledger(), turn=9, budget_tokens=floor_est
    )
    assert floor is False  # the core fits exactly, so no floor
    assert state["budget"]["evicted"] == {
        "regions": 3,
        "diffs": 2,
        "transitions": 2,
        "goal_probes": 2,
        "candidates": 2,
    }
    assert state["evidence"]["regions_fetched"] == []
    assert state["evidence"]["diffs_fetched"] == []
    assert state["evidence"]["goal_probes"] == []
    # Only the inert (changed == 0) transition rows go; real signal stays.
    assert [r["changed"] for r in state["evidence"]["transitions_index"]] == [4, 2]
    # Keep-set: row 0, the best row (idx 2), and the last two rows.
    assert [r["idx"] for r in state["candidates"]] == [0, 2, 4, 5]
    # Never truncate the refinement target.
    assert state["best"]["code"] == best_code


def test_floor_short_circuit_ships_whole_when_keep_set_alone_busts_budget():
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-5 (2026-08-19 review, finding 5).
    One token below the core's size, no eviction can reach the budget -- so the
    state ships WHOLE: floor flag set, NOTHING evicted, every digest intact,
    best.code verbatim. Destroying re-fetchable evidence for zero gain is the
    defect this pins against."""
    session = tools_mod.InductionToolSession(_mover_window(6), cell=1)
    session.candidates = [_cand(i, m) for i, m in _SIX_CANDIDATES]
    best_code = session.candidates[2].code
    full, _ = compact_mod.build_carried_state(
        session, _loaded_ledger(), turn=9, budget_tokens=10**6
    )
    floor_est = full["budget"]["tokens_floor"]
    state, floor = compact_mod.build_carried_state(
        session, _loaded_ledger(), turn=9, budget_tokens=floor_est - 1
    )
    assert floor is True
    assert state["budget"]["evicted"] == {
        "regions": 0,
        "diffs": 0,
        "transitions": 0,
        "goal_probes": 0,
        "candidates": 0,
    }
    assert len(state["evidence"]["regions_fetched"]) == 3
    assert len(state["evidence"]["diffs_fetched"]) == 2
    assert len(state["evidence"]["goal_probes"]) == 2
    assert len(state["candidates"]) == 6
    assert state["best"]["code"] == best_code
    assert state["budget"]["tokens_floor"] == floor_est


# ---------------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6540-6: repair seed is row 0 in the LIVE carried state
# ---------------------------------------------------------------------------------


def test_repair_seed_is_candidate_row_zero_in_live_carried_state(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-6. A seeded (repair-mode) loop
    carries the seed as candidate row 0: its fingerprint appears in the carried
    state after a live compaction event, and the seed is the best block while it
    is the only candidate."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "10")
    fake = _FakeHTTP(
        chat_replies=[
            _diff_reply(0, "s0", prompt_tokens=1000),
            _diff_reply(1, "s1", prompt_tokens=1200),  # crosses -> fire before request 2
            _engine_reply(GOOD_CODE, "s2", prompt_tokens=900),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = loop_mod.induce_with_tool_loop(
        proposer, "seedgame", _mover_window(6), 1, seed_engine_code=IDENTITY_CODE
    )
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["seeded"] is True
    assert stats["compactions"] == 1
    state = json.loads(_carried_messages(fake.chat_payloads[2])[0]["content"])
    assert state["candidates"][0]["idx"] == 0
    assert state["candidates"][0]["code_sha8"] == compact_mod.code_sha8(IDENTITY_CODE)
    assert state["best"]["code"] == IDENTITY_CODE  # the seed floor, verbatim


def test_seed_resubmission_counts_as_duplicate(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 rule 5. The seed's sha8 is registered at seeding time, so
    the model re-submitting the seed verbatim is a counted duplicate."""
    fake = _FakeHTTP(
        chat_replies=[
            _engine_reply(IDENTITY_CODE, "d1"),  # resubmits the seed -> duplicate
            _engine_reply(GOOD_CODE, "d2"),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = loop_mod.induce_with_tool_loop(
        proposer, "seeddup", _mover_window(6), 1, seed_engine_code=IDENTITY_CODE
    )
    assert ok
    assert proposer.last_tool_loop_stats["duplicate_candidate_submissions"] == 1


# ---------------------------------------------------------------------------------
# Env readers
# ---------------------------------------------------------------------------------


def test_env_readers_defaults_overrides_and_garbage(monkeypatch):
    """REQ-ARC-WMTE-6540 rule 2/4 knobs. Only the exact value "1" enables; the
    numeric knobs fall back to their defaults on garbage."""
    assert compact_mod.compaction_enabled() is False  # unset (autouse fixture)
    for off in ("0", "true", "yes", ""):
        monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", off)
        assert compact_mod.compaction_enabled() is False
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    assert compact_mod.compaction_enabled() is True
    assert compact_mod._growth_tokens() == compact_mod.DEFAULT_GROWTH_TOKENS
    assert compact_mod._state_budget_tokens() == compact_mod.DEFAULT_STATE_BUDGET_TOKENS
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET", "64")
    assert compact_mod._growth_tokens() == 100
    assert compact_mod._state_budget_tokens() == 64
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "not-a-number")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET", "not-a-number")
    assert compact_mod._growth_tokens() == compact_mod.DEFAULT_GROWTH_TOKENS
    assert compact_mod._state_budget_tokens() == compact_mod.DEFAULT_STATE_BUDGET_TOKENS


def test_measured_prompt_tokens_usage_first_then_timings_fallback():
    """REQ-ARC-WMTE-6540 rule 2. usage.prompt_tokens is primary; timings.prompt_n
    is the llama-server fallback; neither -> None (fails toward NO compaction)."""
    assert compact_mod.measured_prompt_tokens({"usage": {"prompt_tokens": 7}}) == 7
    assert compact_mod.measured_prompt_tokens({"timings": {"prompt_n": 9}}) == 9
    assert (
        compact_mod.measured_prompt_tokens(
            {"usage": {"prompt_tokens": 7}, "timings": {"prompt_n": 9}}
        )
        == 7
    )
    assert compact_mod.measured_prompt_tokens({"usage": {"completion_tokens": 3}}) is None
    assert compact_mod.measured_prompt_tokens({}) is None
    # 2026-08-19 review, finding 9: a float count (some backends emit 17000.0)
    # must not silently disable the trigger; bool is never a token count.
    assert compact_mod.measured_prompt_tokens({"usage": {"prompt_tokens": 17000.0}}) == 17000
    assert compact_mod.measured_prompt_tokens({"timings": {"prompt_n": 9.0}}) == 9
    assert compact_mod.measured_prompt_tokens({"usage": {"prompt_tokens": True}}) is None


# ---------------------------------------------------------------------------------
# 2026-08-19 review, finding 2: mutation-proven guard pins. Each of these tests was
# verified to go RED under the exact mutation that previously survived the suite,
# then GREEN with the guard restored.
# ---------------------------------------------------------------------------------


def test_should_compact_fires_at_the_exact_boundary():
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-2. The trigger is `>=`: a
    measured prompt EXACTLY at baseline + growth fires. Pins the `>=` against a
    `>` mutation the prior suite let survive."""
    c = compact_mod.CompactionController(
        enabled=True,
        growth_tokens=100,
        state_budget_tokens=2048,
        baseline_prompt_tokens=1000,
        last_prompt_tokens=1100,
    )
    assert c.should_compact() is True  # exactly baseline + growth
    c.last_prompt_tokens = 1099
    assert c.should_compact() is False  # one token under


def test_failed_result_is_not_evidence():
    """REQ-ARC-WMTE-6540 rule 1/5. A failed tool result must touch NOTHING in the
    ledger: no digest row, no dispatch key, and -- the counting consequence -- no
    sha8 registration, so a later identical SUCCESSFUL submission is the first of
    its kind, not a duplicate. Pins the `if not result.get("ok"): return` guard,
    whose deletion the prior suite let survive."""
    ledger = compact_mod.EvidenceLedger()
    stats = _fresh_stats()
    args = json.dumps({"code": GOOD_CODE})
    ledger.observe("run_engine_on_transitions", args, {"ok": False, "error": "raised"}, stats)
    assert ledger.first_mismatch_by_sha8 == {}  # the failure registered nothing
    ledger.observe("diff_grids", '{"t": 99}', {"ok": False, "error": "out of range"}, stats)
    assert ledger.diffs == []  # a failed fetch is not evidence
    ledger.observe("run_engine_on_transitions", args, {"ok": True, "mismatches": []}, stats)
    assert stats["duplicate_candidate_submissions"] == 0  # first REAL submission
    ledger.observe("run_engine_on_transitions", args, {"ok": True, "mismatches": []}, stats)
    assert stats["duplicate_candidate_submissions"] == 1  # genuine resubmission


def test_diff_refetch_replaces_the_row_instead_of_growing_the_list():
    """REQ-ARC-WMTE-6540 rule 1. A diff digest is deterministic per t: a re-fetch
    replaces the row. Pins the dedupe line whose deletion the prior suite let
    survive (the digest list would grow without bound on re-fetch loops)."""
    session = tools_mod.InductionToolSession(_mover_window(6), cell=1)
    ledger = compact_mod.EvidenceLedger()
    stats = _fresh_stats()
    result = session.diff_grids(0)
    ledger.observe("diff_grids", '{"t": 0}', result, stats)
    ledger.observe("diff_grids", '{"t": 0}', result, stats)
    assert len(ledger.diffs) == 1
    assert ledger.diffs[0]["t"] == 0


def test_region_refetch_does_not_duplicate_the_row():
    """REQ-ARC-WMTE-6540 rule 1. Same pin for regions: identical fetches keep one
    digest row. Pins the `if row not in self.regions` guard against an
    unconditional-append mutation the prior suite let survive."""
    session = tools_mod.InductionToolSession(_mover_window(6), cell=1)
    ledger = compact_mod.EvidenceLedger()
    stats = _fresh_stats()
    result = session.query_region(0, 2, 3, 0, 3, "before")
    ledger.observe("query_region", '{"t": 0}', result, stats)
    ledger.observe("query_region", '{"t": 0}', result, stats)
    assert len(ledger.regions) == 1


def test_floor_hit_reaches_stats_through_the_live_loop(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 rule 4/5, through the LIVE loop (the prior suite only
    pinned the floor at unit level, so deleting the loop's floor wiring survived).
    A 1-token state budget trips the keep-set short-circuit on a real compaction
    event: the stats flag sets, and the injected state ships WHOLE -- digests
    intact, nothing evicted (review finding 5, live)."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "10")
    fake = _FakeHTTP(
        chat_replies=[
            _diff_reply(0, "f0", prompt_tokens=1000),
            _diff_reply(1, "f1", prompt_tokens=1200),  # crosses -> fire before request 2
            _engine_reply(GOOD_CODE, "f2", prompt_tokens=900),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("floorgame", _mover_window(6), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["compactions"] == 1
    assert stats["compact_floor_hit"] is True
    state = json.loads(_carried_messages(fake.chat_payloads[2])[0]["content"])
    assert state["budget"]["tokens_floor"] > 1
    assert set(state["budget"]["evicted"].values()) == {0}  # short-circuit: nothing evicted
    assert {d["t"] for d in state["evidence"]["diffs_fetched"]} == {0, 1}  # digests intact


# ---------------------------------------------------------------------------------
# 2026-08-19 review, finding 1: the tail is the last TOOL round
# ---------------------------------------------------------------------------------


def test_tail_is_last_tool_round_when_a_prose_turn_crosses_the_threshold(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-3 (2026-08-19 review, finding 1).
    The reviewed defect: a prose (no-tool-call) turn crossing the threshold made
    the rebuild keep only [prose, nudge] and compact the last MISMATCH REPORT to a
    digest -- the report the design keeps verbatim. The tail must start at the last
    assistant turn WITH tool_calls and carry the prose turn and nudge after it."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "10")
    fake = _FakeHTTP(
        chat_replies=[
            _diff_reply(0, "p0", prompt_tokens=1000),
            _chat_reply(  # prose turn, no tool calls: crosses the threshold
                {"role": "assistant", "content": "Let me think about the rule."},
                prompt_tokens=1200,
            ),
            _engine_reply(GOOD_CODE, "p2", prompt_tokens=1300),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("prosegame", _mover_window(6), 1)
    assert ok
    assert proposer.last_tool_loop_stats["compactions"] == 1
    rebuilt = fake.chat_payloads[2]["messages"]
    # [base, carried, tool-round assistant, its tool result, prose turn, nudge]
    assert len(rebuilt) == 6
    assert rebuilt[1]["role"] == "user"
    assert compact_mod.CARRIED_STATE_KIND in rebuilt[1]["content"]
    assert rebuilt[2]["role"] == "assistant"
    assert rebuilt[2]["tool_calls"][0]["id"] == "p0"
    assert rebuilt[3]["role"] == "tool" and rebuilt[3]["tool_call_id"] == "p0"
    assert rebuilt[4] == {"role": "assistant", "content": "Let me think about the rule."}
    assert rebuilt[5]["role"] == "user"  # the no-tool-call nudge rides along
    _assert_tool_ids_resolve(fake.chat_payloads[2])


# ---------------------------------------------------------------------------------
# 2026-08-19 review, finding 3: thrash floor + alarm
# ---------------------------------------------------------------------------------


def test_thrash_floor_refire_requires_growth_beyond_the_post_rebuild_prompt(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 / SCENARIO-ARC-WMTE-6540-2 (2026-08-19 review, finding 3).
    After a rebuild whose compacted prompt still sits above the design threshold,
    the trigger must NOT re-fire until the prompt grows by `growth` beyond the
    first post-rebuild measurement. Under the pre-fix rule, the 10300 and 10350
    turns here would BOTH have re-fired (>= 10100); a probe measured 10 events in
    11 turns that way."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "99")
    fake = _FakeHTTP(
        chat_replies=[
            _diff_reply(0, "h0", prompt_tokens=10000),  # baseline
            _diff_reply(1, "h1", prompt_tokens=10200),  # >= 10100 -> fire 1
            _diff_reply(0, "h2", prompt_tokens=10300),  # post-rebuild floor = 10300
            _diff_reply(1, "h3", prompt_tokens=10350),  # < 10400 -> NO re-fire
            _diff_reply(0, "h4", prompt_tokens=10450),  # >= 10400 -> fire 2
            _engine_reply(GOOD_CODE, "h5", prompt_tokens=10500),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("thrashgame", _mover_window(6), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["compactions"] == 2
    assert stats["compaction_thrash_alarm"] is False  # 2 events, no alarm
    carried_per_payload = [len(_carried_messages(p)) for p in fake.chat_payloads]
    assert carried_per_payload == [0, 0, 1, 1, 1, 1]
    # The second event injected a FRESH state (its turn stamp moved on).
    first_state = json.loads(_carried_messages(fake.chat_payloads[2])[0]["content"])
    second_state = json.loads(_carried_messages(fake.chat_payloads[5])[0]["content"])
    assert first_state["turn"] == 2
    assert second_state["turn"] == 5


def test_compaction_alarm_raises_past_the_design_threshold(monkeypatch, proposer, caplog):
    """REQ-ARC-WMTE-6540 rule 2 (2026-08-19 review, finding 3). Design section 10:
    expected <= 3 events per loop, alarm above 5. Past 5 events the loop sets
    `compaction_thrash_alarm` and logs a warning -- active visibility, not a
    number a reader must notice in the artifact."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_TURNS", "20")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "99")
    # Prompt sizes climb 3 tokens per turn against growth=1, so the trigger
    # re-fires as fast as the thrash floor allows: every second turn.
    replies = [_diff_reply(0, f"a{i}", prompt_tokens=1000 + 3 * i) for i in range(15)]
    replies.append(_engine_reply(GOOD_CODE, "a15", prompt_tokens=1100))
    fake = _FakeHTTP(chat_replies=replies)
    _patch_http(monkeypatch, fake)
    with caplog.at_level(logging.WARNING, logger=loop_mod.__name__):
        ok, _ = proposer.induce("alarmgame", _mover_window(6), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["compactions"] > compact_mod.COMPACTION_ALARM_THRESHOLD
    assert stats["compaction_thrash_alarm"] is True
    assert "compaction thrash" in caplog.text


# ---------------------------------------------------------------------------------
# 2026-08-19 review, finding 4: refetch keys normalize tool-side defaults
# ---------------------------------------------------------------------------------


def test_refetch_key_normalizes_tool_side_defaults():
    """REQ-ARC-WMTE-6540 rule 5 (2026-08-19 review, finding 4). The refetch key is
    derived from the RESULT, so `query_region(..., which="before")` and the same
    region with `which` omitted -- identical cells -- key equal, and a real
    post-compaction re-fetch counts. Raw-kwargs keys silently counted zero here."""
    session = tools_mod.InductionToolSession(_mover_window(6), cell=1)
    ledger = compact_mod.EvidenceLedger()
    stats = _fresh_stats()
    explicit = session.query_region(0, 2, 3, 0, 3, "before")
    ledger.observe(
        "query_region",
        '{"t": 0, "r0": 2, "r1": 3, "c0": 0, "c1": 3, "which": "before"}',
        explicit,
        stats,
    )
    ledger.note_compaction()
    defaulted = session.query_region(0, 2, 3, 0, 3)  # `which` omitted, same cells
    ledger.observe("query_region", '{"t": 0, "r0": 2, "r1": 3, "c0": 0, "c1": 3}', defaulted, stats)
    assert stats["refetch_tool_calls_post_compaction"] == 1
    # A genuinely DIFFERENT region after compaction is not a re-fetch.
    other = session.query_region(1, 2, 3, 0, 3)
    ledger.observe("query_region", '{"t": 1, "r0": 2, "r1": 3, "c0": 0, "c1": 3}', other, stats)
    assert stats["refetch_tool_calls_post_compaction"] == 1


# ---------------------------------------------------------------------------------
# 2026-08-19 review, finding 6: compaction-attributable transport failure
# ---------------------------------------------------------------------------------


class _FailingHTTP(_FakeHTTP):
    """Scripted server that raises on the Nth chat call (0-based), the shape of a
    strict-alternation chat template rejecting the rebuild's consecutive user
    messages."""

    def __init__(self, chat_replies=(), fail_on_call=0):
        super().__init__(chat_replies=chat_replies)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, req, timeout=None):
        n = self.calls
        self.calls += 1
        if n == self.fail_on_call:
            raise urllib.error.URLError("template rejected consecutive user messages")
        return super().__call__(req, timeout=timeout)


def test_transport_error_on_the_first_post_rebuild_request_is_attributed(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 rule 3 (2026-08-19 review, finding 6). When the FIRST
    request after a rebuild fails, `transport_error_on_compacted_request` is set,
    so a strict-alternation template rejection is distinguishable from the plain
    transport failures `terminated_by=transport_error` also covers."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    fake = _FailingHTTP(
        chat_replies=[
            _diff_reply(0, "e0", prompt_tokens=1000),
            _diff_reply(1, "e1", prompt_tokens=1200),  # crosses -> fire before call 2
        ],
        fail_on_call=2,  # the first post-rebuild request
    )
    _patch_http(monkeypatch, fake)
    ok, note = loop_mod.induce_with_tool_loop(proposer, "xportgame", _mover_window(6), 1)
    assert ok is False
    stats = proposer.last_tool_loop_stats
    assert stats["terminated_by"] == "transport_error"
    assert stats["compactions"] == 1
    assert stats["transport_error_on_compacted_request"] is True


def test_transport_error_without_compaction_is_not_attributed(monkeypatch, proposer):
    """REQ-ARC-WMTE-6540 rule 3, the control: a transport failure on an ordinary
    (never-compacted) request must NOT carry the compaction attribution."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", "100")
    fake = _FailingHTTP(chat_replies=[_diff_reply(0, "e0", prompt_tokens=1000)], fail_on_call=0)
    _patch_http(monkeypatch, fake)
    ok, _ = loop_mod.induce_with_tool_loop(proposer, "xportgame2", _mover_window(6), 1)
    assert ok is False
    stats = proposer.last_tool_loop_stats
    assert stats["terminated_by"] == "transport_error"
    assert stats["compactions"] == 0
    assert "transport_error_on_compacted_request" not in stats
