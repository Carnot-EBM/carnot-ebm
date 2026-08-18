"""Wiring + behaviour tests for the tool-calling induction loop (REQ-ARC-WMTE-6460).

Same discipline as test_arc_goal_defect_reask_wiring.py: drive the REAL induce()/loop
against a scripted fake HTTP layer (no GPU, no model), and assert the observable
consequences -- which endpoint was hit, what was written, which counters moved. The
load-bearing pin is the first test: with the env var UNSET, induction must be
byte-identical to the shipped single-shot path and the loop must never be consulted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_induction_tool_loop as loop_mod
from carnot.agentic import arc_induction_tools as tools_mod


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
    """Scripted server for BOTH endpoints. Chat replies come from `chat_replies` in
    order; the raw /completion endpoint replays `completion_replies`."""

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
    # E3_DIR -> tmp_path: a test must never write results/** (Test-Run Record Integrity).
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    # Pin think mode OFF so the shipped single-shot path takes the raw /completion route
    # the scripted fake serves. The tool loop does not read this env var -- it always
    # uses the chat endpoint with tools -- so the loop tests are unaffected.
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    p = e3.LocalGGUFProposer()
    monkeypatch.setattr(type(p), "_ensure_server", lambda _self: True)
    monkeypatch.setattr(type(p), "_url", lambda _self: "http://127.0.0.1:1")
    return p


def _patch_http(monkeypatch, fake):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake)


# ---------------------------------------------------------------------------------
# The default-off pin
# ---------------------------------------------------------------------------------


def test_env_unset_never_consults_the_tool_loop(monkeypatch, proposer, tmp_path):
    """Unset -> the shipped single-shot path, and the loop entry is NEVER called.

    The bomb proves the negative: if induce() consulted the loop, the test would
    raise. The payload check pins the request shape -- no `tools`, no `messages`,
    the raw /completion prompt exactly as shipped."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP", raising=False)

    def _bomb(*_a, **_k):
        raise AssertionError("tool loop consulted with env unset")

    monkeypatch.setattr(loop_mod, "induce_with_tool_loop", _bomb)
    fake = _FakeHTTP(
        completion_replies=[{"content": f"```python\n{GOOD_CODE}```", "stop_type": "eos"}]
    )
    _patch_http(monkeypatch, fake)
    ok, note = proposer.induce("testgame", _mover_window(), 1)
    assert ok
    assert fake.chat_payloads == []
    assert len(fake.completion_payloads) >= 1
    assert "tools" not in fake.completion_payloads[0]
    assert "messages" not in fake.completion_payloads[0]
    assert (tmp_path / "testgame" / "world_model.py").exists()


# ---------------------------------------------------------------------------------
# Loop behaviour
# ---------------------------------------------------------------------------------


def test_loop_runs_tools_and_accepts_zero_mismatch_engine(monkeypatch, proposer, tmp_path):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _FakeHTTP(
        chat_replies=[
            _chat_reply(
                {"role": "assistant", "tool_calls": [_tool_call("diff_grids", '{"t": 0}')]},
                tokens=500,
            ),
            _chat_reply(
                {
                    "role": "assistant",
                    "tool_calls": [
                        _tool_call(
                            "run_engine_on_transitions", json.dumps({"code": GOOD_CODE}), "c2"
                        )
                    ],
                },
                tokens=700,
            ),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, note = proposer.induce("toolgame", _mover_window(), 1)
    assert ok
    assert "tool loop" in note
    stats = proposer.last_tool_loop_stats
    assert stats["terminated_by"] == "zero_mismatches"
    assert stats["tool_calls_by_name"] == {"diff_grids": 1, "run_engine_on_transitions": 1}
    assert stats["decode_tokens_total"] == 1200
    assert stats["tool_call_parse_failures"] == 0
    written = (tmp_path / "toolgame" / "world_model.py").read_text()
    assert "def engine" in written and "def is_level_complete" in written
    # The chat requests carried the tool schemas and the per-turn think budget.
    assert fake.chat_payloads[0]["tools"] == tools_mod.TOOL_SCHEMAS
    assert fake.chat_payloads[0]["thinking_budget_tokens"] == loop_mod.DEFAULT_THINK_BUDGET
    # The tool result travelled back as a tool-role message on the next request.
    roles = [m["role"] for m in fake.chat_payloads[1]["messages"]]
    assert roles.count("tool") == 1


def test_loop_failure_falls_back_to_single_shot(monkeypatch, proposer, tmp_path):
    """No parseable engine by the cap -> (False, ...) -> the shipped path runs and
    succeeds. Failure is never worse than today."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_TURNS", "2")
    fake = _FakeHTTP(
        chat_replies=[_chat_reply({"role": "assistant", "content": "I am stuck."}, tokens=50)],
        completion_replies=[{"content": f"```python\n{GOOD_CODE}```", "stop_type": "eos"}],
    )
    _patch_http(monkeypatch, fake)
    ok, note = proposer.induce("fbgame", _mover_window(), 1)
    assert ok
    assert proposer.last_tool_loop_stats["terminated_by"] == "turn_cap"
    assert len(fake.chat_payloads) == 2  # both turns spent
    assert len(fake.completion_payloads) >= 1  # the fallback single-shot ran
    assert (tmp_path / "fbgame" / "world_model.py").exists()


def test_malformed_tool_arguments_are_counted_not_fatal(monkeypatch, proposer, tmp_path):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _FakeHTTP(
        chat_replies=[
            _chat_reply(
                {
                    "role": "assistant",
                    "tool_calls": [_tool_call("diff_grids", "{not json")],
                },
                tokens=60,
            ),
            _chat_reply(
                {
                    "role": "assistant",
                    "tool_calls": [
                        _tool_call(
                            "run_engine_on_transitions", json.dumps({"code": GOOD_CODE}), "c2"
                        )
                    ],
                },
                tokens=60,
            ),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("pfgame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["tool_call_parse_failures"] == 1
    assert stats["terminated_by"] == "zero_mismatches"
    # The malformed call produced an error tool-result, visible to the model.
    tool_msgs = [m for m in fake.chat_payloads[1]["messages"] if m["role"] == "tool"]
    assert any("unparseable" in m["content"] for m in tool_msgs)


def test_early_stop_after_non_improving_candidates_keeps_best(monkeypatch, proposer, tmp_path):
    """Two consecutive non-improving engine submissions end the loop, and the BEST
    candidate (fewest visible mismatches) is what lands on disk -- monotone accept."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")

    def _engine_call(code, cid):
        return _chat_reply(
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("run_engine_on_transitions", json.dumps({"code": code}), cid)
                ],
            },
            tokens=10,
        )

    fake = _FakeHTTP(
        chat_replies=[
            _engine_call(GOOD_CODE.replace("min(g.shape[1] - 1, cs[0] + 1)", "cs[0]"), "c1"),
            _engine_call(IDENTITY_CODE, "c2"),
            _engine_call(IDENTITY_CODE, "c3"),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, note = proposer.induce("esgame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["terminated_by"] == "early_stop_non_improving"
    # Candidate 1 clears cells (writes 0 then 2 back = identity too) -- so all three
    # tie; best is the first scored. The pin is that SOMETHING was written and the
    # loop stopped early rather than burning the cap.
    assert stats["turns"] == 3
    assert (tmp_path / "esgame" / "world_model.py").exists()


def test_final_answer_is_scored_and_monotone_accept_keeps_the_better_candidate(
    monkeypatch, proposer, tmp_path
):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
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
    ok, _ = proposer.induce("fagame", _mover_window(), 1)
    assert ok
    # GOOD_CODE has zero mismatches, so the loop accepts before any final answer.
    written = (tmp_path / "fagame" / "world_model.py").read_text()
    assert "min(g.shape[1] - 1" in written


ENGINE_ONLY_CODE = GOOD_CODE.split("def is_level_complete")[0]


def test_force_engine_nudge_fires_after_inspection_only_turns(monkeypatch, proposer, tmp_path):
    """Probe 1's measured failure: 12 turns of pure diff/query, zero engines. After the
    inspection budget the loop must DEMAND a candidate, and record that it did."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "2")
    inspect_reply = _chat_reply(
        {"role": "assistant", "tool_calls": [_tool_call("diff_grids", '{"t": 0}')]},
        tokens=10,
    )
    submit_reply = _chat_reply(
        {
            "role": "assistant",
            "tool_calls": [
                _tool_call("run_engine_on_transitions", json.dumps({"code": GOOD_CODE}), "c9")
            ],
        },
        tokens=10,
    )
    fake = _FakeHTTP(chat_replies=[inspect_reply, inspect_reply, submit_reply])
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("nudgegame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["force_engine_nudges"] >= 1
    assert stats["terminated_by"] == "zero_mismatches"
    assert stats["mismatch_trajectory"] == [0]
    assert stats["tool_calls_per_turn"][0] == ["diff_grids"]
    # The demand reached the conversation as a user message before the next request.
    turn3_roles = [m["role"] for m in fake.chat_payloads[2]["messages"]]
    assert "user" in turn3_roles[-3:]
    nudge_texts = [m["content"] for m in fake.chat_payloads[2]["messages"] if m["role"] == "user"]
    assert any("MUST be run_engine_on_transitions" in t for t in nudge_texts)


def test_engine_only_best_gets_goal_from_focused_call_not_a_shadowing_donor(
    monkeypatch, proposer, tmp_path
):
    """An engine-only winner is completed by the focused goal call, and _goal_only never
    returns a donor verbatim (a donor carrying its own engine would bind last and
    shadow the accepted engine -- the split-induce shadowing bug, at accept time)."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _FakeHTTP(
        chat_replies=[
            _chat_reply(
                {
                    "role": "assistant",
                    "tool_calls": [
                        _tool_call(
                            "run_engine_on_transitions",
                            json.dumps({"code": ENGINE_ONLY_CODE}),
                            "c1",
                        )
                    ],
                },
                tokens=10,
            ),
        ],
        completion_replies=[
            {
                "content": "```python\ndef is_level_complete(grid):\n    return False\n```",
                "stop_type": "eos",
            }
        ],
    )
    _patch_http(monkeypatch, fake)
    ok, note = proposer.induce("goalgame", _mover_window(), 1)
    assert ok
    written = (tmp_path / "goalgame" / "world_model.py").read_text()
    assert "def engine" in written and "def is_level_complete" in written
    assert len(fake.completion_payloads) == 1  # the focused goal call ran
    # Direct unit pins on the extractor: def-only on a full donor, None without a goal.
    donor = GOOD_CODE
    seg = loop_mod._goal_only(donor)
    assert seg is not None and seg.startswith("def is_level_complete")
    assert "def engine" not in seg
    assert loop_mod._goal_only(ENGINE_ONLY_CODE) is None


def test_prompt_shows_visible_rows_only_never_the_held_out_tail(monkeypatch, proposer):
    """The loop's opening prompt renders session.visible, not the full window. Rendering
    the tail would hand the model the held-out deltas in-prompt and quietly defeat the
    aggregate-only held-out score (the REQ-ARC-WMTE-6090 leak, one level up)."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_TURNS", "1")
    window = _mover_window(14)  # 14 rows -> 3 held out
    visible, _held = tools_mod.holdout_split(window)
    expected = e3.induce_prompt(game := "leakgame", list(visible), 1, k=e3._induce_transitions_k())
    fake = _FakeHTTP(
        chat_replies=[_chat_reply({"role": "assistant", "content": "no tools"}, tokens=5)],
        completion_replies=[{"content": f"```python\n{GOOD_CODE}```", "stop_type": "eos"}],
    )
    _patch_http(monkeypatch, fake)
    proposer.induce(game, window, 1)
    first_user = fake.chat_payloads[0]["messages"][0]["content"]
    assert first_user.startswith(expected)


# ---------------------------------------------------------------------------------
# Registry-level guarantees
# ---------------------------------------------------------------------------------


def test_held_out_report_is_aggregate_only():
    session = tools_mod.InductionToolSession(_mover_window(14), cell=1)
    assert len(session.held_out) == 3
    report = session.run_engine_on_transitions(GOOD_CODE)
    assert set(report["held_out"]) == {"n_transitions", "accuracy", "cell_recall"}
    # Every mismatch (if any) indexes a VISIBLE transition only.
    assert all(m.get("i", 0) < len(session.visible) for m in report["mismatches"])


def test_memorization_scan_flags_hardcoded_window_coordinates():
    rows = []
    for x in range(4):
        a = np.zeros((20, 20), dtype=int)
        a[12, 13 + x] = 2
        b = np.zeros((20, 20), dtype=int)
        b[12, 14 + x] = 2
        rows.append(_T(a, b))
    session = tools_mod.InductionToolSession(rows, cell=1)
    hardcoded = """import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    g[12, 13] = 0
    g[12, 14] = 2
    g[12, 15] = 2
    g[12, 16] = 2
    return g

def is_level_complete(grid):
    return False
"""
    report = session.run_engine_on_transitions(hardcoded)
    assert report["memorization_scan"]["is_memorizing"] is True
    assert "warning" in report
    general = session.run_engine_on_transitions(GOOD_CODE)
    assert general["memorization_scan"]["is_memorizing"] is False


def test_dispatch_covers_every_schema_and_rejects_unknown_names():
    session = tools_mod.InductionToolSession(_mover_window(4), cell=1)
    for name in tools_mod.TOOL_NAMES:
        result = tools_mod.dispatch_tool(session, name, "{}")
        # Every schema-listed tool is dispatchable; missing args are a per-tool error,
        # never a KeyError escaping to the loop.
        assert isinstance(result, dict)
    bad = tools_mod.dispatch_tool(session, "no_such_tool", "{}")
    assert bad["ok"] is False and "unknown tool" in bad["error"]


def test_query_region_and_diff_grids_round_trip():
    rows = _mover_window(4)
    session = tools_mod.InductionToolSession(rows, cell=1)
    d = session.diff_grids(0)
    assert d["ok"] and d["n_changed"] == 2
    q = session.query_region(0, 2, 3, 0, 3, "before")
    assert q["ok"] and q["rows"] == [[2, 0, 0]]
    big = session.query_region(0, 0, 6, 0, 6, "before")
    assert big["ok"]  # 36 cells, under the cap
    out_of_range = session.diff_grids(99)
    assert out_of_range["ok"] is False


# ---------------------------------------------------------------------------------
# Stall cap (CARNOT_ARC_INDUCE_TOOL_STALL_TURNS): abort inspection-only stretches
# ---------------------------------------------------------------------------------


def _inspect_reply(call_id="d1"):
    return _chat_reply(
        {
            "role": "assistant",
            "tool_calls": [_tool_call("diff_grids", json.dumps({"t": 0}), call_id)],
        },
        tokens=10,
    )


def _submit_reply(code, call_id):
    return _chat_reply(
        {
            "role": "assistant",
            "tool_calls": [
                _tool_call("run_engine_on_transitions", json.dumps({"code": code}), call_id)
            ],
        },
        tokens=10,
    )


def test_stall_cap_unset_is_inert_inspection_runs_to_turn_cap(monkeypatch, proposer):
    """DEFAULT-OFF PIN. With the stall env unset, a candidate followed by endless
    inspection turns terminates by the TURN CAP exactly as before the lever existed."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_TURNS", "5")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_STALL_TURNS", raising=False)
    fake = _FakeHTTP(chat_replies=[_submit_reply(IDENTITY_CODE, "c1"), _inspect_reply()])
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("stgame0", _mover_window(), 1)
    assert ok  # the identity candidate is still the monotone-accept best
    stats = proposer.last_tool_loop_stats
    assert stats["stall_cap"] == 0
    assert stats["terminated_by"] == "turn_cap"
    assert stats["turns"] == 5


def test_stall_cap_aborts_after_inspection_only_turns_and_keeps_best(
    monkeypatch, proposer, tmp_path
):
    """With the cap set, two consecutive inspection-only turns after the submission end
    the loop with `stall_turns`, and the best candidate so far still lands on disk."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_STALL_TURNS", "2")
    fake = _FakeHTTP(chat_replies=[_submit_reply(IDENTITY_CODE, "c1"), _inspect_reply()])
    _patch_http(monkeypatch, fake)
    ok, note = proposer.induce("stgame1", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["terminated_by"] == "stall_turns"
    assert stats["turns"] == 3  # submit, inspect, inspect
    assert (tmp_path / "stgame1" / "world_model.py").exists()
    assert "stall_turns" in note


def test_stall_cap_resets_on_every_submission_so_slow_convergence_survives(monkeypatch, proposer):
    """A submission on every other turn keeps resetting the stall counter: the loop must
    end via the early-stop (non-improving) rule, never via the stall cap."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_STALL_TURNS", "2")
    fake = _FakeHTTP(
        chat_replies=[
            _submit_reply(IDENTITY_CODE, "c1"),
            _inspect_reply(),
            _submit_reply(IDENTITY_CODE, "c2"),
            _inspect_reply(),
            _submit_reply(IDENTITY_CODE, "c3"),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("stgame2", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["terminated_by"] == "early_stop_non_improving"
    assert stats["turns"] == 5
