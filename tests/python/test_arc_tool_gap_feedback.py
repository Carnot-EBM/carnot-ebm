"""Tool-gap feedback: capture, curated candidate tools, offline analyzer (REQ-ARC-WMTE-6770).

Same discipline as test_arc_induction_tool_loop.py: the loop tests drive the REAL
induce() entry point against a scripted fake HTTP layer, so a rule whose call site
is deleted goes RED here, not only in a pure-helper test. The analyzer tests write
only under tmp_path (Test-Run Record Integrity).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import carnot.agentic.arc_executable_world_model as e3
import carnot.agentic.arc_induction_tool_loop as loop_mod
import carnot.agentic.arc_induction_tools as tools_mod
import carnot.agentic.arc_tool_gap_refinement as gap_mod
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


@pytest.fixture(autouse=True)
def _clean_candidate_registry(monkeypatch):
    """Snapshot/restore the module-global registry, and start each test with
    the candidate env flag unset so no state leaks between tests."""
    monkeypatch.delenv(tools_mod.CANDIDATE_TOOLS_ENV, raising=False)
    saved = dict(tools_mod.CANDIDATE_TOOLS)
    yield
    tools_mod.CANDIDATE_TOOLS.clear()
    tools_mod.CANDIDATE_TOOLS.update(saved)


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


COUNT_COLORS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "count_colors",
        "description": "Count distinct colors in one observed grid.",
        "parameters": {
            "type": "object",
            "properties": {
                "t": {"type": "integer", "description": "transition index (0-based)"},
                "which": {"type": "string", "enum": ["before", "after"]},
            },
            "required": ["t"],
        },
    },
}


def _count_colors_factory(session):
    def count_colors(t, which="before"):
        tr = session.visible[int(t)]
        grid = np.asarray(tr.next_grid if which == "after" else tr.grid)
        return {"ok": True, "t": int(t), "n_colors": int(len(np.unique(grid)))}

    return count_colors


def _session():
    return tools_mod.InductionToolSession(_mover_window(), cell=1)


# ---------------------------------------------------------------------------
# Capture at the dispatch chokepoint (SCENARIO-ARC-WMTE-6770-2 / -3 / -4)
# ---------------------------------------------------------------------------


def test_unknown_tool_dispatch_records_gap_event():
    """SCENARIO-ARC-WMTE-6770-2: the strongest gap signal keeps its identity."""
    s = _session()
    out = tools_mod.dispatch_tool(s, "get_full_grid", '{"t": 0, "which": "before"}')
    assert out["ok"] is False and "unknown tool" in out["error"]
    assert s.tool_gap_events == [
        {
            "kind": "unknown_tool",
            "requested_tool": "get_full_grid",
            "argument_keys": ["t", "which"],
        }
    ]


def test_bad_arguments_dispatch_records_gap_event():
    """SCENARIO-ARC-WMTE-6770-3: an imagined signature on a real tool is retained."""
    s = _session()
    out = tools_mod.dispatch_tool(s, "diff_grids", '{"t": 0, "layer": 2}')
    assert out["ok"] is False and "bad arguments" in out["error"]
    assert len(s.tool_gap_events) == 1
    event = s.tool_gap_events[0]
    assert event["kind"] == "bad_arguments" and event["tool"] == "diff_grids"
    assert "layer" in event["error"]


def test_gap_events_are_bounded_with_visible_overflow():
    """SCENARIO-ARC-WMTE-6770-4 (bound half): capped at MAX_TOOL_GAP_EVENTS,
    overflow counted rather than silently lost."""
    s = _session()
    for i in range(tools_mod.MAX_TOOL_GAP_EVENTS + 5):
        tools_mod.dispatch_tool(s, f"phantom_{i}", "{}")
    assert len(s.tool_gap_events) == tools_mod.MAX_TOOL_GAP_EVENTS
    assert s.tool_gap_events_dropped == 5


# ---------------------------------------------------------------------------
# Entry point: the loop itself (SCENARIO-ARC-WMTE-6770-2 loop half, -4, -5, -6)
# ---------------------------------------------------------------------------


def _two_turn_fake(first_call):
    return _FakeHTTP(
        chat_replies=[
            _chat_reply({"role": "assistant", "tool_calls": [first_call]}, tokens=60),
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


def test_loop_carries_unknown_tool_gap_event_into_stats(monkeypatch, proposer, tmp_path):
    """SCENARIO-ARC-WMTE-6770-2 (loop half): an unknown-tool call made through the
    REAL induce() entry point lands in last_tool_loop_stats, which every row
    consumer already copies."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _two_turn_fake(_tool_call("get_full_grid", '{"t": 0}'))
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("gapgame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["tool_gap_events"] == [
        {"kind": "unknown_tool", "requested_tool": "get_full_grid", "argument_keys": ["t"]}
    ]
    assert stats["tool_gap_events_dropped"] == 0
    # The pre-existing conflated counter still moves; the event resolves it.
    assert stats["tool_call_parse_failures"] == 1


def test_clean_loop_run_reports_gap_keys_present_and_empty(monkeypatch, proposer, tmp_path):
    """SCENARIO-ARC-WMTE-6770-4 (presence half): absence is not zero — a clean
    run must say [] out loud, so an artifact reader can tell 'no demand' from
    'capture never ran'."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _two_turn_fake(_tool_call("diff_grids", '{"t": 0}'))
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("cleangame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["tool_gap_events"] == []
    assert stats["tool_gap_events_dropped"] == 0
    assert stats["candidate_tools_enabled"] == []
    assert stats["candidate_tools_rejected"] == []


def test_registered_candidate_stays_dark_with_env_unset(monkeypatch, proposer, tmp_path):
    """SCENARIO-ARC-WMTE-6770-1: default byte-identical. A registered candidate
    with the env unset must not leak into the request payload, and calling it
    is an unknown-tool refusal that records a gap event."""
    tools_mod.register_candidate_tool(COUNT_COLORS_SCHEMA, _count_colors_factory)
    assert tools_mod.active_tool_schemas() is tools_mod.TOOL_SCHEMAS
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _two_turn_fake(_tool_call("count_colors", '{"t": 0}'))
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("darkgame", _mover_window(), 1)
    assert ok
    assert fake.chat_payloads[0]["tools"] == tools_mod.TOOL_SCHEMAS
    stats = proposer.last_tool_loop_stats
    assert stats["tool_gap_events"] == [
        {"kind": "unknown_tool", "requested_tool": "count_colors", "argument_keys": ["t"]}
    ]


def test_enabled_candidate_reaches_request_and_dispatch(monkeypatch, proposer, tmp_path):
    """SCENARIO-ARC-WMTE-6770-5: the introduction path, end to end through the
    REAL induce() entry point — schema in the payload, dispatch executed,
    usage measurable in tool_calls_by_name."""
    tools_mod.register_candidate_tool(COUNT_COLORS_SCHEMA, _count_colors_factory)
    monkeypatch.setenv(tools_mod.CANDIDATE_TOOLS_ENV, "count_colors")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _two_turn_fake(_tool_call("count_colors", '{"t": 0}'))
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("candgame", _mover_window(), 1)
    assert ok
    assert fake.chat_payloads[0]["tools"] == [*tools_mod.TOOL_SCHEMAS, COUNT_COLORS_SCHEMA]
    tool_msgs = [m for m in fake.chat_payloads[1]["messages"] if m["role"] == "tool"]
    assert any("n_colors" in m["content"] for m in tool_msgs)
    stats = proposer.last_tool_loop_stats
    assert stats["tool_calls_by_name"]["count_colors"] == 1
    assert stats["candidate_tools_enabled"] == ["count_colors"]
    assert stats["tool_gap_events"] == []


def test_unregistered_env_name_is_visibly_rejected(monkeypatch, proposer, tmp_path):
    """SCENARIO-ARC-WMTE-6770-6: an env typo must be visible in the stats, not
    silently disable the tool it meant to arm."""
    monkeypatch.setenv(tools_mod.CANDIDATE_TOOLS_ENV, "no_such_tool")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    fake = _two_turn_fake(_tool_call("diff_grids", '{"t": 0}'))
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("typogame", _mover_window(), 1)
    assert ok
    assert fake.chat_payloads[0]["tools"] == tools_mod.TOOL_SCHEMAS
    stats = proposer.last_tool_loop_stats
    assert stats["candidate_tools_enabled"] == []
    assert stats["candidate_tools_rejected"] == ["no_such_tool"]


def test_selfparse_prompt_and_coercion_cover_enabled_candidate():
    """SCENARIO-ARC-WMTE-6770-5 (selfparse half): the prompt text announces the
    candidate and the XML parser coerces by its schema, so both transports
    serve the same active set."""
    import os

    tools_mod.register_candidate_tool(COUNT_COLORS_SCHEMA, _count_colors_factory)
    os.environ[tools_mod.CANDIDATE_TOOLS_ENV] = "count_colors"
    try:
        text = tools_mod.render_tool_schemas_for_prompt()
        assert "- count_colors(" in text
        calls, n_blocks, n_unparsed = tools_mod.parse_xml_tool_calls(
            "<tool_call>\n<function=count_colors>\n<parameter=t>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        assert n_blocks == 1 and n_unparsed == 0
        assert json.loads(calls[0]["function"]["arguments"]) == {"t": 3}
    finally:
        del os.environ[tools_mod.CANDIDATE_TOOLS_ENV]
    # Unset again: the candidate must vanish from the rendered prompt.
    assert "- count_colors(" not in tools_mod.render_tool_schemas_for_prompt()


def test_register_candidate_refuses_core_name_collision():
    """REQ-ARC-WMTE-6770 rule: a candidate must ADD, never shadow a core tool."""
    bad = json.loads(json.dumps(COUNT_COLORS_SCHEMA))
    bad["function"]["name"] = "diff_grids"
    with pytest.raises(ValueError, match="collides"):
        tools_mod.register_candidate_tool(bad, _count_colors_factory)


# ---------------------------------------------------------------------------
# Entry point: the LIVE agent's repair record (SCENARIO-ARC-WMTE-6770-7)
# ---------------------------------------------------------------------------

_IDENTITY_CODE = """import numpy as np

def engine(grid, action, data):
    return grid

def is_level_complete(grid):
    return False
"""

_RECOVER_CODE = """import numpy as np

def engine(grid, action, data):
    out = np.asarray(grid).copy()
    out[:] = 5
    return out

def is_level_complete(grid):
    return False
"""


def test_live_repair_record_carries_tool_gap_evidence(tmp_path, monkeypatch):
    """SCENARIO-ARC-WMTE-6770-7: the live E3 policy's tool_loop record subset
    must carry the gap fields — without this the mechanism is blind on
    exactly the scored path."""
    game = "gp01"
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "repair")
    monkeypatch.delenv("CARNOT_ARC_RECALL_RESAMPLE", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRUST_METRIC", raising=False)
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    store = tmp_path / game / "world_model.py"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(_IDENTITY_CODE)

    trans = []
    for i in range(4):
        trans.append(
            e3.Transition(
                grid=np.zeros((4, 4), dtype=int),
                action=1 + (i % 3),
                data=None,
                next_grid=np.full((4, 4), 5, dtype=int),
                level_before=0,
                level_after=0,
            )
        )
    old_engine, old_is_done = e3.load_engine(game)
    vr = e3.WorldModelVerifier(trans).score(old_engine)
    assert vr.cell_recall == 0.0

    events = [{"kind": "unknown_tool", "requested_tool": "get_full_grid", "argument_keys": []}]

    def _fake_loop(prop, g, rows, cell, **kwargs):
        prop.last_tool_loop_stats = {
            "turns": 2,
            "terminated_by": "zero_mismatches",
            "candidates_scored": 2,
            "best_visible_mismatches": 0,
            "seed_visible_mismatches": 4,
            "decode_tokens_total": 500,
            "tool_call_parse_failures": 1,
            "tool_gap_events": list(events),
            "tool_gap_events_dropped": 0,
            "candidate_tools_enabled": [],
            "candidate_tools_rejected": [],
        }
        path = e3.E3_DIR / g / "world_model.py"
        path.write_text(_RECOVER_CODE)
        return True, "tool loop: zero_mismatches"

    monkeypatch.setattr(loop_mod, "induce_with_tool_loop", _fake_loop)

    class _FakeProposer:
        last_tool_loop_stats: dict = {}

        def induce(self, *a, **k):
            raise AssertionError("blind induce must not run in repair mode")

    policy = object.__new__(E3AgentPolicy)
    policy.short = game
    policy.cell = 1
    policy.proposer = _FakeProposer()
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
    record = attempt["recall_resample"]["tool_loop"]
    assert record["tool_gap_events"] == events
    assert record["tool_gap_events_dropped"] == 0
    assert record["candidate_tools_enabled"] == []
    assert record["candidate_tools_rejected"] == []


# ---------------------------------------------------------------------------
# The offline analyzer (SCENARIO-ARC-WMTE-6770-8 / -9)
# ---------------------------------------------------------------------------


def _row(game, seed, events, key="tool_loop_stats", extra=None, dropped=0):
    stats = {
        "tool_calls_by_name": {"diff_grids": 2},
        "tool_call_parse_failures": len(events),
        "tool_gap_events": events,
        "tool_gap_events_dropped": dropped,
    }
    row = {"game": game, "seed": seed, key: stats}
    if extra:
        row.update(extra)
    return row


def _unknown(name, keys=("t",)):
    return {"kind": "unknown_tool", "requested_tool": name, "argument_keys": list(keys)}


def _run_cli(tmp_path, *inputs):
    ledger = tmp_path / "ledger.json"
    rc = gap_mod.main([*map(str, inputs), "--ledger", str(ledger)])
    return rc, ledger


def test_analyzer_specifies_above_floor_and_renders_markdown(tmp_path, capsys):
    """SCENARIO-ARC-WMTE-6770-8: 3 events across 2 distinct rows crosses the
    floor; the output is a HUMAN specification with a ready-to-append entry
    in the ops/verifier_gaps.md schema convention — and no generated code."""
    f1 = tmp_path / "a" / "rows.json"
    f1.parent.mkdir()
    f1.write_text(
        json.dumps(
            [
                _row("tg01", 1, [_unknown("count_colors"), _unknown("count_colors")]),
                _row("tg01", 2, [_unknown("count_colors")], key="stats"),
            ]
        )
    )
    rc, ledger_path = _run_cli(tmp_path, f1)
    assert rc == 0
    out = capsys.readouterr().out
    ledger = json.loads(ledger_path.read_text())
    spec = ledger["specification"]
    assert spec["status"] == "specification_available"
    assert spec["recommendation_only"] is True
    (item,) = spec["specifications"]
    assert item["audience"] == "human"
    assert item["gap"] == {
        "kind": "unknown_tool",
        "name": "count_colors",
        "events": 3,
        "distinct_rows": 2,
    }
    assert "### TOOLGAP-UNKNOWN_TOOL-count_colors" in item["markdown_entry"]
    assert "- status: open" in item["markdown_entry"]
    assert "register_candidate_tool" in item["markdown_entry"]
    # The entry carries the exact evidence, and the analyzer's own report
    # names the gap. (An earlier cut asserted `"def " not in entry` here — a
    # tautology no input could violate; replaced per the 2026-08-29 review.)
    assert "3 events across 2 distinct run rows" in item["markdown_entry"]
    assert "SPECIFY unknown_tool:count_colors" in out


def test_analyzer_floors_are_exact(tmp_path):
    """SCENARIO-ARC-WMTE-6770-8 (floor half): 3 events in ONE row, or 2 events
    across two rows, are both insufficient — said loudly, not ranked."""
    one_row = tmp_path / "one" / "rows.json"
    one_row.parent.mkdir()
    one_row.write_text(json.dumps([_row("tg01", 1, [_unknown("x"), _unknown("x"), _unknown("x")])]))
    rc, ledger_path = _run_cli(tmp_path / "one", one_row)
    assert rc == 0
    spec = json.loads(ledger_path.read_text())["specification"]
    assert spec["status"] == "insufficient_evidence"
    assert spec["specifications"] == []
    assert spec["per_gap"][0]["rows_shortfall"] == 1

    two_rows = tmp_path / "two" / "rows.json"
    two_rows.parent.mkdir()
    two_rows.write_text(
        json.dumps([_row("tg01", 1, [_unknown("x")]), _row("tg01", 2, [_unknown("x")])])
    )
    rc, ledger_path = _run_cli(tmp_path / "two", two_rows)
    spec = json.loads(ledger_path.read_text())["specification"]
    assert spec["status"] == "insufficient_evidence"
    assert spec["per_gap"][0]["events_shortfall"] == 1


def test_analyzer_empty_and_precapture_statuses(tmp_path):
    """SCENARIO-ARC-WMTE-6770-8 (status half): pre-capture rows are absence of
    evidence and say so; capture rows with zero events are the honest empty."""
    rc, ledger_path = _run_cli(tmp_path / "e0")
    assert rc == 0
    assert json.loads(ledger_path.read_text())["specification"]["status"] == "no_rows_ingested"

    pre = tmp_path / "pre" / "rows.json"
    pre.parent.mkdir()
    pre.write_text(
        json.dumps([{"game": "tg01", "tool_loop_stats": {"tool_calls_by_name": {"x": 1}}}])
    )
    rc, ledger_path = _run_cli(tmp_path / "pre", pre)
    spec = json.loads(ledger_path.read_text())["specification"]
    assert spec["status"] == "no_capture_capable_rows"
    assert spec["ingest_counts"]["pre_capture_rows"] == 1

    clean = tmp_path / "clean" / "rows.json"
    clean.parent.mkdir()
    clean.write_text(json.dumps([_row("tg01", 1, [])]))
    rc, ledger_path = _run_cli(tmp_path / "clean", clean)
    spec = json.loads(ledger_path.read_text())["specification"]
    assert spec["status"] == "no_gap_events_nothing_to_specify"


def test_analyzer_dedupes_prunes_clones_and_fails_loud(tmp_path):
    """SCENARIO-ARC-WMTE-6770-9: a byte-identical row ingests once; a nested
    repo clone's rows.json is pruned; a missing input exits non-zero."""
    scan = tmp_path / "scan"
    good = scan / "run1"
    good.mkdir(parents=True)
    (good / "rows.json").write_text(json.dumps([_row("tg01", 1, [_unknown("x")])]))
    clone = scan / "clone"
    (clone / ".git").mkdir(parents=True)
    (clone / "rows.json").write_text(json.dumps([_row("tg01", 9, [_unknown("cloned")])]))

    ledger = tmp_path / "ledger.json"
    assert gap_mod.main([str(scan), "--ledger", str(ledger)]) == 0
    assert gap_mod.main([str(scan), "--ledger", str(ledger)]) == 0  # second pass: all dupes
    data = json.loads(ledger.read_text())
    assert len(data["entries"]) == 1
    assert data["specification"]["ingest_counts"]["capture_rows_duplicate"] == 1
    names = {g["name"] for g in data["specification"]["per_gap"]}
    assert names == {"x"}  # the clone's event never entered

    assert gap_mod.main([str(tmp_path / "missing"), "--ledger", str(ledger)]) == 1


def test_analyzer_survives_source_deletion(tmp_path):
    """SCENARIO-ARC-WMTE-6770-9 (durability half): the ledger stores the
    events, so re-evaluation after the scratch source is gone reproduces the
    same totals."""
    f = tmp_path / "rows.json"
    f.write_text(json.dumps([_row("tg01", 1, [_unknown("x")]), _row("tg01", 2, [_unknown("x")])]))
    ledger = tmp_path / "ledger.json"
    assert gap_mod.main([str(f), "--ledger", str(ledger)]) == 0
    before = json.loads(ledger.read_text())["specification"]["per_gap"]
    f.unlink()
    assert gap_mod.main(["--ledger", str(ledger)]) == 0
    after = json.loads(ledger.read_text())["specification"]["per_gap"]
    assert after == before


def test_candidate_factory_bug_costs_a_turn_not_the_induction(monkeypatch):
    """REQ-ARC-WMTE-6770 rule 3 hardening: dispatch_tool must keep its
    never-raises contract even when a human-authored candidate factory has a
    bug — the error is the tool result, same as a tool-body failure."""
    import os

    schema = json.loads(json.dumps(COUNT_COLORS_SCHEMA))
    schema["function"]["name"] = "broken_tool"

    def _broken_factory(session):
        raise RuntimeError("factory bug")

    tools_mod.register_candidate_tool(schema, _broken_factory)
    monkeypatch.setenv(tools_mod.CANDIDATE_TOOLS_ENV, "broken_tool")
    out = tools_mod.dispatch_tool(_session(), "broken_tool", '{"t": 0}')
    assert out["ok"] is False
    assert "setup raised RuntimeError" in out["error"]


# ---------------------------------------------------------------------------
# Adversarial-review fixes, 2026-08-29 (R/F findings; REQ-ARC-WMTE-6770
# amendment). Each test here is the regression for one review finding.
# ---------------------------------------------------------------------------


def test_unknown_tool_with_malformed_arguments_still_records_the_name():
    """F3: the NAME needs no JSON parse. An unknown name with malformed or
    non-object arguments is still demand; a KNOWN tool with malformed
    arguments is transport noise, not demand."""
    s = _session()
    out = tools_mod.dispatch_tool(s, "warp_grid", "{t: 0}")
    assert out["ok"] is False and "unparseable" in out["error"]
    assert s.tool_gap_events == [
        {"kind": "unknown_tool", "requested_tool": "warp_grid", "argument_keys": None}
    ]
    s2 = _session()
    tools_mod.dispatch_tool(s2, "diff_grids", "{t: 0}")
    assert s2.tool_gap_events == []
    s3 = _session()
    tools_mod.dispatch_tool(s3, "warp_grid", "[0]")
    assert s3.tool_gap_events[0]["requested_tool"] == "warp_grid"


def test_model_controlled_event_strings_are_bounded():
    """F9: the name and argument keys are model-controlled text headed for a
    durable ledger and a human-pasted heading; both are capped like `error`."""
    s = _session()
    big_args = json.dumps({f"k{i:04d}": 1 for i in range(500)})
    tools_mod.dispatch_tool(s, "x" * 5000, big_args)
    event = s.tool_gap_events[0]
    assert len(event["requested_tool"]) == tools_mod.MAX_TOOL_GAP_NAME_CHARS
    assert len(event["argument_keys"]) == tools_mod.MAX_TOOL_GAP_ARG_KEYS


def test_session_freeze_pins_advertised_and_dispatched_set(monkeypatch):
    """F5/F7: enablement is frozen at session creation. A mid-run env or
    registry mutation - including one made by model-executed code - changes
    neither what the session serves nor what its record claims; and a set
    enabled at creation survives a mid-run env clear, so the record cannot
    contradict what actually dispatched."""
    s = _session()  # env unset at creation
    tools_mod.register_candidate_tool(COUNT_COLORS_SCHEMA, _count_colors_factory)
    monkeypatch.setenv(tools_mod.CANDIDATE_TOOLS_ENV, "count_colors")
    out = tools_mod.dispatch_tool(s, "count_colors", '{"t": 0}')
    assert out["ok"] is False and "unknown tool" in out["error"]
    assert tools_mod.active_tool_schemas_for(s) is tools_mod.TOOL_SCHEMAS

    s2 = _session()  # env set at creation
    monkeypatch.delenv(tools_mod.CANDIDATE_TOOLS_ENV)
    assert s2.enabled_candidates == ("count_colors",)
    assert tools_mod.dispatch_tool(s2, "count_colors", '{"t": 0}')["ok"] is True


def test_unknown_tool_error_names_the_session_active_set(monkeypatch):
    """F8 (decorative-assertion fix): the refusal text must list the ACTIVE
    set - reverting it to the bare core names would hide an enabled candidate
    from the model that just failed to call it."""
    tools_mod.register_candidate_tool(COUNT_COLORS_SCHEMA, _count_colors_factory)
    monkeypatch.setenv(tools_mod.CANDIDATE_TOOLS_ENV, "count_colors")
    out = tools_mod.dispatch_tool(_session(), "nope", "{}")
    assert "count_colors" in out["error"]


def test_dark_candidate_does_not_change_xml_coercion():
    """F6: a registered-but-dark candidate must not change coercion - the
    default path stays byte-identical until the env enables the tool."""
    tools_mod.register_candidate_tool(COUNT_COLORS_SCHEMA, _count_colors_factory)
    calls, n_blocks, n_unparsed = tools_mod.parse_xml_tool_calls(
        "<tool_call>\n<function=count_colors>\n<parameter=t>\n3\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert n_blocks == 1 and n_unparsed == 0
    # String passthrough, exactly as before the registry existed.
    assert json.loads(calls[0]["function"]["arguments"]) == {"t": "3"}


def test_loop_finalizes_the_dropped_counter(monkeypatch, proposer, tmp_path):
    """F8: the dropped counter must reach the run's stats THROUGH the loop -
    an earlier cut proved the session attribute only, and replacing the
    finalize with a literal 0 stayed green."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    n = tools_mod.MAX_TOOL_GAP_EVENTS + 5
    calls = [_tool_call(f"ph_{i}", "{}", f"c{i}") for i in range(n)]
    fake = _FakeHTTP(
        chat_replies=[
            _chat_reply({"role": "assistant", "tool_calls": calls}, tokens=60),
            _chat_reply(
                {
                    "role": "assistant",
                    "tool_calls": [
                        _tool_call(
                            "run_engine_on_transitions", json.dumps({"code": GOOD_CODE}), "cz"
                        )
                    ],
                },
                tokens=60,
            ),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("dropgame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert len(stats["tool_gap_events"]) == tools_mod.MAX_TOOL_GAP_EVENTS
    assert stats["tool_gap_events_dropped"] == 5


def test_selfparse_unparsed_block_keeps_the_demanded_name(monkeypatch, proposer, tmp_path):
    """F4 (selfparse half): a block the strict parser refuses still holds the
    demanded name; the loop keeps it as gap evidence with its source marked."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "selfparse")
    malformed = (
        'thinking</think>\n<tool_call>\n<function="fetch_layers">\n'
        "<parameter=t>\n0\n</parameter>\n</function>\n</tool_call>"
    )
    fake = _FakeHTTP(
        chat_replies=[
            _chat_reply({"role": "assistant", "content": malformed}, tokens=60),
            _chat_reply({"role": "assistant", "content": f"```python\n{GOOD_CODE}```"}, tokens=60),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("loosegame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["selfparse_blocks_unparsed"] == 1
    assert stats["tool_gap_events"] == [
        {
            "kind": "unknown_tool",
            "requested_tool": "fetch_layers",
            "argument_keys": None,
            "source": "unparsed_text",
        }
    ]


def test_server_lift_failure_text_keeps_the_demanded_name(monkeypatch, proposer, tmp_path):
    """F4 (server-lifted half): tool-call JSON left as text still names the
    demanded tool; a known-tool name in the same text is transport noise."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    text = '{"name": "fetch_layers", "arguments": {"t": 0}}'
    fake = _FakeHTTP(
        chat_replies=[
            _chat_reply({"role": "assistant", "content": text}, tokens=60),
            _chat_reply({"role": "assistant", "content": f"```python\n{GOOD_CODE}```"}, tokens=60),
        ]
    )
    _patch_http(monkeypatch, fake)
    ok, _ = proposer.induce("liftgame", _mover_window(), 1)
    assert ok
    stats = proposer.last_tool_loop_stats
    assert stats["unparsed_tool_call_text_turns"] == 1
    assert stats["tool_gap_events"] == [
        {
            "kind": "unknown_tool",
            "requested_tool": "fetch_layers",
            "argument_keys": None,
            "source": "unparsed_text",
        }
    ]


def test_primary_live_induce_records_tool_gap_evidence(monkeypatch):
    """F2 - the finding that mattered most: the PRIMARY live induction (the
    only branch that runs under the live '1'/'selfparse' transports) must
    copy gap evidence onto the attempt record, and must never copy a STALE
    run's stats."""
    from types import SimpleNamespace

    from carnot.agentic import arc_competition_agent as agent

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")
    events = [{"kind": "unknown_tool", "requested_tool": "get_full_grid", "argument_keys": None}]
    prop = SimpleNamespace(model_specs="stub")

    def _induce(*_a, **_k):
        prop.last_tool_loop_stats = {
            "tool_gap_events": list(events),
            "tool_gap_events_dropped": 0,
            "candidate_tools_enabled": [],
            "candidate_tools_rejected": [],
            "terminated_by": "turn_cap",
            "tool_calls_total": 3,
        }
        return (False, "no engine")

    prop.induce = _induce
    policy = agent.E3AgentPolicy("gaplive", proposer=prop, value_head=lambda _f: 0.0)
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"
    policy._induce_and_plan()
    attempt = policy.induction_attempts[-1]
    assert attempt["tool_gap"]["tool_gap_events"] == events
    assert attempt["tool_gap"]["terminated_by"] == "turn_cap"

    # Stale-stats guard: the site clears the diagnostic BEFORE inducing, so a
    # loop-less induction must not inherit the previous run's events.
    prop.induce = lambda *_a, **_k: (False, "no loop this time")
    policy._pending_induction_reason = "stall"
    policy._induce_and_plan()
    assert "tool_gap" not in policy.induction_attempts[-1]


def test_analyzer_writes_only_its_ledger_and_reports_dropped(tmp_path, monkeypatch):
    """Spec rule 5, previously asserted by nobody: the analyzer writes its
    ledger and NOTHING else; and dropped-at-capture counts surface in the
    evidence so per-gap counts read as floors (F8)."""
    monkeypatch.chdir(tmp_path)
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([_row("tg01", 1, [_unknown("x")], dropped=3)]))
    ledger = tmp_path / "led" / "ledger.json"
    before = {p for p in tmp_path.rglob("*")}
    assert gap_mod.main([str(rows), "--ledger", str(ledger)]) == 0
    after = {p for p in tmp_path.rglob("*")}
    assert after - before == {ledger.parent, ledger}
    spec = json.loads(ledger.read_text())["specification"]
    assert spec["evidence"]["gap_events_dropped_total"] == 3
