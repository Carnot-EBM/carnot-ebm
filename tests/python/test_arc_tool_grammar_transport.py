"""Observed serving shapes reach live dispatch.

Spec: REQ-ARC-WMTE-7043, REQ-ARC-WMTE-7044, REQ-ARC-WMTE-7045.
"""

from __future__ import annotations

import io
import json
import urllib.request

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_induction_tool_loop as loop
from carnot.agentic.arc_induction_tools import TOOL_SCHEMAS

# Policy construction populates large third-party import caches. Tests still run;
# this marker excludes that known cache growth from the separate leak watchdog.
pytestmark = pytest.mark.memory_watchdog_skip

CODE = """import numpy as np
def engine(grid, action, data):
    return grid + 1
def is_level_complete(grid):
    return False
"""


def rows():
    return [
        e3.Transition(np.full((2, 2), i), 1, None, np.full((2, 2), i + 1), 0, 0) for i in range(8)
    ]


def reply(content, finish="stop"):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": {"completion_tokens": 20, "prompt_tokens": 100},
    }


@pytest.fixture
def transport(monkeypatch, tmp_path):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path / "engines")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_GRAMMAR", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_TURNS", "2")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_SEED", "42")
    monkeypatch.setenv("CARNOT_ARC_LLM_BACKEND", "llamacpp")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", raising=False)
    p = e3.LocalGGUFProposer(ffn_cpu_layers=0, mtp=False, max_tokens=1024, tries=1)
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    sent = []
    answers = [
        reply(json.dumps({"name": "diff_grids", "arguments": {"t": 0}})),
        reply(json.dumps({"name": "run_engine_on_transitions", "arguments": {"code": CODE}})),
    ]

    def request(req, timeout=None):
        sent.append(json.loads(req.data))
        response = (
            (answers.pop(0) if answers else reply(""))
            if req.full_url.endswith("/chat/completions")
            else {"content": "```python\n" + CODE + "```", "stop_type": "eos"}
        )
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    return p, sent, answers


@pytest.mark.parametrize("mode", ["1", "selfparse"])
def test_live_induce_dispatches_and_delivers_feedback(monkeypatch, transport, tmp_path, mode):
    """REQ-7044/7045: inspect actual HTTP and the real published engine."""
    p, sent, _ = transport
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", mode)
    ok, note = p.induce("grammar", rows(), 1)
    assert ok, note
    assert len(sent) == 2
    for payload in sent:
        assert payload.get("grammar", "").startswith("root ::=")
        # REQ-ARC-WMTE-7046: one call rule per tool, and run_engine_on_transitions
        # (schema index 0) must carry a code string that defines engine.
        assert payload["grammar"].splitlines()[0].startswith("root ::= call-0 | call-1")
        assert (
            'args-0 ::= "{" ws "\\"code\\"" ws ":" ws "\\"" char* "def engine(" char* "\\"" ws'
            in payload["grammar"]
        )
        assert payload.get("grammar_lazy") is False
        assert payload.get("chat_template_kwargs") == {"enable_thinking": False}
        assert payload.get("thinking_budget_tokens") == 0
        assert "tools" not in payload and "tool_choice" not in payload
        assert payload["max_tokens"] == 1024
        assert all(s["function"]["name"] in payload["grammar"] for s in TOOL_SCHEMAS)
    assert sent[1]["seed"] == sent[0]["seed"] + 1
    messages = sent[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert json.loads(messages[1]["content"]) == {"name": "diff_grids", "arguments": {"t": 0}}
    assert '"before": 0' in messages[2]["content"]
    assert '"after": 1' in messages[2]["content"]
    assert "JSON" in messages[0]["content"] and '"name"' in messages[0]["content"]
    assert "Return exactly one JSON" in messages[0]["content"]
    assert "reply with ONLY one final" not in messages[0]["content"]
    written = tmp_path / "engines" / "grammar" / "world_model.py"
    assert written.exists()
    assert written.read_text().strip() == CODE.strip()
    stats = p.last_tool_loop_stats
    assert stats.get("grammar_json") is True
    assert stats.get("grammar_calls_parsed") == 2
    assert stats["tool_calls_by_name"] == {"diff_grids": 1, "run_engine_on_transitions": 1}
    assert stats["terminated_by"] == "zero_mismatches"


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_default_payload_preserved(monkeypatch, transport, flag):
    """REQ-7044: grammar is an exact opt-in at the request call site."""
    p, sent, answers = transport
    if flag is None:
        monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_GRAMMAR")
    else:
        monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_GRAMMAR", flag)
    answers[:] = [reply("```python\n" + CODE + "```")]
    assert p.induce("grammar", rows(), 1)[0]
    assert len(sent) == 1
    assert "grammar" not in sent[0] and "grammar_lazy" not in sent[0]
    assert sent[0]["tools"] == TOOL_SCHEMAS
    assert sent[0]["tool_choice"] == "auto"
    assert sent[0]["thinking_budget_tokens"] == loop.DEFAULT_THINK_BUDGET
    assert "Return exactly one JSON" not in sent[0]["messages"][0]["content"]


def test_flag_alone_does_not_enable_tools(monkeypatch, transport):
    """REQ-7044: the existing master switch still controls induction."""
    p, sent, _ = transport
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP")
    assert p.induce("grammar", rows(), 1)[0]
    assert sent and all("messages" not in s and "grammar" not in s for s in sent)


@pytest.mark.parametrize(
    "content,finish",
    [
        ('{"name":"diff_grids","arguments":{"t":0}}', "length"),
        ('{"name":"diff_grids","arguments":{"t":0}} trailing', "stop"),
        ('{"name":"missing","arguments":{}}', "stop"),
        ('{"name":"diff_grids","arguments":"{}"}', "stop"),
        ('{"name":"diff_grids","arguments":{"t":NaN}}', "stop"),
        ('{"name":"diff_grids","arguments":{"t":Infinity}}', "stop"),
        ('{"name":"diff_grids","arguments":[],"extra":1}', "stop"),
        ('{"name":"diff_grids","arguments":{},"extra":1}', "stop"),
        ('{"name":[],"arguments":{}}', "stop"),
        ("null", "stop"),
        ("[]", "stop"),
        (
            "<tool_call><function=diff_grids><parameter=t>0</parameter></function></tool_call>",
            "stop",
        ),
    ],
)
def test_invalid_envelope_never_dispatches(monkeypatch, transport, content, finish):
    """REQ-7045: transport failures cannot fall through to XML or tool dispatch."""
    p, sent, answers = transport
    answers[:] = [reply(content, finish)]
    dispatches = []
    dispatch = loop.dispatch_tool
    monkeypatch.setattr(
        loop, "dispatch_tool", lambda *a, **kw: (dispatches.append(a), dispatch(*a, **kw))[1]
    )
    ok, _ = loop.induce_with_tool_loop(p, "grammar", rows(), 1)
    assert not ok
    assert len(sent) == 1
    assert dispatches == []
    assert p.last_tool_loop_stats["terminated_by"] == "grammar_invalid_response"
    assert "grammar response" in p.last_tool_loop_stats.get("grammar_error", "")
    assert p.last_tool_loop_stats.get("grammar_invalid_responses") == 1
    assert p.last_tool_loop_stats["decode_tokens_total"] == 20
    assert p.last_tool_loop_stats["turns"] == 1


def test_invalid_response_uses_existing_fallback(transport):
    """REQ-7045: failed constrained turns do not disable ordinary induction."""
    p, sent, answers = transport
    answers[:] = [reply("not JSON")]
    assert p.induce("grammar", rows(), 1)[0]
    assert len(sent) == 2 and "grammar" in sent[0] and "prompt" in sent[1]
    assert p.last_tool_loop_stats["terminated_by"] == "grammar_invalid_response"


def test_valid_json_with_bad_arguments_returns_observed_error(transport):
    """REQ-7044/7045: syntax does not replace the dispatcher's argument checks.

    The grammar admits extra keys after the required ones (REQ-7046), so an unknown
    keyword is the grammar-valid shape that still has to fail at dispatch."""
    p, sent, answers = transport
    answers[0] = reply('{"name":"diff_grids","arguments":{"t":0,"bogus":1}}')
    assert p.induce("grammar", rows(), 1)[0]
    assert "unexpected keyword argument 'bogus'" in sent[1]["messages"][2]["content"]
    assert p.last_tool_loop_stats["grammar_calls_parsed"] == 2
    assert p.last_tool_loop_stats["candidates_scored"] == 1
    assert p.last_tool_loop_stats["tool_gap_events"][0]["kind"] == "bad_arguments"


@pytest.mark.parametrize(
    "content",
    [
        '{"name":"run_engine_on_transitions","arguments":{}}',
        '{"name":"run_engine_on_transitions","arguments":{"code":""}}',
        '{"name":"run_engine_on_transitions","arguments":{"code":" "}}',
        '{"name":"run_engine_on_transitions","arguments":{"code":"x"}}',
        '{"name":"run_engine_on_transitions","arguments":{"code":"def is_level_complete(g): 0"}}',
        '{"name":"run_goal_on_states","arguments":{"code":"def engine(g, a, d): return g"}}',
        '{"name":"query_region","arguments":{"t":0}}',
        (
            '{"name":"find_objects","arguments":{"t":0,"which":" ",'
            '"predicate_code":"def accept(obj): return True","max_objects":5}}'
        ),
    ],
)
def test_payload_less_envelope_is_a_grammar_failure(monkeypatch, transport, content):
    """SCENARIO-ARC-WMTE-7046-B: a missing, blank, or definition-less required argument
    never dispatches.

    The first of these is exactly what the 0.8B trial returned twice and counted as
    two parsed calls. The one-space and one-character forms are the review's finding 2:
    "non-empty" alone delivered no program."""
    p, sent, answers = transport
    answers[:] = [reply(content)]
    dispatches = []
    dispatch = loop.dispatch_tool
    monkeypatch.setattr(
        loop, "dispatch_tool", lambda *a, **kw: (dispatches.append(a), dispatch(*a, **kw))[1]
    )
    ok, _ = loop.induce_with_tool_loop(p, "grammar", rows(), 1)
    assert not ok
    assert dispatches == []
    stats = p.last_tool_loop_stats
    assert stats["terminated_by"] == "grammar_invalid_response"
    err = stats["grammar_error"]
    assert "missing required argument" in err or "does not define" in err
    assert stats["grammar_calls_parsed"] == 0
    assert stats["grammar_invalid_responses"] == 1
    assert stats["tool_calls_total"] == 0


def test_request_grammar_rejects_empty_shell_model_free(transport):
    """SCENARIO-ARC-WMTE-7046-A: the grammar the live request carries, read by the
    model-free reader, refuses the empty shell and accepts a full call."""
    from carnot.testing.gbnf_match import accepts

    p, sent, _ = transport
    assert p.induce("grammar", rows(), 1)[0]
    grammar = sent[0]["grammar"]
    full = json.dumps(
        {"name": "run_engine_on_transitions", "arguments": {"code": CODE}},
        separators=(",", ":"),
    )
    assert not accepts(grammar, '{"name":"run_engine_on_transitions","arguments":{}}')
    assert not accepts(grammar, '{"name":"run_engine_on_transitions","arguments":{"code":""}}')
    assert not accepts(grammar, '{"name":"run_engine_on_transitions","arguments":{"code":" "}}')
    assert not accepts(grammar, '{"name":"run_engine_on_transitions","arguments":{"code":"x"}}')
    assert accepts(
        grammar, '{"name":"run_engine_on_transitions","arguments":{"code":"def engine("}}'
    )
    assert accepts(grammar, full)
    # list_transitions takes no arguments: `{}` is its complete call, not an empty shell.
    assert accepts(grammar, '{"name":"list_transitions","arguments":{}}')
    assert not accepts(grammar, '{"name":"diff_grids","arguments":{}}')
    assert '"arguments": {}' not in sent[0]["messages"][0]["content"]
    assert "required parameter" in sent[0]["messages"][0]["content"]


def test_force_turn_sends_submission_only_grammar(monkeypatch, transport):
    """SCENARIO-ARC-WMTE-7046-C: after the inspection budget the grammar itself admits
    only a run_engine_on_transitions submission, so the argument-less tools cannot be
    chosen forever and the prompt nudge is no longer the only enforcement."""
    from carnot.testing.gbnf_match import accepts

    p, sent, answers = transport
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", "1")
    answers[:] = [
        reply('{"name":"list_transitions","arguments":{}}'),
        reply(json.dumps({"name": "run_engine_on_transitions", "arguments": {"code": CODE}})),
    ]
    ok, note = loop.induce_with_tool_loop(p, "grammar", rows(), 1)
    assert ok, note
    assert len(sent) == 2
    first, second = sent[0]["grammar"], sent[1]["grammar"]
    assert first.splitlines()[0].startswith("root ::= call-0 | call-1")
    assert second.splitlines()[0] == "root ::= call-0"
    assert accepts(first, '{"name":"list_transitions","arguments":{}}')
    assert not accepts(second, '{"name":"list_transitions","arguments":{}}')
    assert not accepts(second, '{"name":"diff_grids","arguments":{"t":0}}')
    assert accepts(
        second,
        json.dumps(
            {"name": "run_engine_on_transitions", "arguments": {"code": CODE}},
            separators=(",", ":"),
        ),
    )
    assert sent[1]["messages"][-1]["content"] == loop._FORCE_ENGINE_NUDGE
    stats = p.last_tool_loop_stats
    assert stats["force_engine_nudges"] == 1
    assert stats["grammar_submit_only_turns"] == 1
    assert stats["terminated_by"] == "zero_mismatches"


@pytest.mark.parametrize(
    "choices", [None, [], [None], [{"message": None}], [{"message": []}], "invalid"]
)
def test_invalid_outer_response_is_rejected(transport, choices):
    """REQ-7045: malformed server shapes still record spent tokens and fall back."""
    p, sent, answers = transport
    answers[:] = [{"choices": choices, "usage": {"completion_tokens": 20}}]
    try:
        outcome = p.induce("grammar", rows(), 1)
    except Exception as exc:
        outcome = exc
    assert not isinstance(outcome, Exception), f"fallback raised instead of returning: {outcome!r}"
    assert outcome[0]
    assert len(sent) == 2 and "prompt" in sent[1]
    assert p.last_tool_loop_stats["terminated_by"] == "grammar_invalid_response"
    assert p.last_tool_loop_stats["decode_tokens_total"] == 20


def test_compaction_combination_is_explicitly_rejected(monkeypatch, transport):
    """REQ-7044: unsupported message compaction cannot silently claim success."""
    p, sent, _ = transport
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_COMPACT", "1")
    assert p.induce("grammar", rows(), 1)[0]
    assert len(sent) == 1 and "prompt" in sent[0]
    assert p.last_tool_loop_stats["terminated_by"] == "grammar_compaction_unsupported"
    assert p.last_tool_loop_stats["compactions"] == 0


def test_vllm_rejected_before_http(monkeypatch, transport):
    """REQ-7044: the unconfirmed backend gets no GBNF request."""
    p, sent, _ = transport
    monkeypatch.setattr(e3, "_vllm_backend_active", lambda: True)
    ok, _ = loop.induce_with_tool_loop(p, "grammar", rows(), 1)
    assert not ok and sent == []
    assert "llama.cpp" in p.last_tool_loop_stats.get("transport_error", "")


def test_candidate_names_and_json_are_preserved(monkeypatch, transport):
    """REQ-7044/7045: session names freeze; literal tags and nested values survive."""
    from carnot.agentic import arc_induction_tools as tool_module

    p, sent, answers = transport
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "selfparse")
    monkeypatch.setattr(tool_module, "CANDIDATE_TOOLS", {})
    schema = {
        "type": "function",
        "function": {
            "name": "echo_probe",
            "description": "Return the observed JSON arguments unchanged.",
            "parameters": {"type": "object"},
        },
    }
    received = []

    def factory(session):
        def echo(**kwargs):
            received.append(kwargs)
            schema["function"]["name"] = "changed_after_request"
            return {"observed": kwargs}

        return echo

    tool_module.register_candidate_tool(schema, factory)
    monkeypatch.setenv(tool_module.CANDIDATE_TOOLS_ENV, "echo_probe")
    arguments = {"literal": '</think>\\"\nλ', "nested": [None, True, False, -1.25, {"x": 3}]}
    envelope = json.dumps({"name": "echo_probe", "arguments": arguments})
    answers[0] = reply(envelope)
    ok, note = p.induce("grammar", rows(), 1)
    assert ok, note
    assert received == [arguments]
    assert sent[1]["messages"][1]["content"] == envelope
    assert '"observed"' in sent[1]["messages"][2]["content"]
    for payload in sent:
        assert "echo_probe" in payload["grammar"]
        assert "changed_after_request" not in payload["grammar"]
    assert p.last_tool_loop_stats["candidate_tools_enabled"] == ["echo_probe"]
    assert p.last_tool_loop_stats["selfparse"] is False


def test_offline_twin_reaches_grammar(monkeypatch, transport, tmp_path):
    """REQ-7045: CLI construction reaches the same policy and transport."""
    from scripts import arc_leaderboard_eval as evaluate
    from scripts import arc_loop_solve as twin
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    p, sent, _ = transport
    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")
    monkeypatch.setenv("CARNOT_ARC_STRUCTURED_NAV", "0")
    monkeypatch.setenv("CARNOT_ARC_LIVE_TTT", "0")
    monkeypatch.delenv("CARNOT_ARC_DISABLE_INDUCTION", raising=False)

    def run(game, policy, *, budget):
        assert isinstance(policy, E3AgentPolicy) and game == "grammar" and budget == 7
        policy.proposer = p
        policy.transitions = rows()
        policy.root_grid = policy.transitions[0].grid
        policy.cell = 1
        policy._episode_transition_start = 0
        policy.program_synthesis_filter_enabled = False
        policy.active_probe_controller_enabled = False
        policy.think_arm_fallback_enabled = False
        policy._induce_and_plan()
        return {"grammar": p.last_tool_loop_stats}

    monkeypatch.setattr(evaluate, "run_game", run)
    output = tmp_path / "twin.json"
    status = twin.main(
        ["--game", "grammar", "--mechanism", "e3", "--max-actions", "7", "--output", str(output)]
    )
    assert status == 0
    assert sent and "grammar" in sent[0]
    assert json.loads(output.read_text())["grammar"]["grammar_calls_parsed"] == 2


@pytest.mark.parametrize("bounded", [False, True])
def test_scored_policy_reaches_grammar(monkeypatch, transport, bounded):
    """REQ-7045: the scored policy reaches HTTP, dispatch and publication."""
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic import arc_llm_reinduction as reinduce

    p, sent, _ = transport
    dispatched = []
    dispatch = loop.dispatch_tool

    def recording(session, name, *args, **kwargs):
        dispatched.append(name)
        return dispatch(session, name, *args, **kwargs)

    monkeypatch.setattr(loop, "dispatch_tool", recording)
    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "1" if bounded else "0")
    monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
    monkeypatch.setenv("CARNOT_ARC_STRUCTURED_NAV", "0")
    monkeypatch.setenv("CARNOT_ARC_LIVE_TTT", "0")
    monkeypatch.delenv("CARNOT_ARC_DISABLE_INDUCTION", raising=False)
    monkeypatch.setattr(reinduce, "MAX_REFINEMENT_ROUNDS", 1)
    policy = E3AgentPolicy("grammar", proposer=p, value_head=None)
    policy.transitions = rows()
    policy.root_grid = policy.transitions[0].grid
    policy.cell = 1
    policy._episode_transition_start = 0
    policy.program_synthesis_filter_enabled = False
    policy.active_probe_controller_enabled = False
    policy.think_arm_fallback_enabled = False
    policy.max_refinement_rounds = 1
    policy._induce_and_plan()
    assert sent and "grammar" in sent[0], policy.induction_attempts
    assert dispatched[:2] == ["diff_grids", "run_engine_on_transitions"]
    assert '"after": 1' in sent[1]["messages"][2]["content"]
    attempt = policy.induction_attempts[-1]
    stats = (
        attempt["refinement_rounds"][0].get("tool_loop", {})
        if bounded
        else attempt.get("grammar_transport", {})
    )
    assert stats.get("grammar_json") is True
    assert stats.get("grammar_calls_parsed") == 2


def test_repair_policy_records_real_grammar_dispatch(monkeypatch, transport):
    """REQ-7045: repair receipts retain counters from actual HTTP and dispatch."""
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    p, sent, _ = transport
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "repair")
    policy = object.__new__(E3AgentPolicy)
    policy.short, policy.cell, policy.proposer = "grammar", 1, p
    old = CODE.replace("return grid + 1", "return grid")
    path = e3.E3_DIR / "grammar" / "world_model.py"
    path.parent.mkdir(parents=True)
    path.write_text(old)
    engine, goal = e3.load_engine("grammar")
    verdict = e3.WorldModelVerifier(rows()).score(engine)
    assert verdict.cell_recall == 0
    attempt = {}
    _, _, kept = policy._maybe_recall_gated_resample(
        attempt=attempt,
        transitions=rows(),
        hud_mask=None,
        engine=engine,
        is_done=goal,
        vr=verdict,
        induce_rows=rows(),
        induce_kwargs={},
    )
    stats = attempt["recall_resample"]["tool_loop"]
    assert stats.get("grammar_json") is True
    assert stats.get("grammar_calls_parsed") == 2
    assert len(sent) == 2 and "grammar" in sent[0]
    assert kept.cell_recall == 1


def test_bounded_round_does_not_reuse_stale_diagnostics(monkeypatch, transport):
    """REQ-7045: earlier grammar success is not evidence of this round's transport."""
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    p, sent, _ = transport
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP")
    p.last_tool_loop_stats = {"grammar_json": True, "grammar_calls_parsed": 999}
    outcome = execute_bounded_llm_reinduction(
        game="grammar",
        transitions=rows(),
        cell=1,
        root_grid=rows()[0].grid,
        proposer=p,
        max_rounds=1,
        candidate_provider=lambda engine, goal: [],
        load_engine=e3.load_engine,
        plan_in_model=e3.plan_in_model,
    )
    assert sent and "prompt" in sent[0]
    assert "tool_loop" not in outcome.rounds[0]


@pytest.mark.parametrize("route", ["repair", "refactor"])
@pytest.mark.parametrize("failure", ["server_false", "server_raise", "evidence_raise"])
def test_early_failure_receipts_are_fresh(monkeypatch, transport, route, failure):
    """REQ-ARC-WMTE-7045: a dead server cannot reuse prior successful call counts."""
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_llm_reinduction import _tool_loop_refactor

    p, sent, _ = transport
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "repair")
    prior = {"grammar_json": True, "grammar_calls_parsed": 999, "terminated_by": "zero_mismatches"}
    p.last_tool_loop_stats = prior

    def fail(*args, **kwargs):
        raise RuntimeError("injected initialization failure")

    if failure == "evidence_raise":
        monkeypatch.setattr(p, "_begin_engine_evidence", fail)
    else:
        monkeypatch.setattr(
            p, "_ensure_server", fail if failure == "server_raise" else lambda: False
        )
    path = e3.E3_DIR / "grammar" / "world_model.py"
    path.parent.mkdir(parents=True)
    path.write_text(CODE.replace("return grid + 1", "return grid"))
    if route == "refactor":
        outcome = _tool_loop_refactor(p, "grammar", rows(), 1)
        assert outcome is not None and outcome[0] is False
        stats = outcome[2]
    else:
        policy = object.__new__(E3AgentPolicy)
        policy.short, policy.cell, policy.proposer = "grammar", 1, p
        engine, goal = e3.load_engine("grammar")
        verdict = e3.WorldModelVerifier(rows()).score(engine)
        attempt = {}
        policy._maybe_recall_gated_resample(
            attempt=attempt,
            transitions=rows(),
            hud_mask=None,
            engine=engine,
            is_done=goal,
            vr=verdict,
            induce_rows=rows(),
            induce_kwargs={},
        )
        stats = attempt["recall_resample"]["tool_loop"]
    assert sent == []
    assert p.last_tool_loop_stats is not prior
    assert stats.get("grammar_json") is True
    assert stats.get("grammar_calls_parsed") == 0
    assert stats.get("decode_tokens_total") == 0
    assert stats.get("terminated_by") == "initialization_failed"
