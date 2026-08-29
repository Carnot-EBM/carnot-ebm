"""Selfparse tool-call transport (REQ-ARC-WMTE-6730) + goal-prompt RLE render (REQ-ARC-WMTE-6740).

WHY THIS EXISTS. The scored vLLM server is launched with no tool-parser flags: a
request carrying a `tools` field returns HTTP 400, and with `--tool-call-parser
hermes` the model's calls stay unlifted text, because Qwen3.8 emits the Qwen3-coder
XML convention (measured: offplay_out5 tool_transport_probe, 2026-08-20). Selfparse
removes the server dependency: schemas travel as prompt text, no `tools` field is
sent, and the loop parses the XML itself. These tests pin every rule of that
transport, each one mutation-proved (deleting the rule turns a test here red).

Spec: REQ-ARC-WMTE-6730 SCENARIO-ARC-WMTE-6731..6736, REQ-ARC-WMTE-6740
SCENARIO-ARC-WMTE-6741.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from carnot.agentic.arc_induction_tools import (
    TOOL_NAMES,
    parse_xml_tool_calls,
    render_tool_schemas_for_prompt,
)

# The verbatim emission captured from the scored-shape vLLM probe (offplay_out5,
# content_head). This is the exact string the server-side hermes parser failed to
# lift; the agent-side parser must lift it.
_CAPTURED_EMISSION = (
    "We need answer user's request: call run_engine_on_transitions with code "
    "`def engine(grid, action, data): return grid`. Use tool. Do not answer in "
    "prose. Need just tool call.\n</think>\n\n"
    "<tool_call>\n"
    "<function=run_engine_on_transitions>\n"
    "<parameter=code>\n"
    "def engine(grid, action, data): return grid\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


# --------------------------------------------------------------------------- #
# The parser (SCENARIO-ARC-WMTE-6731: the captured cross-backend emission).    #
# --------------------------------------------------------------------------- #


def test_parses_the_captured_vllm_emission_verbatim() -> None:
    """The stored emission the hermes server parser could not lift must parse to one
    dispatchable call with the exact code payload, newline-trimmed."""
    calls, seen, unparsed = parse_xml_tool_calls(_CAPTURED_EMISSION)
    assert seen == 1 and unparsed == 0 and len(calls) == 1
    assert calls[0]["function"]["name"] == "run_engine_on_transitions"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"code": "def engine(grid, action, data): return grid"}


def test_only_text_after_think_close_is_scanned() -> None:
    """SCENARIO-ARC-WMTE-6732: a call the model merely sketched while reasoning is
    not a call it made -- only text after the last </think> is parsed."""
    sketched = (
        "I could call <tool_call>\n<function=diff_grids>\n<parameter=t>\n0\n"
        "</parameter>\n</function>\n</tool_call> here...\n</think>\nNo call after all."
    )
    calls, seen, unparsed = parse_xml_tool_calls(sketched)
    assert calls == [] and seen == 0 and unparsed == 0


def test_integer_params_coerced_from_the_schema_types() -> None:
    """SCENARIO-ARC-WMTE-6733: the XML carries raw text; dispatch expects JSON types.
    Types come from TOOL_SCHEMAS so prompt, parser, and dispatch cannot drift."""
    xml = (
        "<tool_call>\n<function=query_region>\n"
        "<parameter=t>\n2\n</parameter>\n"
        "<parameter=r0>\n0\n</parameter>\n"
        "<parameter=r1>\n4\n</parameter>\n"
        "<parameter=c0>\n1\n</parameter>\n"
        "<parameter=c1>\n5\n</parameter>\n"
        "<parameter=which>\nafter\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls, seen, unparsed = parse_xml_tool_calls(xml)
    assert seen == 1 and unparsed == 0
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"t": 2, "r0": 0, "r1": 4, "c0": 1, "c1": 5, "which": "after"}


def test_code_value_keeps_inner_indentation_drops_only_wrapping_newlines() -> None:
    """A full strip() would corrupt an indented first code line; only the single
    wrapping newline on each side is the convention's, not the payload's."""
    xml = (
        "<tool_call>\n<function=run_goal_on_states>\n<parameter=code>\n"
        "def is_level_complete(grid):\n    return False\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    calls, _, _ = parse_xml_tool_calls(xml)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["code"] == "def is_level_complete(grid):\n    return False"


def test_unterminated_trailing_block_counts_as_unparsed_never_dispatched() -> None:
    """SCENARIO-ARC-WMTE-6734: a length-truncated call must be counted (the parse-rate
    stat's denominator) but never dispatched with half a payload."""
    truncated = (
        "</think>\n<tool_call>\n<function=run_engine_on_transitions>\n"
        "<parameter=code>\ndef engine(grid, ac"
    )
    calls, seen, unparsed = parse_xml_tool_calls(truncated)
    assert calls == [] and seen == 1 and unparsed == 1


def test_multiple_calls_parse_in_order() -> None:
    xml = (
        "<tool_call>\n<function=list_transitions>\n</function>\n</tool_call>\n"
        "then\n"
        "<tool_call>\n<function=diff_grids>\n<parameter=t>\n0\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls, seen, unparsed = parse_xml_tool_calls(xml)
    assert seen == 2 and unparsed == 0
    assert [c["function"]["name"] for c in calls] == ["list_transitions", "diff_grids"]


def test_block_without_function_tag_is_seen_but_unparsed() -> None:
    calls, seen, unparsed = parse_xml_tool_calls("<tool_call>\nrubbish\n</tool_call>")
    assert calls == [] and seen == 1 and unparsed == 1


# --------------------------------------------------------------------------- #
# The prompt-embedded schemas (SCENARIO-ARC-WMTE-6735).                        #
# --------------------------------------------------------------------------- #


def test_prompt_schema_render_names_every_tool_and_the_call_format() -> None:
    """The schemas must travel as prompt text (the only payload shape the flag-less
    server accepts), rendered FROM TOOL_SCHEMAS so they cannot drift from dispatch."""
    text = render_tool_schemas_for_prompt()
    for name in TOOL_NAMES:
        assert name in text
    assert "TOOL CALL FORMAT" in text
    assert "<tool_call>" in text and "<function=TOOL_NAME>" in text
    assert "<parameter=PARAM_NAME>" in text
    assert "<tool_response>" in text


# --------------------------------------------------------------------------- #
# The loop wiring (SCENARIO-ARC-WMTE-6736): no `tools` field, XML dispatched,  #
# results as user-side <tool_response> blocks, never a tool-role message.      #
# --------------------------------------------------------------------------- #


class _FakeProposer:
    """The transport surface induce_with_tool_loop needs, with no HTTP and no writes."""

    max_tokens = 1024
    timeout = 60.0
    include_playbook_exemplars = False

    def __init__(self) -> None:
        self.written: list[tuple[str, str]] = []

    def _url(self) -> str:
        return "http://127.0.0.1:1"

    def _ensure_server(self) -> bool:
        return True

    @staticmethod
    def sampling_seed(attempt: int = 0) -> None:
        return None

    def _record_completion_diagnostics(self, normalized: dict[str, Any]) -> None:
        return None

    def _write_world_model(self, game: str, code: str, note: str = "") -> tuple[bool, str]:
        self.written.append((game, code))
        return True, f"fake write: {note}"


def _identity_transitions(n: int = 3) -> list[Any]:
    from carnot.agentic.arc_executable_world_model import Transition

    g = np.zeros((4, 4), dtype=np.int16)
    return [Transition(g.copy(), 1, None, g.copy(), 0, 0) for _ in range(n)]


_ENGINE_XML_TURN = (
    "thinking about the rules here\n</think>\n\n"
    "<tool_call>\n<function=run_engine_on_transitions>\n<parameter=code>\n"
    "import numpy as np\n"
    "def engine(grid, action, data):\n    return grid\n"
    "def is_level_complete(grid):\n    return False\n"
    "</parameter>\n</function>\n</tool_call>"
)


def test_selfparse_loop_dispatches_xml_with_no_tools_field_and_no_tool_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through induce_with_tool_loop with a canned model turn: the XML call
    must be lifted agent-side, dispatched through the shared path, and fed back as a
    user-side <tool_response> -- with no `tools` request field and no tool-role
    message anywhere in the stream."""
    from carnot.agentic import arc_induction_tool_loop as loop

    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "selfparse")
    captured: dict[str, Any] = {}

    def fake_post_chat(
        proposer: Any,
        messages: list[dict[str, Any]],
        *,
        turn: int,
        timeout_s: float,
        selfparse: bool = False,
    ) -> dict[str, Any]:
        captured["messages"] = messages  # the loop mutates this same list
        captured["selfparse_flag"] = selfparse
        return {
            "choices": [{"message": {"content": _ENGINE_XML_TURN}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10},
        }

    monkeypatch.setattr(loop, "_post_chat", fake_post_chat)
    prop = _FakeProposer()
    ok, note = loop.induce_with_tool_loop(prop, "testgame", _identity_transitions(), 1)
    assert ok, note
    stats = prop.last_tool_loop_stats  # type: ignore[attr-defined]
    assert stats["selfparse"] is True
    assert stats["selfparse_turns_with_tool_call_text"] == 1
    assert stats["selfparse_blocks_seen"] == 1
    assert stats["selfparse_calls_parsed"] == 1
    assert stats["selfparse_blocks_unparsed"] == 0
    assert stats["tool_calls_total"] == 1
    assert stats["terminated_by"] == "zero_mismatches"
    assert captured["selfparse_flag"] is True
    msgs = captured["messages"]
    # The schemas travel in the FIRST user message (prompt text, not a request field).
    assert "TOOL CALL FORMAT" in msgs[0]["content"]
    # Never a tool-role message; the result returns as a user-side <tool_response>.
    assert all(m["role"] != "tool" for m in msgs)
    assert any(m["role"] == "user" and "<tool_response>" in m["content"] for m in msgs[1:])
    # The assistant turn is plain text: no tool_calls field, think channel stripped.
    assistant = [m for m in msgs if m["role"] == "assistant"]
    assert assistant and "tool_calls" not in assistant[0]
    assert "</think>" not in assistant[0]["content"]
    assert "<tool_call>" in assistant[0]["content"]
    # The accepted world model was written from the dispatched candidate.
    assert prop.written and "def engine" in prop.written[0][1]


def test_post_chat_selfparse_payload_carries_no_tools_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured HTTP 400 fires only when the request CARRIES `tools`; the
    selfparse payload must therefore never include it (nor tool_choice)."""
    from carnot.agentic import arc_induction_tool_loop as loop

    sent: dict[str, Any] = {}

    class _Resp:
        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _Resp:
        sent["payload"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(loop.urllib.request, "urlopen", fake_urlopen)
    prop = _FakeProposer()
    loop._post_chat(prop, [{"role": "user", "content": "x"}], turn=0, timeout_s=5, selfparse=True)
    assert "tools" not in sent["payload"] and "tool_choice" not in sent["payload"]
    loop._post_chat(prop, [{"role": "user", "content": "x"}], turn=0, timeout_s=5, selfparse=False)
    assert sent["payload"]["tools"] and sent["payload"]["tool_choice"] == "auto"


def test_induce_hook_enters_the_loop_on_selfparse_but_not_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """induce()'s hook must accept 'selfparse' as a primary mode; 'repair' must still
    bypass it (that mode fires only through the recall-gated resample)."""
    from carnot.agentic import arc_executable_world_model as awm
    from carnot.agentic import arc_induction_tool_loop as loop

    called: list[str] = []

    def fake_loop(*a: Any, **k: Any) -> tuple[bool, str]:
        called.append("loop")
        return True, "fake loop note"

    monkeypatch.setattr(loop, "induce_with_tool_loop", fake_loop)

    class _SentinelError(Exception):
        pass

    def boom(*a: Any, **k: Any) -> str:
        raise _SentinelError

    # The single-shot path starts at induce_prompt; a sentinel there proves which
    # branch ran without executing the heavy path.
    monkeypatch.setattr(awm, "induce_prompt", boom)
    prop = object.__new__(awm.LocalGGUFProposer)
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "selfparse")
    ok, note = prop.induce("g", _identity_transitions(), 1)
    assert ok and note == "fake loop note" and called == ["loop"]
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "repair")
    with pytest.raises(_SentinelError):
        prop.induce("g", _identity_transitions(), 1)
    assert called == ["loop"]


# --------------------------------------------------------------------------- #
# Goal-prompt RLE render (REQ-ARC-WMTE-6740, SCENARIO-ARC-WMTE-6741).          #
# --------------------------------------------------------------------------- #


def test_goal_only_prompt_renders_previous_grid_rle_not_ascii() -> None:
    """The last full-grid ASCII render in the induce-prompt family: to_ascii costs
    4,159 tokens on ANY 64x64 (1 token/digit char, pinned Qwen3.8-27B GGUF tokenizer);
    _rle_grid costs 1,002-2,776. Measured saving on real reset frames: tu93
    4,309 -> 1,928 (-2,381), ft09 4,309 -> 2,995 (-1,314), full goal prompts."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    grid = np.zeros((64, 64), dtype=np.int16)
    prompt = LocalGGUFProposer._goal_only_prompt(object.__new__(LocalGGUFProposer), "tu93", grid)
    # RLE form present; the raw one-char-per-cell row must be gone.
    assert "r0:0x64" in prompt
    assert "0" * 64 not in prompt
    # The model is told how to decode the encoding it is shown.
    assert "run-length encoded" in prompt
    # The 2026-07-29 polarity correction survives the re-render.
    assert "is_level_complete must return False here" in prompt
    assert "board at the START of the current level" in prompt
