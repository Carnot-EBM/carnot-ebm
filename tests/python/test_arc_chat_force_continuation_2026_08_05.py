"""The empty-answer-channel fallback (CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK) reads a
different channel when the answer is empty. Measured live 2026-08-05 (n=5 cells, exp6091):
that does not help when NEITHER channel has code -- the model reasons correctly then emits
EOS at </think> without ever writing an answer, well under the token budget. This tests the
fix: CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION re-issues one request with the model's own
reasoning fed back as an assistant-role prefill, forcing continuation into the code fence
rather than a fresh turn.

SCENARIO-ARC-CHAT-CONTINUATION-2026-08-05
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

THINK_NO_CODE = "Let's implement this.\n    * Wait, what about is_level_complete?\n    * I'll just return False."
REAL_CODE = "import numpy as np\n\n\ndef engine(grid, action, data=None):\n    return grid.copy()\n"


class _ChatServer:
    """Fakes the /v1/chat/completions HTTP boundary only. Distinguishes a continuation
    retry from a fresh call by checking whether the posted payload carries a trailing
    assistant-role message -- the actual signal `_chat_complete_request` sends, not call
    order, so the test cannot pass by accident on ordering alone."""

    def __init__(self, first_reply: dict, continuation_reply: dict | None = None) -> None:
        self.first_reply = first_reply
        self.continuation_reply = continuation_reply
        self.calls: list[dict] = []

    def urlopen(self, req: Any, timeout: float | None = None) -> Any:  # noqa: ANN401
        payload = json.loads(req.data.decode())
        self.calls.append(payload)
        is_continuation = payload["messages"][-1]["role"] == "assistant"
        body = (
            self.continuation_reply
            if is_continuation and self.continuation_reply
            else self.first_reply
        )

        class _Resp:
            def __enter__(_s):  # noqa: ANN001, N805
                return _s

            def __exit__(_s, *a):  # noqa: ANN001, N805
                return False

            def read(_s):  # noqa: ANN001, N805
                return json.dumps(body).encode()

        return _Resp()


def _openai_shape(*, content: str = "", reasoning: str = "", finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": reasoning},
                "finish_reason": finish_reason,
            }
        ]
    }


def _proposer(monkeypatch: pytest.MonkeyPatch, server: _ChatServer):
    import urllib.request

    prop = LocalGGUFProposer(repo_substr="gemma-4-31B-it", port=65512)
    monkeypatch.setattr(prop, "_ensure_server", lambda: True)
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)
    return prop


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION",
        "CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK",
    ):
        monkeypatch.delenv(var, raising=False)


class TestFlagDefaultsOffAndIsANoOp:
    def test_flag_off_leaves_the_bug_reproducing(self, monkeypatch: pytest.MonkeyPatch):
        """OFF must reproduce the exact 2026-08-05 failure: one call, empty extraction, no
        retry issued. This is what makes the default a true no-op."""
        srv = _ChatServer(_openai_shape(content="", reasoning=THINK_NO_CODE))
        prop = _proposer(monkeypatch, srv)
        normalized, extraction = prop._chat_complete_request(
            "induce prompt", max_tokens=16384, temperature=0.0, stop=None
        )
        assert extraction == "", "flag OFF must not read any fallback or continuation"
        assert len(srv.calls) == 1, "flag OFF must never issue a second request"

    def test_flag_on_but_answer_already_present_is_untouched(self, monkeypatch: pytest.MonkeyPatch):
        """No-op whenever extraction already has a code marker -- the property the docstring
        claims. Proves the flag cannot regress an already-passing cell."""
        monkeypatch.setenv("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION", "1")
        srv = _ChatServer(_openai_shape(content=REAL_CODE, reasoning="reasoning here"))
        prop = _proposer(monkeypatch, srv)
        _normalized, extraction = prop._chat_complete_request(
            "induce prompt", max_tokens=16384, temperature=0.0, stop=None
        )
        assert "def engine" in extraction
        assert len(srv.calls) == 1, "a passing call must not trigger a retry"

    def test_flag_on_but_reasoning_also_empty_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch):
        """Nothing to continue from -- must not retry with an empty prefix."""
        monkeypatch.setenv("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION", "1")
        srv = _ChatServer(_openai_shape(content="", reasoning=""))
        prop = _proposer(monkeypatch, srv)
        _normalized, extraction = prop._chat_complete_request(
            "induce prompt", max_tokens=16384, temperature=0.0, stop=None
        )
        assert extraction == ""
        assert len(srv.calls) == 1


class TestTheFixRecoversCode:
    def test_continuation_only_reply_is_fence_prepended(self, monkeypatch: pytest.MonkeyPatch):
        """Server returns ONLY the newly-generated tokens after the supplied prefix (no
        echo) -- the defensive fence-prepend path. The reply body deliberately has NEITHER
        a fence NOR "def " (adversarial-review finding: the first draft of this fixture
        accidentally contained both, so it silently exercised the OTHER branch instead)."""
        monkeypatch.setenv("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION", "1")
        srv = _ChatServer(
            first_reply=_openai_shape(content="", reasoning=THINK_NO_CODE),
            continuation_reply=_openai_shape(content="    return grid.copy()\n"),
        )
        prop = _proposer(monkeypatch, srv)
        _normalized, extraction = prop._chat_complete_request(
            "induce prompt", max_tokens=16384, temperature=0.0, stop=None
        )
        assert extraction.startswith("```python\n"), "the prepend arm must have fired"
        assert "return grid.copy()" in extraction
        assert len(srv.calls) == 2, "must issue exactly one retry"
        assert srv.calls[1]["messages"][-1]["role"] == "assistant"
        assert srv.calls[1]["messages"][-1]["content"].endswith("```python\n")
        assert srv.calls[1].get("add_generation_prompt") is False

    def test_echoed_prefix_reply_is_not_double_fenced(self, monkeypatch: pytest.MonkeyPatch):
        """Server echoes the supplied prefix back plus new tokens -- the retry's own
        extraction already carries the fence, so it must be used AS-IS, not re-prepended
        (a naive prepend would corrupt the extraction with a duplicate fence line)."""
        monkeypatch.setenv("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION", "1")
        echoed = "```python\n" + REAL_CODE + "```"
        srv = _ChatServer(
            first_reply=_openai_shape(content="", reasoning=THINK_NO_CODE),
            continuation_reply=_openai_shape(content=echoed),
        )
        prop = _proposer(monkeypatch, srv)
        _normalized, extraction = prop._chat_complete_request(
            "induce prompt", max_tokens=16384, temperature=0.0, stop=None
        )
        assert extraction.count("```python") == 1, "must not double the fence"
        assert "def engine" in extraction

    def test_retry_never_retries_itself(self, monkeypatch: pytest.MonkeyPatch):
        """The recursion guard: if the retry ALSO comes back empty, there must be exactly
        one retry attempt total, never a chain. A missing guard here would let a
        persistently-empty model spin the call budget."""
        monkeypatch.setenv("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION", "1")
        srv = _ChatServer(
            first_reply=_openai_shape(content="", reasoning=THINK_NO_CODE),
            continuation_reply=_openai_shape(content="", reasoning=""),
        )
        prop = _proposer(monkeypatch, srv)
        _normalized, extraction = prop._chat_complete_request(
            "induce prompt", max_tokens=16384, temperature=0.0, stop=None
        )
        assert len(srv.calls) == 2, "must stop after exactly one retry, not chain"
        assert extraction == ""
