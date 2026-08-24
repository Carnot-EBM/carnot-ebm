"""The opt-in thinking budget on the chat induce path (REQ-ARC-WMTE-6420).

WHY THIS EXISTS. On a reasoning generator the think channel is nearly all of the tokens: measured
induction completions ran 38k-94k while the final answer was 1-8k. llama.cpp accepts a per-request
`thinking_budget_tokens` that forces `</think>` after N think tokens and still lets the model
answer. Capping `max_tokens` instead hard-truncates mid-answer and yields an unusable engine, so
the two are not interchangeable.

The budget ships DEFAULT OFF. These tests pin that, because a lever that silently changes every
induction request would be a behaviour change disguised as a knob.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from carnot.agentic.arc_executable_world_model import LocalGGUFProposer


class _CapturingProposer(LocalGGUFProposer):
    """Capture the request body without a server. `_chat_complete_request` builds the payload and
    then POSTs it; we only care about the payload, so the POST is replaced by a recorder."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips the real __init__ (no server)
        self.captured: dict | None = None
        self.repo_substr = "test"
        self.port = 9
        self._seed_base = None
        self.timeout = 1.0

    def sampling_seed(self, attempt: int = 0):  # type: ignore[override]
        return None

    def _url(self) -> str:  # type: ignore[override]
        return "http://127.0.0.1:9"


def _payload(monkeypatch, env: dict[str, str]) -> dict:
    """Run the payload-building half of _chat_complete_request and return the body."""
    for k in ("CARNOT_ARC_INDUCE_THINKING_BUDGET",):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    p = _CapturingProposer()
    seen: dict = {}

    import urllib.request

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def _fake_urlopen(req, *a, **kw):
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    # Normalisation may object to the stub reply; the payload is already captured by then, which
    # is all these tests inspect.
    with contextlib.suppress(Exception):
        p._chat_complete_request("prompt", max_tokens=64, temperature=0.2, stop=None)
    return seen.get("body", {})


def test_absent_env_sends_no_thinking_budget(monkeypatch) -> None:
    """Default OFF. An unset env var must leave the request byte-identical to before this lever."""
    body = _payload(monkeypatch, {})
    assert "thinking_budget_tokens" not in body


def test_set_env_sends_the_budget(monkeypatch) -> None:
    body = _payload(monkeypatch, {"CARNOT_ARC_INDUCE_THINKING_BUDGET": "3072"})
    assert body.get("thinking_budget_tokens") == 3072


@pytest.mark.parametrize("bad", ["0", "-5", "not-a-number", ""])
def test_malformed_or_nonpositive_env_is_ignored(monkeypatch, bad: str) -> None:
    """A bad value must fall back to OFF rather than sending garbage the server may reject --
    a broken env var should not be able to fail every induction."""
    body = _payload(monkeypatch, {"CARNOT_ARC_INDUCE_THINKING_BUDGET": bad})
    assert "thinking_budget_tokens" not in body


def test_budget_is_not_max_tokens(monkeypatch) -> None:
    """The two are separate fields. Conflating them is the failure this lever exists to avoid:
    capping max_tokens truncates the answer, capping the think budget closes the tag."""
    body = _payload(monkeypatch, {"CARNOT_ARC_INDUCE_THINKING_BUDGET": "2048"})
    assert body.get("thinking_budget_tokens") == 2048
    assert body.get("max_tokens") == 64
