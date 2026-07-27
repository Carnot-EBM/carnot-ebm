"""The shared-pool truncation detector must be LIVE on the chat-completions path.

WHY THIS FILE EXISTS (2026-07-27 adversarial review of commit 776161963).

That commit added `_limit_diagnostic()` to separate the two faults that both report
`stop_type == "limit"`:

  * INTENDED BUDGET LIMIT -- the model generated the whole `max_tokens` we asked for. The fix is
    a BIGGER max_tokens.
  * SHARED-POOL TRUNCATION (mode C) -- the model was cut off far short of `max_tokens` because
    the prompt had already consumed most of the server's shared context pool. The fix is a
    BIGGER `-c` / `CARNOT_ARC_INDUCE_N_CTX`, and a bigger `max_tokens` would make it WORSE.

The prescriptions are OPPOSITE, which is what makes conflating them expensive. But the detector
was STRUCTURALLY DEAD on one of the two endpoints it has to cover. `_chat_complete_request()`
normalizes the OpenAI-shaped reply into llama.cpp's native `{content, stop_type, truncated}`
shape -- and carried NO `timings` key. So `_record_completion_diagnostics()` set
`last_generated_tokens = -1`, and `_limit_diagnostic()`'s guard (`0 <= got < max_tokens - 8`)
could never be true when `use_chat_template=True`. It always fell through to the
actively-misleading "HIT n_predict OUTPUT LIMIT" branch.

That is the SAME dead-channel class the diagnostic was added to fix -- a field that exists, is
read, and can never report anything -- reintroduced on the sibling endpoint in the same commit.
The scored Kaggle path uses `use_chat_template=False` so it worked there, which is exactly why it
went unnoticed; `use_chat_template=True` is the Qwen3.6 / ThinkingCap path (REQ-ARC-WMTE-5725).

These tests drive the real `_chat_complete_request` -> `_record_completion_diagnostics` ->
`_limit_diagnostic` chain against a stubbed transport, asserting the branch FIRES when it should
and does NOT fire when it should not. A test that only asserted the counter would have passed
against the broken version.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from carnot.agentic.arc_executable_world_model import LocalGGUFProposer


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture()
def chat_proposer(monkeypatch):
    """A proposer pinned to the chat endpoint, with the transport stubbed.

    Stubbing at `urllib.request.urlopen` rather than at `_chat_complete_request` is deliberate:
    the bug lived INSIDE that method's normalization, so a stub above it would have reproduced
    the reconstruction-agrees-with-reconstruction failure this project keeps hitting.
    """
    p = LocalGGUFProposer()
    p.max_tokens = 4096
    p.use_chat_template = True

    def _install(payload: dict) -> None:
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload), raising=True
        )

    p._install_payload = _install  # type: ignore[attr-defined]
    return p


def _drive(p: LocalGGUFProposer) -> str:
    resp, _ = p._chat_complete_request(
        "prompt", max_tokens=p.max_tokens, temperature=0.2, stop=None
    )
    p._record_completion_diagnostics(resp)
    return p._limit_diagnostic()


def test_pool_truncation_fires_on_the_chat_path(chat_proposer) -> None:
    """THE REGRESSION. 300 generated tokens against a 4096 budget, stopped on `length`, is mode C
    -- the silent quality degradation. Before this fix the chat path reported it as
    'HIT n_predict OUTPUT LIMIT', whose prescription (raise max_tokens) makes it WORSE."""
    chat_proposer._install_payload(
        {
            "choices": [{"message": {"content": "x" * 900}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 300},
        }
    )
    diag = _drive(chat_proposer)
    assert chat_proposer.last_generated_tokens == 300, (
        "generated-token count did not survive normalization; the detector is dead again"
    )
    assert "TRUNCATED BY SHARED CONTEXT POOL" in diag
    assert "CARNOT_ARC_INDUCE_N_CTX" in diag, "the message must name the lever that fixes it"
    assert "HIT n_predict" not in diag


def test_intended_budget_limit_does_not_over_fire_on_the_chat_path(chat_proposer) -> None:
    """The other side of the discrimination. Generating the FULL budget is not pool exhaustion,
    and reporting it as such would send an operator to raise n_ctx for no reason."""
    chat_proposer._install_payload(
        {
            "choices": [{"message": {"content": "y" * 9000}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 4096},
        }
    )
    diag = _drive(chat_proposer)
    assert chat_proposer.last_generated_tokens == 4096
    assert "HIT n_predict=4096 OUTPUT LIMIT" in diag
    assert "TRUNCATED BY SHARED CONTEXT POOL" not in diag


def test_native_timings_are_preferred_when_the_build_provides_them(chat_proposer) -> None:
    """Newer llama.cpp builds attach a native top-level `timings` to the OpenAI-compatible reply;
    older ones only fill `usage`. Read whichever is present, preferring the native one, so the
    detector does not depend on which build the eval happens to bundle."""
    chat_proposer._install_payload(
        {
            "choices": [{"message": {"content": "z" * 40}, "finish_reason": "length"}],
            "timings": {"predicted_n": 17},
            "usage": {"completion_tokens": 9999},  # deliberately disagrees
        }
    )
    diag = _drive(chat_proposer)
    assert chat_proposer.last_generated_tokens == 17
    assert "generated only 17" in diag


def test_absent_token_count_degrades_to_the_old_message_rather_than_lying(chat_proposer) -> None:
    """If a build supplies NEITHER source, the honest fallback is the generic message -- not a
    fabricated count and not a truncation claim we cannot support."""
    chat_proposer._install_payload(
        {"choices": [{"message": {"content": "w" * 40}, "finish_reason": "length"}]}
    )
    diag = _drive(chat_proposer)
    assert chat_proposer.last_generated_tokens == -1
    assert "TRUNCATED BY SHARED CONTEXT POOL" not in diag
    assert "HIT n_predict" in diag


def test_normalized_reply_still_carries_everything_the_model_emitted(chat_proposer) -> None:
    """Guard the pre-existing contract while adding to it: reasoning_content must still be folded
    back into `content` so truncation detection sees EVERYTHING the model produced."""
    chat_proposer._install_payload(
        {
            "choices": [
                {
                    "message": {"content": "answer", "reasoning_content": "thinking"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 12},
        }
    )
    resp, extraction = chat_proposer._chat_complete_request(
        "prompt", max_tokens=4096, temperature=0.2, stop=None
    )
    assert "<think>" in resp["content"] and "thinking" in resp["content"]
    assert "answer" in resp["content"]
    assert extraction == "answer", "extraction text must be the final answer only"
    assert resp["stop_type"] == "eos"
    assert resp["timings"]["predicted_n"] == 12
