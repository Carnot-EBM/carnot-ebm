"""The chat endpoint must not throw away an answer that landed in the reasoning channel.

Spec: REQ-ARC-WMTE-5725 (the `use_chat_template` path these tests drive).

WHY THIS FILE EXISTS (2026-08-05, exp6091 refine-engine-visible A/B).

That run produced `induce_ok = 0/19`. Every recorded completion ended at `</think>` and contained
no `def engine`, which reads as "the model reasoned and never wrote code". On the chat endpoint
that reading is UNSAFE, because `_chat_complete_request` deliberately returns only the
post-thought `content` as the extraction text (so `_extract_python` cannot grab a draft block the
model wrote INSIDE its own reasoning). When this build's `content` is empty and everything the
model wrote sits in `reasoning_content`, the candidate is a guaranteed miss regardless of what the
model actually produced -- and the folded `last_raw_completion` view cannot distinguish the two
cases, because folding puts both channels into one string.

A live probe against the pinned gemma-4-31B-it-qat weights on 2026-08-05 showed the split is
working normally on an easy prompt (`content` 86 chars carrying the fenced `def engine`,
`reasoning_content` 630 chars), so this is NOT a claim that the transport is structurally broken.
It is the guard for the case where the model closes its thought channel and stops without
nominating an answer.

Two properties are pinned here, and the SECOND one is the reason the behaviour is behind a flag:

  1. WITH the flag, an empty answer channel falls back to the reasoning channel instead of to
     nothing.
  2. WITHOUT the flag -- and, separately, in EVERY case where the answer channel is non-empty,
     flag or not -- the extraction text is byte-identical to the pre-change behaviour. The frozen
     live/scored generator path cannot inherit this silently.

The per-channel observation fields (`last_final_content` / `last_reasoning_content`) are asserted
too: they are what makes the 0/19 failure diagnosable next time, and a field that is written but
never read is exactly the dead-channel class this module has been bitten by before (see
test_arc_chat_path_pool_truncation_detector.py).
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

FLAG = "CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK"


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
    """A proposer pinned to the chat endpoint with the HTTP transport stubbed.

    Stubbed at `urllib.request.urlopen`, not at `_chat_complete_request`, because the behaviour
    under test lives INSIDE that method's normalization -- a stub above it would only prove the
    test agrees with itself.
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
    _resp, extraction = p._chat_complete_request(
        "prompt", max_tokens=p.max_tokens, temperature=0.2, stop=None
    )
    return extraction


CODE = "```python\ndef engine(grid, action, data=None):\n    return grid\n```"


def _empty_answer_payload() -> dict:
    """The exp6091 shape: the model spent its whole emission in the thought channel, closed it,
    and stopped. `finish_reason` is `stop`, NOT `length` -- this is not truncation."""
    return {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": f"thinking...\n{CODE}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 6603},
    }


def test_empty_answer_channel_falls_back_to_reasoning_with_flag_on(
    chat_proposer, monkeypatch
) -> None:
    """DIRECTION 1. Flag on: the answer that exists is used instead of being discarded."""
    monkeypatch.setenv(FLAG, "1")
    chat_proposer._install_payload(_empty_answer_payload())
    extraction = _drive(chat_proposer)
    assert "def engine" in extraction, (
        "with the fallback on, an empty answer channel must read the reasoning channel; "
        "returning nothing here is the 0/19 failure this flag exists to fix"
    )


def test_empty_answer_channel_stays_empty_with_flag_off(chat_proposer, monkeypatch) -> None:
    """DIRECTION 2. Flag off (the shipped default) is byte-identical to the pre-change path."""
    monkeypatch.delenv(FLAG, raising=False)
    chat_proposer._install_payload(_empty_answer_payload())
    assert _drive(chat_proposer) == "", (
        "the default must not change: the frozen live/scored generator path must not start "
        "reading the reasoning channel without an explicit opt-in"
    )


def test_flag_off_is_only_off_for_the_exact_string_1(chat_proposer, monkeypatch) -> None:
    """A truthy-looking value that is not `1` must NOT enable it -- no accidental opt-in from a
    stray `CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK=true` in some inherited environment."""
    monkeypatch.setenv(FLAG, "true")
    chat_proposer._install_payload(_empty_answer_payload())
    assert _drive(chat_proposer) == ""


@pytest.mark.parametrize("flag", ["1", None])
def test_nonempty_answer_channel_is_unaffected_in_both_directions(
    chat_proposer, monkeypatch, flag
) -> None:
    """THE NO-OP PROPERTY, which is what makes the fallback safe to turn on for a run.

    Whenever the model DID nominate an answer, the extraction text is exactly that answer with the
    flag on or off. The fallback can therefore only ever convert a guaranteed miss into an
    attempt; it can never change a path that was already working.
    """
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    chat_proposer._install_payload(
        {
            "choices": [
                {
                    "message": {"content": CODE, "reasoning_content": "a DRAFT I rejected"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 208},
        }
    )
    extraction = _drive(chat_proposer)
    assert extraction == CODE
    assert "DRAFT" not in extraction, (
        "the reasoning channel must never leak into extraction when an answer exists -- that is "
        "the draft-grabbing hazard the split was introduced to prevent"
    )


def test_whitespace_only_answer_counts_as_empty(chat_proposer, monkeypatch) -> None:
    """`content` is frequently a lone newline rather than a true empty string; treating that as a
    real answer would leave the fallback dead in the most common form of the failure."""
    monkeypatch.setenv(FLAG, "1")
    payload = _empty_answer_payload()
    payload["choices"][0]["message"]["content"] = "\n \n"
    chat_proposer._install_payload(payload)
    assert "def engine" in _drive(chat_proposer)


def test_both_channels_are_recorded_separately_regardless_of_flag(
    chat_proposer, monkeypatch
) -> None:
    """The observation fields are what make the next failure diagnosable. They must be populated
    on the DEFAULT path too -- diagnosis cannot be gated behind the remedy."""
    monkeypatch.delenv(FLAG, raising=False)
    chat_proposer._install_payload(_empty_answer_payload())
    resp, _extraction = chat_proposer._chat_complete_request(
        "prompt", max_tokens=chat_proposer.max_tokens, temperature=0.2, stop=None
    )
    chat_proposer._record_completion_diagnostics(resp)
    assert chat_proposer.last_final_content == ""
    assert "def engine" in chat_proposer.last_reasoning_content
    # The folded view still carries everything, as every other consumer expects -- the two new
    # fields ADD a channel-resolved view, they do not replace the existing one.
    assert "def engine" in chat_proposer.last_raw_completion
    assert chat_proposer.last_raw_completion.startswith("<think>")
