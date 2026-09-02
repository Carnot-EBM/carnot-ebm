"""REQ-ARC-WMTE-6890: a timed-out generator request is visible in the record.

Origin: the 2026-09-02 chars_final gate run. Two 2400s client timeouts left
`channel_totals` ALL ZERO and `last_generated_tokens=-1` -- indistinguishable, from the
artifact, from a run that never called the generator. The server log held the real numbers
(24,942 and 26,563 tokens processed, `truncated=0`); the artifact held nothing.

Spec refs: REQ-ARC-WMTE-6890, SCENARIO-ARC-WMTE-6890-A, SCENARIO-ARC-WMTE-6890-B.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from carnot.agentic import arc_executable_world_model as awm
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

pytestmark = pytest.mark.memory_watchdog_skip


def _proposer(monkeypatch: pytest.MonkeyPatch) -> LocalGGUFProposer:
    # Raw-completion path (think OFF): both transports share the same failure seam, and the
    # raw path needs no chat-shape normalizer in the fake.
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP", raising=False)
    p = LocalGGUFProposer(
        repo_substr="X",
        model_path="/x.gguf",
        port=59995,
        no_think_prefix="/no_think\n",
        max_tokens=64,
        tries=1,
    )
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    return p


def _raise_on_request(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    def fake(req, timeout=None):  # noqa: ANN001
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", fake)


@pytest.mark.parametrize(
    "exc",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))],
    ids=["bare_timeout", "urlerror_wrapped_timeout"],
)
def test_generate_counts_a_timeout(monkeypatch, exc) -> None:
    """SCENARIO-ARC-WMTE-6890-A, both observed shapes."""
    p = _proposer(monkeypatch)
    _raise_on_request(monkeypatch, exc)
    ok, msg = p.generate("prompt", ("engine",))
    assert ok is False
    assert p.channel_totals["request_timeouts"] == 1, p.channel_totals


def test_complete_text_counts_a_timeout(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-6890-A on the second seam."""
    p = _proposer(monkeypatch)
    _raise_on_request(monkeypatch, urllib.error.URLError(TimeoutError("timed out")))
    ok, _ = p.complete_text("prompt")
    assert ok is False
    assert p.channel_totals["request_timeouts"] == 1, p.channel_totals


def test_a_non_timeout_failure_does_not_masquerade(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-6890-B: connection-refused is a server failure, not a timeout."""
    p = _proposer(monkeypatch)
    _raise_on_request(monkeypatch, urllib.error.URLError(ConnectionRefusedError(111, "refused")))
    ok, _ = p.generate("prompt", ("engine",))
    assert ok is False
    assert p.channel_totals["request_timeouts"] == 0, p.channel_totals


def test_fresh_proposers_carry_the_key_at_zero() -> None:
    """The counter must exist from construction, so an artifact reading the dict can rely on
    the key being present rather than absent-means-zero (the field-names-lie trap)."""
    assert awm._EMPTY_CHANNEL_TOTALS["request_timeouts"] == 0
