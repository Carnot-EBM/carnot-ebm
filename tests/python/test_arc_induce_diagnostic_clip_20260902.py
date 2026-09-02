"""REQ-ARC-WMTE-6860: stored induce-failure notes carry the whole diagnostic.

Origin: 2026-09-02. Three live r11l induce failures stored 179-character notes -- the
29-char `split induce: engine failed: ` prefix plus a 150-char clip -- cut mid-word at
`budg`. The dropped tail carried the n_ctx pool size and the `RAISE -c /
CARNOT_ARC_INDUCE_N_CTX` instruction, i.e. the fix. The record said THAT truncation
happened while hiding what to do about it.

These tests drive the real `induce()` split path against a scripted server whose reply
is a pool-truncated non-answer (stop_type "limit", predicted_n far below the budget),
so `_limit_diagnostic()`'s shared-pool branch builds the real message, and assert the
instruction tail survives into the returned note.

Spec refs: REQ-ARC-WMTE-6860, SCENARIO-ARC-WMTE-6860-A, SCENARIO-ARC-WMTE-6860-B.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as awm
from carnot.agentic.arc_executable_world_model import (
    INDUCE_FAILURE_NOTE_CLIP,
    LocalGGUFProposer,
    Transition,
)

pytestmark = pytest.mark.memory_watchdog_skip


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._b = payload

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def read(self, *_a: object) -> bytes:
        return self._b


def _truncated_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every call replays the incident shape: the server answered (HTTP 200), the model
    reasoned at length, no code reached the answer channel, and generation stopped at
    `stop_type == "limit"` far short of the requested budget -- the shared-pool cut."""

    body = json.dumps(
        {
            "content": "<think>very long reasoning that never ends</think>",
            "stop_type": "limit",
            "timings": {"predicted_n": 10},
        }
    ).encode()

    def fake(req, timeout=None):  # noqa: ANN001
        return _FakeResp(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def _proposer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> LocalGGUFProposer:
    # Raw-completion path (think OFF) so the scripted body's stop_type/timings reach
    # _record_completion_diagnostics unmediated by the chat-shape normalizer.
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP", raising=False)
    # E3_DIR rebound off the tracked evidence store, same reason as every sibling test.
    monkeypatch.setattr(awm, "E3_DIR", tmp_path)
    p = LocalGGUFProposer(
        repo_substr="X",
        model_path="/x.gguf",
        port=59996,
        no_think_prefix="/no_think\n",
        max_tokens=128,
        tries=1,
    )
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    return p


def _trans() -> list[Transition]:
    return [
        Transition(
            grid=np.zeros((2, 2), dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.ones((2, 2), dtype=np.int16),
            level_before=1,
            level_after=1,
        )
    ]


def test_pool_truncation_fix_instruction_survives_the_split_induce_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """SCENARIO-ARC-WMTE-6860-A. The old [:150] clip stored exactly 179 characters and cut
    the diagnostic at `budg`, before the pool size and the RAISE -c instruction."""

    p = _proposer(monkeypatch, tmp_path)
    _truncated_urlopen(monkeypatch)
    ok, msg = p.induce("g", _trans(), 1)
    assert ok is False
    assert msg.startswith("split induce: engine failed:")
    assert "TRUNCATED BY SHARED CONTEXT POOL" in msg
    # The tail the incident lost: the pool size and the complete fix instruction.
    assert "n_ctx=" in msg
    assert "RAISE -c / CARNOT_ARC_INDUCE_N_CTX" in msg
    assert "raising max_tokens would make this worse]" in msg


def test_clip_constant_covers_the_longest_message_generate_returns() -> None:
    """SCENARIO-ARC-WMTE-6860-B arithmetic. generate() caps server-failure messages at 400
    chars at their own site; the longest wrapper prefix is 29 chars. The shared clip must
    cover their sum or a max-length message loses its tail again."""

    longest_prefix = len("split induce: engine failed: ")
    server_failure_cap = 400
    assert INDUCE_FAILURE_NOTE_CLIP >= longest_prefix + server_failure_cap


def test_no_storage_site_narrows_the_clip_below_the_constant() -> None:
    """SCENARIO-ARC-WMTE-6860-B. The incident chain had four independent local literals
    (150, 160, 240, 300); widening any one still lost the tail at the next. Pin each site
    to the shared constant so they cannot drift apart again."""

    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    sites = {
        "python/carnot/agentic/arc_executable_world_model.py": [
            r'f"split induce: engine failed: \{str\(eng\)\[:INDUCE_FAILURE_NOTE_CLIP\]\}"',
            r'f"split induce: goal failed: \{str\(goal\)\[:INDUCE_FAILURE_NOTE_CLIP\]\}"',
        ],
        "python/carnot/agentic/arc_competition_agent.py": [
            r'attempt\["proposer_note"\] = str\(_induce_note\)\[:INDUCE_FAILURE_NOTE_CLIP\]',
            r'record\["error"\] = str\(msg\)\[:INDUCE_FAILURE_NOTE_CLIP\]',
        ],
        "python/carnot/agentic/arc_llm_reinduction.py": [
            r'row\["message"\] = str\(message\)\[:INDUCE_FAILURE_NOTE_CLIP\]',
        ],
    }
    for rel, patterns in sites.items():
        text = (repo / rel).read_text()
        for pat in patterns:
            assert re.search(pat, text), f"{rel} lost its shared-clip site: {pat}"
