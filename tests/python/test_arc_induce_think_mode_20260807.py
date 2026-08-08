"""REQ-ARC-WMTE-6198 (lever #6, 2026-08-07): think-mode toggle for the gemma-4-31B induction path.

Origin: operator directive "we want /think with the 31B model" (ops/known-issues.md, amending the
2026-08-03 one-slot ruling; the June /no_think freeze is lifted). Gemma-4 has no Qwen-style
/think-/no_think soft-switch token; its native thought channel only splits into `reasoning_content`
on the /v1/chat/completions endpoint (exp5764). `induce_think_on()` resolves env
CARNOT_ARC_INDUCE_THINK, falling back to ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT.

FLIPPED TO ON 2026-08-08 (operator directive, on the evidence in
experiment_6199_gemma_think_mode_ab.json / REQ-ARC-WMTE-6198's spec entry: 10/10 games induced on
both arms, reasoning engaged 10/10 vs 0/10, a consistent induction-quality edge for think). These
tests pin two properties, each exercised via an EXPLICIT env override rather than "leave it
unset" -- the shipped default is a value that can change again; the request-construction logic
each branch pins should not depend on which value happens to be current:

  1. Think OFF (CARNOT_ARC_INDUCE_THINK=0) is BYTE-IDENTICAL to pre-lever-#6 behaviour -- the
     load-bearing property, mirroring test_codeonly_induce_scoping.py's own off-arm pins.
  2. Think ON (CARNOT_ARC_INDUCE_THINK=1) skips the codeonly directive + pre-opened fence and
     routes through the chat endpoint instead of raw /completion -- the actual mechanism the
     lever adds, and the shipped default as of 2026-08-08.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as awm
from carnot.agentic.arc_executable_world_model import (
    LocalGGUFProposer,
    Transition,
    induce_think_on,
)

pytestmark = pytest.mark.memory_watchdog_skip

_VALID_CODE = (
    "```python\nimport numpy as np\n"
    "def engine(grid, action, data):\n    return np.asarray(grid)\n"
    "def is_level_complete(grid):\n    return False\n```"
)


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._b = payload

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def read(self, *_a: object) -> bytes:
        return self._b


def _capture_raw(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Intercept the raw /completion POST and capture the request body."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["body"] = json.loads(req.data.decode())
        captured["url"] = req.full_url
        return _FakeResp(json.dumps({"content": _VALID_CODE}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


@pytest.fixture(autouse=True)
def _redirect_engine_store(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Same guard as test_codeonly_induce_scoping.py: never let induce()/refactor() write into the
    tracked results/arc_e3 evidence store during a test run."""
    monkeypatch.setattr(awm, "E3_DIR", tmp_path / "e3")


def _proposer(monkeypatch: pytest.MonkeyPatch) -> LocalGGUFProposer:
    p = LocalGGUFProposer(
        repo_substr="X",
        model_path="/x.gguf",
        port=59999,
        no_think_prefix="",  # matches the live gemma pin, not the historical Qwen prefix
        max_tokens=128,
        tries=1,
    )
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    return p


def _one_transition() -> list[Transition]:
    return [
        Transition(
            grid=np.zeros((2, 2), dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.ones((2, 2), dtype=np.int16),
            level_before=0,
            level_after=0,
        )
    ]


# --------------------------------------------------------------------------- #
# induce_think_on() resolver                                                  #
# --------------------------------------------------------------------------- #
def test_think_defaults_on_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shipped default as of 2026-08-08 (see module docstring). Pinned so a future silent
    flip back to off is caught here rather than only in the live/scored config."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_THINK", raising=False)
    assert induce_think_on() is True


def test_think_env_override_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    assert induce_think_on() is True


def test_think_env_override_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    assert induce_think_on() is False


# --------------------------------------------------------------------------- #
# think OFF (explicit override): byte-identical to pre-lever-#6 behaviour     #
# --------------------------------------------------------------------------- #
def test_think_off_is_byte_identical_codeonly_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CARNOT_ARC_INDUCE_THINK=0, generate() must take the EXACT codeonly branch
    test_codeonly_induce_scoping.py::test_induce_eligible_is_codeonly_with_stop pins. Explicit
    override rather than "leave it unset" -- the shipped default flipped to on 2026-08-08, but
    this branch's request-construction logic is unchanged and still needs pinning."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    cap = _capture_raw(monkeypatch)
    p = _proposer(monkeypatch)
    ok, _ = p.generate("BASE_PROMPT", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok is True
    assert cap["body"]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert cap["body"].get("stop") == ["```"]
    assert "/v1/chat/completions" not in cap["url"]


def test_think_off_induce_appends_pre_opened_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """With think explicitly off, induce()'s combined call still pre-opens the ```python fence
    (unchanged from before lever #6)."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    cap = _capture_raw(monkeypatch)
    p = _proposer(monkeypatch)
    ok, _ = p.induce("g", _one_transition(), 1)
    assert ok is True
    # The codeonly branch wraps the whole prompt as DIRECTIVE + prompt + pre-opened fence, so the
    # induce()-appended fence ends up in the MIDDLE of the final request body, not at the tail.
    assert cap["body"]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert "\n```python\n" in cap["body"]["prompt"]
    assert cap["body"]["prompt"].rstrip().endswith("```python")  # the codeonly-added closing fence


# --------------------------------------------------------------------------- #
# think ON: no codeonly directive, no pre-opened fence, chat-endpoint routing #
# --------------------------------------------------------------------------- #
def test_think_on_skips_codeonly_directive_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    p = _proposer(monkeypatch)
    calls: list[dict] = []

    def fake_chat(
        self, prompt, *, max_tokens, temperature, stop, attempt=0, _continuation_prefix=None
    ):
        calls.append({"prompt": prompt, "stop": stop})
        return ({"content": _VALID_CODE, "stop_type": "eos"}, _VALID_CODE)

    monkeypatch.setattr(LocalGGUFProposer, "_chat_complete_request", fake_chat)
    ok, _ = p.generate("BASE_PROMPT", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok is True
    assert len(calls) == 1
    assert not calls[0]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert calls[0]["prompt"] == "BASE_PROMPT"  # no directive, no no_think_prefix prepended either
    assert calls[0]["stop"] is None  # no forced fence-close stop


def test_think_on_routes_through_chat_endpoint_not_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Think mode must use /v1/chat/completions even when self.use_chat_template is False --
    raw /completion never splits gemma's native thought channel into reasoning_content."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    cap = _capture_raw(monkeypatch)  # would record a call if generate() fell through to raw
    p = _proposer(monkeypatch)
    assert p.use_chat_template is False

    def fake_chat(
        self, prompt, *, max_tokens, temperature, stop, attempt=0, _continuation_prefix=None
    ):
        return ({"content": _VALID_CODE, "stop_type": "eos"}, _VALID_CODE)

    monkeypatch.setattr(LocalGGUFProposer, "_chat_complete_request", fake_chat)
    ok, _ = p.generate("BASE_PROMPT", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok is True
    assert "body" not in cap  # raw /completion was never called


def test_think_on_not_eligible_call_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Think mode is gated on codeonly_eligible, exactly like codeonly itself: a non-eligible
    call (refactor) must not be routed to the chat endpoint by this flag."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    cap = _capture_raw(monkeypatch)
    p = _proposer(monkeypatch)
    ok, _ = p.generate("BASE_PROMPT", ("engine", "is_level_complete"), codeonly_eligible=False)
    assert ok is True
    assert cap["body"]["prompt"] == "BASE_PROMPT"  # no_think_prefix is "" for this fixture
    assert "/v1/chat/completions" not in cap["url"]


def test_think_on_induce_omits_pre_opened_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """induce()'s combined call must not pre-open the fence when think is on (exp5714: the fence
    alone suppresses reasoning, independent of the codeonly directive)."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    p = _proposer(monkeypatch)
    calls: list[dict] = []

    def fake_chat(
        self, prompt, *, max_tokens, temperature, stop, attempt=0, _continuation_prefix=None
    ):
        calls.append({"prompt": prompt})
        return ({"content": _VALID_CODE, "stop_type": "eos"}, _VALID_CODE)

    monkeypatch.setattr(LocalGGUFProposer, "_chat_complete_request", fake_chat)
    ok, _ = p.induce("g", _one_transition(), 1)
    assert ok is True
    assert len(calls) == 1
    assert not calls[0]["prompt"].rstrip().endswith("```python")
    assert "Return ONLY one ```python code block" in calls[0]["prompt"]
