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
        self,
        prompt,
        *,
        max_tokens,
        temperature,
        stop,
        attempt=0,
        repeat_penalty=None,
        repeat_last_n=None,
        _continuation_prefix=None,
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
        self,
        prompt,
        *,
        max_tokens,
        temperature,
        stop,
        attempt=0,
        repeat_penalty=None,
        repeat_last_n=None,
        _continuation_prefix=None,
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
        self,
        prompt,
        *,
        max_tokens,
        temperature,
        stop,
        attempt=0,
        repeat_penalty=None,
        repeat_last_n=None,
        _continuation_prefix=None,
    ):
        calls.append({"prompt": prompt})
        return ({"content": _VALID_CODE, "stop_type": "eos"}, _VALID_CODE)

    monkeypatch.setattr(LocalGGUFProposer, "_chat_complete_request", fake_chat)
    ok, _ = p.induce("g", _one_transition(), 1)
    assert ok is True
    assert len(calls) == 1
    assert not calls[0]["prompt"].rstrip().endswith("```python")
    assert "Return ONLY one ```python code block" in calls[0]["prompt"]


# --------------------------------------------------------------------------- #
# CORRECTED 2026-08-08 (adversarial review): the two OTHER induce prompt      #
# shapes had the same unconditional pre-opened fence as the combined call     #
# above, so think mode was only actually respected on the combined path.      #
# --------------------------------------------------------------------------- #
_ENGINE_ONLY_CODE = "```python\nimport numpy as np\ndef engine(grid, action, data):\n    return np.asarray(grid)\n```"
_GOAL_ONLY_CODE = "```python\ndef is_level_complete(grid):\n    return False\n```"


def test_think_on_goal_only_prompt_omits_pre_opened_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_goal_only_prompt` is a pure string builder -- test it directly, no HTTP needed."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    p = _proposer(monkeypatch)
    prompt = p._goal_only_prompt("g", np.zeros((2, 2), dtype=np.int16))
    assert not prompt.rstrip().endswith("```python")
    assert "Return ONLY one ```python code block defining is_level_complete." in prompt


def test_think_off_goal_only_prompt_appends_pre_opened_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin: think explicitly off keeps the old unconditional fence, unchanged."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    p = _proposer(monkeypatch)
    prompt = p._goal_only_prompt("g", np.zeros((2, 2), dtype=np.int16))
    assert prompt.rstrip().endswith("```python")


def test_think_on_split_engine_fallback_omits_pre_opened_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split-induce engine-half fallback (induce()'s combined call failed, so it retries
    engine-only then goal-only) must not prime a fence either, for the same reason as the
    combined call and the goal-only prompt above."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    p = _proposer(monkeypatch)
    calls: list[dict] = []

    def fake_chat(
        self,
        prompt,
        *,
        max_tokens,
        temperature,
        stop,
        attempt=0,
        repeat_penalty=None,
        repeat_last_n=None,
        _continuation_prefix=None,
    ):
        calls.append({"prompt": prompt})
        if len(calls) == 1:
            # Combined call: engine-only code, missing is_level_complete -> required=("engine",
            # "is_level_complete") fails -> induce() falls through to the split path.
            return ({"content": _ENGINE_ONLY_CODE, "stop_type": "eos"}, _ENGINE_ONLY_CODE)
        if len(calls) == 2:
            return ({"content": _ENGINE_ONLY_CODE, "stop_type": "eos"}, _ENGINE_ONLY_CODE)
        return ({"content": _GOAL_ONLY_CODE, "stop_type": "eos"}, _GOAL_ONLY_CODE)

    monkeypatch.setattr(LocalGGUFProposer, "_chat_complete_request", fake_chat)
    ok, _ = p.induce("g", _one_transition(), 1)
    assert ok is True
    assert len(calls) == 3  # combined (fails) -> engine-only -> goal-only
    engine_fallback_prompt = calls[1]["prompt"]
    assert not engine_fallback_prompt.rstrip().endswith("```python")
    assert (
        "Return ONLY one ```python code block defining engine(grid, action, data)."
        in engine_fallback_prompt
    )


def test_think_off_split_engine_fallback_appends_pre_opened_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin: think explicitly off keeps the old unconditional fence on the split
    engine-half fallback too, unchanged."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    cap = _capture_raw(monkeypatch)
    p = _proposer(monkeypatch)
    # A single-URL raw-completion capture can't distinguish 3 sequential POSTs by content, so
    # this pins the LAST request body only -- sufficient to confirm the fence is still present
    # in whichever call it lands on when think is off (codeonly wraps every codeonly-eligible
    # call in this branch identically).
    ok, _ = p.induce("g", _one_transition(), 1)
    assert ok is True
    assert cap["body"]["prompt"].rstrip().endswith("```python")


def test_repeat_penalty_forwarded_through_chat_completions_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORRECTED 2026-08-08 (adversarial review): the chat-completions route silently dropped
    repeat_penalty/repeat_last_n, even though the raw /completion route (generate()'s other
    branch) has applied them to the engine-induce prompt since REQ-ARC-WMTE-6198's fix (13/36
    -> 22/36 usable engines). Since think mode FORCES the chat route regardless of
    use_chat_template, every think-mode engine-induce call ran without repetition control until
    this fix. _INDUCE_REPEAT_PENALTY's shipped default (1.1) applies with no env override
    needed."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    p = _proposer(monkeypatch)
    calls: list[dict] = []

    def fake_chat(
        self,
        prompt,
        *,
        max_tokens,
        temperature,
        stop,
        attempt=0,
        repeat_penalty=None,
        repeat_last_n=None,
        _continuation_prefix=None,
    ):
        calls.append({"repeat_penalty": repeat_penalty, "repeat_last_n": repeat_last_n})
        return ({"content": _VALID_CODE, "stop_type": "eos"}, _VALID_CODE)

    monkeypatch.setattr(LocalGGUFProposer, "_chat_complete_request", fake_chat)
    ok, _ = p.generate("BASE_PROMPT", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok is True
    assert len(calls) == 1
    assert calls[0]["repeat_penalty"] == pytest.approx(1.1)
    assert calls[0]["repeat_last_n"] == 256


def test_repeat_penalty_not_forwarded_when_engine_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The penalty stays scoped to engine-induce calls (REQ-ARC-WMTE-6198's own scoping,
    unchanged by this fix): a goal-only call ("is_level_complete" only, no "engine" in
    required) must not carry it through the chat route either."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    p = _proposer(monkeypatch)
    calls: list[dict] = []

    def fake_chat(
        self,
        prompt,
        *,
        max_tokens,
        temperature,
        stop,
        attempt=0,
        repeat_penalty=None,
        repeat_last_n=None,
        _continuation_prefix=None,
    ):
        calls.append({"repeat_penalty": repeat_penalty, "repeat_last_n": repeat_last_n})
        return ({"content": _GOAL_ONLY_CODE, "stop_type": "eos"}, _GOAL_ONLY_CODE)

    monkeypatch.setattr(LocalGGUFProposer, "_chat_complete_request", fake_chat)
    ok, _ = p.generate("BASE_PROMPT", ("is_level_complete",), codeonly_eligible=True)
    assert ok is True
    assert len(calls) == 1
    assert calls[0]["repeat_penalty"] is None
    assert calls[0]["repeat_last_n"] is None
