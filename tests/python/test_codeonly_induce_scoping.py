"""The code-only truncation fix must be scoped to INDUCE only -- never refactor or gap-fillers.

Origin: 2026-06-25 adversarial verification (workflow verify-goal-repair-and-default-on) caught that
keying the code-only gate on ``"is_level_complete" in required`` alone ALSO swept refactor() into the
"skip all reasoning" path, because both induce() and refactor() route through _gen_to_file with the
same ``required`` tuple. refactor() is a REASONING task (debug BEFORE/PREDICTED/OBSERVED mismatches),
so suppressing reasoning degrades it. The gate now keys on an explicit ``codeonly_eligible`` flag set
True only by induce(). These tests guard that scoping (and the DEFAULT-ON behaviour) without a server.
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
    VerifyResult,
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


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Intercept the /completion POST and capture the request body (prompt + stop)."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(json.dumps({"content": _VALID_CODE}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _proposer(monkeypatch: pytest.MonkeyPatch) -> LocalGGUFProposer:
    p = LocalGGUFProposer(
        repo_substr="X",
        model_path="/x.gguf",
        port=59999,
        no_think_prefix="/no_think\n",
        max_tokens=128,
        tries=1,
    )
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    return p


# --------------------------------------------------------------------------- #
# generate()-level gate behaviour                                             #
# --------------------------------------------------------------------------- #
def test_induce_eligible_is_codeonly_with_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEFAULT ON: a codeonly_eligible induce call prepends the directive + adds stop=['```']."""
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    cap = _capture(monkeypatch)
    p = _proposer(monkeypatch)
    ok, _ = p.generate("BASE_PROMPT", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok is True
    assert cap["body"]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert cap["body"].get("stop") == ["```"]


def test_not_eligible_is_not_codeonly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NON-eligible call (refactor) keeps the normal /no_think path and NO stop-sequence."""
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    cap = _capture(monkeypatch)
    p = _proposer(monkeypatch)
    ok, _ = p.generate("BASE_PROMPT", ("engine", "is_level_complete"), codeonly_eligible=False)
    assert ok is True
    assert not cap["body"]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert cap["body"]["prompt"].startswith("/no_think\n")
    assert "stop" not in cap["body"]


def test_optout_env_disables_even_when_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """CARNOT_ARC_CODEONLY_INDUCE=0 disables code-only even for an eligible induce call."""
    monkeypatch.setenv("CARNOT_ARC_CODEONLY_INDUCE", "0")
    cap = _capture(monkeypatch)
    p = _proposer(monkeypatch)
    ok, _ = p.generate("BASE", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok is True
    assert not cap["body"]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert "stop" not in cap["body"]


# --------------------------------------------------------------------------- #
# method-level scoping: induce() opts in, refactor() opts out                 #
# --------------------------------------------------------------------------- #
def test_induce_method_requests_codeonly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    cap = _capture(monkeypatch)
    p = _proposer(monkeypatch)
    trans = [
        Transition(
            grid=np.zeros((2, 2), dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.ones((2, 2), dtype=np.int16),
            level_before=0,
            level_after=0,
        )
    ]
    ok, _ = p.induce("g", trans, 1)
    assert ok is True
    assert cap["body"]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert cap["body"].get("stop") == ["```"]


def test_refactor_method_does_not_request_codeonly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression the verification caught: refactor() must NOT be code-only even default-ON."""
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    cap = _capture(monkeypatch)
    p = _proposer(monkeypatch)
    vr = VerifyResult(n=2, n_correct=1, accuracy=0.5, mismatches=[{"kind": "x"}])
    ok, _ = p.refactor("g", vr)
    assert ok is True
    assert not cap["body"]["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert "stop" not in cap["body"]
