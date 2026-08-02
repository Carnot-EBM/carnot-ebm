"""Content failures eat the attempts the defect gates run on, so the gates go quiet on hard games.

THE INCIDENT (2026-08-01, found in a live A/B's first treatment cell, not in review). ar25
accepted a textbook `return False` -- the model's own comment reads "no win state was given ...
maybe just return False" -- with `goal_defect_reasks == 0` AND `engine_defect_reasks == 0`. Both
gates silent simultaneously is what pointed at a shared cause rather than a bug in either one.

`attempt < tries - 1` guards both gates, and it exists for a real reason: a `continue` on the
final attempt falls out of the loop into the content-failure return, converting an accept into a
hard failure. But `tries` is ALSO the budget content failures draw from -- a reply with no code
block, a missing `def`, a syntax error. So every attempt the model wastes on malformed output is
an attempt the defect gate needed, and an answer that finally parses on the last attempt is never
checked at all.

The gate is therefore quietest exactly where it is most needed: a game the model finds hard burns
its attempts on unusable output, then lands its one parseable answer where nothing is armed.

NOT A REGRESSION FROM THE NEWER GOAL GATE. The shipped ENGINE gate carries the identical guard and
blind spot, so its measured 13/36 -> 22/36 improvement is a FLOOR on that gate, not a ceiling.

WHAT IS ASSERTED HERE. Both directions, because either alone would be misleading:
  * with the flag OFF the bug REPRODUCES exactly -- that is what makes the default a true A/B and
    what keeps the in-flight measurement that found this interpretable;
  * with it ON the gate fires on the same input, and no accept is turned into a failure.

SCENARIO-ARC-FCP-5699-43-DEFECT-GATE-OWNS-ATTEMPTS
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

CONTENT_FAILURE = "I cannot produce that."  # no code block, no `def` -- unusable
# A GENUINE engine defect, in the shape the 2026-07-30 audit measured on ft09: the `action == 6`
# path computes and then falls off the end with no `return`, so every click evaluates to None --
# and `WorldModelVerifier.score` wraps that in `np.asarray(None)` rather than raising, so it
# degrades SILENTLY into a model that predicts nothing.
#
# Deliberately NOT the identity function. `return grid.copy()` on every path is what the first
# draft of this fixture used, and it is not a defect at all -- this module's own docstring records
# the identity engine clearing every check while modelling nothing. The test failed and was right
# to: a fixture that is not defective cannot exercise a defect gate.
DEFECTIVE = """```python
def engine(grid, action, data=None):
    out = grid.copy()
    if action != 6:
        return out
    row = 0
    col = 0
```"""


class _ScriptedServer:
    """Replays a fixed reply list. Only the HTTP boundary is faked, so the real accept/reject
    control flow -- the thing under test -- runs unmodified."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def urlopen(self, req: Any, timeout: float | None = None) -> Any:  # noqa: ANN401
        body = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1

        class _Resp:
            def __enter__(_s):  # noqa: ANN001, N805
                return _s

            def __exit__(_s, *a):  # noqa: ANN001, N805
                return False

            def read(_s):  # noqa: ANN001, N805
                return json.dumps({"content": body}).encode()

        return _Resp()


def _proposer(monkeypatch: pytest.MonkeyPatch, replies: list[str]):
    import urllib.request

    srv = _ScriptedServer(replies)
    prop = LocalGGUFProposer(repo_substr="gemma-4-31B-it", port=65511)
    monkeypatch.setattr(prop, "_ensure_server", lambda: True)
    monkeypatch.setattr(urllib.request, "urlopen", srv.urlopen)
    return prop, srv


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS",
        "CARNOT_ARC_INDUCE_DEFECT_REASKS",
        "CARNOT_ARC_GOAL_DEFECT_REASKS",
    ):
        monkeypatch.delenv(var, raising=False)


def _generate(prop, **kw):
    return prop.generate(
        "irrelevant",
        required=("engine",),
        codeonly_eligible=True,
        engine_transitions=None,
        **kw,
    )


class TestTheBugReproducesWhenTheFlagIsOff:
    """The default must still be the OLD behaviour, or the A/B that found this is uninterpretable."""

    def test_flag_defaults_off(self):
        from carnot.agentic.arc_executable_world_model import _defect_gate_owns_attempts

        assert _defect_gate_owns_attempts() is False

    def test_defect_on_the_FIRST_attempt_is_re_asked(self, monkeypatch):
        """The healthy case, which is why the bug was invisible: with attempts to spare, the gate
        fires normally. Any test that only covered this would have passed throughout."""
        prop, srv = _proposer(monkeypatch, [DEFECTIVE, DEFECTIVE, DEFECTIVE])
        ok, _ = _generate(prop, tries=3)
        assert ok is True
        assert prop.n_induce_defect_reasks >= 1, "the gate must fire when attempts remain"

    def test_content_failures_SILENCE_the_gate_on_the_same_input(self, monkeypatch):
        """THE BUG. Identical defective answer, identical budget -- but two wasted attempts first,
        and the gate never runs. This is the ar25 cell."""
        prop, srv = _proposer(monkeypatch, [CONTENT_FAILURE, CONTENT_FAILURE, DEFECTIVE])
        ok, _ = _generate(prop, tries=3)
        assert ok is True, "the defective answer is still accepted -- that part is unchanged"
        assert prop.n_induce_defect_reasks == 0, (
            "if this is non-zero the bug is fixed with the flag OFF, which breaks the A/B"
        )


class TestTheFlagFixesIt:
    def test_the_gate_fires_even_after_content_failures(self, monkeypatch):
        """The fix: the defect gate gets an attempt a content failure cannot consume."""
        monkeypatch.setenv("CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS", "1")
        prop, srv = _proposer(monkeypatch, [CONTENT_FAILURE, CONTENT_FAILURE, DEFECTIVE, DEFECTIVE])
        ok, _ = _generate(prop, tries=3)
        assert prop.n_induce_defect_reasks >= 1, (
            "with the flag on, the answer that finally parses must still be checked"
        )
        assert ok is True

    def test_it_never_turns_an_accept_into_a_failure(self, monkeypatch):
        """THE GUARANTEE THE OLD GUARD EXISTED TO PROTECT, and the reason the fix GRANTS an
        attempt rather than relaxing the guard. A `continue` on the final attempt would fall out
        of the loop into the content-failure return. Here every reply is defective and the re-ask
        budget is spent, so the loop must still end in an accept."""
        monkeypatch.setenv("CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS", "1")
        prop, srv = _proposer(monkeypatch, [DEFECTIVE] * 8)
        ok, code = _generate(prop, tries=3)
        assert ok is True, "a spent re-ask budget must accept, exactly as the shipped path does"
        assert "def engine" in code

    def test_the_grant_is_bounded_by_the_reask_budget(self, monkeypatch):
        """No cap is needed and none is used: the grant is gated on `_reasks_left`, which is
        finite. If the loop could grow without bound a defective model would spin forever, so
        this pins the bound rather than trusting it."""
        monkeypatch.setenv("CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS", "1")
        monkeypatch.setenv("CARNOT_ARC_INDUCE_DEFECT_REASKS", "2")
        prop, srv = _proposer(monkeypatch, [DEFECTIVE] * 20)
        ok, _ = _generate(prop, tries=3)
        assert ok is True
        # 3 content attempts + at most 2 granted re-asks.
        assert srv.calls <= 5, f"loop grew beyond the re-ask budget: {srv.calls} calls"
        assert prop.n_induce_defect_reasks <= 2


def test_a_clean_answer_is_unaffected_either_way(monkeypatch):
    """The fix must be inert on the happy path in BOTH modes -- a change that only shows up when
    something is already broken is easier to trust."""
    clean = """```python
def engine(grid, action, data=None):
    g = grid.copy()
    g[0, 0] = 1
    return g
```"""
    for flag in ("0", "1"):
        monkeypatch.setenv("CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS", flag)
        prop, srv = _proposer(monkeypatch, [clean])
        ok, _ = _generate(prop, tries=3)
        assert ok is True
        assert srv.calls == 1, f"flag={flag}: a clean answer must cost exactly one call"
        assert prop.n_induce_defect_reasks == 0
