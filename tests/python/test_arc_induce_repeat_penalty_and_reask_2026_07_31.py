"""REQ-ARC-FCP-5699-41: the induce path sends a repetition penalty, and re-asks a defective engine.

WHAT WAS MEASURED, AND WHAT WAS NOT. Over 36 attempt-matched pairs on 6 games, scored
OUT-OF-SAMPLE against transitions the induce prompt never rendered
(`docs/research-notes/arc-induce-repeat-penalty-confirm-2026-07-31.md`):

  * attempts producing a mechanically-usable engine   13/36 -> 22/36, sign test p = 0.049
  * wall clock per attempt                            100.3 s -> 47.2 s (65.2 s charging re-asks)
  * calls hitting the 4096-token cap                  20/36 -> 2/36
  * `missing_return` defects                          13 -> 2

  * attempts clearing a STRICT out-of-sample quality bar   6/36 -> 7/36, p = 1.000

The last line is why these tests assert plumbing and control flow and NOTHING about engine
quality. The quality channel produced 5 discordant pairs, whose best possible two-sided p is
0.0625, so it could not have shown an effect at that n in either direction. A test here that
implied the wiring makes engines better would be asserting something the measurement explicitly
does not support -- on 2 of the 6 games no arm ever produced a correct engine at any budget.

THE FAILURE THE DEFECT GATE CLOSES. `generate()` accepted any completion that parsed and defined
the requested names. 22 of 36 measured attempts cleared exactly that bar while returning code that
could not work -- most often an `engine()` with a path that returns None, which
`WorldModelVerifier.score` wraps in `np.asarray(None)` rather than raising, degrading silently
into a world model that predicts nothing.

THE INVARIANT THAT MAKES THIS SAFE TO SHIP, pinned by
`test_a_defective_candidate_on_the_last_try_is_still_accepted`: the gate NEVER converts an accept
into a hard failure. It can only spend an extra ask, and only while budget remains.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from carnot.agentic.arc_executable_world_model import (
    _INDUCE_REPEAT_LAST_N,
    _INDUCE_REPEAT_PENALTY,
    LocalGGUFProposer,
)

# An engine with a real `return` on every path. Passes the defect check.
GOOD_ENGINE = """
import numpy as np


def engine(grid, action, data):
    out = np.array(grid).copy()
    return out


def is_level_complete(grid, data):
    return False
"""

# The measured failure shape: `engine()` falls off the end on some path and yields None. It PARSES
# and it DEFINES both required names, so the pre-2026-07-31 path accepted it.
DEFECTIVE_ENGINE = """
import numpy as np


def engine(grid, action, data):
    out = np.array(grid).copy()
    # the model rambled here and never wrote a return


def is_level_complete(grid, data):
    return False
"""


class _ScriptedServer:
    """Replays a fixed list of completions and records every outbound payload.

    Deliberately NOT a mock of `generate()` itself: the whole point is to exercise the real
    accept/reject control flow, so the only thing faked is the HTTP boundary.
    """

    def __init__(self, completions: list[str]) -> None:
        self.completions = list(completions)
        self.payloads: list[dict[str, Any]] = []

    def urlopen(self, req: Any, timeout: float | None = None) -> Any:  # noqa: ANN401
        self.payloads.append(json.loads(req.data.decode()))
        body = self.completions[min(len(self.payloads) - 1, len(self.completions) - 1)]

        class _Resp:
            def __enter__(_s) -> Any:  # noqa: ANN001, N805
                return _s

            def __exit__(_s, *a: Any) -> bool:  # noqa: ANN001, N805
                return False

            def read(_s) -> bytes:  # noqa: ANN001, N805
                return json.dumps({"content": body}).encode()

        return _Resp()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from the shipped configuration: no operator overrides in play."""
    for var in (
        "CARNOT_ARC_INDUCE_REPEAT_PENALTY",
        "CARNOT_ARC_INDUCE_DEFECT_REASKS",
        "CARNOT_ARC_GENERATOR_SEED",
    ):
        monkeypatch.delenv(var, raising=False)


def _proposer(monkeypatch: pytest.MonkeyPatch, completions: list[str]) -> tuple:
    import urllib.request

    srv = _ScriptedServer(completions)
    prop = LocalGGUFProposer(repo_substr="gemma-4-31B-it", port=65501)
    monkeypatch.setattr(prop, "_ensure_server", lambda: True)
    monkeypatch.setattr(urllib.request, "urlopen", srv.urlopen)
    return prop, srv


# ---- SCENARIO-ARC-FCP-5699-41-1: the penalty reaches the induce payload -----------------------
def test_induce_payload_carries_the_repeat_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION-VERIFIED: fails against the pre-fix payload, which sent no penalty at all and so
    inherited llama-server's default of 1.0 (read from a running server's own /props)."""
    prop, srv = _proposer(monkeypatch, [f"```python\n{GOOD_ENGINE}\n```"])
    ok, _ = prop.generate("p", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok
    assert srv.payloads[0]["repeat_penalty"] == _INDUCE_REPEAT_PENALTY
    assert srv.payloads[0]["repeat_last_n"] == _INDUCE_REPEAT_LAST_N


def test_refactor_shaped_calls_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope discipline. `refactor()` is a REASONING task on a different prompt shape with no
    measurement behind it here; penalising repetition in a debugging explanation is a different
    intervention, not this one applied more widely."""
    prop, srv = _proposer(monkeypatch, [f"```python\n{GOOD_ENGINE}\n```"])
    prop.generate("p", ("engine", "is_level_complete"), codeonly_eligible=False)
    assert "repeat_penalty" not in srv.payloads[0]
    assert "repeat_last_n" not in srv.payloads[0]


def test_penalty_of_one_restores_the_old_payload_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The off switch must be a true identity, not merely a small value -- llama.cpp treats 1.0 as
    no penalty, so an A/B that disables this must produce the byte-identical pre-fix payload."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_REPEAT_PENALTY", "1.0")
    prop, srv = _proposer(monkeypatch, [f"```python\n{GOOD_ENGINE}\n```"])
    prop.generate("p", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert "repeat_penalty" not in srv.payloads[0]
    assert "repeat_last_n" not in srv.payloads[0]


def test_malformed_penalty_falls_back_rather_than_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd env var must not take down a live episode."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_REPEAT_PENALTY", "not-a-float")
    prop, srv = _proposer(monkeypatch, [f"```python\n{GOOD_ENGINE}\n```"])
    ok, _ = prop.generate("p", ("engine", "is_level_complete"), codeonly_eligible=True)
    assert ok
    assert srv.payloads[0]["repeat_penalty"] == _INDUCE_REPEAT_PENALTY


# ---- SCENARIO-ARC-FCP-5699-41-2: a defective engine is re-asked, not accepted -----------------
def test_defective_engine_triggers_exactly_one_plain_reask(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION-VERIFIED: the pre-fix path accepted the defective candidate on call 1 and never
    made call 2. This is the 22-of-36 finding in miniature."""
    prop, srv = _proposer(
        monkeypatch,
        [f"```python\n{DEFECTIVE_ENGINE}\n```", f"```python\n{GOOD_ENGINE}\n```"],
    )
    ok, code = prop.generate(
        "PROMPT", ("engine", "is_level_complete"), tries=3, codeonly_eligible=True
    )
    assert ok
    assert "the model rambled here" not in code, "returned the defective candidate"
    assert len(srv.payloads) == 2, "expected round 0 + one re-ask"
    assert prop.n_induce_defect_reasks == 1


def test_the_reask_names_nothing_about_the_defect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A measured choice, not an oversight. `arc_engine_static_validation.repair_prompt_block()`
    builds a defect-NAMING block; head-to-head against this neutral block over 5 discordant pairs
    on disjoint games it came back p = 1.000. The defect text bought nothing and is left unwired;
    the second ASK is the entire effect. If a future change starts naming defects here, that is a
    new claim and needs its own measurement."""
    prop, srv = _proposer(
        monkeypatch,
        [f"```python\n{DEFECTIVE_ENGINE}\n```", f"```python\n{GOOD_ENGINE}\n```"],
    )
    prop.generate("PROMPT", ("engine", "is_level_complete"), tries=3, codeonly_eligible=True)
    assert len(srv.payloads) == 2, "no re-ask was made, so there is no re-ask text to inspect"
    reask = srv.payloads[1]["prompt"]
    assert "TRY AGAIN" in reask.upper() or "PREVIOUS ANSWER" in reask.upper()
    for leaked in ("missing_return", "engine_returned_none", "syntax_error"):
        assert leaked not in reask, f"re-ask leaked the defect kind {leaked!r}"


def test_a_clean_engine_is_never_reasked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate must cost nothing on the happy path -- otherwise it is a latency tax on every
    induction, and the measured win was partly a COST win."""
    prop, srv = _proposer(monkeypatch, [f"```python\n{GOOD_ENGINE}\n```"])
    ok, _ = prop.generate("p", ("engine", "is_level_complete"), tries=3, codeonly_eligible=True)
    assert ok
    assert len(srv.payloads) == 1
    assert prop.n_induce_defect_reasks == 0


def test_a_defective_candidate_on_the_last_try_is_still_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE SAFETY INVARIANT. With one try there is no room to re-ask, so the gate must accept the
    defective candidate exactly as the old path did. If this ever fails, the wiring has converted
    an accept into a hard failure and can strand a live episode with no world model at all --
    strictly worse than the silently-useless engine it was meant to replace."""
    prop, srv = _proposer(monkeypatch, [f"```python\n{DEFECTIVE_ENGINE}\n```"])
    ok, code = prop.generate("p", ("engine", "is_level_complete"), tries=1, codeonly_eligible=True)
    assert ok, "the defect gate must never fail a call the old path would have passed"
    assert "the model rambled here" in code
    assert len(srv.payloads) == 1


def test_exhausting_the_reask_budget_accepts_rather_than_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same invariant one level up: every try defective, budget spent, still an accept."""
    prop, srv = _proposer(monkeypatch, [f"```python\n{DEFECTIVE_ENGINE}\n```"] * 3)
    ok, _ = prop.generate("p", ("engine", "is_level_complete"), tries=3, codeonly_eligible=True)
    assert ok
    assert prop.n_induce_defect_reasks == 1, "budget is 1 re-ask, not one per try"


def test_reasks_can_be_disabled_independently_of_the_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two ship independently on purpose: the penalty carries 11 of the 13 paired wins and the
    re-ask 2, so an operator must be able to run either alone."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_DEFECT_REASKS", "0")
    prop, srv = _proposer(monkeypatch, [f"```python\n{DEFECTIVE_ENGINE}\n```"])
    ok, _ = prop.generate("p", ("engine", "is_level_complete"), tries=3, codeonly_eligible=True)
    assert ok
    assert len(srv.payloads) == 1
    assert srv.payloads[0]["repeat_penalty"] == _INDUCE_REPEAT_PENALTY


def test_goal_only_calls_do_not_run_the_engine_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """`induce()`'s split fallback asks for `is_level_complete` alone. The defect checks are
    written against the `engine(grid, action, data)` contract and mean nothing there."""
    goal_only = "def is_level_complete(grid, data):\n    return False\n"
    prop, srv = _proposer(monkeypatch, [f"```python\n{goal_only}\n```"])
    ok, _ = prop.generate("p", ("is_level_complete",), tries=3, codeonly_eligible=True)
    assert ok
    assert len(srv.payloads) == 1
    assert prop.n_induce_defect_reasks == 0


def test_goal_only_calls_do_not_carry_the_penalty_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NARROWED 2026-07-31 (adversarial review). The penalty gate was `codeonly_eligible` alone,
    which is ALSO True for `_split_induce`'s focused goal-only call -- so `repeat_penalty` was
    shipping to a prompt shape Phase 1 never measured (the confirm harness sent ENGINE prompts
    only). The defect gate above was already correctly narrower; the penalty now matches it.

    MUTATION-VERIFIED: reverting the condition to `codeonly_eligible` makes this fail, because
    the goal-only payload then carries the penalty.

    This is the conservative direction, not the adventurous one: with no measurement in either
    direction the goal-only call's correct state is the long-standing no-penalty baseline that
    every banked result was produced under. It is also the call `_split_induce` exists to
    protect precisely BECAUSE it does not exhibit the repetition-loop failure the penalty
    treats -- it is 'valid in ~3.5s where the combined call fails'.
    """
    goal_only = "def is_level_complete(grid, data):\n    return False\n"
    prop, srv = _proposer(monkeypatch, [f"```python\n{goal_only}\n```"])
    ok, _ = prop.generate("p", ("is_level_complete",), tries=3, codeonly_eligible=True)
    assert ok
    assert "repeat_penalty" not in srv.payloads[0]
    assert "repeat_last_n" not in srv.payloads[0]


def test_the_engine_call_still_carries_the_penalty_after_the_narrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the narrowing: it must not have turned the measured intervention off.

    Paired with the goal-only test above, these two pin the boundary from BOTH sides -- a
    mutation that widens the condition back to `codeonly_eligible` fails the first, and one
    that narrows it to something stricter (or drops the penalty entirely) fails this one.
    `engine` alone in `required` is the `_split_induce` engine call, which is a real shipped
    call shape and was measured.
    """
    prop, srv = _proposer(monkeypatch, [f"```python\n{GOOD_ENGINE}\n```"])
    ok, _ = prop.generate("p", ("engine",), tries=3, codeonly_eligible=True)
    assert ok
    assert srv.payloads[0]["repeat_penalty"] == _INDUCE_REPEAT_PENALTY
    assert srv.payloads[0]["repeat_last_n"] == _INDUCE_REPEAT_LAST_N


# ---- SCENARIO-ARC-FCP-5699-41-3: the gate can never break the path it guards ------------------
def test_a_raising_defect_checker_accepts_rather_than_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A defect DETECTOR that can break the thing it is checking is worse than no detector. If the
    validator raises for any reason -- an import failure in the sandbox, a malformed transition --
    the induce path must degrade to its old accept-everything behaviour, not to an exception."""
    from carnot.agentic import arc_engine_static_validation as sv

    def _boom(*a: Any, **k: Any) -> Any:  # noqa: ANN401
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(sv, "validate_engine_code", _boom)

    # The real validator is now a landmine...
    with pytest.raises(RuntimeError):
        sv.validate_engine_code("x")

    # ...and the induce path steps on it and keeps walking.
    prop, srv = _proposer(monkeypatch, [f"```python\n{DEFECTIVE_ENGINE}\n```"])
    assert prop._engine_defects(DEFECTIVE_ENGINE, None) == [], "must swallow to 'no defects'"
    ok, _ = prop.generate("p", ("engine", "is_level_complete"), tries=2, codeonly_eligible=True)
    assert ok, "a broken validator must not fail the induction it was only advising on"
    assert len(srv.payloads) == 1, "nor spend a re-ask on a defect it could not actually measure"


def test_the_defect_checker_really_does_detect_the_measured_defect() -> None:
    """The positive control for the test above. Without it, `_engine_defects` returning [] on a
    genuinely broken engine would look like a pass everywhere in this file -- a detector wired to
    a swallow-all `except` and a detector that works are indistinguishable unless one test insists
    the thing actually fires."""
    prop = LocalGGUFProposer(repo_substr="gemma-4-31B-it", port=65502)
    assert prop._engine_defects(GOOD_ENGINE, None) == []
    assert "missing_return" in prop._engine_defects(DEFECTIVE_ENGINE, None)
