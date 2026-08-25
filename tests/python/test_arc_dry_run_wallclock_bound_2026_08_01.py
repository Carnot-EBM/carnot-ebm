"""The dry run executes LLM-written code, so it must be bounded by something that can be killed.

THE INCIDENT (2026-07-31). The best-of-N generation loop wedged for 13 minutes in state R at 32%
CPU with no open socket and both GPUs idle -- spinning inside `dry_run_defects`' execution of
ft09 candidate 5, a generated engine that does not terminate. The completion had already been
received, so no HTTP timeout applied; the agent's wall budget is checked BETWEEN actions, not
inside one. The shipped induce path reaches the same code via
`LocalGGUFProposer.generate -> _engine_defects -> validate_engine_code -> dry_run_defects`, so a
non-terminating induced engine hangs a live episode -- on a scored submission, the whole run.

WHY THE OBVIOUS FIX IS THE WRONG ONE, and why these tests assert a process boundary rather than a
signal. `dry_run_defects` wraps every engine invocation in `except Exception` because the
exception IS the observation it wants, and `_engine_defects` upstream catches broadly and returns
`[]`, which means ACCEPT. So a SIGALRM-raised exception is caught and becomes either an ordinary
`engine_raised` defect or a clean bill of health -- a hang converted into a FALSE CLEAN, strictly
worse than the hang, because the hang is at least visible. A signal is also main-thread-only and
cannot interrupt a tight loop inside a C-level call.

SCENARIO-ARC-FCP-5699-42-NONTERMINATING-ENGINE-IS-A-DEFECT
SCENARIO-ARC-FCP-5699-42-VALIDATE-BEFORE-SCORE
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pytest

from carnot.agentic import arc_engine_static_validation as sv

SPIN = "def engine(grid, action, data=None):\n    while True:\n        pass\n"
GOOD = "def engine(grid, action, data=None):\n    g = grid.copy()\n    g[0, 0] = 1\n    return g\n"
RAISES = "def engine(grid, action, data=None):\n    raise ValueError('boom-42')\n"
RETURNS_NONE = "def engine(grid, action, data=None):\n    pass\n"


class _DuckTransition:
    """A transition double defined LOCALLY, on purpose.

    The module's contract is duck-typed and says so: "Any object with those three attributes
    works, so callers can pass test doubles." The first version of the subprocess fix pickled the
    caller's object, so the child died with `AttributeError: Can't get attribute` on exactly this
    class -- and, because a non-zero child exit is reported as `engine_crashed_validator`, blamed
    the ENGINE for an artifact of how work crosses the process boundary. Keeping the double local
    is what pins that regression.
    """

    def __init__(self, grid: Any, action: int, data: Any = None) -> None:
        self.grid, self.action, self.data = grid, action, data


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", raising=False)


@pytest.fixture
def transitions() -> list[_DuckTransition]:
    return [_DuckTransition(np.zeros((8, 8), dtype=int), 1)]


class TestTheBoundFires:
    def test_a_nonterminating_engine_is_reported_as_a_defect_not_a_hang(self, transitions):
        """THE INCIDENT, reduced. Bound at 3s so the test is fast; the shipped default is 30s."""
        started = time.monotonic()
        defects = sv.dry_run_defects(SPIN, transitions, timeout_s=3.0)
        elapsed = time.monotonic() - started

        assert [d.kind for d in defects] == ["engine_nonterminating"]
        assert elapsed < 15.0, f"the bound did not fire: {elapsed:.1f}s"
        assert defects[0].evidence["timeout_s"] == 3.0
        assert defects[0].evidence["isolation"] == "subprocess"

    def test_the_timeout_is_a_DEFECT_and_never_an_exception(self, transitions):
        """Load-bearing, and the reason this is not implemented by raising. `_engine_defects`
        catches broadly and returns [] == ACCEPT, so a raised timeout would be converted into
        acceptance -- the exact failure being fixed. Returning it in the ordinary defect list
        means the existing reject path handles it with no new machinery."""
        defects = sv.dry_run_defects(SPIN, transitions, timeout_s=2.0)
        assert defects, "a timeout must produce a defect, not an empty (== accepted) list"

    def test_a_nonterminating_engine_is_not_marked_repairable(self, transitions):
        """There is no exception text to feed back -- we never observed the code end. Marking it
        repairable would send a repair prompt describing a failure nobody saw."""
        d = sv.dry_run_defects(SPIN, transitions, timeout_s=2.0)[0]
        assert d.repairable is False
        assert d.retryable is False


class TestNothingElseChanged:
    """The bound must not cost a single existing observation. Each of these is a defect kind the
    in-process version reported, asserted to survive the process boundary intact."""

    def test_a_good_engine_still_passes_clean(self, transitions):
        assert sv.dry_run_defects(GOOD, transitions, timeout_s=20.0) == []

    def test_a_raising_engine_keeps_its_EXCEPTION_TEXT(self, transitions):
        """The text is not decoration: `repair_prompt_block` feeds it back to the model, and that
        is the difference between a veto and a fix. If it were lost in serialisation the check
        would still 'work' while quietly becoming useless."""
        defects = sv.dry_run_defects(RAISES, transitions, timeout_s=20.0)
        assert [d.kind for d in defects] == ["engine_raised"]
        assert "boom-42" in defects[0].evidence["message"]
        assert defects[0].repairable is True

    def test_a_None_returning_engine_is_still_caught(self, transitions):
        kinds = [d.kind for d in sv.dry_run_defects(RETURNS_NONE, transitions, timeout_s=20.0)]
        assert "engine_returned_none" in kinds

    def test_a_locally_defined_transition_double_still_works(self, transitions):
        """The duck-typed contract. This is the regression the first implementation broke."""
        defects = sv.dry_run_defects(GOOD, transitions, timeout_s=20.0)
        assert defects == [], "a caller-defined transition must not be reported as a defect"


class TestTheKnob:
    def test_shipped_default_is_thirty_seconds(self):
        assert sv.dry_run_timeout_default() == 30.0

    @pytest.mark.parametrize("raw", ["0", "-1", "-0.5"])
    def test_a_nonpositive_value_restores_the_in_process_path_exactly(
        self, monkeypatch, transitions, raw
    ):
        """A true A/B switch, not an approximation: the disabled path calls the SAME function the
        pre-fix code called, so 'off' is the old behaviour rather than a re-implementation."""
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", raw)
        assert sv.dry_run_timeout_default() == float(raw)
        assert sv.dry_run_defects(GOOD, transitions) == []

    @pytest.mark.parametrize("raw", ["", "   ", "abc", "1.2.3"])
    def test_a_malformed_value_falls_back_to_the_default_rather_than_raising(
        self, monkeypatch, raw
    ):
        """Fail-safe direction is BOUNDED. Falling back to 'no bound' on a typo would silently
        reinstate the hang, which is the failure mode, so a malformed knob keeps the protection."""
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", raw)
        assert sv.dry_run_timeout_default() == 30.0

    def test_an_explicit_caller_timeout_beats_the_environment(self, monkeypatch, transitions):
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", "999")
        started = time.monotonic()
        defects = sv.dry_run_defects(SPIN, transitions, timeout_s=2.0)
        assert [d.kind for d in defects] == ["engine_nonterminating"]
        assert time.monotonic() - started < 15.0


class TestTheChildEntryPoint:
    def test_the_module_is_runnable_as_the_child(self):
        """A process boundary is only a boundary if something lives on the far side. If the
        `__main__` entry breaks, `_dry_run_defects_subprocess` returns None on every call and the
        parent falls back in-process -- silently reinstating the hazard while every other test
        here still passes, because they would all take the fallback path."""
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "carnot.agentic.arc_engine_static_validation"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 2
        assert "--dry-run-job" in proc.stdout

    def test_the_subprocess_path_is_the_one_actually_taken(self, transitions):
        """Guards against the whole fix degrading to the fallback unnoticed. `isolation` is
        stamped by the subprocess branch only, so its presence proves which path ran."""
        d = sv.dry_run_defects(SPIN, transitions, timeout_s=2.0)[0]
        assert d.evidence.get("isolation") == "subprocess"


def test_validate_engine_code_runs_BEFORE_the_trust_gate_scores_the_candidate(monkeypatch):
    """The integration test this module's own docstring demanded when it was wired, and did not get.

    That docstring said: "When this module is wired into the induce path, the wiring change MUST
    add an integration test asserting `validate_engine_code` is called BEFORE
    `WorldModelVerifier.score` on the same candidate." The module WAS wired on 2026-07-31
    (`LocalGGUFProposer._engine_defects`), the orphan-lint allow-list entry was removed, and the
    docstring was left asserting the module is unwired. The obligation went unmet in both
    directions, so it is discharged here.

    WHY THE ORDER MATTERS. This module reports mechanical defects; the trust gate judges quality.
    If scoring ran first, a defective engine would be graded as merely WRONG -- which is what the
    2026-07-30 audit found: four of five rejections were broken code reported as bad predictions,
    sending attention to the model's understanding rather than to a missing `return`.
    """
    import urllib.request

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    # This fixture returns llama.cpp's native /completion response shape (`content` at the
    # top level), so explicitly exercise that generation mode. The shipped think-mode default
    # uses /v1/chat/completions and correctly requires an OpenAI-shaped `choices` response;
    # letting an unrelated default flip choose the endpoint makes this ordering test fail before
    # either operation it exists to order can run.
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")

    calls: list[str] = []

    class _Server:
        def urlopen(self, req: Any, timeout: float | None = None) -> Any:  # noqa: ANN401
            class _Resp:
                def __enter__(_s):  # noqa: ANN001, N805
                    return _s

                def __exit__(_s, *a):  # noqa: ANN001, N805
                    return False

                def read(_s):  # noqa: ANN001, N805
                    return json.dumps({"content": f"```python\n{GOOD}\n```"}).encode()

            return _Resp()

    real_validate = sv.validate_engine_code

    def _tracked_validate(*a: Any, **kw: Any):  # noqa: ANN401
        calls.append("validate_engine_code")
        return real_validate(*a, **kw)

    monkeypatch.setattr(sv, "validate_engine_code", _tracked_validate)

    from carnot.agentic import arc_executable_world_model as e3

    real_score = e3.WorldModelVerifier.score

    def _tracked_score(self, engine):  # noqa: ANN001
        calls.append("WorldModelVerifier.score")
        return real_score(self, engine)

    monkeypatch.setattr(e3.WorldModelVerifier, "score", _tracked_score)

    prop = LocalGGUFProposer(repo_substr="gemma-4-31B-it", port=65503)
    monkeypatch.setattr(prop, "_ensure_server", lambda: True)
    monkeypatch.setattr(urllib.request, "urlopen", _Server().urlopen)

    ok, _code = prop.generate(
        "irrelevant prompt",
        required=("engine",),
        codeonly_eligible=True,
        engine_transitions=[_DuckTransition(np.zeros((8, 8), dtype=int), 1)],
    )
    assert ok is True

    assert "validate_engine_code" in calls, (
        "the induce path did not reach the defect check at all -- if this fails, the module has "
        "been unwired and the orphan-lint allow-list entry needs restoring with it"
    )
    if "WorldModelVerifier.score" in calls:
        assert calls.index("validate_engine_code") < calls.index("WorldModelVerifier.score"), (
            "the trust gate scored the candidate before the mechanical defect check ran"
        )
