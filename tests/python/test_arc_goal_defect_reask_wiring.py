"""END-TO-END WIRING for the goal defect re-ask, against a FAKE server (no GPU, no model).

REQ-ARC-WMTE-6071: the goal defect check is REACHED by `generate()` on both induce call shapes.
See `tests/python/test_arc_goal_defect_check.py` for the spec-entry debt these ids carry.

WHY THIS EXISTS AND WHY IT RUNS BEFORE ANY GPU TIME. Three separate knobs were found DEAD on
this code path in a single week -- present, documented, tested at the unit level, and never
reached by the live agent (`_induce_transitions_k` returned a value nothing consulted;
`repair_prompt_block` was built and never wired; a HUD-mask factor was a silent no-op across
162 arms and was read as "both settings measured"). A unit test on `_goal_defects` proves the
DETECTOR works; it proves nothing about whether `generate()` ever calls it, whether the re-ask
suffix reaches the prompt, or whether the budget is spent from the right counter.

So this drives the REAL `generate()` with a scripted fake HTTP layer and asserts the observable
consequences: how many calls went out, what the later prompts contained, which counter moved,
and that the engine's own budget was not touched. If the wiring is dead, these fail here at
zero GPU cost rather than producing a clean-looking null after hours on a card.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3

GOOD_ENGINE = (
    "def engine(grid, action, data):\n    g = grid.copy()\n    g[0, 0] = 3\n    return g\n"
)
BAD_GOAL = "def is_level_complete(grid):\n    return False\n"
GOOD_GOAL = "import numpy as np\ndef is_level_complete(grid):\n    return bool(grid[0, 0] == 3)\n"


class _T:
    def __init__(self, grid, next_grid):
        self.grid = np.asarray(grid)
        self.next_grid = np.asarray(next_grid)
        self.action = 1
        self.data = None
        self.level_before = 0
        self.level_after = 0


def _trans():
    a = np.zeros((4, 4), dtype=int)
    b = a.copy()
    b[0, 0] = 3
    return [_T(a, b), _T(b, a)]


class _FakeHTTP:
    """Records every outgoing prompt and replays a scripted list of completions."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, req, timeout=None):  # urlopen(req, timeout=...)
        self.prompts.append(json.loads(req.data.decode())["prompt"])
        body = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]

        class _R:
            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def read(self):
                return json.dumps({"content": body, "stop_type": "eos"}).encode()

        return _R()


@pytest.fixture
def proposer(monkeypatch):
    p = e3.LocalGGUFProposer()
    monkeypatch.setattr(type(p), "_ensure_server", lambda _self: True)
    monkeypatch.setattr(type(p), "_url", lambda _self: "http://127.0.0.1:1")
    return p


def _drive(monkeypatch, proposer, replies):
    fake = _FakeHTTP(replies)
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    ok, code = proposer.generate(
        "PROMPT-BODY",
        ("engine", "is_level_complete"),
        tries=3,
        codeonly_eligible=True,
        engine_transitions=_trans(),
    )
    return ok, code, fake


@pytest.fixture(autouse=True)
def _json_load(monkeypatch):
    """The real code does `_json.load(r)`; our fake response exposes `.read()`."""
    real = json.load
    monkeypatch.setattr(json, "load", lambda r, *a, **k: json.loads(r.read()))
    yield
    monkeypatch.setattr(json, "load", real)


def test_flag_off_accepts_the_defective_goal_on_the_first_call(monkeypatch, proposer):
    """The shipped path: one call, a `return False` goal, accepted. This is the control arm,
    and it must be exactly one call or the A/B's compute comparison is wrong."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", raising=False)
    ok, code, fake = _drive(monkeypatch, proposer, [f"```python\n{GOOD_ENGINE}{BAD_GOAL}```"])
    assert ok
    assert "return False" in code
    assert len(fake.prompts) == 1
    assert proposer.n_goal_defect_reasks == 0


def test_flag_on_reasks_and_the_suffix_reaches_the_prompt(monkeypatch, proposer):
    """THE WIRING CLAIM. Not 'the detector can detect' but 'generate() acts on it': a second
    call goes out, and it carries the goal re-ask block."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    ok, code, fake = _drive(
        monkeypatch,
        proposer,
        [
            f"```python\n{GOOD_ENGINE}{BAD_GOAL}```",
            f"```python\n{GOOD_ENGINE}{GOOD_GOAL}```",
        ],
    )
    assert ok
    assert "grid[0, 0] == 3" in code, "the re-asked, non-constant goal should be what is kept"
    assert len(fake.prompts) == 2, "exactly one re-ask should have gone out"
    assert "RETURNED THE SAME ANSWER" in fake.prompts[1], "the goal re-ask block is missing"
    assert "RETURNED THE SAME ANSWER" not in fake.prompts[0]
    assert proposer.n_goal_defect_reasks == 1


def test_budget_is_exhausted_then_the_candidate_is_accepted(monkeypatch, proposer):
    """NEVER FAILS WHERE THE SHIPPED PATH SUCCEEDED. With a persistently defective goal the
    budget runs out and the last candidate is ACCEPTED -- the re-ask can only ever cost samples,
    never convert an accept into a hard failure."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    ok, code, fake = _drive(monkeypatch, proposer, [f"```python\n{GOOD_ENGINE}{BAD_GOAL}```"])
    assert ok, "a spent budget must still accept, exactly as the shipped path does"
    assert "return False" in code
    assert len(fake.prompts) == 3, "2 re-asks (the budget) then accept, inside tries=3"
    assert proposer.n_goal_defect_reasks == 2


def test_goal_reask_does_not_spend_the_engine_budget(monkeypatch, proposer):
    """THE CONFOUND GUARD, end to end. If the two shared a counter, the treatment arm's ENGINES
    would silently lose their re-ask and the measured arm difference would be part goal-check
    and part engine-check-removal."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    _ok, _code, _fake = _drive(monkeypatch, proposer, [f"```python\n{GOOD_ENGINE}{BAD_GOAL}```"])
    assert proposer.n_goal_defect_reasks == 2
    assert proposer.n_induce_defect_reasks == 0, "the engine's budget must be untouched"


def test_a_clean_goal_is_not_reasked(monkeypatch, proposer):
    """Selectivity, such as it is: a goal that discriminates on the observed frames is accepted
    on the first call even with the flag on."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    ok, _code, fake = _drive(monkeypatch, proposer, [f"```python\n{GOOD_ENGINE}{GOOD_GOAL}```"])
    assert ok
    assert len(fake.prompts) == 1
    assert proposer.n_goal_defect_reasks == 0


def test_engine_defect_is_reported_before_the_goal_defect(monkeypatch, proposer):
    """Ordering is load-bearing on the combined call: one answer carries both functions, so a
    candidate whose ENGINE is broken must be re-asked as an engine defect and never relabelled
    a goal defect."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    broken_engine = "def engine(grid, action, data):\n    pass\n"
    _ok, _code, fake = _drive(monkeypatch, proposer, [f"```python\n{broken_engine}{BAD_GOAL}```"])
    assert proposer.n_induce_defect_reasks >= 1, "the engine defect should have fired first"
    # Match on a span that cannot be broken by the block's own line wrapping. An earlier draft
    # asserted "SAME SHAPE", which is real text but is wrapped as "SAME\nSHAPE" -- the test
    # failed while the behaviour was correct.
    assert "WAS RUN AGAINST THE OBSERVED TRANSITIONS" in fake.prompts[1], "engine block missing"
    assert "RETURNED THE SAME ANSWER" not in fake.prompts[1], "goal block must not shadow it"


def test_gap_filler_calls_are_untouched(monkeypatch, proposer):
    """`codeonly_eligible=False` is the refactor / gap-filler shape. It must not acquire a goal
    check: those are different artifacts with different contracts."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    fake = _FakeHTTP([f"```python\n{BAD_GOAL}```"])
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    ok, _code = proposer.generate(
        "PROMPT", ("is_level_complete",), tries=3, codeonly_eligible=False
    )
    assert ok
    assert len(fake.prompts) == 1
    assert proposer.n_goal_defect_reasks == 0


def test_goal_only_call_shape_is_reachable_by_the_check(monkeypatch, proposer):
    """THE ASYMMETRY THIS WHOLE CHANGE EXISTS TO CLOSE. `_split_induce`'s focused goal call
    passes `required=("is_level_complete",)`, which the ENGINE gate -- keyed on `"engine" in
    required` -- can never match. If this regresses, the goal-only path silently reverts to
    unchecked and the fix covers only the combined call."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    fake = _FakeHTTP([f"```python\n{BAD_GOAL}```"])
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    ok, _code = proposer.generate(
        "PROMPT",
        ("is_level_complete",),
        tries=3,
        codeonly_eligible=True,
        engine_transitions=_trans(),
    )
    assert ok
    assert len(fake.prompts) == 3, "the goal-only call shape must be checked and re-asked"
    assert proposer.n_goal_defect_reasks == 2
