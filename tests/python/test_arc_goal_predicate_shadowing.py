"""The split-induce path must not ship two definitions of `is_level_complete`.

THE DEFECT. `LocalGGUFProposer.induce` falls back to two focused generations when the
combined engine+goal call fails: an engine-only call whose prompt (`induce_prompt`)
carries the observed transitions and the opening grid, then a goal-only call whose
prompt (`_goal_only_prompt`) by default carries neither. `_combine_world_model`
concatenates them. The model routinely writes an `is_level_complete` in the ENGINE-only
response too -- the base prompt describes the whole interface -- so the file ends up
with two top-level definitions and Python binds the SECOND, the evidence-free one.

Measured 2026-08-02 over the frozen corpus (`results/arc_goal_predicate_shadowing_20260802/`):
23 of 116 concatenated world models carry two definitions, against 0 of 40 raw
single-call completions -- the duplication comes from the concatenation, not from the
model redefining the function.

WHAT THE MEASUREMENT DID NOT ESTABLISH, and why these tests are scoped the way they
are. Grading BOTH definitions of all 23 cells through the shipped goal gate does NOT
show the shadowed one is better: 2 satisfiable against 1, with 20 of 23 cells tied at
unsatisfiable. So nothing here asserts the preserved predicate is a better goal. What
it DID establish is one-sided VALIDITY: 4 of 23 bound definitions are not usable
predicates at all (two return `None`, one raises `NameError`, one does not terminate),
against 0 of 23 shadowed. These tests therefore pin STRUCTURE -- how many definitions
reach the file, and which one runs -- and claim nothing about goal quality.

Spec refs: REQ-ARC-WMTE-6044 (the induced `is_level_complete` contract) and
SCENARIO-ARC-WMTE-6044-WIN-STATE-IS-A-TRANSITION-NOT-A-FRAME, whose corrected
`_goal_only_prompt` is the evidence-free half of the pair this file is about. No new
REQ is added here: the capability spec has unrelated uncommitted work in the tree at
time of writing, and appending to a file another session is editing is how a merge
silently eats someone's paragraph.

`test_shipped_split_induce_binds_the_second_definition` documents CURRENT shipped
behaviour and passes before and after the fix -- that is the binding-order pin. The
`dedup_on` tests FAIL against the pre-fix code, where the flag and the excision do not
exist.
"""

from __future__ import annotations

import ast
import json
import urllib.request

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as awm
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, Transition

pytestmark = pytest.mark.memory_watchdog_skip

# The engine half as the corpus actually shows it: an engine PLUS a grounded predicate that
# names a concrete row and colour, which is only sayable because this prompt carried the board.
_ENGINE_WITH_GROUNDED_GOAL = (
    "import numpy as np\n"
    "def engine(grid, action, data):\n"
    "    return np.asarray(grid)\n"
    "def is_level_complete(grid):\n"
    "    return bool(np.all(grid[63] == 4))\n"
)
# The engine half when the model declined -- 11 of the 23 corpus cells look like this.
_ENGINE_WITH_DECLINED_GOAL = (
    "import numpy as np\n"
    "def engine(grid, action, data):\n"
    "    return np.asarray(grid)\n"
    "def is_level_complete(grid):\n"
    "    # No win state was provided, so we cannot induce one.\n"
    "    return False\n"
)
_ENGINE_ONLY = "import numpy as np\ndef engine(grid, action, data):\n    return np.asarray(grid)\n"
# The goal-only half: the whole-board uniformity trope, 8 of 23 corpus cells.
_GOAL_TROPE = (
    "import numpy as np\n"
    "def is_level_complete(grid):\n"
    "    return bool(np.all(grid == grid[0, 0]))\n"
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


def _proposer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> LocalGGUFProposer:
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    # E3_DIR is rebound to tmp_path so nothing here can reach the tracked evidence store at
    # results/arc_e3/. `_guard_engine_write` only fires under pytest, and relying on it rather
    # than redirecting would leave this test one refactor away from rewriting committed evidence.
    monkeypatch.setattr(awm, "E3_DIR", tmp_path)
    p = LocalGGUFProposer(
        repo_substr="X",
        model_path="/x.gguf",
        port=59997,
        no_think_prefix="/no_think\n",
        max_tokens=128,
        tries=1,
    )
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    return p


def _seq_urlopen(monkeypatch: pytest.MonkeyPatch, contents: list[str]) -> list:
    bodies: list = []
    it = iter(contents)

    def fake(req, timeout=None):  # noqa: ANN001
        bodies.append(json.loads(req.data.decode()))
        content = next(it)
        return _FakeResp(
            json.dumps(
                {
                    "content": content,
                    "stop_type": "eos",
                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return bodies


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


def _goal_defs(code: str) -> list[ast.FunctionDef]:
    return [
        n
        for n in ast.parse(code).body
        if isinstance(n, ast.FunctionDef) and n.name == "is_level_complete"
    ]


def _bound_predicate(code: str):
    """Exec the module the way the live loader does and hand back what `is_level_complete` binds to."""
    ns: dict = {"__name__": "wm_under_test"}
    exec(compile(code, "<wm>", "exec"), ns)  # noqa: S102
    return ns["is_level_complete"]


def test_shipped_split_induce_binds_the_second_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """DEFAULT (flag off): two definitions reach the file and the GOAL-ONLY one wins.

    This is the binding-order pin. It documents the defect rather than the fix, so it passes
    both before and after -- if a later change makes the file bind the first definition
    instead, that is a semantic change to every world model on this path and this test is
    where it should surface.
    """
    monkeypatch.delenv("CARNOT_ARC_GOAL_DEDUP", raising=False)
    p = _proposer(monkeypatch, tmp_path)
    _seq_urlopen(monkeypatch, [_ENGINE_ONLY, _ENGINE_WITH_GROUNDED_GOAL, _GOAL_TROPE])
    ok, msg = p.induce("g", _trans(), 1)
    assert ok is True
    assert "split induce" in msg
    wm = (tmp_path / "g" / "world_model.py").read_text()

    assert len(_goal_defs(wm)) == 2, "the shipped path concatenates both definitions"
    # Python binds the last; the grounded predicate written next to the engine is dead code.
    board = np.full((64, 64), 7, dtype=int)
    board[63, :] = 4
    assert _bound_predicate(wm)(board) is False, "the trope is what runs"
    uniform = np.zeros((64, 64), dtype=int)
    assert _bound_predicate(wm)(uniform) is True, "and it is the uniformity trope specifically"


def test_dedup_on_emits_exactly_one_definition_and_keeps_the_grounded_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """FLAG ON: the engine half already answered usefully, so no second definition is generated.

    Fails against the pre-fix code, where `_goal_dedup_on` does not exist and the goal call is
    unconditional.
    """
    monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", "1")
    p = _proposer(monkeypatch, tmp_path)
    bodies = _seq_urlopen(monkeypatch, [_ENGINE_ONLY, _ENGINE_WITH_GROUNDED_GOAL, _GOAL_TROPE])
    ok, msg = p.induce("g", _trans(), 1)
    assert ok is True
    assert "dedup" in msg
    # combined + focused engine only. The goal call is never made, so it also costs one fewer
    # generation than the shipped path.
    assert len(bodies) == 2

    wm = (tmp_path / "g" / "world_model.py").read_text()
    assert len(_goal_defs(wm)) == 1, "exactly one definition, so binding order cannot matter"
    board = np.full((64, 64), 7, dtype=int)
    board[63, :] = 4
    assert _bound_predicate(wm)(board) is True, "the grounded predicate is what runs now"


def test_dedup_on_preserves_the_numpy_import_guarantee(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Skipping the join must not drop the `import numpy as np` prefix the join guarantees.

    `_combine_world_model` prepends that import on every split-induce write. The dedup path
    returns the engine half WITHOUT going through the join, so an engine completion that uses
    `np` but never imports it at top level would load under the shipped path and raise
    `NameError` under this one -- a regression the flag would have introduced silently.

    No completion in the frozen corpus actually omits the import (0 of 40 checked), so this
    guards an invariant the corpus does not currently exercise. That is the point: the next
    corpus might.
    """
    monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", "1")
    no_import = (
        "def engine(grid, action, data):\n"
        "    return np.asarray(grid)\n"
        "def is_level_complete(grid):\n"
        "    return bool(np.all(grid[63] == 4))\n"
    )
    p = _proposer(monkeypatch, tmp_path)
    _seq_urlopen(monkeypatch, [_ENGINE_ONLY, no_import, _GOAL_TROPE])
    ok, msg = p.induce("g", _trans(), 1)
    assert ok is True
    assert "dedup" in msg
    wm = (tmp_path / "g" / "world_model.py").read_text()
    assert len(_goal_defs(wm)) == 1
    board = np.full((64, 64), 7, dtype=int)
    board[63, :] = 4
    # Would raise NameError on `np` without the prepended import.
    assert _bound_predicate(wm)(board) is True


def test_dedup_on_still_generates_a_goal_when_the_engine_half_declined(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A constant-false engine half carries no information, so the focused goal call must still run.

    The dedup must not be "always keep the engine half": that would trade an evidence-free
    trope for an unconditional `return False`, which the goal gate rejects as degenerate
    anyway. The file must STILL define the function exactly once.
    """
    monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", "1")
    p = _proposer(monkeypatch, tmp_path)
    bodies = _seq_urlopen(monkeypatch, [_ENGINE_ONLY, _ENGINE_WITH_DECLINED_GOAL, _GOAL_TROPE])
    ok, _ = p.induce("g", _trans(), 1)
    assert ok is True
    assert len(bodies) == 3, "combined + engine + goal: the declined half does not short-circuit"

    wm = (tmp_path / "g" / "world_model.py").read_text()
    assert len(_goal_defs(wm)) == 1, "the declined definition is excised before the join"
    assert _bound_predicate(wm)(np.zeros((4, 4), dtype=int)) is True


def test_dedup_on_still_generates_a_goal_when_the_engine_half_is_defective(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A predicate whose body falls through returns None, which is not an answer.

    This is the corpus's own largest bound-side defect (two cells return `None`, one raises).
    Whichever half it appears in, it must not be treated as "the engine already answered".
    """
    monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", "1")
    falls_through = (
        "import numpy as np\n"
        "def engine(grid, action, data):\n"
        "    return np.asarray(grid)\n"
        "def is_level_complete(grid):\n"
        "    x = np.sum(grid)\n"  # no return on any path -> None
    )
    p = _proposer(monkeypatch, tmp_path)
    bodies = _seq_urlopen(monkeypatch, [_ENGINE_ONLY, falls_through, _GOAL_TROPE])
    ok, _ = p.induce("g", _trans(), 1)
    assert ok is True
    assert len(bodies) == 3
    wm = (tmp_path / "g" / "world_model.py").read_text()
    assert len(_goal_defs(wm)) == 1
    assert _bound_predicate(wm)(np.zeros((4, 4), dtype=int)) is True


def test_dedup_off_leaves_the_combined_output_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The control arm must be the SHIPPED path, not a re-rendering of it."""
    monkeypatch.delenv("CARNOT_ARC_GOAL_DEDUP", raising=False)
    p = _proposer(monkeypatch, tmp_path)
    off = p._combine_world_model(_ENGINE_WITH_GROUNDED_GOAL, _GOAL_TROPE)
    expected = (
        "import numpy as np\n\n"
        + _ENGINE_WITH_GROUNDED_GOAL.strip()
        + "\n\n"
        + _GOAL_TROPE.strip()
        + "\n"
    )
    assert off == expected
    assert len(_goal_defs(off)) == 2


def test_combine_world_model_dedups_even_when_called_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Belt as well as braces: no caller of `_combine_world_model` can produce a shadowed file.

    `induce` normally avoids the situation by not making the goal call, but this function is
    the last place both halves are in scope, and a future caller that assembles the halves
    differently must not be able to reintroduce the defect.
    """
    monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", "1")
    p = _proposer(monkeypatch, tmp_path)
    combined = p._combine_world_model(_ENGINE_WITH_GROUNDED_GOAL, _GOAL_TROPE)
    assert len(_goal_defs(combined)) == 1
    assert "def engine" in combined
    ast.parse(combined)


def test_strip_preserves_engine_comments_and_helpers() -> None:
    """The excision must not reformat the half it edits.

    The engine half carries the model's own comments and any helper it defined; a round-trip
    through `ast.unparse` would silently rewrite all of it, which would make every downstream
    sha256 and every human diff of a world model meaningless.
    """
    src = (
        "import numpy as np\n"
        "\n"
        "def _helper(g):\n"
        "    # a comment that must survive verbatim\n"
        "    return g\n"
        "\n"
        "def is_level_complete(grid):\n"
        "    return False\n"
        "\n"
        "def engine(grid, action, data):\n"
        "    # trailing comment\n"
        "    return _helper(grid)\n"
    )
    out = awm._strip_top_level_goal_defs(src)
    assert "is_level_complete" not in out
    assert "# a comment that must survive verbatim" in out
    assert "# trailing comment" in out
    assert "def _helper" in out and "def engine" in out
    ast.parse(out)


def test_nested_goal_definition_is_not_treated_as_a_competing_one() -> None:
    """`exec` never binds a nested def at module level, so it is not a second definition."""
    src = (
        "import numpy as np\n"
        "def engine(grid, action, data):\n"
        "    def is_level_complete(g):\n"
        "        return True\n"
        "    return np.asarray(grid)\n"
    )
    assert awm._strip_top_level_goal_defs(src) == src
    assert awm._engine_half_goal_usable(src) is False


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("def is_level_complete(g):\n    return False\n", True),
        ('def is_level_complete(g):\n    """doc"""\n    return False\n', True),
        ("def is_level_complete(g):\n    # note\n    return False\n", True),
        ("def is_level_complete(g):\n    return True\n", False),
        ("def is_level_complete(g):\n    return bool(g.sum())\n", False),
        (
            "def is_level_complete(g):\n    if g.sum():\n        return False\n    return True\n",
            False,
        ),
        ("def not_it(g):\n    return False\n", False),
        ("def is_level_complete(g):\n    return False\n(", False),  # unparseable
    ],
)
def test_constant_false_detection(code: str, expected: bool) -> None:
    assert awm._goal_predicate_is_constant_false(code) is expected


def test_flag_defaults_off_and_rejects_malformed_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd env var must not change how the scored agent behaves."""
    monkeypatch.delenv("CARNOT_ARC_GOAL_DEDUP", raising=False)
    assert awm._goal_dedup_on() is False
    for bad in ("", "0", "true", "yes", "2", " "):
        monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", bad)
        assert awm._goal_dedup_on() is False, bad
    monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", "1")
    assert awm._goal_dedup_on() is True
