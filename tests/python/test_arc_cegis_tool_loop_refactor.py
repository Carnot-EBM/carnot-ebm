"""REQ-ARC-WMTE-6480 / SCENARIO-ARC-WMTE-6480-TOOL-ROUTED-REFINEMENT.

WHAT THIS PINS. `execute_bounded_llm_reinduction`'s refactor rounds can route through the
induction tool loop's repair mode under `CARNOT_ARC_CEGIS_TOOL_LOOP=1`. The shipped text
refactor is measured-destructive (exp5760 + exp5766: 0 of 84 refinement cells improved
held-out accuracy; all 8 partially-correct engines collapsed to 0.0), and REQ-ARC-WMTE-6091
traced the cause: the refactor prompt never contains the engine being fixed. The tool loop's
repair mode shows the model the failed engine WITH its measured mismatch report and lets it
execute candidate repairs before submitting.

THE CONTRACT, in order of importance:

  1. FLAG UNSET -> byte-identical shipped behaviour. The tool-loop module is never even
     imported (pinned with a poisoned sys.modules sentinel).
  2. FLAG ON -> refactor rounds call the tool loop, seeded with the engine currently on
     disk; the shipped text refactor is skipped for a round the tool loop completes.
  3. ANY tool-loop failure (returns False, proposer lacks the transport surface) -> the
     shipped text refactor still runs. Worst case is today's behaviour.
  4. The tool loop receives the REFINEMENT corpus -- with the acceptance split on, the
     reserved grading rows are withheld, same purity rule as the text path's
     counterexamples.

EVIDENCE SAFETY: every test redirects the engine store to `tmp_path`. Nothing here can
write `results/arc_e3`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_llm_reinduction import (
    _tool_loop_refactor,
    cegis_tool_loop_enabled,
    execute_bounded_llm_reinduction,
)

GAME = "ctlr"

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
# Same shape as the retention tests: the held-out tail must contain an action-3 tick so the
# good engine fails the live 1.0 gate and the loop actually reaches a refactor round.
ACTIONS = [1, 3, 0, 2, 1, 3, 0, 2, 3]


def _true_next(grid: np.ndarray, action: int) -> np.ndarray:
    g = grid.copy()
    pos = np.argwhere(g == 3)
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    if int(action) % 4 == 3:
        g[5, 5] = int(g[5, 5]) + 1
    return g


def _corpus() -> tuple[list[Transition], np.ndarray]:
    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows: list[Transition] = []
    for action in ACTIONS:
        nxt = _true_next(grid, action)
        rows.append(
            Transition(
                grid=grid.copy(),
                action=action,
                data=None,
                next_grid=nxt.copy(),
                level_before=0,
                level_after=0,
            )
        )
        grid = nxt
    return rows, root


# Models the move, misses the action-3 counter tick: held-out accuracy < 1.0, so the loop
# refactors. Identical role to the retention tests' GOOD_SRC.
GOOD_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    return False
"""

# A distinguishable second source, so a test can tell WHO wrote the store for round 2.
NOOP_SRC = """
import numpy as np


def engine(grid, action, data):
    return np.asarray(grid).copy()


def is_level_complete(grid):
    return False
"""


class _ScriptedProposer:
    """Writes a fixed source per round, like the retention tests' proposer.

    Deliberately has NONE of the tool loop's transport surface (`_url`, `_ensure_server`,
    ...), which is exactly the shape of every experiment-script proposer.
    """

    model_specs = "scripted-cegis-tool-loop-test-proposer"

    def __init__(self, store: Path, game: str, sources: list[str]) -> None:
        self.store = Path(store)
        self.game = game
        self.sources = list(sources)
        self.refactor_calls = 0
        self.writes: list[int] = []

    def _write(self, index: int) -> tuple[bool, str]:
        src = self.sources[min(index, len(self.sources) - 1)]
        path = self.store / self.game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src)
        self.writes.append(index)
        return True, f"wrote round {index + 1}"

    def induce(self, game, trans, cell, *, previous_level_complete_grid=None):
        return self._write(0)

    def refactor(self, game, vr):
        self.refactor_calls += 1
        return self._write(len(self.writes))


class _TransportProposer(_ScriptedProposer):
    """Same scripted proposer, plus the attribute surface `_tool_loop_refactor` checks for.

    The attributes are never CALLED in these tests (the loop itself is monkeypatched); they
    exist so the capability check passes.
    """

    max_tokens = 4096
    timeout = 60

    def _url(self) -> str:  # pragma: no cover - presence only
        return "http://127.0.0.1:1"

    def _ensure_server(self) -> bool:  # pragma: no cover - presence only
        return True

    def _write_world_model(self, game, code, note=""):  # pragma: no cover - presence only
        return True, note


class _PoisonedToolLoopModule:
    """sys.modules sentinel: ANY attribute access is proof the shipped path imported the
    tool loop, which the flag-off contract forbids."""

    def __getattr__(self, name):  # noqa: D105
        raise AssertionError(f"tool loop module touched (attribute {name!r}) with flag off")


def _run(monkeypatch, tmp_path, proposer):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    # The shipped MAX_REFINEMENT_ROUNDS default was capped 3 -> 1 on 2026-08-17
    # (operator-approved; see _max_refinement_rounds_default). These tests exercise the
    # multi-round refactor contract, which needs rounds to exist. The loop reads the
    # module global at call time, so patching it here raises the cap for this test only.
    import carnot.agentic.arc_llm_reinduction as _reind

    monkeypatch.setattr(_reind, "MAX_REFINEMENT_ROUNDS", 3)
    transitions, root = _corpus()
    result = execute_bounded_llm_reinduction(
        game=GAME,
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=lambda engine, goal, grid: None,
        max_rounds=3,
        min_heldout_accuracy=1.0,
    )
    return result, transitions


def test_flag_default_off() -> None:
    """The flag helper reads the env var and nothing else; unset means OFF."""

    assert (
        cegis_tool_loop_enabled() is False
        or __import__("os").environ.get("CARNOT_ARC_CEGIS_TOOL_LOOP") == "1"
    )


def test_flag_off_is_inert_and_never_imports_the_tool_loop(monkeypatch, tmp_path) -> None:
    """CONTRACT 1. MUTATION PROOF: make `cegis_tool_loop_enabled` return True and the
    poisoned sentinel raises inside the first refactor round."""

    monkeypatch.delenv("CARNOT_ARC_CEGIS_TOOL_LOOP", raising=False)
    monkeypatch.setitem(
        sys.modules, "carnot.agentic.arc_induction_tool_loop", _PoisonedToolLoopModule()
    )
    proposer = _ScriptedProposer(tmp_path, GAME, [GOOD_SRC, NOOP_SRC, NOOP_SRC])
    result, _ = _run(monkeypatch, tmp_path, proposer)

    assert proposer.refactor_calls == 2, "both refactor rounds must use the shipped path"
    actions = [r.get("action") for r in result.rounds]
    assert actions == ["induce", "refactor", "refactor"]
    assert all("tool_loop" not in r for r in result.rounds)


def test_flag_on_routes_refactor_through_tool_loop_with_seed(monkeypatch, tmp_path) -> None:
    """CONTRACT 2. The spy stands in for the real loop: it must receive the engine
    currently on disk as the seed, and a completed tool round must skip the text refactor."""

    monkeypatch.setenv("CARNOT_ARC_CEGIS_TOOL_LOOP", "1")
    calls: list[dict] = []

    def spy(
        proposer,
        game,
        trans,
        cell,
        *,
        previous_level_complete_grid=None,
        win_transition=None,
        seed_engine_code=None,
        hud_mask=None,
    ):
        calls.append(
            {
                "game": game,
                "n_trans": len(trans),
                "cell": cell,
                "seed": seed_engine_code,
            }
        )
        # Simulate a successful repair write, like the real loop's _write_world_model.
        path = tmp_path / game / "world_model.py"
        path.write_text(NOOP_SRC)
        proposer.last_tool_loop_stats = {"turns": 1, "tool_calls_total": 2}
        return True, "spy repair"

    import carnot.agentic.arc_induction_tool_loop as tl

    monkeypatch.setattr(tl, "induce_with_tool_loop", spy)
    proposer = _TransportProposer(tmp_path, GAME, [GOOD_SRC])
    result, transitions = _run(monkeypatch, tmp_path, proposer)

    assert len(calls) == 2, "both refactor rounds must route through the tool loop"
    assert calls[0]["seed"] == GOOD_SRC, "round 2 must be seeded with round 1's engine"
    assert calls[1]["seed"] == NOOP_SRC, "round 3 must be seeded with round 2's write"
    assert calls[0]["n_trans"] == len(transitions), "accept split off -> full corpus"
    assert proposer.refactor_calls == 0, "a completed tool round must skip the text refactor"
    actions = [r.get("action") for r in result.rounds]
    assert actions == ["induce", "refactor_tool_loop", "refactor_tool_loop"]
    assert result.rounds[1]["tool_loop"]["turns"] == 1, "stats must land on the round row"


def test_tool_loop_failure_falls_back_to_text_refactor(monkeypatch, tmp_path) -> None:
    """CONTRACT 3a. A loop that produced nothing scoreable must not cost the round."""

    monkeypatch.setenv("CARNOT_ARC_CEGIS_TOOL_LOOP", "1")

    def failing_spy(proposer, game, trans, cell, **kwargs):
        proposer.last_tool_loop_stats = {"terminated_by": "transport_error"}
        return False, "tool loop: no scoreable engine (transport_error)"

    import carnot.agentic.arc_induction_tool_loop as tl

    monkeypatch.setattr(tl, "induce_with_tool_loop", failing_spy)
    proposer = _TransportProposer(tmp_path, GAME, [GOOD_SRC, NOOP_SRC, NOOP_SRC])
    result, _ = _run(monkeypatch, tmp_path, proposer)

    assert proposer.refactor_calls == 2, "text refactor must run when the tool loop fails"
    actions = [r.get("action") for r in result.rounds]
    assert actions == ["induce", "refactor", "refactor"]
    # The attempt is still visible in the record, so a null can be attributed.
    assert result.rounds[1]["tool_loop"]["terminated_by"] == "transport_error"


def test_proposer_without_transport_surface_falls_back(monkeypatch, tmp_path) -> None:
    """CONTRACT 3b. An experiment-script proposer (no `_url`) must degrade to the shipped
    path WITHOUT importing the tool loop -- the capability check runs before the import."""

    monkeypatch.setenv("CARNOT_ARC_CEGIS_TOOL_LOOP", "1")
    monkeypatch.setitem(
        sys.modules, "carnot.agentic.arc_induction_tool_loop", _PoisonedToolLoopModule()
    )
    proposer = _ScriptedProposer(tmp_path, GAME, [GOOD_SRC, NOOP_SRC, NOOP_SRC])
    result, _ = _run(monkeypatch, tmp_path, proposer)

    assert proposer.refactor_calls == 2
    actions = [r.get("action") for r in result.rounds]
    assert actions == ["induce", "refactor", "refactor"]


def test_accept_split_withholds_grading_rows_from_the_tool_loop(monkeypatch, tmp_path) -> None:
    """CONTRACT 4. With CARNOT_ARC_CEGIS_ACCEPT_SPLIT=1 the tool loop sees the refinable
    rows only -- the same purity rule the text path's counterexamples follow."""

    from carnot.agentic.arc_world_model_trust_energy import split_refinement_acceptance

    monkeypatch.setenv("CARNOT_ARC_CEGIS_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
    seen: list[int] = []

    def spy(proposer, game, trans, cell, **kwargs):
        seen.append(len(trans))
        (tmp_path / game / "world_model.py").write_text(NOOP_SRC)
        proposer.last_tool_loop_stats = {"turns": 1}
        return True, "spy repair"

    import carnot.agentic.arc_induction_tool_loop as tl

    monkeypatch.setattr(tl, "induce_with_tool_loop", spy)
    proposer = _TransportProposer(tmp_path, GAME, [GOOD_SRC])
    _, transitions = _run(monkeypatch, tmp_path, proposer)

    expected = len(split_refinement_acceptance(list(transitions)).refinable)
    assert seen, "the tool loop must have been called"
    assert seen[0] == expected, "the reserved acceptance rows must be withheld"
    assert seen[0] < len(transitions)


def test_helper_returns_none_without_seed(monkeypatch, tmp_path) -> None:
    """No engine on disk -> nothing to seed a repair with -> fall back (None), and the
    poisoned sentinel proves the import was never reached."""

    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    monkeypatch.setitem(
        sys.modules, "carnot.agentic.arc_induction_tool_loop", _PoisonedToolLoopModule()
    )
    proposer = _TransportProposer(tmp_path, GAME, [GOOD_SRC])
    assert _tool_loop_refactor(proposer, GAME, [], 1) is None
