"""REQ-ARC-WMTE-6710, the invariants its own test file does not pin.

WHY A SECOND FILE. `tests/python/test_arc_induction_diagnosis_6710.py` pins the three defects
REQ-ARC-WMTE-6710 was written to fix: a diagnosed cause reaching the caller, the per-round record
surviving into the harness row, and the channel split accumulating. This file pins what a FIX
CAN BREAK ON ITS WAY PAST -- the properties that were true before the REQ and must stay true.
The two files were built in parallel by two agents against the same brief; rather than ship a
duplicate, the overlapping cases were dropped and only the non-overlapping ones kept.

The invariants, and what each would catch:

  1. THE SUCCESS PATH. The widening writes an outer `skipped` variable that a planned return must
     still clear. Nothing in the sibling file runs a planning attempt to the end, so a change that
     made a successful induction report a skip would pass the whole 6710 suite.
  2. LAST-DIAGNOSED-CAUSE-WINS. The two widened sites overwrite each round. A future edit that
     made the field sticky, or first-wins, would silently report round 1's cause for a run whose
     last round failed differently.
  3. WHY BOTH RECORD VIEWS ARE KEPT. The exception site appends to the attempt's `counterexamples`
     and never writes `row["counterexample"]`, so the round view is BLANK for exactly the failure
     a reader most needs. Keeping only `rounds` rebuilds the original blind spot one level down.
  4. THE DATACLASS DEFAULT. The sibling file builds its proposer with
     `LocalGGUFProposer.__new__`, which bypasses `__init__` and therefore never exercises
     `field(default_factory=...)`. If that field were ever written as a bare `{}` every proposer
     in the process would share ONE counter dict and every per-cell number would be wrong, with
     the whole 6710 suite still green.
  5. NO FORGED CHANNEL SPLIT. The raw `/completion` endpoint has no `reasoning_content`, so the
     split is accumulated inside `_chat_complete_request` and NOT at the
     `_record_completion_diagnostics` seam. A refactor that "simplified" by moving it to the seam
     would attribute the previous CHAT call's channels to a request that never had them.
  6. MONOTONICITY. Differencing two reads is only valid on counters that never decrease.
  7. INSTRUMENTATION MUST NOT FAIL A CELL. A malformed attempt must not raise out of the digest.

WHY THE FIX EXISTS, IN ONE SENTENCE. Because the masking label is ALSO the correct label for a
genuine no-plan, the `induction_skipped` counter can never size its own mislabelling -- which is
the defect restated, and it means no prevalence figure taken from that field is trustworthy until
the per-round record lands.

A PREVALENCE FIGURE THAT DID NOT SURVIVE RE-DERIVATION. The brief this work was built from cited
`no_reachable_plan_after_refinement` as 9 of 18 skip records. Measured over `results/**` the count
is 7 of 56 non-`disabled_by_env` records; no file or window reproduces 9-of-18. Recorded here
rather than dropped, because the sentence above is the reason the number could not be checked
against anything: an unknown share of those 7 are held-out dynamics failures wearing a planning
label, and the field cannot say which.

EVIDENCE SAFETY: the tests that run the loop redirect the engine store to `tmp_path`. Nothing
here writes `results/**`. No network, no GPU: the chat transport is stubbed at `urlopen`.

Spec refs: REQ-ARC-WMTE-6710.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_llm_reinduction as reinduction
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, Transition
from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

GAME = "diag6710inv"
_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# =============================================================================================
# Loop fixtures. A corpus where every row genuinely changes state, so an identity engine scores
# zero and the held-out gate rejects it, while the perfect engine below is accepted.
# =============================================================================================


def _true_next(grid: np.ndarray, action: int) -> np.ndarray:
    g = grid.copy()
    pos = np.argwhere(g == 3)
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def _corpus(n: int = 12) -> tuple[list[Transition], np.ndarray]:
    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows: list[Transition] = []
    for i in range(n):
        action = i % 4
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


IDENTITY_SRC = """
import numpy as np


def engine(grid, action, data):
    return np.asarray(grid).copy()


def is_level_complete(grid):
    return False
"""

# Predicts the corpus exactly, so the held-out gate ACCEPTS it. Its goal is the token at row 0,
# column 2 -- two UP moves from the root, so a planner reaches it and the goal gate can prove it.
PERFECT_SRC = """
import numpy as np

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    return bool(np.asarray(grid)[0, 2] == 3)
"""

_REACHING_PLAN = [{"action": 0, "data": None}, {"action": 0, "data": None}]


class _ScriptedProposer:
    """Writes a fixed engine source into the redirected store on every induce/refactor."""

    model_specs = "scripted-6710-invariants-test-proposer"

    def __init__(self, store: Path, source: str) -> None:
        self.store = Path(store)
        self.source = source
        self.writes = 0

    def _write(self) -> tuple[bool, str]:
        path = self.store / GAME / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source)
        self.writes += 1
        return True, f"wrote {self.writes}"

    def induce(self, game, trans, cell, *, previous_level_complete_grid=None):
        return self._write()

    def refactor(self, game, vr):
        return self._write()


def _run_loop(
    monkeypatch,
    tmp_path: Path,
    source: str,
    *,
    min_heldout_accuracy: float,
    plan_in_model,
    rounds: int = 1,
):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    monkeypatch.setattr(reinduction, "MAX_REFINEMENT_ROUNDS", rounds)
    rows, root = _corpus()
    return execute_bounded_llm_reinduction(
        game=GAME,
        transitions=rows,
        cell=1,
        root_grid=root,
        proposer=_ScriptedProposer(tmp_path, source),
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=plan_in_model,
        max_rounds=rounds,
        min_heldout_accuracy=min_heldout_accuracy,
    )


# =============================================================================================
# INVARIANT 1 and 2 -- the widened `skipped` variable must not disturb the paths around it.
# =============================================================================================


def test_a_planned_return_still_clears_the_skip_field(monkeypatch, tmp_path):
    """INVARIANT 1. A run that PLANS must report no skip at all.

    The three planned returns pass `skipped=""` as a literal, so the widening cannot reach them.
    That is an argument, not a measurement -- this is the measurement. A fix that reported a
    stale cause on a SUCCESSFUL induction would be a worse bug than the one being fixed, because
    it would make every working run look broken.
    """

    result = _run_loop(
        monkeypatch,
        tmp_path,
        PERFECT_SRC,
        min_heldout_accuracy=0.5,
        plan_in_model=lambda engine, goal, grid: list(_REACHING_PLAN),
    )

    assert result.planned is True, "the scenario must actually plan -- otherwise it proves nothing"
    assert result.skipped == ""


def test_the_last_diagnosed_cause_wins_across_rounds(monkeypatch, tmp_path):
    """INVARIANT 2. Each round overwrites the field; the report is the LAST round's cause.

    This is the shipped semantics of the two sites that already wrote the outer variable before
    REQ-ARC-WMTE-6710. The widening had to match it, not invent first-wins or sticky.
    """

    result = _run_loop(
        monkeypatch,
        tmp_path,
        IDENTITY_SRC,
        min_heldout_accuracy=1.0,
        plan_in_model=lambda engine, goal, grid: None,
        rounds=3,
    )

    per_round = [r.get("skipped") for r in result.rounds]
    assert len(per_round) >= 2, "this test needs more than one round to say anything"
    assert result.skipped == per_round[-1]


# =============================================================================================
# INVARIANT 3 and 7 -- the harness digest.
# =============================================================================================

_HARNESS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "arc_scored_path_lever_harness.py"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    """Load the harness by path -- it lives in scripts/, not an installed package."""

    spec = importlib.util.spec_from_file_location("arc_scored_path_lever_harness", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_exception_counterexample_survives_although_no_round_carries_it(harness):
    """INVARIANT 3. The reason BOTH record views are kept, stated as a test.

    The exception site appends to the attempt's `counterexamples` but never writes
    `row["counterexample"]`. So the round view is genuinely blank here. A future simplification
    that kept only `rounds` would drop the traceback evidence for the one failure class that
    cannot be diagnosed any other way -- and every other test would stay green.
    """

    attempt = {
        "reason": "stagnation",
        "skipped": "selection_or_planning_exception",
        "refinement_rounds": [{"round": 1, "skipped": "selection_or_planning_exception"}],
        "counterexamples": [
            {"kind": "selection_or_planning_exception", "error": "RuntimeError('boom')"}
        ],
    }

    records = harness._induction_round_records([attempt])

    assert records[0]["rounds"][0]["counterexample"] == {}, (
        "the round view is empty for a raising round -- that is the gap the attempt view fills"
    )
    assert records[0]["counterexamples"][0]["error"] == "RuntimeError('boom')"


def test_a_long_string_in_a_counterexample_is_capped(harness):
    """INVARIANT 7 (bounds). A traceback repr is unbounded and the row is written once per cell.

    The sibling suite pins that LISTS become lengths. Strings are the other unbounded field, and
    an uncapped `error` from a deep traceback can be far larger than the mismatch list it
    replaced.
    """

    digest = harness._counterexample_digest({"kind": "x", "error": "E" * 5000})

    assert len(digest["error"]) == harness._DIGEST_STR_CAP
    assert digest["error"].startswith("E")


def test_the_digest_tolerates_malformed_input(harness):
    """INVARIANT 7. Instrumentation must never be the thing that fails a cell.

    These records are built from whatever the loop produced, on a path that already went wrong.
    A digest that raised on an unexpected shape would turn a diagnosable failure into a crashed
    cell -- losing the row that was supposed to explain it.
    """

    assert harness._induction_round_records([None, "nonsense", 7]) == []
    assert harness._round_digest("not a list") == []
    assert harness._counterexample_digest(None) == {}


def test_an_attempt_with_no_evidence_adds_no_entry(harness):
    """An attempt short-circuited before induction carries nothing to diagnose, so it adds no
    entry. The category counters already record that it happened."""

    assert (
        harness._induction_round_records([{"reason": "stagnation", "skipped": "disabled_by_env"}])
        == []
    )


# =============================================================================================
# INVARIANT 4, 5 and 6 -- the proposer counters.
# =============================================================================================


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _chat_payload(final: str, reasoning: str) -> dict:
    return {
        "choices": [
            {
                "message": {"content": final, "reasoning_content": reasoning},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 123},
    }


@pytest.fixture()
def chat_proposer(monkeypatch):
    """A FULLY CONSTRUCTED proposer -- built through `__init__`, not `__new__`.

    That difference is the point of `test_every_proposer_instance_owns_its_own_counters` below:
    only a real construction exercises the dataclass default factory.
    """

    p = LocalGGUFProposer()
    p.max_tokens = 4096
    p.use_chat_template = True

    def _install(payload: dict) -> None:
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload), raising=True
        )

    p._install_payload = _install  # type: ignore[attr-defined]
    return p


def _drive_chat(p: LocalGGUFProposer) -> None:
    resp, _extraction = p._chat_complete_request(
        "prompt", max_tokens=p.max_tokens, temperature=0.2, stop=None
    )
    p._record_completion_diagnostics(resp)


def test_every_proposer_instance_owns_its_own_counters(chat_proposer):
    """INVARIANT 4. The classic shared-mutable-default trap, pinned.

    `channel_totals` is declared `field(default_factory=lambda: dict(_EMPTY_CHANNEL_TOTALS))`.
    Written as a bare default, or as a factory returning the module constant ITSELF rather than a
    copy, every proposer in the process would share one dict. Per-cell differencing would then
    mix cells together and no existing test would notice, because the sibling suite constructs
    its proposer with `__new__` and assigns the field by hand.
    """

    chat_proposer._install_payload(_chat_payload(final="answer", reasoning="think"))
    _drive_chat(chat_proposer)

    other = LocalGGUFProposer()
    assert other.channel_totals["completions"] == 0, "a fresh proposer must start at zero"
    assert chat_proposer.channel_totals["completions"] == 1
    assert e3._EMPTY_CHANNEL_TOTALS["completions"] == 0, (
        "the module constant is a TEMPLATE -- writing through it would poison every later instance"
    )


def test_a_raw_completion_response_does_not_forge_a_channel_split(chat_proposer):
    """INVARIANT 5. The staleness hazard, and the reason the split lives where it lives.

    `_record_completion_diagnostics` is reached by BOTH endpoints, but only the chat endpoint has
    a `reasoning_content` field. Accumulating the split there would read whatever the last CHAT
    call left in `last_final_content` / `last_reasoning_content` and count it again for a raw
    `/completion` request that never had those channels -- a stale read published as a
    measurement. The split therefore accumulates inside `_chat_complete_request`, and this test
    is what stops a later refactor from moving it.
    """

    chat_proposer._install_payload(_chat_payload(final="answer", reasoning="think"))
    _drive_chat(chat_proposer)
    before = dict(chat_proposer.channel_totals)

    chat_proposer._record_completion_diagnostics({"content": "raw text", "stop_type": "eos"})

    after = chat_proposer.channel_totals
    assert after["completions"] == before["completions"] + 1, "the raw call is still a completion"
    assert after["chars_raw"] == before["chars_raw"] + len("raw text")
    assert after["chat_completions"] == before["chat_completions"], (
        "a raw /completion request must not be counted as a chat request"
    )
    assert after["chars_final"] == before["chars_final"]
    assert after["chars_reasoning"] == before["chars_reasoning"]


def test_the_counters_never_go_backwards(chat_proposer):
    """INVARIANT 6. Differencing is only valid on monotone counters.

    A row's numbers are `after - before`. If any counter could decrease -- a reset, a reassignment,
    a per-call overwrite instead of an increment -- a cell could publish a negative character
    count, and nothing downstream would know which of the two reads was wrong.
    """

    seen = [dict(chat_proposer.channel_totals)]
    for final, reasoning in (("a", "bb"), ("", "ccc"), ("dddd", ""), ("   ", "")):
        chat_proposer._install_payload(_chat_payload(final=final, reasoning=reasoning))
        _drive_chat(chat_proposer)
        seen.append(dict(chat_proposer.channel_totals))

    for earlier, later in zip(seen, seen[1:]):
        for key, value in earlier.items():
            assert later[key] >= value, f"{key} decreased: {value} -> {later[key]}"

    assert seen[-1]["completions"] == 4, "all four calls must have been counted"
