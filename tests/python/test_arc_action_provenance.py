"""Tests for the per-action provenance instrument on the SCORED ARC path.

REQ-ARC-WMTE-6070 / SCENARIO-ARC-WMTE-6070-*: the instrument must (a) be OFF by default,
(b) not change what the agent does when off, (c) attribute every emitted action to a branch
drawn from the closed vocabulary, and (d) never propagate its own failure into the agent.

The load-bearing test in this file is `test_flag_unset_leaves_action_sequence_identical`:
an instrument that perturbs the run answers a question about the instrument. It runs the
REAL `E3AgentPolicy` (the SCORED policy, reached the same way `make_carnot_agent` reaches
it) against a scripted stub environment, twice, and asserts the two action sequences are
equal element-for-element.

The environment here is a hand-written stub rather than the offline arcade because a unit
test must not depend on `environment_files/` being present, must not take seconds, and must
not open a scorecard. The arcade-level version of the same comparison is the three-arm
probe in `scripts/arc_action_provenance_probe.py`, whose artifact carries the
byte-identity result over a real game.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_action_provenance as prov
from carnot.agentic.arc_competition_agent import E3AgentPolicy


# ----------------------------------------------------------------------------------------
# A minimal frame stub. The policy reads `levels_completed` (via `_level_of`) and the grid
# (via `grid_of`), and nothing else that matters here.
# ----------------------------------------------------------------------------------------


class _Frame:
    def __init__(self, grid: np.ndarray, level: int = 0) -> None:
        self.frame = [grid.tolist()]
        self.levels_completed = level
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _grid(seedval: int, n: int = 8) -> np.ndarray:
    rng = np.random.RandomState(seedval)
    return rng.randint(0, 4, size=(n, n)).astype(int)


def _drive(policy: E3AgentPolicy, n_actions: int) -> list[str]:
    """Step the policy `n_actions` times against a deterministic scripted world.

    The 'world' advances one canned frame per action. It is not a game -- it does not have
    to be. What is under test is which BRANCH of the policy emits each action and whether
    arming the recorder changes that, and both are decided by policy-internal state plus the
    frame stream, which this reproduces exactly across runs.
    """
    frames: list[_Frame] = []
    latest = None
    out: list[str] = []
    for i in range(n_actions):
        kind, data = policy.next_move(frames, latest)
        out.append(json.dumps({"a": kind, "d": data}, sort_keys=True, default=str))
        if kind is None:
            break
        latest = _Frame(_grid(i + 1), level=0)
        frames.append(latest)
    return out


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Keep every test in this file off the LLM and off any GPU.

    `CARNOT_ARC_DISABLE_INDUCTION=1` makes `_induce_and_plan` short-circuit before it can
    lazily construct a `LocalGGUFProposer`, so no 31B GGUF is ever loaded by the unit suite.
    """
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv(prov.PROVENANCE_ENV_FLAG, raising=False)
    monkeypatch.delenv(prov.PROVENANCE_DIR_ENV, raising=False)


# ----------------------------------------------------------------------------------------
# (a) OFF by default
# ----------------------------------------------------------------------------------------


def test_provenance_is_off_by_default(monkeypatch):
    monkeypatch.delenv(prov.PROVENANCE_ENV_FLAG, raising=False)
    assert prov.provenance_enabled() is False
    assert prov.maybe_make_recorder("xx11") is None
    pol = E3AgentPolicy("xx11", proposer=None)
    assert pol.action_provenance() is None


def test_flag_must_be_exactly_one(monkeypatch):
    """A stray `=0` must not arm the instrument.

    Strict equality rather than truthiness is deliberate: `"0"` is a truthy Python string,
    so a truthiness test would arm on the exact value an operator would type to turn it OFF.
    """
    for value, expected in (("0", False), ("", False), ("true", False), ("1", True)):
        monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, value)
        assert prov.provenance_enabled() is expected, value


# ----------------------------------------------------------------------------------------
# (b) INERT -- the claim the whole exercise rests on
# ----------------------------------------------------------------------------------------


def test_flag_unset_leaves_action_sequence_identical(monkeypatch):
    """Arming the recorder must not change a single action the policy emits.

    Three runs, not two. `off` vs `off2` establishes that this policy is deterministic under
    the fixture at all -- without it, `off == on` could be luck and `off != on` could be
    ordinary nondeterminism misattributed to the instrument. The determinism assertion is
    made FIRST, so a failure reports the right cause.
    """
    monkeypatch.delenv(prov.PROVENANCE_ENV_FLAG, raising=False)
    off = _drive(E3AgentPolicy("xx11", proposer=None, explore_budget=6), 40)
    off2 = _drive(E3AgentPolicy("xx11", proposer=None, explore_budget=6), 40)
    assert off == off2, "policy is not deterministic under this fixture; inertness untestable"

    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol_on = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    on = _drive(pol_on, 40)

    assert pol_on.action_provenance() is not None, "arming did not construct a recorder"
    assert on == off, (
        "arming the provenance instrument CHANGED the agent's action sequence; "
        f"first divergence at index "
        f"{next((i for i, (x, y) in enumerate(zip(on, off)) if x != y), None)}"
    )


def test_armed_run_records_one_row_per_action(monkeypatch):
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    actions = _drive(pol, 25)
    rec = pol.action_provenance()
    assert rec is not None
    assert len(rec.rows) == len(actions)
    assert rec.errors == [], rec.errors


# ----------------------------------------------------------------------------------------
# (c) every action is attributed, and only to a KNOWN branch
# ----------------------------------------------------------------------------------------


def test_every_action_carries_a_known_top_branch(monkeypatch):
    """An unrecognised label means the agent grew a decision path the accounting would have
    silently mis-bucketed. That must fail loudly rather than land in an 'other' bin."""
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 40)
    rec = pol.action_provenance()
    assert rec is not None and rec.rows
    for row in rec.rows:
        assert row["top_branch"] in prov.TOP_BRANCHES, row
    summary = rec.summary()
    assert summary["unknown_top_branches"] == []
    assert summary["unknown_explorer_branches"] == []
    assert summary["actions_recorded"] == len(rec.rows)


def test_explorer_rows_carry_an_explorer_branch_and_plan_rows_do_not(monkeypatch):
    """The two label layers must not bleed into each other.

    A plan-step row carrying a stale explorer branch would inflate the explorer's share of
    the accounting -- which is the single number this instrument exists to produce -- so the
    separation is asserted directly rather than assumed from the clearing code.
    """
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 40)
    rec = pol.action_provenance()
    assert rec is not None
    for row in rec.rows:
        if str(row["top_branch"]).endswith("explorer"):
            assert row["explorer_branch"] in prov.EXPLORER_BRANCHES, row
        else:
            assert row["explorer_branch"] is None, row
            assert row["explorer_serve_kind"] is None, row
        if row["explorer_serve_kind"] is not None:
            assert row["explorer_serve_kind"] in prov.SERVE_KINDS, row


def test_serve_kind_only_appears_on_branches_that_call_serve(monkeypatch):
    """`_serve` is reached from exactly two explorer branches. A serve kind on any other
    branch means a label leaked across actions."""
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 40)
    rec = pol.action_provenance()
    assert rec is not None
    for row in rec.rows:
        if row["explorer_serve_kind"] is not None:
            assert row["explorer_branch"] in ("pending_drain", "frontier.navigate"), row


def test_summary_counts_partition_the_actions(monkeypatch):
    """plan-derived + explorer + reset-for-replay must account for every recorded action.

    This is the arithmetic the headline rests on. If the buckets do not partition, a
    '0% of actions are plan-derived' headline could be hiding actions in an unnamed
    remainder.
    """
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 40)
    s = pol.action_provenance().summary()
    total = s["plan_derived_actions"] + s["explorer_actions"] + s["reset_for_plan_replay_actions"]
    assert total == s["actions_recorded"], s


# ----------------------------------------------------------------------------------------
# (d) the instrument must never break the agent
# ----------------------------------------------------------------------------------------


def test_recorder_failure_does_not_break_the_agent(monkeypatch):
    """A recorder that raises on every call must cost rows, never actions."""
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)

    def _boom(*_a, **_k):
        raise RuntimeError("instrument exploded")

    monkeypatch.setattr(pol, "_provenance_pre_state", _boom)
    monkeypatch.setattr(pol, "_provenance_post_state", _boom)
    actions = _drive(pol, 15)
    assert len(actions) == 15
    rec = pol.action_provenance()
    assert rec.rows == []
    assert any("pre_state" in e or "post_state" in e for e in rec.errors)


def test_flush_writes_a_readable_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    monkeypatch.setenv(prov.PROVENANCE_DIR_ENV, str(tmp_path))
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 10)
    path = pol.action_provenance().flush()
    assert path is not None
    data = json.loads(Path(path).read_text())
    assert data["schema"] == prov.SCHEMA
    assert len(data["rows"]) == 10
    assert Path(path).parent == tmp_path


def test_flush_failure_returns_none_and_is_recorded(monkeypatch):
    """Flushing into an unwritable location must be a recorded error, not an exception."""
    rec = prov.ActionProvenanceRecorder("xx11", path="/proc/definitely/not/writable/x.json")
    assert rec.flush() is None
    assert any("flush" in e for e in rec.errors)


# ----------------------------------------------------------------------------------------
# plan-epoch accounting
# ----------------------------------------------------------------------------------------


def test_plan_epoch_counts_installs_not_clears():
    """`self.plan = []` at a level boundary is a CLEAR, not a plan install. Counting it
    would credit the induce->plan pipeline with a plan it never produced."""
    rec = prov.ActionProvenanceRecorder("xx11")
    rec.note_plan_object([], None)  # initial empty plan
    assert rec.plan_epoch == 0
    rec.note_plan_object([{"action": 1}], 0)  # a real install
    assert rec.plan_epoch == 1
    assert rec.plan_installed_by_attempt == 0
    rec.note_plan_object([], 1)  # cleared at a level boundary
    assert rec.plan_epoch == 1
    rec.note_plan_object([{"action": 2}], 2)  # a second real install
    assert rec.plan_epoch == 2
    assert rec.plan_installed_by_attempt == 2


def test_note_plan_object_is_identity_based_not_length_based():
    """Two DIFFERENT plans that happen to be the same length are two installs."""
    rec = prov.ActionProvenanceRecorder("xx11")
    a = [{"action": 1}]
    b = [{"action": 1}]
    rec.note_plan_object(a, 0)
    rec.note_plan_object(a, 0)  # same object again: not a new install
    assert rec.plan_epoch == 1
    rec.note_plan_object(b, 1)
    assert rec.plan_epoch == 2


# ----------------------------------------------------------------------------------------
# The instrumented source itself: the labels must stay wired.
# ----------------------------------------------------------------------------------------


def test_every_declared_top_branch_is_assigned_somewhere_in_the_source():
    """A vocabulary entry no return site assigns is a branch the accounting claims to cover
    and does not. Read from source because the plan branches need an LLM to reach at
    runtime, so a runtime-only check would silently stop covering them."""
    src = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "python",
            "carnot",
            "agentic",
            "arc_competition_agent.py",
        )
    ).read_text()
    for branch in prov.TOP_BRANCHES:
        assert f'self._prov_top = "{branch}"' in src, branch
    for branch in prov.EXPLORER_BRANCHES:
        assert f'self._prov_branch = "{branch}"' in src, branch
    for kind in prov.SERVE_KINDS:
        assert f'self._prov_serve_kind = "{kind}"' in src, kind


def test_no_return_site_in_the_two_choosers_is_unlabelled():
    """Every `return` in `E3AgentPolicy._next_move_routed` must be preceded by a
    `_prov_top` assignment.

    This is the test that catches the failure mode the whole instrument is vulnerable to:
    someone adds a seventh exit to the routing function, actions start flowing through it,
    and the accounting attributes them to whatever label happened to be left over from the
    previous action. Counting labels is not enough -- an unlabelled NEW exit passes a
    count-based check.
    """
    import ast
    import inspect
    import textwrap

    from carnot.agentic import arc_competition_agent as mod

    src = textwrap.dedent(inspect.getsource(mod.E3AgentPolicy._next_move_routed))
    tree = ast.parse(src)
    fn = tree.body[0]

    labelled: set[int] = set()
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for prev, cur in zip(block, block[1:]):
                if not isinstance(cur, ast.Return):
                    continue
                if (
                    isinstance(prev, ast.Assign)
                    and len(prev.targets) == 1
                    and isinstance(prev.targets[0], ast.Attribute)
                    and prev.targets[0].attr == "_prov_top"
                ):
                    labelled.add(cur.lineno)

    returns = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return)]
    unlabelled = sorted(set(returns) - labelled)
    assert not unlabelled, (
        "return site(s) in E3AgentPolicy._next_move_routed with no immediately preceding "
        f"`self._prov_top = ...` label, at relative line(s) {unlabelled}. Every exit must "
        "be labelled or the per-action accounting silently mis-attributes actions."
    )
    assert len(returns) == len(prov.TOP_BRANCHES), (
        f"{len(returns)} return sites but {len(prov.TOP_BRANCHES)} declared top branches -- "
        "the vocabulary in arc_action_provenance.TOP_BRANCHES is out of date."
    )


def test_next_move_delegates_without_recording_when_unarmed(monkeypatch):
    """The default path must not touch the recorder machinery at all."""
    monkeypatch.delenv(prov.PROVENANCE_ENV_FLAG, raising=False)
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    called: list[int] = []
    real = pol._next_move_recorded

    def _spy(*a, **k):
        called.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pol, "_next_move_recorded", _spy)
    _drive(pol, 8)
    assert called == [], "_next_move_recorded ran with the instrument unarmed"


# ----------------------------------------------------------------------------------------
# The measurement driver must not write the tracked evidence store.
# ----------------------------------------------------------------------------------------


def test_probe_redirects_the_engine_store_for_every_arm():
    """SCENARIO-ARC-WMTE-6070-9. A live induction writes
    `<E3_DIR>/<game>/world_model.py`, and the default `E3_DIR` is `results/arc_e3` --
    TRACKED, READ-ONLY evidence. The module's own `_guard_engine_write` is deliberately
    scoped to pytest, because the LIVE agent writing there is what the store is FOR, so a
    measurement driver is precisely the caller nothing protects.

    This is a regression test for an incident, not a hypothetical: the first live-generator
    attempt at this measurement rewrote `results/arc_e3/tn36/world_model.py` (40 insertions,
    14 deletions) within 90 seconds, and it surfaced from `git status` rather than from any
    guard. The file was restored from git.

    Asserted on the source because the alternative -- running a live induction to observe
    the write -- would need a 31B GGUF and would itself be the thing being prevented.
    """
    src = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts",
            "arc_action_provenance_probe.py",
        )
    ).read_text()
    assert 'env["CARNOT_ARC_E3_DIR"]' in src, (
        "the probe must redirect the engine store per arm, or a live arm overwrites "
        "results/arc_e3/<game>/world_model.py"
    )
    # PER-ARM, not one shared directory: with a shared store, arm A's induced engine is on
    # disk when arm B starts, so the A/B comparison silently becomes a comparison of two
    # different situations.
    assert 'f"e3_store_{label}"' in src, "the engine store redirect must be per-arm"


def test_worker_refuses_to_run_against_the_tracked_evidence_store():
    """The child's own belt-and-braces check, asserted on the source for the same reason.

    It lives in the WORKER rather than only in the probe because `E3_DIR` is resolved at
    module import in the child, so the child is the only place that can check what the run
    will REALLY use rather than what the environment was asked to set.
    """
    src = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts",
            "arc_action_provenance_worker.py",
        )
    ).read_text()
    assert "_TRACKED_E3_EVIDENCE_DIR" in src
    assert "E3_DIR.resolve() == _TRACKED_E3_EVIDENCE_DIR.resolve()" in src


def test_recorded_action_data_is_a_copy_not_the_live_object(monkeypatch):
    """A recorded row must not change after the fact.

    The same `data` dict the policy returns is handed straight to the environment step, and
    `record()` only shallow-copies the row -- so storing the REFERENCE would let a
    downstream mutation rewrite an action that was already accounted for. A record that
    changes retroactively is the one failure an accounting cannot survive.

    Driven through `_provenance_post_state` directly rather than through an episode. The
    scripted `_Frame` fixture provably never emits a coordinate action -- 60 steps produce
    60 rows with `data is None` -- so an episode-driven version of this test asserts nothing
    at all. That vacuity is the failure mode the QA-Layer discipline calls a decorative
    test; it was written first and caught by its own non-vacuity guard, and is recorded here
    rather than quietly replaced.
    """
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 4)  # give the policy real state to snapshot

    live_data = {"x": 31, "y": 35}
    pre = pol._provenance_pre_state([], None)
    row = pol._provenance_post_state(pre, (6, live_data), pol.plan, [], None)

    assert row["data"] == {"x": 31, "y": 35}
    assert row["data"] is not live_data, "the row holds the LIVE dict, not a copy"

    live_data["x"] = -999
    live_data["injected_by_a_later_mutation"] = True
    assert row["data"] == {"x": 31, "y": 35}, (
        "mutating the dict after the action was recorded changed the recorded row"
    )


# ----------------------------------------------------------------------------------------
# The PLAN branches, reached without an LLM.
#
# The offline-arcade probe's induction-disabled configuration never enters a plan-execution
# branch, so its byte-identity result covers the explorer path only -- it says so in its own
# artifact. These reach the plan labels at RUNTIME by installing a plan on the policy
# directly, which is the only way to exercise them without a 31B GGUF, and closes the gap
# between "the label exists in the source" and "the label is what a plan action carries".
# ----------------------------------------------------------------------------------------


def test_execute_phase_actions_are_labelled_as_plan_steps(monkeypatch):
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 3)  # build real explorer state first

    pol.plan = [{"action": 1, "data": None}, {"action": 2, "data": None}]
    pol.pi = 0
    pol.phase = "execute"
    pol.induced = True

    frames = [_Frame(_grid(9))]
    rows_before = len(pol.action_provenance().rows)
    kind_a, _ = pol.next_move(frames, frames[-1])
    kind_b, _ = pol.next_move(frames, frames[-1])
    rows = pol.action_provenance().rows[rows_before:]

    assert (kind_a, kind_b) == (1, 2)
    assert [r["top_branch"] for r in rows] == ["execute.plan_step", "execute.plan_step"]
    # A plan row must not carry an explorer label, and must report the plan's shape.
    assert [r["explorer_branch"] for r in rows] == [None, None]
    assert [r["plan_remaining"] for r in rows] == [1, 0]
    assert [r["plan_len"] for r in rows] == [2, 2]
    assert all(r["plan_present"] for r in rows)


def test_plan_step_rows_count_as_plan_derived_in_the_accounting(monkeypatch):
    """The headline number must actually move when plan steps occur.

    A summary that reported 0% plan-derived no matter what would be indistinguishable from
    the finding this instrument exists to establish -- which is exactly the kind of
    measurement floor the project has been bitten by before (`hv_progress` pinned at 0.0).
    """
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 3)
    pol.plan = [{"action": 1, "data": None}, {"action": 2, "data": None}]
    pol.pi = 0
    pol.phase = "execute"
    pol.induced = True
    frames = [_Frame(_grid(9))]
    pol.next_move(frames, frames[-1])
    pol.next_move(frames, frames[-1])

    s = pol.action_provenance().summary()
    assert s["plan_step_actions"] == 2
    assert s["plan_derived_actions"] == 2
    assert s["plan_derived_fraction"] > 0
    # and the partition still holds with both kinds of action present
    assert (
        s["plan_derived_actions"] + s["explorer_actions"] + s["reset_for_plan_replay_actions"]
        == s["actions_recorded"]
    )


def test_plan_exhaustion_is_labelled_exhausted_not_execute(monkeypatch):
    """Stepping past the end of a plan falls back to the explorer, and must say so.

    If the exhaustion fallback were mislabelled `execute.plan_step`, every action after a
    short plan ran out would be credited to the planner forever.
    """
    monkeypatch.setenv(prov.PROVENANCE_ENV_FLAG, "1")
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 3)
    pol.plan = [{"action": 1, "data": None}]
    pol.pi = 0
    pol.phase = "execute"
    pol.induced = True
    frames = [_Frame(_grid(9))]

    rows_before = len(pol.action_provenance().rows)
    pol.next_move(frames, frames[-1])  # consumes the only step
    pol.next_move(frames, frames[-1])  # plan exhausted
    rows = pol.action_provenance().rows[rows_before:]

    assert rows[0]["top_branch"] == "execute.plan_step"
    assert rows[1]["top_branch"] == "exhausted.explorer"
    assert rows[1]["explorer_branch"] in prov.EXPLORER_BRANCHES
    assert rows[1]["phase_after"] == "explore"
