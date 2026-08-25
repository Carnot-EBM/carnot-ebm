"""An INERT induced engine -- one that predicts no action changes anything -- is rejectable.

WHY THIS EXISTS (`results/outer_loop_arc_generation_taxonomy_20260801.json`). Classifying 172
gemma-4-31B induce candidates by failure mode found that inertness is the LARGEST single class:
26 of 172 (15.1%), 15 of 124 (12.1%) on the shipped 3-try path, exceeding every code-validity
class combined. It was also the only class the live induce path took NO ACTION on, despite the
detector (`engine_changes_anything`) being shipped in `arc_engine_static_validation` and already
imported at the `_engine_defects` call site.

TWO THINGS HAD TO BE TRUE BEFORE IT COULD BE WIRED, and both are pinned here:

  1. `engine_changes_anything` is UNBOUNDED. Measured, not assumed: on ft09 candidate 5 --
     the engine behind the 2026-07-31 13-minute wedge -- `validate_engine_code` returns in
     30.07s with `engine_nonterminating` while `engine_changes_anything` does not return at all
     (`probe_unbounded_inertness.json`). Wiring the unbounded function into the live path would
     have reintroduced the exact hang the subprocess bound was shipped to remove.
  2. UNDETERMINED IS NOT INERT. A probe that could not reach a verdict must not reject. The two
     ways to reach "undetermined" that really are the engine's fault (it hangs, it crashes the
     validator) are already rejected upstream by `dry_run_defects`.

AND IT SHIPS OFF. `CARNOT_ARC_INDUCE_REJECT_INERT` defaults to off so the control arm of the A/B
is the shipped path itself rather than a reimplementation of it.

SCENARIO-ARC-FCP-5699-43-INERT-ENGINE-IS-A-DEFECT
SCENARIO-ARC-FCP-5699-43-INERTNESS-PROBE-IS-BOUNDED
SCENARIO-ARC-FCP-5699-43-DEFAULT-OFF-IS-BYTE-FOR-BYTE-THE-OLD-PATH
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from carnot.agentic import arc_engine_static_validation as sv
from carnot.agentic import arc_executable_world_model as e3

IDENTITY = "def engine(grid, action, data=None):\n    return grid.copy()\n"
IDENTITY_DRESSED = (
    "import numpy as np\n"
    "def engine(grid, action, data=None):\n"
    "    out = np.array(grid, copy=True)\n"
    "    for i in range(out.shape[0]):\n"
    "        for j in range(out.shape[1]):\n"
    "            out[i, j] = grid[i, j]\n"
    "    return out\n"
)
LIVE = "def engine(grid, action, data=None):\n    g = grid.copy()\n    g[0, 0] = 7\n    return g\n"
SPIN = "def engine(grid, action, data=None):\n    while True:\n        pass\n"
RAISES = "def engine(grid, action, data=None):\n    raise ValueError('boom-43')\n"
RETURNS_NONE = "def engine(grid, action, data=None):\n    pass\n"
# Inert on the transitions it answers AND broken on one. The only engine shape that isolates the
# `not defects` ordering guard -- see test_a_real_defect_still_wins_over_inertness.
MOSTLY_IDENTITY_ONE_RAISE = (
    "def engine(grid, action, data=None):\n"
    "    if action == 9:\n"
    "        raise ValueError('boom-43')\n"
    "    return grid.copy()\n"
)
WRONG_SHAPE = (
    "import numpy as np\ndef engine(grid, action, data=None):\n    return np.zeros((2, 2), int)\n"
)


class _DuckTransition:
    """Locally defined on purpose -- the module's contract is duck-typed, and a child process
    that unpickles by importing the class would break it. See the sibling wall-clock test."""

    def __init__(self, grid: Any, action: int, data: Any = None) -> None:
        self.grid, self.action, self.data = grid, action, data


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCE_REJECT_INERT", raising=False)


@pytest.fixture
def transitions() -> list[_DuckTransition]:
    g = np.zeros((6, 6), dtype=int)
    return [_DuckTransition(g.copy(), a) for a in (1, 2, 3)]


# ---------------------------------------------------------------------------
# 1. The detector itself
# ---------------------------------------------------------------------------


class TestTheDetector:
    def test_an_identity_engine_is_reported_inert(self, transitions):
        d = sv.engine_inertness_defect(IDENTITY, transitions)
        assert d is not None
        assert d.kind == "engine_inert"
        assert d.evidence["changed_on_any"] is False
        assert d.evidence["transitions_tried"] == 3

    def test_inertness_is_about_BEHAVIOUR_not_about_the_source_text(self, transitions):
        """An engine that copies every cell one at a time is still the identity function. A
        source-text check could not see that; running it can, which is why this is a dry-run
        style check and not a static one."""
        d = sv.engine_inertness_defect(IDENTITY_DRESSED, transitions)
        assert d is not None and d.kind == "engine_inert"

    def test_an_engine_that_changes_the_grid_is_NOT_flagged(self, transitions):
        assert sv.engine_inertness_defect(LIVE, transitions) is None

    def test_the_defect_is_repairable_because_we_can_say_what_we_saw(self, transitions):
        """Contrast with `engine_nonterminating`, which is NOT repairable: there the run was
        killed and there is no observation to feed back. Here there is one -- 'your engine never
        changed the grid' -- so the model can act on it."""
        d = sv.engine_inertness_defect(IDENTITY, transitions)
        assert d is not None and d.repairable is True
        assert sv.repair_prompt_block([d]) != ""


# ---------------------------------------------------------------------------
# 2. UNDETERMINED IS NOT INERT -- the fail-open direction
# ---------------------------------------------------------------------------


class TestUndeterminedIsNotInert:
    def test_a_nonterminating_engine_is_NOT_reported_inert(self, transitions, monkeypatch):
        """It is killed at the bound, which is UNDETERMINED, not "changes nothing". Reporting it
        as inert would be a fabricated observation -- and would double-count one broken engine,
        since `dry_run_defects` already rejects it as `engine_nonterminating`."""
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", "2")
        t0 = time.monotonic()
        assert sv.engine_inertness_defect(SPIN, transitions) is None
        assert time.monotonic() - t0 < 30, "the probe was not bounded"

    def test_the_nonterminating_engine_is_still_rejected_BY_THE_DRY_RUN(
        self, transitions, monkeypatch
    ):
        """The fail-open above is only safe because something else catches this case. Assert
        that, rather than assuming it: a fail-open whose backstop had quietly moved would let a
        hanging engine through."""
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", "2")
        kinds = {d.kind for d in sv.validate_engine_code(SPIN, transitions=transitions)}
        assert "engine_nonterminating" in kinds

    def test_an_engine_that_cannot_be_run_at_all_is_NOT_reported_inert(self, transitions):
        assert sv.engine_inertness_defect("def not_engine(): pass\n", transitions) is None

    @pytest.mark.parametrize("code", [RAISES, RETURNS_NONE, WRONG_SHAPE])
    def test_an_engine_that_never_RAN_is_NOT_reported_inert(self, code, transitions):
        """THE BUG THIS TEST FOUND. `engine_changes_anything` `continue`s past a raise, past a
        None return and past a wrong-shaped return, so an engine that does any of those on
        EVERY transition returns False from it -- indistinguishable, at that return value, from
        an engine that ran cleanly and predicted nothing ever changes. The first version of
        `engine_inertness_defect` read that False and labelled a raising engine 'inert'. Those
        are different failures with different repairs, and only the second is inertness."""
        assert sv.engine_inertness_defect(code, transitions) is None

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(RAISES, "engine_raised"), (RETURNS_NONE, "engine_returned_none")],
    )
    def test_those_engines_are_still_rejected_UNDER_THEIR_OWN_KIND(
        self, code, expected, transitions
    ):
        """The fail-open above is only safe because something else catches these. Assert it."""
        kinds = {d.kind for d in sv.validate_engine_code(code, transitions=transitions)}
        assert expected in kinds

    def test_the_census_separates_ran_and_did_nothing_from_never_ran(self, transitions):
        inert = sv._change_census_inprocess(IDENTITY, transitions)
        broken = sv._change_census_inprocess(RAISES, transitions)
        assert inert["changes_anything"] is False and inert["n_usable_predictions"] == 3
        assert broken["changes_anything"] is False and broken["n_usable_predictions"] == 0

    def test_the_bounded_probe_is_three_valued(self, transitions, monkeypatch):
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", "2")
        assert sv.engine_changes_anything_bounded(LIVE, transitions) is True
        assert sv.engine_changes_anything_bounded(IDENTITY, transitions) is False
        assert sv.engine_changes_anything_bounded(SPIN, transitions) is None

    def test_the_bounded_probe_agrees_with_its_unbounded_sibling(self, transitions, monkeypatch):
        """They must stay the same function, one of them merely fenced. If the census had
        changed the VERDICT rather than only adding a count, this would fail."""
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", "10")
        for code in (LIVE, IDENTITY, IDENTITY_DRESSED, RAISES, RETURNS_NONE, WRONG_SHAPE):
            assert sv.engine_changes_anything_bounded(code, transitions) == (
                sv.engine_changes_anything(code, transitions)
            ), code


# ---------------------------------------------------------------------------
# 3. Bounded, in a killable child -- the precondition from the taxonomy
# ---------------------------------------------------------------------------


class TestTheBound:
    def test_the_probe_runs_in_a_subprocess_by_default(self, transitions, monkeypatch):
        seen: dict = {}
        real = sv._run_isolated_job

        def spy(*a, **kw):
            seen["job_kind"] = kw.get("job_kind")
            return real(*a, **kw)

        monkeypatch.setattr(sv, "_run_isolated_job", spy)
        sv.engine_changes_anything_bounded(IDENTITY, transitions)
        assert seen["job_kind"] == "changes_anything"

    def test_the_child_starts_without_package_owned_pickle_or_blas_fanout(
        self, transitions, monkeypatch
    ):
        """SCENARIO-ARC-FCP-5699-43-INERTNESS-PROBE-IS-BOUNDED keeps startup
        infrastructure outside the generated-code deadline under parallel pytest workers."""
        import json
        import pickle
        from pathlib import Path
        from types import SimpleNamespace

        observed: dict = {}

        def fake_run(argv, **kwargs):
            with open(argv[-1], "rb") as handle:
                observed["job"] = pickle.load(handle)
            observed["argv"] = list(argv)
            observed["env"] = dict(kwargs["env"])
            return SimpleNamespace(
                returncode=0,
                stderr="",
                stdout=json.dumps(
                    {
                        "changes_anything": False,
                        "n_usable_predictions": 3,
                        "n_tried": 3,
                    }
                ),
            )

        monkeypatch.setattr(sv.subprocess, "run", fake_run)
        status, payload = sv._run_isolated_job(
            IDENTITY,
            transitions,
            limit=25,
            func_name="engine",
            timeout_s=30.0,
            job_kind="changes_anything",
        )

        assert status == "ok" and payload["n_usable_predictions"] == 3
        assert observed["argv"][1] == str(Path(sv.__file__).resolve())
        assert all(isinstance(row, dict) for row in observed["job"]["transitions"])
        assert set(
            observed["env"].get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ) == {"1"}

    def test_the_disable_knob_restores_the_unbounded_in_process_path_exactly(
        self, transitions, monkeypatch
    ):
        monkeypatch.setenv("CARNOT_ARC_DRY_RUN_TIMEOUT_S", "0")
        called: dict = {}
        monkeypatch.setattr(
            sv, "_run_isolated_job", lambda *a, **kw: called.setdefault("spawned", True)
        )
        assert sv.engine_changes_anything_bounded(IDENTITY, transitions) is False
        assert "spawned" not in called

    def test_the_unbounded_public_helper_is_left_alone(self, transitions):
        """`engine_changes_anything` keeps its old name AND its old semantics. Changing them
        under existing callers would be a behaviour change smuggled in as a refactor."""
        assert sv.engine_changes_anything(IDENTITY, transitions) is False
        assert sv.engine_changes_anything(LIVE, transitions) is True

    def test_the_child_entry_point_answers_the_changes_anything_job(self, tmp_path):
        import json
        import pickle
        import subprocess
        import sys

        job = tmp_path / "job.pkl"
        g = np.zeros((4, 4), dtype=int)
        with open(job, "wb") as fh:
            pickle.dump(
                {
                    "code": IDENTITY,
                    "transitions": [sv._ShimTransition(grid=g, action=1, data=None)],
                    "limit": 5,
                    "func_name": "engine",
                    "job_kind": "changes_anything",
                },
                fh,
            )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "carnot.agentic.arc_engine_static_validation",
                "--dry-run-job",
                str(job),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(proc.stdout.strip().splitlines()[-1]) == {
            "changes_anything": False,
            "n_usable_predictions": 1,
            "n_tried": 1,
        }

    def test_a_job_with_no_kind_still_means_dry_run(self, tmp_path):
        """Backward compatibility of the staged job, asserted rather than hoped for."""
        import json
        import pickle
        import subprocess
        import sys

        job = tmp_path / "job.pkl"
        g = np.zeros((4, 4), dtype=int)
        with open(job, "wb") as fh:
            pickle.dump(
                {
                    "code": RAISES,
                    "transitions": [sv._ShimTransition(grid=g, action=1, data=None)],
                    "limit": 5,
                    "func_name": "engine",
                },
                fh,
            )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "carnot.agentic.arc_engine_static_validation",
                "--dry-run-job",
                str(job),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert [d["kind"] for d in payload["defects"]] == ["engine_raised"]


# ---------------------------------------------------------------------------
# 4. `validate_engine_code` is UNCHANGED -- the measurement definition must not move
# ---------------------------------------------------------------------------


class TestTheMeasurementDefinitionDidNotMove:
    def test_validate_engine_code_still_calls_an_inert_engine_CLEAN(self, transitions):
        """Load-bearing for the A/B. `validate_engine_code` is the definition usable-engine
        yield is measured with; if inertness were folded into it, the treatment and the outcome
        would be the same object and the measurement circular."""
        assert sv.validate_engine_code(IDENTITY, transitions=transitions) == []

    def test_inertness_is_clean_under_validate_even_with_the_flag_on(
        self, transitions, monkeypatch
    ):
        monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", "1")
        assert sv.validate_engine_code(IDENTITY, transitions=transitions) == []


# ---------------------------------------------------------------------------
# 5. The wiring, and that it is OFF by default
# ---------------------------------------------------------------------------


class _Proposer:
    """The narrowest object `_engine_defects` needs. Constructing a real `LocalGGUFProposer`
    would try to reach a GPU server; the method under test reads exactly two attributes."""

    last_stop_type = None
    last_requested_n_predict = 0
    max_tokens = 4096
    _engine_defects = e3.LocalGGUFProposer._engine_defects


class TestTheFlag:
    def test_the_shipped_default_is_off(self, monkeypatch):
        monkeypatch.delenv("CARNOT_ARC_INDUCE_REJECT_INERT", raising=False)
        assert e3._reject_inert_engines() is False

    @pytest.mark.parametrize("raw", ["0", "", "yes", "true", "True", "2", " "])
    def test_only_an_exact_1_turns_it_on(self, raw, monkeypatch):
        """A malformed value falls back to OFF. A typo must not change how the scored agent
        behaves."""
        monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", raw)
        assert e3._reject_inert_engines() is False

    def test_it_turns_on(self, monkeypatch):
        monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", "1")
        assert e3._reject_inert_engines() is True

    def test_default_off_reports_no_defect_for_an_inert_engine(self, transitions, monkeypatch):
        monkeypatch.delenv("CARNOT_ARC_INDUCE_REJECT_INERT", raising=False)
        assert _Proposer()._engine_defects(IDENTITY, transitions) == []

    def test_default_off_DOES_NOT_EVEN_RUN_THE_PROBE(self, transitions, monkeypatch):
        """ "Ships inert" is a claim about BEHAVIOUR, not about the return value.

        The sibling test above only pins that the RESULT is [] with the flag off -- which would
        still hold if the probe ran on every induce and its finding were discarded. That would
        cost a subprocess spawn per induction on the live scored path, which is a real
        regression wearing an identical return value. `and` short-circuits, so the probe is
        genuinely not reached; this asserts it rather than trusting the operator precedence."""
        monkeypatch.delenv("CARNOT_ARC_INDUCE_REJECT_INERT", raising=False)
        called: list = []
        monkeypatch.setattr(
            sv, "engine_inertness_defect", lambda *a, **kw: called.append(a) or None
        )
        assert _Proposer()._engine_defects(IDENTITY, transitions) == []
        assert called == [], "the flag is off but the inertness probe still ran"

    def test_flag_on_reports_engine_inert(self, transitions, monkeypatch):
        monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", "1")
        assert _Proposer()._engine_defects(IDENTITY, transitions) == ["engine_inert"]

    def test_flag_on_does_not_touch_a_live_engine(self, transitions, monkeypatch):
        monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", "1")
        assert _Proposer()._engine_defects(LIVE, transitions) == []

    @pytest.mark.parametrize("trans", [None, []])
    def test_flag_on_with_no_transitions_NEVER_REACHES_THE_PROBE(self, trans, monkeypatch):
        """Without observed transitions there is nothing to run the engine against, so
        inertness is unobservable and must not be guessed at.

        ASSERTED AS 'THE PROBE IS NOT CALLED', not as 'the result is []'. Mutation testing
        (2026-08-01) showed the result-shaped version of this test stays GREEN when the
        `and transitions` guard is deleted: with `None` the probe raises TypeError into
        `_engine_defects`' broad `except Exception: return []`, and with `[]` the census
        reports zero usable predictions. Both give [] by accident, through paths that would
        equally hide a real bug. The guard is what makes it deliberate, so the guard is what
        gets asserted."""
        monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", "1")
        called: list = []
        monkeypatch.setattr(
            sv, "engine_inertness_defect", lambda *a, **kw: called.append(a) or None
        )
        assert _Proposer()._engine_defects(IDENTITY, trans) == []
        assert called == [], "the inertness probe was reached with nothing to run against"

    def test_a_real_defect_still_wins_over_inertness(self, monkeypatch):
        """ORDERING IS LOAD-BEARING, and this engine is why an all-raising one does not test it.

        `MOSTLY_IDENTITY_ONE_RAISE` is inert on the transitions it answers (2 usable
        predictions, neither changing anything) AND raises on one -- so the inertness probe
        would return a real `engine_inert`, and only the `not defects` ordering guard stops it.
        The earlier version of this test used an ALL-raising engine, where the probe's own
        `n_usable >= 1` requirement already returns None; deleting the ordering guard left the
        suite green. Two rules double-covering one case is how a named guard gets silently
        removed."""
        monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", "1")
        g = np.zeros((6, 6), dtype=int)
        trans = [_DuckTransition(g.copy(), a) for a in (1, 2, 9)]
        assert sv.engine_inertness_defect(MOSTLY_IDENTITY_ONE_RAISE, trans) is not None
        kinds = _Proposer()._engine_defects(MOSTLY_IDENTITY_ONE_RAISE, trans)
        assert kinds == ["engine_raised"]
        assert "engine_inert" not in kinds
