"""BOTH-DIRECTIONS gate for the CEGIS acceptance/refinement purity split (defect C fix).

WHAT IS BEING FIXED. Acceptance is scored on the held-out tail `_split_prefix_heldout` returns,
and `execute_bounded_llm_reinduction` builds its refinement feedback from
`WorldModelVerifier(list(transitions))` -- the FULL corpus, that same tail included, with the
OBSERVED next state (`true_change`) attached to every mismatch. The rows that decide whether an
engine is trusted are the rows the LLM is handed to fix it with.

WHY BOTH DIRECTIONS, AND WHY OFF MUST STILL LEAK. Every measurement this harness has produced --
including the `retire_if_same_verdict` carried by exp5766 -- was taken against the leaking split.
If the OFF arm quietly stopped leaking, those measurements would become uninterpretable and the
A/B that is supposed to value this fix would have no control. So the OFF assertions here are
deliberately assertions that the BUG IS STILL PRESENT, byte for byte, and the companion
characterization file (test_arc_cegis_purity_leak_repro_20260803.py) is the fuller statement of
that same contract.

WHAT THE ON DIRECTION MUST BUY, stated as the measured comparison rather than as an intention.
Design (i) -- "just refine on the prefix" -- was rejected on evidence, not preference: with a
prefix-perfect engine, prefix-only refinement yields ZERO counterexamples on 13 of 13 real
offline windows, i.e. it trades the leak for no CEGIS signal at all in exactly the round where
refinement is the only route to acceptance. The ON arm must therefore close the leak AND keep a
non-empty counterexample budget where design (i) would have emptied it.
`test_on_keeps_the_counterexample_budget_design_i_would_have_destroyed` is that assertion.

Spec: REQ-ARC-WMTE-6090. Related: REQ-ARC-WMTE-4544 (the mismatch evidence attached to the
refactor call), REQ-ARC-WMTE-4791 (`select_trusted_world_model`, which owns the split),
REQ-ARC-FCP-5699-26 (`_bounded_mismatches`, the five-mismatch render cap that makes the leak
worst precisely when the engine is good).

Artifact: results/outer_loop_arc_cegis_purity_leak_20260803.json

SCENARIO-ARC-WMTE-6090-CEGIS-ACCEPT-SPLIT-GATE
"""

from __future__ import annotations

import json
import traceback
from typing import Any

import numpy as np
import pytest

from carnot.agentic import arc_llm_reinduction as reinduction
from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    _delta,
    refactor_prompt,
)
from carnot.agentic.arc_llm_reinduction import _counterexample_result, _proposal_prefix
from carnot.agentic.arc_world_model_trust_energy import (
    _CEGIS_ACCEPT_SPLIT_DEFAULT,
    WorldModelCandidate,
    _split_prefix_heldout,
    cegis_accept_split_enabled,
    select_trusted_world_model,
    split_refinement_acceptance,
)


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from the SHIPPED environment. A stray export in the operator's shell must not make
    a default-OFF assertion pass for the wrong reason."""

    monkeypatch.delenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", raising=False)


def _corpus(n: int, *, terminal_levelup: bool = True) -> list[Transition]:
    """Each step changes exactly one, DISTINCT cell -- so a rendered delta identifies its row.

    This is the same corpus construction the reproduction used, and the single-changed-cell
    property is what makes `_delivered_rows` a measurement rather than an inference: a row is
    counted as delivered only when its own unique cell appears in the prompt TEXT, never by
    trusting the mismatch's `i` label.
    """

    g = np.zeros((8, 8), dtype=int)
    rows: list[Transition] = []
    for i in range(n):
        nxt = g.copy()
        nxt[i // 8, i % 8] = i + 3
        rows.append(
            Transition(
                g.copy(), i % 4, None, nxt.copy(), 0, 1 if (terminal_levelup and i == n - 1) else 0
            )
        )
        g = nxt
    return rows


def _engine(rows: list[Transition], correct: set[int]):
    """A memorising engine that is right on exactly `correct` and a no-op elsewhere."""

    table = {
        (rows[i].grid.tobytes(), int(rows[i].action)): rows[i].next_grid.copy()
        for i in sorted(correct)
    }

    def engine(grid, action, data=None):
        hit = table.get((np.asarray(grid).tobytes(), int(action)))
        return hit.copy() if hit is not None else np.asarray(grid).copy()

    return engine


def _delivered_rows(rows: list[Transition], prompt: str) -> set[int]:
    """Rows whose OBSERVED answer is present in the rendered refactor prompt TEXT."""

    start = prompt.find("MISMATCHES:\n")
    assert start >= 0, "refactor_prompt no longer emits a MISMATCHES block"
    body = prompt[start + len("MISMATCHES:\n") :]
    parsed = json.loads(body[: body.rfind("]") + 1])
    tup2row: dict[tuple, int] = {}
    for i, t in enumerate(rows):
        for tup in _delta(t.grid, t.next_grid):
            tup2row.setdefault(tuple(tup), i)
    out: set[int] = set()
    for m in parsed:
        for tup in m.get("true_change") or []:
            hit = tup2row.get(tuple(tup))
            if hit is not None:
                out.add(hit)
    return out


def _refactor_prompt_for(rows: list[Transition], correct: set[int], corpus: list[Transition]):
    """Run the SHIPPED render path end to end: verifier -> counterexample -> refactor prompt."""

    eng = _engine(rows, correct)
    vr = WorldModelVerifier(list(corpus), hud_mask=None).score(eng)
    prompt = refactor_prompt(
        "SYNTH",
        _counterexample_result(
            {
                "real_n": vr.n,
                "real_n_correct": vr.n_correct,
                "real_accuracy": float(vr.accuracy),
                "real_mismatches": list(vr.mismatches),
            }
        ),
    )
    return vr, prompt


# =========================================================================================
# THE FLAG ITSELF
# =========================================================================================


class TestTheDefaultIsOff:
    def test_flag_defaults_off(self) -> None:
        assert cegis_accept_split_enabled() is False

    def test_the_module_constant_is_off(self) -> None:
        """Pinned separately from the function: reading the env var correctly while defaulting
        the constant to "1" would still ship the ungated behaviour."""

        assert _CEGIS_ACCEPT_SPLIT_DEFAULT == "0"

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_truthy_values_enable(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", raw)
        assert cegis_accept_split_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "banana"])
    def test_anything_else_stays_off(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail-closed on garbage: a typo'd export must not silently move a scored-path split."""

        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", raw)
        assert cegis_accept_split_enabled() is False


# =========================================================================================
# OFF REPRODUCES THE SHIPPED BEHAVIOUR -- INCLUDING THE BUG
# =========================================================================================


class TestOffReproducesTheLeak:
    def test_off_still_grades_on_the_legacy_prefix_heldout_split(self) -> None:
        """The acceptance split is untouched with the flag off, on every corpus size."""

        for n in range(2, 31):
            rows = _corpus(n)
            prefix, heldout = _split_prefix_heldout(rows)
            eng = _engine(rows, set(range(len(prefix))))
            sel = select_trusted_world_model(
                list(rows), [WorldModelCandidate("c", eng, None)], hidden_state=True
            )
            # heldout_accuracy is computed on `heldout`; recompute it directly and compare.
            direct = WorldModelVerifier(list(heldout), hud_mask=None).score(eng)
            assert float(sel.selected_score.heldout_accuracy) == pytest.approx(
                float(direct.accuracy)
            ), f"n={n}: flag-off acceptance is no longer the legacy held-out tail"
            assert sel.acceptance_split_enabled is False
            assert sel.acceptance_reason == "legacy_prefix_heldout_split"
            assert sel.n_acceptance_gradeable == -1

    def test_off_still_leaks_the_acceptance_rows_into_the_refactor_prompt(self) -> None:
        """THE BUG, ASSERTED AS STILL PRESENT.

        A prefix-perfect engine at the real offline window sizes hands the refiner every
        gradeable held-out row's observed answer. If this ever goes green-by-fixing rather than
        by flipping the flag, every prior measurement on this harness silently changes meaning.
        """

        for n in (9, 12, 25):
            rows = _corpus(n)
            n_prefix = len(_split_prefix_heldout(rows)[0])
            _vr, prompt = _refactor_prompt_for(rows, set(range(n_prefix)), rows)
            gradeable_tail = [
                i for i in range(n_prefix, len(rows)) if rows[i].level_after <= rows[i].level_before
            ]
            leaked = sorted(i for i in _delivered_rows(rows, prompt) if i >= n_prefix)
            # THE EXACT LAW, not a subset claim. Written as `issubset` first, which FAILED at
            # n=25 with 5 of 7 gradeable tail rows delivered -- because `_bounded_mismatches`
            # renders only the first five and a prefix-perfect engine has no prefix mismatches
            # to spend them on. That cap bounds the leak only when the gradeable tail exceeds
            # five rows, which happens at the n=25 LIVE window shape and never at the n=3..12
            # offline harness shape the null was actually measured on. Asserting the law keeps
            # both regimes honest instead of overclaiming one of them.
            assert leaked == gradeable_tail[:5], (
                f"n={n}: the flag-OFF path must still leak min(5, gradeable tail); "
                f"got {leaked} for gradeable tail {gradeable_tail}"
            )
            assert leaked, f"n={n}: OFF leaked nothing -- the control arm stopped being a control"

    def test_off_leaves_the_reinduction_refinement_corpus_at_the_full_corpus(self) -> None:
        """Trajectory invariance for OFF, measured at the CALLEE.

        `execute_bounded_llm_reinduction` is driven for real with a recording proposer, and the
        rows the refinement verifier is built from are read back off the actual call -- not
        inferred from reading the source. With the flag off it must be the whole corpus,
        acceptance rows included.
        """

        seen = _drive_reinduction(_corpus(12))
        assert seen["refinement_rows"] == 12, (
            f"flag-OFF refinement must still score the FULL corpus; got {seen['refinement_rows']}"
        )


# =========================================================================================
# ON ENABLES THE FIX
# =========================================================================================


class TestOnEnablesTheFix:
    def test_on_grades_acceptance_on_rows_the_refiner_never_sees(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE CONTRACT: no row that shapes refinement may also grade the final engine."""

        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
        for n in (5, 9, 12, 25):
            rows = _corpus(n)
            split = split_refinement_acceptance(rows)
            n_prefix = len(_split_prefix_heldout(rows)[0])
            _vr, prompt = _refactor_prompt_for(rows, set(range(n_prefix)), split.refinable)
            acceptance_idx = set(range(len(rows) - len(split.acceptance), len(rows)))
            assert not (_delivered_rows(rows, prompt) & acceptance_idx), (
                f"n={n}: an acceptance row's observed answer reached the refactor prompt"
            )

    def test_on_moves_the_ACCEPTANCE_GRADE_onto_the_reserved_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`select_trusted_world_model` must grade on the reserved block, not the legacy tail.

        Added after a deletion proof: reverting that one line inside the function left the whole
        suite GREEN, because every other ON assertion computed the split itself and never asked
        the shipped selector what it had graded. The engine below is deliberately chosen so the
        two splits DISAGREE -- right on rows 0..9, wrong on 10 and 11 -- otherwise the assertion
        would hold for both and prove nothing.
        """

        rows = _corpus(12)
        eng = _engine(rows, set(range(10)))
        cands = [WorldModelCandidate("c", eng, None)]
        legacy = float(
            select_trusted_world_model(
                list(rows), cands, hidden_state=True
            ).selected_score.heldout_accuracy
        )
        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
        sel = select_trusted_world_model(list(rows), cands, hidden_state=True)
        split = split_refinement_acceptance(rows)
        reserved = WorldModelVerifier(list(split.acceptance), hud_mask=None).score(eng)
        assert legacy == pytest.approx(2.0 / 3.0), (
            "the legacy tail no longer disagrees with the reserved block; pick a different "
            "engine or this assertion cannot distinguish the two splits"
        )
        assert float(sel.selected_score.heldout_accuracy) == pytest.approx(float(reserved.accuracy))
        assert float(sel.selected_score.heldout_accuracy) != pytest.approx(legacy)
        assert sel.acceptance_split_enabled is True
        assert sel.n_acceptance_gradeable == 1

    def test_on_keeps_the_counterexample_budget_design_i_would_have_destroyed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The measured reason design (ii) was chosen over design (i).

        Same engine, same round, three refinement corpora. Prefix-only (design (i)) yields ZERO
        counterexamples -- no CEGIS signal at all. The reserved split keeps a non-empty budget.
        """

        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
        for n in (9, 12, 25):
            rows = _corpus(n)
            n_prefix = len(_split_prefix_heldout(rows)[0])
            correct = set(range(n_prefix))
            split = split_refinement_acceptance(rows)
            design_i = WorldModelVerifier(rows[:n_prefix], hud_mask=None).score(
                _engine(rows, correct)
            )
            design_ii = WorldModelVerifier(list(split.refinable), hud_mask=None).score(
                _engine(rows, correct)
            )
            assert len(design_i.mismatches) == 0, (
                f"n={n}: prefix-only refinement was expected to be STARVED; "
                "if this ever finds counterexamples the design rationale needs re-measuring"
            )
            assert len(design_ii.mismatches) > 0, (
                f"n={n}: the reserved split must still supply counterexamples, else it is "
                "design (i) wearing a different name"
            )

    def test_on_wires_the_split_through_the_real_reinduction_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Availability is not delivery: read the refinement corpus off the real call."""

        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
        rows = _corpus(12)
        split = split_refinement_acceptance(rows)
        seen = _drive_reinduction(rows)
        assert seen["refinement_rows"] == len(split.refinable) == 10


class TestTheSplitItself:
    def test_the_partition_is_disjoint_and_total(self) -> None:
        """Purity is a property of the SPLIT, so assert it on the split, at every size."""

        for n in range(2, 41):
            rows = _corpus(n)
            split = split_refinement_acceptance(rows)
            ids_refine = [id(r) for r in split.refinable]
            ids_accept = [id(r) for r in split.acceptance]
            assert not (set(ids_refine) & set(ids_accept)), f"n={n}: blocks overlap"
            assert len(ids_refine) + len(ids_accept) == n, f"n={n}: not a partition"

    def test_the_acceptance_block_is_sized_in_GRADEABLE_rows(self) -> None:
        """Companion to `_n_gradeable`: the terminal level-up row is always LAST, so a block
        sized in RAW rows reproduces the unfalsifiable-gate bug inside the new split. At the
        four real window sizes whose entire legacy tail IS that one row (n=3 and n=4), the raw
        sizing gives 0 gradeable acceptance rows."""

        for n in (4, 5, 9, 12, 25):
            split = split_refinement_acceptance(_corpus(n))
            assert split.decidable is True, f"n={n} should be decidable"
            assert split.n_acceptance_gradeable >= 1, (
                f"n={n}: acceptance block has no gradeable row -- a perfect engine would score "
                "0.0 and be rejected, which is the C2 unfalsifiability bug relocated"
            )

    def test_undecidable_is_reported_not_scored_as_zero(self) -> None:
        """MISSING IS NOT ZERO. At n=3 with a terminal level-up (the r11l / vc33 shape) no
        disjoint split can supply a gradeable acceptance row without starving induction. That
        must be REPORTED, because `n_correct / max(1, n)` returns 0.0 for an ungradeable block
        and every aggregation downstream reads 0.0 as "the engine failed"."""

        split = split_refinement_acceptance(_corpus(3))
        assert split.decidable is False
        assert split.reason == "no_gradeable_acceptance_row_within_refinable_floor"
        assert split.n_acceptance_gradeable == 0

    def test_induce_prefix_is_clamped_at_the_n4_terminal_levelup_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE BUG THIS FIX SHIPPED WITH FIRST, kept as a regression.

        `_proposal_prefix` cuts at 1/3 and the acceptance block is the last ~1/6, so the induce
        rows are a subset of the refinable rows -- at 38 of the 39 sizes checked. At n=4 WITH a
        terminal level-up the gradeable grow loop extends acceptance to 2 rows while
        `_proposal_prefix` still returns 3, so row 2 would be in the induce prompt AND in the
        acceptance block. That is the measured shape of sp80 and ft09, 2 of the 13 real windows.
        """

        rows = _corpus(4)
        split = split_refinement_acceptance(rows)
        assert len(_proposal_prefix(rows)) == 3 and len(split.refinable) == 2, (
            "the n=4 overlap precondition changed; re-measure before trusting the clamp"
        )
        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
        seen = _drive_reinduction(rows)
        reserved = {id(r) for r in split.acceptance}
        assert not [r for r in seen["induce_rows"] if id(r) in reserved], (
            "an acceptance row reached the induce prompt at n=4"
        )
        assert len(seen["induce_rows"]) <= len(split.refinable)

    def test_a_caller_supplied_prompt_set_is_filtered_by_identity_not_by_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The identity filter, proven to bite by the case only it can catch.

        Added after a deletion proof showed the filter was DECORATIVE: on the default path the
        length clamp alone already removes the n=4 overlap, so replacing the filter with a no-op
        left the suite green. `proposal_transitions` need not be a prefix of anything -- here an
        acceptance row is FIRST and the list is shorter than the refinable block, so no
        length-based clamp can reach it. If the filter is ever removed as redundant, this goes red.
        """

        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
        rows = _corpus(12)
        split = split_refinement_acceptance(rows)
        reserved_row = split.acceptance[0]
        supplied = [reserved_row, rows[0], rows[1]]
        assert len(supplied) < len(split.refinable), "a length clamp must not be able to fix this"
        seen = _drive_reinduction(rows, proposal_transitions=supplied)
        assert [id(r) for r in seen["induce_rows"]] == [id(rows[0]), id(rows[1])], (
            "the reserved row survived into the induce prompt"
        )


# =========================================================================================
# DELIVERY PROBE -- instrument the callee, read the caller off the stack
# =========================================================================================


def _run_reinduction_rounds(
    rows: list[Transition], *, min_heldout_accuracy: float
) -> list[dict[str, Any]]:
    """Run the REAL loop and hand back its per-round record rows.

    Separate from `_drive_reinduction` because this one asserts on what the loop WROTE DOWN,
    and the two must not share a stub whose behaviour one of them tunes.
    """

    def floor_engine(grid: Any, _action: Any, _data: Any = None) -> Any:
        return np.asarray(grid).copy()

    class _Prop:
        def induce(self, *_a: Any, **_kw: Any) -> tuple[bool, str]:
            return True, "ok"

        def refactor(self, *_a: Any, **_kw: Any) -> tuple[bool, str]:
            return False, "stop"

    outcome = reinduction.execute_bounded_llm_reinduction(
        game="stub",
        transitions=rows,
        cell=8,
        root_grid=rows[0].grid.copy(),
        proposer=_Prop(),
        candidate_provider=lambda e, g: [WorldModelCandidate("floor", e, g)],
        load_engine=lambda _g: (floor_engine, lambda _grid: False),
        plan_in_model=lambda _e, _g, _s: None,
        min_heldout_accuracy=min_heldout_accuracy,
    )
    return list(outcome.rounds)


def _drive_reinduction(
    rows: list[Transition], *, proposal_transitions: list[Transition] | None = None
) -> dict[str, Any]:
    """Run the REAL `execute_bounded_llm_reinduction` and record what each channel was handed.

    The proposer fails at round 1 on purpose: this measures the INPUTS the loop assembles, and
    a fuller harness would only add ways for the assertion to pass for an unrelated reason. The
    refinement corpus is captured by wrapping `WorldModelVerifier` in the reinduction module's
    own namespace, which is what that code path actually resolves.
    """

    seen: dict[str, Any] = {"induce_rows": [], "refinement_rows": None, "induce_caller": []}

    class _RecordingProposer:
        def induce(self, _game: str, trans: Any, _cell: int, **_kw: Any) -> tuple[bool, str]:
            seen["induce_rows"] = list(trans)
            seen["induce_caller"] = [
                f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}"
                for f in traceback.extract_stack()[-3:-1]
            ]
            # TRUE, not False. The first version of this probe returned False and the refinement
            # corpus came back None on every arm -- the loop breaks at `proposer_failed` before
            # the refinement block is ever reached, so the probe was measuring nothing and the
            # OFF/ON assertions would have compared None to None. Round 1 must SUCCEED, be
            # graded, and FAIL the gate; only then does the counterexample path run.
            return True, "recording_stub_round_1_ok"

        def refactor(self, _game: str, _vr: Any) -> tuple[bool, str]:
            return False, "recording_stub_stops_after_round_2"

    def _floor_engine(grid: Any, _action: Any, _data: Any = None) -> Any:
        """Predicts nothing, so `heldout_accuracy` misses the 1.0 threshold and the loop takes
        the `not accepted` branch that builds the refinement corpus."""

        return np.asarray(grid).copy()

    real_verifier = reinduction.WorldModelVerifier

    class _RecordingVerifier(real_verifier):  # type: ignore[misc,valid-type]
        def __init__(self, transitions: Any, *a: Any, **kw: Any) -> None:
            seen["refinement_rows"] = len(list(transitions))
            super().__init__(transitions, *a, **kw)

    reinduction.WorldModelVerifier = _RecordingVerifier  # type: ignore[assignment]
    try:
        reinduction.execute_bounded_llm_reinduction(
            game="stub",
            transitions=rows,
            cell=8,
            root_grid=rows[0].grid.copy(),
            proposer=_RecordingProposer(),
            candidate_provider=lambda e, g: [WorldModelCandidate("floor", e, g)],
            load_engine=lambda _g: (_floor_engine, lambda _grid: False),
            plan_in_model=lambda _e, _g, _s: None,
            min_heldout_accuracy=1.0,
            **(
                {}
                if proposal_transitions is None
                else {"proposal_transitions": proposal_transitions}
            ),
        )
    finally:
        reinduction.WorldModelVerifier = real_verifier  # type: ignore[assignment]
    assert seen["induce_caller"], "the induce channel was never reached"
    assert any("execute_bounded_llm_reinduction" in frame for frame in seen["induce_caller"]), (
        f"induce was not called from the reinduction loop: {seen['induce_caller']}"
    )
    return seen


def test_off_and_on_disagree_on_the_refinement_corpus_and_that_is_the_whole_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag whose two arms are byte-identical is decorative. Assert they differ, on the real
    call, at the real window size -- and that the OFF arm is the LARGER (leaking) one."""

    rows = _corpus(12)
    monkeypatch.delenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", raising=False)
    off = _drive_reinduction(rows)["refinement_rows"]
    monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
    on = _drive_reinduction(rows)["refinement_rows"]
    assert off == 12 and on == 10 and on < off


def test_an_undecidable_gate_refuses_to_admit(monkeypatch: pytest.MonkeyPatch) -> None:
    """UNDECIDABLE MUST NOT ADMIT, and this is not a theoretical guard.

    `execute_bounded_llm_reinduction`'s `min_heldout_accuracy` DEFAULTS TO 0.0, and `0.0 >= 0.0`
    is True -- so a caller taking the default, on a corpus whose acceptance block has nothing
    gradeable in it, would ACCEPT every engine it was handed. The live agent passes 1.0 and is
    therefore safe today by coincidence of its threshold, not by construction. n=3 with a
    terminal level-up is the r11l / vc33 shape.
    """

    monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
    rows = _corpus(3)
    assert split_refinement_acceptance(rows).decidable is False
    result = _run_reinduction_rounds(rows, min_heldout_accuracy=0.0)
    assert result, "no round was recorded"
    first = result[0]
    assert first.get("acceptance_split_decidable") is False
    assert first.get("acceptance_split_reason") == (
        "no_gradeable_acceptance_row_within_refinable_floor"
    )
    assert first.get("accepted_by_heldout_verifier") is False, (
        "an UNDECIDABLE acceptance gate admitted an engine at the default 0.0 threshold"
    )


def test_off_records_nothing_about_a_split_it_did_not_take() -> None:
    """Trajectory invariance for OFF, at the RECORD level. Adding fields unconditionally would
    make every flag-off artifact differ from the ones the prior measurements were taken on,
    which is the same interpretability loss as changing the behaviour itself."""

    rows = _corpus(12)
    first = _run_reinduction_rounds(rows, min_heldout_accuracy=1.0)[0]
    for key in (
        "acceptance_split_decidable",
        "acceptance_split_reason",
        "acceptance_split_gradeable_rows",
        "refinement_corpus_rows",
    ):
        assert key not in first, f"flag-OFF round row gained {key!r}"


# =========================================================================================
# THE THIRD CHANNEL: the agent's NON-CEGIS induce branch, which is the DEFAULT path
# =========================================================================================


class TestTheAgentInduceBranch:
    """`_induce_and_plan`'s `execute_bounded_llm_reinduction` branch runs only for a level-up
    reinduction (`attempt['reason'] == 'level_up_reinduction' or next_level_episode`); every
    other induction lands on the plain branch, which handed the proposer the WHOLE corpus and
    then graded acceptance on the tail of that same list a few dozen lines below."""

    def _drive(self, win_row: Any, rows: list[Transition]) -> list[Any]:
        """The gated block verbatim, kept in one place so drift against the shipped source is a
        failure rather than a silently stale copy (the sibling win-transition gate's pattern)."""

        import carnot.agentic.arc_competition_agent as mod

        seen: list[Any] = []

        class _Prop:
            def induce(self, _game: str, trans: Any, _cell: int, **_kw: Any) -> tuple[bool, str]:
                seen.append(list(trans))
                return False, "recording_stub"

        induce_rows = rows
        if mod._cegis_accept_split_enabled():
            induce_rows = mod._split_refinement_acceptance(rows).refinable
        _Prop().induce("vc33", induce_rows, 8)
        return seen[0]

    def test_off_still_shows_the_proposer_the_whole_corpus(self) -> None:
        rows = _corpus(12)
        assert len(self._drive(None, rows)) == 12

    def test_on_withholds_the_acceptance_rows_from_the_induce_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
        rows = _corpus(12)
        shown = self._drive(None, rows)
        reserved = {id(r) for r in split_refinement_acceptance(rows).acceptance}
        assert len(shown) == 10
        assert not [r for r in shown if id(r) in reserved]

    def test_the_shipped_source_actually_reads_the_gate(self) -> None:
        """A gate the call site does not consult is the "trusted and silent guard" failure this
        project has already been bitten by. Crude, and the alternative is worse."""

        import inspect

        import carnot.agentic.arc_competition_agent as mod

        src = inspect.getsource(mod.E3AgentPolicy._induce_and_plan)
        assert "_cegis_accept_split_enabled()" in src, (
            "_induce_and_plan no longer consults the purity gate -- the plain induce branch is "
            "handing the proposer the acceptance rows again"
        )
        assert (
            "self._proposer().induce(\n                self.short,\n                _induce_rows,"
            in src
        ), "the plain induce call site no longer passes the gated row list"
        # PINNED VERBATIM after a deletion proof: mutating the assignment to
        # `_induce_rows = active_transitions` (i.e. removing the withholding while leaving
        # the `if` in place) left the suite GREEN, because the behavioural test above drives
        # a local copy of the block rather than the shipped method. Until a harness exists
        # that can run `_induce_and_plan` end to end, the assignment itself is what must be
        # pinned -- crude, and strictly better than a pattern nothing checks.
        assert (
            "                _induce_rows = _split_refinement_acceptance(\n"
            "                    active_transitions\n"
            "                ).refinable"
            in src
            or "_induce_rows = _split_refinement_acceptance(active_transitions).refinable" in src
        ), "the gated branch no longer withholds the acceptance rows from the induce prompt"
