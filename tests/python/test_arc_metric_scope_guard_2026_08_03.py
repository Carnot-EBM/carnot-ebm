"""REQ-ARC-WMTE-6090: the cross-scope subtraction guard, tested on the real incident.

WHAT IS PINNED HERE, AND WHY IT IS NOT A SYNTHETIC CASE. The numbers in these tests are not
fixtures invented to exercise the code. They are re-derived, every run, from the two committed
shards that produced the defect:

    results/exp5764_gemma31b_singleshot_shard.jsonl
    results/exp5766_gemma31b_cegis_refinement_shard.jsonl

and checked against the block those shards were rendered into:

    results/experiment_5766_gemma31b_cegis_refinement_ab.json
        comparison_to_gemma31b_singleshot_baseline

`test_off_reproduces_the_shipped_comparison_bit_for_bit` rebuilds all thirteen per-game deltas
and the pooled `-0.318658` by routing every subtraction through `compare_scoped` with the guard
OFF, and asserts they equal the recorded values exactly. That is the both-directions contract's
OFF half: the shipped arithmetic, bug included, so the historical record stays interpretable.
`test_on_refuses_the_exact_invalid_comparison` runs the identical rebuild with the guard ON and
asserts it refuses.

ONE SUBTLETY THAT DECIDES THE WHOLE DESIGN, pinned by
`test_on_permits_the_correct_like_for_like_comparison`. The guard keys on the RELATION a score
stands in to its prompt evidence, not on the row set. exp5764 grades the whole window it was
prompted on; exp5766 round 0's `prefix_accuracy` grades the two-thirds prefix it was prompted
on. Different row sets, same relation -- both are fit scores -- and their comparison
(0.378487 vs 0.319444, sign test p = 1.0) is the CORRECT one. A guard keyed on the row-set name
would have refused it, which would have blocked the right answer along with the wrong one.

MISSING IS NOT ZERO. exp5764's vc33 trial 0 errored (`load_engine: FileNotFoundError`) and
recorded `heldout_accuracy: null`. It is EXCLUDED from vc33's mean, not counted as 0.0 --
which is what makes vc33's baseline 1.0 rather than 0.667, and what makes the pooled baseline
0.378487 rather than 0.352846. The exclusion is asserted explicitly rather than left implicit,
because silently counting it as zero reproduces the artifact's number for the wrong reason.

SCENARIO-ARC-WMTE-6090-METRIC-SCOPE-GUARD
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.agentic.arc_metric_scope import (
    DISJOINT_TAIL,
    HELDOUT_METRIC_SCOPES,
    IN_SAMPLE,
    OUT_OF_SAMPLE,
    PROMPT_OVERLAPPING_PREFIX,
    WHOLE_WINDOW,
    MetricScopeMismatch,
    compare_scoped,
    metric_scope_guard_enabled,
    relation_of,
    scope_for,
)
from carnot.agentic.arc_metric_scope import (
    _METRIC_SCOPE_GUARD_DEFAULT,  # noqa: PLC2701 -- pinned deliberately, see below
)

_ALL_SCOPES = sorted([WHOLE_WINDOW, DISJOINT_TAIL, PROMPT_OVERLAPPING_PREFIX])

REPO = Path(__file__).resolve().parents[2]
SHARD_5764 = REPO / "results" / "exp5764_gemma31b_singleshot_shard.jsonl"
SHARD_5766 = REPO / "results" / "exp5766_gemma31b_cegis_refinement_shard.jsonl"
ARTIFACT_5766 = REPO / "results" / "experiment_5766_gemma31b_cegis_refinement_ab.json"


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from the SHIPPED environment so a stray export in the operator's shell cannot make
    a default-OFF assertion pass for the wrong reason."""

    monkeypatch.delenv("CARNOT_ARC_METRIC_SCOPE_GUARD", raising=False)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _singleshot_means_5764() -> dict[str, float]:
    """exp5764's per-game baseline, EXCLUDING the errored null cell (missing is not zero)."""

    by_game: dict[str, list[float]] = collections.defaultdict(list)
    for row in _load_jsonl(SHARD_5764):
        value = row.get("heldout_accuracy")
        if value is None:
            continue
        by_game[row["game"]].append(float(value))
    return {game: round(_mean(vals), 6) for game, vals in by_game.items()}


def _cegis_best_means_5766() -> dict[str, float]:
    """exp5766's per-game best-achieved heldout across round 0 and every refactor round."""

    by_game: dict[str, list[float]] = collections.defaultdict(list)
    for row in _load_jsonl(SHARD_5766):
        scored = [
            float(rd["heldout_accuracy"])
            for rd in row["rounds"]
            if rd.get("heldout_accuracy") is not None
        ]
        by_game[row["game"]].append(max(scored) if scored else 0.0)
    return {game: round(_mean(vals), 6) for game, vals in by_game.items()}


def _round0_prefix_means_5766() -> dict[str, float]:
    """exp5766 round 0's `prefix_accuracy` -- the field that was on disk and never read."""

    by_game: dict[str, list[float]] = collections.defaultdict(list)
    for row in _load_jsonl(SHARD_5766):
        for rd in row["rounds"]:
            if rd["round"] == 1 and rd["action"] == "induce":
                by_game[row["game"]].append(float(rd["prefix_accuracy"] or 0.0))
    return {game: round(_mean(vals), 6) for game, vals in by_game.items()}


def _recorded_comparison() -> dict[str, Any]:
    return json.loads(ARTIFACT_5766.read_text())["comparison_to_gemma31b_singleshot_baseline"]


class TestTheDefaultIsOff:
    def test_flag_defaults_off(self) -> None:
        assert metric_scope_guard_enabled() is False

    def test_the_module_constant_is_off(self) -> None:
        """Pinned separately from the resolver: reading the env correctly while defaulting the
        constant to "1" would still ship an armed refusal."""

        assert _METRIC_SCOPE_GUARD_DEFAULT == "0"

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_truthy_values_enable(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_METRIC_SCOPE_GUARD", raw)
        assert metric_scope_guard_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "banana"])
    def test_anything_else_stays_off(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail-closed on garbage: a typo'd export must not arm a refusal mid-run."""

        monkeypatch.setenv("CARNOT_ARC_METRIC_SCOPE_GUARD", raw)
        assert metric_scope_guard_enabled() is False


class TestTheExactIncident:
    """The 2026-08-03 reproduction, rebuilt from the shards on every run."""

    def test_the_shards_reproduce_the_reported_pooled_values(self) -> None:
        """Guards the guard's own premise. If the shards ever stop reproducing 0.378487 /
        0.059829 the tests below would still pass while measuring something else entirely."""

        recorded = _recorded_comparison()
        singleshot = _singleshot_means_5764()
        best = _cegis_best_means_5766()
        assert round(_mean(list(singleshot.values())), 6) == pytest.approx(
            recorded["gemma31b_singleshot_5764_pooled_heldout"]
        )
        assert round(_mean(list(best.values())), 6) == pytest.approx(
            recorded["cegis_best_achieved_pooled_heldout"]
        )

    def test_the_errored_cell_is_excluded_not_zeroed(self) -> None:
        """MISSING IS NOT ZERO. vc33 trial 0 errored with a null score; counting it as 0.0 would
        put vc33's baseline at 0.666667 and the pooled baseline at 0.352846 -- close enough to
        look right, wrong for a reason no reader would see."""

        rows = _load_jsonl(SHARD_5764)
        errored = [r for r in rows if r.get("heldout_accuracy") is None]
        assert [(r["game"], r["trial"]) for r in errored] == [("vc33", 0)]
        assert errored[0]["error"].startswith("load_engine: FileNotFoundError")
        assert _singleshot_means_5764()["vc33"] == 1.0

    def test_off_reproduces_the_shipped_comparison_bit_for_bit(self) -> None:
        """OFF HALF OF THE CONTRACT. Every per-game delta and the pooled -0.318658, rebuilt
        through `compare_scoped`, must equal what the artifact recorded -- exactly, not
        approximately. The subtraction is incommensurable and the guard reproduces it anyway,
        because that is what keeps the historical record re-derivable."""

        recorded = _recorded_comparison()
        singleshot = _singleshot_means_5764()
        best = _cegis_best_means_5766()

        deltas: dict[str, float] = {}
        for game in sorted(recorded["per_game"]):
            deltas[game] = round(
                compare_scoped(
                    best[game],
                    scope_for("exp5766", "rounds[].heldout_accuracy"),
                    singleshot[game],
                    scope_for("exp5764", "heldout_accuracy"),
                    label=f"cegis_best_minus_singleshot[{game}]",
                ),
                6,
            )

        expected = {
            game: cell["cegis_best_minus_singleshot"] for game, cell in recorded["per_game"].items()
        }
        assert deltas == expected
        assert (
            round(_mean(list(deltas.values())), 6)
            == (recorded["pooled_delta_cegis_best_minus_singleshot"])
        )
        assert recorded["pooled_delta_cegis_best_minus_singleshot"] == -0.318658

    def test_on_refuses_the_exact_invalid_comparison(self) -> None:
        """ON HALF OF THE CONTRACT. The same rebuild, guard armed, refuses -- and the message
        names both relations, so a reader hitting it learns what was actually mixed."""

        singleshot = _singleshot_means_5764()
        best = _cegis_best_means_5766()
        with pytest.raises(MetricScopeMismatch) as excinfo:
            for game in sorted(best):
                compare_scoped(
                    best[game],
                    scope_for("exp5766", "rounds[].heldout_accuracy"),
                    singleshot[game],
                    scope_for("exp5764", "heldout_accuracy"),
                    label=f"cegis_best_minus_singleshot[{game}]",
                    enabled=True,
                )
        assert excinfo.value.treatment_scope == DISJOINT_TAIL
        assert excinfo.value.baseline_scope == WHOLE_WINDOW
        assert excinfo.value.treatment_relation == OUT_OF_SAMPLE
        assert excinfo.value.baseline_relation == IN_SAMPLE

    def test_on_permits_the_correct_like_for_like_comparison(self) -> None:
        """THE GUARD MUST NOT BE A SLEDGEHAMMER. exp5764's whole-window fit and exp5766 round
        0's prefix fit grade DIFFERENT row sets in the SAME relation to their prompts, so their
        comparison is the valid one -- 0.378487 vs 0.319444. Armed, the guard permits it. A
        guard keyed on the row-set name instead of the relation would refuse here and would
        have suppressed the correct answer along with the wrong one."""

        singleshot = _singleshot_means_5764()
        prefix = _round0_prefix_means_5766()
        assert round(_mean(list(singleshot.values())), 6) == 0.378487
        assert round(_mean(list(prefix.values())), 6) == 0.319444

        deltas = [
            compare_scoped(
                prefix[game],
                scope_for("exp5766", "rounds[].prefix_accuracy"),
                singleshot[game],
                scope_for("exp5764", "heldout_accuracy"),
                label=f"like_for_like[{game}]",
                enabled=True,
            )
            for game in sorted(singleshot)
        ]
        # The like-for-like gap is real but small, and points the way the one-third evidence
        # handicap predicts. It is NOT asserted to be zero.
        assert round(_mean(deltas), 6) == round(0.319444 - 0.378487, 6)


class TestOffIsAStrictPassthrough:
    """Trajectory invariance for the OFF arm: with the guard off the function must be
    indistinguishable from the bare subtraction it replaced, for every input it accepts."""

    @pytest.mark.parametrize("treatment_scope", _ALL_SCOPES)
    @pytest.mark.parametrize("baseline_scope", _ALL_SCOPES)
    def test_off_never_raises_and_equals_bare_subtraction(
        self, treatment_scope: str, baseline_scope: str
    ) -> None:
        for treatment in (0.0, 0.059829, 0.319444, 0.378487, 1.0, -0.5):
            for baseline in (0.0, 0.148133, 0.378487, 1.0):
                assert (
                    compare_scoped(treatment, treatment_scope, baseline, baseline_scope)
                    == treatment - baseline
                )

    def test_off_leaves_the_environment_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A guard that mutated process state to remember it had fired would change the
        trajectory of anything downstream that reads the same state."""

        import os

        before = dict(os.environ)
        compare_scoped(0.5, DISJOINT_TAIL, 0.25, WHOLE_WINDOW)
        assert dict(os.environ) == before

    def test_the_explicit_enabled_false_override_also_passes_through(self) -> None:
        """`enabled=False` must be the OFF arm even when the env says otherwise, or an A/B
        harness cannot pin its control arm against a process-wide export."""

        assert (
            compare_scoped(0.059829, DISJOINT_TAIL, 0.378487, WHOLE_WINDOW, enabled=False)
            == 0.059829 - 0.378487
        )


class TestTheTagsDescribeTheCode:
    """A scope tag is a claim about what the code does. These assertions run the real functions
    so the claim cannot drift away from the implementation silently."""

    @pytest.mark.parametrize("n_rows", [3, 4, 5, 9, 12, 25])
    def test_the_prefix_and_the_tail_partition_the_window_at_the_same_index(
        self, n_rows: int
    ) -> None:
        """`disjoint_tail` and `prompt_overlapping_prefix` are only honest names if
        `_proposal_prefix` (what goes in the prompt) and `_split_prefix_heldout` (what gets
        graded) cut at the SAME place. They are written independently, in two modules, with two
        different arithmetic spellings of one third."""

        from carnot.agentic.arc_llm_reinduction import _proposal_prefix
        from carnot.agentic.arc_world_model_trust_energy import _split_prefix_heldout

        rows = list(range(n_rows))
        prompt_rows = _proposal_prefix(rows)
        graded_prefix, graded_tail = _split_prefix_heldout(rows)

        assert prompt_rows == graded_prefix, "the prompt prefix is the graded prefix"
        assert set(prompt_rows).isdisjoint(graded_tail), "the tail is out of sample"
        assert prompt_rows + graded_tail == rows, "and together they are the whole window"
        assert graded_tail, "a non-empty tail, or `disjoint_tail` grades nothing"

    def test_relations_are_assigned_by_prompt_overlap_not_by_row_count(self) -> None:
        assert relation_of(WHOLE_WINDOW) == IN_SAMPLE
        assert relation_of(PROMPT_OVERLAPPING_PREFIX) == IN_SAMPLE
        assert relation_of(DISJOINT_TAIL) == OUT_OF_SAMPLE

    def test_every_registered_scope_is_a_known_scope(self) -> None:
        """A registry entry naming a scope the guard cannot resolve would raise at the
        subtraction, in a harness, rather than here."""

        for key, scope in HELDOUT_METRIC_SCOPES.items():
            assert relation_of(scope) in {IN_SAMPLE, OUT_OF_SAMPLE}, key

    def test_the_two_exp5766_round_fields_are_registered_as_opposites(self) -> None:
        """The whole incident is that these two sat side by side on the same round record,
        computed from the same engine, and were treated as interchangeable."""

        assert scope_for("exp5766", "rounds[].heldout_accuracy") == DISJOINT_TAIL
        assert scope_for("exp5766", "rounds[].prefix_accuracy") == PROMPT_OVERLAPPING_PREFIX
        assert relation_of(scope_for("exp5766", "rounds[].heldout_accuracy")) != relation_of(
            scope_for("exp5766", "rounds[].prefix_accuracy")
        )

    def test_an_unregistered_field_refuses_rather_than_defaulting(self) -> None:
        with pytest.raises(KeyError, match="no recorded metric scope"):
            scope_for("exp9999", "heldout_accuracy")

    @pytest.mark.parametrize("bad", ["held_out", "tail", "", "WHOLE_WINDOW"])
    def test_an_unknown_scope_name_is_refused_in_both_directions(self, bad: str) -> None:
        """Fail-closed on a typo'd tag whether the guard is on or off. A tag the guard cannot
        resolve is a pattern list narrower than the concept it protects."""

        for enabled in (False, True):
            with pytest.raises(ValueError, match="unknown metric scope"):
                compare_scoped(0.5, bad, 0.25, WHOLE_WINDOW, enabled=enabled)
            with pytest.raises(ValueError, match="unknown metric scope"):
                compare_scoped(0.5, WHOLE_WINDOW, 0.25, bad, enabled=enabled)
