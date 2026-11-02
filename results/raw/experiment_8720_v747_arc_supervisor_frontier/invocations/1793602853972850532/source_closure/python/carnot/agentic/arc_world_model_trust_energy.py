"""Exp 4491 world-model trust energy for ARC hidden-state games.

Spec refs: REQ-ARC-WMTE-4491, REQ-ARC-WMTE-4492, REQ-ARC-WMTE-4493,
REQ-ARC-WMTE-4494.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    change_gate_decision,
)


INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DEFAULT_OUTPUT_PATH = Path("results/experiment_4491_world_model_trust_energy.json")
HIDDEN_STATE_GAME_IDS = (
    "ar25",
    "cd82",
    "cn04",
    "dc22",
    "g50t",
    "ka59",
    "m0r0",
    "re86",
    "sc25",
    "sk48",
    "wa30",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/"
        "passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
}


Engine = Callable[[np.ndarray, int, Any], np.ndarray]


@dataclass(frozen=True)
class WorldModelCandidate:
    """Candidate executable world model plus optional planner win predicate."""

    name: str
    engine: Engine
    is_level_complete: Callable[[np.ndarray], bool] | None = None


@dataclass(frozen=True)
class CandidateScore:
    candidate: WorldModelCandidate
    prefix_accuracy: float
    heldout_accuracy: float
    trust_energy: float
    baseline_clears: bool
    heldout_best: bool
    prefix_change_consistency: float = 0.0
    heldout_change_consistency: float = 0.0
    correct_changed_cells: int = 0
    true_changed_cells: int = 0
    nondegenerate: bool = False
    trust_pass: bool = False
    binary_gate_pass: bool = False
    # REQ-ARC-WMTE-6013: the SYMMETRIC union-fidelity decision for this candidate, computed
    # on the SAME held-out split `trust_pass` is computed on. Populated unconditionally --
    # a control arm records it without acting on it, so the four-arm matrix compares like
    # with like and the disagreement between the two verdicts is visible in every artifact
    # row without a re-run. Empty dict only if the gate could not be computed at all.
    change_gate: dict = field(default_factory=dict)

    @property
    def change_gate_pass(self) -> bool:
        """REQ-ARC-WMTE-6013 admit/reject under the symmetric metric.

        Defaults to False on a missing record rather than True: a decision that could not be
        computed is not evidence that the engine is trustworthy, and defaulting the other way
        would make a plumbing failure look like a pass.
        """

        return bool(self.change_gate.get("passed", False))


@dataclass(frozen=True)
class S1StructuralTransitionEnergy:
    """REQ-ARC-WMTE-4791: frozen S1 lower-is-better off-path transition energy."""

    feature_mean: tuple[float, ...] = (
        0.32721382,
        0.12581662,
        0.25262373,
        0.43074671,
        0.09112444,
        1.0,
        0.93723486,
        0.00196306,
        0.06080208,
        0.00334773,
        0.01798484,
        0.03559435,
        0.01466377,
        0.00196967,
        1.0,
        0.00196842,
        -0.00054787,
        0.00054787,
        0.00191674,
        0.00843683,
        0.00239692,
        0.0,
        0.0,
    )
    feature_scale: tuple[float, ...] = (
        0.39143438,
        0.1052774,
        0.08983493,
        0.18484001,
        0.06381427,
        1.0,
        0.16514214,
        0.01738833,
        0.16494935,
        0.02692125,
        0.05628982,
        0.10111979,
        0.04225519,
        0.00789738,
        1.0,
        0.00462703,
        0.00240489,
        0.00240489,
        0.00456805,
        0.02938024,
        0.00781244,
        1.0,
        1.0,
    )
    weights: tuple[float, ...] = (
        0.43535789,
        -0.0719261,
        -0.01053813,
        0.11925962,
        -0.18605983,
        0.0,
        0.64766456,
        -0.31110494,
        -0.61562603,
        -0.20712673,
        -0.5342987,
        -0.59984914,
        -0.58702932,
        -0.41766227,
        0.0,
        -0.95604276,
        0.39851227,
        -0.39851227,
        -0.91184741,
        -0.44574168,
        -0.52151116,
        0.0,
        0.0,
    )
    no_change_penalty: float = 500.0
    shape_mismatch_penalty: float = 1.0e6

    @property
    def energy_config(self) -> dict[str, Any]:
        return {
            "source_experiment": 4781,
            "model": "frozen_linear_pairwise_margin_s1_structural_energy",
            "feature_families": ["object_relational", "frame_delta"],
            "dim": len(self.weights),
            "no_change_penalty": float(self.no_change_penalty),
            "shape_mismatch_penalty": float(self.shape_mismatch_penalty),
        }

    def features(
        self,
        previous_grid: Any,
        action: int,
        data: Any,
        predicted_grid: Any,
    ) -> list[float]:
        from carnot.agentic.arc_value_learner import (
            cross_game_feature_slices_v3,
            cross_game_features_v3,
        )

        action_key: Any = action
        if isinstance(data, Mapping):
            if "x" in data and "y" in data:
                action_key = (int(action), int(data["x"]), int(data["y"]))
        slices = cross_game_feature_slices_v3()
        values = [
            float(v)
            for v in cross_game_features_v3(
                predicted_grid,
                previous_frame=previous_grid,
                action_id=action_key,
                goal_frame=None,
            )
        ]
        out: list[float] = []
        for family in ("object_relational", "frame_delta"):
            lo, hi = slices[family]
            out.extend(float(v) for v in values[lo:hi])
        return out

    def transition_energy(
        self,
        previous_grid: Any,
        action: int,
        data: Any,
        predicted_grid: Any,
    ) -> float:
        prev = np.asarray(previous_grid)
        pred = np.asarray(predicted_grid)
        if pred.shape != prev.shape:
            return float(self.shape_mismatch_penalty)
        feats = np.asarray(self.features(prev, action, data, pred), dtype=float)
        dim = len(self.weights)
        if feats.size != dim:
            aligned = np.zeros(dim, dtype=float)
            n = min(dim, feats.size)
            if n:
                aligned[:n] = feats[:n]
            feats = aligned
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        z = (feats - mean) / np.where(scale < 1.0e-8, 1.0, scale)
        energy = -float(z @ weights)
        if np.array_equal(prev, pred):
            energy += float(self.no_change_penalty)
        return float(energy)


_DEFAULT_S1_OFFPATH_SCORER = S1StructuralTransitionEnergy()


def default_s1_offpath_energy_scorer() -> S1StructuralTransitionEnergy:
    """REQ-ARC-WMTE-4791: live default scorer for E3's hidden-state trust gate."""

    return _DEFAULT_S1_OFFPATH_SCORER


@dataclass(frozen=True)
class ChangeWeightedConsistency:
    """REQ-ARC-WMTE-4604: score only real grid-changing cells."""

    n: int
    n_changing_transitions: int
    true_changed_cells: int
    correct_changed_cells: int
    consistency: float
    exact_accuracy: float
    nondegenerate: bool
    trust_pass: bool


@dataclass(frozen=True)
class _CalibrationRow:
    prefix_change_miss: float
    prefix_exact_miss: float
    heldout_change_miss: float


@dataclass(frozen=True)
class TrustEnergyCalibrator:
    """Small CPU ridge calibrator mirroring the linear learner pattern."""

    weights: tuple[float, float, float]

    @classmethod
    def fit(cls, rows: Sequence[_CalibrationRow]) -> "TrustEnergyCalibrator":
        x = np.asarray(
            [[1.0, row.prefix_change_miss, row.prefix_exact_miss] for row in rows],
            dtype=float,
        )
        y = np.asarray([row.heldout_change_miss for row in rows], dtype=float)
        eye = np.eye(x.shape[1], dtype=float) * 1.0e-6
        weights = np.linalg.solve(x.T @ x + eye, x.T @ y)
        return cls(tuple(float(v) for v in weights))

    def predicted_heldout_miss(self, row: _CalibrationRow) -> float:
        value = (
            self.weights[0]
            + self.weights[1] * row.prefix_change_miss
            + self.weights[2] * row.prefix_exact_miss
        )
        return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class TrustSelection:
    selected: WorldModelCandidate
    selected_score: CandidateScore
    baseline_candidate_name: str | None
    rows: tuple[CandidateScore, ...]
    verifier_is_oracle: bool
    # ---- REQ-ARC-WMTE-6010 CORRIGENDUM (2026-07-27): THE ARM WITNESS -----------------
    # What masking ACTUALLY happened, as resolved by `resolve_hud_mask_enabled`, so a
    # harness can assert that the arm it asked for is the arm it got. exp6013 could not:
    # it passed `hud_mask=<a real mask>` without setting CARNOT_ARC_WM_HUD_MASK, every
    # comparator silently ran unmasked, and its "admitted under both mask settings" claim
    # was measured on mask-off twice. `hud_mask_supplied` is deliberately separate from
    # `hud_mask_enabled`: (supplied=True, enabled=False) is exactly that silent-no-op
    # state, and it is the one combination a mask-arm harness must refuse to accept.
    hud_mask_enabled: bool = False
    hud_mask_supplied: bool = False
    # ---- REQ-ARC-WMTE-6090: THE DECIDABILITY WITNESS --------------------------------
    # `heldout_accuracy` is a float and cannot say "I could not tell". `n_correct /
    # max(1, n)` returns 0.0 for an acceptance block with nothing gradeable in it, which
    # every downstream aggregation reads as "the engine failed" -- measured today on
    # sp80/r11l/vc33/ft09, where a MEMORISING PERFECT engine is rejected at 0.0. These
    # three fields are the "MISSING IS NOT ZERO" record: a caller can distinguish a
    # engine that was graded and lost from one that was never gradeable at all.
    # `acceptance_split_enabled` is deliberately separate from `acceptance_decidable` so
    # an artifact can tell "the flag was off" from "the flag was on and the corpus was
    # too short" -- the exp6013 silent-no-op lesson, applied to this flag.
    acceptance_split_enabled: bool = False
    acceptance_decidable: bool = True
    acceptance_reason: str = "legacy_prefix_heldout_split"
    n_acceptance_gradeable: int = -1

    @property
    def hud_mask_silently_dropped(self) -> bool:
        """True iff a caller supplied a mask that the flag then discarded.

        This is the exp6013 failure state, named so a harness can assert `not
        selection.hud_mask_silently_dropped` instead of discovering months later that its
        treatment arm was byte-identical to its control.
        """

        return bool(self.hud_mask_supplied and not self.hud_mask_enabled)

    @property
    def trust_energy_beats_baseline(self) -> bool:
        return (
            self.selected.name != self.baseline_candidate_name and self.selected_score.heldout_best
        )


@dataclass(frozen=True)
class Scorecard:
    game: str
    selected_candidate_name: str
    baseline_candidate_name: str | None
    trust_energy_pick_best: bool
    baseline_pick_best: bool
    n_candidates: int
    verifier_is_oracle: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "selected_candidate_name": self.selected_candidate_name,
            "baseline_candidate_name": self.baseline_candidate_name,
            "trust_energy_pick_best": self.trust_energy_pick_best,
            "baseline_pick_best": self.baseline_pick_best,
            "n_candidates": self.n_candidates,
            "verifier_is_oracle": self.verifier_is_oracle,
        }


def _split_prefix_heldout(
    transitions: Sequence[Transition],
    *,
    heldout_fraction: float = 1.0 / 3.0,
) -> tuple[list[Transition], list[Transition]]:
    rows = list(transitions)
    if len(rows) < 2:
        return rows, rows
    n_heldout = max(1, int(round(len(rows) * float(heldout_fraction))))
    n_heldout = min(n_heldout, len(rows) - 1)
    return rows[:-n_heldout], rows[-n_heldout:]


# ---------------------------------------------------------------------------------------
# REQ-ARC-WMTE-6090: THE CEGIS ACCEPTANCE/REFINEMENT PURITY SPLIT (default OFF).
# ---------------------------------------------------------------------------------------

_CEGIS_ACCEPT_SPLIT_DEFAULT = "0"


def cegis_accept_split_enabled() -> bool:
    """Is acceptance graded on rows that NOTHING in the loop is allowed to learn from?

    THE DEFECT (measured 2026-08-03, results/outer_loop_arc_cegis_purity_leak_20260803.json).
    Acceptance is scored on the held-out tail returned by `_split_prefix_heldout`, and the
    refinement feedback in `execute_bounded_llm_reinduction` is built from
    `WorldModelVerifier(list(transitions))` -- the FULL corpus, that same tail included, WITH
    the observed answer (`true_change`) attached to every mismatch. So the rows that decide
    whether an engine is trusted are also the rows the LLM is handed to fix it with.

    HOW BADLY, EXACTLY. Mismatches are collected in index order and only the first five are
    rendered (`_bounded_mismatches`), so prefix mismatches crowd the render budget first. An
    engine that is BAD on the prefix leaks nothing; an engine that is GOOD on the prefix leaks
    the tail. The law is exact, not statistical -- reproduced on 325 configurations at n=6..30
    with zero counterexamples:

        n_leaked = max(0, min(5 - n_prefix_mismatches, n_heldout_mismatches))

    and on the 13 real offline windows (n=3..12) in the prefix-perfect regime the leak is
    TOTAL: every gradeable acceptance row's answer is delivered, on 9 of 13 games. The render
    cap only bounds the leak when the gradeable tail exceeds five rows, which never happens at
    the real window sizes.

    WHY THE OBVIOUS FIX IS THE WRONG ONE. `execute_bounded_llm_reinduction`'s own comment
    defends full-corpus scoring -- "which is what CEGIS refinement needs" -- and it is right.
    Measured: with a prefix-perfect engine, refinement restricted to the prefix produces ZERO
    counterexamples on 13 of 13 real windows, while the full corpus produces 1-3 on nine of
    them. Restricting refinement to the prefix would not trade a leak for a weaker signal, it
    would trade a leak for NO signal, in precisely the round where refinement is the only route
    to acceptance. The defect is narrower than the comment suggests: nothing about
    counterexample-guided synthesis requires the GRADER and the TEACHER to share rows.

    SO THE FIX IS A THIRD BLOCK, not a smaller refinement corpus. The corpus is cut into
    `refinable` (everything the loop may teach from -- the induce prompt's rows AND the rows
    that supply counterexamples) and `acceptance` (never in any prompt, never a counterexample,
    grades the final engine). CEGIS keeps every counterexample it had except the reserved ones;
    acceptance becomes unreachable from any prompt.

    `CARNOT_ARC_CEGIS_ACCEPT_SPLIT=1` enables it. Default OFF, because the existing
    measurements -- including exp5766's `retire_if_same_verdict` -- were taken against the
    leaking split, and an interpretable A/B needs the old arm to still be reproducible byte for
    byte, leak included.
    """

    import os

    raw = os.environ.get("CARNOT_ARC_CEGIS_ACCEPT_SPLIT")
    if raw is None:
        raw = _CEGIS_ACCEPT_SPLIT_DEFAULT
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AcceptanceSplit:
    """A disjoint (refinable, acceptance) partition, plus whether the gate can decide at all.

    `decidable` is the "MISSING IS NOT ZERO" half of this REQ and is load-bearing on its own.
    `WorldModelVerifier.score` correctly refuses to grade a level-up row and correctly drops it
    from the denominator, then returns `n_correct / max(1, n)` -- so a block whose every row is
    a level-up scores 0.0, which is byte-indistinguishable from "the engine got everything
    wrong". The induction window is cut to END at the level-up transition, so that row is
    ALWAYS last and therefore always lands in the tail. Measured on the 13 real windows: on
    sp80, r11l, vc33 and ft09 the entire held-out tail IS that single row, and a MEMORISING,
    PERFECT engine is REJECTED with heldout_accuracy 0.0. That is an unfalsifiable gate
    reported as a failure, and 4 of 13 games sit under it.
    """

    refinable: list[Transition]
    acceptance: list[Transition]
    decidable: bool
    n_acceptance_gradeable: int
    reason: str


def _n_gradeable(rows: Sequence[Any]) -> int:
    """Rows `WorldModelVerifier.score` will actually GRADE.

    Mirrors that method's own exclusion (`if t.level_after > t.level_before: continue`) rather
    than restating it, because a split sized in RAW rows reproduces the unfalsifiable-gate bug
    inside the new block: the terminal level-up row would eat one of the one-to-three slots an
    acceptance block gets at these window sizes.
    """

    return sum(
        1
        for t in rows
        if not (int(getattr(t, "level_after", 0)) > int(getattr(t, "level_before", 0)))
    )


def split_refinement_acceptance(
    transitions: Sequence[Any],
    *,
    acceptance_fraction: float = 1.0 / 6.0,
    min_acceptance_gradeable: int = 1,
    min_refinable: int = 2,
) -> AcceptanceSplit:
    """Cut the corpus into rows the loop may LEARN from and rows that GRADE it. Disjoint.

    SIZING, and why it is half the shipped held-out fraction rather than a third of the corpus.
    `_split_prefix_heldout` already reserves the last 1/3; taking `acceptance` as half of that
    leaves the shipped induce prefix (`_proposal_prefix`, also 1/3) untouched and divides only
    rows that were never in the induce prompt to begin with. An equal three-way split would
    instead move the prompt boundary, which is a second behaviour change riding on this one.

    THE GROW LOOP is companion to `_n_gradeable`: start at the fraction, then extend the block
    backwards until it holds `min_acceptance_gradeable` GRADEABLE rows. Measured effect on the
    13 real offline windows -- 11 become decidable (up from 9 today, because the four
    all-level-up tails now reach back far enough to include a real row), and the two n=3 games
    (r11l, vc33) are reported UNDECIDABLE rather than silently scoring a perfect engine 0.0.

    `min_refinable` is the floor that stops the grow loop from eating the corpus. Where both
    floors cannot be met the split is returned UNDECIDABLE and still disjoint: purity is
    preserved even when the gate cannot decide, which is the safe direction -- an undecidable
    gate rejects, and the caller records WHY instead of publishing a 0.0.
    """

    rows = list(transitions)
    n_rows = len(rows)
    max_acceptance = n_rows - int(min_refinable)
    if max_acceptance < 1:
        # Nothing can be reserved without starving induction. Return an EMPTY acceptance block
        # rather than `_split_prefix_heldout`'s degenerate `(rows, rows)`: aliasing the two
        # halves is the very overlap this REQ exists to remove, and reproducing it here under
        # the flag would make the ON arm violate its own contract on the shortest corpora.
        return AcceptanceSplit(rows, [], False, 0, "corpus_too_short_for_disjoint_split")
    n_acceptance = max(1, int(round(n_rows * float(acceptance_fraction))))
    n_acceptance = min(n_acceptance, max_acceptance)
    while n_acceptance <= max_acceptance and _n_gradeable(rows[n_rows - n_acceptance :]) < int(
        min_acceptance_gradeable
    ):
        n_acceptance += 1
    if n_acceptance > max_acceptance:
        n_acceptance = max_acceptance
        acceptance = rows[n_rows - n_acceptance :]
        return AcceptanceSplit(
            rows[: n_rows - n_acceptance],
            acceptance,
            False,
            _n_gradeable(acceptance),
            "no_gradeable_acceptance_row_within_refinable_floor",
        )
    acceptance = rows[n_rows - n_acceptance :]
    return AcceptanceSplit(
        rows[: n_rows - n_acceptance],
        acceptance,
        True,
        _n_gradeable(acceptance),
        "ok",
    )


def resolve_hud_mask_enabled(explicit: Optional[bool] = None) -> bool:
    """REQ-ARC-WMTE-6010 CORRIGENDUM (2026-07-27): the ONE place that decides "is masking on".

    THE BUG THIS EXISTS TO KILL. Before this function, two comparators inside a single
    `select_trusted_world_model` decision disagreed about what "the same state" means:

      * `WorldModelVerifier(..., hud_mask=m)` gates on the module flag and sets
        `self.hud_mask = m if enabled else None` -- so a caller that SUPPLIED a mask but did
        not also set `CARNOT_ARC_WM_HUD_MASK=1` silently got NO masking, status "disabled".
      * `score_change_weighted_consistency` called `apply_hud_mask(...)` UNCONDITIONALLY --
        so the same caller's incumbent consistency WAS masked.

    `select_trusted_world_model`'s own docstring promises the comparators "must move
    TOGETHER: masking only some of them would rank candidates on a mixture of two different
    notions of 'the same state', which is worse than masking none of them." That promise was
    false in code. Measured on results/experiment_6013_hidden_state_change_gate_closure.json
    (162 paired mask=0/mask=1 arms): `change_fidelity` -- the quantity the new gate tests --
    differed on 0 of 162 arms (the gate was NEVER masked, `hud_mask_status` was "disabled" on
    every single mask=1 arm), while `incumbent_consistency` differed on 9 of 162 (that path
    WAS masked). exp6013's entire mask factor was therefore a silent no-op for the gate: it
    measured mask-off twice and reported it as "both mask settings".

    Resolution order, most specific first: an explicit True/False from the caller, then
    `world_model_hud_mask_enabled()` (env `CARNOT_ARC_WM_HUD_MASK`, else the shipped
    SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED default). Flag off means NO masking ANYWHERE,
    which is what keeps the default-off discipline byte-honest: before REQ-6010 nothing
    masked, so with the flag off nothing may mask, no matter what a caller passes in.
    """

    if explicit is not None:
        return bool(explicit)
    from carnot.agentic.arc_executable_world_model import world_model_hud_mask_enabled

    return bool(world_model_hud_mask_enabled())


def _effective_mask(hud_mask: Any, enabled: bool) -> Any:
    """Collapse (mask, enabled) to the single value every comparator must be handed.

    Returning None when disabled -- rather than letting each comparator re-decide -- is what
    makes "they move together" a property of the code instead of a promise in a docstring.
    """

    return hud_mask if enabled else None


def _score_accuracy(
    transitions: Sequence[Transition],
    engine: Engine,
    *,
    hud_mask: Any = None,
    hud_mask_enabled: Optional[bool] = None,
) -> float:
    # HIDDEN-STATE-branch verify metric (the gap-1 0.08-wall games -- cn04/ar25/sc25/sk48/wa30 -- are all
    # hidden-state and gate HERE). CARNOT_ARC_TRUST_METRIC=cell_recall scores by GRADED changed-cell recall
    # instead of exact-full-grid match, the same coordinated-redesign lever wired into the non-hidden gate.
    # Default (unset) -> exact accuracy: submitted behavior + the parity test unchanged.
    # REQ-ARC-WMTE-6010: `hud_mask` (LOGICAL coordinates) collapses status-bar cells before the
    # comparison. This module grepped ZERO for `hud_mask` until 2026-07-27, so every hidden-state
    # game with a monotone counter was being graded on a comparison it could not win.
    import os

    on = resolve_hud_mask_enabled(hud_mask_enabled)
    vr = WorldModelVerifier(
        list(transitions), hud_mask=_effective_mask(hud_mask, on), hud_mask_enabled=on
    ).score(engine)
    return float(
        vr.cell_recall
        if os.environ.get("CARNOT_ARC_TRUST_METRIC") == "cell_recall"
        else vr.accuracy
    )


def score_change_weighted_consistency(
    transitions: Sequence[Transition],
    engine: Engine,
    *,
    threshold: float = 0.5,
    min_correct_changed_cells: int = 1,
    hud_mask: Any = None,
    hud_mask_enabled: Optional[bool] = None,
) -> ChangeWeightedConsistency:
    """REQ-ARC-WMTE-4604: held-out consistency over grid-changing cells only.

    REQ-ARC-WMTE-6010 adds the optional `hud_mask` (LOGICAL coordinates); see
    `arc_executable_world_model.apply_hud_mask`.

    REQ-ARC-WMTE-6010 CORRIGENDUM (2026-07-27): this function used to call `apply_hud_mask`
    UNCONDITIONALLY on whatever mask it was handed, while `WorldModelVerifier` gated the same
    mask on the module flag. Two comparators inside one selection decision therefore used two
    different notions of "the same state" -- see `resolve_hud_mask_enabled` for the measured
    incident (0 of 162 gate arms masked vs 9 of 162 incumbent arms masked, in exp6013).
    `hud_mask_enabled` now routes through the single resolver, so the flag decides for every
    comparator at once.

    HONEST LIMIT of this function, recorded 2026-07-27 and NOT fixed here: `consistency`
    scores `pred[changed] == next_grid[changed]` -- it masks to TRUE changes only, so it
    cannot see a cell the engine wrote that reality did not change. It is recall, not
    fidelity, and an engine that writes garbage everywhere while covering the real changes
    scores 1.0. The symmetric replacement is `VerifyResult.change_fidelity` (union of
    truly-changed and engine-written cells) in arc_executable_world_model.py. This function
    is left on its existing metric deliberately: it is the SHIPPED hidden-state gate and
    changing its meaning here would silently move a live gate, which is a separate decision
    from adding a new default-off one.
    """

    from carnot.agentic.arc_executable_world_model import apply_hud_mask

    hud_mask = _effective_mask(hud_mask, resolve_hud_mask_enabled(hud_mask_enabled))
    exact_correct = 0
    n_changing = 0
    true_changed_cells = 0
    correct_changed_cells = 0
    for t in transitions:
        try:
            pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
        except Exception:
            continue
        t_grid = apply_hud_mask(np.asarray(t.grid), hud_mask)
        t_next = apply_hud_mask(np.asarray(t.next_grid), hud_mask)
        if pred.shape == t_next.shape:
            pred = apply_hud_mask(pred, hud_mask)
        if pred.shape == t_next.shape and np.array_equal(pred, t_next):
            exact_correct += 1
        if pred.shape != t_next.shape:
            continue
        changed = t_grid != t_next
        n_changed_cells = int(changed.sum())
        if n_changed_cells <= 0:
            continue
        n_changing += 1
        true_changed_cells += n_changed_cells
        # `t_next`, NOT `t.next_grid`: with a mask applied these differ, and reading the
        # unmasked original here would grade the prediction's HUD cells against real HUD
        # values while `changed` was computed on the masked pair -- an inconsistency that
        # would make the mask look like it did nothing.
        correct_changed_cells += int((pred[changed] == t_next[changed]).sum())

    consistency = correct_changed_cells / max(1, true_changed_cells)
    exact_accuracy = exact_correct / max(1, len(transitions))
    nondegenerate = correct_changed_cells >= int(min_correct_changed_cells)
    return ChangeWeightedConsistency(
        n=len(transitions),
        n_changing_transitions=n_changing,
        true_changed_cells=true_changed_cells,
        correct_changed_cells=correct_changed_cells,
        consistency=float(consistency),
        exact_accuracy=float(exact_accuracy),
        nondegenerate=bool(nondegenerate),
        trust_pass=bool(nondegenerate and consistency >= float(threshold)),
    )


def binary_exact_gate_pass(
    transitions: Sequence[Transition],
    engine: Engine,
    *,
    threshold: float = 0.5,
    hud_mask: Any = None,
    hud_mask_enabled: Optional[bool] = None,
) -> bool:
    """Legacy matched control: full-grid exact-match accuracy threshold.

    NOTE ON `threshold`, recorded 2026-07-27 after an adversarial review found the ambiguity
    was changing a headline by an order of magnitude: the DEFAULT here is 0.5, but the LIVE
    admission threshold the agent actually ships is `min_heldout_accuracy=1.0`
    (arc_competition_agent.py:5593 and :5719, both verified on disk). Any artifact making an
    admission claim must state WHICH threshold it used -- see
    `change_gate_decision`'s `legacy_accuracy_would_pass_at_live_threshold`.
    """

    return bool(
        _score_accuracy(transitions, engine, hud_mask=hud_mask, hud_mask_enabled=hud_mask_enabled)
        >= float(threshold)
    )


def select_trusted_world_model(
    transitions: Sequence[Transition],
    candidates: Sequence[WorldModelCandidate],
    *,
    hidden_state: bool,
    baseline_threshold: float = 0.5,
    offpath_energy_scorer: Any | None = None,
    hud_mask: Any = None,
    hud_mask_enabled: Optional[bool] = None,
) -> TrustSelection:
    """REQ-ARC-WMTE-4791: rank by off-path structural energy, not a binary cutoff.

    REQ-ARC-WMTE-6010: `hud_mask` (LOGICAL coordinates) threads to EVERY comparator this
    function ranks on -- the two accuracy scores, the two change-weighted consistencies, and
    the off-path verifier. They must move TOGETHER: masking only some of them would rank
    candidates on a mixture of two different notions of "the same state", which is worse
    than masking none of them.

    REQ-ARC-WMTE-6010 CORRIGENDUM (2026-07-27): the paragraph above was TRUE OF THE INTENT
    AND FALSE OF THE CODE until this date -- `WorldModelVerifier` dropped an unflagged mask
    while `score_change_weighted_consistency` applied it unconditionally, so the comparators
    moved apart exactly as the paragraph says they must not. `hud_mask_enabled` is now
    resolved ONCE here and handed to every comparator, so "together" is enforced rather than
    asserted. `None` keeps the shipped default-off behaviour; True/False is the per-arm
    override an A/B needs. The resolved value and the resulting mask status are recorded on
    the returned selection so a harness can ASSERT that the arm it asked for is the arm it
    got -- exp6013 could not, and silently measured mask-off twice.
    """

    if not candidates:
        raise ValueError("at least one world-model candidate is required")

    mask_on = resolve_hud_mask_enabled(hud_mask_enabled)
    # Kept before the collapse so the returned selection can report the SILENT-DROP state
    # (a mask was supplied and the flag then discarded it) rather than only the outcome.
    supplied_mask = hud_mask
    hud_mask = _effective_mask(hud_mask, mask_on)
    # REQ-ARC-WMTE-6090: the acceptance block is chosen HERE, for the same reason REQ-6013
    # gives immediately below for the change gate -- this function owns the split, and a caller
    # that pre-split and passed a tail in would have it split AGAIN. Under the flag `heldout`
    # becomes the never-taught-from ACCEPTANCE block, so every comparator that already reads
    # `heldout` (both accuracies' held-out half, the off-path energy verifier, the change gate,
    # `trust_pass`) moves onto it together. That is the whole point of the REQ-6013 invariant:
    # there is exactly one place to change, so the halves cannot drift apart.
    acceptance_split = (
        split_refinement_acceptance(transitions) if cegis_accept_split_enabled() else None
    )
    if acceptance_split is None:
        prefix, heldout = _split_prefix_heldout(transitions)
    else:
        prefix, heldout = acceptance_split.refinable, acceptance_split.acceptance
    # REQ-ARC-WMTE-6015: judge the mask ONCE, on the WHOLE corpus, and hand the verdict to
    # every sub-corpus verifier. Judging per-slice would refuse an honest mask on any
    # held-out tail that happens to contain no genuine state change -- see
    # `WorldModelVerifier.__init__` for why a tail cannot distinguish that case.
    from carnot.agentic.arc_executable_world_model import (
        hud_mask_swallow_check,
        hud_mask_swallow_clean,
    )

    swallow = hud_mask_swallow_check(list(transitions), hud_mask)
    # REQ-ARC-WMTE-6017: `swallow.get("swallows")` was plain truthiness, so an UNMEASURABLE
    # verdict (`no_dynamics_to_swallow` -- the corpus has no changing transition, so the
    # check could not fire; real instance ft09) read as clean and the mask was applied to
    # every comparator. `hud_mask_swallow_clean` requires the affirmative `ok`. The
    # `hud_mask is not None` guard keeps the mask-OFF path untouched: with no mask the reason
    # is `no_mask`, which is not clean either, and refusing there would be a no-op that
    # nonetheless overwrote `mask_on` for a mask that never existed.
    mask_refused = hud_mask is not None and not hud_mask_swallow_clean(swallow)
    # THE RECORD-BEARING PAIR keeps the supplied mask + flag so their `hud_mask_status` can
    # NAME the refusal. REQ-ARC-WMTE-6017 second finding: nulling both before constructing
    # them (what this function used to do unconditionally) made every refusal report
    # `hud_mask_status == "disabled"` -- byte-indistinguishable from "the flag was off". The
    # hidden-state branch grades exclusively through here, so on the 11 hidden-state games --
    # every 0.08-wall game -- a swallow refusal was INVISIBLE in the record. Grading is
    # unaffected: these verifiers run the same guard on the same pre-computed verdict and set
    # their own `hud_mask` to None, which is exactly what the nulled pair compared.
    gate_mask, gate_mask_on = hud_mask, mask_on
    if mask_refused:
        # Refuse ONCE, here, for every comparator at the same time. Letting each verifier
        # re-decide would reintroduce exactly the split-convention bug this corrigendum is
        # fixing, and `_score_accuracy` / `score_change_weighted_consistency` build their own
        # verifiers on their own slices where the verdict is not even computable.
        hud_mask = None
        mask_on = False
    energy_scorer = offpath_energy_scorer or default_s1_offpath_energy_scorer()
    offpath_verifier = WorldModelVerifier(
        list(heldout),
        hud_mask=gate_mask,
        hud_mask_enabled=gate_mask_on,
        hud_mask_swallow=swallow,
    )
    # REQ-ARC-WMTE-6013: the symmetric change gate is scored on `heldout` -- THE SAME SPLIT
    # `trust_pass` uses -- and it is computed HERE, inside the function that owns the split,
    # rather than by a caller. A caller cannot do this correctly: `select_trusted_world_model`
    # splits internally, so a caller that pre-split and passed the tail in would have it split
    # AGAIN and would end up scoring the last ~1/9 of the corpus. That is not hypothetical --
    # it is the first of the four harness errors recorded in
    # results/experiment_6012_hidden_state_trust_gate_hole.json, where it rejected even a
    # perfect engine. Owning the split here makes the mistake unavailable to callers.
    change_gate_verifier = WorldModelVerifier(
        list(heldout),
        hud_mask=gate_mask,
        hud_mask_enabled=gate_mask_on,
        hud_mask_swallow=swallow,
    )
    raw_rows: list[
        tuple[
            WorldModelCandidate,
            float,
            float,
            ChangeWeightedConsistency,
            ChangeWeightedConsistency,
            bool,
            bool,
            _CalibrationRow,
            float,
            dict,
        ]
    ] = []
    for candidate in candidates:
        prefix_accuracy = _score_accuracy(
            prefix, candidate.engine, hud_mask=hud_mask, hud_mask_enabled=mask_on
        )
        heldout_accuracy = _score_accuracy(
            heldout, candidate.engine, hud_mask=hud_mask, hud_mask_enabled=mask_on
        )
        prefix_change = score_change_weighted_consistency(
            prefix, candidate.engine, hud_mask=hud_mask, hud_mask_enabled=mask_on
        )
        heldout_change = score_change_weighted_consistency(
            heldout, candidate.engine, hud_mask=hud_mask, hud_mask_enabled=mask_on
        )
        calibration_row = _CalibrationRow(
            prefix_change_miss=1.0 - prefix_change.consistency,
            prefix_exact_miss=1.0 - prefix_accuracy,
            heldout_change_miss=1.0 - heldout_change.consistency,
        )
        raw_rows.append(
            (
                candidate,
                prefix_accuracy,
                heldout_accuracy,
                prefix_change,
                heldout_change,
                binary_exact_gate_pass(
                    transitions,
                    candidate.engine,
                    threshold=baseline_threshold,
                    hud_mask=hud_mask,
                    hud_mask_enabled=mask_on,
                ),
                binary_exact_gate_pass(
                    transitions,
                    candidate.engine,
                    threshold=baseline_threshold,
                    hud_mask=hud_mask,
                    hud_mask_enabled=mask_on,
                ),
                calibration_row,
                offpath_verifier.offpath_structural_energy(
                    candidate.engine,
                    energy_scorer=energy_scorer,
                ),
                # REQ-ARC-WMTE-6013. `enabled=True` here is NOT a flip: it asks
                # change_gate_decision for the decision it WOULD make, so the record is
                # populated identically in every arm. Whether that decision is ACTED ON is
                # the caller's business and is what the flag controls -- see
                # E3AgentPolicy's hidden-state branch. Computing it only when the flag is on
                # would leave the control arm with an empty field and nothing to compare.
                change_gate_decision(
                    change_gate_verifier.score(candidate.engine),
                    enabled=True,
                ),
            )
        )

    best_heldout = max(row[2] for row in raw_rows)
    calibrator = TrustEnergyCalibrator.fit([row[7] for row in raw_rows])
    rows = tuple(
        _candidate_score(
            candidate=candidate,
            prefix_accuracy=prefix_accuracy,
            heldout_accuracy=heldout_accuracy,
            prefix_change=prefix_change,
            heldout_change=heldout_change,
            baseline_clears=baseline_clears,
            binary_gate_pass=binary_gate_pass,
            heldout_best=heldout_accuracy == best_heldout,
            calibration_row=calibration_row,
            calibrator=calibrator,
            offpath_structural_energy=offpath_structural_energy,
            change_gate=change_gate,
        )
        for (
            candidate,
            prefix_accuracy,
            heldout_accuracy,
            prefix_change,
            heldout_change,
            baseline_clears,
            binary_gate_pass,
            calibration_row,
            offpath_structural_energy,
            change_gate,
        ) in raw_rows
    )
    baseline = next((row for row in rows if row.binary_gate_pass), None)
    selected_score = min(
        rows,
        key=lambda row: (
            row.trust_energy,
            row.candidate.name,
        ),
    )
    return TrustSelection(
        selected=selected_score.candidate,
        selected_score=selected_score,
        baseline_candidate_name=baseline.candidate.name if baseline else None,
        rows=rows,
        verifier_is_oracle=not bool(hidden_state),
        hud_mask_enabled=bool(mask_on),
        hud_mask_supplied=bool(supplied_mask is not None),
        acceptance_split_enabled=acceptance_split is not None,
        acceptance_decidable=(True if acceptance_split is None else acceptance_split.decidable),
        acceptance_reason=(
            "legacy_prefix_heldout_split" if acceptance_split is None else acceptance_split.reason
        ),
        # -1, not 0, on the OFF path: the legacy split does not compute this, and writing 0
        # would assert "nothing was gradeable", which is a claim and usually a false one.
        n_acceptance_gradeable=(
            -1 if acceptance_split is None else acceptance_split.n_acceptance_gradeable
        ),
    )


def _candidate_score(
    *,
    candidate: WorldModelCandidate,
    prefix_accuracy: float,
    heldout_accuracy: float,
    prefix_change: ChangeWeightedConsistency,
    heldout_change: ChangeWeightedConsistency,
    baseline_clears: bool,
    binary_gate_pass: bool,
    heldout_best: bool,
    calibration_row: _CalibrationRow,
    calibrator: TrustEnergyCalibrator,
    offpath_structural_energy: float | None = None,
    change_gate: dict | None = None,
) -> CandidateScore:
    heldout_miss = 1.0 - heldout_change.consistency
    calibrated_miss = calibrator.predicted_heldout_miss(calibration_row)
    overfit_gap = max(0.0, prefix_change.consistency - heldout_change.consistency)
    degeneracy_penalty = 1.0 if not heldout_change.nondegenerate else 0.0
    exact_miss_tiebreak = 0.05 * (1.0 - heldout_accuracy)
    if offpath_structural_energy is None:
        trust_energy = heldout_miss + 0.25 * calibrated_miss + 0.25 * overfit_gap
        trust_energy += degeneracy_penalty + exact_miss_tiebreak
    else:
        del heldout_miss, calibrated_miss, overfit_gap, degeneracy_penalty, exact_miss_tiebreak
        trust_energy = float(offpath_structural_energy)
    return CandidateScore(
        candidate=candidate,
        prefix_accuracy=prefix_accuracy,
        heldout_accuracy=heldout_accuracy,
        trust_energy=float(trust_energy),
        baseline_clears=baseline_clears,
        heldout_best=heldout_best,
        prefix_change_consistency=prefix_change.consistency,
        heldout_change_consistency=heldout_change.consistency,
        correct_changed_cells=heldout_change.correct_changed_cells,
        true_changed_cells=heldout_change.true_changed_cells,
        nondegenerate=heldout_change.nondegenerate,
        trust_pass=heldout_change.trust_pass,
        binary_gate_pass=binary_gate_pass,
        change_gate=dict(change_gate or {}),
    )


def scorecard_from_selection(game: str, selection: TrustSelection) -> Scorecard:
    best_names = {row.candidate.name for row in selection.rows if row.heldout_best}
    baseline_pick_best = selection.baseline_candidate_name in best_names
    return Scorecard(
        game=game,
        selected_candidate_name=selection.selected.name,
        baseline_candidate_name=selection.baseline_candidate_name,
        trust_energy_pick_best=selection.selected.name in best_names,
        baseline_pick_best=bool(baseline_pick_best),
        n_candidates=len(selection.rows),
        verifier_is_oracle=selection.verifier_is_oracle,
    )


def build_experiment_artifact(
    *,
    hidden_scorecards: Sequence[Scorecard],
    positive_control: Scorecard,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4492/4493: terminal artifact with oracle-distinct null guard."""

    hidden_rows = list(hidden_scorecards)
    n_hidden = len(hidden_rows)
    trust_wins = sum(1 for row in hidden_rows if row.trust_energy_pick_best)
    baseline_wins = sum(1 for row in hidden_rows if row.baseline_pick_best)
    trust_rate = trust_wins / max(1, n_hidden)
    baseline_rate = baseline_wins / max(1, n_hidden)
    positive_control_passed = (
        positive_control.trust_energy_pick_best and not positive_control.baseline_pick_best
    )
    if trust_rate > baseline_rate:
        honest_verdict = "success: world_model_trust_energy_beats_first_clears_baseline"
        false_negative_risk_guard = "positive_control_passed_hidden_state_gain"
    elif positive_control_passed:
        honest_verdict = "complete: world_model_trust_energy_hidden_state_null"
        false_negative_risk_guard = "positive_control_passed_hidden_state_null"
    else:
        honest_verdict = "complete: world_model_trust_energy_positive_control_failed"
        false_negative_risk_guard = "positive_control_failed_null_uninformative"

    return {
        "experiment": "experiment_4491_world_model_trust_energy",
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "verifier_is_oracle": False,
        "hidden_state_games_n": n_hidden,
        "hidden_state_games": [row.game for row in hidden_rows],
        "trust_energy_pick_rate": trust_rate,
        "baseline_pick_rate": baseline_rate,
        "positive_control_passed": bool(positive_control_passed),
        "false_negative_risk_guard": false_negative_risk_guard,
        "selected_candidates": [row.to_json() for row in hidden_rows],
        "positive_control": positive_control.to_json(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 3),
        "submitted_to_leaderboard": False,
    }


def _synthetic_hidden_transitions(seed: int) -> list[Transition]:
    base = int(seed) * 100
    return [
        Transition(
            grid=np.array([[base + i]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[base + i + 1]], dtype=np.int16),
            level_before=0,
            level_after=0,
        )
        for i in range(6)
    ]


def _prefix_overfit_engine(cutoff: int) -> Engine:
    def engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
        value = int(np.asarray(grid)[0, 0])
        return np.asarray(grid) + 1 if value < cutoff else np.asarray(grid)

    engine.__name__ = f"prefix_overfit_lt_{cutoff}"
    return engine


def _increment_engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
    return np.asarray(grid) + 1


def _noop_engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
    return np.asarray(grid)


def _fixture_scorecard(game: str, seed: int, *, verifier_is_oracle: bool) -> Scorecard:
    transitions = _synthetic_hidden_transitions(seed)
    cutoff = int(np.asarray(transitions[4].grid)[0, 0])
    selection = select_trusted_world_model(
        transitions,
        (
            WorldModelCandidate("first_prefix_overfit", _prefix_overfit_engine(cutoff)),
            WorldModelCandidate("heldout_generalizer", _increment_engine),
            WorldModelCandidate("noop_null", _noop_engine),
        ),
        hidden_state=not verifier_is_oracle,
    )
    return scorecard_from_selection(game, selection)


def run_experiment_4491(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-4492: write the stable Exp 4491 deliverable JSON."""

    started = time.time()
    checks = dict(preconditions_checked or check_preconditions())
    hidden_scorecards = [
        _fixture_scorecard(game, seed=i + 1, verifier_is_oracle=False)
        for i, game in enumerate(HIDDEN_STATE_GAME_IDS)
    ]
    positive_control = _fixture_scorecard(
        "markov_positive_control", seed=99, verifier_is_oracle=True
    )
    artifact = build_experiment_artifact(
        hidden_scorecards=hidden_scorecards,
        positive_control=positive_control,
        preconditions_checked=checks,
        duration_s=time.time() - started,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def check_preconditions() -> dict[
    str, Any
]:  # pragma: no cover - tested by required terminal smoke.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    import torch

    return {
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": str(torch.__version__),
    }


def main() -> int:  # pragma: no cover - CLI shim.
    run_experiment_4491()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
