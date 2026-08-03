"""REQ-ARC-WMTE-6090: make an incommensurable metric subtraction unrepresentable.

THE INCIDENT (2026-08-03). `results/experiment_5766_gemma31b_cegis_refinement_ab.json`
reported `pooled_delta_cegis_best_minus_singleshot: -0.318658` with a bootstrap CI that
`excludes_0`, under the heading "does CEGIS refinement improve on gemma's OWN 0.378
single-shot baseline". Read as written, that number says the refinement loop is 6.33x worse
than the single-shot control it was built to extend. It was the headline of the run that
carries `retire_if_same_verdict`.

The subtraction is invalid. Both operands are called `heldout_accuracy`, and they are
computed by two different functions over two different row sets, standing in two different
relations to the evidence the model was actually shown:

  * exp5764 (via `exp5726.run_reason_cell_budget`, reused VERBATIM) prompts on the WHOLE
    progress window -- `_induce_no_fence(prop, game, list(window), cell)` -- and then grades
    on the WHOLE progress window -- `WorldModelVerifier(list(window)).score(engine).accuracy`.
    Prompt set and grading set are THE SAME ROWS. That is an in-sample fit score. It is not a
    held-out quantity at all, whatever the field is named.

  * exp5766 round 0 reaches induction through `execute_bounded_llm_reinduction`, which -- when
    the caller passes no `proposal_transitions`, and `run_cegis_cell` never does -- falls back
    to `_proposal_prefix(transitions)`, the FIRST two thirds. It is then graded by
    `select_trusted_world_model` -> `_score_accuracy(heldout, ...)` on the LAST third from
    `_split_prefix_heldout`. Prompt set and grading set are DISJOINT BY CONSTRUCTION. That is
    an out-of-sample generalisation score.

Subtracting an out-of-sample score from an in-sample score and reporting the difference as a
treatment effect measures the split, not the treatment. Reproduced exactly, 2026-08-03, by
re-deriving every per-game cell and the pooled value from the two committed shards:

    reported     (whole_window vs disjoint_tail) 0.378487 vs 0.059829  ratio 6.3261  p = 0.0386
    like-for-like(whole_window vs prompt-overlapping prefix) 0.378487 vs 0.319444  p = 1.0

Both quantities were on disk the whole time. `prefix_accuracy` sits beside `heldout_accuracy`
in EVERY exp5766 round record, computed by the same function on the same engine, and
`_proposal_prefix` and `_split_prefix_heldout` cut at the same index -- so `prefix_accuracy`
grades precisely the pool the prompt was drawn from. Nobody read it. Nothing in the code could
have told them to: a float is a float, and the scope lived only in the reader's head.

WHY A GUARD AT THE SUBTRACTION, AND NOT A LINT OVER ARTIFACTS. An artifact lint only ever sees
what reached disk, and this comparison is computed IN PROCESS at artifact-build time -- the
number is already fabricated by the time a linter could look at it, and the linter would have
to re-derive the scopes it cannot see. That is the same structural blind spot
`operator_curated_docs_lint.py` has, and it is why this guard sits at the arithmetic instead.

WHAT THIS MODULE IS NOT. It is not a claim that the two arms are equal. It is a refusal to
produce a number that cannot mean what its label says. The residual like-for-like gap is
1.18-1.29x, which is smaller than this harness's own documented cell-level nondeterminism
(`LocalGGUFProposer.sampling_seed` returns None absent `CARNOT_ARC_GENERATOR_SEED`, so
llama-server draws a fresh sampler seed per call at temperature 0.2), and 13 games cannot
resolve an effect that size either way.

DEFAULT OFF. `compare_scoped` with the guard off returns exactly `treatment - baseline` --
the shipped arithmetic, bug included -- so every measurement taken before this module existed
stays interpretable and re-derivable. `CARNOT_ARC_METRIC_SCOPE_GUARD=1` turns the refusal on.

Cross-references:
  * results/experiment_5766_round0_regression_reproduction.json -- the reproduction
  * results/experiment_5766_gemma31b_cegis_refinement_ab.json -- the voided comparison block
  * python/carnot/agentic/arc_world_model_trust_energy.py:_split_prefix_heldout -- the tail
  * python/carnot/agentic/arc_llm_reinduction.py:_proposal_prefix -- the matching prefix
  * CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" -- the sibling rule that
    catches fabricated numbers; this one catches numbers that are real but incommensurable.
"""

from __future__ import annotations

import os
from typing import Final

__all__ = [
    "DISJOINT_TAIL",
    "HELDOUT_METRIC_SCOPES",
    "IN_SAMPLE",
    "KNOWN_SCOPES",
    "METRIC_SCOPE_GUARD_ENV",
    "OUT_OF_SAMPLE",
    "PROMPT_OVERLAPPING_PREFIX",
    "SCOPE_RELATION",
    "WHOLE_WINDOW",
    "MetricScopeMismatch",
    "compare_scoped",
    "metric_scope_guard_enabled",
    "relation_of",
    "scope_for",
]


# --------------------------------------------------------------------------------------
# The vocabulary. Each name answers ONE question: what is this score's grading set, relative
# to the rows the model was shown? That RELATION -- not the row count, not the field name --
# is what decides whether two scores may be subtracted.
# --------------------------------------------------------------------------------------

WHOLE_WINDOW: Final = "whole_window"
"""Graded on the ENTIRE window, which is also the entire prompt evidence. The model is being
asked to reproduce rows it was shown, with the answers. Produced by
`WorldModelVerifier(list(window)).score(engine).accuracy` in `exp5726.run_reason_cell_budget`,
which exp5764 reuses verbatim."""

DISJOINT_TAIL: Final = "disjoint_tail"
"""Graded on the last third, which was deliberately withheld from the prompt. Produced by
`select_trusted_world_model` via `_split_prefix_heldout` + `_score_accuracy`, and recorded as
`heldout_accuracy` on every exp5766 round record."""

PROMPT_OVERLAPPING_PREFIX: Final = "prompt_overlapping_prefix"
"""Graded on the first two thirds -- exactly the rows `_proposal_prefix` put in the prompt.
Recorded as `prefix_accuracy` beside `heldout_accuracy` on every exp5766 round record."""

KNOWN_SCOPES: Final[frozenset[str]] = frozenset(
    {WHOLE_WINDOW, DISJOINT_TAIL, PROMPT_OVERLAPPING_PREFIX}
)


# --------------------------------------------------------------------------------------
# THE RELATION IS THE COMPARABILITY KEY, NOT THE ROW SET. This distinction is the whole
# design, and getting it wrong in either direction breaks something real:
#
#   * keying on the row-set NAME would refuse `whole_window` vs `prompt_overlapping_prefix` --
#     which is the CORRECT like-for-like comparison (0.378487 vs 0.319444, p = 1.0) that the
#     2026-08-03 reproduction identified. A guard that blocks the right answer is worse than
#     no guard, because it teaches people to switch it off.
#   * keying on nothing at all is the shipped bug.
#
# Two scores are commensurable when they stand in the SAME relation to their prompt evidence.
# Both `whole_window` and `prompt_overlapping_prefix` grade exactly the rows their model was
# shown, so both are fit scores and their difference is a treatment effect. `disjoint_tail`
# grades rows the model never saw, so subtracting it from either measures the split.
#
# HONEST LIMIT, deliberately NOT enforced here: equal relation does not imply equal evidence
# QUANTITY. exp5764 fits on 3/3 of the window and exp5766 round 0 on 2/3, so even the
# like-for-like comparison carries a one-third evidence handicap whose sign matches the
# residual 1.18-1.29x. Refusing that too would leave no admissible comparison at all; it is a
# caveat to state in the artifact, not a subtraction to block.
# --------------------------------------------------------------------------------------

IN_SAMPLE: Final = "in_sample"
OUT_OF_SAMPLE: Final = "out_of_sample"

SCOPE_RELATION: Final[dict[str, str]] = {
    WHOLE_WINDOW: IN_SAMPLE,
    PROMPT_OVERLAPPING_PREFIX: IN_SAMPLE,
    DISJOINT_TAIL: OUT_OF_SAMPLE,
}


def relation_of(scope: str) -> str:
    """The prompt-relation a scope stands in: `IN_SAMPLE` or `OUT_OF_SAMPLE`."""

    try:
        return SCOPE_RELATION[scope]
    except KeyError:
        raise ValueError(
            f"unknown metric scope {scope!r}; known scopes are {sorted(KNOWN_SCOPES)}"
        ) from None


# --------------------------------------------------------------------------------------
# The registry. A future comparator should not have to re-derive, from two experiment scripts
# and a call chain three modules deep, what a recorded field's grading set was -- that
# re-derivation is precisely the step that did not happen in exp5766. Keys are
# (producer, recorded field path).
# --------------------------------------------------------------------------------------

HELDOUT_METRIC_SCOPES: Final[dict[tuple[str, str], str]] = {
    # exp5726 is the origin of the whole-window pattern; exp5764 reuses its cell VERBATIM, so
    # both shards carry a fit score under the name `heldout_accuracy`.
    ("exp5726", "heldout_accuracy"): WHOLE_WINDOW,
    ("exp5764", "heldout_accuracy"): WHOLE_WINDOW,
    # exp5766 records BOTH relations on every round, from the same engine. The tail is the one
    # that was read; the prefix is the one that was comparable to exp5764 and was not.
    ("exp5766", "rounds[].heldout_accuracy"): DISJOINT_TAIL,
    ("exp5766", "rounds[].prefix_accuracy"): PROMPT_OVERLAPPING_PREFIX,
    ("exp5766", "round0_heldout"): DISJOINT_TAIL,
    # The live producer, named so a future caller can tag a fresh measurement without going
    # back through the experiment scripts.
    ("select_trusted_world_model", "heldout_accuracy"): DISJOINT_TAIL,
    ("select_trusted_world_model", "prefix_accuracy"): PROMPT_OVERLAPPING_PREFIX,
}


def scope_for(producer: str, field: str) -> str:
    """Look up the grading scope of a recorded metric field.

    Raises rather than returning a default. A silent fallback would reintroduce the exact
    failure this module exists to stop: an unknown provenance rendered as a confident tag, and
    then subtracted from something.
    """

    try:
        return HELDOUT_METRIC_SCOPES[(producer, field)]
    except KeyError:
        raise KeyError(
            f"no recorded metric scope for producer={producer!r} field={field!r}; "
            f"add it to HELDOUT_METRIC_SCOPES after reading the code that produces it "
            f"(what rows does it grade, and were those rows in the prompt?)"
        ) from None


# --------------------------------------------------------------------------------------
# The flag. Default OFF, fail-closed on anything unrecognised, following
# `_supply_win_transition_enabled` -- a typo'd export must never silently arm a refusal in a
# harness that is mid-run.
# --------------------------------------------------------------------------------------

METRIC_SCOPE_GUARD_ENV: Final = "CARNOT_ARC_METRIC_SCOPE_GUARD"
_METRIC_SCOPE_GUARD_DEFAULT: Final = "0"


def metric_scope_guard_enabled() -> bool:
    """True when the cross-scope refusal is armed. Shipped default is False."""

    raw = os.environ.get(METRIC_SCOPE_GUARD_ENV)
    if raw is None:
        raw = _METRIC_SCOPE_GUARD_DEFAULT
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class MetricScopeMismatch(ValueError):
    """Raised when two metrics graded on differently-related row sets would be subtracted.

    Carries both scopes so the message names the actual defect rather than "invalid input" --
    the operator reading a stack trace needs to know WHICH two relations were mixed.
    """

    def __init__(self, treatment_scope: str, baseline_scope: str, label: str = "") -> None:
        self.treatment_scope = treatment_scope
        self.baseline_scope = baseline_scope
        self.treatment_relation = relation_of(treatment_scope)
        self.baseline_relation = relation_of(baseline_scope)
        self.label = label
        where = f" ({label})" if label else ""
        super().__init__(
            f"refusing to subtract metrics graded in different relations to their prompt"
            f"{where}: treatment {treatment_scope!r} is {self.treatment_relation}, "
            f"baseline {baseline_scope!r} is {self.baseline_relation}. Their difference "
            f"measures the split rather than the treatment (REQ-ARC-WMTE-6090; see exp5766's "
            f"voided comparison_to_gemma31b_singleshot_baseline, which read -0.318658 with a "
            f"CI excluding 0 for exactly this reason). Compare like with like -- for a "
            f"{WHOLE_WINDOW!r} baseline the commensurable counterpart is "
            f"{PROMPT_OVERLAPPING_PREFIX!r}, not {DISJOINT_TAIL!r}."
        )


def compare_scoped(
    treatment: float,
    treatment_scope: str,
    baseline: float,
    baseline_scope: str,
    *,
    label: str = "",
    enabled: bool | None = None,
) -> float:
    """Return `treatment - baseline`, refusing when the two grade in different prompt relations.

    With the guard OFF (the shipped default) this is a strict passthrough: it returns exactly
    the float the bare subtraction returned, for every pair of known scopes including
    incommensurable ones. That is deliberate and load-bearing -- every number measured before
    this module existed must stay re-derivable through it, or the historical record becomes
    uninterpretable in a second, different way.

    With the guard ON it refuses only a RELATION mismatch. `whole_window` minus
    `prompt_overlapping_prefix` is permitted, because both grade the rows their model was
    shown; that is the comparison the 2026-08-03 reproduction found to be the correct one. Note
    that permitted is not the same as unconfounded -- see the module's honest limit on evidence
    quantity.

    `enabled` overrides the env for a single call, so an A/B harness can assert that the arm it
    asked for is the arm it got instead of trusting process-wide state.

    Unknown scope NAMES are refused in both directions. That is not a behaviour change to any
    historical path -- no pre-existing call site passes a scope at all -- and it stops the
    guard being defeated by a typo, which would make its pattern list narrower than the concept
    it protects.
    """

    treatment_relation = relation_of(treatment_scope)
    baseline_relation = relation_of(baseline_scope)
    on = metric_scope_guard_enabled() if enabled is None else bool(enabled)
    if on and treatment_relation != baseline_relation:
        raise MetricScopeMismatch(treatment_scope, baseline_scope, label)
    return float(treatment) - float(baseline)
