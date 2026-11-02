"""Recall-gated induction resample: one bounded second draw when induction collapses.

WHY THIS EXISTS. Live world-model induction is single-shot. Measured over 13 scored
cells (5 games x 3 seeds), roughly 1 in 4 draws collapses on fitted-window recall
(0.0 / 0.294 / 0.354 observed) while the SAME game passes at every other seed. The
pipeline's own A/A note (`arc_executable_world_model.LocalGGUFProposer.sampling_seed`)
documents a ~40% cell-divergence floor under identical code. A collapsed draw is
therefore usually a SAMPLE failure, not a WINDOW failure, and one fresh draw
recovers it with high probability.

THE TRAP THIS DESIGN IS BUILT AROUND. Recall here is IN-SAMPLE: the verifier scores
the engine against the same transition window the model saw in the prompt. An
engine that hardcodes the window's coordinates scores 1.0 by construction, so a
gate that prefers HIGHER in-sample recall among usable engines selects for
memorization -- it would improve this number and worsen hidden-game play. The
measured discriminator between high and low cells is exactly that (`mem_scan`,
`results/arc_qwen38_h2h_partial_20260817`: on tr87 the two ~1.0 draws hardcode 33
and 38 window coordinates; the 0.354 draw does not).

THE DESIGN, and why it dodges the trap rather than merely shrinking it:

1. TRUST-SUBORDINATE. The gate may fire ONLY on an engine the downstream trust
   gate already rejects -- an engine the agent would refuse to plan with and
   would replace with nothing. It never discards or ranks an engine the pipeline
   would use, so it applies zero selection pressure among usable engines. The
   memorization bias of the SECOND draw is the same bias the first draw already
   has; acceptance criteria are unchanged.
2. CATASTROPHIC-ONLY. It fires on recall far below ceiling (default < 0.6), never
   on near-ceiling differences (a 0.97 generalizer vs a 1.0 memorizer), which is
   the exact region where in-sample selection is toxic. An engine wrong on 40%+
   of the changed cells it was SHOWN is bad by any measure -- in-sample recall is
   a necessary condition, and only its gross violation is acted on.
3. EVIDENCE-FLOORED. `cell_recall` is 0.0 by definition on a window with no
   state-changing transitions, and is noise on 1-2. Below `min_changing` (default
   3) the window, not the draw, is the problem, and a fresh draw from the same
   window would fail the same way -- so the gate refuses to spend a generation.
4. HARD-BOUNDED COST. One retry per induce call (constant, not tunable), a small
   per-game budget (default 2), and default OFF: a full generation costs tens of
   minutes and induction already dominates the live eval budget, so the lever
   ships dark until an A/B measures the trade.
5. SEED-PINNED REFUSAL. With CARNOT_ARC_GENERATOR_SEED set, `sampling_seed(0)` is
   the same for every induce call, so a re-call replays the identical tokens. The
   gate refuses rather than paying a generation for a byte-identical answer.

This is DISJOINT from the defect re-ask in `LocalGGUFProposer.generate` (static
defects and dry-run raises, caught INSIDE the generation call before acceptance).
This gate runs AFTER acceptance, on behavioral evidence the re-ask cannot see: an
engine that runs cleanly and predicts wrong. See REQ-ARC-WMTE-6410.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Hard per-call ceiling, deliberately NOT env-tunable: a second retry would only
# fire after two consecutive collapses (rare) and doubles worst-case added cost.
MAX_RETRIES_PER_CALL = 1

DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_CHANGING = 3
DEFAULT_MAX_PER_GAME = 2

_ENABLE_ENV = "CARNOT_ARC_RECALL_RESAMPLE"
_THRESHOLD_ENV = "CARNOT_ARC_RECALL_RESAMPLE_THRESHOLD"
_MIN_CHANGING_ENV = "CARNOT_ARC_RECALL_RESAMPLE_MIN_CHANGING"
_MAX_PER_GAME_ENV = "CARNOT_ARC_RECALL_RESAMPLE_MAX_PER_GAME"
_GENERATOR_SEED_ENV = "CARNOT_ARC_GENERATOR_SEED"
_TOOL_LOOP_ENV = "CARNOT_ARC_INDUCE_TOOL_LOOP"


def resample_enabled() -> bool:
    """Default OFF: the scored path is byte-identical until an operator opts in."""
    return str(os.environ.get(_ENABLE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def tool_repair_active() -> bool:
    """REQ-ARC-WMTE-6470: `CARNOT_ARC_INDUCE_TOOL_LOOP=repair` activates this gate with
    the seeded tool loop as the re-draw. Read here (not just at the call site) so the
    decision function can honor repair enablement without a second env contract."""
    return str(os.environ.get(_TOOL_LOOP_ENV, "")).strip().lower() == "repair"


def resample_threshold() -> float:
    """Catastrophic-recall ceiling. A malformed env value falls back to the default
    rather than raising -- a typo must not take down a live episode."""
    raw = os.environ.get(_THRESHOLD_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_THRESHOLD
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD


def resample_min_changing() -> int:
    raw = os.environ.get(_MIN_CHANGING_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_MIN_CHANGING
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MIN_CHANGING


def resample_max_per_game() -> int:
    raw = os.environ.get(_MAX_PER_GAME_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_PER_GAME
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_PER_GAME


def generator_seed_pinned() -> bool:
    """Mirror of `LocalGGUFProposer.sampling_seed`'s parse: a set, int-parseable value
    pins the sampler, so a plain re-call would reproduce the first draw exactly."""
    raw = os.environ.get(_GENERATOR_SEED_ENV)
    if raw is None or raw.strip() == "":
        return False
    try:
        int(raw)
    except (TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True)
class ResampleDecision:
    fire: bool
    reason: str


def decide_resample(
    *,
    cell_recall: float,
    n_changing: int,
    downstream_rejects: bool,
    resamples_used_this_game: int,
    retries_used_this_call: int = 0,
    tool_repair: bool = False,
) -> ResampleDecision:
    """Should this induce call spend one more generation? Pure; the caller supplies
    the downstream trust verdict so this module never re-implements the gate.

    Check order is the audit order: each early refusal names the cheapest reason
    that makes the later, more expensive questions moot.

    `tool_repair` (REQ-ARC-WMTE-6470, default False = behaviour unchanged): the
    re-draw will be the SEEDED TOOL LOOP, not a blind re-generation. Two rules move:
    enablement is satisfied by repair mode alone, and the seed-pinned refusal is
    skipped -- that refusal exists because a pinned blind re-draw replays the same
    tokens for a full generation's cost, while the loop's multi-turn conversation is
    a different token stream by construction. Every safety rule (trust-subordinate,
    budgets, evidence floor, catastrophic-only) applies identically in both modes.
    """
    if not (resample_enabled() or tool_repair):
        return ResampleDecision(False, "disabled")
    if not downstream_rejects:
        # The structural safety property: an engine the pipeline would use is
        # never re-rolled, so no selection happens among usable engines.
        return ResampleDecision(False, "downstream_accepted_engine")
    if generator_seed_pinned() and not tool_repair:
        return ResampleDecision(False, "seed_pinned_resample_would_reproduce")
    if retries_used_this_call >= MAX_RETRIES_PER_CALL:
        return ResampleDecision(False, "per_call_retry_exhausted")
    if resamples_used_this_game >= resample_max_per_game():
        return ResampleDecision(False, "per_game_budget_exhausted")
    if int(n_changing) < resample_min_changing():
        return ResampleDecision(False, "insufficient_changing_evidence")
    threshold = resample_threshold()
    if float(cell_recall) >= threshold:
        return ResampleDecision(False, "recall_not_catastrophic")
    # HONEST REASON LABEL. A threshold above 1.0 makes the recall test vacuous (recall
    # cannot exceed 1.0), turning the effective fire condition into "trust rejected the
    # engine". Recording `catastrophic_recall` on a recall-1.0/accuracy-0.0 cell would
    # write the opposite of the truth into the audit field, so the reason names the
    # condition that actually fired.
    if threshold > 1.0:
        if tool_repair:
            return ResampleDecision(True, "trust_rejected_tool_loop_repair")
        return ResampleDecision(True, "trust_rejected_resample")
    if tool_repair:
        return ResampleDecision(True, "catastrophic_recall_tool_loop_repair")
    return ResampleDecision(True, "catastrophic_recall_resample")


def keep_resample(
    *,
    new_passes_downstream: bool,
    new_cell_recall: float,
    old_cell_recall: float,
) -> bool:
    """Keep the second draw iff it is usable, else keep the better of two rejects.

    The recall tie-break only ever compares two engines the trust gate rejects
    (fire requires the old one rejected; this branch requires the new one too),
    so it cannot prefer a 1.0 memorizer over a usable generalizer -- a 1.0
    in-sample engine is not in the rejected set.
    """
    if new_passes_downstream:
        return True
    return float(new_cell_recall) > float(old_cell_recall)
