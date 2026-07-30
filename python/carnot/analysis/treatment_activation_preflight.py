"""TREATMENT-ACTIVATION PRE-FLIGHT: refuse an A/B grid the treatment cannot move.

WHY THIS EXISTS (the incident that paid for it, 2026-07-29)
-----------------------------------------------------------
Three A/B measurements on the live ARC agent were run in one day. All three returned a
null. Hours of wall-clock went into each. Then the post-mortem diffed the two arms' ACTION
TRACES cell by cell and found the nulls were not findings at all:

    6 of 12 cells   byte-identical action traces, both arms complete -- the SAME AGENT
    3 of 12 cells   one arm truncated by a wall-clock cap, the other ran to budget
    2 of 12 cells   BOTH arms truncated, agreeing on every action either was allowed to take
    1 of 12 cells   genuinely different behaviour

(The first published version of this partition read "8 identical, 3 truncated, 1 perturbed".
That was wrong in this module's own favour and is corrected rather than reworded: the
classifier tested `a == b` BEFORE consulting whether either arm finished, so cn04 and r11l --
where NEITHER arm got past action 27 of a 400-action budget -- were counted as byte-identical
measurements. They are missing observations. See `BOTH_TRUNCATED`.)

A byte-identical pair cannot possibly differ on ANY downstream endpoint. It is not a tie,
it is a non-observation: comparing an agent against itself. With one genuinely-perturbed
cell out of twelve, the largest number of discordant pairs the grid could ever have
produced was one, and the smallest two-sided sign-test p-value it could ever have reported
was 1.0. **No outcome of that experiment could have reached p<0.05.** The null was
guaranteed before the first cell started, and it was knowable in minutes.

That is what this module computes. Given two arms and a SHORT probe per cell, it diffs the
action traces, classifies each pair, and REFUSES the full grid unless the treatment
actually perturbs enough cells for the planned test to be able to detect anything.

A REFUSAL IS A SUCCESSFUL OUTCOME. It converts "we measured it and found nothing" (which
reads as evidence AGAINST the treatment, and which future planners will cite as a dead
direction) into the honest and much more useful "the treatment never activated, so we
learned nothing about it yet."

HOW CHEAP THE PROBE ACTUALLY IS -- measured, and it is NOT "minutes"
-------------------------------------------------------------------
An earlier draft of this docstring claimed the pre-flight "costs minutes and saves hours".
On the ARC live path that is FALSE and was corrected rather than softened: one induction on
`gemma-4-31B-it` takes 200-1200 s wall-clock, so a 24-cell probe costs roughly what the
24-cell grid it screens costs. The saving is real but it is a saving of the LARGER grid the
probe would have licensed, not an order-of-magnitude screen.

Two things do make it cheap, and they are where the leverage actually is:

  1. **Retrospective use is FREE.** Running it over cells already on disk costs nothing and
     is how the 2026-07-29 nulls were diagnosed. Do this whenever the cells come from THE SAME
     GRID.
     **But do NOT assemble a noise floor out of a DIFFERENT experiment's cells.** An earlier
     version of this note recommended exactly that -- pair cells that "share treatment, seed,
     model and game" -- and doing it produced a retracted finding: the two sets were >=6 agentic
     commits apart, so the supposed A/A was an A/B of the treatment under test. Cheap and wrong
     is worse than expensive and right. If you must reach across experiments, first diff the
     fields the two cells RECORD (`arm_commit`, the treatment's own diagnostic fields) rather
     than reasoning about how they were configured; a control whose validity rests on an
     argument is not a control.
  2. **The verdict often settles early.** Only a PERTURBED cell can ever be discordant, so
     the answer is decided the moment `n_perturbed + n_pending < required` (REFUSE certain)
     or `n_perturbed >= required` (PASS certain). Every cell after that point is pure cost.
     Check it as cells land instead of running the grid to completion out of habit. Note the
     asymmetry: an early PASS is final, an early REFUSE is only final once no cells remain --
     which is what the `INCONCLUSIVE` verdict exists to keep honest.

A genuinely minutes-scale pre-flight on this path needs a cheaper induction -- a smaller
SOTA model, or a replay of cached inductions -- which is not built.

WHAT IT IS NOT
--------------
Passing this pre-flight does NOT mean the experiment is powered. Perturbation is
NECESSARY but not SUFFICIENT: a treatment can change every action the agent takes and
still land on the same banked-level count in every cell. The pre-flight can only rule out
the specific failure mode where the arms are behaviourally identical, i.e. where the
experiment is DEAD ON ARRIVAL. Read a PASS as "not obviously dead", never as "powered".

WHY ACTION TRACES, AND NOT THE ENDPOINT ITSELF
----------------------------------------------
The endpoint (banked levels) is coarse, expensive, and mostly zero -- exactly the
properties that make an underpowered grid look like a clean null. The action trace is the
agent's complete observable behaviour, it is dense, it is recorded by the existing harness
at no extra cost, and it is UPSTREAM of every endpoint: if two arms emit the same actions,
every metric computed downstream of those actions is identical by construction. So the
trace diff is the cheapest possible test of "did the treatment reach the agent at all".

THE THREE CLASSES, AND WHY THE MIDDLE ONE MATTERS
-------------------------------------------------
`IDENTICAL`         Byte-identical traces. The treatment did not change behaviour. This
                    pair can never be discordant on any endpoint. It contributes ZERO
                    information, and counting it as a "tie" understates how little the
                    grid knows.

`TRUNCATION_ONLY`   The two arms agreed on every action they both took, and the arm that
                    stopped first did not finish (wall-clock cap, hard timeout, crash) --
                    whether its trace is a strict prefix of the other's or exactly as long.
                    This is a MISSING OBSERVATION, not a zero and not a perturbation. One
                    arm simply ran out of time. Scoring it as "0 banked levels for the slow
                    arm" is the selection effect that silently hands a comparison to
                    whichever arm is faster -- and since a treatment that does more work is
                    systematically slower, that bias points AGAINST the treatment every time.

`BOTH_TRUNCATED`    Same agreement, but NEITHER arm finished. Strictly less informative than
                    TRUNCATION_ONLY, where at least one arm's full behaviour is known, and it
                    is broken out as its own class because it is the one that produced a real
                    FALSE PASS: the class that certifies "the harness repeats itself here" is
                    IDENTICAL, and a both-truncated A/A pair mislabelled IDENTICAL will
                    license an A/B perturbation as treatment-attributable while having
                    measured nothing whatsoever.

`PERTURBED`         The traces diverge at an index that exists in both, OR one arm
                    terminated early *of its own accord* (a completed run that is a strict
                    prefix of the other is a real behavioural difference, not a resource
                    accident). Only these pairs can ever be discordant.

THE A/A NOISE FLOOR, AND WHY IT IS NOT OPTIONAL
-----------------------------------------------
Perturbation on its own is ambiguous. A trace pair can diverge because the treatment changed
the agent's behaviour, or because the HARNESS is not deterministic -- and the ARC harness is
not. Two mechanisms, and the second is the one that matters:

  1. By default the generator samples at ``temperature = 0.2 + 0.1*attempt`` and sends no seed
     to the server, so the server picks a fresh random one per request. The harness `seed`
     seeds `random`/`numpy` inside the driver, never the sampler. `CARNOT_ARC_GENERATOR_SEED`
     (REQ-ARC-WMTE-6046) closes that channel, but it is opt-in and default-OFF, so the shipped
     path is still unseeded.
  2. **Seeding it is NOT sufficient.** Measured 2026-07-30 on a provably-same-code A/A -- both
     arms at commit ``f9a458e87``, identical effective sampler seed, identical GGUF, n_ctx,
     action budget and game -- 1 of 2 pairs still diverged (ft09, first divergence at action 26,
     and the two runs disagreed on whether a plan was found at all: 1 vs 0). vc33 was
     byte-identical over 18 actions. So a residual nondeterminism survives the seed, and its
     size is unknown: n=2 supports "it happens", not a rate.

The practical consequence is that the noise floor must be MEASURED per grid, not assumed from
the fact that a seed was set.

That matters in both directions:

  * without a noise floor, a nondeterministic harness makes EVERY cell perturb, the
    pre-flight passes trivially, and the grid is just as dead as before -- the perturbation
    was noise, so nothing in it is attributable to the treatment
  * a cell that perturbs under A/A tells us nothing about the treatment even when it also
    perturbs under A/B, because the difference cannot be attributed

So the honest quantity is ATTRIBUTABLE perturbation: cells that perturb under A/B **and are
byte-identical under A/A**. Those are the cells where the harness demonstrably repeats itself
and the treatment nonetheless changed something. `preflight_verdict` accepts ``noise_pairs``
and, when given, uses attributable perturbation as the ceiling. When it is NOT given the
verdict says so loudly, because an unmeasured noise floor makes a PASS uninterpretable.

The 2026-07-29 retention grid is the cautionary case: its single "perturbed" cell (vc33)
first diverges at action index 17 -- BEFORE the explore budget of 24 transitions is spent, so
before the first induction, so before retention could act at all. Its own pre-registration
specified `ret0b`/`ret1b` A/A control arms for exactly this reason. They were never run, so
whether that one cell was treatment or noise is UNATTRIBUTABLE.

(A first attempt to recover that noise floor retrospectively -- by pairing the retention grid's
`ret1` cells against cells from a DIFFERENT experiment that shared treatment, seed, model and
game -- was RETRACTED, not softened. Those two sets are >=6 agentic commits apart: the held-out
cells ran before ``11cd3c3a9`` introduced engine retention at all, so the "A/A" pair was in fact
an A/B of the very treatment under test, and its cells carry no retention fields to compare.
The claim it supported -- "2 of 5 cells diverge under IDENTICAL code", and hence "vc33's
attributable perturbation is ZERO" -- did not follow and has been removed. The replacement above
is a real same-commit A/A. The generalisable lesson is the one this whole module is about: a
noise floor assembled out of convenient nearby data is not a noise floor, and "same treatment
flag" is not "same code".)

THE THRESHOLD IS DERIVED, NOT PICKED
------------------------------------
The planned analysis is a two-sided sign test over matched pairs -- the standard test when
the endpoint is a small count per cell and the pairing is by game. Under the null, each
discordant pair independently favours either arm with probability 1/2, so the most
favourable possible outcome (every discordant pair pointing the same way) gives

    p = 2 * (1/2) ** d           for d one-way discordant pairs

    d = 4  ->  p = 0.1250     still not significant
    d = 5  ->  p = 0.0625     still not significant
    d = 6  ->  p = 0.0312     significant

So **six** one-way discordant pairs are the arithmetic floor at alpha=0.05, and since a
pair can only be discordant if the treatment perturbed it, the treatment must perturb at
least 6 of the grid's cells. At the customary n=12 grid that is half the cells. This is
computed by `min_one_way_discordant_pairs()` rather than hardcoded, so the number moves
correctly if alpha ever does.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "IDENTICAL",
    "TRUNCATION_ONLY",
    "BOTH_TRUNCATED",
    "PERTURBED",
    "MISSING",
    "PASS",
    "REFUSE",
    "INCONCLUSIVE",
    "MISSING_OBSERVATION_CLASSES",
    "two_sided_sign_test_p",
    "min_one_way_discordant_pairs",
    "classify_trace_pair",
    "preflight_verdict",
    "format_report",
]

# ---- pair classes -------------------------------------------------------------------------
IDENTICAL = "IDENTICAL"
TRUNCATION_ONLY = "TRUNCATION_ONLY"
# BOTH arms were cut short, and they agreed on every action they both took. Added 2026-07-30
# after review: the previous code checked `a == b` BEFORE consulting completeness, so a pair in
# which NEITHER arm got past action 27 of a 400-action budget was labelled IDENTICAL and its
# recorded reason asserted "no downstream endpoint can differ" -- a claim the data cannot
# support, because neither arm was allowed to reach an endpoint. That mislabel was not cosmetic:
# IDENTICAL is the class that certifies the harness as deterministic on a cell, so a
# both-truncated A/A pair could license an A/B perturbation as treatment-attributable while
# actually measuring nothing at all. See `preflight_verdict`'s attribution block.
BOTH_TRUNCATED = "BOTH_TRUNCATED"
PERTURBED = "PERTURBED"
# A cell where one or both arms produced no usable trace at all (blocked, crashed before
# acting, arm absent). Distinct from TRUNCATION_ONLY, where both arms DID act and agreed on
# every shared action. Both are missing observations; only MISSING has nothing to compare.
MISSING = "MISSING"

# The classes that are MISSING OBSERVATIONS rather than measurements. Grouped in one place so
# every consumer (rate denominator, charitable ceiling, attribution) treats them consistently;
# the 2026-07-30 review found the rate denominator and the class docstring disagreeing about
# exactly this set.
MISSING_OBSERVATION_CLASSES = (TRUNCATION_ONLY, BOTH_TRUNCATED, MISSING)

# ---- verdicts -----------------------------------------------------------------------------
PASS = "PASS"
REFUSE = "REFUSE"
# The grid is not finished, and the cells still outstanding could still carry it to a PASS.
# Added 2026-07-30: without this, a partially-run probe returned REFUSE, because every unrun cell
# classifies as MISSING and MISSING is not PERTURBED. That is this module's own central error --
# scoring a missing observation as an unfavourable one -- committed at the verdict level rather
# than the cell level. It matters most in the case that actually arose: a probe stopped early for
# an unrelated reason (a bug found in the arm under test) would otherwise have produced an
# artifact saying REFUSE, i.e. laundering a process decision into a finding that the treatment
# is inert.
INCONCLUSIVE = "INCONCLUSIVE"


def two_sided_sign_test_p(n_one_way_discordant: int) -> float:
    """Best (smallest) two-sided sign-test p-value reachable with ``d`` discordant pairs.

    This is the p-value you get when EVERY discordant pair favours the same arm -- the most
    favourable outcome the data could possibly produce. It is therefore an upper bound on
    how much the experiment could ever prove, and computing it before the experiment runs is
    the whole point of a power pre-flight.

    Under the null hypothesis a discordant pair is a fair coin, so the probability of all
    ``d`` pairs landing the same way is ``(1/2) ** d``, doubled because the test is
    two-sided (we would be equally impressed by ``d`` pairs favouring the control).

    ``d == 0`` returns 1.0: with no discordant pairs there is literally nothing to test,
    and reporting anything below 1.0 would be a claim the data cannot support.
    """
    d = int(n_one_way_discordant)
    if d <= 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**d))


def min_one_way_discordant_pairs(alpha: float = 0.05) -> int:
    """Smallest number of one-way discordant pairs that can reach significance at ``alpha``.

    Derived by walking ``d`` upward until ``two_sided_sign_test_p(d) < alpha`` rather than
    hardcoding the answer, so the threshold cannot silently drift away from the alpha it
    claims to enforce. At the conventional alpha=0.05 this returns 6.
    """
    d = 1
    # 60 is far past any practical grid size and guarantees termination for any alpha > 0.
    while d < 60 and two_sided_sign_test_p(d) >= alpha:
        d += 1
    return d


def _is_prefix(short: Sequence[Any], long: Sequence[Any]) -> bool:
    """Is ``short`` a strict prefix of ``long``? (Equal-length inputs return False.)"""
    return len(short) < len(long) and list(long[: len(short)]) == list(short)


def classify_trace_pair(
    trace_a: Optional[Sequence[Any]],
    trace_b: Optional[Sequence[Any]],
    *,
    a_complete: bool = True,
    b_complete: bool = True,
) -> dict[str, Any]:
    """Classify one matched pair of action traces into one of the four classes.

    ``a_complete`` / ``b_complete`` say whether that arm's run FINISHED on its own terms --
    False when it was cut short by a wall-clock cap, a hard timeout, or a crash. This flag is
    what separates a MISSING OBSERVATION from a real behavioural difference, and it cannot be
    inferred from the traces alone:

      * a truncated trace whose arm ran out of TIME tells us nothing about the treatment;
        the two arms agreed on every action they both took (TRUNCATION_ONLY)
      * a truncated trace whose arm STOPPED ON PURPOSE -- it decided it was done, or its
        action budget ran out earlier because it spent actions differently -- is a genuine
        behavioural difference (PERTURBED)

    Getting this backwards is what turns a resource accident into a fake treatment effect,
    or a real early-stop into an ignored signal. So the caller must supply it honestly from
    the cell's own `timed_out` / `status` / `error` record.

    Returns a dict (not a bare string) because the diagnostic detail -- where the traces
    first diverged, how long each was -- is what makes a REFUSE actionable rather than
    merely discouraging.
    """
    a = list(trace_a) if trace_a is not None else None
    b = list(trace_b) if trace_b is not None else None

    out: dict[str, Any] = {
        "len_a": None if a is None else len(a),
        "len_b": None if b is None else len(b),
        "a_complete": bool(a_complete),
        "b_complete": bool(b_complete),
        "common_prefix_len": None,
        "first_divergence_index": None,
    }

    # Nothing to compare. An empty trace counts as absent: an arm that never emitted an
    # action cannot evidence a behavioural difference either way.
    if not a or not b:
        out["cls"] = MISSING
        out["why"] = "one or both arms produced no action trace"
        return out

    n = min(len(a), len(b))
    pref = 0
    while pref < n and a[pref] == b[pref]:
        pref += 1
    out["common_prefix_len"] = pref

    if a == b:
        # Equal traces are only an IDENTICAL *measurement* if both arms were allowed to finish.
        # If either was cut short, the agreement covers only the actions both happened to take
        # and says nothing about what either would have done next -- so it is a missing
        # observation, exactly like the strict-prefix case below (2026-07-30 review).
        if a_complete and b_complete:
            out["cls"] = IDENTICAL
            out["why"] = (
                "byte-identical action traces, both arms run to completion: the two arms are "
                "the same agent on this cell, so no downstream endpoint can differ"
            )
        elif not a_complete and not b_complete:
            out["cls"] = BOTH_TRUNCATED
            out["why"] = (
                f"both arms were cut short after the same {len(a)} actions and agreed on all "
                f"of them. NEITHER arm reached an endpoint, so this pair cannot show that the "
                f"treatment is inert and cannot serve as a determinism witness -- it is a "
                f"MISSING observation, not a tie"
            )
        else:
            out["shorter_arm"] = "a" if not a_complete else "b"
            out["cls"] = TRUNCATION_ONLY
            out["why"] = (
                f"traces agree on all {len(a)} actions, but arm {out['shorter_arm']} was cut "
                f"short at exactly that point, so whether it would have continued to diverge "
                f"is unobserved. MISSING observation, never a zero"
            )
        return out

    if pref < n:
        # Divergence at an index BOTH traces reach -- unambiguously different behaviour.
        out["first_divergence_index"] = pref
        out["cls"] = PERTURBED
        out["why"] = f"traces diverge at action index {pref}"
        return out

    # One trace is a strict prefix of the other. Which class depends entirely on WHY the
    # shorter run stopped.
    shorter_is_a = len(a) < len(b)
    shorter_complete = a_complete if shorter_is_a else b_complete
    out["shorter_arm"] = "a" if shorter_is_a else "b"
    if shorter_complete:
        out["cls"] = PERTURBED
        out["why"] = (
            f"the shorter arm ({out['shorter_arm']}) terminated on its own terms after "
            f"{min(len(a), len(b))} actions while the other continued to "
            f"{max(len(a), len(b))} -- a real behavioural difference, not a resource cap"
        )
    else:
        out["cls"] = TRUNCATION_ONLY
        out["why"] = (
            f"the shorter arm ({out['shorter_arm']}) was cut short before finishing; both "
            f"arms agreed on all {min(len(a), len(b))} actions they both took. This is a "
            f"MISSING observation and must never be scored as a zero"
        )
    return out


def preflight_verdict(
    pairs: Mapping[str, Mapping[str, Any]] | Iterable[tuple[str, Mapping[str, Any]]],
    *,
    alpha: float = 0.05,
    planned_n_cells: Optional[int] = None,
    noise_pairs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    noise_pairs_b: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """PASS or REFUSE the full grid, from the per-cell classifications.

    ``pairs`` maps a cell label (conventionally the game) to the dict returned by
    `classify_trace_pair` for the TREATMENT-vs-CONTROL comparison. ``planned_n_cells`` is the
    size of the grid the caller intends to run if this passes; it defaults to the number of
    probed cells, and it exists so a caller can probe 12 cells while planning 24. When it
    EXCEEDS the probed count (and a noise floor was measured) the decision switches from the
    strict at-probed-n ceiling to the projected attributable rate at the planned size -- see
    the decision block below for why, and for why projection needs a noise floor.

    ``noise_pairs`` is the same mapping for an A/A comparison -- one arm against a byte-
    identical replicate of itself. See the module docstring: without it, a nondeterministic
    harness makes every cell perturb and the pre-flight passes on noise. When it is supplied,
    the ceiling on discordant pairs is ATTRIBUTABLE perturbation (perturbs under A/B AND
    byte-identical under A/A) rather than raw perturbation.

    ``noise_pairs_b`` is the SECOND arm's A/A comparison, and it closes a hole this module had
    from the day it shipped (found by review 2026-07-30, on the first real grid to use it).

    An A/B comparison spans TWO arms. A single ``noise_pairs`` measures the determinism of ONE
    of them. The module nonetheless treated that single measurement as licensing attribution for
    the whole comparison -- so a grid whose control arm was the noisy one would credit the
    control arm's self-perturbation to the treatment, and the module would certify it. In the
    grid that exposed this, the measured floor was head-vs-headb 0/6 while every A/B pair held
    ONE unreplicated base run; and the treatment under test changed a search DEDUP KEY, i.e.
    exactly the kind of change that can stabilise iteration order and so make the two arms
    differ in determinism. "The treatment arm repeats itself" is not evidence that "the control
    arm repeats itself", and the whole point of a noise floor is that it must not rest on an
    argument.

    When both mappings are supplied, a cell is attributable only when BOTH arms were separately
    witnessed to repeat themselves on THAT cell. A cell present in one mapping and absent from
    the other is UNWITNESSED on the missing side and is excluded -- a missing observation, never
    a pass. When only one is supplied the behaviour is unchanged for compatibility, but the
    result now carries ``single_arm_noise_floor_warning`` saying so, because a caller who reads
    ``noise_floor_measured: True`` and stops there is reading the same over-claim this
    parameter exists to prevent.

    The decision rule, and why each part of it is there:

      * ``required`` one-way discordant pairs come from `min_one_way_discordant_pairs`.
      * The CEILING on discordant pairs is the number of ATTRIBUTABLE perturbed cells (or of
        raw perturbed cells if no noise floor was measured), because a pair whose two arms
        emitted identical actions cannot differ on any endpoint, and a truncated pair is a
        missing observation that will be EXCLUDED from the analysis rather than scored.
      * If that ceiling is below ``required``, no possible outcome of the grid reaches
        ``alpha``. Running it cannot produce a finding. REFUSE.
      * UNLESS the caller is probing a strict subset (``planned_n_cells`` > probed cells) and a
        noise floor exists, in which case the ceiling at the PROBED size is not the relevant
        bound -- the expected attributable count at the PLANNED size is. A 33%-perturbation
        probe of 12 cells cannot reach 6 discordant pairs, but the 1000-cell grid it is
        screening expects ~333, and refusing that grid would be arithmetically wrong.

    A second, deliberately CHARITABLE ceiling is also reported, which pretends every
    truncation-only cell would have been discordant had it been allowed to finish. It is not
    used for the decision -- coding a missing observation as a favourable one is precisely
    the error this module exists to prevent -- but when even the charitable ceiling falls
    short, the refusal is beyond argument, and saying so is more persuasive than the strict
    number alone.
    """
    items = list(pairs.items() if isinstance(pairs, Mapping) else pairs)
    counts = {IDENTICAL: 0, TRUNCATION_ONLY: 0, BOTH_TRUNCATED: 0, PERTURBED: 0, MISSING: 0}
    for _label, rec in items:
        counts[str(rec.get("cls", MISSING))] = counts.get(str(rec.get("cls", MISSING)), 0) + 1

    n_probed = len(items)
    n_comparable = n_probed - counts[MISSING]
    n_perturbed = counts[PERTURBED]
    required = min_one_way_discordant_pairs(alpha)
    n_truncation_affected = counts[TRUNCATION_ONLY] + counts[BOTH_TRUNCATED]

    # ---- attribution against the A/A noise floor -----------------------------------------
    # A cell earns "attributable" only when the harness demonstrably repeats itself there
    # (A/A byte-identical, BOTH A/A arms run to completion) AND the treatment nonetheless
    # changed the trace. Anything else -- A/A perturbed, A/A truncated, A/A missing -- leaves
    # the difference unattributable, and the conservative reading is the only defensible one:
    # we cannot tell treatment from noise.
    #
    # The completeness requirement is stated TWICE on purpose (2026-07-30 review). Since the
    # `classify_trace_pair` fix, a both-truncated A/A pair can no longer come back IDENTICAL,
    # so this is already structurally impossible via that path -- but the check is repeated
    # here because `noise_pairs` is plain data a caller can hand-build, and this is the exact
    # direction that produces a FALSE PASS: a truncated A/A pair measures nothing, yet under
    # the old rule it would certify the harness as deterministic on that cell and license an
    # A/B perturbation as treatment-attributable. A false REFUSE only wastes an experiment; a
    # false PASS spends hours and then reports a number nobody can attribute.
    noise_measured = noise_pairs is not None
    noise_maps: list[Mapping[str, Mapping[str, Any]]] = [
        m for m in (noise_pairs, noise_pairs_b) if m is not None
    ]
    n_noise_arms = len(noise_maps)
    attributable: list[str] = []
    unattributable: dict[str, str] = {}
    lacking_second_witness: list[str] = []
    if noise_measured:
        for label, rec in items:
            if str(rec.get("cls")) != PERTURBED:
                continue
            # EVERY supplied noise mapping must witness this cell as deterministic. `all()` over
            # the mappings, rather than a check against one of them, is the whole fix: with a
            # single mapping this is identical to the previous behaviour, and with two it refuses
            # to let one arm's determinism speak for the other's.
            reasons: list[str] = []
            for m in noise_maps:
                noise_rec = m.get(label, {})
                noise_cls = str(noise_rec.get("cls", MISSING))
                both_complete = bool(noise_rec.get("a_complete")) and bool(
                    noise_rec.get("b_complete")
                )
                if noise_cls == IDENTICAL and both_complete:
                    continue
                reasons.append(
                    "IDENTICAL_BUT_AA_ARM_TRUNCATED" if noise_cls == IDENTICAL else noise_cls
                )
            if not reasons:
                attributable.append(label)
            else:
                # Report the FIRST failing reason for compatibility with the single-arm output
                # shape, and name the multi-arm case explicitly so a reader can tell "one arm
                # said no" from "one arm never spoke".
                unattributable[label] = reasons[0] if n_noise_arms == 1 else "+".join(reasons)
                if n_noise_arms > 1 and MISSING in reasons:
                    lacking_second_witness.append(label)
    n_attributable = len(attributable) if noise_measured else None

    ceiling_strict = n_attributable if noise_measured else n_perturbed
    ceiling_charitable = ceiling_strict + n_truncation_affected

    # Perturbation RATE is computed over cells that YIELDED A USABLE OBSERVATION -- comparable
    # cells minus the truncation-affected ones (2026-07-30 review).
    #
    # The previous denominator was every comparable cell, which put truncation-affected cells in
    # the denominator but never the numerator: i.e. it scored them as zeros. That is precisely
    # what this module's own TRUNCATION_ONLY documentation forbids ("must never be scored as a
    # zero"), and the bias has a direction -- a treatment that does MORE work is systematically
    # slower, so it is systematically the arm that gets truncated, so the error always points
    # against the treatment. On the 2026-07-29 retention grid it reported 1/12 = 0.083 (and
    # "72 cells needed") where the usable-observation rate is 1/9 = 0.111 (54 cells needed).
    #
    # The comparable-cell rate is still emitted, under a name that says what it does, because it
    # is the conservative bound and a REFUSE that holds under both is stronger than one that
    # needs the favourable denominator.
    n_usable = n_comparable - n_truncation_affected
    rate = (ceiling_strict / n_usable) if n_usable else 0.0
    rate_truncations_as_zero = (ceiling_strict / n_comparable) if n_comparable else 0.0
    planned = int(planned_n_cells) if planned_n_cells else n_probed
    cells_needed = math.ceil(required / rate) if rate > 0 else None

    # ---- the decision ---------------------------------------------------------------------
    # Two regimes, because a caller can legitimately probe a SUBSET of the grid it intends to
    # run, and the advertised workflow ("probe 12 cells while planning 24") was previously
    # inert: only `ceiling_strict >= required` decided, so a 33%-perturbation probe of 12 cells
    # was REFUSED even at planned_n_cells=1000, where ~333 perturbed cells are expected against
    # the 6 needed. The tool computed `cells_needed` and then ignored it (2026-07-30 review).
    #
    # Projection is only permitted when a noise floor was measured, because projecting from an
    # UNATTRIBUTED rate just forecasts how much noise the bigger grid will contain.
    if planned > n_probed and noise_measured:
        projected = rate * planned
        decision_basis = "projected_attributable_rate_at_planned_n"
        verdict = PASS if projected >= required else REFUSE
    else:
        projected = None
        decision_basis = "strict_attributable_ceiling_at_probed_n"
        verdict = PASS if ceiling_strict >= required else REFUSE

    # A REFUSE is only a FINDING when the grid actually ran. If cells are still outstanding and
    # they could still carry the ceiling to `required`, the honest answer is INCONCLUSIVE -- the
    # probe has not decided yet. Only PASS is unaffected: a ceiling already at `required` cannot
    # be undone by running more cells.
    max_reachable_if_all_missing_land_favourably = ceiling_strict + counts[MISSING]
    still_open = (
        verdict == REFUSE
        and counts[MISSING] > 0
        and max_reachable_if_all_missing_land_favourably >= required
    )
    if still_open:
        verdict = INCONCLUSIVE

    return {
        "decision_basis": decision_basis,
        "n_missing_cells_still_outstanding": counts[MISSING],
        "max_reachable_ceiling_if_every_missing_cell_were_attributable": (
            max_reachable_if_all_missing_land_favourably
        ),
        "grid_is_incomplete_and_still_arithmetically_open": still_open,
        "projected_attributable_cells_at_planned_n": (
            None if projected is None else round(projected, 3)
        ),
        "n_usable_observations": n_usable,
        "n_truncation_affected": n_truncation_affected,
        "perturbation_rate_over_comparable_counting_truncations_as_zero": round(
            rate_truncations_as_zero, 4
        ),
        "verdict": verdict,
        "alpha": alpha,
        "required_one_way_discordant_pairs": required,
        "sign_test_arithmetic": {
            str(d): round(two_sided_sign_test_p(d), 6) for d in range(1, required + 2)
        },
        "n_probed_cells": n_probed,
        "n_comparable_cells": n_comparable,
        "counts": counts,
        "noise_floor_measured": noise_measured,
        "n_perturbed_raw": n_perturbed,
        "n_perturbed_attributable": n_attributable,
        "attributable_cells": sorted(attributable),
        "unattributable_cells_with_aa_class": dict(sorted(unattributable.items())),
        "perturbation_rate_over_usable_observations": round(rate, 4),
        "discordance_ceiling_strict": ceiling_strict,
        "discordance_ceiling_charitable_counts_truncations": ceiling_charitable,
        "best_reachable_p_strict": round(two_sided_sign_test_p(ceiling_strict), 6),
        "best_reachable_p_charitable": round(two_sided_sign_test_p(ceiling_charitable), 6),
        "planned_n_cells": planned,
        "cells_needed_at_observed_perturbation_rate": cells_needed,
        "per_cell": {label: dict(rec) for label, rec in items},
        "per_cell_noise": (
            None if not noise_measured else {k: dict(v) for k, v in (noise_pairs or {}).items()}
        ),
        "per_cell_noise_b": (
            None if noise_pairs_b is None else {k: dict(v) for k, v in noise_pairs_b.items()}
        ),
        "n_noise_arms_measured": n_noise_arms,
        "cells_perturbed_but_lacking_a_second_arm_noise_witness": sorted(lacking_second_witness),
        "single_arm_noise_floor_warning": (
            None
            if n_noise_arms >= 2 or not noise_measured
            else (
                "ONLY ONE ARM'S DETERMINISM WAS MEASURED. An A/B comparison spans two arms, so a "
                "noise floor measured on one of them does not license attribution for the pair: "
                "if the UNREPLICATED arm is the noisy one, its self-perturbation is credited to "
                "the treatment and this verdict certifies it. Supply the other arm's A/A as "
                "`noise_pairs_b`. Note that a treatment which touches hashing, dedup keys, "
                "iteration order or tie-breaking can CHANGE how deterministic an arm is, so the "
                "two arms' floors cannot be assumed equal -- that is precisely when this matters "
                "most."
            )
        ),
        "noise_floor_warning": (
            None
            if noise_measured
            else (
                "NO A/A NOISE FLOOR WAS MEASURED. The ARC generator samples at a temperature "
                "and sends no seed, so the harness is not deterministic; raw perturbation "
                "therefore cannot be attributed to the treatment. A PASS computed without a "
                "noise floor is uninterpretable -- run one arm against a byte-identical "
                "replicate of itself and pass it as `noise_pairs`."
            )
        ),
        # Stated on every verdict, PASS included, because the single most likely way to
        # misuse this tool is to read a PASS as "the experiment is powered".
        "interpretation": (
            "PASS means only that the treatment perturbs enough cells for the planned sign "
            "test to be ABLE to reach significance. It does NOT mean the experiment is "
            "powered: perturbation is necessary but not sufficient for discordance on the "
            "endpoint. REFUSE means no possible outcome of the grid reaches alpha, so "
            "running it cannot produce a finding. INCONCLUSIVE means the probe itself is not "
            "finished -- cells are still outstanding that could carry it to a PASS -- so it is "
            "NOT evidence that the treatment is inert and must never be cited as one."
        ),
    }


def format_report(verdict: Mapping[str, Any], *, title: str = "") -> str:
    """Human-readable rendering of a verdict, for a log or an artifact's prose field."""
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    v = verdict
    lines.append(f"VERDICT: {v['verdict']}   (alpha={v['alpha']})")
    if v.get("grid_is_incomplete_and_still_arithmetically_open"):
        lines.append(
            f"  NOTE: the grid is INCOMPLETE -- {v['n_missing_cells_still_outstanding']} cells "
            f"outstanding, and the ceiling could still reach "
            f"{v['max_reachable_ceiling_if_every_missing_cell_were_attributable']} vs "
            f"{v['required_one_way_discordant_pairs']} required. This is NOT a refusal."
        )
    lines.append(
        f"cells probed {v['n_probed_cells']}, comparable {v['n_comparable_cells']}, "
        f"counts {v['counts']}"
    )
    lines.append(
        f"one-way discordant pairs required for p<{v['alpha']}: "
        f"{v['required_one_way_discordant_pairs']}   "
        f"(2*(1/2)^d: {v['sign_test_arithmetic']})"
    )
    lines.append(
        f"ceiling on discordant pairs: strict {v['discordance_ceiling_strict']} "
        f"(best reachable p = {v['best_reachable_p_strict']}), charitable "
        f"{v['discordance_ceiling_charitable_counts_truncations']} "
        f"(best reachable p = {v['best_reachable_p_charitable']})"
    )
    lines.append(
        f"perturbation rate over USABLE observations "
        f"{v['perturbation_rate_over_usable_observations']} "
        f"(n_usable {v['n_usable_observations']}, truncation-affected "
        f"{v['n_truncation_affected']} EXCLUDED as missing, not scored as zeros; "
        f"counting them as zeros instead would give "
        f"{v['perturbation_rate_over_comparable_counting_truncations_as_zero']}); "
        f"cells needed at that rate: {v['cells_needed_at_observed_perturbation_rate']}"
    )
    lines.append(
        f"decision basis: {v['decision_basis']}"
        + (
            ""
            if v.get("projected_attributable_cells_at_planned_n") is None
            else f" -- projected {v['projected_attributable_cells_at_planned_n']} attributable "
            f"cells at planned n={v['planned_n_cells']} vs "
            f"{v['required_one_way_discordant_pairs']} required"
        )
    )
    if v.get("noise_floor_measured"):
        lines.append(
            f"A/A noise floor measured on {v.get('n_noise_arms_measured', 1)} arm(s): "
            f"raw perturbed {v['n_perturbed_raw']}, "
            f"attributable {v['n_perturbed_attributable']} {v['attributable_cells']}; "
            f"unattributable {v['unattributable_cells_with_aa_class']}"
        )
        # Printed as a WARNING rather than tucked into the dict, because the failure mode is a
        # reader who sees "noise floor measured" and stops.
        if v.get("single_arm_noise_floor_warning"):
            lines.append(f"WARNING: {v['single_arm_noise_floor_warning']}")
    else:
        lines.append(f"WARNING: {v['noise_floor_warning']}")
    for label, rec in v["per_cell"].items():
        lines.append(
            f"  {label:8s} {str(rec.get('cls')):16s} "
            f"len {rec.get('len_a')}/{rec.get('len_b')} "
            f"prefix {rec.get('common_prefix_len')} -- {rec.get('why')}"
        )
    lines.append(v["interpretation"])
    return "\n".join(lines)
