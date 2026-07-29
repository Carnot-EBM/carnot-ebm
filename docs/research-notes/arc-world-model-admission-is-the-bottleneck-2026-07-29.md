# The ARC live agent's bottleneck is world-model induction, not the gate that rejects it

**Date:** 2026-07-29
**Status:** measurement note; three candidate levers ruled out with evidence
**Origin:** the engine-retention -> banked-levels A/B (REQ-ARC-WMTE-6035). While establishing
whether that A/B *could* detect anything, the support analysis walked upstream and found the
binding constraint.

## The one metric that has not moved

Banked levels on held-out games has been flat at 3 (31B) / 3 (9B) across an inducer migration
that made the generator roughly 6x better. That non-result is usually read as "the lever we
tried was weak." It is not. It is the same non-result every lever in this area will produce
until the finding below is addressed.

## The chain, each link measured

**1. Retention acts downstream of admission.** Best-engine retention decides WHICH refinement
round's engine the store ends up holding. That can only change the agent's behaviour if the
stored engine reaches the policy at all.

**2. Multi-round refinement and reaching the policy are anti-correlated — perfectly.** Across
the 13 cells of the 2026-07-28 held-out run:

| | reached policy = True | reached policy = False |
|---|---|---|
| any multi-round attempt | **0** | 11 |
| all attempts single-round | 2 | 0 |

The mechanism is structural, not incidental: a second refinement round runs *because* round 1
failed the admission gate. If the later rounds fail too, nothing is accepted and nothing
reaches the policy. If round 1 passes, refinement stops at one round and retention is a
definitional no-op. Retention's precondition and its opportunity are mutually exclusive in
practice.

**3. Admission is where it dies.** Of 17 induction attempts, **12 had no round accepted at
all**, and 9 were blocked specifically by `world_model_accuracy_below_threshold`.

**4. Relaxing the threshold does not help.** `heldout_accuracy` over 45 rounds is degenerate-
bimodal — 32 rounds score exactly 0.0, 9 score exactly 1.0, and only 4 land in between.
Thresholds from the shipped 1.0 all the way down to 0.6 admit the **identical 9 rounds**.
There is no continuum to trade precision against recall on.

**5. Nor does switching to the graded metric.** `CARNOT_ARC_TRUST_METRIC=cell_recall` already
exists as a lever. Computing it directly from `experiment_6012`'s recorded
`correct_changed_cells / true_changed_cells` for the real induced (`ondisk`) engines, over
11 games x 3 seeds:

    median cell_recall 0.0033    max 0.0870    n = 33
    a graded gate at 0.90 / 0.75 / 0.50 / 0.25 / 0.10 admits 0/33 in every case

**6. The gate has its own separate hole — which does NOT rescue the induced engines.** An
earlier draft of this note claimed "the gate is fine, it admits a correct hand-written `base`
engine 31/33." That was wrong, and exp6012's own docstring says so. In 30 of those 33 rows
`base` is `visible_lookup` — a table tabulated from the corpus, which that experiment
designates an **attack baseline, explicitly "never the must-not-fire control"**, because a
function of the visible grid cannot be right on a hidden-state game. The genuine control is
dc22's hand-written rule, and there the measured result is the opposite of my claim: REQ-6011
admits it 3/3, but **the LIVE hidden-state gate rejects it on 2/3** — that gate is, in
exp6012's words, "simultaneously too STRICT on an honest partial model and BLIND to a spurious
writer."

So there are two independent problems, and only one of them is this note's subject. The gate
hole is real and already logged (exp6012's verdict:
`complete_hidden_state_gate_hole_measured_req6011_not_wired_into_that_branch`). It does not
explain the induced engines' rejection, because the induced engines are not honest partial
models being unfairly excluded — at ~0.3% changed-cell recall, **any** correctly calibrated
gate rejects them. Fixing the gate hole is worth doing on its own merits; it will not move
banked levels while induction is where it is. The one claim that survives unchanged is that
`identity` is rejected 0/33.

## What the induced world models are actually doing

The two numbers together are diagnostic. Exact-match `heldout_accuracy` sits at intermediate
values (0.125-0.6) while cell recall on the same rows is ~0.003. That combination has one
explanation: the engines predict *no-op frames* correctly — most cells do not change, so
copying the input scores well — and get essentially every *changed* cell wrong. On the cells
that carry the dynamics, the induced world model is behaving like the identity engine, which
the gate correctly rejects.

So the failure is not "the model is slightly imperfect and the gate is too strict." It is
"the model has not learned the dynamics at all, and the gate is correctly saying so."

## Consequences

* **Retention cannot move banked levels**, and neither can any further work on the admission
  gate, its threshold, or its metric. All three are downstream of, or orthogonal to, a world
  model that is wrong about ~99.7% of changed cells.
* **This explains the inducer migration's null.** A 6x better generator still has to produce a
  world model that predicts changed cells. Making the proposer stronger did not change what it
  is being asked to produce or how that is scored, so it did not convert.
* **The live agent's own note already suspected this** — `arc_competition_agent.py:5751`
  records that the `min_heldout_accuracy=1.0` gate "is rarely met (0/6 rounds across a full
  real run on g50t)". This note supplies the reason and rules out the obvious fixes.

## Caveat, stated rather than buried

The cell-recall figures come from `results/arc_e3_origin_fixtures`, whose on-disk engines
predate best-engine retention and may therefore include stores corrupted by the
last-write-wins overwrite that REQ-ARC-WMTE-6035 fixes (the ka59 fidelity 1.0000 -> 0.0000
case). That weakens point 5 as a statement about *freshly induced* engines specifically.
It does not weaken the chain, because points 3 and 4 are measured on fresh live inductions:
71% of fresh rounds score exactly 0.0, and 12 of 17 attempts had no accepted round.

## The error class, characterised (the question this note originally deferred)

The note first ended by asking "what error class is this?" and calling it a cheap CPU-only
follow-up. It is cheap enough to have done immediately, from the `correct_changed_cells` /
`spurious_changed_cells` / `noop_hallucination_rate` already recorded per row in exp6012. The
33 rows split cleanly into two structural failure modes, neither of which is "close but
imperfect":

**Mode A — the no-op world model (12/33 rows; ar25, ka59, re86, and cd82 on changed frames).**
`correct_changed_cells = 0` and `spurious_changed_cells = 0`. The engine predicts that nothing
changes, ever. It is the identity engine in all but name — which is why it scores respectably
on exact-match accuracy (most cells genuinely do not change) and zero on changed-cell recall.

**Mode B — the runaway writer (21/33 rows; cn04, dc22, g50t, m0r0, sc25, sk48, wa30).** The
engine does write, but wildly over-writes:

| game | true changed cells | correct | spurious | spurious / true |
|---|---|---|---|---|
| cn04@1 | 4346 | 600 | 33021 | **7.6x** |
| g50t@0 | 1504 | 168 | 14612 | **9.7x** |
| sc25@0 | 453 | 28 | 4687 | **10.3x** |
| sk48@0 | 2817 | 515 | 7714 | 2.7x |
| wa30@1 | 722 | 18 | 2200 | 3.0x |

It invents between 3x and 10x more changed cells than reality contains. And
`noop_hallucination_rate` reaches **1.0** on cn04@0 — on frames where reality did nothing, the
engine invents a change every single time. (cd82 is a mixture: no-op on changed frames like
Mode A, but hallucinating at 0.44-0.55 on no-op frames. It is counted in Mode A on the
changed-cell criterion and flagged here rather than forced into one bucket.)

This matters for what to try next. The two modes want opposite corrections — Mode A needs the
inducer to predict *any* dynamics at all, Mode B needs it *constrained* from over-writing — so
a single undifferentiated "make the inducer better" intervention is unlikely to move both. A
change-magnitude prior or a sparsity constraint on predicted writes addresses Mode B directly
and is cheap to test; Mode A looks like a failure to extract the action semantics at all.

## The next question worth asking

Not "how do we admit more engines" but **"why does induction produce a world model that is
right about static cells and wrong about moving ones, and what representation or supervision
would change that?"** Concretely, the cheapest discriminating experiment is to take a game
where a correct `base` engine exists, diff it against the induced engine on the changed-cell
predictions, and characterise the error class — is the induced engine missing the action
semantics, the object identity, or the update rule? That is a CPU-only analysis over existing
fixtures and needs no GPU time.
