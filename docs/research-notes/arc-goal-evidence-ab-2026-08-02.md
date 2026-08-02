# Does giving the ARC goal prompt the agent's own evidence stop the model declining?

**Date:** 2026-08-02
**Artifact:** `results/outer_loop_arc_goal_evidence_ab_20260802.json`
**Harness:** `results/arc_goal_evidence_20260802/`
**Status:** measured; both shipped knobs remain DEFAULT OFF. Flipping either is an operator decision.

## The defect this was about

Of 138 induced engines from 21 games, 93 are live and 71 fail on the **goal**, not the dynamics.
Reading all 71:

| count | shape | note |
|---|---|---|
| 34 | unconditional `return False` | four say why in their own docstring: *"Given no WIN STATE grid was provided..."* |
| 11 | "the whole board becomes one colour" | a generic ARC trope referencing nothing observed |
| 9 | a real observed region with an unreachable threshold | the same cluster as 11 of the 22 successes — calibration, not kind |
| 17 | other | colour-elimination, no-return, self-contradictory object tests, connectivity |

62 of the 71 had the information **available** from the agent's own observations.
`is_level_complete` appears exactly once in a ~5,000-character induction prompt, as an interface
stub. Given zero information about the goal, `return False` is arguably the honest answer.

Two knobs already exist and both ship OFF:

- `CARNOT_ARC_GOAL_PROMPT_TRANSITIONS=1` attaches the observed transitions to
  `_goal_only_prompt` (`arc_executable_world_model.py`).
- `CARNOT_ARC_GOAL_DEDUP=1` stops an evidence-grounded predicate being shadowed by an
  evidence-free one.

Nobody had measured whether either helps.

## What the pre-flight found before any GPU time

Classifying all 116 world models in the frozen shipped-path corpus
(`results/arc_object_perception_ab_change_fidelity_20260801/engines`) with this run's own AST
classifier:

| measure | value |
|---|---|
| DECLINED rate overall | 0.431 |
| TROPE rate overall | 0.207 |
| split-induce rate | 0.198 |
| **DECLINED on split-induce cells** | **0.174** |
| **DECLINED on combined-path cells** | **0.495** |

This is the single most important number in the study and it was available for free.
**Both knobs live only in the split-induce fallback**, and `_goal_only_prompt` is never built at
all when the combined induce call succeeds. So the declines are overwhelmingly *not where the
knobs are*: 46 of the 50 frozen declines came from the combined path, which neither flag can
touch.

Arithmetic ceiling on the live-path (ITT) effect: `0.198 x 0.174 = 3.4 percentage points`.
No design at a feasible n resolves 3.4 points. That was **declared before the run**, not
discovered afterwards — see `out/preregistration.json`.

## Design

Two stages, one generator (`gemma-4-31B-it-qat` Q4_K_XL, one llama.cpp CUDA server, non-default
port, `n_ctx` read back from `/props`, build proven from `/proc/<pid>/exe`).

**Stage 2 — the live induce path, the arm structure the brief specifies.**
`A` both knobs off (shipped) · `B` prompt=1 · `C` prompt=1 + dedup=1 · `AA` = `A` at a different
seed base (the noise floor). Per cell, **whether the goal-only call actually ran** is recorded by
wrapping the bound method — an arm whose mechanism never fired is an *untested* arm, not a
refuted one. Declared underpowered on the primary before running, per the ceiling above.

**Stage 1 — the goal-only call, made directly.** `gA` evidence-free · `gB` transitions attached ·
`gAA` noise floor. Same shipped `_goal_only_prompt`, same shipped
`generate(required=("is_level_complete",), tries=3, codeonly_eligible=True)`. The mechanism fires
in 100% of cells by construction, so the contrast is measured rather than diluted to nothing.
This is a **component** measurement of a live-path component and is labelled as such.

**The primary is predicate SHAPE, not the goal gate.** `plan_found` is an exact function of the
gate's kind (0 mismatches in 138), so grading a goal intervention with it grades the gate using
the gate. Shape is decided from the syntax tree by `classify.py`, which reuses the 2026-08-01
anatomy clusters rather than re-deriving them:

- **DECLINED** — constant `False`, no return at all, or raises `NameError`.
- **TROPE** — whole-board uniformity, or colour elimination over the whole board.
- **GROUNDED** — names a region, cell or object whose literal actually appears in the agent's own
  observed deltas. The grounding check is the one thing added on top of the cluster layer.

Clustering is at the **game**. Replicates within a game share the window, the transitions and the
prompt; treating them as independent inflated a sibling experiment's p from 0.125 to 0.049 on
2026-07-31 and had to be corrected.

## Nothing crosses the line

The only thing added to any prompt is the agent's **own observed transitions**, rendered by the
same `_transitions_block` the engine prompt already uses. No game source, no hand-written
adapter, no curated win example. `_previous_level_complete_grid` is passed as `None` in every arm
— it is the *next* level's opening board, not a win state (the win-state poison corrected
2026-07-29).

The transitions shown are the prefix split, so **`levelup_rows_in_shown` is 0 for all 20 games**:
every predicate in this run was written by a model that had never seen the game won. Verified by
inspecting a rendered treatment prompt directly — every delta line reads `level 0->0`. This is
exactly the condition a hidden game presents, which is why stall games are in the roster on
purpose.

`solve_provenance: development_proxy` — no game is solved and no level is banked; the measurement
is offline on public games.

## The mechanism finding

The treatment prompt is **365 characters in control and 3,248–7,654 in treatment**. The
control call returns in ~3 seconds; the treatment call runs to the 4096-token cap, three
attempts, and the code block is truncated mid-definition:

```
local model code unusable after 3 tries
(syntax error line 192: expected an indented block after function definition on line 3)
```

This is the same failure mode `_split_induce`'s own docstring documents for the *combined*
call — "the model rambles its analysis INTO engine() comments, exhausts the token budget, and
never writes is_level_complete". The focused goal call was fast **because** it had nothing to
ramble about. Attaching the evidence reintroduces the very failure the split was designed to
avoid.

Because of this, **differential missingness is tested first and reported before any shape
number**. If the treatment loses cells the control keeps, every shape rate is conditioned on a
post-treatment variable and none of them supports a causal claim on its own. `analyse.py` runs
the missingness contrast as the first test for exactly this reason.

## Two structural results that do not depend on n

**1. The byte-identity check fired — and found a harness defect rather than confirming the
assumption.** `B` and `C` share `A`'s seed within a `(game, replicate)`, so on a cell where the
combined induce call succeeded and the goal-only prompt was never built, all three arms should
produce the same bytes. On `ar25` they do. **On `bp35` they do not**: `A` wrote 3,722 bytes
(`bcf42b01…`) while `B` and `C` both wrote 4,150 bytes (`64b7c69b…`) — same seed 8300, zero
content failures, `goal_only_call_ran` false in all three, so neither knob could have acted.

`B` and `C` are byte-identical to *each other* on both games (2/2), and they differ in the dedup
knob — so the divergence is not knob-driven. What differs between `A` and `(B, C)` is **position**:
`A` is the first cell of the game block and `B`/`C` follow it against a KV cache warmed by the
identical preceding request.

**Consequence, and it matters for the whole design:** a fixed `CARNOT_ARC_GENERATOR_SEED` does
**not** guarantee a byte-identical completion on this server — llama.cpp's sampled output depends
on cache and batch state as well as on the seed. The pairing this design leans on is therefore
*approximate*, not exact, and non-firing cells contribute **nuisance variation** rather than the
exact zero the pre-run reasoning assumed. That also promotes the arm-order confound below from a
timing-only issue to one that touches shape. This is reported, not dropped: surfacing exactly
this is what the check was written for.

**2. The control arm reproduces the defect verbatim.** `ar25`'s control world model contains:

```python
def is_level_complete(grid):
    # Win state usually involves reaching a certain position or clearing blocks.
    # No win state provided, but typically it's when patterns reach the bottom or a target.
    # We'll return False unless we see a clear pattern.
    return False
```

That is the 34-of-71 class, live, in the shipped path, saying so in its own comments.

## What the treatment writes when it survives

On `bp35`, same seed, the three stage-1 arms diverge exactly as the hypothesis predicts:

| arm | prompt | what it wrote |
|---|---|---|
| `gA` control | 365 chars, no evidence | "one solid rectangle of a single colour" — invents a shape rule referencing nothing observed |
| `gAA` control, other seed | 365 chars, no evidence | `len(non_zero_colors) == 1` — the canonical uniformity trope |
| `gB` treatment | 7,654 chars of its own deltas | reads a **progress bar** out of the deltas — "cells in r63 are being filled one by one" — and returns `np.all(grid[63, :] != 0)` |

This is the tn36 mechanism the brief names as the existence proof: a model with **zero observed
wins** inducing a satisfiable goal by reading progress out of what it saw. The treatment does the
intended thing *when it survives*. The same completion also shows the pathology: it repeats lines,
degenerates into `# return np.// a bit too a bit bit bit`, and reaches a valid predicate with
little budget to spare. Survival and rambling are the same phenomenon seen from two sides.

## A grounding caveat found by inspection, and what was done about it

The pre-registered `GROUNDED` rule counts a predicate as grounded if *any* small integer literal
in it matches an observed row, column or changed colour. `0` and `1` appear in essentially every
numpy predicate as indexing and background constants, so they can ground a predicate by accident.
Measured on `bp35`:

- control: literals `[0, 1]`, hits `rows=[1] colours=[0]` → GROUNDED **by accident**
- treatment: literals `[0, 63]`, hits `rows=[63] colours=[0]` → GROUNDED **because 63 is the row
  it read the progress bar from**

Both score GROUNDED under the pre-registered rule and only one means anything. The pre-registered
rule was **not changed** — swapping in a definition devised after seeing which arm it favours is
the instrument-tuning these disciplines exist to prevent. Instead `analyse.py` reports a labelled
post-hoc **sensitivity** (`SENSITIVITY_grounded_excluding_trivial_literals`) requiring a hit on a
literal `>= 2`, alongside the pre-registered number, never in place of it.

## Two confounds found by adversarially reviewing this harness against its own brief

Both are recorded in `analysis.json` under `known_confounds` so they travel with the numbers
rather than living only here.

**1. `GROUNDED` is partially determined by the treatment — the circularity class the brief
warns about.** `GROUNDED` means "the predicate names a literal that appears in the agent's
observed deltas". The treatment prompt *is* those deltas rendered as text. A model that copies a
row index out of its own prompt scores GROUNDED without having understood anything, and the
control — shown no deltas at all — can only score GROUNDED by coincidence. So the GROUNDED
contrast is **not** a clean read on goal quality and is not reported as one; it reports whether
the treatment gets the model to *reference* its observations.

The PRIMARY is unaffected: a constant-`False` predicate references nothing, so no prompt content
can mechanically produce or prevent a DECLINED. TROPE is unaffected for the same reason — a
whole-board uniformity claim names nothing observed by construction. MISSING is a parse outcome.
The circularity is confined to GROUNDED. It was **not** "fixed", because removing it would mean
changing a pre-registered outcome definition after seeing data.

**2. Arm order within a game is fixed, not randomised — and that confounds the timing measure.**
Cells run `A, B, C, AA` (and `gA, gB, gAA`) always in that order, so anything drifting with
position — llama.cpp prompt-cache warmth, GPU thermal state — is confounded with the arm label.
Shape is protected by the fixed seed, and the byte-identity result above is direct evidence that
order is not perturbing the sampled output. **Timing is not protected**, and the run shows it: on
`ar25` the control took 111.4s against 55.2/55.1s for the treatments, while on `bp35` the control
took 44.5s against 67.5/67.6s — the sign flips between games. Stage-2 `median_elapsed_s` is
therefore descriptive only and carries no p-value.

Stage-1 timing *is* load-bearing, and for a reason that survives this confound: 3s against 340s
is a 100x gap in the direction cache warmth cannot produce (warmth makes calls faster), and the
mechanism is visible in the artefact itself — three full 4096-token attempts, code block
truncated mid-definition. A future run should counterbalance arm order regardless.

## Honest limitations

- **The run was truncated by the session's real-time budget, not by the job list.** The stopping
  rule, every amendment to it, and what had been observed at each amendment are recorded in
  `out/stopping_rule.json`. Truncation removes a prefix of the alphabetically-fixed roster with
  every arm of both stages measured on the surviving games, so it cannot select for games where
  the treatment looks good — but the realised n is far below the pre-flight power table in
  `pre/power.json`, and a null at this n is close to uninformative.
- **One replicate per arm** means the A/A floor carries the whole burden of describing run-to-run
  noise, and that noise is large: the identical 365-character control prompt produced
  `TROPE` in 2.6s on one seed and `DECLINED` in 114.5s on another.
- **Stage 1 is a component measurement**, not an end-to-end rate. It answers "does evidence in
  the goal prompt change what the goal call writes", not "does the live agent induce better
  goals".
- **The win transition is not supplied** to either arm (`win_transition=None`). Whether showing
  the agent its own single self-produced positive example helps is a separate, untested question
  and is the obvious next experiment.
- **`GROUNDED` is deliberately easy to pass**: any small integer in the predicate can match a row,
  column or colour. That biases toward calling a predicate grounded, so the analysis separates
  `groundable_rate` (did the model write a region-naming predicate at all) from
  `grounded_given_groundable` (was the thing it named something it had actually seen).

## What this does not license

Neither default is changed. In particular, a treatment that raises the rate at which the goal
call returns nothing at all is **not** a fix for the decline defect, however the surviving cells'
shapes come out — and the 62-of-71 "the information was available" split remains unexplained by
this flag as currently wired.
