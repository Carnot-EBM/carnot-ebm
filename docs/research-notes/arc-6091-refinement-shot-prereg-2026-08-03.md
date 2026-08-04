# exp6091 refinement shot — pre-registration, recorded before any number existed

**Written 2026-08-03T23:41:52Z.** At that instant
`results/exp6091_refine_engine_visible_shard.jsonl` did not exist (verified, `ls` returned
`No such file or directory`) and the measurement process was still building its offline windows,
having launched no `llama-server` and generated nothing. Everything below is therefore a
commitment, not a description.

## Why this record exists at all

`ops/known-issues.md` (2026-08-03 ruling) closes the ARC induction line and adds one instruction
to anyone who ever revisits it:

> Any future reconsideration must settle which metric carries the decision BEFORE the shot, not
> after.

That instruction exists because of a real design property of this A/B, also recorded there: all
11 gradeable games carry **exactly one** gradeable acceptance row each, so the primary metric is a
single Bernoulli draw per game. `min reachable p = 0.00098` therefore requires **11 of 11**
concordant games. A merely-good effect of 8 of 11 lands at **p = 0.23**. That is a design in which
the honest reading depends entirely on which metric was named first — so it is named here.

## Authorization — this is not a unilateral reopening of a closed axis

The same ruling says: *"Do NOT propose a follow-up on the refinement / single-shot-GGUF-induction
axis … not a re-run of the engine-visible A/B, however tempting the fixed instrument makes it,"*
and, in the next breath, that the fixed capability *"stays in the tree, tested and default-OFF, for
whoever later has a healthy host and an operator decision to spend a slot on it — it is banked, not
queued."*

Both conditions of that second sentence now hold, which is the only reason this ran:

- **A healthy host.** The operator identified the conductor and its user timers as the process that
  was reaping every `llama-server`, and stopped them. Verified before launch:
  `carnot-conductor.service`, `carnot-outer-loop.timer`, `carnot-orphan-cleanup.timer` all
  `inactive`; no `research_conductor` / `codex exec` / `pytest` children; both GPUs at 2 MiB.
- **An operator decision to spend the slot**, issued as this task.

The prior attempt was **BLOCKED, not null** — 2 attempted cells, 0 vouched, every server killed
within seconds to minutes. This run completes the pre-committed test that could not be taken. It
does not relitigate the closure, and the closure does not rest on it: the line was closed on an
independent measured negative (0 of 296 clean engine-units with `n_changing>=3` reaching held-out
`change_accuracy >= 0.5`), which stands regardless of what this shot returns.

## The metric that carries the decision

**PRIMARY, and the one the stopping rule keys on: held-out `change_accuracy` on the gradeable
acceptance block, clustered at the GAME level, paired two-sided exact sign test.** This is the
quantity the operator's stopping rule names. It decides, and it decides alone.

**SECONDARY, pre-registered as secondary and explicitly NOT decision-carrying:**
`change_fidelity` (continuous, symmetric cell-level union fidelity) and `cell_recall`, on the same
rows. Their job is to make a null *interpretable* — to distinguish "the treatment moved nothing"
from "the primary metric was too coarse to move at these window sizes." A secondary win alongside a
primary null does **not** overturn the stopping rule; it is reported as a caveat, and any decision
to act on it belongs to the operator, not to this run.

## Stated before results

| Quantity | Committed value |
|---|---|
| Roster | 13 games (`tu93 tr87 sp80 sb26 re86 g50t r11l cd82 ar25 lp85 sk48 vc33 ft09`) |
| Leaving the denominator BY NAME | **r11l, vc33** — no gradeable acceptance row; reported, never silently dropped |
| Gradeable games | 11 |
| Min reachable two-sided p (11 of 11 concordant) | **0.00098** |
| p at 8 of 11 | 0.2266 |
| p at 6 of 11 | 1.0 |
| Significance threshold | p < 0.05 on the PRIMARY |
| Arms | `single_shot` · `refine_control` (engine invisible) · `refine_treatment` (engine visible) |
| Baseline that matters | `single_shot` — refinement must beat NOT refining, not merely beat zero |
| Identity control on the acceptance block | expected **VACUOUS** (`n_noop = 0` ⇒ 0.0 by construction); also run on the full window where it can move |
| Oracle control | must reach `change_accuracy = 1.0` on every gradeable block, else that game is excluded as unfalsifiable |

## Addendum, 2026-08-03T23:45:30Z — still pre-result (verified: shard absent)

Ties are DROPPED by a sign test, so the effective n is `n_discordant`, not 11. The
known-issues example ("8 of 11 lands at p=0.23") assumes 8 wins **and 3 losses**; 8 wins with 3
*ties* is n=8 and p=0.0078. The two readings differ by a factor of 29, so the tie count has to be
read off the artifact, never assumed.

The consequence that matters for interpreting a null:

| n_discordant | min reachable two-sided p |
|---|---|
| 11 | 0.0010 |
| 8 | 0.0078 |
| 6 | 0.0313 |
| **5** | **0.0625 — p < 0.05 is already UNREACHABLE** |
| 3 | 0.2500 |
| 1 | 1.0000 |
| 0 | no test exists |

**So if 5 or fewer games are discordant, this design cannot reject the null no matter how
one-sided the result is.** A `p = 1.0` from 2 discordant games is not evidence of no effect; it is
an absent test. I will therefore quote `min_reachable_p_at_this_n_discordant` alongside the
observed p in every comparison, and will say plainly whether the shot was capable of rejecting
before saying whether it did.

This does not change which metric decides — `change_accuracy` still carries the stopping rule. It
changes what a null is allowed to be *called*: a null that could not have been anything else is
reported as underpowered, not as a refutation.

## Outcomes that are NOT a null, and must not be reported as one

1. **The treatment never reached the generator.** If no generator prompt carried the engine, the
   run measured the control twice. The analyser computes this from a wrapper on `generate` — the
   deepest callee before transport — and emits
   `blocked_treatment_never_delivered_to_generator_not_a_null`, with `line_closes: false`.
2. **The substrate could not be vouched for.** Cells whose serving PID changed mid-cell are
   recorded and excluded, never silently dropped.
3. **No cell produced a round-0 engine.** Missing is not zero; such cells are counted and named.

A clean null **is** the expected outcome and reporting it is a success. The stopping rule governs:
if this returns null, the line stays closed and **no follow-up on this axis is proposed.**

## Delivery was proven before the shot, not assumed

`results/experiment_6091_delivery_probe.json` (adversarial-verify clean) wraps
`LocalGGUFProposer.generate` and reads the caller chain off the interpreter stack, confirming
`refactor:6702 -> _gen_to_file:6469 -> generate` on the live path. Flag ON delivers 9 of the
engine's substantive source lines; flag OFF delivers 2, and both of those are the prompt quoting
its own output template — an independent reproduction of the original defect's characterisation
(0 of 454 substantive lines, 13 of 13 games). A mutation arm neutralises the source resolver and
the ON assertion goes red, so the green result is not vacuous.

---

## POST-HOC ADDENDUM, 2026-08-03 ~00:20Z — corrections applied after the shot began

**This section is written AFTER cells started landing and is marked as such. Nothing above it has
been edited; every claim above stands as pre-registered, including the ones this addendum
corrects.** Four defects were found by adversarial review and independently verified here before
being acted on. Three are in the analyser and are fixed in
`scripts/experiments/outer_loop_arc_refine_engine_visible_analyse_20260803.py`; one is a factual
error in the prose above.

**A-1. The identity-control rationale above is FALSE IN BOTH CLAUSES.** The pre-registration says
the identity control is vacuous on the acceptance block because `n_noop = 0` there, and that it is
therefore ALSO run on the full window, "where no-op rows exist and the score can move." Measured
on the landed cells: `n_noop` is **0 on the FULL window as well** (tu93 11 changing / 0 no-op,
tr87 11/0, sp80 3/0, sb26 likewise), and identity's full-window `change_accuracy` is **0.0**
everywhere. The mechanism is not no-ops at all — `change_accuracy = n_changes_correct /
n_changing`, so an identity engine scores exactly 0 on ANY block with `n_changing > 0`,
irrespective of how many no-op rows sit beside it. **The identity control is vacuous everywhere in
this design, not only on the acceptance block, and the stated remedy does not remedy it.** The
analyser now computes and reports `VACUOUS` for the full-window control too. A non-vacuous identity
control would need a block containing no-op rows AND would have to be scored on `accuracy`, not
`change_accuracy`.

**A-2. The pre-registered PRIMARY is not like-for-like.** `cell_value` reads `best_<metric>` — the
MAX over the R refactor rounds, maximised on the very held-out block that grades — for both
refinement arms, while reading `single_shot`'s ONE grade. Per-cell `change_accuracy` is BINARY at
these window sizes (exactly one gradeable acceptance row per game), so a refinement arm that
changes nothing still scores `1-(1-p)^R` against single-shot's `p` — 0.36 vs 0.20 at p=0.2, R=2 —
**from the extra draw alone**. `treatment_vs_control` is symmetric and unaffected, and remains the
only clean contrast in the original set. Fix: the shard already stores each arm's `final` round, so
the analyser now ALSO computes the symmetric final-round contrast and a POSITIVE verdict requires
BOTH to point the same way. This can only make a win harder to claim.

**A-3. `line_closes` had no power term, contradicting this document's own prose.** It was
`(not beat) and treatment_administered`, which fires identically at 1 discordant game (min
reachable p = 1.0) and at 11 (0.00098) — while the section above commits, in words, to reporting a
null that could not have been anything else as underpowered rather than as a refutation. That
commitment is now machine-readable: `line_closes` additionally requires
`min_reachable_p_at_this_n_discordant < 0.05` and requires that at least one cell produced a
round-0 engine.

**A-4. Outcome #3 above ("No cell produced a round-0 engine") had no verdict of its own.** It was
pre-registered as not-a-null but the analyser had no branch for it, so such a run emitted
`blocked_treatment_never_delivered_to_generator_not_a_null` — naming the DELIVERY instrument, which
is independently proven working, as the blocker. There is now a distinct
`blocked_round0_induction_produced_no_engine_on_any_cell_not_a_null` keyed on
`round0_engine_loaded`, carrying the induce messages so the operator investigates the token budget
rather than the witness.

**Also corrected, not in the analyser: the LLAMA_SERVER isolation comment in the AB script
describes protection that is not in effect.** It states the binary "is COPIED to a privately-named
path and run from there"; there is no copy step — `LLAMA_SERVER` reads an env var and defaults to
the original build path, `CARNOT_6091_LLAMA_SERVER` is unset in the serving process's environment,
and `/proc/<pid>/exe` resolves to `~/.cache/llama.cpp-master/build/bin/llama-server`. Benign now
that the reaper is stopped, but the prose overstates the isolation.

**And one guard that was inert:** the "stratification by changed-cell count (failure mode 7)" block
tallied rows and games per stratum but never recomputed the primary on the `>=2`-cell stratum, so a
headline carried entirely by 1-changed-cell rows would not have been caught by anything. It now
recomputes. Harmless on this data — the gradeable acceptance rows measured so far have
changed-cell counts of 20, 11, 34 and up, with no 1-cell rows — but the guard was decorative.

All six corrections were self-tested on synthetic shards (all-dead, clean-win,
best-of-R-artifact-with-flat-final, 1-discordant-null, 11-discordant-null, all-1-cell-rows); each
classifies as intended, and the 1-discordant case now returns `line_closes: false` where it
previously returned `true`.
