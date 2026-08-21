# Conductor self-improvement assessment — 2026-08-21

Read-only strategy assessment. No code was changed. Every number below was
re-measured in this session against the live checkout (commit `12fee38ba2`,
both worktrees in sync). Where this note corrects the briefing evidence, the
correction is marked **CORRECTION**.

The question under assessment: the loop produces many artifacts. Does it
compound — does it get better at producing true, useful results, and does it
correct itself without a human?

## Summary

The loop has one defect class, expressed three ways: **claims of completed
work are written by control flow, not derived from the work's artifact.**

- The milestone archiver stamps `result: "OK (conductor)"` on every task as a
  hardcoded literal (`scripts/research_conductor.py:3727`). It never checks
  that the task ran or that the deliverable exists.
- The QA-layer audit advances its coverage rotation state before it runs, and
  writes its report only after it finishes. It has not finished since
  2026-07-29. The rotation offset says 78 units covered. The real coverage
  since 2026-07-29 is zero.
- The activation-refusal retry loop re-archives the same milestone every two
  minutes. `research-complete.yaml` now holds 1,891 milestone entries for 50
  distinct milestones. About 97 percent of the ledger is duplicate spam.

The fix is one invariant, applied at three places, not a thirty-first rule:
**a completion claim must be a receipt the completed work itself wrote, and
the reader must verify the receipt.** The three top-ranked mechanisms below
are that invariant applied to the archiver, the audits, and the stall loop.

Ranked by expected value per unit of effort:

1. Guard-stall recovery: stop re-archiving on refusal; bounded replan with
   the lint report as planner input; hard cap, then park and escalate.
2. Truthful archival: derive `result` from the conductor log and deliverable
   existence; refuse duplicate milestone appends.
3. Run receipts for every audit invocation: the caller checks report
   freshness; rotation state advances only when the report lands.
4. Declared verdict class: replace the four drifting verdict token lists with
   one structured enum, cross-checked against structural fields.
5. Loop instrument panel: a small per-milestone metrics artifact about the
   loop itself, fed to the planner.

## Corrections to the briefing evidence

The three briefing findings are real. Several numbers and one framing need
correction.

**CORRECTION 1 — the ledger gap is larger and has a named mechanism.** The
briefing said 504 unique (id, deliverable) pairs, 57 missing, 51 recorded OK.
Re-measured: 519 unique pairs, 57 missing, and **all 57** are recorded
`OK (conductor)`. Two spot-checked deliverables
(`results/experiment_5710_fr11_isolated_act_on_advice_canary.json`,
`results/experiment_5792_arc_calibration_only_selector.json`) have no entry in
`git log --all --follow` — never created, confirming false-success logging.
The mechanism is not subtle: `_archive_current_milestone`
(`scripts/research_conductor.py:3690`) writes the literal string
`"OK (conductor)"` for every task in the roadmap, including tasks that were
3-fail-skipped or gate-blocked. `pick_next_task` (line 2458) counts an
exhausted task as done, so a milestone can close with unrun tasks, and the
archiver then certifies them.

The bigger uncounted problem: the ledger holds **1,891 milestone entries for
50 distinct milestones** (17,929 task rows for 519 unique pairs). Milestone
2026.07.510 was archived 684 times, .527 641 times, .557 109 times. Cause:
on an activation refusal, the loop returns, the roadmap file stays in place,
and the next 2-minute iteration runs `_archive_current_milestone` again. The
function appends unconditionally. The planner reads this file as its failure
record.

**CORRECTION 2 — the 22 percent flag rate is overstated and mislabeled.**
Nine of 42 artifacts touched since the restart carry `flagged_adversarial`,
but four of the nine are OLD artifacts (exp833, exp1736, exp3361, exp3377,
exp3392) with fresh mtimes from restoration and checkpoint commits — for
example commit `20b2aefaa8` ("Restore 5 dropped fabrication-gate
determinations"). They are not new production. The new-production flag rate
is 4 of about 38 (~11 percent). Of those four:

| Artifact | Flag | What it actually is |
|---|---|---|
| exp6487 | NONTERMINAL_DECLARED_ARTIFACT | honest negative; verdict prefix `disqualified:` is not in the terminal whitelist |
| exp6490 | NONTERMINAL_DECLARED_ARTIFACT | same — a control caught a shortcut, which is the discipline working |
| exp6484 | DURATION_TOO_SHORT | 0.32s with vestigial compute markers — a substrate-declaration miss |
| exp6491 | DURATION_TOO_SHORT | 54.5s against the 60s floor — the exp5178 false-positive class again |

So the recent flag signal is mostly taxonomy friction (unregistered verdict
vocabulary, duration-floor near-misses), not a fabrication surge. The
`disqualified:` prefix is the sixth recorded incident of the terminal-prefix
whitelist being narrower than honest practice. That matters for mechanism 4.

**CORRECTION 3 — the QA-layer audit is worse than silently dead.** Confirmed:
`ops/qa_layer_authenticity_audit_report.md` is dated Jul 29 in both
checkouts while the sibling verifier report regenerated today. But the audit
is not failing to start. `ops/.qa_layer_audit_rotation.json` has mtime today
(12:00) and offset 78. The script writes rotation state at
`scripts/qa_layer_authenticity_audit.py:1374`, before the LLM loop, and
writes the report only at line 1476, after ALL units complete. The caller
(`scripts/research_conductor.py:4800`) runs it with `timeout=900` and
`check=False`; a `TimeoutExpired` is caught and logged as a warning. With
`--limit 20` LLM units, 900 seconds is structurally too short since the
2026-07-29 scope extension grew the per-unit prompts. Net effect every
milestone close: the audit starts, burns rotation coverage for 20 units,
dies at the timeout, writes nothing. The coverage ledger advances while zero
coverage happens — the same defect class as the phantom OKs, inside the tool
built to catch that defect class.

One more fact that shapes the fix: journal retention on this host is a few
hours (the earliest retained conductor entry today is 15:29). A
`logger.warning` is not a durable failure record here. Receipts must be
files.

## Mechanism 1 — Guard-stall recovery (bounded replan, then park)

**What it changes.** Today, when `_activate_next_roadmap` is refused by the
exclusion-manifest lint, the loop retries an identical activation every two
minutes, forever, and re-archives the milestone each time. Measured cost:
5,125 `Activation REFUSED` lines in `ops/conductor-log.md`. At the 2-minute
cadence that is roughly 170 hours of dead loop time across the project.
Worst stalls: .511 (1,998 refusals, ~66 hours), .528 (942), .505 (907). Both
briefing stalls reproduce (.557: 27 refusals; .558: 108, ~3.6 hours).

Proposed behavior on refusal:

1. Do not re-archive. Archive at most once per milestone (mechanism 2 makes
   this a dedup check in the archiver itself).
2. On the first refusal, rename `research-roadmap-next.yaml` to a quarantine
   name and re-run the planner ONCE with the lint's violation report embedded
   verbatim in the prompt: the violation class, the matched prior experiment
   ids, and the exact `prior_failures:` / `operator_override:` structure the
   lint accepts. The lint already computes all of this
   (`scripts/exclusion_manifest_lint.py`); it is currently thrown away.
3. Re-lint the replanned roadmap. The guard is unchanged and still decides.
4. Cap at 2 replans. On the cap, park the milestone: write a dated
   `OPERATOR-ATTENTION` entry to `ops/known-issues.md` and a `BLOCK` line to
   the conductor log, then stop retrying that activation (idle or work the
   backlog). An indefinite retry and a silent halt are both worse than a
   visible parked state.

**Why this is safe.** The team-lead's worry — auto-repair defeats the guard —
does not apply, because the repair path is a replan that goes back through
the same guard. The planner is allowed to fix its own roadmap; it is not
allowed to touch the lint. Precedent exists: the lint already auto-downgrades
when task prose addresses the prior (`_prose_addresses_prior`, 2026-07-03).
This mechanism feeds the planner the information it needs to write the
structured block the guard wants, which is what the human did by hand twice
this week.

**Cost.** Roughly one day. The delicate part is the once-per-milestone
archive latch and the replan-count persistence across conductor restarts.

**How you know it worked.** Refusal streaks per milestone drop from hundreds
to under 5. Zero new duplicate milestone entries. Median
milestone-close-to-activation time drops from hours to minutes.

**Abandon if.** Two consecutive milestones hit the replan cap. That means
the planner cannot use the lint report, which is a planner-prompt or model
problem — raise the cap never; escalate instead.

**Feedback path (why this compounds).** The guard's output becomes the
planner's input. Today the guard only vetoes; the planner never learns why.
With the violation report in the replan prompt, and with the same report
summarized in the NEXT milestone's planning prompt ("last milestone was
refused N times for these classes"), the planner's error rate on
`prior_failures:` blocks should fall milestone over milestone. That is a
measurable learning curve: refusals per planned roadmap.

## Mechanism 2 — Truthful archival (receipts, not assertions)

**What it changes.** `_archive_current_milestone` stops writing a literal.
For each task it derives `result` from evidence at archive time:

- conductor log says OK and the deliverable file exists → `OK`
- deliverable exists but is stamped `flagged_adversarial` → `FLAGGED`
- failure count exhausted, no deliverable → `SKIPPED (3-fail)`
- gate-blocked → `GATE_BLOCKED`
- deliverable missing despite a log OK → `OK_NO_DELIVERABLE` (this row is
  the alarm the briefing asked for — it fires at archive time, not a month
  later)

And one structural rule: refuse to append a milestone id that is already
archived (update-in-place or skip, log the collision).

This check answers the briefing's question 3 directly: the moment a phantom
OK would have been logged is `_archive_current_milestone`, and a one-line
`Path(deliverable).exists()` there would have caught all 57.

**Cost.** Two to four hours. The signals already exist — `pick_next_task`
already parses the log for exactly these states; the archiver just does not
consult them.

Separately, with operator authorization: repair the historical file. Dedup
the 1,841 duplicate entries and re-stamp the 57 phantoms. Keep the original
as a compressed sidecar (never-prune applies to substantive records; this is
machine-generated corruption, but the operator should make that call, not an
agent). Cost: one script, one review.

**How you know it worked.** `research-complete.yaml` invariants hold on
every commit: one entry per milestone id; every `OK` row has an existing
deliverable. A ten-line lint asserts both; it can ride the existing
pre-commit hook set.

**Abandon if.** Truthful stamping causes the planner to propose rerun storms
of newly-revealed `SKIPPED` tasks. The doomed-rerun ledger should absorb
this; if it does not, gate reruns of re-stamped tasks behind
`prior_failures:` review rather than reverting to false OKs.

**Feedback path.** The ledger is planner input. The Failed-Experiment Rerun
Discipline and every `requires:` chain assume the failure record is true.
exp5789 lost a month to an upstream recorded OK that never ran. A true
ledger makes the next roadmap's gating decisions correct by construction —
this is the cheapest compounding fix on the list.

## Mechanism 3 — Run receipts for every audit and guard invocation

**What it changes.** The general pattern the briefing asked for, in three
rules:

1. **The audited program's last act is a dated receipt.** Each audit writes a
   small run-manifest (date, exit status, units reviewed, report path) as the
   final step — or simply its report, dated. The QA-layer audit already dates
   its report; the problem is ordering and the caller.
2. **The caller verifies the receipt, not the exit code.** After
   `subprocess.run`, the conductor checks that the report/manifest mtime is
   newer than the call start. A stale receipt writes a `BLOCK`-severity line
   to `ops/conductor-log.md` — a tracked file the operator reads — not a
   `logger.warning` that the journal deletes within hours. `check=False`
   plus `except Exception` can stay; the receipt check replaces them as the
   truth signal.
3. **Consumption state moves only with the receipt.** Rotation offsets, "last
   audited" markers, and any coverage accounting are written after the
   report, or reverted when the receipt check fails. The QA-layer audit
   currently does the opposite (state at line 1374, report at 1476).

Plus the immediate unblocking fix for the QA-layer audit specifically: pass
a wall-clock budget into the script (finish early, write the report with
whatever units completed) instead of killing it from outside at 900s. A
deadline the program knows about produces a partial report; a timeout the
caller imposes produces nothing. Alternatively size `--limit` to measured
per-unit time. Either is under an hour.

**Cost.** Half a day for the receipt check across the five audit invocations
in `research_step`, plus the ordering fix in the QA-layer script.

**How you know it worked.** All audit reports in `ops/` carry the current
milestone-close date after every close. A deliberately-broken audit (rename
the script) produces a visible `BLOCK` line within one close.

**Abandon if.** Nothing plausible — this is the project's own stated
doctrine ("a guard that is trusted and silent is worse than no guard")
applied to the audit callers. If the receipt check itself starts
false-positiving on slow-but-successful runs, widen the freshness window,
not the principle.

**Feedback path.** The QA-layer audit's `## MISSED INPUT` section is the
loop's strongest designed compounding mechanism: each line is a widening
task with its regression test pre-specified. That channel has been dead for
23 days. Receipts do not add a new loop; they restore the best existing one.

## Mechanism 4 — Declared verdict class, cross-checked

**What it changes.** Verdict semantics currently live in four substring
token lists (`_PARTIAL_TOKENS`, `_BLOCKED_TOKENS`, `_FAILED_TOKENS`, and the
terminal-prefix whitelist), patched at least six times, and still wrong this
week (`disqualified:` — an honest negative — drew a critical flag). This is
the "pattern list narrower than its concept" bug class, applied to verdicts.

Replace inference with declaration plus cross-check:

- Every artifact declares `verdict_class`, a closed enum:
  `positive | circular_positive | null | blocked | disqualified | partial`.
  The free-text `honest_verdict` stays for humans.
- `adversarial_verify.py` cross-checks the declared class against structural
  fields it already reads. `verifier_is_oracle: true` forbids `positive`
  (must be `circular_positive`). `flip_count == 0` forbids an informative
  `null` (FALSE_NEGATIVE_RISK already implements this). A failed acceptance
  gate forbids `positive`.
- Capstones and headline aggregation consume only the enum, never the
  string.

This is what the exp6478 case needs. That artifact was honest: it declared
`verifier_is_oracle: true` and its verdict even says "exact backend remains
the oracle." No detector failed. The defect is that `complete_positive`
connotes a research win to every downstream reader that matches on the
prefix, and nothing forces the declared circularity into the claim. A
cross-check that renames the class to `circular_positive` closes that gap
without any new pattern list, because the enum is closed and the
cross-checks are structural, not textual.

**Cost.** Two to three days: the enum, the cross-checks, the planner prompt
update, and a compatibility shim that maps legacy prefixes to classes so the
reconciler's token lists can retire instead of growing.

**How you know it worked.** The four token lists stop receiving patches
(their patch cadence has been roughly monthly). Zero
NONTERMINAL_DECLARED_ARTIFACT flags on honest terminal verdicts. Capstone
claims trace to enum values.

**Abandon if.** More than about 10 percent of artifacts need a class outside
the enum. That would mean verdicts genuinely need open vocabulary; keep the
enum for aggregation eligibility only.

**Feedback path.** Weaker than 1–3. This mechanism raises the floor on what
a "result" is, which protects the record, but it improves the next milestone
only through cleaner capstone claims.

## Mechanism 5 — Loop instrument panel

**What it changes.** A tiny derived artifact per milestone close: refusals
this milestone, replans used, flag rate split by cause (fabrication vs
taxonomy), audit receipt freshness, phantom-OK count (should be
structurally zero after mechanism 2), ledger invariant status. Fed into the
planner and retrospective prompts.

**Cost.** Half a day, after 1–3 exist (it mostly reads their receipts).

**How you know it worked.** The retrospective starts citing loop-health
numbers instead of only experiment outcomes, and the numbers trend: refusals
down, receipt freshness at 100 percent.

**Abandon if.** The panel becomes a metric the planner games (for example,
proposing trivially-activatable roadmaps to keep refusals at zero). Watch
for headline-progress stagnation while loop metrics improve.

## On consolidation instead of a thirty-first rule

The briefing asked whether some rules need consolidating. Yes, narrowly:

- The four verdict token lists collapse into mechanism 4's enum. That
  retires a monthly patch treadmill.
- The Verifier Authenticity, QA-Layer, Test-Run Record Integrity, and
  Adversarial Artifact disciplines already share one thesis, stated in
  CLAUDE.md itself: a trusted-and-silent guard is worse than no guard.
  Mechanism 3 is that thesis as a single enforceable invariant (receipts,
  verified by the caller, owning their consumption state). Writing the
  invariant once — as a spec requirement with a lint, not as prose — would
  let several of the per-incident prose sections point at it rather than
  each restating it.
- Do not consolidate the exclusion-manifest lint or the fabrication gate.
  They fire correctly. The stall problem is the caller's retry behavior, not
  the guards.

## What actually compounds (question 5, answered directly)

Ordered by strength of the feedback path:

1. **Mechanism 2** (true ledger): planner gating decisions become correct on
   the next plan. Shortest path from fix to better milestone.
2. **Mechanism 1** (lint report into replan): the planner is corrected at
   the moment of error, with the exact structure it failed to produce.
   Measurable as refusals-per-roadmap over time.
3. **Mechanism 3** (receipts): restores the MISSED-INPUT widening loop,
   which is the only mechanism in the project that makes the GUARDS better
   over time rather than the experiments.
4. Mechanisms 4 and 5 protect the record and measure the loop; they compound
   indirectly.

## Measurements that would sharpen this assessment

- Per-unit wall-clock of the QA-layer audit under the current model config,
  to size the budget correctly (one `--file` run measures it).
- Refusals-per-roadmap trend for the last 20 milestones, as the baseline for
  mechanism 1's success metric.
- How many of the 57 phantom-OK tasks are load-bearing in later `requires:`
  chains. exp5789 is one confirmed month-long casualty; the count bounds the
  value of backfilling history versus only fixing the archiver forward.
- Flag-rate split (fabrication vs taxonomy) over the last 100 artifacts, to
  confirm the ~11 percent recent rate is mostly taxonomy friction, as the
  four-artifact sample suggests.
