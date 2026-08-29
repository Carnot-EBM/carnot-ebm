# Gate-cascade settlement — 2026-08-29 conductor stall diagnosis

Spec: REQ-CONDUCTOR-GATECASCADE-1 (`openspec/capabilities/research-harnesses/spec.md`).
Tests: `tests/python/test_gate_cascade_settlement.py`.

## The symptom

The last 10 conductor outcomes on 2026-08-29 were all GATE_BLOCK, cycling
every ~2 minutes (20:47Z to 21:03Z). The milestone advanced (.588 to .589),
so the process was not wedged.

## Diagnosis A — the .589 requires-chain collapse (CONFIRMED, with the root)

The chain: exp6755 -> exp6756 -> exp6757 -> exp6758 -> exp6759 -> exp6760.

- exp6755 completed honestly at 20:45Z. Its artifact records
  `environment_grammar_targetable_rows = 21` and status `complete`
  (`results/experiment_6755_lossless_gguf_output_reparse.json`).
- exp6756's gate demands `environment_grammar_targetable_rows >= 24`.
  The measured 21 falls short. The gate failure is LEGITIMATE — the
  planner set a falsifiable bar and reality missed it. That part is the
  system working.
- The waste: the value is frozen in a finished artifact, so no retry can
  ever pass, yet the conductor re-evaluated the gate 3 times (20:47,
  20:49, 20:51), spent exp6756's full retry budget on it, retired
  exp6756, and then let each downstream link burn 3 GATE_BLOCK rows
  before the next link could even be skipped. Total: ~18 minutes of
  2-minute cycling, 16 log rows, and a delayed start for the 7 runnable
  non-chain tasks (exp6761..exp6767).
- The same shape ran repeatedly on 2026-08-28 (three windows of paired
  GATE_BLOCK cycling after the "Cold/Bounded audit" tasks retired on
  wall-clock caps). This is chronic, not new today.

Why `scripts/exclusion_manifest_lint.py:REQUIRES_RETIRED_EXP` did not
stop it: that class checks a roadmap's requires-chain against
`ops/exclusion_manifest.yaml` — experiments retired in PRIOR milestones,
known at activation time. The .589 chain was healthy at activation;
exp6756 retired at RUNTIME when exp6755 measured 21. No activation-time
lint can know that. The lint is not defective for this case; the fix
belongs in the conductor's runtime gate handling.

## Diagnosis B — GPU contention from the A/B (REFUTED as framed)

Measured base rates from `ops/conductor-log.md` (108 logged days):

| day | OK | FAIL | wall-clock caps | bootstrap fails |
|---|---|---|---|---|
| 08-24 | 17 | 9 | 4 | 5 |
| 08-25 | 33 | 5 | 2 | 0 |
| 08-26 | 16 | 10 | 7 | 3 |
| 08-27 | 16 | 14 | 5 | 5 |
| 08-28 | 9 | 10 | 9 | 1 |
| 08-29 | 18 | 6-7 | 3-4 | 3 |

- Today's cap count (3-4) is BELOW yesterday's 9 and below the trailing
  week's mean. Yesterday's 9 caps happened with NO A/B running.
- exp6754 ("V588 branch disposition and PRD gap reconciliation") needs
  no GPU and still capped/bootstrapped 3 times. GPU contention cannot
  explain it.
- Caps recover on retry sometimes (09:36 FAIL -> 09:55 OK; 12:25 FAIL ->
  12:44 OK), so they are codex-runtime variance, chronic since long
  before today (179 occurrences all-time, mean 1.24/day, spread across
  tasks — the worst repeat offender has 3).

Conclusion: today is within normal variance. The A/B on GPU 1 is not
implicated. RECOMMENDATION: do NOT pause the 6-hour measurement
(pid 2941040, `results/holdout_equalized_ab_selfparse_20260829`).

What DID reduce today's afternoon throughput: the .588 tail tasks
(exp6752/6753/6754) burned ~6 consecutive 80-minute codex slots on caps
and bootstrap failures, exactly as on 08-28 — chronic, not acute.

## The mechanical fix (Error Lifecycle step 6)

Three rules, all live in the conductor's real code path:

1. `conductor_gates.evaluate_gates` prefixes the failure summary with
   `gate-unsat(final): ` when any failed gate read a FINISHED upstream
   artifact (`_upstream_is_final`: terminal status or terminal-prefix
   honest_verdict). Missing/unreadable/running/blocked/failed upstream
   artifacts stay transient. The summary flows unchanged through the
   existing `log_step(task["title"], "GATE_BLOCK", gate_check.summary)`
   call site, so no new call-site wiring was needed.
2. `pick_next_task` retires a task on the FIRST GATE_BLOCK row whose
   details start with a terminal marker (`gate-unsat(final):` or
   `Pre-emptive skip: upstream retired`). Unmarked rows keep the
   3-strike budget.
3. `pick_next_task` closes the retired set transitively over `gated_on`
   edges, so a dead chain settles in one iteration.

Post-fix cost of this incident's shape: 1 wasted iteration and ~5 log
rows, versus 12+ iterations and 16 rows. The retirement outcome itself
is unchanged — the chain was structurally dead per the plan and stays
dead; only the burn and the noise go away.

## Proofs

- 16 tests at the real entry points (`evaluate_gates`,
  `pick_next_task`), including the incident's exact values (21 vs >= 24,
  status `complete`) and a not-fire proof on the passing threshold.
- Read-only run of the NEW `evaluate_gates` over the LIVE .589 roadmap +
  live `results/`: the marker fires on exp6756 only; every other failing
  gate in the milestone stays transient.
- 11 mutations (each rule deleted or inverted): all RED; restores
  byte-identical by `cmp`; final suite GREEN; zero decorative rules.
- Found at baseline, fixed in passing:
  `test_pick_next_task_gate_block.py::test_pick_next_task_source_includes_gate_block_in_failure_set`
  had been RED since the `DOOMED_RERUN_BLOCK` tuple extension (April) —
  its source-substring needle matched the pre-extension tuple. Needle
  updated to the live tuple.

## Considered and NOT changed (with reasons)

- **Wall-clock-cap FAILs still consume a retry.** Removing that would
  let a task that caps 3 times in a row (observed twice on 08-28) burn
  80 minutes per extra retry, unbounded. The measured recovery rate
  after a cap does not justify the certain wall-time cost. If the cap
  policy is revisited, the better lever is honoring
  `estimated_wall_time_min` (exp6757 declares 240 min against a fixed
  ~80-min cap) — flagged as PLAUSIBLE follow-up, not built.
- **No planning-time lint on chain depth.** A 4-deep single-parent chain
  is fragile, but a hard lint would fire on nearly every legitimate
  roadmap. Per the Error Lifecycle: a check that cries wolf trains
  people to bypass it.

## Deployment note

The conductor runs as a long-lived systemd `--loop` process from the
main checkout. This fix lands via the worktree branch
`worktree-agent-aff05cc9b49c41da6`; it takes effect after merge to the
conductor's checkout and the process's next natural restart.
