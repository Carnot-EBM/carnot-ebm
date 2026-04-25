# Failed-experiment rerun enforcement: ledger + conductor pre-launch check

**Status:** Draft change proposal.
**Origin:** User directive 2026-04-25 in response to the .60–.65 retros'
  consistent finding that the slow-5 carryover (Exps 786/527/491/627/603)
  has run unguarded for six consecutive milestones. The pattern is the
  same ~224 min/milestone of wall-time waste, with the manifest patch
  recommended in five separate retros and applied in zero of them.
**Target milestone:** 2026.04.67 — earliest practical milestone after
  the rule lands in CLAUDE.md (commit pending) and the planner has at
  least one cycle to internalise it.
**Priority:** High. Six consecutive milestone-level retros have
  identified this as the single largest wall-time bottleneck. The
  CLAUDE.md rule binds the planner immediately at the policy layer;
  this proposal binds the conductor at the enforcement layer so the
  rule is not waivable by future-Claude attention drift.
**Depends on:** the Failed-Experiment Rerun Discipline rule in CLAUDE.md
  (pending commit) — defines the structure this enforcement implements.

## Summary

The CLAUDE.md rule says: do not re-run a failed experiment in a new
milestone unless the new task explicitly addresses the suspected root
cause. Without mechanical enforcement, the rule is a hope. This
proposal:

1. Adds a `FailureLedger` module that reads
   `research-complete.yaml` + `results/operational_retro_*.json` and
   produces a queryable record: for each prior experiment ID, what
   was its terminal verdict, was it a failure, what was its scope.
2. Adds a `validate_prior_failures` step to the conductor's
   pre-launch sequence: before spawning Sonnet for a task whose
   scope overlaps a prior failure, the task's YAML must include a
   `prior_failures:` field that satisfies the four-part discipline
   from CLAUDE.md.
3. Adds a `retire_experiment_id` action that promotes a
   `retire_if_same_verdict: true` task to the permanent exclusion
   manifest (`ops/exclusion_manifest.yaml`) when the rerun
   reproduces the same verdict.
4. Seeds the initial ledger with the known-doomed carryovers
   (Exps 786, 527, 491, 627, 603, plus 753 / 776 / 794 / 804 /
   827 / 839 / 851 if iCE40 PnR continues to fail).

## What this proposal IS NOT

- Not a substitute for the planner respecting the CLAUDE.md rule at
  design time. Mechanical enforcement is the safety net, not the
  primary discipline.
- Not an excuse to stop iterating on hard problems. A previously-
  failed experiment that genuinely has a new approach (different
  technique, different upstream prerequisite resolved, different
  parameters, different gate) is *encouraged* to re-run. The rule
  is "named root cause + named change," not "never iterate."
- Not a replacement for the existing exclusion manifest. The
  exclusion manifest is the *permanent* retirement list; this
  proposal adds the *conditional* failure ledger that feeds into
  it when retirement is triggered.

## Proposed experiments

### Exp A — `FailureLedger` module + pre-launch validator

**Deliverable:**
`scripts/failure_ledger.py` (new module) +
edits to `scripts/research_conductor.py` (pre-launch validator) +
`tests/python/test_failure_ledger.py` (unit tests, no LLM) +
`results/experiment_<N>_failure_ledger_primitive.json`.

**What it does:**

1. `class FailureLedger`:
   - `load_from_artifacts(repo_root)` — reads
     `research-complete.yaml` + `results/operational_retro_*.json`
     + `ops/changelog.md` to construct an `{experiment_id ->
     LedgerEntry}` map.
   - `is_doomed_rerun(task: dict) -> tuple[bool, str]` — given a
     YAML task, returns `(blocked, reason)`. Blocked when the task
     scope matches a prior failure AND the task lacks an adequate
     `prior_failures:` field.
   - `validate_prior_failures(task: dict) -> ValidationResult` —
     checks the four-part discipline from CLAUDE.md:
     names-prior-failure, names-root-cause, names-what-is-different,
     states-falsifiable-acceptance-gate.
   - `record_outcome(experiment_id, verdict, retired)` — when a
     task with `retire_if_same_verdict: true` reproduces the same
     verdict, this method promotes it to the exclusion manifest.

2. Edit to `research_step()` in `scripts/research_conductor.py`:
   inserted between the pre-flight reaper and the Sonnet
   `run_agent` call. If the ledger blocks the task, write a
   `blocked_doomed_rerun_no_root_cause` artifact and skip Sonnet.
   Same pattern as the existing pre-gate check
   (`scripts/conductor_gates.py`, commit `7f4e8125`).

**Acceptance gates:**

1. **Replays correctly on the .60–.65 record**: when seeded with
   the known carryovers (Exps 786/527/491/627/603), the ledger
   correctly identifies them as doomed reruns when proposed in
   .67's roadmap (synthetic test case — feed the ledger a hand-
   crafted YAML proposing those experiments without
   `prior_failures:`, expect block).
2. **No false-positive blocks on legitimate iterations**: when fed
   the .65 → .66 transition (where Exp 836 → 847 is a genuine
   iteration with new technique), the ledger does NOT block.
3. **Validator mechanically detects each of the four discipline
   parts**: tests cover (a) names-prior-failure, (b) names-root-
   cause, (c) names-what-is-different, (d) states-acceptance-gate.
   A YAML missing any one of these is rejected with a specific
   reason.
4. **Honest-verdict enum**: `failure_ledger_loads_correctly`,
   `failure_ledger_blocks_known_carryovers`,
   `failure_ledger_false_positive_above_budget`,
   `failure_ledger_unable_to_parse_artifacts`.

### Exp B — Initial ledger seed + retirement promotion

**Deliverable:** `ops/exclusion_manifest.yaml` updated with the
known-doomed carryovers + `data/failure_ledger_seed.yaml`
(committed to repo, machine-readable seed of prior failures) +
`results/experiment_<N>_initial_ledger_seed.json`.

**What it does:** seeds the failure ledger with the experiments
that have failed three or more times across recent milestones:

- Exp 786 — slow-5 carryover, 6 consecutive milestones unguarded;
  `retire_if_same_verdict: true`
- Exp 527 — same
- Exp 491 — same
- Exp 627 — partially addressed by Exp 841 (.65) batching;
  retire status pending one more re-evaluation
- Exp 603 — same
- iCE40 PnR experiments (Exps 776 / 794 / 804 / 827 / 839 / 851)
  — five consecutive PnR failures; `retire_if_same_verdict: true`
  with explicit "ECP5 / Nexus track is the recommended pivot per
  proposal `issue-004-ecp5-nexus-open-fpga.md`"
- Live-data path bypass (Exps 840 / 853) — recurring
  `simulated_no_verdict` verdict; addressed by .66 Exp 855 LIVE-ENV
  permanent fix; the ledger entry expires when 855 ships clean.

**Acceptance gates:**

1. **Audit each seed entry against the four-part discipline.** For
   each seeded entry, either the prior failures + suspected root
   cause + what changed + acceptance gate are all named, or the
   entry is `retire_if_same_verdict: true` with a clear pivot
   note.
2. **Verify that the seed correctly classifies recent .65 / .66
   activity**: experiments that were *iterations* (.66 Exp 855
   building on .65 Exp 853, etc.) do not match seed entries; the
   seed is for *unaddressed carryovers*, not iterations.
3. **Honest-verdict enum**: `seed_audit_passed_all_four_parts`,
   `seed_audit_failed_some_entries`,
   `seed_misclassifies_iterations_as_carryovers`.

## Risks and honest concerns

- **Scope-matching is not trivial**. "Same experiment" can be
  detected by experiment number prefix (`exp857-` vs `exp850-`) or
  by deliverable shape or by prompt-text overlap. The first
  implementation should use a conservative scope-match (numeric
  prefix + slug stem) and accept some false negatives over false
  positives. False negatives mean a legitimate iteration runs
  un-checked; false positives mean a genuine new experiment is
  blocked. The latter is more expensive.
- **Ledger over-fires on early milestones**. Some of the seed
  entries (like the slow-5) are *known* to be doomed; others
  (like recurring `simulated_no_verdict`) might genuinely be
  fixed by a single milestone's work and should expire from the
  ledger automatically when the fix lands. The proposal needs a
  ledger-expiry mechanism: when an upstream prerequisite ships
  clean, downstream entries automatically leave the ledger.
- **The four-part discipline is hard to mechanically check**.
  "Names a falsifiable acceptance gate" is a content question
  that pattern-matching cannot fully decide. The validator should
  do a heuristic check (presence of "acceptance gate" / "honest
  verdict enum" sections in the prompt) and surface borderline
  cases for human review rather than auto-pass them.
- **The planner may simply not write `prior_failures:` on tasks**
  even after the rule lands. We saw exactly this with the
  `gated_on:` / `max_turns:` planner-prompt update (commit
  `eec2cda4`) — the planner read the documentation but didn't
  populate the new fields on .66 tasks. The fallback if the
  planner doesn't comply is the conductor's pre-launch validator
  rejecting the task with a clear reason and re-running the
  planner with a stronger prompt. We accept the cost of one
  extra planner-loop iteration if the alternative is another
  unaddressed carryover milestone.
- **Risk of over-aggressive retirement**: a task retired too
  eagerly may have been a fixable iteration that just needed one
  more honest attempt. Mitigation: retirement requires
  *reproducing the same verdict* (by string match in the
  honest_verdict whitelist), not just any failure. A task that
  fails differently each time is being iterated on.

## Tie-ins to other drafted proposals

- **CLAUDE.md "Failed-Experiment Rerun Discipline" rule**
  (pending commit) — this proposal is the mechanical enforcement
  of that rule. The CLAUDE.md update lands first, the
  enforcement code lands second.
- **`conductor-self-protection-safeguard.md`** (drafted, queued):
  the dogfood-safeguard pre-exec script guard is a separate
  enforcement layer for *generated code*; this proposal is the
  enforcement layer for *experiment proposals*. Both share the
  pattern of a conductor-level pre-launch validator.
- **Exclusion manifest** (`ops/exclusion_manifest.yaml`,
  mechanism in `_ensure_exclusion_manifest_loaded` in
  `scripts/research_conductor.py`): this proposal feeds new
  retirements into that manifest when
  `retire_if_same_verdict: true` triggers.
- **Planner-prompt update** (commit `eec2cda4`) — same
  intervention pattern. The planner needs to be told (in
  CLAUDE.md and in the planning prompt) about the new YAML
  field. This proposal adds `prior_failures:` to the list.

## Decentralization implications

- Per CLAUDE.md rule 1 (local-first using open models): the
  failure ledger reads only project-local artifacts
  (research-complete.yaml, operational_retro_*.json,
  ops/changelog.md). No external service dependency. ✓
- Per rule 7 (no vendor-specific abstractions in the core): the
  ledger is plain YAML + Python; no LLM or vendor coupling. ✓
- Per rule 4 (multiple integration surfaces): the ledger should
  be queryable both via the conductor's pre-launch validator AND
  via a CLI tool (`scripts/check_failure_ledger.py`) so humans
  can audit the ledger without running the conductor. The CLI is
  a small additional deliverable in Exp A.
