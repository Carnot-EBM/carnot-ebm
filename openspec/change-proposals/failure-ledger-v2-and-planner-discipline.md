# Failure-Ledger v2 + Planner-Discipline Improvements

**Author:** operator (Ian Blenke + Claude session 2026-05-01)
**Status:** Draft, ready for .86 planner pickup
**Origin:** 2026-05-01 .85 mid-milestone incident analysis

## Problem statement

Milestone .85 lost 4 of 14 tasks to **conductor-mechanism bugs and
planner-discipline gaps**, not to legitimately-doomed research. Each
retirement was either prevented or recovered through manual operator
patches; each represents an avoidable loss of ~17-50 min of
research-iteration wall time, and (for some) a permanent loss of the
research finding itself.

The .85 milestone's substantive findings landed (4 of 5 mandatory
phase-validation experiments — exp1090, exp1093, exp1094, exp1095),
but the carry-forward tasks and the cross-vendor codex first attempt
all retired before the discipline checks could even evaluate the
prior_failures field.

This proposal codifies 5 structural fixes to the conductor and 3
planner-prompt deltas so .86+ does not relive the same dance.

## Empirical evidence (from milestone .85)

| Task | Mechanism that retired it | Root cause |
|------|---------------------------|------------|
| exp1092 Phase 1a Verifier Robustness Audit | 3-fail cap (2× GATE_BLOCK + 1× DOOMED_RERUN) | Stacked: missing gate field on exp1090 artifact + missing prior_failures on exp1092; operator patches landed AFTER cap |
| exp1096 SemEnergy Probe v1 | "failed 4 times" auto-skip | Title-prefix matched .84's retired exp1080 SemEnergy Probe v1 SKIPs; conductor counts failures by title-prefix, not experiment_id |
| exp1097 WOPR N-Queens Cartridge | "failed 3 times" auto-skip | Title-prefix matched .84's retired exp1086 WOPR N-Queens GATE_BLOCKs |
| exp1098 Potts Verilog (1st attempt) | codex CLI HTTP 400: "opus model not supported" | Planner emitted `model: opus` on cross-vendor `agent_type: codex` task |

(exp1098 was recovered when the operator patched `_build_agent_command`
to snap cross-vendor model overrides — commit `1f1aef51`. The other
three were not recovered.)

## The 5 unfixed structural issues

### Issue 1: Title-prefix failure-count inheritance

**Problem.** `pick_next_task()` calls `_count_failures_for_task(task)`
which scans `ops/conductor-log.md` for log entries whose title-prefix
matches the current task's title. If a task in milestone .85 has the
same title as a retired task from .84, it inherits .84's failure
count. The planner's `prior_failures:` block is never evaluated
because `pick_next_task` skips before the discipline check fires.

**Fix.** Count failures by `experiment_id`, not title prefix. The
conductor log already carries the experiment_id in the artifact path;
extending the log entry format to include `id:` field is a
non-breaking schema change. Also extend `_count_failures_for_task` to
use that field, falling back to title-prefix only for legacy entries
written before the schema change.

**Code locations.**
- `scripts/research_conductor.py:_count_failures_for_task` (or
  whatever the exact name is — operator should grep)
- `scripts/research_conductor.py:log_step`
- `ops/conductor-log.md` schema: add `id:` field to row format

**Acceptance test.** Manually inject a milestone .Y task with the
same title as a retired .X task; confirm `pick_next_task` does NOT
count the .X failures against the .Y task as long as their experiment
IDs differ.

### Issue 2: 3-fail cap races operator-patch

**Problem.** When a task fails repeatedly with 10-min sleeps between
attempts, the 3-fail cap fires within ~30 minutes. Operator response
time (diagnose → patch → commit → restart) is sometimes longer than
that, especially when patches touch multiple files (artifact JSON +
roadmap YAML + manifest). Today exp1092 retired 7 minutes before the
operator patch could land.

**Fix.** When a fix-shaped commit lands between two attempts of the
same task — specifically a commit that touches that task's roadmap
entry, deliverable artifact, or upstream dependency artifact — reset
the failure counter for that task. Detect by:
- Look at the commits between the most recent failure and now
- If any commit touches `research-roadmap.yaml` AND modifies the
  task's entry, OR touches the task's deliverable JSON, OR touches
  any of the task's `gated_on` upstream artifacts, treat the prior
  failure as superseded
- Soft-reset: zero out the count

**Acceptance test.** Manually inject 3 failures for a task, then
commit a fix touching its deliverable. The conductor's next iteration
must NOT skip the task as "failed 3 times".

### Issue 3: Stable-deliverable detection (60s-unchanged → kill)

**Problem.** The conductor kills a Sonnet/Opus subagent if its
deliverable artifact's mtime hasn't changed in 60 seconds. The check
fires on the first iteration regardless of whether the agent has had
time to write the artifact. If a stale `blocked_gate_check_v1`
artifact pre-exists from a prior iteration, the new agent is killed
within 60s of starting — before it can update the artifact.

This bit exp1090 today: the operator's patched artifact was
`blocked` from a prior DOOMED_RERUN_BLOCK. When iteration 6 spawned
Opus, the artifact was already on disk and unchanged; the conductor
killed Opus 60s later thinking the deliverable was "stable".

**Fix.** The check should require `mtime > task_start_time`, not
just "unchanged for 60s". Specifically:
- Record `task_start_time` when the subagent is spawned
- The stable-deliverable check returns true ONLY if
  `mtime > task_start_time AND (now - mtime) > 60s`
- If mtime is older than `task_start_time`, the artifact is stale
  from a prior iteration and the agent has not yet started writing

**Code locations.**
- Wherever the conductor watches for deliverable updates during a
  Claude Code subagent run

**Acceptance test.** Pre-create a `blocked` artifact, then run a
substantive task expected to take >60s. The agent must not be killed
prematurely; the deliverable must be updated past the
`task_start_time` before the check fires.

### Issue 4: Cache-fingerprint race (saves START fingerprint)

**Problem.** `run_tests()` computes the fingerprint at pre-test
START, runs the ~17-min pre-test, and saves the START fingerprint at
GREEN exit. If the operator commits during the pre-test run (e.g.,
fixing a different task while the conductor is busy), the saved
fingerprint reflects the pre-commit state. The next iteration sees
the post-commit fingerprint, mismatches, and runs the full pre-test
again — even though both runs would pass.

**Fix.** Compute the fingerprint at pre-test END as well as START.
Save the END fingerprint. Operator commits during the pre-test run
get captured; subsequent iterations cache-hit on the post-commit
state.

**Acceptance test.** Run a pre-test, commit a `.py` change midway,
verify the next iteration cache-hits the post-commit fingerprint.

### Issue 5: Failure-ledger keyword matching is too coarse

**Problem.** `FailureLedger.is_doomed_rerun()` matches a task's
title against prior failure titles using single-keyword substring
overlap. This produces 18 matches on "verifier" or "adversarial",
10 matches on "verifier" or "null-space", 2 matches on "diagnostic",
etc. — all false positives because the substantive scope is novel.

**Fix.** Tighten the matcher to one of:
- **Option A (cheap):** require ≥2 keyword overlap from a curated
  scope-vocabulary (e.g., {verifier, robustness, audit, false-pass,
  attack-pattern} as a 5-word set; require any 2 to match before
  flagging the prior).
- **Option B (semantic):** embed both titles via the existing
  diagnostics library (or a small sentence-transformer) and require
  cosine ≥ 0.7 before flagging.
- **Option C (operator override):** allow the planner to declare
  `expected_keyword_matches:` per task; the matcher accepts those
  as in-scope and only flags priors that match additional keywords.

Option A is cheapest and likely catches 90% of false positives.
Option B is more robust but adds an embedding model dependency.

**Acceptance test.** A task titled "Phase 1c Verifier Joint
Null-Space Measurement" must NOT match "Phase 1a Adversarial Verifier
Robustness Audit" as a doomed prior, despite both containing
"Verifier".

## Planner-prompt deltas (3 changes for .86 planner)

The planner reads `_plan_next_milestone()` prompt + CLAUDE.md +
`ops/known-issues.md`. The following must be added.

### Delta P1: Always emit prior_failures: blocks for any task whose
title or scope words appear in `research-complete.yaml`

**Current state.** The .85 planner emitted `prior_failures:` for 6
of 14 tasks. Operator patched 6 more during .85 execution (exp1090,
1092, 1093, 1094, 1095, 1101). Net: 12 of 14 tasks needed the
block; planner emitted half.

**Required.** Before drafting any task, the planner MUST query
`research-complete.yaml` for prior failures with overlapping scope
(by title keywords or by deliverable filename pattern). For every
overlapping prior, emit a `prior_failures:` entry with:
- `experiment_id`: the prior failure's id
- `verdict`: from research-complete.yaml
- `addressed_by`: explicit explanation (one of "false-positive scope
  match — actual scope is X", "real follow-up — different approach
  Y", or "real follow-up — same approach but Z has changed since")
- `retire_if_same_verdict`: true unless the addressed_by is "false-
  positive scope match"

The planner MUST NOT emit a task without this block if any prior
title-word match exists.

### Delta P2: Never emit cross-vendor `model:` overrides on tasks
with non-default `agent_type:`

**Current state.** The .85 planner emitted `model: opus` on
exp1097 (`agent_type: codex`) and exp1098 (`agent_type: codex`),
causing the codex CLI to reject the Anthropic model name. Operator
shipped `1f1aef51` to snap cross-vendor overrides at the conductor
layer; that fix neutralizes the bug going forward, but the planner
should not emit invalid combinations in the first place.

**Required.** The planner-prompt must enumerate the model namespace
per agent_type:
- `agent_type: claude` → models: sonnet, opus, haiku, claude-*
- `agent_type: codex` → models: gpt-5.5, gpt-5, codex-*, o1-*
- `agent_type: gemini` → models: gemini-3.1-pro-preview, gemini-3-*
- `agent_type: opencode` → models: opencode/*

Emitting any model name outside this list for the chosen agent_type
is a planner discipline error. The activation-guard (next section)
catches it before milestone activation.

### Delta P3: Document gate-required artifact fields explicitly

**Current state.** The .85 roadmap's `gated_on:` blocks reference
artifact fields that downstream tasks check (e.g.,
`exp1090.diagnostics_library_written == true`). The .85 planner
documented these in the upstream task's prompt but the upstream
task's experiment script did not always produce that field. Operator
patches were required (exp1090's gate field added manually).

**Required.** For any task with `gated_on:`, the upstream task's
`prompt:` must explicitly enumerate the gate-required fields under a
"REQUIRED ARTIFACT FIELDS:" section, and the planner must verify
those fields are listed in the upstream task's deliverable schema
description. The activation-guard validates this cross-reference.

## Activation-guard validation enhancements

The conductor's existing activation-guard (the planner-output
validator that runs before activating a milestone) currently
validates basic schema. Extend it to:

1. **prior_failures completeness check** — for each task, verify
   that for every prior in `research-complete.yaml` whose title
   shares ≥2 scope-vocabulary keywords, the task carries a matching
   `prior_failures:` entry. Reject the milestone if any task fails.

2. **agent_type/model coherence check** — for each task, verify
   that the model name matches the agent_type's vendor namespace
   (per Delta P2). Reject the milestone if any cross-vendor combo.

3. **gate field cross-reference check** — for each task with
   `gated_on: { artifact_field: X }`, verify that the upstream
   task's prompt enumerates `X` in its REQUIRED ARTIFACT FIELDS
   section. Reject the milestone if any gate references an
   undocumented field.

If validation fails, the activation-guard does NOT swap
`research-roadmap-next.yaml` → `research-roadmap.yaml`. Instead it
writes a `planner-validation-failed` artifact and pings the operator.

## Implementation phases

| Phase | Scope | Effort |
|-------|-------|--------|
| 1 | Issue 4 (cache fingerprint) — single function change | 30 min |
| 2 | Issue 3 (stable-deliverable mtime check) — small | 1 hr |
| 3 | Issue 5 Option A (2-keyword matcher) | 1 hr |
| 4 | Issue 1 (count by experiment_id) — schema change | 2 hr |
| 5 | Issue 2 (3-fail cap reset on patch commit) — needs git introspection | 3 hr |
| 6 | Planner-prompt deltas P1-P3 — markdown + prompt rewrite | 2 hr |
| 7 | Activation-guard checks — Python validation | 3 hr |

Total: ~12 hours of operator work spread across .86. Each phase is
independently shippable and reduces the manual-patch burden by an
estimated 30-50% per phase.

## Why this matters

Today's .85 mid-milestone incidents cost the operator ~3 hours of
manual patching. .86 will hit identical walls without these fixes.
.87+ will compound the discipline drift. The phase-validation
discipline mandated yesterday in CLAUDE.md is producing real
findings (4 of 5 in .85), but the surrounding plumbing is not yet
honest about what it can and cannot do automatically.

This proposal converts today's incidents into durable improvements
so the conductor + planner system structurally encodes the lessons.

## Decentralization implications

All 5 fixes are local-first by design — no external service
dependencies. Issue 5 Option B (semantic embedding) would add an
embedding model, but Option A (curated keyword vocabulary) does
not. Default to Option A unless empirically insufficient.

## Sources

- 2026-05-01 .85 mid-milestone incident analysis (operator session)
- `ops/conductor-log.md` — empirical retirement evidence
- `scripts/research_conductor.py` — current implementation
- `scripts/failure_ledger.py` — current matcher
- CLAUDE.md "Failed-Experiment Rerun Discipline" — the rule this
  proposal protects from being silently violated
