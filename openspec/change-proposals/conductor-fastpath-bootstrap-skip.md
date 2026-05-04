# Conductor Fast-Path Bootstrap-Status Skip

**Status:** Draft change proposal, ready for .81 pickup.

**Origin:** 2026-04-29 milestone .80 wedge incident. Filed as a
`prior_failures`-compliant infrastructure task to land in .81 after
the .80 retro completes.

**Priority:** **HIGH.** Same severity tier as
`conductor-supervisor.md` and `roadmap-schema-validation.md` —
i.e. an operator-attention-reduction infrastructure fix that
prevents wedged milestones. Belongs in `MANDATORY-NEXT-MILESTONE
PRIORITIES` until shipped.

## Problem statement

`scripts/research_conductor.py:_deliverable_exists()` short-circuited
on **any** file at the deliverable path, regardless of status. This
caused milestone .80 to wedge on 2026-04-29:

1. exp1028 (preflight v30) ran. Per its prompt's "CRITICAL: write
   artifact FIRST" defensive pattern (see
   `research-roadmap.yaml:210-216`), Sonnet wrote a bootstrap-only
   artifact:
   ```json
   {"experiment": 1028, "status": "running",
    "preflight_started": true,
    "pre_test_fixed": false, "gpu_probe_rocm": false,
    "manifest_786_retired": false, "manifest_641_retired": false,
    "manifest_906_retired": false, "gate_schema_v80": false}
   ```
2. Sonnet hit max-turns or otherwise short-circuited before step 5
   ("FINAL ARTIFACT UPDATE").
3. Conductor's pytest self-heal at the iteration boundary passed
   (`88 passed, 1 warning in 5.79s`) and the conductor logged
   `Preflight v30 — OK | 88 passed`. The OK was misleading: it
   measured the conductor's *own* unit-test health, NOT Sonnet's
   completion of the experiment task.
4. Subsequent iterations hit `_deliverable_exists()` → True →
   `Deliverable already exists in repo` fast-path skip. The
   bootstrap-only stub never got overwritten with real values.
5. exp1030 (triple-integration-v6) gated on
   `exp1028.pre_test_fixed == True` → GATE_BLOCK with
   `actual=False == expected=True`.
6. Milestone wedged: 1028 looked done but wasn't; 1030/1031/1032
   couldn't progress; retro couldn't fire.

This is the **second occurrence** of the bug:
- 2026-04-28 17:41Z: `Preflight v30 — INVALIDATED | Prior OK was
  from pretest pass but Sonnet max-turns'd; artifact quarantined
  18:30Z` (rescued manually).
- 2026-04-29 10:09Z: same pattern, no manual rescue, milestone
  wedge persisted across multiple status-check cycles before
  detection.

The first occurrence triggered a manual quarantine. The second
occurred *under* that fix. The structural root cause —
fast-path-without-status-check — was never closed.

## Proposed fix (already implemented in this proposal)

Update `_deliverable_exists()` to read the deliverable JSON when it
exists and skip-fast only when the artifact's `status` field
indicates a finished run.

**Patch summary:** added a `_BOOTSTRAP_STATUSES` frozenset
(`{"running", "blocked", "partial", "in_progress"}`) and made
`_deliverable_exists()` read the JSON, check `status.lower()` against
that set, and return `False` (i.e. *not* finished) when the status is
bootstrap-only. Legacy paths (no JSON, no `status` field, malformed
JSON) preserve old behavior so this is non-breaking for older
artifacts.

**Test coverage** (`tests/python/test_conductor_deliverable_status.py`,
12 tests passing):
- SCENARIO-INFRA-080-A: tasks with no `deliverable` key → False.
- SCENARIO-INFRA-080-B: file missing → False.
- SCENARIO-INFRA-080-C: `status: "running"` → False (the bug).
- SCENARIO-INFRA-080-D: `status` in {`blocked`, `partial`,
  `in_progress`}, any case → False.
- SCENARIO-INFRA-080-E: `status: "success"` → True.
- Legacy artifacts (no `status` field) → True (no regression).
- Non-JSON deliverables (.md, .txt) → True (no regression).
- Malformed JSON → True (no crash, no regression).

## prior_failures (mandatory)

```yaml
prior_failures:
  - experiment_id: exp1028-preflight-v30-bootstrap-1
    verdict: bootstrap_artifact_invalidated_by_quarantine
    addressed_by: |
      First incident 2026-04-28 17:41Z: manually quarantined the
      bootstrap-only artifact at 18:30Z. Did NOT close the
      structural root cause (fast-path skipped any file regardless
      of status). This proposal closes that root cause via a
      status-aware fast-path; manual quarantine becomes unnecessary.
    retire_if_same_verdict: false
  - experiment_id: exp1028-preflight-v30-bootstrap-2
    verdict: bootstrap_artifact_wedged_milestone_80
    addressed_by: |
      Second incident 2026-04-29 10:09Z: same bootstrap-only
      artifact pattern, no quarantine ran, milestone .80 wedged.
      This proposal patches _deliverable_exists() to read the
      artifact's status field and refuse fast-path skip on
      status ∈ {running, blocked, partial, in_progress}.
      With the patch in place, the bootstrap artifact triggers a
      re-run on next iteration; downstream gates wait correctly.
      Acceptance gate: a bootstrap-only artifact written by Sonnet
      then never updated must NOT cause downstream GATE_BLOCK on a
      later iteration.
    retire_if_same_verdict: true
```

## Acceptance criteria

1. `tests/python/test_conductor_deliverable_status.py` passes 12/12
   in CI.
2. `scripts/research_conductor.py:_deliverable_exists()` reads the
   deliverable JSON and rejects bootstrap statuses.
3. Replay: clear `results/experiment_1028_preflight_v30.json` from
   the .80 wedge, restart the conductor, confirm it RE-RUNS exp1028
   (does not fast-path skip), and confirm exp1030 unblocks.
4. No regression: pytest `tests/python/` passes the full suite (or
   the smart-subset the conductor uses for self-heal).

## Replay plan for the .80 wedge

After this patch lands:

```bash
# 1. Stop the conductor (graceful — it's idempotent on restart):
systemctl --user stop carnot-conductor

# 2. Clear the bootstrap-only artifact:
rm results/experiment_1028_preflight_v30.json

# 3. Restart the conductor:
systemd-run --user --unit=carnot-conductor \
  --working-directory=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm \
  --setenv=PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/research_conductor.py \
  --loop --interval 10 --in-process-docs --async-doc-recon
```

Expected: next iteration re-runs exp1028 properly, writes a real
`pre_test_fixed: true` artifact, exp1030 gate clears, milestone
.80 unblocks.

## Why this is in CLAUDE.md's `MANDATORY-NEXT-MILESTONE PRIORITIES`

This is the *third* operator-attention-reduction infrastructure
proposal in three consecutive milestones (after
`conductor-supervisor.md` and `roadmap-schema-validation.md`). The
pattern is: planner Sonnet drifts toward research breadth; the
operator catches a wedge after-the-fact; the structural fix gets
filed as a proposal; and unless the proposal is mandatorily picked
up in the next milestone, the same wedge recurs.

This proposal must be in milestone .81 to land before .82
research-breadth work resumes.

## Strategic rationale

The verifier-saturation theorem (Round 12 of the Zenil chain)
proves Carnot's architecture is provably optimal *if the underlying
infrastructure preserves verdicts faithfully*. Wedged milestones
break that premise — the verdict trail becomes a mix of "1028
shipped (false)" with the actual status hidden in the artifact's
internal fields. Fixing the fast-path to honor status is a
prerequisite for the position paper's reproducibility claims.

## Out of scope (deliberate)

- Stricter Sonnet-side enforcement (e.g. the experiment script
  refusing to exit until the artifact reaches `status: success`).
  That's a separate proposal — Sonnet may need to legitimately
  exit with `partial` if a probe genuinely partially succeeded.
- Auto-quarantine of bootstrap-only artifacts older than N hours.
  Useful but more invasive; this proposal's narrower change is
  enough to close the wedge.

## Estimated effort

Already implemented. Drafting + tests + this proposal: ~1 hour.
.81 pickup is a paperwork formality (validate, retro, close).
