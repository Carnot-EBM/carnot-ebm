# ARC-AGI-3 Submission Refresh — Pre-Flight Report (2026-07-14)

**Status: agent-code payload refreshed and locally smoke-tested end-to-end. One severe bug found
and fixed. Local regression gate is IMPROVED but not yet fully clean — see "Open item" below
before submitting.**

This is a same-day follow-up to `arc-agi3-cuda-submission-runbook-2026-06-30.md` (the last staged
submission, 6/30). That runbook's steps A (build) and staging are still valid as a process — this
note covers what changed and what needs re-verifying before the operator runs steps B/C/D again.

## What changed on the live path since 6/30 (summarized; see `ops/known-issues.md` for detail)

Real bug fixes landed on the scored `E3AgentPolicy` path between the 6/30 staging and today:

1. **Tier-3 world-model induction was completely dead** since 6/28 (a missing `import os` caused
   a silently-swallowed `NameError` on every call) — fixed 7/13. The 6/30 submission ran this
   entire escalation tier non-functional.
2. **Large-grid re-induction couldn't run at all** (induce prompt overflowed the token budget on
   64x64 boards) — fixed 7/14 via RLE encoding.
3. **A crash-on-empty-frame bug** killed the remaining action budget on 2/6 roster games — fixed.
4. **A goal-predicate-consistency veto** now rejects bad LLM-induced goal predicates before
   installing them.
5. **`SUBMITTED_AUTO_HUD_MASK_ENABLED`** flipped True (proven zero-harm efficiency lever).

## New finding from THIS pre-flight (2026-07-14)

Running `scripts/kaggle/arc_local_submission_gate.py --check` against the current live config
surfaced a **severe regression**: 0/8 core games solved vs the verified baseline's 4/8, 7/8 timed
out at the gate's 115s/game cap.

**Root cause (found via a live `faulthandler` stack trace on a real hung `lp85` run, not
guesswork):** `SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED` (flipped True on 7/7) recomputed a full-grid
flood-fill from scratch on EVERY candidate click action instead of reusing the one decomposition
its own caller (`action_tier_rows`) had already computed once. O(candidates x grid_cells) per
step — on a 64x64 grid with thousands of click candidates, a de facto hang.

**Fixed** (`REQ-ARC-FCP-5591-3`, `python/carnot/agentic/arc_color_blob_salience.py`): `score()`
now accepts an optional per-frame cache, threaded through from `action_tier_rows()`. Verified: a
previously-hanging `lp85` run now completes in 25-68s instead of never finishing.

**Flag still disabled** pending re-validation: even fixed, this feature measured meaningfully
slower per action than the pre-feature baseline, for zero demonstrated benefit (three follow-on
live-path attempts using it, same day, all returned `honest_null`).

Committed: `620bf5f65`.

## Open item — local gate not yet fully clean (be honest about this before submitting)

A gate re-run after the fix+disable, on a quieter system (load average dropped from 33.93 to
0.75-6.95 between runs), showed real improvement — **1/8 solved (`vc33` recovered), median actions
on the solve (7777) essentially matches baseline (7761.5)** — but still **7/8 timed out**, and
`lp85`, `m0r0`, `sp80` remain unsolved within the cap.

An isolated, quiet-system `lp85` run (`budget=500`, no induction) completed 496 actions in ~25s —
about 3x slower per action than the baseline's implied rate (7761 actions in <=115s). This gap is
**not fully explained** by the color-blob-salience bug alone (that fix + disable already applied)
and **not fully explained** by system contention (measured on a quiet system). It may be:

- A second, smaller, still-undiagnosed per-step cost (possibly from one of the other 6/30->now
  changes: the goal-predicate-consistency veto, the AUTO_HUD_MASK masking pass, or the
  `_world_model_candidates` NameError fix now actually EXECUTING tier-3 logic that used to
  silently no-op — restoring a real code path can also restore its real cost).
- A genuinely game-specific cost (lp85's 64x64 grid is unusually large; some non-color-blob
  per-step processing may scale with grid size and always have been slower than the gate's tight
  cap allows for THIS specific game, independent of anything changed since 6/30 -- worth checking
  against a pre-6/30 measurement if one exists).

**Recommendation before submitting:** re-run `scripts/kaggle/arc_local_submission_gate.py --check`
yourself on a guaranteed-quiet system (conductor paused) as the final go/no-go signal. If it still
fails to recover the 3 remaining core games, that's real information worth having before spending
the day's one submission slot -- either investigate further, or accept the trade (the fixed hang
bug alone is a large, unambiguous win over the 6/30 build even if this specific local proxy isn't
fully clean; the REAL Kaggle eval's time budget may be far more generous than this local gate's
115s/game local-testing cap, in which case this residual slowdown may not matter at all).

## Payload status (all 4)

| Payload | Status |
|---|---|
| Agent code | **REFRESHED** at `/tmp/cac_stage_daily/` (today's fixes incl. the color-blob-salience fix; guards pass: `MAX_ACTIONS=400`, no `.so` leak, import+build smoke OK) |
| CUDA binary | **UNCHANGED since 6/30** — verified present, same as `/home/ianblenke/carnot_submission_staging/carnot-llamacpp-mtp-binary/` |
| GGUF | **UNCHANGED since 6/30** — verified via SHA-256 checksum match (`e8dd9481...`) against the live HF cache copy |
| `kernel-metadata.json` / `main.py` | **UNCHANGED since 6/30** — `git diff 7124225b8..HEAD` on both files is empty |

## Local end-to-end smoke test — PASSED

Extracted the exact `AGENT_SRC` block from `scripts/kaggle/submission_kernel/main.py`, ran it
locally against the real staged binary + GGUF + refreshed agent-code (file-level symlinks so
`Path.rglob` traverses correctly, matching the real Kaggle mount discovery logic). Real GPU load,
real HTTP health check, real `make_carnot_agent()` build:

```
LLM TIER RESOLVED: server=.../llama-server gguf=Qwen3.5-9B-Q4_K_M.gguf mtp=False ctx=16384 kv=q8_0
LLM GENERATOR HEALTHY -- loaded on GPU, /health ok (generator tier ENGAGED)
```
Exit code 0. No orphan processes; GPU returned to idle after.

## Operator steps to ship this refresh (operator-only, per Operator-Only External Publication)

1. **Re-version the `carnot-agent-code` dataset** with today's fixes:
   ```bash
   .venv/bin/python scripts/kaggle/prep_daily_submission.py
   ```
   This re-stages fresh (re-downloads current dataset, overlays `python/carnot` + `ops/`, re-runs
   guards), re-versions the dataset, re-pushes the kernel, waits for save-run, and writes
   `ops/arc-daily-prep-status.json` with the exact operator-approved submit command. It does
   **not** submit.
2. **Confirm the kernel log prints `LLM TIER RESOLVED: ... mtp=False`** (not `LLM TIER DISABLED`).
3. Optionally, **re-run the local gate** on a quiet system per the Open Item above, as an extra
   confidence check before spending the day's submission slot.
4. **Submit** (operator-approved):
   ```bash
   .venv/bin/python scripts/kaggle/prep_daily_submission.py --submit-only --kver <N>
   ```
   This runs the local submission gate automatically before submitting and refuses on a
   regression (use `--force` only if knowingly accepting one).

## Cross-references

- `docs/research-notes/arc-agi3-cuda-submission-runbook-2026-06-30.md` — the prior full runbook (build/staging process, still valid)
- `scripts/kaggle/prep_daily_submission.py` — the refresh/submit driver
- `scripts/kaggle/arc_local_submission_gate.py` — the pre-submit regression gate
- `openspec/capabilities/arc-human-replay-frame-change/spec.md` `REQ-ARC-FCP-5591-3` — the color-blob-salience fix
- `ops/known-issues.md` 2026-07-15 entry — full incident writeup
