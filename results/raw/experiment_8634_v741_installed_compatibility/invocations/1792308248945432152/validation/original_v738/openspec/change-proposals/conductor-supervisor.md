# Conductor supervisor: external observer + automated recovery

**Status:** Draft change proposal. **REQUESTED FOR MILESTONE 2026.04.77.**
The 2026-04-27 session demonstrated that the operator role of "manual
watchdog for the conductor" is unsustainable — over a 24h window the
operator had to manually reap orphans, kill runaway Sonnets, recover
from broken log handles, translate schema-mismatched planner output,
and commit accumulated work that the conductor's own commit pipeline
failed to push. Every one of those failure modes is in scope for this
proposal. .76 did not pick this up; .77 should.
**Origin:** 2026-04-27 root-cause analysis after a multi-hour overnight
  session where the conductor accumulated three independent failure
  modes — broken log handle, schema-mismatch on a planner output, and
  silent wall-clock-burn on Exp 942/953/954 retries — that none of
  the conductor's existing safeguards detected. Each was caught
  manually by an attendant Claude operator. The handoff notes flagged
  this as an unsustainable pattern: the conductor is the source of
  truth, but no process checks whether it's actually doing what it
  claims.
**Target milestone:** 2026.04.75 — after `conductor-process-isolation.md`
  (process-group ownership, cgroup ceilings) and
  `roadmap-schema-validation.md` (Pydantic at planner boundary)
  have shipped.
**Priority:** **High.** The two earlier proposals fix specific
  failure modes; this one fixes the *meta* failure mode of
  "nothing notices when the conductor drifts." Without it, every
  new failure shape requires a human attendant to catch it.
**Depends on:**
  - `openspec/change-proposals/conductor-process-isolation.md` —
    the supervisor needs the orphan-tracker registry (Exp E) to
    do its claim-and-cleanup work.
  - `openspec/change-proposals/roadmap-schema-validation.md` —
    the supervisor's "claimed vs actual state" check uses the
    Pydantic model as the contract.

## Summary

The conductor's existing safeguards are all internal: pre-flight
GPU reaper, wall-clock timeout, exclusion manifest, failure ledger.
None of these catch the case where **the conductor itself misbehaves
or its observation channels break**:

  - Log handle severs (4× this session) → conductor runs blind for
    hours → operator attention required to detect.
  - Wall-clock reaper fires but subagent leaves orphans → swap
    saturation → no one knows until manual `pgrep`.
  - Conductor commits the same task 3× under different names →
    history accumulates noise → no one sees the duplication.
  - Pre-commit chain truncates output mid-stream → conductor parses
    as "Commit failed" → in-process docs path rescues but the
    failure is invisible to the operator.

The fix is an **external supervisor process** that runs alongside
the conductor and watches it. The supervisor is small (~300 LOC),
single-file, no Sonnet calls, no LLM logic — pure Linux process
inspection + git verification + structured-state reconciliation.

## What this proposal IS NOT

- **Not a Sonnet-driven secondary research agent.** Zero LLM calls.
  All decisions are deterministic file/process/git checks.
- **Not a service / daemon / k8s pod.** Single Python process started
  by `systemd-run --user` alongside the conductor; if the supervisor
  itself crashes it auto-restarts via systemd's Restart=always, but
  it does not run as a system-level service.
- **Not a Sonnet-rewrite-the-roadmap recovery agent.** When the
  supervisor detects "broken state," it logs + alerts (PushNotification
  to operator) + on a small, well-defined whitelist, takes
  *minimal* recovery actions. It does not synthesise YAML, does not
  rewrite specs, does not invoke planners.

## Proposed experiments

### Exp A — Heartbeat + iteration-cadence watchdog

**Deliverable:**
new `python/carnot/supervisor/__init__.py` +
`python/carnot/supervisor/heartbeat.py` +
`scripts/conductor_supervisor.py` (entry point) +
`tests/python/test_supervisor_heartbeat.py` +
systemd unit at `ops/systemd/carnot-supervisor.service` +
`results/experiment_<N>_supervisor_heartbeat.json`.

**What it does:**

The supervisor polls `logs/conductor.log` every 60 sec and tracks
the last-update timestamp. The conductor is expected to log at least
one line per 10-min iteration. If `now - last_update > 30 min`:

  1. Check whether the conductor PID is still alive
  2. If alive: the log handle has severed. Log to a SUPERVISOR-OWNED
     file at `ops/supervisor/log/incidents.jsonl` and surface via
     `PushNotification` if configured.
  3. If alive AND `now - last_update > 60 min`: the conductor is
     wedged. Send SIGTERM to the conductor process group; let
     systemd respawn. Tag the incident in the supervisor log so the
     restart cadence is auditable.
  4. If dead: log + PushNotification + don't auto-restart (manual
     review required).

The supervisor's own log is durable — written via `O_APPEND` to a
JSONL file with explicit `os.fsync()` after each entry. This is the
fallback channel when the conductor's stdout breaks.

**Acceptance:** kill the conductor's log handle (truncate the file
mid-write) and verify the supervisor catches the silence within
30 min.

### Exp B — Claimed-vs-actual state reconciliation

**Deliverable:**
`python/carnot/supervisor/state_reconciler.py` +
`tests/python/test_supervisor_reconcile.py` +
`results/experiment_<N>_supervisor_reconcile.json`.

**What it does:**

Once per iteration (60-sec cadence), the reconciler asks four
questions in sequence:

  1. **What does the conductor claim?** — read the most recent commit
     message from `git log -1`. Extract any "Exp NNN: <title>" or
     "Activate milestone NNN" or "Doomed-rerun block: <task>" tags.
  2. **What is on disk?** — list `results/experiment_*.json` and
     `research-roadmap.yaml`'s `milestone:` field.
  3. **What does the active YAML say is in flight?** — load
     `research-roadmap.yaml` via the Pydantic schema (from
     `roadmap-schema-validation.md`) and identify the next-task that
     hasn't yet produced a deliverable.
  4. **What is the conductor's process tree doing?** — `ps --ppid
     <conductor_pid>` to see whether a Sonnet sub-process is in
     flight. If yes, what `claude -p` flags is it running with
     (max-turns, model)?

The four answers should agree. When they don't, the reconciler logs
a `state_drift` incident with all four observations, so a future
operator (or me) can see what failed. Examples it would catch:

  - Conductor claims to have committed Exp NNN, but no
    `experiment_NNN_*.json` exists on disk
  - Roadmap says milestone .74 active, but the most recent
    conductor commit is `Activate milestone 2026.04.73` (the .74
    schema mismatch from tonight)
  - Conductor's Sonnet sub-process is mid-flight on Exp 954, but
    the active YAML's first-uncomplete is Exp 953 (cross-talk)

**Acceptance:** synthetic test inserts a deliberate drift (e.g.,
delete a deliverable file the conductor committed) and verifies the
reconciler's `state_drift` incident contains all four observations
and a clear "what to look at" pointer.

### Exp C — Bounded auto-recovery (conservative whitelist)

**Deliverable:**
`python/carnot/supervisor/recover.py` (whitelist of recovery
actions) +
`tests/python/test_supervisor_recover.py` +
`results/experiment_<N>_supervisor_recover.json`.

**What it does:**

For a *small, deterministic* set of incident shapes, the supervisor
takes recovery action without operator involvement:

  - **Orphan reap** — when `host_health_check` returns ALERT and
    the offending PID's PPID=1 (true orphan), SIGTERM the PID's
    process group. (The orphan-tracker registry from
    `process-isolation` Exp E is the source of truth for "is this
    actually an orphan?")
  - **Conductor restart** — when heartbeat watchdog confirms
    `wedged > 60 min`, SIGTERM the conductor PG, let systemd
    respawn.
  - **Log handle reset** — when log silence detected and conductor
    is alive, send the conductor SIGUSR1 (handler in conductor
    re-opens its log file). The conductor must support this (Exp D).

For everything else — schema mismatch, planner output drift, doomed
rerun without prior_failures, pre-commit hook chain failure, swap
> CRITICAL — the supervisor logs + PushNotifies the operator and
takes no action.

The whitelist is deliberately small. Adding a new auto-recovery
action requires updating this proposal first, with explicit rationale
for why the deterministic action is safe.

**Acceptance:** the three whitelisted actions each fire correctly
on synthetic conditions; for any other ALERT shape the supervisor
logs + alerts but takes no action.

### Exp D — Conductor-side SIGUSR1 log-reopen handler

**Deliverable:**
edits to `scripts/research_conductor.py` to install a SIGUSR1 handler
that re-opens its stdout/stderr log file +
`tests/python/test_conductor_log_reopen.py` +
`results/experiment_<N>_conductor_log_reopen.json`.

**What it does:**

Cooperative half of Exp C's "log handle reset" recovery. When the
conductor receives SIGUSR1, it closes and re-opens its log file
(`logs/conductor.log`) without restarting. This recovers from the
log-severance pattern we hit 4× this session without losing
in-flight work.

```python
import signal

def _reopen_log_file(signum, frame):
    """SIGUSR1 handler: re-open the log file in case it got severed."""
    for h in logger.handlers:
        if hasattr(h, 'stream') and hasattr(h, 'baseFilename'):
            # Close and re-open
            h.close()
            h.stream = open(h.baseFilename, 'a')
            logger.info("Log handle re-opened by SIGUSR1")
            break

signal.signal(signal.SIGUSR1, _reopen_log_file)
```

**Acceptance:** simulate the log-severance pattern (truncate the
file mid-write), send SIGUSR1, verify subsequent log lines appear
in the file.

## Decentralization implications

- **Rule 1 (local-first):** unaffected. All checks are local file /
  process / git inspection.
- **Rule 7 (no vendor abstractions):** the supervisor lives in
  `python/carnot/supervisor/` (new sub-module). systemd is used as
  a process supervisor, not as a vendor SDK; the unit file is
  swappable for any `init` system.
- **PushNotification** is a Claude-Code-specific feature; if the
  supervisor isn't running under Claude Code, fall back to writing
  to `ops/supervisor/log/alerts.jsonl` and let the operator
  (human or otherwise) read it.

## Why this is in change-proposals, not just a code change

The supervisor is the durable answer to "who watches the watchman."
Every individual conductor failure mode this session had a fix
proposal, but the unified pattern — *no external observer* — has
no scoped fix. The supervisor is that fix. It must be auditable
(small whitelist, durable log), bounded (no LLM calls, no spec
rewrites), and supervised itself (systemd Restart=always so the
supervisor doesn't become a single-point-of-failure of its own).

## Risks

- **Supervisor itself becomes a SPOF.** Mitigation: systemd
  `Restart=always` + a `WatchdogSec=120` so the kernel kills it
  if it stops watchdogging. The supervisor is small enough (<300
  LOC) that its bug surface is bounded.
- **False-positive restart.** If the supervisor's heartbeat
  threshold is too tight, it'd restart the conductor mid-experiment.
  Mitigation: 60-min wedge threshold (well above any legitimate
  iteration including the 50-turn planner Sonnet).
- **Auto-recovery whitelist creep.** Easy to add "just one more
  action" until the supervisor is making policy decisions. The
  proposal explicitly forbids this — extending the whitelist
  requires updating the proposal with rationale.
- **Log churn.** The supervisor's JSONL incident log grows
  unboundedly. Mitigation: log-rotate at 10 MB / 30 days, archive
  to `ops/supervisor/log/incidents-<date>.jsonl.gz`.

## Acceptance criteria (overall)

1. The supervisor catches the four failure modes from tonight's
   session: broken log (within 30 min), schema mismatch (within
   60 min), orphan accumulation (within one health-check cycle),
   conductor wedge (within 60 min).
2. The supervisor has zero LLM calls. It is auditable end-to-end
   in <300 LOC.
3. The supervisor is itself supervised by systemd; its own log
   is durable via `O_APPEND + fsync`.
4. The whitelist of auto-recovery actions has exactly three
   entries (orphan reap, conductor restart, log handle reset).
   Adding a fourth requires a proposal amendment.
5. The supervisor is the unifying answer to "no external observer"
   — the third root cause from the 2026-04-27 RCA. After it ships,
   the operator role of "manual watchdog" should not be required
   for routine milestone runs.
