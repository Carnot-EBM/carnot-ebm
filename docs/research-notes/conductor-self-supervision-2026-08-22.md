# Conductor Self-Supervision — Plan and Design (2026-08-22)

Operator goal, verbatim:

> "the goal here is to eventually remove the need for an outer loop to
>  constantly keep the conductor and experiments in check."

## 1. The unifying defect

The system WRITES validity signals and never READS them while it matters.

Today's incident is the exemplar. A supervisor A/B ran ~2.5 hours and looked
healthy: harness alive, 3 of 3 rows written, no runner-log errors. Every row
was `llm_on_row_valid: false`. The llama-server had died mid-run:

```
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 629440768
graph_reserve: failed to allocate compute buffers
failed to create context with model '...Qwen3.8-27B-Q4_K_M.gguf'
```

A stranded 3,744 MiB VRAM fragment from a model that never finished loading
starved later cells. One game generated 59,667 tokens; a later one recorded
`resp=0, tok_out=0`. The dangerous part: ar25 read `levels=0` against a prior
valid `1`. Read naively, that says "the supervisor makes things worse" — a
wrong conclusion about the exact mechanism under evaluation. Only a human
reading `llm_on_row_valid` prevented it.

Same shape as two prior incidents: the 57 phantom `OK (conductor)` rows
(written, never checked) and the QA-layer audit that wrote no report for four
weeks while its caller reported success.

## 2. What already exists (do not duplicate)

| Layer | What it closes | Gap left open |
|---|---|---|
| `arc_llm_on_liveness_lint.py` (`check_row`, primitives-only) | POST-HOC row refusal at commit time + advisory at experiment completion | Nothing reads rows while the run is ALIVE |
| `_run_audit_with_receipt` (conductor) | Audits must prove they ran; receipts, not exit codes | Scoped to milestone-close audits only |
| Truthful archival + `research_complete_ledger_lint.py` | No phantom OK in the ledger | Post-hoc |
| Guard-stall recovery (REQ-CONDUCTOR-STALL-1) | Bounded replan, then park with OPERATOR-ATTENTION | Activation path only |
| `experiment_claim_audit.py` | Adversarial claim refutation at milestone close | Findings accumulate with no disposition; nobody must decide |
| `arc_generator_pin_guard.py` | Refuses a retired pin at harness start | Start-time only; says nothing about the server actually loaded |
| `arc_trajectory_supervisor.py` | In-EPISODE stall detection (inside the agent) | Different layer; not the conductor |
| `~/.carnot/orphan-cleanup.sh` (30-min timer) | Kills stale pytest/python orphans | Process-tree only; blind to VRAM, server logs, row validity |
| `conductor_supervisor.py` daemon | Heartbeat staleness, conductor orphans | Blind to run CONTENT |
| `run_agent` deliverable-watch | Polls the deliverable file every 30 s | Reads only size/mtime/status, never row validity |

## 3. What this work builds

Two new tools plus wiring. Both fail closed and both write durable records.

### 3.1 `scripts/conductor_run_sentinel.py` — in-flight run sentinel

One scan pass per invocation. Pure evaluator functions; a thin CLI. Three
detector classes:

**A. Live-run row validity.** Discover live ARC harness runs from `/proc`:
a python process whose cmdline names a `scripts/arc_*.py` file and carries
`--out <path>.json`. `/proc/<pid>/cmdline` is the trustworthy source; a
launcher's claims are not. Parse the out path from the cmdline, read the
JSON, and evaluate each row with `check_row()` IMPORTED from
`arc_llm_on_liveness_lint.py`. Reuse, not a second pattern list: the lint
already recomputes validity from primitive fields. Escalate when >= 2
consecutive LLM-on rows carry FAIL findings. Never kill: a wasted run is
cheaper than a killed legitimate one, and the harness also carries valid
llm-off arms.

Mid-write tolerance: the harness rewrites the whole out file after each row.
An unparseable read with a fresh mtime (< 60 s) is a normal race — skip this
cycle. An unparseable read with a stale mtime is a finding. Absent is not
zero: an LLM-on row with no witness fields at all is a WITNESS_MISSING
finding (the lint already defines it), never silently valid.

**B. Server-log allocation failures.** Locate the llama-server stderr log:
read `CARNOT_ARC_SERVER_LOG_DIR` from `/proc/<pid>/environ` of the discovered
run (fallback: tempdir), then look in `<dir>/carnot_llama_server_logs/` for
`llama_server_p{port}_*.log` matching the run's `--port`. Scan for the
unambiguous failure lines the incident produced:
`failed to allocate`, `ggml_gallocr_reserve`, `failed to create context`,
`CUDA error`, `out of memory`, `ggml_abort`. A hit escalates CRITICAL. No
auto-kill in v1 even here: the evidence is unambiguous about the LLM TIER,
not about the whole run (llm-off arms still measure). A later `--enforce`
flag can add the stop once the escalation path has run clean for a while.

**C. GPU / resource health.** Three checks, all read-only:

1. Stranded VRAM: per GPU, `memory.used` minus the sum of
   `compute-apps.used_memory` beyond a 1024 MiB slack. Measured on the live
   box today: 4 MiB (GPU 0) and 15 MiB (GPU 1) unaccounted — zero false
   positives at this threshold. The incident fragment was 3,744 MiB.
2. Orphaned llama-server: a llama-server PID (from `/proc` cmdline scan)
   whose parent is init (PPID 1) and whose cgroup shows no systemd service.
   A systemd-managed server is legitimate; a reparented one has no owner.
   Escalate WARN; never kill (the janitor owns process reaping).
3. Wrong model loaded: a llama-server whose `-m <path>` does not contain the
   live pin (`ARC_LIVE_GENERATOR_REPO_SUBSTR`, imported — one constant,
   never duplicated; measured import cost 0.43 s). Escalate WARN. If the
   import fails the sentinel reports `pin_check_unavailable` rather than
   silently skipping the check — fail closed means a check that cannot run
   says so.

**Escalation (class D, shared).** Findings append one row to
`ops/conductor-log.md` in the exact `log_step` format with an
`OPERATOR-ATTENTION:` task prefix — journald retention here is hours, so
only the tracked log is durable. CRITICAL findings also append a
`## OPERATOR-ATTENTION <date>:` section to `ops/known-issues.md` (the parked-
milestone precedent). Dedupe by finding fingerprint in
`ops/.run_sentinel_state.json` so a 30-minute cadence cannot spam; the state
file also records `last_scan_utc` on EVERY run — that is the sentinel's own
receipt, per REQ-CONDUCTOR-RECEIPT-1's lesson that a silent monitor is the
worst state.

### 3.2 `scripts/audit_findings_ledger.py` — audits someone must answer

The claim audit found two CLAIM_OVERSTATED verdicts today and nobody decided
anything. The mechanism: a ledger with aging, not another report.

1. Parse flagged verdicts out of `ops/experiment_claim_audit_report.md`.
2. Maintain `ops/audit-findings-ledger.md`: one row per finding —
   first-seen date, artifact, verdict, disposition. New findings enter as
   `OPEN`. Existing rows are never rewritten (never-prune); a human closes a
   row by editing its disposition to `ACCEPTED | FIXED | WONTFIX` plus a
   note.
3. Any `OPEN` row older than 7 days escalates through the same class-D
   writer. Escalation repeats weekly (re-fingerprinted by age bucket) until
   the disposition changes. Unread findings now have a visible, growing age
   instead of silence.

v1 ingests the claim-audit report only. The other five audit reports keep
their existing operator flow; extending the ledger to them is mechanical
once the shape proves out.

### 3.3 Wiring

1. `scripts/research_conductor.py` `research_step()`: run the sentinel via
   subprocess (short timeout, non-fatal) once per iteration, before task
   pick. The conductor is a long-lived `--loop` process; the change takes
   effect at its next natural restart.
2. `~/.carnot/orphan-cleanup.sh`: invoke the sentinel after the
   anomaly-escalation block, `|| true`, same venv-python pattern. This is
   the path that covers OUTER-LOOP-launched runs — today's incident class —
   because the janitor timer fires regardless of conductor liveness. Note
   the janitor's process-reaping half still exits early when the conductor
   is dead; the sentinel invocation sits ABOVE that early-exit.
3. Milestone close: the ledger ingest runs next to the claim audit's
   existing receipt-checked invocation.

## 4. Fail-direction table

| Check | On inability to run | Direction |
|---|---|---|
| Row validity | JSON unreadable + stale mtime -> finding; fresh mtime -> skip cycle | closed (visible) |
| Server log | log dir/file absent -> note in state, no finding (externally-managed server is legitimate) | open, stated |
| nvidia-smi | binary missing/error -> `gpu_check_unavailable` finding once per day | closed (visible) |
| /proc reads | permission/vanished-pid -> skip that pid, count in state | open, stated (races are normal) |
| Live pin import | ImportError -> `pin_check_unavailable` finding | closed (visible) |
| Ledger parse | report absent -> no-op (audit may not have run); malformed row -> finding | mixed, stated |
| Escalation write | conductor-log unwritable -> stderr + non-zero exit | closed |

## 5. Tests and proof

Spec: new REQ-CONDUCTOR-SENTINEL-1/2/3 and REQ-OPS-AUDIT-LEDGER-1 in
`openspec/capabilities/research-harnesses/spec.md`, with scenarios including
a literal replay of the 2026-08-22 incident (3 invalid rows + the allocation
lines above -> CRITICAL escalation). Tests write only under `tmp_path`; the
escalation writer takes explicit paths so no test touches tracked state.
Every check gets a mutation proof: disable the check, show RED, restore,
show GREEN.

False-positive measurement before any wiring: run the row detector over the
historical `results/**` row corpus (the 2026-07-26 contention artifacts are
TRUE positives and must fire); run the GPU detectors against the live box
(expected: zero findings — one healthy pinned server on GPU 1, clean GPU 0).
Ship anything that fires on legitimate history as WARN-only, per the
`SUBSTRATE_HAS_NO_DURATION_FLOOR` precedent.

## 6. Deliberately not built

- Auto-kill of any process. A false stop is worse than a slow human. The
  janitor keeps process reaping; the sentinel only escalates.
- A new daemon. Two existing schedulers (conductor loop, janitor timer)
  already provide cadence; a third resident process adds failure surface.
- Ledger ingestion of all six audit reports. v1 proves the disposition
  mechanism on the claim audit only.
- systemd unit changes. The janitor script edit reuses the existing timer.
- A CLAUDE.md discipline section. The spec REQs plus this note carry the
  contract; CLAUDE.md grows one-per-incident, and this is prevention, not
  an incident writeup.

## 7. What still genuinely requires a human

- Deciding a ledger disposition. The sentinel can force the question; it
  cannot answer whether a CLAIM_OVERSTATED verdict warrants a corrigendum.
- Acting on a parked escalation (restart a server, free a fragment via
  driver reset, approve `--enforce`).
- Judging a slow-but-valid run. The sentinel reads validity, not progress
  quality; "this run is legitimate but pointless" stays human.
- Anything that changes the scored path or kills work.
