# Conductor Self-Sufficiency — Case Record and Decisions (2026-08-23)

Operator goal, verbatim:

> "spawn a fable 5 task to plan and implement prevention for those required
>  operator intervention cases from occurring again. our goal here is to be
>  able to soon remove the need for this outer loop and be able to trust the
>  conductor to monitor and fix the inner loop itself."

Predecessor: `conductor-self-supervision-2026-08-22.md` built the DETECTION
layer (sentinel + findings ledger). This work builds the step AFTER
detection: action, or a yes/no escalation where action is not yet safe.

## 1. The verified case list

Every case below was verified against primary evidence (git log, job-dir
run logs, /proc, journalctl, systemd state), not from session memory.
Corrections to the remembered version are marked.

### A. GPU runs burned hours on provably-invalid rows; only humans stopped them

Four incidents in two days, all with rows stamped `llm_on_row_valid: false`
while the run looked alive:

| run | duration | rows | end |
|---|---|---|---|
| supab2 off-arm | 2h50m | 3/3 invalid | killed by hand; dir renamed `_VOID_oom` |
| supab3 off-arm | 4h28m | 3/3 invalid | ran to natural exit |
| supab3 on-arm | 1h08m | 0 rows ever | CUDA OOM into a stranded fragment; SIGKILL by hand (exit 137) |
| baseline25 first attempt | 2h38m | 1/1 invalid | SIGTERM by hand; relaunch with thinking OFF did 10 games in 22 min |

The sentinel fired CRITICAL at 22:39Z on supab3 (2/2 rows at that point).
The run then continued 1h33m; the sibling arm then launched into the same
broken GPU. Detection was machine. Action was human-only, by design
(the sentinel has no kill primitive, enforced by a source test).

### B. Thinking-mode default: the silent 43-hour class

Correction to the remembered version: no `enable_thinking` commit exists.
The repo fix (512eca0e6b) raised the induce budget 4096 -> 131072 to
ACCOMMODATE Qwen3.8's inline thinking (median 62,490 tokens/induction).
Even fixed, local throughput was ~1 row per 2h38m. The thinking-OFF
relaunch mechanism lives ONLY in the outer-loop job dir
(`baseline25/run_nothink.sh`), not in the repo; the repo default remains
think-ON with a recorded held-out-accuracy rationale
(`arc_executable_world_model.py:6400`).

The near-miss is worse than remembered: FOUR live prompt constants still
embed a literal `/no_think\n` that the current pin consumes as prompt
text (`arc_executable_world_model.py:3159` `_L2_CODEONLY_DIRECTIVE` — the
live default — plus `arc_reactive_verifier_filter.py:135`,
`arc_tool_loop_lookahead.py:126`, `arc_llm_guided_solve.py:82`), and
`arc_actions_to_progress.py:104-115` still describes a generator two pins
back. No test relates thinking state to the pin.

The live signal existed: 156 consecutive generations, every one ending
"[HIT n_predict=4096 OUTPUT LIMIT before completing]", recorded and read
by nothing.

### C. The conductor ran stale code ~11.5 hours

Three commits changed `scripts/research_conductor.py` on 2026-08-22
(12:35, 16:21 — the sentinel wiring — and 23:52 EDT) while a process
started near noon kept running old code until the 00:06 restart. No
mechanism anywhere restarted or re-exec'd on HEAD movement (0 execv, 0
rev-parse, no systemd path units). `ops/known-issues.md` documented the
class; nothing automated it.

### D. QA-layer audit dead 23 days at its caller's timeout

Fixed before this work (a76b5f03f8: `AUDIT_TIMEOUT_SLACK_S`, an AST test
pinning the budget<timeout inequality class-wide). The receipt mechanism
(REQ-CONDUCTOR-RECEIPT-1/2) now catches the silence; the cause fix was
human but is now regression-locked. Nothing further built here.

### E. Orphaned llama-server: detectable, unreapable

The sentinel flags ORPHANED_LLAMA_SERVER (WARN, conservative
port-reference check). The janitor's kill filter is exactly
`comm in {python3, pytest}` — a llama-server was structurally unreapable
by any machine path. The 2026-08-22 orphan held 20.7 GB with PPID 1 and
zero connections until a human killed it; a second orphan cost the tr87
acceptance cell through GPU-1 contention.

### F. Unexplained SIGTERMs

exp6227 (2026-08-09): ten server deaths classified
`server_received_interrupt_sender_unknown`; every sender-identification
tool needs root (auditctl/ausearch/bpftrace denied). The 2026-08-23 00:28
exit-143 of the baseline25 relaunch bracket-matches a deliberate human
SIGTERM (resumed by hand 12 min later), but no repo record names any
actor. The conductor's reaper logs both branches; the janitor uses
SIGKILL; earlyoom logged zero kills with memory >= 96% free. Residual:
external/human kills stay unattributable without root. What is buildable
is making OUR OWN automation never add to this class — every kill this
work adds writes a durable actor line.

### G. The outer loop itself was wrong, repeatedly

The class that makes "keep a human watching" a bad answer, verified:
three wrong "verdict poisoning" reports (the enum-first conductor was
correct every time; proven by stripping `verdict_class`); a misread
`adversarial_verify --json` key that called a CRITICAL artifact clean; a
check built and deleted the same day after a corpus sweep showed 221
false fires; near-misreading its own pre-commit stash mtimes as fresh
audit output; claim-audit precision measured at 0.60 strict. Structural
consequence adopted here: no stop predicate ships without a recorded
corpus sweep (REQ-CONDUCTOR-AUTHORITY-2 rule 5).

### H. Ledger findings sat unactioned

Three CLAIM_OVERSTATED rows (two from 2026-08-22) were dispositioned only
after an operator prompt. First escalation was 7 days against a loop
closing several milestones per day.

### I/J. The watchers were themselves dead (cases the brief missed)

- `carnot-outer-loop.timer`: the ONLY automated conductor-restart path,
  enabled, last fired 2026-08-19, NextElapse empty — silently dead 3.5
  days.
- `conductor_supervisor.py`: not running for months; no unit, no pidfile;
  the conductor docstring about it is inverted.
- `conductor_commit_watchdog.sh`: wired nowhere; no crontab on this host.
- The janitor early-exited its reap half for 4h39m+ while the conductor
  was down, logging "not alive — skipping" every 31 minutes.
- The 2026-08-22 stop WAS deliberate (`ops/status.md`: "Conductor is
  STOPPED and must be restarted once the running local work finishes") —
  but the intent lived in prose no machine reads. That is the actual
  defect: no machine-readable stop-intent.
- journald retention on this host measured at 1h45m (SystemMaxUse=50M):
  every durable record must be a tracked file. Confirmed, already the
  established pattern.

## 2. What was built (all spec'd in openspec/capabilities/research-harnesses/spec.md)

| Case | Mechanism | Default |
|---|---|---|
| E | REQ-CONDUCTOR-AUTHORITY-1: janitor-scheduled orphan reap, six conjunctive conditions + 25-min two-scan persistence, durable actor lines | ON |
| A | REQ-CONDUCTOR-AUTHORITY-2: dead-tier run stop, rows x server x persistence x opt-out; yes/no packet when disarmed | OFF (arm file) |
| C | REQ-CONDUCTOR-FRESHEXEC-1: loop-boundary re-exec on committed self-change (HEAD-bytes only, compile gate, exec-storm guard) | ON at next restart |
| I/J | REQ-CONDUCTOR-RESTART-1: janitor starts a dead conductor unless `~/.carnot/conductor-hold` exists; stale hold WARNs | ON |
| H | REQ-OPS-AUDIT-LEDGER-1 amendment: first escalation 1 day, weekly re-bucket unchanged | ON |
| F (partial) | Actor lines on every authority kill; our automation can no longer kill silently | ON |

Corpus evidence recorded before build: the run-stop row predicate fires on
0 of 18,539 `results/**` artifacts (413 with LLM-on rows) and on exactly
the two true incident row files. Live-box dry-run after wiring: zero
candidates.

## 3. The autonomous-stop decision, argued

The evidence threshold that separates machine from human:

- **Validity is machine.** A run whose LLM tier is PROVABLY dead — every
  LLM-on row invalid (or no rows at all after 30 min) AND independent
  server-side evidence (an allocation-failure line in its own fresh log,
  or no listener on its declared port) — measures nothing on that tier.
  All four A-incidents satisfy the conjunction except baseline25.
- **Efficiency stays human.** baseline25 had invalid rows and a HEALTHY,
  generating server (thinking misconfigured). Stopping it was a judgment
  about wastefulness. The predicate deliberately does not fire there; no
  mechanical predicate should.

Costs, measured: false-continue cost this window was ~11 GPU-hours plus a
near-miss wrong conclusion (ar25 read as "the supervisor makes things
worse"). False-stop cost is bounded by the harness's incremental
whole-file row rewrites: at most the in-flight cell (~2h worst observed)
plus a scripted relaunch; and the llm-off arms of a --both run, which are
cheap to re-run.

Middle paths considered and rejected:

- **Pause (SIGSTOP):** keeps VRAM held, still needs a human to resume,
  and delays a legitimate run if wrong — worse than kill on every axis
  here.
- **Kill only the llama-server:** the harness's retry/relaunch behavior on
  a vanished server is undefined per-path; could convert a measurable
  failure into a hang.
- **Dead-man auto-stop after T unless disarmed:** punishes every future
  absent human for every future false positive, forever. A ONE-TIME
  operator arming decision against recorded evidence is strictly better.

Decision: build the full stop mechanism, ship it DISARMED. While
disarmed, a qualifying candidate emits a packet with the exact kill / arm
/ opt-out commands — the human decision is yes/no with evidence attached,
never an investigation. Recommendation to the operator: arm after the
packet path has produced 1-2 correct packets, or immediately if the next
incident costs hours again. Arming is:
`echo "$(date -u +%FT%TZ) armed" > ~/.carnot/stop-authority-armed`.
The orphan reap ships armed: every one of its six conditions says
"nothing owns this process", and the residual risk (a deliberately
detached interactive server idle >2h with no client and no connections)
has the `CARNOT_STOP_AUTHORITY_ALLOW=1` opt-out.

## 4. Deliberately not built, and why

- **A per-cell generator health gate inside the lever harness.** Right
  layer in principle (fail-fast beats external kill), but the harness has
  a live acceptance run in flight, and any harness edit triggers the
  freshness lint's rebuild machinery across its dependent artifacts. The
  external authority covers the class at <=30-min latency. Revisit when
  the harness is quiet.
- **A cap-hit/degenerate-induction sentinel detector (case B live
  signal).** The row-validity path already fired in every historical
  B-incident; a notes-based detector would be a second pattern list over
  log-format-fragile text with poor latency (rows land hourly in the bad
  state). The class is double-covered upstream (budget moves with the
  pin + kernel parity test).
- **Fixing the four stale `/no_think` literals.** A prompt change to the
  scored path with measured baselines behind it — operator-adjacent, and
  a live run is mid-flight. Reported here with exact locations instead.
- **Auto-disposition of ledger findings.** The answer stays human by
  design; only the cadence changed.
- **A new daemon, and any watcher-of-the-watcher beyond the janitor.**
  The janitor timer (Persistent=true, verified firing) is the surviving
  cadence; the sentinel/authority receipts make THEIR death visible. The
  dead `carnot-outer-loop.timer` should simply be retired by the operator
  once the outer loop goes away — flagged, not changed here.
- **Root-privileged signal-sender forensics (case F).** Needs auditd
  configuration only the operator can do; recorded as the standing
  limit.

## 5. What still genuinely requires a human

- Arming the run-stop authority (one command, one decision).
- Ledger dispositions: whether a CLAIM_OVERSTATED verdict warrants a
  corrigendum.
- Slow-but-valid runs: the baseline25 class. Efficiency judgment.
- Prompt/scored-path changes (the `/no_think` literals; thinking on/off
  policy).
- External/root-cause forensics for kills our automation did not perform.
- Deciding to stop the conductor on purpose — now one `touch` of a hold
  file instead of a status.md paragraph nobody executes.

## 6. Session note: the shared-checkout hazard, measured again

During this build, uncommitted work in the shared checkout was clobbered
three separate times by concurrent activity (pre-commit stash-restore
failures during other agents' commits; a sibling session twice reverting
an edit its own brief forbade IT from making, and adding a forbidden
skipif to a test). Only commits protect work here — the
commit-first discipline is not optional, and narrow pathspec commits plus
immediate re-verification after every landing are the working method.
