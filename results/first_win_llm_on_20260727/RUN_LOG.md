# Run log — ARC first-win re-measurement with a working generator (2026-07-27)

Append-only. Nothing here is rewritten; superseded partial runs are moved aside, not deleted
(never-prune).

## Scope decisions, with the measurement that forced each

**07:30 — pilot, K=2, 2 cells (lp85, tu93), n_ctx=81920.**
Server `/props` read directly: `total_slots: 4`, `n_ctx_reported: 81920`. Device verdict
`CONFIRMED_GPU1_BY_PER_PID_RESIDENCY`, 13452 MiB, GPU 0 untouched at 4 MiB. Both cells
LLM-engaged (`calls` 4 and 6, `responses` 4 and 5, `errors` 0). Per-cell wall 164 s and 272 s.
Cells preserved under `pilot_k2/`; they are NOT pooled into the K=4 arms because mixing
concurrency inside one arm would confound the very variable under test.

**07:36 — llm_off launched at the full N=100** (25 games x variants 1,2,3,4), K=4.
Throughput ~26 s/cell. This arm is the fidelity control and is worth the full N because it
has to reproduce the baseline's 4/100 AND its exact winning variants.

**07:38 — llm_on_fix launched at N=100. SUPERSEDED at 08:07.**
First completed cell measured 330.7 s (ar25~color04, LLM-engaged, `calls`/`responses` 4/4).
At 330 s/cell over 4 workers that is ~2.3 h for one LLM arm, and two LLM arms plus the
control did not fit the session budget. Torn down by explicit PID (`kill -15 2259615`, then
`kill -15 2259672`) — never `pkill -f`, whose pattern would have matched this session's own
command line. GPU 1 returned to 306 MiB, GPU 0 unchanged at 2882 MiB (the conductor's own
work, untouched throughout). The partial cell + log are preserved in
`superseded_v4_partial/`.

**08:08 — llm_on_fix relaunched at N=25** (25 games x variant 1), K=4.
Why variant 1 specifically: the baseline's four wins are lp85~color01..04, so the variant-1
slice contains exactly one baseline winner out of 25 games — a matched-subset control rate of
1/25 = 0.04, numerically identical to the published baseline. That makes the reduced scope
directly comparable instead of a differently-defined quantity, and keeps all 25 games (a
corpus statement, not a few-cell witness).

Power consequence, stated up front: the paired exact test's smallest reachable two-sided p is
`2 * 0.5^n` at n discordant pairs, so >= 6 variants must flip in one direction for ANY result
here to reach p < 0.05. N=25 still admits up to 25 discordant pairs, so reducing N from 100 to
25 does not remove the ability to reach significance in kind; it widens the interval on each
arm's own rate (Clopper-Pearson at 1/25 is [0.001, 0.204]).

## Bugs found in my own analyser, by testing it rather than trusting it

1. **Sentinel summed as a count.** `E3AgentPolicy.generator_liveness_witness` writes
   `llm={calls:-1,responses:-1,errors:-1}` when the installed proposer has no
   `liveness_witness()` — which is exactly the `_NoOpProposer` control arm. Summing that gave
   `total_llm_calls = -15` on a 15-cell arm. Fixed by treating negatives as UNDETERMINED and
   counting them separately (never coercing them to 0, which would assert "asked nothing").
2. **Falsy-zero trap in the fix for (1).** Writing the dead-generator predicate as
   `int(b.get("responses") or -1) == 0` reads a genuine `responses: 0` as `-1` (zero is falsy),
   so the check silently stopped firing — on the 8 real origin-incident cells it was written
   for. Caught by guard G2, fixed with an explicit `_num()` reader, and pinned by mutation
   proof M6. This is the same guard-that-does-not-fire-on-its-own-origin-incident shape this
   repo has shipped before.

## Observation about the shipped liveness witness (found by using it)

`llm_enabled` is derived from `CARNOT_ARC_DISABLE_INDUCTION != "1"`, so it reports `true` even
when a stub proposer is installed — the `_NoOpProposer` control arm shows
`llm_enabled: true`. The composite `llm_on_row_valid` *does* correctly report `false` there
(it requires `calls > 0` and `responses > 0`), so the witness as a whole is not fooled; but
`llm_enabled` alone must not be read as "the generator was engaged". Reported, not patched:
changing shipped behaviour is outside this workflow's authorised scope.

## 12:11:48 UTC — the fixed arm died silently at 10/25 cells

**What was observed, not inferred.** `log_llm_on_fix.txt` ends mid-batch: the 10th cell
(`ls20~color01`) was written at 12:11:48 UTC, three further games (`m0r0`, `r11l`, `re86`) had
just been loaded, and then the file stops. No traceback, no `DONE`, no `BLOCKED`. The process
(2261356) was gone. A traceback would have been printed for any Python-level exception, and an
exception inside a worker thread cannot terminate the interpreter — so this was an external
signal, not a crash in my code.

**Ruled out: memory.** 125 GB total, ~92 GB available, ~41 GB free, `/proc/pressure/memory`
avg10 = 0.00 on every line, and nothing matching `oom` or `killed process` in `dmesg` or the
user journal.

**Plausible but NOT proven: process-group collateral from the conductor.**
`scripts/research_conductor.py` SIGKILLs whole process groups (`os.killpg(pgid, SIGKILL)` at
:517 and :940), and its iteration cycled either side of the death (`ops/conductor-log.md`
records activation attempts at 12:10 and 12:12 UTC). My `nohup ... &` jobs inherited the tool
shell's process group rather than getting their own. That is a mechanism plus a timing
coincidence, which is not proof — I am recording it as a hypothesis, not a diagnosis. Notably
the `llm_off` arm launched the same way survived, so if this was the cause it was not
deterministic.

**Mitigation applied (cheap either way).** The remaining arms run from
`chain_rest.sh` under `setsid`, each in its OWN session and process group
(verified: pid = pgid = sid = 2274367), which makes group-directed signal collateral
structurally impossible regardless of whether that was the actual cause.

**Resumption is lossless, not a patch-over.** `run_cell()` returns the cached row whenever the
cell file already exists, so the 10 banked cells are reused byte-for-byte and only the missing
15 are executed. `measurement_wall_s_from_rows` sums each row file's OWN `elapsed_s`, cached
rows included, so the published measurement clock covers the entire arm rather than just the
resumed tail. Per-cell determinism was independently established earlier
(`serialcheck.json`: two repetitions of each probe cell bit-identical), so a resumed cell is
not distinguishable from one run in the original pass.

## 12:12:18 UTC — the llm_off arm died the same way, 30 s after the fixed arm

Checked because the fixed arm's death made it worth checking rather than assumed: process
2258792 is also gone, its log ending at 12:12:18 UTC mid-load of `vc33`, at 90/100 cells. Two
independent processes dying 30 seconds apart is not coincidence, which strengthens the
external-signal reading and effectively rules out anything specific to the LLM arm (the
`llm_off` arm holds no GPU server and makes no network call at all). Resumed under `setsid` in
its own session, same lossless cell-cache resumption as the fixed arm. Both deaths are recorded
here rather than quietly resumed, because a silently-truncated arm reported as complete is
exactly the class of defect this whole workflow exists to correct.

## 12:50 UTC — the fixed arm's resume was BLOCKED by a leaked server, and why that matters

**The harness reported it honestly rather than fabricating.** `run_llm_on_fix.json` (preserved
as `superseded_v4_partial/run_llm_on_fix_BLOCKED_server_launch_vram.json`) recorded
`launch_ok: false`, `launch_s: 600.29`, `blocked_generator_server_launch_failed`. That is the
Pre-Launch-Preconditions behaviour working: a missing resource produced a `blocked_*` verdict
and no measurement, instead of a plausible-looking arm.

**Root cause — a real defect in MY harness's teardown, found by its consequence.**
`kill_server()` killed only the pid that the FIRST `_ensure_server()` spawned. But every worker
thread calls `_ensure_server()`, and when a server dies mid-run a worker RELAUNCHES it. On the
`llm_on_16k` arm that is exactly what happened: the recorded pid 2271222 was already gone (its
teardown logged `ProcessLookupError`), while pid **2279120** was still holding **11818 MiB** on
GPU 1 and still LISTENING on that arm's port 8953 — a server this harness never knew existed.
The next arm then could not fit its own 81920-context server (~13.5 GiB) in the remaining VRAM,
so it waited 600 s and blocked. Note the shape: the recorded pid and the actual server had
diverged — the same declared-versus-actual class as the fault this whole workflow is about.

**Fix, with its own self-test.** `kill_server(pid, port)` now also reaps whatever OWNS the
arm's port, resolved from `ss -ltnpH` rather than from a name pattern (`pkill -f` would match
this session's own command line). Self-tested against the real leaked process BEFORE using it:
`_pids_on_port(8953)` returned `[2279120]`, `_pids_on_port(8919)` correctly returned the
pre-existing iGPU server that must NOT be touched, and `_pids_on_port(8956)` correctly returned
`[]`. The reap then cleared 8953, returned GPU 1 to 306 MiB, and left 8919 alive.

**Second incident, same cleanup: a duplicate chain.** The original `chain_rest.sh` (2274367) had
not exited — after its fix-arm attempt blocked it had moved on and started a
`llm_on_fix_probe` run whose winner-game list had been computed BEFORE `vc33` was won, so it was
probing an incomplete set. Two chains would also have put two generator servers on one card.
Killed by explicit pid (2280851 child, then 2274367), then every port the dead chain could have
owned (8954-8958) was swept and confirmed clear. Relaunched on fresh ports 8961/8962/8963 so no
resumed run can inherit a half-dead listener.
