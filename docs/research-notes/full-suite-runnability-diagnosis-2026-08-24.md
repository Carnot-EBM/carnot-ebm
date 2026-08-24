# Making the full test suite runnable — diagnosis and first partial baseline (2026-08-24)

**Scope.** "Can run", not "is green". Fixing product failures was out of scope.
This note answers four questions: is the conductor's `full=True` branch broken,
is the "hangs serially" claim still true, what does a clean serial run actually
measure, and where should the check live.

**Status.** Questions 1, 2 and 4 are answered. Question 3 has a PARTIAL receipt:
the run was killed at 24m47s having covered 20% of the suite. The cause of the
kill is NOT determined. That partial number is still the first serial,
no-coverage measurement in the corpus, and it refutes the estimate the work was
scoped against.

---

## 1. The `full=True` branch: broken AND uncalled

Both. The distinction matters, because "merely uncalled" would make this a
one-line wiring fix, and it is not.

**Uncalled — confirmed.** `run_tests(full: bool = False)` at
`scripts/research_conductor.py:1467`. All four call sites pass the default or
`full=False`: `:6125`, `:6181`, `:6671`, `:6798`. Two carry the comment
"full suite hangs serially".

**Broken — three independent defects, found by reading it.**

### Defect A — the timeout is far shorter than the run (CONFIRMED)

The full branch passes `timeout=600` (`:1537`). The independent reviewer's
finding is **confirmed, and is worse than stated**: the reviewer compared 600s
against an 18-minute estimate. Measured here, the run had covered only 20% at
24m47s (see §3), which projects past two hours. So `timeout=600` is not
marginally short, it is short by roughly an order of magnitude.

`git log -S "timeout=600"` dates the value to `0118d190f9` (2026-04-04) — the
conductor's first commit, when the suite was a fraction of today's size. It was
never revisited, because the branch was abandoned ten days later and nobody ran
it again.

The subset branch, by contrast, was raised to `timeout=1200` (`:1623`) with a
comment recording that the 5-minute cap had been SIGKILLing pytest mid-run. So
the cheap gate carries twice the budget of the expensive one. That is backwards,
and it is a direct consequence of only the cheap one being exercised.

**Consequence for the recommendation:** reviving the branch as written produces a
guard that times out permanently. Wire it up AND fix the timeout.

### Defect B — `-o addopts=` is too blunt

`-o addopts=` blanks the whole ini `addopts` list. It removes `--cov` and `-n 4`
as intended, but it also removes six `--ignore` entries that `pyproject.toml`
applies deliberately. Four of those files still exist and carry 183 test
definitions:

| ignored file | test defs |
|---|---|
| `tests/python/test_experiment_337_retro.py` | 58 |
| `tests/python/test_experiment_355_adversarial_benchmark.py` | 51 |
| `tests/python/test_experiment_368_precision_live.py` | 74 |
| `tests/python/test_experiment_692_preflight_v5.py` | 4 |
| `test_experiment_1033_thinkprm_v4.py` | file no longer exists |
| `test_boltzmann_repair.py` | file no longer exists |

`pyproject.toml` records why they are ignored: they "target an artifact schema
that NEVER materialised". So the `full=True` command deliberately runs 183 tests
the project has decided should not run, and its failure count is inflated
relative to the project's own documented command by however many of those fail.

The fix is to strip only what needs stripping (`-p no:cacheprovider`-style
targeted overrides, or `--override-ini` of just the coverage/xdist flags), not to
blank the list.

### Defect C — a timeout reports an empty summary

`_pytest_run` returns `(-1, "", "Command timed out")` on timeout (`:1518-1519`).
The summary parser at `:1633` then scans `stdout or stderr` — the string
"Command timed out" — for a line containing "passed" or "failed", finds none,
and leaves `summary` empty. `run_tests` returns `(False, "")`.

Every call site treats that as a test failure and launches an agent self-heal
loop. So on timeout the conductor asks an agent to fix failing tests **and tells
it nothing about which tests failed**. Combined with Defect A this is the whole
mechanism: revive the branch as-is and the conductor enters a permanent,
information-free self-heal loop.

---

## 2. "Hangs serially" — refuted as a deadlock claim

The comment traces to commit `15fa328316`, 2026-04-14. Its message reads: "The
full 4000+ test suite hangs when run serially (individual files pass but
interaction effects cause deadlocks)."

Two reasons to stop treating that as current:

**It is four months stale and describes a much smaller suite.** It says "4000+
tests". Collection today is far larger (§3).

**The observed behaviour is not a deadlock.** Throughout the run I sampled the
process directly:

- state `SNl`, 103–106% CPU sustained, never blocked at 0% CPU;
- `utime` accumulating continuously (363s of CPU at the 8-minute mark);
- the open file descriptor walking through `data/arc_transition_corpus/*.npz` —
  `ls20.npz`, then `sk48.npz`, then `ar25.npz`;
- `rchar` at 3.36 GB read.

That is a process doing work, not one waiting on a lock.

**What is really there is a slow phase, not a hang.** Output stalled twice for
~200 seconds at a time while CPU stayed pinned. Both stalls were inside tests
reading the ARC transition corpus. The corpus is only 5.3 MB across 25 files but
3.36 GB had been read, so those tests reload it repeatedly. Seven test files
reference it:

```
tests/python/test_preflight_aa_floor_config_witness_2026_07_30.py
tests/python/test_arc_live_asset_arm_confound_2026_07_30.py
tests/python/test_experiment_4568_clickability_action_effect_predictor.py
tests/python/test_experiment_5641_arc_counterexample_executable_model.py
tests/python/test_arc_transition_effect_rows_hoisted_actions.py
tests/python/test_experiment_6471_arc_generic_safety_shield_objective_ab.py
tests/python/test_experiment_6458_arc_representation_objective_generalization_ab.py
```

**Honest limit.** A ~200s test is not a hang, but I did not run to completion, so
I cannot claim there is no deadlock later in the suite. What I can say is that
the 20% I covered contained no deadlock, and that the two stalls that look like
hangs from the outside are slow ARC-corpus tests. Naming them is the actionable
part: `--durations` on those seven files is the next cheap step.

---

## 3. The baseline — PARTIAL, and the run was killed

### Exact command

```bash
.venv/bin/pytest tests/python -q --no-header -n 0 --no-cov -o addopts=
```

This is byte-for-byte the argument list the `full=True` branch builds at
`:1525-1538`, and it is the invocation `ops/known-issues.md` asks for. The branch
also strips `CARNOT_FORCE_LIVE` from the child env; that variable was unset in my
shell, so the environments match.

### What was measured

| | |
|---|---|
| started | 2026-08-24T18:45:53Z |
| last output | 2026-08-24T19:10:40Z |
| wall time | **1487 s (24 m 47 s)** |
| progress reached | **20%** |
| tests reported | **11905** |
| failures so far | **74 F** |
| skips so far | **9 s** |
| errors so far | **0 E** |
| exit code | none — process died without writing one |

### Collection — the suite is far larger than any prior record says

Same command with `--collect-only`, run separately and to completion:

```
57013 tests collected in 34.02s        (exit 0, no collection errors)
```

11905 / 57013 = 20.9%, which matches the `[ 20%]` marker the run reached — so the
partial measurement and the collection count corroborate each other.

**57,013 is far above every prior figure in the corpus**: exp6582 reported 10,422
total, and the 2026-07-27 known-issues entry reported 29,948. Both of those runs
crashed or were stopped, which is the likely explanation — nobody had seen the
whole collection before.

Two consequences. First, collection itself is cheap (34s), so "can the suite even
be collected" was never the problem. Second, the extrapolation below rests on a
measured denominator rather than a guess.

**Projected full serial run:** 1487s / 11905 tests = 0.125 s/test, so 57,013 tests
≈ **7100 s ≈ 1 h 58 m**. Call it two hours, with the caveat that the ARC-corpus
slow phase sits inside the measured 20% and the remaining 80% is unmeasured in
both directions.

### ROOT CAUSE FOUND (second run, appended 2026-08-24 23:00Z)

**The box's own janitor SIGKILLs the run at exactly two hours. The serial suite
needs more than two hours. So a serial full-suite run cannot complete on this
machine, and never could.**

This is the answer to "make the full suite runnable". It is not a pytest problem,
a deadlock, or a product problem. It is infrastructure killing the measurement.

**The evidence, in one line each:**

| fact | value |
|---|---|
| run 2 exit code, captured outside the owning shell | **137 = SIGKILL** |
| run 2 wall time | 7386 s (2 h 03 m 06 s) |
| run 2 END stamp | `2026-08-24T22:55:51Z` |
| `/tmp/orphan-cleanup.log` | `2026-08-24T22:55:50Z killed 1 orphan workers (conductor=4109305)` |

One second apart. That is the kill.

**The mechanism**, `~/.carnot/orphan-cleanup.sh`:

- `THRESHOLD_MIN=120` — two hours (`:41`).
- It matches on `comm` being exactly `python3` or `pytest` (`:226`).
- It skips any pid that descends from the running conductor (`:241`).
- Everything else over the threshold gets `kill -9` (`:242`).
- `carnot-orphan-cleanup.timer` is `OnUnitActiveSec=30min`, so a run that crosses
  two hours dies at the next fire, within 30 minutes.

**The trap that makes this hard to see.** Detaching the run with `setsid` — which
I did specifically so the agent harness could not kill it — is exactly what makes
it an orphan in the janitor's eyes. Hardening the run against one killer handed it
to another. A run launched *under the conductor* is exempt; a careful standalone
run is not.

**This retro-explains run 1 as a different kill.** Run 1 died at 24 m 47 s, far
under the threshold, so the janitor is not responsible for it. Run 1 was a
harness-owned background task; run 2 was not. Two runs, two different killers,
neither of them a hang.

**It probably explains the 2026-04-14 comment too.** "The full 4000+ test suite
hangs when run serially" is what an unexplained `kill -9` looks like from the
outside: output stops, no traceback, no summary. I cannot prove the janitor
existed in that form in April, so this is a hypothesis, not a finding — but the
observable signature matches exactly, and no deadlock was found in 20% of the
suite across two independent runs.

**Both runs died at the same place, which is itself informative:**

| | run 1 | run 2 |
|---|---|---|
| wall | 1487 s | 7386 s |
| tests reported | 11905 | 11910 |
| F / s / E | 74 / 9 / 0 | 74 / 9 / 4 |

Identical failure counts at nearly identical indices, five times apart in wall
clock. Test ordering is therefore deterministic (no `pytest-randomly` shuffle),
and the 74 failures in the first 20% are reproducible. That is a real, if partial,
result.

The death index is NOT a poison test. The neighbourhood
(`test_experiment_1745_retro.py` … `test_experiment_1752_expand_ltlzinc.py`,
19 tests) was run in isolation: **33 s, 1 failed / 18 passed, exit 1** — a normal
test failure, no kill. So run 2 stopped where it did because that is simply how
far it got in two hours, not because of anything at that index.

### What actually killed each run



The process stopped writing at 19:10:40Z and was gone when next sampled. There is
no traceback, no `KeyboardInterrupt`, no pytest summary — output ends mid-line.
That is the signature of an unhandled fatal signal, not of a test failure or a
Python exception.

Candidates I could not separate, stated as open:

- the agent harness stopping its own background task;
- `ExperimentTemplate.kill_gpu_zombies()`, which per
  `incident_reaper_was_kill_gpu_zombies` fires inside `setup()` **at pytest
  import time** and SIGTERMs GPU-holding processes. It was fixed 2026-08-23 to
  exempt llama-server/vLLM and to attribute utilization per GPU — but a pytest
  process holding GPU memory is not obviously covered by a server-cmdline
  exemption. This is a hypothesis with a plausible mechanism and NO direct
  evidence; I did not capture the signal.
- memory pressure. Unlikely: RSS was 3.39 GB against 89 GB available.

**How to settle it next time, cheaply:** run under `strace -e trace=signal` or arm
`auditd` on SIGTERM (`a1=0xf`) — and note the trap recorded in that same memory,
that a previous investigation armed on SIGINT only and therefore found nothing.
Simply capturing the shell's `$?` would also distinguish 143 (SIGTERM) from 137
(SIGKILL); my wrapper lost it because the whole shell was torn down with the
process.

### What the partial number is worth

It refutes the estimate this work was scoped against. `ops/known-issues.md` says
the baseline "takes 18+ minutes". At 24m47s the run was one fifth done. Linear
extrapolation puts a complete serial run near **two hours**, and the true figure
is likely higher because the ARC-corpus slow phase is front-loaded in the 20% I
saw but the remaining 80% is unmeasured.

That changes the cadence decision in §4 materially: a two-hour job is not
something to attach to a per-commit or per-experiment path under any
circumstances.

**Do not quote 74 failures as the suite's failure count.** It is the count within
the first 20%, it includes whatever share comes from the 183 deliberately-ignored
tests Defect B re-admits, and it is not comparable to the 68/87/433/470 figures,
which came from coverage+xdist runs.

### Contention caveat

Throughout, a concurrent conductor task ran `pytest tests/python -q` with four
live xdist workers at 4.0–6.0 GB RSS. The box has 24 cores and load average was
~5.3, so a single-threaded run had a core available and the wall-time distortion
should be modest — but it is not zero, and memory-bandwidth contention is not
captured by load average. Treat 1487s/20% as an upper-bounded estimate, not a
clean-room number.

### One thing that did NOT go wrong: the record survived

`ops/known-issues.md` states that "Running the full suite is DESTRUCTIVE to the
working tree", with 45 files rewritten on 2026-07-27. **That did not reproduce.**
A mutation snapshot was taken before the run
(`CARNOT_MUTATION_RUN_ID=fullsuite-baseline-20260824`) and checked after. Nothing
under `results/` was modified by the run. The only `results/` path dirty at any
point was `results/experiment_1822_rtl_synth.log`, which was **already dirty
before the run started**.

The credit belongs to `python/carnot/testing/child_results_guard.py`, added
2026-08-24, which carries the results-write redirect across the subprocess
boundary. On this evidence the destructive-run problem is closed for Python
writers, though the guard's own docstring is explicit that a `/bin/sh` redirect
still lands.

---

## 4. Where the check should live

### It must not go on any existing `run_tests` call site

All four are **launch gates**: `run_tests` is called, and on failure the
conductor enters an agent self-heal loop before the experiment may proceed
(`:6125` "Run tests first — ensure clean state"; `:6671`/`:6798` inside the
fix-attempt loop). The suite is red. Wiring `full=True` into a launch gate makes
every experiment in the loop block on unrelated repository-wide failures — which
is precisely the defect that was removed from exp6581/exp6582 today and that
`test_verification_scope_excludes_repository_suite` now refuses.

### CI already has this job, and it has never run

`.github/workflows/ci.yml:98-123` defines `python-test`, "Python Tests (100%
coverage)", running the full suite. It declares `needs: python-lint` (`:101`).
`python-lint` fails. So the test job is **skipped**, not failed.

Sampled every 8th CI run back to 2026-08-23T20:54 — eight runs, `Python Tests`
`skipped` in all eight. Recent runs are 12-for-12 `failure` at the workflow
level, all from `Rust Build & Lint` and `Python Lint & Type Check`.

This is the more important half of the finding. The 2026-04-14 commit that
abandoned the conductor branch justified it with "Full suite runs via CI or
explicit invocation." That justification was false at the time or became false
since. **There are two dead reporters, not one**, and adding a third without
fixing the second would repeat the mistake.

Incidentally: `tests/integration` (3 files) is in `testpaths` but both the
conductor branch and CI pass `tests/python` explicitly, so it is run by nothing.

### Recommendation ZERO — the janitor, which blocks everything else

Nothing below matters until a full-suite run is allowed to finish. Pick one:

1. **Make the run fit inside two hours.** `-n 4 --no-cov` should land near 30
   minutes, comfortably inside the window, and needs no infrastructure change at
   all. This is the cheapest path to a complete receipt and is what I ran next.
   Caveat: it reintroduces xdist, and `ops/known-issues.md` wants `-n 0`
   precisely because a dying worker contaminates counts. Report the worker state
   alongside the numbers.
2. **Exempt a marked run from the janitor.** The janitor already has an exemption
   concept — "descends from the conductor". Add a second: an env marker or a
   pidfile the operator sets for a deliberate long run. Preferable to raising
   `THRESHOLD_MIN`, which weakens the janitor for its actual job.
3. **Run it under the conductor**, which is already exempt. Works today, but
   couples the measurement to the loop it is supposed to observe independently.

Do NOT "fix" this by evading the `comm` match (for example invoking
`.venv/bin/python -m pytest`, whose `comm` is `python` and therefore misses the
janitor's `python3|pytest` pattern). That works, and it is the wrong answer: it
leaves a genuinely orphaned run unkillable and hides the problem from the next
person. Note the fragility though — the janitor's matcher is narrower than its
concept, which is the same class of defect CLAUDE.md's QA-Layer discipline exists
to catch.

### Then, in order

1. **Fix `python-lint` first.** It is the cheapest action with the largest
   effect: it un-skips a full-suite job that already exists, already runs on
   every push, already reports, and costs this project nothing in local
   wall-clock. Until it is fixed, any other reporter is a workaround for a
   working mechanism that is switched off.
2. **Then add a `carnot-full-suite.timer`, weekly, not daily.** Precedent and
   template: `~/.config/systemd/user/carnot-arc-daily-prep.{timer,service}`,
   `Type=oneshot` with `Persistent=true`. Cadence: **weekly**, at a low-traffic
   hour, because the job is ~2 hours of one core and the suite's failure set does
   not change fast enough to justify daily. Cost: ~2 core-hours/week, one core of
   24, no GPU. It must write a receipt (counts, wall time, commit SHA) to a
   tracked path and must **not** gate anything.
3. **Only then consider `run_tests(full=True)`**, and if it is revived, fix all
   three defects in §1 — timeout to 4+ hours or removed entirely, targeted ini
   overrides instead of `-o addopts=`, and a real summary on timeout. Do not
   attach it to a launch gate.

---

## 5. What I could not determine

- **The complete baseline under `-n 0`.** 20%, not 100%, and now known to be
  UNOBTAINABLE under the current janitor: the serial run needs >2h and is
  SIGKILLed at 2h. §3 "ROOT CAUSE FOUND".
- ~~**Why the run died.**~~ RESOLVED for run 2: `kill -9` from
  `~/.carnot/orphan-cleanup.sh`, exit 137, correlated to the janitor's own log
  within one second. Run 1's kill (at 24m47s, under the threshold) is a
  *different* event, most likely the agent harness stopping its background task —
  that one is still inferred, not proven.
- **Whether the April "hangs serially" report was this same janitor.** The
  signature matches; I did not check the janitor's history. Worth one `git log`
  by whoever picks this up.
- **Whether a genuine deadlock exists in the unmeasured 80%.** The 2026-04-14
  claim is unsupported for the part I measured and untested for the rest.
- ~~**The true total test count.**~~ RESOLVED, see §3 "Collection". The count is
  57,013. What remains unexplained is why earlier records report ~10,400 and
  ~29,900 for the same tree; those runs were themselves crashed or partial, so
  the discrepancy is most likely that no one had ever seen the whole collection.
- **How much of the 74 failures is Defect B's fault.** Separating that needs a
  run with the `--ignore` entries preserved, which no one has done either.

## 6. Reproducing this

```bash
export CARNOT_MUTATION_RUN_ID=<something-unique-to-you>
.venv/bin/python scripts/test_suite_mutation_check.py --snapshot
.venv/bin/pytest tests/python -q --no-header -n 0 --no-cov -o addopts=
.venv/bin/python scripts/test_suite_mutation_check.py --check --run-id <same>
git status          # BEFORE any git add
```

Budget four hours, not twenty minutes. Capture the exit code separately from the
shell that owns the process, so a fatal signal is distinguishable from a test
failure — that is the single thing this run most needed and did not have.

**Cross-references:** `ops/known-issues.md` 2026-08-24 full-suite entry ·
`scripts/research_conductor.py:1467-1687` · `.github/workflows/ci.yml:98-123` ·
`python/carnot/testing/child_results_guard.py` · commit `15fa328316` (origin of
"hangs serially") · commit `0118d190f9` (origin of `timeout=600`) ·
CLAUDE.md "Test-Run Record Integrity Discipline".
