# Conductor process isolation: prevent orphan accumulation + swap saturation

**Status:** Draft change proposal.
**Origin:** 2026-04-26 19:42 UTC memory-pressure incident. A separate
  Claude Code session found the host swap-thrashing (123/143 GB used,
  load avg 119) and traced it to orphaned long-running processes from
  prior Claude sessions that had exited without reaping their children:
    - 3 concurrent `experiment_942_math_repair_sota_v2.py` runs
      (combined ~22 GB RSS + 22 GB swap)
    - 7 concurrent `pytest tests/python` runs (combined ~30+ GB RSS +
      25 GB swap), the oldest 5h 52m old
    - 161 defunct python3 multiprocessing zombies
  The intervening Claude SIGTERMed all 20 PIDs (10 bash wrappers + 10
  python parents). Full handoff notes at `/tmp/memory-fix.txt`.
**Target milestone:** 2026.04.74 — earliest practical milestone after
  .73 work on the existing experiments stabilises.
**Priority:** **Critical** — the host became fully non-responsive
  during the incident; another instance would block all autoresearch
  work plus any other process on the machine. The conductor's design
  currently leaks orphan processes by construction.
**Depends on:** nothing — uses standard Linux primitives (setsid,
  flock, systemd-run, cgroup memory limits) already available on the
  host (CachyOS / systemd).

## Summary

The conductor spawns long-running children (Sonnet subagents, the
experiment scripts they invoke, and any pytest those scripts launch)
through `subprocess.Popen` from a shell wrapper. When the conductor
process or its parent Claude Code session exits, those children
**reparent to init** (PPID=1) and survive — the conductor's wall-clock
reaper sends SIGTERM to the subagent it knows about, but it has no
record of the grandchildren the subagent itself spawned. Each leaked
generation can hold gigabytes of mmapped GGUF model + JAX arrays.

The fix is a layered set of OS-level isolation primitives that turn
"long-running children outlive the launching session" into "all
children share a process-group ID the launcher tracks, and the kernel
guarantees they can be killed atomically; runaway memory hits a
cgroup ceiling instead of swap; old runs are detected and refused
rather than stacked."

The six gaps from the incident memo become five proposed experiments
plus one explicit non-goal.

## What this proposal IS NOT

- **Not a rewrite of the experiment scripts.** The scripts continue
  to use `BatchedInferenceRunner`, `experiment_template`, and the
  rest of the existing harness. The wrapping changes — the scripts
  themselves do not.
- **Not a removal of the wall-clock timeout.** The 60-min timeout
  stays as a backstop. The proposal makes the timeout actually able
  to kill the workload (currently it kills only the bash wrapper or
  the immediate child, leaving the GPU-bound python orphaned).
- **Not coupled to systemd specifically.** `systemd-run --user`
  is the recommended path because it gives clean cgroup
  isolation + accounting on this host. A pure-`setsid` fallback is
  in scope as the portable alternative for hosts without systemd
  user instances.
- **Not a workflow change for the human operator.** Direct CLI
  invocation of experiment scripts (`python scripts/experiment_NNN.py`)
  still works; the isolation only wraps the conductor's automated
  spawn path.

## Proposed experiments

### Exp A — Process-group isolation for conductor subagents

**Deliverable:**
edits to `scripts/research_conductor.py:_spawn_subagent` (or
equivalent — the function that calls `subprocess.Popen` for the
Sonnet `claude -p` invocation) +
`tests/python/test_conductor_process_group.py` +
`results/experiment_<N>_conductor_process_group.json`.

**What it does:**

Currently the conductor spawns subagents via plain `subprocess.Popen`.
The shell wrapper from Claude Code sits in between, and when the
conductor SIGTERMs the wrapper PID, the python child (which has its
own PID) survives the wrapper's death.

Wrap the spawn in `setsid` (or `start_new_session=True` in Popen,
which is the Python equivalent) so the subagent's whole tree gets a
new session ID and process-group ID. Kill the entire group on
timeout via `os.killpg(pg, SIGTERM)` then SIGKILL after a grace
period:

```python
proc = subprocess.Popen(
    cmd,
    start_new_session=True,  # creates new PGID matching new PID
    ...
)
pgid = os.getpgid(proc.pid)

# On timeout:
try:
    os.killpg(pgid, signal.SIGTERM)
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    os.killpg(pgid, signal.SIGKILL)
```

**Acceptance:** in a synthetic test, spawn a Sonnet-shaped command
that itself spawns python children. SIGTERM the parent and verify
all descendants are dead within 15 sec. With the existing code, the
descendants survive — that's the regression case the test pins.

### Exp B — Single-run guards via flock

**Deliverable:**
new module `python/carnot/conductor/single_run_guard.py` with
`acquire(name)` / `release(name)` context manager +
`tests/python/test_single_run_guard.py` +
edits to `scripts/research_conductor.py` (wrap research_step's
subagent call) +
`results/experiment_<N>_single_run_guard.json`.

**What it does:**

The incident memo found 3 concurrent `experiment_942` runs and 7
concurrent `pytest tests/python` runs. Both come from the same
pattern: a previous launch is still running, and a new launch
doesn't notice. flock on a per-experiment-script lockfile prevents
this:

```python
@contextmanager
def acquire(name: str, timeout_s: float = 0):
    """Hold a flock(name) for the duration of the block.

    timeout_s=0 → fail immediately if held; the conductor logs and
    skips the launch. timeout_s>0 → wait that long. Lockfile lives
    in /tmp/carnot-locks/<name>.lock so it survives daemon restarts
    but not host reboots.
    """
    lockdir = Path("/tmp/carnot-locks")
    lockdir.mkdir(exist_ok=True)
    fd = os.open(lockdir / f"{name}.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Already held by another process — don't stack
        os.close(fd)
        raise SingleRunHeld(name)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
```

The conductor wraps the experiment-spawn call. If `SingleRunHeld`
fires, the conductor logs `"Skipping <task> — another instance is
already running"` and moves on (treats it as a soft-skip, not a
failure).

**Acceptance:** a synthetic test launches the same experiment
twice in parallel; the second launch raises `SingleRunHeld` within
1 sec. With this guard in production, the .74 milestone cannot
produce the 3-concurrent-942 pattern even if the conductor is
spawned twice (e.g. operator doesn't realise one is already running).

### Exp C — Cgroup memory ceiling per experiment

**Deliverable:**
edits to `scripts/research_conductor.py` to wrap subagent spawn in
`systemd-run --user --scope --property=MemoryMax=...` when available +
`tests/python/test_memory_ceiling.py` +
`results/experiment_<N>_memory_ceiling.json`.

**What it does:**

The incident's worst component was 22 GB of swap consumed by 3
concurrent experiment_942 processes. Swap saturation, not RAM
exhaustion, was what made the host non-responsive — the OOM killer
never fired because each individual process was within its budget.

Wrap subagent spawns in a transient systemd scope with
`MemoryMax=10G` (configurable per-task via YAML field, default 10).
When the scope exceeds the limit, the kernel OOM-kills only that
scope, leaving the rest of the host untouched.

```python
cmd = [
    "systemd-run", "--user", "--scope", "--quiet",
    f"--property=MemoryMax={mem_max_gb}G",
    f"--property=MemorySwapMax=2G",
    "--",
    *original_cmd,
]
```

A `_systemd_available()` helper checks at startup; if systemd-user
isn't reachable (unlikely on CachyOS but portable), fall back to
plain `setsid` + a python-side `resource.RLIMIT_AS` cap.

**Acceptance:** a synthetic experiment that allocates 15 GB while
wrapped in a 10 GB scope is OOM-killed by the kernel within seconds.
Without the cap, it would saturate swap (the incident's
behaviour). The conductor logs the kill and treats it as a normal
timeout — same handling, faster blast-radius limit.

### Exp D — Zombie reaping audit + fix

**Deliverable:**
new module `python/carnot/conductor/zombie_audit.py` with a
periodic `audit_and_log()` call invoked at iteration start +
edits to the experiment harness (`scripts/experiment_template.py`)
to ensure `multiprocessing.Pool.close()` + `.join()` even on
exceptions +
`tests/python/test_zombie_audit.py` +
`results/experiment_<N>_zombie_audit.json`.

**What it does:**

The 161 defunct workers indicate that pytest's parallel runners
(pytest-xdist? the project's own ProcessPoolExecutor wrapper?) are
not calling `wait()` on their multiprocessing children when a test
fails or the parent is killed.

Two-pronged fix:

1. **Audit.** At every iteration start, `audit_and_log()` walks
   `/proc/*/status` for `State: Z` processes whose parent is the
   conductor or any of its descendants. Logs the count + parent PID.
   If count > 50 (configurable threshold), the conductor refuses
   to start a new experiment until the operator clears them — same
   shape as the existing pre-flight reaper for stale GPU processes.

2. **Source fix.** In the experiment harness, wrap any
   `multiprocessing.Pool` / `ProcessPoolExecutor` instantiation
   in a context manager that guarantees `close()` + `join()` on
   exit (including exception paths). The `with ... as pool:`
   pattern from the stdlib already does this; the project may
   have ad-hoc Pool() calls that don't.

**Acceptance:** the audit runs at every iteration without slowing
the conductor (<100 ms). The harness fix eliminates new zombie
accumulation from experiment scripts. The total live-zombie
count over a milestone (.74) stays under 20.

### Exp E — Orphan detection on conductor + Claude session shutdown

**Deliverable:**
new module `python/carnot/conductor/orphan_tracker.py` with
`record(pid, name)` / `claim_orphans()` / `kill_orphans()` +
edits to `scripts/research_conductor.py` to call `claim_orphans()`
at startup and register an `atexit` hook that logs surviving
descendants +
`tests/python/test_orphan_tracker.py` +
`results/experiment_<N>_orphan_tracker.json`.

**What it does:**

The root cause from the memo: when Claude Code launches a long
process via `bash -c`, the bash exits as soon as it's done writing
the command line; the python child reparents to init. If the Claude
session itself ends, nothing kills these python children.

The conductor cannot fix Claude Code's orphan behaviour, but it
can:

1. Maintain a `~/.cache/carnot/orphans.json` registry recording
   `(pid, pgid, start_time, command, conductor_pid)` for every
   subagent it spawns.
2. At conductor startup, `claim_orphans()` scans the registry,
   filters to PIDs whose conductor_pid is no longer running and
   whose pgid is still live (i.e. orphaned), and offers an option:
   `--reap-orphans` flag → SIGKILL the lot; default → log and
   continue (operator's call).
3. `atexit` hook on the conductor: on graceful shutdown, log the
   list of subagent pgids that are still live, so the next
   conductor startup knows what to claim.

**Acceptance:** kill the conductor with SIGKILL (simulating Claude
session crash). Restart with `--reap-orphans`. All previously-
spawned subagent process groups are dead within 30 sec. Without
`--reap-orphans`, the conductor logs the orphans it detected and
the operator can decide.

## Non-goals

- **Not building a process supervisor.** The conductor stays the
  single source of truth; we are not introducing supervisord /
  systemd unit files / PM2 / etc. The proposal uses
  `systemd-run --user --scope` for transient one-shot isolation,
  not for daemon supervision.
- **Not changing the experiment-failure semantics.** The conductor's
  `MAX_FAILURES_PER_TASK = 3`, `CARNOT_CONDUCTOR_TIMEOUT_MINUTES = 60`,
  and existing wall-clock reaper all stay. The proposal makes them
  *actually effective* against orphaned descendants — currently they
  kill only the immediate child.

## Decentralization implications

- **Rule 1 (local-first):** unaffected. systemd-run, setsid, flock,
  cgroups are all kernel-/userland-local primitives.
- **Rule 5 (hardware portability):** the systemd-run path is
  Linux-only with user systemd. The setsid + RLIMIT_AS fallback
  works on any POSIX system, including the FPGA-host development
  workstations and CI runners.
- **Rule 7 (no vendor abstractions):** the new modules live in
  `python/carnot/conductor/` (already a conductor-specific
  submodule). No vendor-specific hooks.

## Why this is in change-proposals, not just a code change

Five interacting OS primitives (process groups, flock, systemd-user
scopes, cgroup memory limits, atexit hooks) need to land together to
fix the failure mode. Documenting the *interaction* — particularly
how `--reap-orphans` interacts with the existing exclusion-manifest
flow and the failure-ledger discipline — matters more than any
single diff.

The proposal is also the locus where the future "what to do when
the conductor itself crashes" runbook lives. The incident memo at
`/tmp/memory-fix.txt` is ephemeral; the proposal preserves the
findings durably.

## Risks

- **systemd-run --user requires user systemd instance.** On hosts
  where it's not present, the cgroup-memory ceiling silently
  degrades to "no ceiling." Mitigation: `_systemd_available()`
  check at startup logs a clear warning and the operator can
  enable user-systemd (one-shot setup) or accept the looser
  guarantee.
- **flock on /tmp survives reboots only if /tmp persists.** On
  CachyOS, /tmp is tmpfs by default — flock state is wiped on
  reboot, which is the desired behaviour (a reboot resets the
  "is anything running?" question).
- **Process-group kill can race with normal subagent exit.** The
  10-sec grace before SIGKILL is the standard mitigation; if the
  subagent has already exited, `os.killpg` raises `ProcessLookupError`
  which we catch silently.
- **`claim_orphans()` could kill processes the operator wants to
  preserve.** Default is "log only" — `--reap-orphans` is opt-in,
  printed as a separate flag, so no surprise SIGKILLs.

## Acceptance criteria (overall)

1. After this proposal lands and the conductor restarts under it,
   the host can run a full milestone without producing a
   memory-pressure incident even when Claude Code sessions are
   crashing/restarting around it.
2. Live-zombie count and live-orphan count for each milestone are
   bounded (say: zombies < 20, orphans < 5) and logged in the
   operational retrospective alongside wall-time and slowest-5.
3. A re-run of the .73 Exp 942 wall-clock-timeout scenario kills
   the entire process group on timeout — no swap leak.
4. Restarting the conductor while an experiment is in flight cleans
   up the prior subagent's whole descendant tree within 30 sec
   (Exp E `--reap-orphans` path).
5. Two concurrent attempts to run the same experiment script
   produce one running instance and one logged "skipped — single-
   run guard" (Exp B).
