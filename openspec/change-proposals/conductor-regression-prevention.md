# Conductor regression prevention — harden the autoresearch harness

Status: Draft change proposal. Origin: 2026-04-18 session retrospective. Target milestone: 2026.04.35 (next milestone after the one actively planning).

## Why this exists

Milestone 2026.04.33 lost ~7.5 hours to a single off-by-whitespace bug in `scripts/research_conductor.py::pick_next_task()` (title prefix not stripped, so fail counts never matched) and another ~3 hours to 60-minute wall-clock kills of subagents that had already produced their deliverables. Both bugs were silent: the conductor kept reporting "running" and the user only noticed because the same task kept retrying.

The research code itself was fine — the orchestration harness was the single point of failure. If we don't harden the harness now, the next long-autonomous run will hit a similar class of bug and burn another multi-hour window. This proposal captures four concrete experiments the planner should schedule into 2026.04.35 or 2026.04.36.

These are infrastructure experiments (CPU-only, no GPU, no Anthropic-API cost) so they fit the "cheap, dependency-free" slot the planner already allocates for retro closure work.

## Experiment proposals

Pick at least two per milestone until all four land. The first two are tiny and should be scheduled together.

### Exp N: Regression tests for conductor pick_next_task + run_agent

- **Deliverable:** `results/experiment_<N>_conductor_regression_tests.json` plus `tests/python/test_research_conductor_regressions.py`.
- **Why:** Both bugs that hurt us are 20-line changes with no unit test covering them. A regression test pins each fix so no future refactor silently reintroduces the bug.
- **Scope:**
  - `test_pick_next_task_title_prefix_whitespace_boundary` — construct a YAML task whose `title[:50]` ends on a space character; log 3 FAIL entries for that title via `log_step`; assert `pick_next_task` returns `None` (skipped) rather than returning the same task a fourth time. This is the exact Exp 447 bug.
  - `test_pick_next_task_counts_wall_clock_timeouts_as_failures` — synthesize a log where the only FAIL entries are "Wall-clock timeout after 3603s"; assert the task is skipped after the third entry. Wall-clock timeouts had historically not incremented the fail counter consistently; the contract matters.
  - `test_run_agent_deliverable_watch_kills_subagent_early` — use a trivial shell-script fake agent (not Claude Code) that `touch`es the deliverable then `sleep 999`; assert `run_agent(deliverable_path=...)` returns `(True, ...)` inside DELIVERABLE_STABLE_SECS + a few seconds, not after 60 minutes.
  - `test_run_agent_deliverable_watch_ignores_changing_file` — fake agent that rewrites the deliverable every 10 s; assert `run_agent` does NOT short-circuit (the stability window is what makes it safe).
- **Scale:** 4 unit tests, all CPU-only, all under 30 s runtime. Schema: `carnot.conductor_regression.v1` with fields `tests_run`, `tests_passed`, `honest_verdict in {all_pass, partial_pass, all_fail}`.
- **Size:** 1–2 hour subagent budget. Does not need the SOTA GGUF model set.

### Exp N+1: Startup invariant check for roadmap/log key round-trip

- **Deliverable:** `results/experiment_<N+1>_conductor_startup_invariants.json` plus a new function `scripts/research_conductor.py::assert_startup_invariants()` called once at `main()` entry.
- **Why:** The fail-counter bug only manifested because one specific task's title hit a specific boundary. A 20-line check run once at conductor boot would have caught it in the first second of the first run, turning a 7.5-hour silent loss into a config error before the loop starts.
- **Scope:** At startup, for every task in the active roadmap YAML, simulate `log_step(task["title"], "FAIL")` → parse back via the same logic `pick_next_task` uses → assert the task's title-prefix key matches a FAIL entry it just wrote. If any task fails the round-trip, log the offending title and `sys.exit(2)` with a clear diagnostic (don't try to repair). Belt-and-braces: also check that every task has a non-empty `deliverable`, that no two tasks share a deliverable path, and that every `blocked_by` reference resolves to a task in the roadmap.
- **Scale:** ~40 lines of code, ~4 tests. Tests must include a "poisoned" roadmap fixture that deliberately hits the boundary bug, asserting the check rejects it.
- **Size:** 1 hour. Adds one invariant-check log line per conductor startup.

### Exp N+2: Structured event log for conductor observability

- **Deliverable:** `results/experiment_<N+2>_conductor_events.json` plus `ops/conductor-events.jsonl` as a new live output and a tiny `scripts/conductor_health.py` CLI.
- **Why:** Today the only way to answer "is the conductor making progress?" is to grep `/tmp/conductor.log` and eyeball it. In the 2026-04-18 session the user had to manually notice that two consecutive iterations both hit "Exp 447 timeout" with no fail-counter movement — a structured signal would have surfaced this automatically.
- **Scope:**
  - Emit one JSONL event per major state transition: `iter_start`, `task_picked`, `subagent_start`, `deliverable_stable`, `subagent_killed {reason}`, `tests_passed`, `tests_failed`, `commit_made`, `iter_end {status, elapsed_s}`. One line per event, schema-stable.
  - Wrap `log_step` so every OK/FAIL/SKIP also emits a structured event.
  - `conductor_health.py` reads the tail of `conductor-events.jsonl` and reports: iterations in last hour, distribution of end-of-iter statuses, average subagent elapsed, % of runs killed by deliverable-watch vs wall-clock, any task id that has appeared as `task_picked` more than MAX_FAILURES_PER_TASK times (early warning of another fail-counter regression).
  - The cron we use for hourly monitoring can switch from grepping text to parsing the JSONL.
- **Scale:** ~80 lines of code across the conductor + health script. Tests: golden-file check that a canned set of state transitions produces the expected JSONL; `conductor_health.py` correctly flags a stuck-in-loop scenario from a fixture.
- **Size:** 2 hours. Uses the same sonnet model, no GPU.

### Exp N+3 (stretch): Fake-Claude state-machine test harness

- **Deliverable:** `results/experiment_<N+3>_conductor_state_machine_harness.json` plus `tests/python/test_conductor_state_machine.py` and a new `FakeAgentBackend` injectable into `run_agent`.
- **Why:** Items 1–3 are targeted patches. This one is the structural fix: today `research_conductor.py` is only exercised by the live loop. A harness that swaps in a scripted agent lets us drive the conductor through every state transition in milliseconds, which catches bugs that unit tests miss because they come from the state machine, not individual functions.
- **Scope:**
  - `FakeAgentBackend(scripted_outcomes: list[Outcome])` that `run_agent` uses when `CARNOT_FAKE_AGENT=1`. Each `Outcome` is `{exit_code, elapsed_s, files_touched, stdout}`.
  - A pytest-driven roadmap (`tests/fixtures/roadmap_regression_matrix.yaml`) with 10+ tasks covering: immediate success, timeout-then-success, three-fail-skip, max-turns failure, missing deliverable, half-written deliverable (file exists but no schema field), title on 50-char boundary, task with conflicting deliverable path, task with unresolved `blocked_by`, task that succeeds only on 3rd attempt.
  - Each fixture invokes `main(dry_run=False, --interval 0)` in a subprocess with the fake backend and asserts the expected terminal state: which tasks committed, which were skipped, which left a contract_violation. Runs in < 30 s total.
- **Scale:** ~200 lines of test infrastructure + fixtures. Pays for itself the first time it catches a state-machine regression before it ships.
- **Size:** 4–6 hours. Milestone-scale; schedule after N+1 and N+2 have landed so they can be exercised by the harness.

## Rollout / sequencing

- Milestone 2026.04.35: land **Exp N** and **Exp N+1** together — they are small and high-leverage, the planner should scope them as one batch.
- Milestone 2026.04.36: land **Exp N+2** (observability); let it run one full milestone in the background emitting events, then assess.
- Milestone 2026.04.37: land **Exp N+3** (state-machine harness), using the events log and invariant checks from prior milestones as ground truth for the assertions.

## Constraints the planner should respect

- All four experiments are CPU-only. No SOTA GGUF model requirement. They should NOT consume the same roadmap slot as any live verify-repair benchmark — those remain the headline tasks.
- None of these experiments should modify `scripts/research_conductor.py` in a way that changes running behaviour without a feature flag. Exp N+1 and N+2 both add new code paths; keep them idempotent.
- Every deliverable must include the standard `schema`, `honest_verdict`, and `run_date` fields per `scripts/experiment_template.py::REQUIRED_RESULT_FIELDS`.

## References

- Session post-mortem 2026-04-18 — the fail-counter and deliverable-watch bugs discussed in the retrospective conversation with the maintainer.
- `scripts/research_conductor.py` line 578 (`title_prefix = task["title"][:50].strip()`) — site of the Exp 447 fix.
- `scripts/research_conductor.py` around the stream loop in `run_agent` — site of the deliverable-watch addition.
- `openspec/change-proposals/research-roadmap-vNEXT-dspy-signatures.md` — adjacent hardening proposal; Exp N+1's invariant check should later verify DSPy signatures as well once that proposal lands.
