# Conductor OpenTelemetry tracing + Victoria Trace backend

**Status:** Draft change proposal.
**Origin:** 2026-04-27 root-cause analysis. The supervisor proposal
  (`conductor-supervisor.md`) answers "is the conductor alive and
  doing what it claims" — deterministic, small, 300 LOC. This proposal
  answers the orthogonal question: "**what is the conductor actually
  doing in detail**, across iterations and across Claude sessions?"
  Tonight's incidents would each have been root-caused in seconds
  with proper tracing instead of forensically reconstructed from log
  fragments and pgrep output.
**Target milestone:** 2026.04.76 — after the supervisor lands in .75.
  The supervisor watches trace-export health, so OTel needs a working
  supervisor to be reliable.
**Priority:** Medium-high. Without OTel, every new conductor failure
  mode requires the same forensic dance we did tonight (read log
  fragments, infer state from pgrep, reconstruct timing). With OTel,
  any operator can answer "what happened during Exp NNN at 04:42 UTC"
  with a single trace query.
**Depends on:**
  - `conductor-supervisor.md` — supervisor's heartbeat/watchdog covers
    the "trace export itself fails" failure mode. Without that, OTel
    becomes a new single point of failure.
  - `conductor-process-isolation.md` — process-group ownership
    (Exp A) gives us a clean place to attach trace context to a
    subagent's whole descendant tree.

## Summary

OpenTelemetry instrumentation across the conductor's iteration loop +
each subagent invocation + each git operation, with traces exported
to a locally-hosted Victoria Trace instance (`victoriametrics/victoria-logs`
container with the `victoria-trace` add-on, or `vlogs` directly).

Every conductor iteration becomes a parent span. Each child operation
(load_research_tasks, pre_flight_gpu_reap, run_tests, run_agent,
git_commit_and_push, in_process_doc_recon) is a child span with
explicit timing and tagged attributes. Subagent invocations
(`claude -p ...`) get their own trace ID propagated via env var, so
the Sonnet's internal pytest invocation can be correlated with the
conductor iteration that spawned it.

Result: the seven incidents tonight each become a one-line trace query.

## Where this would have helped tonight (concrete)

| Tonight's incident | Trace query that would have surfaced it instantly |
|---|---|
| Broken log handle (4× this session) | OTel spans continue to export via OTLP even when stdout severs. The trace shows iterations 16, 17, 18 firing; the absence in the log is then known to be a log-channel issue, not a conductor wedge. |
| .74 schema mismatch | `span.load_research_tasks` has `task_count=0` — drop from 11 → 0 visible in a single histogram. |
| Wall-clock burn on Exp 942 | Spans inside the subagent show `gguf_model_load=45min`, `inference=10min`, `commit=0` — the bottleneck is identifiable without re-running the experiment. |
| Orphan accumulation | Each `subprocess.Popen` span has `pid` and `pgid` attributes; orphans are spans whose parent span ended but whose `pid` is still alive in the next iteration. |
| Pre-commit overhead | Per-hook spans (gitleaks, ruff, mypy, pytest, spec-coverage) show pytest is 95% of commit time vs 30 sec for the others. |
| Multi-Claude-session interactions | Each `claude -p` invocation propagates a trace ID via `OTEL_RESOURCE_ATTRIBUTES`; descendants are queryable by trace, even across different Claude sessions. |
| "Deployment theater" detection | Span `DualGPURunner.benchmark()` only fires from one call site (`Exp 932`); absence in `Exp NNN` iterations of `verify_repair` proves it's not on the hot path. |

## What this proposal IS NOT

- **Not a replacement for the supervisor.** Supervisor is the
  deterministic 300-LOC watchdog that catches the "is the conductor
  even alive" case. OTel is the rich observability layer for "what
  is it doing in detail." Both are needed.
- **Not vendor-locked.** OTel is the open standard. Victoria Trace
  is one of several OTLP-compatible backends; the proposal codifies
  Victoria Trace as the default but the export path is swappable
  (Jaeger, Tempo, Grafana Cloud Tempo, even file-based JSONL with
  `OTEL_EXPORTER_OTLP_ENDPOINT=file://...`).
- **Not a full distributed-tracing story.** The conductor is a
  single process tree on a single host; we don't need cross-service
  propagation primitives. The "distributed" part is just
  cross-Claude-session and cross-subprocess.
- **Not coupled to LLM observability frameworks** (Langfuse,
  Phoenix, Helicone, etc.). Those instrument LLM call/response;
  this instruments the conductor's process control flow. Both
  tracing layers can co-exist; this proposal only covers the
  process-control layer.

## Proposed experiments

### Exp A — OTel SDK integration + Victoria Trace local instance

**Deliverable:**
edits to `pyproject.toml` to add `opentelemetry-sdk` +
`opentelemetry-exporter-otlp` as `[tracing]` extras +
new `python/carnot/conductor/tracing.py` with the global
TracerProvider setup +
`docker-compose.tracing.yml` for the Victoria Trace local instance +
`tests/python/test_tracing_init.py` +
`results/experiment_<N>_otel_init.json`.

**What it does:**

Bring up the tracing infrastructure without any actual instrumentation
of conductor code. The goal is verifying:

1. `opentelemetry-sdk` imports cleanly under Python 3.14 with the
   project's existing JAX/torch/PyO3 stack.
2. `docker compose -f docker-compose.tracing.yml up` brings up
   Victoria Trace at `localhost:9428` (default port).
3. `python/carnot/conductor/tracing.py:setup_tracing()` configures
   a global TracerProvider with OTLP HTTP exporter pointing at
   Victoria Trace, plus a console exporter for debugging.
4. A minimal "Hello span" test fires and is queryable in the
   Victoria Trace UI.

The setup is **opt-in via env var**: `CARNOT_TRACING=1` enables
tracing; default is off so the conductor's existing performance
profile is unaffected for users who don't enable it.

**Acceptance:**
  - `docker compose -f docker-compose.tracing.yml up -d` brings up
    Victoria Trace.
  - `CARNOT_TRACING=1 python -c "from carnot.conductor.tracing import
    setup_tracing; setup_tracing(); ..."` produces a trace visible at
    `http://localhost:9428` within 5 sec.
  - With `CARNOT_TRACING` unset, the import + setup is a near-zero-cost
    no-op (`SimpleSpanProcessor` with no exporter).

### Exp B — Conductor iteration-loop instrumentation

**Deliverable:**
edits to `scripts/research_conductor.py` to wrap each iteration's
key operations in spans +
`tests/python/test_conductor_tracing_iteration.py` +
`results/experiment_<N>_otel_iteration.json`.

**What it does:**

Instrument the conductor's iteration loop with 8 nested spans per
iteration:

```
iteration_<N>
├── load_research_tasks            [count=11]
├── preflight_gpu_reap             [reaped=1, vram_freed=1866MiB]
├── host_health_check              [status=OK, swap_gb=40]
├── pick_next_task                 [task_id=exp953-..., milestone=2026.04.74]
├── pre_test_suite                 [count=81 passed]
├── run_agent                      [model=sonnet, max_turns=30]
│   ├── claude_p_subprocess        [pid=1933292, pgid=...]
│   └── (subagent spans propagate via OTEL_RESOURCE_ATTRIBUTES)
├── git_commit_and_push            [files_changed=4, push_remotes=2]
└── doc_reconciliation             [verdict=preflight_complete, label=Complete]
```

Each span carries timing + relevant attributes. The
`run_agent` span propagates `traceparent` to the spawned `claude -p`
subprocess via env var, so the subagent's pytest / experiment-script
spans (Exp C) link back to their parent iteration.

**Acceptance:** running 5 conductor iterations produces 5 parent
spans in Victoria Trace, each with the expected child structure.
Total spans per iteration: ~12 (plus child subagent spans). Storage:
~50 KB per iteration → ~7 MB per milestone (.74 had 12 iterations,
some retried).

### Exp C — Experiment-script + subagent instrumentation

**Deliverable:**
edits to `scripts/experiment_template.py` to read `OTEL_RESOURCE_ATTRIBUTES`
on startup and continue the parent iteration's trace +
spans for the canonical experiment phases (setup, gpu_warm,
batched_inference, build_result) +
edits to `python/carnot/pipeline/three_tier_pipeline.py` and
related core paths +
`tests/python/test_experiment_template_tracing.py`.

**What it does:**

Once the parent iteration's `traceparent` is set in the subagent's
env, every experiment-script span automatically inherits it. The
experiment_template wraps its standard phases:

```
experiment_<NNN>
├── setup_directories
├── setup_gpu                      [models=[Qwen3.6-35B-A3B], healthy=true]
├── batched_inference              [n_questions=25, batch_size=8]
├── verify_repair                  [n_violations=14]
└── build_result                   [verdict=preflight_complete]
```

This is where most of the *actual* time goes — the iteration-loop
spans show "spent 28 min in run_agent"; the experiment spans show
*why* (e.g., `gguf_model_load=15min, inference_loop=12min, commit=1min`).

**Acceptance:** an Exp 942-shaped experiment with `CARNOT_TRACING=1`
produces a full trace from `iteration_N` → `run_agent` →
`experiment_942` → 4-5 phase spans, each timed. The wall-clock
distribution that took ~3 hours to forensically reconstruct tonight
is queryable in seconds.

### Exp D — Supervisor integration with trace-export health

**Deliverable:**
edits to `python/carnot/supervisor/heartbeat.py` (from
`conductor-supervisor.md` Exp A) to also check trace-export health +
`tests/python/test_supervisor_trace_health.py`.

**What it does:**

The supervisor polls Victoria Trace at `http://localhost:9428/metrics`
every 60 sec. If trace export has stalled (no new spans in last 30
min while conductor is alive and iterating), the supervisor logs
a `trace_export_stalled` incident. This is the "OTel itself failing"
failure mode — same shape as the broken log-handle pattern.

The supervisor does NOT auto-recover trace export — it logs +
PushNotifies. Trace recovery is operator-driven (likely
`docker compose restart` on the Victoria Trace container, which
the operator does deliberately rather than the supervisor doing
automatically).

**Acceptance:** stop the Victoria Trace container; verify supervisor
catches the export stall within 30 min and emits a clear incident
to its durable log.

### Exp E — Trace storage retention + decentralization mirroring

**Deliverable:**
config block in `docker-compose.tracing.yml` for retention rules +
documentation in `docs/observability.md` for trace export to a
secondary OTLP endpoint (per decentralization rule 3 — distribution
mirroring) +
`results/experiment_<N>_trace_retention.json`.

**What it does:**

Victoria Trace stores spans on local disk by default. Configure:

  - **Local retention:** 30 days for full traces, 90 days for
    aggregate metrics (per-iteration timing distributions).
  - **Optional secondary export:** the OTel SDK supports a
    `MultiSpanProcessor`. If `CARNOT_TRACING_MIRROR=<url>` is set,
    spans get exported to both local Victoria Trace and the
    operator-chosen secondary (e.g., a Tempo instance, a Grafana
    Cloud account, or another Victoria Trace running on a backup
    host). Per decentralization rule 3, distribution mirroring is
    encouraged but not required for traces.
  - **Local disk usage cap:** 5 GB for traces (rotates oldest first).
    At ~7 MB per milestone, that's 700+ milestones of history.

**Acceptance:** retention policy is enforced by Victoria Trace's
built-in rules; documented in `docs/observability.md`. The mirror
path is exercised in tests (using a second local Victoria Trace
instance as the mirror).

## Decentralization implications

- **Rule 1 (local-first):** Victoria Trace runs locally in Docker;
  no cloud dependency. The default stack is fully local.
- **Rule 3 (distribution mirroring):** Exp E adds optional
  trace-export mirroring to a secondary endpoint. Aligned with
  the rule but opt-in (mirroring is encouraged for HF-publishable
  artifacts, less critical for ephemeral traces).
- **Rule 5 (hardware portability):** OTel SDK works on any platform
  Python supports. Victoria Trace runs on x86_64 + ARM64 Linux +
  macOS Docker Desktop. No GPU dependency.
- **Rule 7 (no vendor abstractions in core):** OTel SDK is in
  `python/carnot/conductor/tracing.py` (conductor-specific
  submodule). The core verifier stack does not import OTel directly;
  if tracing is added there in the future, it's via an explicit
  hook, not as a runtime dependency.

## Why this is in change-proposals, not just a code change

OpenTelemetry instrumentation is invasive — it touches every
subprocess invocation, every git call, every Sonnet spawn. The
proposal is the locus where the instrumentation contract lives:
*which* spans, *which* attributes, *which* trace ID propagation
patterns. Without that contract written down, future Claude-on-
this-project will instrument inconsistently and the value of the
trace data degrades.

The Victoria Trace choice is also deliberate: lightweight (~50 MB
RAM), fast queries, OTLP-native, single-binary deployable, MIT-
licensed. Switching to Jaeger / Tempo / Grafana Cloud later is a
config change, not a re-instrumentation.

## Risks

- **Performance overhead.** OTel SDK adds ~5-10% per-iteration
  overhead in heavy-instrumentation scenarios. Mitigation: all
  tracing is gated by `CARNOT_TRACING=1`; default off. The
  `BatchSpanProcessor` defers export to a background thread, so
  the conductor's hot path sees minimal latency added.
- **Disk usage.** ~7 MB per milestone × 700 milestones = 4.9 GB.
  Within the 5 GB cap. If exceeded, oldest traces drop. Operators
  can bump to 50 GB cap + 10-year retention if they want full
  project history.
- **Trace export itself can fail silently.** Mitigation: Exp D's
  supervisor integration. The supervisor watches export health
  and surfaces stalls.
- **Scope creep into LLM observability.** The proposal explicitly
  excludes LLM call/response capture (Langfuse / Phoenix / etc.
  are dedicated for that). Adding LLM-content spans here would
  bloat retention and conflate two concerns.
- **Pyo3 / JAX interaction with OTel.** OTel's context-propagation
  uses `contextvars`, which can interact poorly with multi-threaded
  JAX dispatch. Mitigation: instrumentation lives in the conductor
  layer; the JAX hot path is not directly instrumented.

## Acceptance criteria (overall)

1. With `CARNOT_TRACING=1` enabled, every conductor iteration
   produces a complete trace in Victoria Trace within 60 sec.
2. Subagent (Sonnet) trace IDs propagate cleanly to experiment-
   script spans, so a `traceID:<id>` query in Victoria Trace shows
   the full call tree from iteration → run_agent → claude_p →
   experiment_phase.
3. The seven incidents from the 2026-04-27 RCA each become
   queryable in <30 sec via single Victoria Trace queries.
4. Default off (`CARNOT_TRACING` unset) → near-zero overhead;
   conductor wall-time within 1% of pre-instrumentation baseline.
5. The supervisor (`conductor-supervisor.md`) catches trace-export
   stalls within 30 min.
6. Documentation in `docs/observability.md` explains how to bring
   up the local Victoria Trace stack, how to enable tracing, and
   how to query the trace data for the seven canonical incident
   shapes.
