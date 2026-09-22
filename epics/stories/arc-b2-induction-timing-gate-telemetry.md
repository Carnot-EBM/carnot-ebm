# ARC B2 induction-timing gate telemetry

**Status:** Complete, feasibility only
**Requirement:** REQ-ARC-WMTE-7530
**Experiments:** 7530, 7531

## Goal

Join each fired induction decision to its generation cost, verifier outcome,
and bounded later progress. Measure whether an induction-timing gate has
headroom. Do not change the existing induction rule or ship a gate.

## Stage 1

The real E3 decision recorder now assigns a stable attempt ID when induction
fires. It joins prompt and completion tokens, wall time, plan state, verifier
result, and frame-change or level-up progress within 32 policy actions. The
window covers short plans and immediate fallback while avoiding credit for
unrelated late exploration. Episode end records a censored row instead of
dropping it. The recorder stays off by default.

The focused CPU suite passed 36 tests. On/off parity preserves actions,
provenance, model calls, environment calls, and random state. Ruff, Ruff
format, mypy, scoped spec coverage, and the paired test-mutation guard passed.
Experiment 7530 records this proof.

## Stage 2

Experiment 7531 reused the frozen E6 12-game panel. It used the current
Qwen3.8-27B Q4_K_M model, no per-game adapters, and
`solve_provenance=live_agent_self_discovery`. The corrected idle check ran
before CUDA initialization. GPU 1 offload reached 18,030 MiB. GPU 0 was not
touched.

The run completed 56 episodes in 10,566.096 seconds. It observed 20,045 gate
opportunities and 60 fired induction attempts. The opportunity floor passed.
The 100-attempt floor did not. The result is feasibility only.

All 60 attempts had progress within the 32-action window. The analysis-only
oracle kept all 60 attempts, suppressed none, and saved zero completion
tokens. It found no headroom in this corpus. No numeric gate-quality or
ship-readiness claim is made.

Both terminal artifacts pass adversarial verification with zero flags. The
known action-vocabulary warning remains a runtime risk, but it did not stop
the measurement.
