# Research Roadmap v235: Codex Recovery Sprint v2 + AUROC Ceiling Assault v4

**Milestone:** 2026.05.235
**Status:** PROPOSED
**Date:** 2026-05-18
**Previous milestone:** 2026.05.234 (AUROC Ceiling Assault + Phase 1 Ship Gate) — 0/13 tasks completed

---

## What .234 Proved

Milestone .234 activated and immediately failed with the same Codex CLI transient backend
error pattern that collapsed .232: ALL 13 non-retro tasks failed with "Codex CLI error:
you finish the real work inside 10 minutes, that is correct" on every retry. The retro
(exp2419) succeeded because it only scans for missing artifacts — no complex tool calls.

**Root cause (confirmed via exp2393 pattern matching):** The Codex CLI backend is
experiencing recurring transient failures. In .232, exp2393 (requires_claude) diagnosed
these as transient OpenAI backend errors that "resolved naturally." However, the same
pattern has recurred in .234. This suggests the issue is NOT a one-time event but a
recurring Codex backend reliability problem that may be capacity-related (peaks, quotas,
or rate limits on the OpenAI side).

**State entering .235:**
- Best AUROC: FregeLogic=0.8831 (from .233, exp2395) — still holds
- HIVE peer ceiling: 0.9236 (gap = 0.0405)
- FR-11 NSVIF online learning: NEVER completed (4 consecutive failures)
- Phase 1 ship gate: NEVER audited (3 consecutive failures)
- KV260 Yosys synthesis: NEVER completed (2 consecutive failures)
- New samplers (Kinetic Langevin, Dikin-Langevin, DE-PSGLD): NEVER implemented

---

## Architecture Diagram

```
Phase 0: Admin + Infra
┌─────────────────────┐   ┌──────────────────────────────────────────────┐
│ exp2420             │   │ exp2421 (requires_claude: true)              │
│ Archive .234 +      │   │ Codex CLI Diagnostic v2                      │
│ Activate .235       │   │ Tests simple+medium+complex Codex tasks      │
│ (codex, ungated)    │   │ → codex_cli_healthy: bool                    │
└─────────────────────┘   └──────────────┬───────────────────────────────┘
                                         │ gated_on: codex_cli_healthy==true
                          ┌──────────────▼───────────────────────────────┐
Phase 1: AUROC Assault    │  exp2422  exp2423  exp2424                   │
                          │  HIVE v4  LogCons  HALT-RAG                  │
                          │  (codex)  (codex)  (codex)                   │
                          └──────────────┬───────────────────────────────┘
                                         │
Phase 2: FR-11 + FST      ┌──────────────▼───────────────────────────────┐
                          │  exp2425 (FR-11 CSL)   exp2426 (FST MCMC)    │
                          │  (codex, gated)        (codex, gated)         │
                          └──────────────┬───────────────────────────────┘
                                         │
Phase 3: HW + Samplers    ┌──────────────▼───────────────────────────────┐
                          │ exp2427  exp2428  exp2429  exp2430  exp2431   │
                          │ Yosys    Kinetic  Dikin    DE-PSGLD ShipGate │
                          │ (codex)  (codex)  (codex)  (codex)  (codex)  │
                          └──────────────┬───────────────────────────────┘
                                         │ gated: exp2422.ensemble_auroc_improved
Phase 4: Synthesis        ┌──────────────▼───────────────────────────────┐
                          │  exp2432 (claude opus)  exp2433 (codex)       │
                          │  Paper-v6 Capstone      .235 Retro            │
                          │  (gated on AUROC)       (ungated)             │
                          └─────────────────────────────────────────────-─┘
```

---

## Three Critical Gaps

### Gap 1: Codex CLI Recurring Backend Failure
**Evidence:** .232 (11/14 FAIL), .233 (partial recovery via exp2393), .234 (13/13 FAIL).
**Root cause hypothesis:** OpenAI Codex backend has intermittent capacity/rate-limit issues
that cause the CLI to echo its stop-when-done postamble without executing the task.
**Fix:** Gate ALL codex tasks on exp2421 (Codex diagnostic v2, requires_claude: true).
If Codex is still broken, gate-block preserves wall time. If working, all tasks proceed.

### Gap 2: AUROC Gap 0.0405 to HIVE Peer 0.9236
**Evidence:** Best AUROC = FregeLogic 0.8831 (exp2395). HIVE 4-verifier ensemble was
attempted in exp2408 (.234) but failed via Codex CLI error before running.
**Fix:** exp2422 (HIVE v4), exp2423 (Hierarchical LogCons v2), exp2424 (HALT-RAG NLI v2).
New technique from .235 sweep: Falkor-IRAC knowledge-graph grounding (arXiv:2605.14665)
informs the LogCons Z3 encoding with graph-based constraint checking.

### Gap 3: FR-11 NSVIF Online Learning Never Completed
**Evidence:** 4 consecutive failures (exp2383: Codex CLI error in .232; exp2400:
no_artifact_session_ended in .233; exp2411: Codex CLI error x3 in .234).
**Fix:** exp2425, gated on exp2421. If Codex works, this WILL run (it's 4th in the queue).

---

## Phase Descriptions

### Phase 0: Administration + Infrastructure
- **exp2420** — Archive milestone .234 to research-complete.yaml, activate .235
- **exp2421** — Codex CLI Diagnostic v2 (requires_claude: true). Runs three Codex tasks
  of increasing complexity: (1) trivial math, (2) file read + JSON write, (3) multi-file
  Python edit. Records pass/fail at each complexity level. Produces codex_cli_healthy
  boolean AND complexity_threshold (max complexity level Codex handles reliably).
  This avoids burning 40+ minutes of retry wall time if Codex is still broken.

### Phase 1: AUROC Ceiling Assault (gated on exp2421.codex_cli_healthy)
- **exp2422** — HIVE Full 4-Verifier Ensemble v4. Fuses Tier 0f (FreqAwareAttn,
  AUROC=0.7045), 0g (SemanticEnergy, 0.6852), 0h (LaaB), 0j (HALT, 0.8539) with
  calibrated soft-vote weights. Target: push past HIVE peer 0.9236.
- **exp2423** — Hierarchical LogCons v2. Z3 instruction-hierarchy partial order with
  TruncProof grammar grounding (arXiv:2605.13076) as a structural validity pre-check.
  Baseline: FregeLogic 0.8831.
- **exp2424** — HALT-RAG NLI Ensemble v2. 3-proxy NLI signals with calibrated abstention;
  abstention threshold tuned based on exp2410 (if it produced data) or a sweep.

### Phase 2: Continuous Self-Learning + FST (gated on exp2421)
- **exp2425** — FR-11 NSVIF Online Self-Learning v4 (MANDATORY CSL task). Verify→feedback
  →update loop on 20 telemetry entries. This is the 4th attempt; gating on exp2421 ensures
  it only runs when Codex is confirmed healthy.
- **exp2426** — FST Constrained MCMC v2. MH-acceptance filter over FST pipeline using
  Carnot Ising energy as the acceptance criterion.

### Phase 3: Hardware + Sampler Track (gated on exp2421)
- **exp2427** — KV260 Yosys Synthesis v4. RTL is lint-clean since exp2372; this is the
  3rd attempt at Yosys synthesis; both prior attempts failed via Codex CLI error.
- **exp2428** — Kinetic Langevin BAOAB vs CASAL v4. Underdamped Langevin with BAOAB
  splitting on 4x4 ferromagnetic Ising (N=16).
- **exp2429** — Dikin-Langevin Polytope Sampler v2. Dikin-metric Langevin on box-
  constrained Ising [-1,+1]^N.
- **exp2430** — DE-PSGLD v2. Decentralized Proximal SGLD (arXiv:2605.00723) vs CASAL.
- **exp2431** — Phase 1 Ship Gate v4. PyPI carnot-ebm, HF mirror, MCP/CLI docs, CI reproducer.

### Phase 4: Synthesis (Phase 5)
- **exp2432** — Paper-v6 Capstone v235 (claude opus, requires_claude). Synthesizes all
  .235 AUROC results; updates ops/status.md. Gated on exp2422.ensemble_auroc_improved==true.
- **exp2433** — Milestone .235 Operational Retrospective (codex, ungated).

---

## Dependency Graph

```
exp2420 (archive)   → unblocks: research-complete.yaml updated
exp2421 (diagnostic, requires_claude)  → gate for all exp2422-exp2431

exp2421 → exp2422 (HIVE v4)
exp2421 → exp2423 (LogCons v2)
exp2421 → exp2424 (HALT-RAG NLI v2)
exp2421 → exp2425 (FR-11 v4, CSL mandatory)
exp2421 → exp2426 (FST MCMC v2)
exp2421 → exp2427 (Yosys v4)
exp2421 → exp2428 (Kinetic Langevin v4)
exp2421 → exp2429 (Dikin-Langevin v2)
exp2421 → exp2430 (DE-PSGLD v2)
exp2421 → exp2431 (Ship Gate v4)

exp2422.ensemble_auroc_improved → exp2432 (Capstone, gated)
exp2433 (retro, ungated)
```

---

## Hardware Requirements

| Task | Hardware | Notes |
|------|----------|-------|
| exp2421 | CPU | Codex diagnostic via Claude |
| exp2422-exp2425 | CPU | No GGUF inference required (telemetry-based) |
| exp2426 | CPU | Cached telemetry, no live GGUF |
| exp2427 | CPU | Yosys synthesis only |
| exp2428-exp2430 | CPU | Sampler benchmarks on N=16 Ising |
| exp2431 | CPU+network | curl checks to PyPI/HF |
| exp2432 | CPU | Claude reads artifacts |

No live GPU inference required. All tasks are CPU-bound or network-bound.

---

## Key Papers (this milestone sweep)

| arXiv | Title | Hook |
|-------|-------|------|
| 2605.14665 | Falkor-IRAC graph-constrained generation | LogCons Z3 graph grounding |
| 2605.13076 | TruncProof grammar-constrained JSON | Structural pre-check for NSVIF extraction |
| 2605.10065 | NCO negative constraints in decoding | Negative energy constraint track |
| 2605.09927 | JSON-Schema guided LLM pipeline | Schema-constrained extraction |
| 2604.26139 | HIVE soft-voting ensemble (0.9236) | Target ceiling |
| 2604.18328 | FregeLogic Z3+neural hybrid (0.8831) | Current best baseline |
| 2509.07475 | HALT-RAG calibrated NLI abstention | Abstention mechanism |
| 2601.14210 | HALT latent probe Tier 0j (0.8539) | HIVE component |
| 2506.05754 | Constrained Sampling MCMC (MH) | FST MCMC track |
| 2510.04582 | Dikin-Langevin polytope sampler | Sampler track |
| 2605.00723 | DE-PSGLD proximal SGLD | Sampler track |
| 2603.23397 | Kinetic Langevin BAOAB splitting | Sampler track |

---

## Success Criteria

| Criterion | Target | Gate |
|-----------|--------|------|
| Codex CLI health | codex_cli_healthy == true | Unblocks all Phase 1-3 |
| AUROC improvement | HIVE v4 > 0.8831 (any improvement) | exp2432 capstone |
| AUROC ceiling | Any verifier AUROC > 0.9236 | Paper-v6 headline |
| FR-11 compliance | cross_domain_retention_rate >= 0.50 | Mandatory per CLAUDE.md |
| Hardware track | synthesis_succeeded == true | KV260 track milestone |
| Phase 1 ship | phase1_ship_gate_met | Publication unlock |

---

## Agent Routing

| Task | Agent | Justification |
|------|-------|---------------|
| exp2420 | codex gpt-5.5 | Simple archive task, proven pattern |
| exp2421 | claude (requires_claude) | Diagnosing Codex CLI; Claude must run Codex |
| exp2422-exp2431 | codex gpt-5.5 | Codex-Default rule; gated on health check |
| exp2432 | claude opus | requires_claude: 12+ artifact synthesis, multi-step reasoning |
| exp2433 | codex gpt-5.5 | Templated retro, no judgment calls |

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml`. Retired experiments: 2091 (gemini CLI bail-out),
260, 308, 309, 346, 380, 381, 382, 383, 410, 425, 491, 527, 603, 627, HalluSAEGeometricProbe,
887, 783, 799, 804, 809, 825 (discriminative JEPA OOD failures).

None of the .235 tasks match retired scopes:
- HIVE ensemble: new scope (4-verifier fusion, not retired)
- NSVIF online learning: not retired (Codex CLI infrastructure failures, not scope failures)
- KV260 Yosys: not retired (infrastructure failures, scope remains valid)
- Samplers: Kinetic Langevin/Dikin/DE-PSGLD — none retired
- Ship gate: not retired (infrastructure failures)

Cross-check: PASSED.
