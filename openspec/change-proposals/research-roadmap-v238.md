# Research Roadmap v238: AUROC Ceiling Assault v3 + KV260 RTL Debug + PolarFire Pure-Python Deploy + arXiv Prep

**Milestone:** 2026.05.238
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.237 — 8/12 tasks completed (AUROC=0.9167, GateMate TERMINAL, PolarFire SSH partial, KV260 missing)

---

## What .237 Proved

Milestone .237 had 8 of 12 tasks complete. Key findings:

**Wins:**
- **GateMate TERMINAL STATE MET**: exp2453 confirmed gatemate_bitstream_flashed=True (n=16 Ising tile synthesized + flashed via DirtyJTAG). GateMate drops from mandatory hardware roster.
- **PolarFire SSH reachable**: exp2454 confirmed ssh_reachable=True (uptime 4+ days). Blocked on jaxlib missing from riscv64 wheels.
- **ODAR routing implemented**: exp2455 added OdarRouter to VerifyRepairPipeline (bad verdict format but odar_routing_implemented=True).
- **NCO tautology fixed**: exp2456 improved AUROC from 0.5 (tautology) to 0.678.
- **FR-11 soundness/completeness**: exp2451 implemented asymmetric Littlestone-dimension tracking (soundness_tracking_enabled=True).
- **Phase 1 ship gate confirmed**: exp2457 (capstone) confirmed phase1_ship_gate_met=True with best_auroc_achieved=0.9167.

**Gaps carried forward:**
1. **AUROC ceiling**: conformal ensemble v2 (exp2448) achieved AUROC=0.9167 — identical to v1. Adding PCIB as 8th verifier via Fisher's method did NOT improve the result. Gap to HIVE peer (0.9236) = 0.0069. Fisher's method has hit a ceiling.
2. **KV260 synthesis unfixed**: exp2452 (claude+opus RTL fix) produced NO ARTIFACT — the task never ran. synthesis_errors=1 remains from prior experiments exp2413 + exp2427. This is 3 consecutive misses on a claude+opus task.
3. **PolarFire carnot-ebm install failed**: jaxlib is not available as a prebuilt binary for riscv64; `pip3 install carnot-ebm` fails at the JAX dependency. Pure-Python workaround needed.

**New findings from .237 experiments:**
- HalluField Tier 0m (exp2449): AUROC=0.539 — below baseline. Not useful for ensemble.
- LaaB meta-judgment v2 (exp2450): AUROC=0.854 — useful signal but was NOT fused into exp2448's ensemble (scores in exp2450_laab_meta_scores.json, 36 entries available).

---

## Three Biggest Gaps vs PRD Vision

**Gap 1: AUROC ceiling at 0.9167 (0.0069 from HIVE peer)**
Fisher's conformal p-value aggregation has been tried with 7 and 8 verifiers. Result is identical (0.9167). The aggregation method is the bottleneck, not the number of verifiers. Solution: (a) Stouffer's Z-score with quality-weighted verifier scores, (b) learned ensemble (LogisticRegression on verifier scores), (c) new Tier 0n verifier from arXiv:2604.16217 using internal representation conformal scores.

**Gap 2: KV260 RTL synthesis_errors=1 (blocking hardware track terminal state)**
Three consecutive failures (exp2413, exp2427, exp2452). Root cause: RTL content error in one of 18 .v files. exp2452 (claude+opus) never ran (artifact missing). The conductor likely did not launch it — opus tasks have consistently lower activation rates in the current conductor setup. Solution: use opus for the primary fix task + add an explicit pre-step to capture yosys stderr. Mark retire_if_same_verdict for the third attempt.

**Gap 3: PolarFire carnot-ebm install failure (blocking sovereignty claim)**
jaxlib is not prebuilt for riscv64-linux. The fix: install carnot-ebm with `--no-deps` and manually install numpy-only dependencies. The IsingModel.energy() can run in pure-Python/numpy mode on any platform.

---

## Architecture Snapshot (entering .238)

```
Tier 0 (logit-based verifiers):
  Tier 0g: SemanticEnergy (AUROC=0.810)
  Tier 0h: LaaB (AUROC=0.854 with meta-judgment v2 from exp2450)
  Tier 0i: FregeLogic Z3+neural hybrid (AUROC=0.8831)
  Tier 0j: HALT logit-space latent probe (AUROC=0.8539)
  Tier 0k: DiffuTruth (AUROC=0.588)
  Tier 0l: PCIB (AUROC=0.802)
  Tier 0m: HalluField (AUROC=0.539 — below baseline, excluded from ensemble)
  Tier 0n: [NEW] Internal Representation Conformal (arXiv:2604.16217)
  Tier 0o: [NEW] Qwen Suppressed-Retrieval NLA Probe
  Conformal Ensemble v2 (exp2448): AUROC=0.9167 (8 verifiers, Fisher's method)
  Conformal Ensemble v3 target: AUROC > 0.9236 (Stouffer + quality weights + LaaB meta)

Tier 1 (neuro-symbolic):
  NSVIF Z3 SMT extractor (online soundness/completeness tracking — exp2451)
  FregeLogic Z3+neural hybrid
  VERGE SMT repair

Tier 2 (cross-session memory):
  [NEW .238] Constraint Memory — per-session fact caching (FR-11 Tier 2)

Samplers:
  CASAL (Carnot Adaptive Sampler, Exp 442), Kinetic Langevin, ODAR routing

Hardware:
  GateMate: TERMINAL (gatemate_bitstream_flashed=True) — COMPLETED
  KV260: synthesis_errors=1, not at terminal state — ACTIVE
  PolarFire: ssh_reachable=True, carnot install blocked by jaxlib — ACTIVE

Paper-v6:
  Phase 1 ship gate: MET (exp2441)
  Best AUROC: 0.9167 (gap -0.0069 vs HIVE peer)
  arXiv submission: HOLDS per operator directive until Phase 4 empirically validated
    (but paper integrity audit queued for .238)
```

---

## Phase Structure

### Phase 0: Admin (ungated)
- exp2459: Archive .237, activate .238

### Phase 1: AUROC Ceiling Assault v3 (ungated — all can run in parallel)
- exp2460: Tier 0n Internal Representation Conformal (arXiv:2604.16217) — layer-wise logit information scores as nonconformity measure
- exp2461: Conformal Ensemble v3 — Stouffer's Z-score with quality weights + LaaB meta fused (9 verifiers)
- exp2462: Qwen Suppressed-Retrieval NLA Probe Tier 0o (MANDATORY — 2026-05-19 operator note)

### Phase 2: Self-Learning + Research (ungated)
- exp2463: FR-11 Constraint Memory Tier 2 — cross-session fact caching (MANDATORY continuous_self_learning_task)
- exp2464: CRANE Balanced Constraint Integration (arXiv:2502.09061)

### Phase 3: Hardware Continuity (ungated, mandatory per CLAUDE.md)
- exp2465: KV260 RTL Synthesis Fix v6 (claude+opus — 3rd real attempt with stderr capture)
- exp2466: PolarFire Pure-Python Carnot Deploy (carnot-ebm --no-deps + numpy fallback)

### Phase 4: Paper Prep (ungated)
- exp2467: KAN Formal Verification Bounds (arXiv:2602.06737)
- exp2468: Paper-v6 arXiv Pre-Submission Integrity Audit

### Phase 5: Synthesis (gated)
- exp2469: Capstone v238 (claude+opus, gated on exp2461.ensemble_auroc > 0.9167)
- exp2470: Retro v238 (ungated)

---

## Dependency Graph

```
exp2459 → (no downstream gate; archive is procedural)

exp2460 → exp2461 (optional: Tier 0n scores fused if available)
exp2461 → exp2469 (gate: ensemble_auroc_improved_v3 == true)
exp2462 → exp2461 (optional: Tier 0o scores fused if available)

exp2463 → (standalone FR-11 task, no downstream gate)
exp2464 → (standalone CRANE task)
exp2465 → (standalone hardware task)
exp2466 → (standalone hardware task)
exp2467 → (standalone research task)
exp2468 → exp2469 (audit reads paper state before capstone update)
exp2469 → exp2470 (retro reads all artifacts)
```

---

## Hardware Requirements

| Board | Current State | .238 Target | Terminal State |
|---|---|---|---|
| KV260 | synthesis_errors=1 (exp2452 never ran) | synthesis_errors=0 (exp2465) | kv260_synthesis_succeeded=true |
| GateMate | **TERMINAL** — gatemate_bitstream_flashed=True (exp2453) | N/A | Achieved |
| PolarFire | ssh_reachable=True, carnot install failed | carnot_runs_on_polarfire=True (exp2466) | polarfire_workload_validated=true |

---

## Decentralization Check (CLAUDE.md Rules 1-7)

- Rule 1 (Local-first): All verifier experiments use local CPU-only computation (no closed-weight calls). CRANE integration uses locally-cached GGUF models. ✓
- Rule 2 (Closed optional): No closed-weight models are required for any .238 task. ✓
- Rule 3 (Distribution mirroring): Phase 1 ship gate already met; HF + IPFS mirroring noted in exp2441. ✓
- Rule 4 (Multiple integration surfaces): Python API + CLI + MCP server all included in ship gate. ✓
- Rule 5 (Hardware portability): PolarFire deploy (exp2466) is directly sovereignty infrastructure. ✓
- Rule 6 (Data minimization): All .238 experiments use local data (telemetry manifest, yosys). ✓
- Rule 7 (No vendor-specific core): All new verifiers added to python/carnot/verify/ (open protocol). ✓

---

## Exclusion Manifest Cross-Check

Retired patterns that must NOT appear in .238:
- GRPO/VPRM v15 — None proposed ✓
- WOPR puzzle cartridge — None proposed ✓
- HardNet++/DSP — None proposed ✓
- THRML scaling sweep — None proposed ✓
- SpecAnn — None proposed ✓
- iCE40 PIMI — None proposed ✓
- exp2091 (gemini CSL grammar) — Not referenced ✓

All 12 tasks pass exclusion manifest check.

---

## Failed-Experiment Rerun Compliance Table

| Task | Prior Experiment | Prior Verdict | Root Cause | What's Different |
|---|---|---|---|---|
| exp2461 (Ensemble v3) | exp2448 (v2, AUROC=0.9167) | complete: | Fisher's method ceiling | Stouffer's Z-score + LaaB meta (9th verifier) |
| exp2461 (Ensemble v3) | exp2438 (v1, JSON malformed) | complete: | JSON + Fisher ceiling | JSON fixed in v2; now switching aggregation method |
| exp2462 (Qwen NLA) | exp1851 (NLA probe fabrication) | complete: TPR=1.0 (FABRICATED) | No preconditions, GGUF not loaded | CPU-only logprob proxy; no GGUF required; explicit preconditions |
| exp2463 (FR-11 Tier 2) | exp2451 (FR-11 v5, passed) | complete: | N/A (passed) | Tier 2 cross-session memory (distinct from Tier 1 online learning) |
| exp2465 (KV260 v6) | exp2452 (v5, MISSING) | (no artifact) | Task never ran | explicit stderr capture + parse loop before fix attempt |
| exp2465 (KV260 v6) | exp2427 (v4, blocked_synthesis_failed) | blocked | synthesis_errors=1 | stderr captured + error line parsed + fix applied before re-run |
| exp2465 (KV260 v6) | exp2413 (v3, blocked_synthesis_failed) | blocked | synthesis_errors=1 | now three explicit fix-attempt loops; retire_if_same_verdict=true |
| exp2466 (PolarFire v2) | exp2454 (v3, install failed) | complete: | jaxlib missing on riscv64 | pip install --no-deps + numpy-only IsingModel |
| exp2469 (Capstone v238) | exp2457 (capstone v237, complete) | complete: | N/A (different milestone) | Different milestone outcomes; gated on exp2461 improvement |

---

## Agent Routing Table

| Task | Agent | Model | Justification |
|---|---|---|---|
| exp2459 (archive) | codex | gpt-5.5 | Formulaic: cp + yaml append |
| exp2460 (Tier 0n) | codex | gpt-5.5 | New verifier implementation, known pattern |
| exp2461 (Ensemble v3) | codex | gpt-5.5 | Algorithmic: load scores, compute Stouffer, evaluate |
| exp2462 (NLA probe) | codex | gpt-5.5 | CPU-only logprob proxy; no live LLM; formulaic |
| exp2463 (FR-11 Tier 2) | codex | gpt-5.5 | SQLite/JSON cache implementation; known pattern |
| exp2464 (CRANE) | codex | gpt-5.5 | Single-file pipeline addition; balance_ratio parameter |
| exp2465 (KV260 RTL) | claude | opus | requires_claude: ALL 3 positive criteria met (codex failed x3, 18-file RTL debug, open-ended error diagnosis) |
| exp2466 (PolarFire) | codex | gpt-5.5 | SSH + install; formulaic steps |
| exp2467 (KAN formal) | codex | gpt-5.5 | MILP implementation, known bounds |
| exp2468 (arXiv audit) | codex | gpt-5.5 | Document inspection + claim comparison |
| exp2469 (capstone) | claude | opus | requires_claude: codex never completed a capstone; 12+ artifact reads; open-ended synthesis |
| exp2470 (retro) | codex | gpt-5.5 | Templated structure |

FR-11 mandate: exp2463 (continuous_self_learning_task: true), gate cross_session_retention_rate >= 0.50.
