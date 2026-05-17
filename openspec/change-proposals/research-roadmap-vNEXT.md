# Research Roadmap — Milestone 2026.05.222

**Title:** FST Continual Adaptation, ODAR Energy Routing, CASAL Hard Sampling, and COOL On-Device KAN

**Milestone:** 2026.05.222
**Date:** 2026-05-17
**Experiment IDs:** exp2239–exp2252 (14 tasks)
**Previous milestone:** 2026.05.221

---

## What Milestone 2026.05.221 Proved

Milestone .221 established four foundational implementation artifacts:

1. **ActFocus token reweighting** (exp2229): Energy-variance-driven gradient redistribution from reasoning tokens to action tokens; addresses the action bottleneck in verify-repair RL training loops.
2. **KAN-CL per-knot importance regularization** (exp2231): B-spline anchoring at per-knot granularity reduces catastrophic forgetting without replay buffers; directly satisfies FR-11 Tier 1 online constraint learning.
3. **AdamFLIP hard constraint optimizer** (exp2235): Adaptive Momentum Feedback Linearization enforces hard equality constraints during EBM verifier training, reducing violation magnitude vs soft-penalty CD.
4. **Wahkon RKHS superposition** (exp2233): Deep RKHS alternative to spline KAN with MAP estimation and minimax-optimal convergence guarantees.
5. **Thermodynamic optimized Langevin initialization** (exp2237): Mpemba-effect suppression of slow relaxation modes reduces thermalization time for Carnot's phase-3 continuous EBM samplers.

**Key gaps entering .222:**
- FR-11 continuous self-learning was addressed via KAN-CL (exp2231/2232), but the Fast-Slow Training (FST) architecture (arXiv:2605.12484) — the mechanism that explains the .96–.150+ FR-11 stalls — was not yet integrated.
- ODAR-style routing (arXiv:2602.23681) was flagged in known-issues (score 400) but not implemented.
- Hard constraint enforcement via CASAL (arXiv:2505.18017) extends AdamFLIP to the sampling phase but was not yet implemented.

---

## Architecture Snapshot

```
Milestone 2026.05.222 additions (bold = new):

Verify-Repair Cascade:
  Tier 0a–0e probes → [ODAR free-energy routing gate] → Tier 1–2.7 → Tier 3 Ising
                             ^
                    **ODAR risk-sensitive fusion** (exp2243)
                    fast agent = Tier 0 probes
                    deliberative = Tier 3 Ising

Continuous Self-Learning (FR-11):
  Slow weights = k=16 verifier ensemble + base LLM (frozen)
  Fast weights = **FST verifier-output-summary** (exp2240/2241)
                 updated via ICL; drives next repair prompt

Hard Constraint Sampling:
  **CASAL** primal-dual Split Augmented Langevin (exp2245)
  vs AdamFLIP Lagrangian gradient (exp2235/2236)
  → zero-false-accept guarantee path

KAN Tier:
  KAN-CL per-knot (.221) → **COOL-style scaling to n=256** (exp2247)
  → THRML parity validation at n=256 (exp2248)

Hardware Track (active):
  KV260 RTL → **Verilator lint + sim** (exp2249)
  THRML/Extropic TSU → **CASAL sampler interface update** (exp2250)
```

---

## The 3 Biggest Gaps vs PRD Vision

1. **FR-11 self-learning stalls (parameter-only RL fails)** — The .96–.150+ retrospectives repeatedly found FR-11 utility metrics stalling or regressing. Fast-Slow Training (arXiv:2605.12484) empirically explains why: parameter-only RL (what Carnot's FR-11 used) is strictly dominated by fast-slow on sample efficiency AND KL drift. Integrating FST's slow-fast decomposition is the load-bearing fix.

2. **Verify-repair cascade uses argmax routing (uniform cost)** — The current cascade tries Tier 0 probes then falls through regardless of confidence. ODAR (arXiv:2602.23681) proves that free-energy-principled risk-sensitive routing reduces computational overhead 82% with accuracy parity. This is the primary candidate for Phase 4 compute reduction.

3. **Hard constraint enforcement still uses soft penalties in training** — AdamFLIP (.221) addressed this for equality constraints during PINN-style training. CASAL (arXiv:2505.18017) extends to the sampling phase: guarantees hard constraint satisfaction during Langevin sampling, not just at optimization time. Together they close the full training+inference hard-constraint gap.

---

## Phase Descriptions

### Phase 0: Archive and Activate
Archive milestone 2026.05.221 into research-complete.yaml. Initialize changelog stub. One task (exp2239), max_turns=20.

### Phase 1: Fast-Slow Training for FR-11 Rescue
Implement the FST slow-fast decomposition (arXiv:2605.12484) for Carnot's verify-repair iteration loop. Slow weights = frozen k=16 verifier ensemble + base LLM. Fast weights = verifier-output-summary context prepended to the next repair prompt (updated via ICL, not RL). Validate: sample efficiency >= 2x baseline, KL drift <= 0.5x. One FR-11 continuous_self_learning_task run. Then integrate ActFocus (.221) + FST for unified gradient assignment on SOTA GGUFs.

### Phase 2: ODAR Free-Energy Routing
Implement the free-energy-principled risk-sensitive fusion selector (arXiv:2602.23681) for Carnot's cascade. Fast agent = Tier 0 probes (sub-1ms). Deliberative agent = Tier 3 Ising verification. Gate on predicted free-energy uncertainty. Benchmark on 30-example reasoning corpus; acceptance gate: >= 30% computation reduction vs uniform routing.

### Phase 3: CASAL Hard Constraint Sampling
Implement CASAL (arXiv:2505.18017) primal-dual Split Augmented Langevin for hard equality constraint enforcement during Carnot's continuous EBM sampling. Compare constraint violation magnitude against AdamFLIP (.221). Acceptance gate: CASAL mean violation magnitude <= AdamFLIP/2 on 100-sample test.

### Phase 4: COOL On-Device KAN Scaling
Scale KAN-CL (.221) to n=256 and n=512 spin Ising models, targeting the COOL-demonstrated 20μs per-knot update budget. Validate THRML/Carnot parity at n=256 (no regression from .221's n=128 baseline). Acceptance gate: bounded KL < 0.05 at n=256.

### Phase 5: Hardware Track
KV260 RTL: continue Verilator source-level lint + simulation (architecture.md active track; no bitfile or board claim until Vivado synthesis runs). THRML/Extropic TSU: update sampler interface compatibility layer with CASAL's primal-dual sampler protocol.

### Phase 6: Capstone and Retrospective
E2E integration test: FST + ODAR routing + CASAL sampling on gemma-4-26B-A4B-it-GGUF. All three components must complete with passing gates before capstone runs. Retrospective analyzes compute reduction, FR-11 utility delta, and constraint violation statistics.

---

## Dependency Graph

```
exp2239 (archive)
  |
  +-- exp2240 (FST impl, codex)
  |     |
  |   exp2241 (FR-11 FST eval, codex) [continuous_self_learning_task]
  |     |
  |   exp2242 (ActFocus+FST integration on SOTA GGUFs)  ──────── exp2251 (capstone)
  |                                                                   |
  +-- exp2243 (ODAR routing impl, codex) ─────────────────────── exp2251
  |     |                                                            |
  |   exp2244 (ODAR benchmark, codex) ─────────────────────────── exp2251
  |
  +-- exp2245 (CASAL impl, codex) ──────────────────────────────── exp2251
  |     |
  |   exp2246 (CASAL vs AdamFLIP, codex)
  |
  +-- exp2247 (KAN-CL n=256 scaling, codex)
  |     |
  |   exp2248 (THRML parity n=256, codex) [gated on exp2247]
  |
  +-- exp2249 (KV260 RTL Verilator, codex)
  +-- exp2250 (THRML/TSU CASAL interface, codex)

exp2251 (capstone, opus) [gated on exp2242+exp2244+exp2246]
exp2252 (retrospective, codex)
```

---

## Hardware Requirements

| Track | Hardware | Tasks |
|-------|----------|-------|
| LLM evaluation (SOTA GGUFs) | RTX 3090 pair (CUDA) | exp2242, exp2251 |
| Ising sampling | CPU (fastest for Carnot sampler) | exp2246, exp2248 |
| KV260 RTL simulation | CPU (Verilator) | exp2249 |
| THRML interface | CPU (JAX simulation) | exp2250 |

---

## Agent Routing

All tasks use `agent_type: codex`, `model: gpt-5.5` except:
- exp2251 (capstone): Claude, model: opus, max_turns: 100 — multi-file coordination + live GPU requires Claude's tool choreography.

---

## Acceptance Gates

| Gate | Condition | Experiment |
|------|-----------|------------|
| FST sample efficiency | sample_efficiency_ratio >= 2.0 | exp2241 |
| FST KL drift | kl_drift_ratio <= 0.5 | exp2241 |
| ODAR compute reduction | compute_reduction_pct >= 30.0 | exp2244 |
| CASAL violation reduction | casal_violation_mean <= adamflip_violation_mean / 2.0 | exp2246 |
| KAN-CL n=256 parity | bounded_kl < 0.05 | exp2248 |
| Capstone gate | exp2242 + exp2244 + exp2246 all contain 'complete' | exp2251 |

---

## Cross-References

- FST: arXiv:2605.12484, known-issues 2026-05-15 operator-flagged candidate
- ODAR: arXiv:2602.23681, known-issues 2026-05-15T21:30Z sweep candidate
- CASAL: arXiv:2505.18017, research-references.md 2026-05-14 post-.169 sweep
- COOL: Springer 2026, research-references.md 2026-05-17 post-.221 sweep
- ActFocus: arXiv:2605.14558, .221 exp2229
- KAN-CL: arXiv:2605.12306, .221 exp2231
- AdamFLIP: arXiv:2605.08408, .221 exp2235
- hardware track: architecture.md "Active hardware tracks (Exp 1460)"
