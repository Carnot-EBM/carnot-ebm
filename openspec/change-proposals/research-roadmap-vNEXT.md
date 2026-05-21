# Research Roadmap vNEXT — Milestone 2026.05.267
# Multi-Corpus Evidence Harvest + RecMem FR-11 + Delta H2 Fix v3

**Milestone:** 2026.05.267
**Author:** outer-loop planning agent (Claude) — 2026-05-21T17:25Z
**Prior milestone:** 2026.05.266 (1 artifact: capstone only; pre-test cascade blocked all research)
**Tasks:** exp2819–exp2829 (11 tasks)

---

## Context: Four Consecutive Zero-Research-Artifact Milestones

Milestones .263, .264, .265, and .266 all produced zero research artifacts. The root cause
was a self-referential pre-test gate deadlock: the conductor's smart-subset test gate was failing
("1 failed, 80 passed"), and the only task that could fix it was itself gated on the same
pre-test that was failing. After 3 consecutive SKIPs, the fix task was retired as "merit-failure",
cascade-blocking all downstream research tasks.

Resolution (outer-loop, 2026-05-21 ~15:35Z): Pre-test smart-subset was run manually in this
planning session. Result: 81 passed. The `.pretest-cache.json` records the green state. Cascade
is broken. The conductor will read the cache on next activation and proceed with tasks.

Accumulated carry-forward (4 milestones):
- FoVer Memory-Leakage Isolation (FR-11 dual-condition)
- MBPP dual-condition corpus evaluation (N=100)
- HumanEval dual-condition corpus evaluation (N=164)
- TruthfulQA dual-condition corpus evaluation (N=200)
- Cross-corpus verifier discriminative matrix
- FUSE zero-labeled ensemble weighting
- Paper v6 §5 multi-corpus results table

Additionally, two critical unresolved gaps from .260–.264:
- Delta H2 repair pipeline regression (empirical_delta=0.000, 5+ milestones)
- RecMem FR-11 memory recurrence trigger (mandatory since .257, never picked up)

---

## What .266 Proved (and Didn't)

| Status | Finding |
|--------|---------|
| Validated | Phase 4 FEP held_out_auroc=0.9989 (exp2766, .262) |
| Validated | Ensemble v12 k=18, +0.011 AUROC lift (exp2756, .261) |
| Validated | FR-11 Tier 4: cycle3 AUROC=0.9275, 34 NEXUS rules (exp2755, .261) |
| Validated | Tier 0y ECE~5e-8 (exp2759, .261) |
| Validated | Phase 1 v0.1.0b1 ship ready (exp2760, .261, operator action pending) |
| Validated | Weak-strong policy fix: t_low=0.018 < t_high=0.14 (exp2758, .266 pre-cascade) |
| Open | Verifier live-GPU: energy_values=[0,0,0,0,0] on 5 live-GGUF examples (exp2765) |
| Open | Delta H2 regression: 0/60 successes (exp2754, confirmed .261) |
| Untested | FoVer overfit thesis (operator hypothesis 2026-05-21) |
| Untested | FR-11 self-learning contribution hypothesis (operator hypothesis 2026-05-21) |
| Untested | MBPP / HumanEval / TruthfulQA cross-corpus generalization |
| Overdue | RecMem FR-11 recurrence trigger (mandatory since .257) |

---

## Three Biggest Gaps

### Gap 1 (CRITICAL): Multi-Corpus Evaluation - 4 Milestones Accumulated

Impact: The FoVer-overfit thesis and the FR-11 self-learning contribution hypothesis remain
entirely untested. The headline AUROC=0.9857 (exp2546) is FoVer-only. Without MBPP, HumanEval,
TruthfulQA data points, the paper-v6 section 5 table cannot be written.

Operator directive (2026-05-21, carries forward): "we want to report both with and without
learning so we know both how effective our naive answers are as well as what benefit the
learning is affording our solution."

Resolution: exp2820 (FoVer leakage isolation, FR-11 dual-condition), exp2821 (MBPP),
exp2822 (HumanEval), exp2823 (TruthfulQA) -- all carry-forwards with no gating on exp2810
(the cascade is now resolved at outer-loop level).

### Gap 2 (CRITICAL): Delta H2 Repair Pipeline Regression - 5+ Milestones

Impact: empirical_delta=0.000 (exp2754, .261). The verify-and-repair pipeline shows zero
improvement over baseline. Without a positive empirical delta, paper-v6 section 5 cannot
claim repair adds value beyond detection.

Prior failures:
- exp2754 (.261): confirmed H2 regression (0/60 successes)
- exp2767 (.262): gemini rate-limited 3x (600s each, abandoned)
- exp2791 (.264): planned Claude Opus, never ran (pre-test cascade)

Resolution: exp2826 -- Claude Opus git-bisect + repair fix.

### Gap 3 (MANDATORY PRIORITY): RecMem FR-11 Memory Recurrence Trigger - 10 Milestones Overdue

Impact: FR-11 Tier 2/3 memory currently extracts per-verification (eager consolidation).
arXiv:2605.16045 (RecMem, ACL 2026 Findings) shows -87% memory-construction token cost via
recurrence-trigger consolidation. Filed in known-issues.md as ".257+ MANDATORY". Now 10 milestones
without pickup; CLAUDE.md "Overdue-Priority Forcing Function" mandates inclusion.

Resolution: exp2827 -- RecMem RecurrenceTrigger implementation + benchmark.

---

## Architecture Snapshot (end of .266)

Carnot Verifier Ensemble (k=18 as of exp2756):
- Tier 0 fast verifiers (k=14 base)
  - Tier 0f: Weak-Strong policy (FIXED exp2758, t_low=0.018 < t_high=0.14, 41% savings)
  - Tier 0v: Set-Consistency Energy (exp2743)
  - Tier 0w: Paraphrastic Consistency (exp2746)
  - Tier 0x: Conformal Selective Acting (exp2757)
  - Tier 0y: Differentiable Conformal Training ECE~5e-8 (exp2759)
- Tier 1 medium verifiers (NLA-class + phase-3)
- Phase 4 FEP Router: held_out_auroc=0.9989 (exp2766, .262)
- FR-11 Tier 4 NEXUS: 34 rules, cycle3 AUROC=0.9275 (exp2755, .261)
- Repair Pipeline: empirical_delta=0.000 (BROKEN -- exp2754, .261)

FoVer corpus headline: AUROC=0.9857 (exp2546, production with FR-11 state loaded)
Peer HIVE ceiling: AUROC=0.924
Cross-corpus: UNMEASURED (4 milestones of blocked carry-forward)

---

## New Research Contributions Incorporated

| Paper | ArXiv | Integration plan |
|-------|-------|-----------------|
| RecMem: Recurrence-based Memory Consolidation | 2605.16045 | exp2827 -- FR-11 Tier 2/3 recurrence trigger |
| FUSE: Ensemble Composition Without Labels | 2604.18547 | exp2825 -- zero-labeled ensemble weighting |
| Behavioral Entanglement Reweighting | 2604.07650 | exp2824/2825 context |
| Conditional Diffusion Under Linear Constraints | 2605.05387 | paper-v6 section 3 cite |
| Tool Receipts: Practical Hallucination Detection | 2603.10060 | paper-v6 section 4 cite |

---

## Phase Structure

### Phase A -- Archive + Admin (exp2819)

Archive both .265 and .266 to research-complete.yaml (both were zero-artifact; neither
was archived in its own activation task). Activate .267.

### Phase B -- Multi-Corpus Dual-Condition Evaluations (exp2820-exp2823)

Four corpus evaluations, all dual-condition (Condition A: full FR-11 production state,
Condition B: FR-11 state reset). Directly answers the operator directive.

All Phase B tasks are independent; they run in parallel after exp2819.

- exp2820: FoVer Memory Leakage Isolation (FR-11 mandate, continuous_self_learning_task: true)
- exp2821: MBPP dual-condition (N=100, execution-based ground truth)
- exp2822: HumanEval dual-condition (N=164, execution-based ground truth)
- exp2823: TruthfulQA dual-condition (N=200, BLEURT semantic similarity)

### Phase C -- Research Advancement (exp2824-exp2827)

Independent research tasks. Run after Phase B.

- exp2824: Cross-Corpus Per-Verifier Matrix (gated on exp2820)
- exp2825: FUSE Zero-Labeled Ensemble Weighting (independent)
- exp2826: Delta H2 Regression Fix v3 (Claude Opus; git-bisect + repair pipeline fix)
- exp2827: RecMem FR-11 Recurrence-Trigger (MANDATORY pickup from .257)

### Phase D -- Publication (exp2828)

- exp2828: Paper v6 section 5 Dual-Condition Table (gated on exp2824)

### Phase E -- Capstone (exp2829)

- exp2829: Capstone v267 (Claude Opus; synthesis + gap analysis for .268)

---

## Dependency Graph

exp2819 (archive .265+.266, activate .267) feeds all downstream tasks.

Phase B (exp2820-exp2823): all independent after exp2819.
- exp2820 (FoVer) -> exp2824 (cross-corpus matrix)
- exp2824 -> exp2828 (paper table)

Phase C (exp2824-exp2827): after Phase B.
- exp2825 (FUSE), exp2826 (Delta H2), exp2827 (RecMem): all independent.

Phase D (exp2828): gated on exp2824.
Phase E (exp2829): synthesis of all prior.

---

## Hardware Requirements

All hardware boards are at terminal state. No mandatory hardware tasks.

| Board | Status |
|-------|--------|
| KV260 | TERMINAL (.260, 3.183 microsecond mean latency) |
| GateMate | TERMINAL (.247, n=16 Ising bitstream flashed and smoke-tested) |
| PolarFire | TERMINAL (.241, end-to-end Carnot dispatch validated) |

---

## Agent Routing

9/11 gemini (81.8%) + 2/11 claude/opus (18.2%, within ceiling).

exp2826 (Delta H2 fix) meets all 3 positive criteria for requires_claude: true:
1. Gemini demonstrably failed (3x rate-limited, exp2767 -- each attempt burned 600s)
2. Multi-file tool choreography: git-bisect across commit history + multi-file pipeline edits
3. Multi-step open-ended hypothesis testing (H1/H2/H3) requires judgment under ambiguity

exp2829 (Capstone) meets criteria 3: cross-artifact synthesis across 4 dual-condition
corpora + per-verifier matrix + paper-v6 narrative implications = open-ended judgment.

---

## Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | Archive .265 + .266 complete (exp2819) |
| 2 | FoVer leakage delta measured (exp2820.learning_contribution not null) |
| 3 | MBPP dual-condition AUROC measured (exp2821) |
| 4 | HumanEval dual-condition AUROC measured (exp2822) |
| 5 | TruthfulQA dual-condition AUROC measured (exp2823) |
| 6 | Cross-corpus matrix built (exp2824) |
| 7 | FUSE weighting evaluated (exp2825) |
| 8 | Delta H2 root cause identified or blocked (exp2826) |
| 9 | RecMem recurrence trigger implemented (exp2827) |
| 10 | Paper section 5 table compiled (exp2828) |
| 11 | FR-11 mandate satisfied (exp2820.continuous_self_learning_task=true) |
| 12 | Capstone written honestly (exp2829) |

---

## CLAUDE.md Compliance

- Gemini-Default: 9/11 gemini, 2/11 claude/opus (both requires_claude: true)
- Prior failures: all carry-forward tasks have full 4-field prior_failures blocks
- PRECONDITIONS step 0: all compute-bound tasks
- Principle-annotated artifact fields: all REQUIRED ARTIFACT FIELDS
- Terminal-prefix verdicts: all tasks specify honest_verdict must start with terminal prefix
- FR-11 mandate: exp2820 (continuous_self_learning_task: true)
- Hardware continuity: all 3 boards TERMINAL
- Exclusion manifest cross-check: 0 scope matches
- Operator-only publication: exp2828 generates table only, never submits
- RecMem mandatory pickup: exp2827 satisfies .257+ MANDATORY overdue priority
- SOTA models: Qwen3.6-35B-A3B-GGUF in all inference tasks

---

## Note on Pre-Staged YAML

The pre-staged YAML (commit 3cd85879a) had 8 tasks (exp2819-exp2826) mapping exp2825 to
the paper table and exp2826 to capstone. This vNEXT planning pass replaces it with 11 tasks
that add: FUSE (exp2825), Delta H2 fix (exp2826), RecMem mandatory (exp2827), and renumber
paper table to exp2828 + capstone to exp2829. Same milestone: "2026.05.267" so the
Pre-Staged Roadmap Convention applies -- conductor will use this YAML.

---

## Historical Content (2026.05.222 -- preserved per "Never remove existing content" rule)

The original content of this file (research-roadmap-vNEXT.md at 2026.05.222) described
FST Continual Adaptation, ODAR Energy Routing, CASAL Hard Sampling, and COOL On-Device KAN.
That milestone closed. This file has been repurposed for the 2026.05.267 planning pass.
The original content is preserved in git history (commit prior to this overwrite).

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
