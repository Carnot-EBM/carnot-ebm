# Research Roadmap — Milestone 2026.05.323 (Depth-Over-Breadth IX)

**P0.1 Now Has Two Narrow Scientific Next Steps, Not Infra Losses:
Real Combinatorial Optimizers on the VALIDATED Sudoku-Ising Encoding +
the Energy-vs-Self-Consistency Crux on the PURPOSE-BUILT In-Band Corpus;
Closing the FoVer Step-vs-Final 0.138 Gap; Closed-Loop Deployment of the
β_min=f(λ_min) Self-Learning Law; G2 Drift-Verify**

- **Milestone:** `2026.05.323`
- **Milestone doc:** this file
- **Planner:** Opus 4.8 (2026-05-31), under the Depth-Over-Breadth Forcing Function
- **Predecessor:** `2026.05.322` (Depth-Over-Breadth VIII)

---

## 1. What the previous milestone (.322) proved

`.322` was the FIRST milestone where the P0.1 architecture actually held: both
infra-robust routes RAN to completion (no `thinking`-400 on the crux tasks, no
idle-timeout on the depth tasks themselves) and produced **honest, actionable
scientific diagnoses** instead of operational losses. P0.1 remains OPEN, but its
blockers are now narrow and scientific, not infrastructural:

| Exp | Verdict | What it actually established |
|---|---|---|
| **exp3494** P0.1 Route 1 (Sudoku, CPU) | `blocked_kona_failure_is_representational_not_optimizer` | **Encoding VALIDATED:** `encoding_validity_E0` total_energy=0.0, all residuals 0, cross-validated vs QUBO arXiv:2403.04816. BUT `easy_tier_solve_rate=0.0` — the task bailed at Step-0b because vanilla/gradient descent can't escape local minima; **it never reached its own SA/PT/restart ladder.** The encoding is correct; the OPTIMIZER is the bug. |
| **exp3495** P0.1 Route 2 (cached) | `blocked_contested_subset_too_small_n=21` | The cached GSM8K (ceiling) + MATH-L5 (floor) corpora structurally lack an in-band contested subset (only 21, min 40). The contested-subset-of-cached approach is **exhausted**; a purpose-built in-band corpus is required. |
| **exp3496** live builder (non-blocking) | `blocked_no_in_band_split_found` | Idle-timed-out but LEFT `data/p01_difficulty_matched_generations.jsonl` with **40 MATH-500 level-3 problems, per-level SC=0.5 (IN BAND).** Only blocked because the *combined* L3+L4 self-check hit 0.70. **A clean in-band level-3 corpus already exists.** |
| **exp3497** calibration v5 | `mathaware_recalibration_recovers_correctness_signal` ✅ CLEAN | MATH-aware recalibration: process-energy correctness AUROC 0.601 → 0.625; **step-vs-final AUROC gap = 0.138** (FoVer is strong at step-ERROR 0.9131, weak at final-CORRECTNESS on MATH). The gap is the mechanism to close. |
| **exp3498** FR-11 β_min=f(λ_min) | `beta_min_predictable_from_lambda_min` ✅ CLEAN | Deployable law fits: `β_min = -0.300 + 1.846·λ_min`, **r²=0.989, p=0.006, holds out of sample.** Now needs in-loop validation. |
| **exp3499** G2 | `fover_g2_package_regression_clean` ✅ CLEAN | Package regression-clean, AUROC within CI; `g2_met=False`. **G2 is the SOLE unmet publication gate**, awaiting a non-operator external run. |
| **exp3502/3503** synthesis + capstone | ✅ ran (UNGATED, cascade-proof) but **FLAGGED** | A TRIVIAL self-inflicted tautology: both set `random_seed == experiment_number`, so adversarial_verify flagged `experiment==random_seed`. Not a fabrication, but logged FLAGGED. **One-line fix for .323.** |

**Net:** the cascade-proof, all-sonnet, ungated-synthesis architecture WORKED. What's
left is purely scientific: a better optimizer, the right corpus, and a closed-loop
validation — plus the trivial seed fix.

## 2. The three biggest gaps between current state and the PRD vision

1. **P0.1 — does energy-based GLOBAL inference actually SOLVE (the non-AR-reasoning
   endgame)?** Two narrow, now-diagnosed sub-gaps: (a) Route 1 — the Sudoku-Ising
   encoding is proven correct (E=0) but the optimizer used was too weak; a real
   combinatorial optimizer (SA / parallel tempering / restarts / exact QUBO) has never
   been run on the validated encoding. (b) Route 2 — the energy-vs-SC crux has never run
   on a corpus with genuine headroom; the purpose-built in-band level-3 corpus exists but
   the crux was pointed at the headroom-free cached corpora.
2. **The FoVer step-vs-final transfer gap (G4 / headline robustness).** The ensemble is
   0.9131 at step-ERROR but ~0.60 at final-CORRECTNESS on MATH (gap 0.138). Closing this
   is what would let a process energy beat self-consistency — the mechanism behind Route 2.
3. **Continuous self-learning deployment (FR-11 / Phase-5).** The β_min=f(λ_min) law is
   fit but never DEPLOYED: no closed-loop run has measured λ_min, set β by the formula, and
   confirmed it prevents depth-N collapse versus a control. A fitted law is not a
   validated deployment.

(G2 remains the sole unmet publication gate but is operator-only — covered by a single
light drift-verify task, not a "gap" the loop can close autonomously.)

## 3. Architecture of the milestone

```
PHASE A  ── ops transition (sonnet, max_turns 20)
  exp3504  archive .322 / activate .323  (records: P0.1 diagnosed, not lost;
                                          synthesis/capstone flagged on a trivial seed bug)

PHASE B  ── DEPTH BLOCK (majority of slots; all sonnet; NO cross-gating)
  P0.1 ROUTE 1 (PRIMARY, CPU, infra-bulletproof):
    exp3505  Real combinatorial-optimizer ladder on the VALIDATED Sudoku-Ising encoding
             (SA / parallel tempering / K-restarts / exact-QUBO baseline). Encoding is
             E=0 (exp3494); this runs the ladder exp3494 bailed before reaching.
  P0.1 ROUTE 2 (the in-band crux, finally on a headroom corpus):
    exp3506  (live, NON-BLOCKING, GPU) extend the level-3 in-band corpus to n>=80
             (resume; per-iteration flush; hard 18-min budget)
    exp3507  (cached, the crux) energy-vs-SC 7-condition flip-count crux on WHATEVER
             level-3 in-band corpus exists (>=40 already cached). UNGATED on exp3506.
  Mechanism + self-learning:
    exp3508  Close the step-vs-final 0.138 gap: step->final reward-aggregation functions
             (last/product/min/uncertainty-weighted, arXiv:2508.01773) on the in-band corpus
    exp3509  Closed-loop FR-11 validation of beta_min=f(lambda_min): DEPLOY the law, confirm
             it prevents depth-N>=200 collapse vs beta=0 and fixed-conservative controls
             (mandatory continuous-self-learning)

PHASE C  ── G2 (sole publication gate; operator-only)
  exp3510  G2 clean-room regression-verify (drift check after .323) + external-ask refresh

PHASE D  ── hardware continuity (opportunistic, north-star §3; minimal)
  exp3511  KV260 terminal latency transcript (SSH precondition; drive-to-terminal)
  exp3512  PolarFire opportunistic reachability (distinct fields, de-flagged)

PHASE E  ── SYNTHESIS (UNGATED, cascade-proof; TAUTOLOGY-FIXED seed)
  exp3513  G1-G4 gate-status synthesis v323  (reads & skips absent/flagged; seed != exp num)
  exp3514  Capstone v323  (gated only on the robust synthesis-ready flag; seed != exp num)
```

**Dependency graph (cascade-proof by construction):**

- No depth task is `gated_on` another depth task. exp3507 does NOT gate on exp3506 (it reads
  whatever corpus exists). The synthesis (exp3513) is UNGATED — it reads and skips
  absent/flagged upstreams. Only the capstone (exp3514) gates on the synthesis-ready flag,
  which always lands. No single task's retirement can pre-emptively GATE_BLOCK downstream —
  the failure mode that lost .321.

## 4. Infra discipline carried into .323 (the loss-mechanism fixes)

1. **ALL tasks `agent_type: claude`, `requires_claude: true`.** gemini-cli is DOWN
   (.315/.316/.318/.321-plan all crashed on it).
2. **NO `model: opus` anywhere.** The opus extended-thinking `API Error 400:
   thinking/redacted_thinking` killed .321's builder and .322's first G2 attempt. The science
   is CPU/cached with clear gates — sonnet with generous `max_turns` is correct.
3. **Per-iteration progress flush + hard wall-clock budget** on every loop (the Sudoku
   optimizer and the corpus builder). exp3496/exp3498 idle-timed-out at 1201s because prints
   were too sparse / loops went silent. Flush every iteration, not every puzzle/problem.
4. **Aggregation tasks set a DISTINCT fixed `random_seed` (20260531), NEVER the experiment
   number.** This is the one-line fix for the exp3502/exp3503 `experiment==random_seed`
   tautology flag.
5. **Ungated synthesis + capstone** (kept from .322 — it worked).

**Depth-Over-Breadth does NOT relax.** P0.1 has no clean verdict yet on either route, and G2
is not externally run. No vN+1 re-measurement: every task answers a question its predecessor
**structurally could not** (encoding now validated → run the real ladder; purpose-built corpus
→ crux on real headroom; gap quantified → close it; law fit → deploy it).

## 5. Hardware requirements

- **exp3505, exp3507, exp3508, exp3509, exp3510, exp3513, exp3514:** CPU only
  (`JAX_PLATFORMS=cpu`). No GPU, no GGUF, no live generation → immune to the
  thinking-400 / tokenizer / CUDA failure classes.
- **exp3506 (optional/non-blocking):** CUDA + SOTA GGUF via the embedded-tokenizer path
  (`llama_cpp.Llama(model_path=..., vocab_only=True)`, NOT `AutoTokenizer` on a `-GGUF`
  repo). Blocks honestly if CUDA/GGUF unavailable — no cascade.
- **exp3511 (KV260):** SSH-reachability precondition ONLY (`ssh kria`); never a host SD-card
  check (KV260 SSH-Not-SD-Card Discipline).
- **exp3512 (PolarFire):** `ssh polarfire` reachability.

## 6. SOTA model usage

Only exp3506 invokes an LLM, and it uses the mandated SOTA GGUFs via the embedded-tokenizer
path: default `unsloth/gemma-4-26B-A4B-it-GGUF`, fallback `unsloth/gemma-4-31B-it-GGUF`
(and `unsloth/Qwen3.6-35B-A3B-GGUF` is available). All other tasks are CPU/cached
verifier-scoring or aggregation and invoke no LLM.

## 7. New references incorporated (filed in research-references.md, Post-.322 sweep)

- **arXiv:2506.04596** — QUBO solver benchmark (SA / parallel tempering / Neal / simulated
  bifurcation / Gurobi); the optimizer-ladder + exact-baseline reference for exp3505.
- **arXiv:2510.19835** — quantum-inspired (classical) Sudoku-QUBO solver (2025).
- **Wang et al. 2026 (SPE spe.70063)** — resource-efficient Ising Sudoku solver.
- **arXiv:2508.01773** — step-wise reward aggregation (last/product/min/uncertainty-weighted)
  for routing PRM step scores into final selection; exp3508 + exp3507 aggregation.
- **arXiv:2504.16828 (ThinkPRM, carried forward)** — PRM beats SC at matched compute; the bar.

## 8. Acceptance / what would let Depth-Over-Breadth relax

The forcing function relaxes only when **P0.1 has a clean (non-blocked, non-flagged) verdict
on at least one route AND G2 has a concrete in-flight external reproducer.** Concretely for
.323, a relax-eligible outcome is either:

- exp3505 reports a real `solve_rate` (encoding E=0 + a working optimizer solves easy, climbs
  to hard) — energy global inference solves a CSP, OR an honest negative that the validated
  encoding still cannot solve hard Sudoku with any standard combinatorial optimizer (retires
  the timing framing); AND
- exp3507 reports a real `flip_count` + `delta_optimal_vs_self_consistency` on the in-band
  level-3 corpus (energy changes selections, with a measurable net effect, in either
  direction).

Either a clean positive or a clean honest negative on a route counts — both are P0.1 verdicts.
