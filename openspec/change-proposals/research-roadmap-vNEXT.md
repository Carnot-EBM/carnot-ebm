# Research Roadmap — Milestone 2026.05.328 (Depth-Over-Breadth XIV: Consolidate the P0.1 Positive)

**Status:** staged (pre-activation)
**Planner:** Claude Opus 4.8, 2026-05-31
**Predecessor:** 2026.05.327 (Depth-Over-Breadth XIII)
**North star:** `ops/north-star.md` — one headline (FoVer 0.9131), one finish line (G1–G4; G2 sole unmet gate).

---

## 1. What the previous milestone proved (.327)

`.327` was the milestone that gave **P0.1 its first clean, terminal, *discriminating* verdict — and it
came back POSITIVE.** All 11 tasks ran. Read via `scripts/summarize_artifact.py`:

| Exp | Verdict | Meaning |
|---|---|---|
| **exp3551** Route-1 graph-coloring TERMINAL (discriminating corpus) | `complete: p01_energy_beats_strong_nonAR_baseline_on_discriminating_corpus_terminal_positive_solve_rate_0.963_vs_strong_0.700_p_0.000` | **CLEAN TERMINAL POSITIVE.** On a corpus near the chromatic/freezing threshold (STRONG DSATUR forced to 0.70 global / 0.56 hard-tier), energy global inference solved 0.9625, hard-tier paired diff **+0.38, p=0.000**; greedy-AR 0.15; exact==1.0; `pt_swap_acceptance_rate=0.346`; 0 CRITICAL flags. Energy descent beats BOTH autoregressive AND a strong classical CSP solver. |
| **exp3553** Route-2 fair test | `complete: blocked_corpus_has_no_selectable_headroom_oracle_le_sc` | BLOCKED again. The greedy-wrong corpus builder (exp3552) produced only n=3. `flip_count=0`. Route-2 headroom-starved 4× (exp3507/3530/3531/3542/3553). |
| **exp3554** aggregation promotion | `complete: blocked_no_second_corpus_for_transfer` | BLOCKED. The exp3543 A→B transfer positive (AUROC 0.861) stands but was not promoted — no second/third transfer corpus was constructible at run time. |
| **exp3555** FR-11 non-degenerate deploy | `complete: conservative_default_beta_deploys_on_nondegenerate_corpus_prevents_collapse_to_N200_real_quality_maintained` | **CLEAN POSITIVE.** Starting true accuracy 0.34; conservative-default beta prevents depth-N≥200 collapse (beta=0 collapses), real quality maintained (final 0.367). First non-vacuous FR-11 deployment. |
| **exp3556** G2 regression | `complete: fover_g2_package_regression_clean_external_ask_current_g2_operator_gated` | CLEAN. Package still reproduces 0.9131 within CI. G2 stays operator-gated. |
| exp3550/3557/3558/3559/3560 | OPS / KV260 blocked SSH / PolarFire reachable / synthesis / capstone | All ran. `depth_forcing_function_can_relax = true`; G1∧G3∧G4 met, **G2 the sole unmet gate.** |

**The honest P0.1 reading after .327:** energy-based global inference (PT/SA on the Ising encoding +
exact CP) **decisively beats autoregressive generation AND significantly beats a strong classical CSP
solver** on a discriminating graph-coloring corpus. This is the non-autoregressive-reasoning thesis
(the Phase-3 / Kona premise) made empirical — the result the Depth-Over-Breadth Forcing Function was
waiting for. But it is **one CSP (graph coloring), one seed, one generator** — not yet a robust
headline. And two tracks blocked on *build* failures (Route-2 corpus n=3; aggregation second corpus
absent), not on science.

---

## 2. The three biggest gaps (current state vs PRD vision)

1. **The P0.1 Route-1 positive is single-CSP / single-seed.** To become a defensible secondary headline
   ("energy global inference beats strong classical + AR on combinatorial reasoning"), it needs (a)
   multi-seed CI + a second graph generator on graph coloring, and (b) **generalization to a SECOND
   discriminating CSP** with the same strong-baseline rigor (the neural-CO critique literature —
   arXiv:2502.03669/2302.03602/2112.12251 — is unforgiving of single-instance-family claims).
2. **Two fresh positives are blocked on build failures, not science.** The cross-corpus aggregation
   transfer (exp3543, AUROC 0.861) is single-pair because exp3554 had no second corpus; the Route-2
   selection premise has never had a fair test because no headroom corpus was ever built. Both are
   *constructible* from corpora that already exist.
3. **G2 — the SOLE unmet publication gate — is external/operator-gated**, and the self-learning thesis
   (FR-11) deployed once (exp3555) but on a single corpus; P0.2 (does verifier *diversity* improve the
   alpha_t grounding signal?) is untested.

---

## 3. Milestone design — 11 tasks, exp3561–exp3571

The Depth-Over-Breadth Forcing Function CAN now relax (P0.1 has a clean terminal verdict), so `.328`
shifts from *chasing an unproven crux* to **consolidation: harden the proven positives into defensible
(secondary) headlines and give the bounded tracks their terminal verdicts.** North-star §1 still
governs — every task advances a headline or it is noise. No `vN+1` re-measurement that does not answer a
new question.

Architecture rules carried verbatim from the working .322–.327 chain (they are why those milestones
landed clean):

- **Agent routing:** all 11 tasks PLANNED `agent_type: claude` + `requires_claude: true` — REQUIRED to
  pass the `MODEL_AGENT_COHERENCE` pre-activation gate audit (`scripts/experiment_1152_gate_audit_pre_activation_v2.py`,
  which does NOT allow gemini). gemini-cli 0.44.0 is up; the outer-loop REROUTES the mechanical tasks
  (3561/3563/3565/3566/3567/3568/3569/3570/3571) to gemini AT ACTIVATION per Gemini-Default + the
  .325/.326/.327 reroute precedent. **exp3564** (Route-2 final attempt + honest terminal framing) STAYS
  claude — the sole genuine-judgment task this milestone.
- **No `model: opus` anywhere** (opus thinking-400 killed .321's builder + .322's first G2).
- **Cascade-proof:** no depth task `gated_on` another depth task; exp3564 READS exp3552's corpus / its
  own freshly-built corpus and blocks honestly rather than gating; the synthesis (exp3570) is UNGATED
  (reads & skips absent/flagged); only the capstone (exp3571) gates on the synthesis-ready flag.
- **Per-iteration progress flush + hard wall-clock budget** on every loop (defeats the 1201s idle-timeout).
- **Anti-tautology:** aggregation/ops/hardware/G2/synthesis/capstone tasks set `random_seed=20260601`
  (NOT the exp number); measurement tasks set a CONTENT-DERIVED seed; NEVER store the same measured
  quantity under two field names; references go ONLY in `methodology_note` strings; CSP corpora must NOT
  be ceiling-saturated for the STRONG baseline (strong-baseline solve-rate on the hard tier < 0.9, per
  the .327 discriminating-corpus discipline).

### Phase A — OPS transition
- **exp3561** — archive .327, activate .328.

### Phase B — DEPTH (consolidate the P0.1 positives into defensible secondary headlines)
- **exp3562** *(#1 priority, CPU)* — **GENERALIZE the P0.1 Route-1 positive to a SECOND discriminating
  CSP.** Apply the exp3551 protocol (discriminating corpus where a STRONG classical baseline is forced
  < 0.9 on the hard tier while the exact solver confirms feasibility, energy vs strong-classical vs
  AR-greedy, bootstrap CI + paired McNemar/bootstrap significance) to a NEW CSP — NOT graph coloring or
  Sudoku. Primary candidate: **k-SAT near the satisfiability threshold (α≈4.26)** with WalkSAT/CDCL as
  the strong baseline (the discrete-diffusion-beats-AR family, arXiv:2410.14157); fallback Max-Cut or
  number partitioning. Turns the positive into "energy global inference beats strong classical + AR on
  TWO independent CSPs." `retire_if_same_verdict: true` (if energy only ties the strong baseline on a
  second discriminating CSP, the Route-1 claim is bounded to graph coloring).
- **exp3563** *(CPU)* — **HARDEN the graph-coloring positive: multi-seed CI + a SECOND graph generator.**
  exp3551 was a single discriminating corpus at a single seed. Re-run the hard-tier paired comparison
  over ≥5 seeds (report mean + CI95 on the per-instance paired diff, must exclude 0) AND on a second
  graph family/generator (e.g. Erdős–Rényi G(n,p) vs Barabási–Albert / planted-partition), to rule out
  a seed/generator artifact. Makes the graph-coloring positive itself defensible-headline-grade.
- **exp3564** *(live GPU, the sole judgment task)* — **Route-2 NL-math FINAL attempt + terminal verdict.**
  The greedy-wrong corpus build has failed twice (exp3530 difficulty-band; exp3552/3541 greedy-wrong,
  n=3). Genuinely-different construction: pull HARDER competition-grade problems (AIME/MATH-L5), a much
  bigger pool, the full GPU budget, k≥16, AND combine the MULTI-verifier ensemble (Weaver/BoN-MAV,
  arXiv:2506.18203/2502.20379) rather than a single energy reranker. If a headroom corpus (oracle > SC,
  n≥40) is built, run the fair test on it; if not, **Route-2 on NL-math is permanently retired as a
  trustworthy terminal negative** (SC is near-optimal on NL-math; the energy-reranking selection
  premise does not hold there). `retire_if_same_verdict: true`.
- **exp3565** *(CPU)* — **PROMOTE the cross-corpus aggregation positive to a defensible SECONDARY
  HEADLINE.** Fix exp3554's "no second corpus" block by building the transfer targets from held-out
  DISJOINT splits of corpora that ALREADY exist (FoVer + the level-3 corpus): fit/freeze the aggregation
  on corpus A, evaluate held-out final-correctness AUROC on B AND C over ≥5 seeds (mean + CI95) with
  per-target shuffle controls. Survives the step-OOD / question-OOD failure modes (arXiv:2502.14361).
  `secondary_headline_eligible` iff transfer beats the un-aggregated floor on BOTH targets with
  collapsing shuffle controls.

### Phase C — SELF-LEARNING (mandatory continuous self-learning + P0.2) + G2 (sole gate)
- **exp3566** *(CPU)* — **FR-11 ADVANCE: multi-corpus deploy + verifier-DIVERSITY grounding (P0.2).**
  exp3555 deployed the conservative-default beta once. This advances it: run the closed loop on a
  3-corpus non-degenerate battery AND test whether DIVERSE multi-verifier alpha_t grounding
  (Weaver-style combination of the verifier ensemble, arXiv:2506.18203) sustains collapse-prevention +
  quality better than single-verifier grounding — the P0.2 verifier-diversity question. The mandatory
  continuous-self-learning experiment.
- **exp3567** *(CPU)* — **G2 regression-verify refresh.** Re-confirm the self-contained package still
  reproduces 0.9131 within CI from a fresh environment after the .328 changes (drift detection); keep
  the one-click external ask current. NEVER pushes / triggers CI / marks G2 met (Operator-Only External
  Publication).

### Phase D — HARDWARE (opportunistic continuity per north-star §3; minimal)
- **exp3568** — KV260 terminal board-level latency transcript (SSH precondition; drive to terminal then
  freeze).
- **exp3569** — PolarFire opportunistic reachability + continuity audit (no terminal mandate;
  tautology-de-flagged distinct fields).

### Phase E — SYNTHESIS (gate status + capstone)
- **exp3570** — G1–G4 gate-status synthesis v328 (UNGATED, cascade-proof, seed-fixed).
- **exp3571** — Capstone v328 (gated on the synthesis-ready flag).

---

## 4. Dependency graph (cascade-proof)

```
exp3561 (ops: archive/activate)
   │
   ├─ exp3562  P0.1 second-CSP generalization  (CPU)      ─┐
   ├─ exp3563  graph-coloring multi-seed CI     (CPU)      │
   ├─ exp3564  Route-2 NL-math final + terminal  (GPU)     │  no cross-gating;
   ├─ exp3565  aggregation A→{B,C} promotion     (CPU)     │  each blocks honestly
   ├─ exp3566  FR-11 multi-corpus + P0.2 diversity (CPU)   │  on its own preconditions
   ├─ exp3567  G2 regression refresh             (CPU)     │
   ├─ exp3568  KV260 terminal transcript         (HW/SSH)  │
   └─ exp3569  PolarFire reachability audit      (HW/SSH) ─┘
                          │
                   exp3570  G1–G4 synthesis v328  (UNGATED — reads & skips absent/flagged)
                          │  gate_status_v328_ready == true
                   exp3571  Capstone v328
```

## 5. Hardware requirements

- **exp3564** needs 2× RTX 3090 (CUDA up) + a cached SOTA GGUF (gemma-4-26B-A4B / 31B / Qwen3.6-35B);
  loads via the GGUF path (embedded tokenizer), NEVER `AutoTokenizer` on a `-GGUF` repo id.
- **exp3562/3563/3565/3566** are CPU-only (Ising/PT global opt; cached verifier scoring).
- **exp3568/3569** need SSH reachability to the KV260 / PolarFire boards (precondition-gated; honest
  `blocked_*` verdict if unreachable).

## 6. Self-learning coverage (PRD requirement)

`exp3566` is the mandatory continuous-self-learning experiment: it advances FR-11 from a single-corpus
deployment to a multi-corpus battery and adds the P0.2 verifier-diversity grounding test (Weaver-style
multi-verifier alpha_t grounding vs single-verifier).

## 7. Exit criteria

- P0.1 Route-1 either generalizes to a second discriminating CSP (robust secondary headline) or is
  bounded to graph coloring (both terminal, both clean).
- The aggregation positive is promoted to a defensible cross-corpus secondary headline, or bounded to
  the A→B pair.
- Route-2 on NL-math gets a headroom-corpus test or is permanently retired as a terminal negative.
- FR-11 deploys across a battery + the P0.2 diversity question is answered.
- G2 stays drift-free and operator-ready; the capstone reports `unmet_gates` (G2), not a count.
