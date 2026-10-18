# Research Roadmap v400 — SCALE the in-generation oracle-distinct verifier-moat win + ATTACK the ARC deep tail with executable world models

**Milestone:** 2026.06.400
**Planned:** 2026-06-17 (Claude Opus 4.8, outer-loop planner)
**Supersedes:** v399 (2026.06.399)
**North star:** Solve ARC-AGI-3, accurately and efficiently (operator directive 2026-06-08; `ops/north-star.md` §0). The energy VERIFIER is Carnot's value-add; the generator is commodity (local LLM / TRM refiner / coding agent).

---

## 1. What .399 proved (the inputs to this milestone)

`.399 was DEPTH on the two underpowered-positive open moats from `.398, plus the ARC unstall + the mandated
self-learning. It produced a sharp, decision-grade scorecard (exp4323 capstone, `verifier_thesis_state =
in_generation_moat_holds`):

| Axis | .399 result | Status for .400 |
|---|---|---|
| **In-generation moat** (exp4315) | **CLOSED, ORACLE-DISTINCT.** Reward-guided step-stitching (2602.22871) with the leak-rechecked partial-state scorer beat the best engaged control (EntRGi, **+0.225**) AND the model's intrinsic self-reward SMC (2602.01849, **+0.35**); `guidance_moat_ci95=[0.075,0.375]` EXCLUDES 0; `controls_differentiated=true`; `verifier_is_oracle=false`. n=40/arm, 1 corpus. | **HEADLINE — SCALE & HARDEN.** First oracle-distinct verifier win *in generation*. Candidate to flip the DiffusionGemma gate, but operator twice-burned (.396/.397) → replicate on a 2nd corpus + power BEFORE any gate-flip claim. |
| **Cross-domain selection moat** (exp4314) | **RETIRED, domain-bound.** IR3DE+CASCAL+ContextPRM rebuild → SAME verdict as exp4305 (held-out-DOMAIN delta +0.231, CI95 [-0.115,0.538] still includes 0; label-ablation robust, no leak). `retire_if_same_verdict` fired. | **DO NOT re-propose.** Genuinely math/ARC-bound (consistent with `verifier-domain-bound-math-only`). Pivot to verifier DOMAIN EXPANSION (a learned ARC-grid representation), not another selection rerun. |
| **Efficiency** (exp4316) | **always-energy already DOMINATES** (the cheap discriminative verifier is Pareto-dominant; the cascade is unnecessary). | Strengthens §5; re-measuring is churn. Fold into the narrative. |
| **ARC** (exp4317 + outer-loop sweep) | **+1; now 13 reproducible levels / 11 games + FIRST LIVE SUBMISSION** (scorecard 0f6273ce, 13 levels, 11/11 env-matched, operator-authorized). Adapter-free graph-explore cracked the SHALLOW tail. DEEP tail (ar25/ka59/tr87/ft09) RESISTS graph-explore even at 30k expansions (mechanic-limited). | **NORTH STAR — E3 on the deep tail + adapter-free sweep on the shallow tail.** |
| **Cross-game value transfer** (exp4318) | **NULL with generic features** (reduction ~1.0; positive control passed). Gap logged: a learned frame encoder is the candidate fix. | **SELF-LEARNING — a LEARNED FRAME ENCODER** (the mandated self-learning experiment). |
| **Off-ARC execution transfer** (exp4319) | **marginal WIN, execution-grounded** (+0.02, n=200, CI excl 0, `verifier_is_oracle=true`). | Settled; re-accumulating is churn. Not headline. |

`paper_ready=True` throughout (FoVer 0.9131, G1–G4 pass; publication is operator-only).

---

## 2. The three biggest gaps (current state vs PRD / north-star vision)

1. **ARC-AGI-3 solve DEPTH — the deep tail is unsolved and L2+ is mechanic-locked.** 13 reproducible levels are
   almost all L1 first-contact solves (lp85 L3 is the exception). The deep-tail games resist the training-free
   graph-explore solver (mechanic-limited, confirmed at 30k expansions); the SOTA for FULL solves is the
   executable-world-model coding agent (arXiv:2605.05138, GPT-5.5 solved 15/25, RHAE 58.12%). Closing this gap is the
   direct path to a higher ARC score. **Operator MANDATORY (2026-06-17): E3 on ar25/ka59/tr87/ft09.**

2. **The verifier moat in GENERATION — proven once (n=40, one corpus), needs hardening to headline.** exp4315 is the
   first oracle-distinct verifier win in generation and is potentially gate-flipping for DiffusionGemma — but a single
   modest-n multiple-choice corpus is not a headline. Scaling/replicating it (2nd corpus + power, then adaptive guided
   generation) is the highest-leverage DEPTH on the project's strongest open claim (north-star §1: advance the headline
   or it's noise).

3. **The verifier is DOMAIN-BOUND — cross-domain selection retired, cross-game transfer nulled.** The constructive
   response (north-star §0 step 2: "verifier domain expansion … toward the perception/grid/rule-induction domains
   ARC-AGI-3 needs") is a LEARNED FRAME ENCODER that carries a game-invariant progress signal where generic grid
   features failed — the self-learning frontier AND the ARC efficiency axis (fewer search states on a new game) in one.

---

## 3. Architecture — where .400 acts

```
            ARC-AGI-3 (the north star: accuracy = solved levels, efficiency = actions + cost)
                                          │
         ┌────────────────────────────────┼───────────────────────────────────┐
         │ GENERATOR (commodity)           │ VERIFIER (Carnot's value-add)      │ SELF-LEARNING
         │                                 │                                    │
   ┌─────▼─────────────┐         ┌─────────▼──────────────┐          ┌──────────▼───────────┐
   │ coding agent       │         │ WorldModelVerifier      │          │ learned frame encoder │
   │ (codex/gpt-5.5)    │ induces │ grounds the induced     │  PHASE D │ over solved-game      │
   │ → Python world     │────────▶│ model (reproduction %)  │◀─────────│ traces → cross-game   │
   │   model  [PHASE B] │         │ [execution-grounded]    │          │ value transfer        │
   └────────────────────┘         └─────────────────────────┘          └───────────────────────┘
                                          │
   ┌────────────────────┐         ┌───────▼─────────────────┐
   │ DiffusionGemma 26B  │ guided  │ partial-state scorer     │  PHASE A (HEADLINE)
   │ (open, Apache-2.0)  │────────▶│ as EXTERNAL process      │  reward-guided step-stitching /
   │ denoising  [PHASE A]│ by      │ reward [oracle-distinct] │  adaptive guided generation
   └────────────────────┘         └──────────────────────────┘  → flip the DiffusionGemma gate?
                                          │
   ┌────────────────────┐         ┌───────▼─────────────────┐
   │ graph-explore (v2/v3)│ acts   │ reproduction gate        │  PHASE C (shallow tail)
   │ training-free        │───────▶│ (only reproduced counts) │  + tn36 ACTION6 click-schema RE
   └────────────────────┘         └──────────────────────────┘
```

The energy verifier appears in THREE load-bearing roles: (A) an EXTERNAL process reward steering DiffusionGemma
generation (oracle-distinct, the headline); (B) the `WorldModelVerifier` grounding an induced executable world model
(execution-grounded, the ARC deep tail); (C) a learned value-head over solver state for search efficiency
(oracle-distinct, self-learning).

---

## 4. Phases

### PHASE 0 — TRANSITION
- **exp4324** archive `.399 → activate `.400; record the TRUE `.399 close-state (in-generation moat CLOSED
  oracle-distinct; cross-domain RETIRED domain-bound; efficiency always-energy-dominates; ARC 13 reproduced + first
  live submission; cross-game transfer null; off-ARC execution marginal-win). `agent_type: codex` (mechanical).

### PHASE A — HEADLINE: scale & harden the in-generation oracle-distinct moat win (→ DiffusionGemma gate)
- **exp4325** REPLICATE the exp4315 reward-guided step-stitching win on a **2nd oracle-distinct corpus + more power**
  (n≥80, more seeds), same skeptic-proof harness (engaged control + self-reward SMC + no-op guard + independent leak
  re-check). Confirms the moat is not corpus-specific. `verifier_is_oracle=false`.
- **exp4326** ADAPTIVE guided-generation **scale-up** (2606.08501 reward-state alignment / 2606.13565 A2D2 / 2509.25171
  TR2-D2 — verify the IDs in PRECONDITIONS): the leak-checked scorer as the reward for bounded adaptive DiffusionGemma
  generation, with a NO-ADAPTATION control + frozen held-out + leak check. Aspiration: run it on **ARC-grid generation**
  (unifying the headline with the north star). `verifier_is_oracle=false`.

### PHASE B — ARC NORTH STAR: executable-world-model solver on the deep tail (operator MANDATORY 2026-06-17)
- **exp4327** E3 **ar25** (the validated game — gpt-5.5 induced a genuine flood-fill world model, verifier grounded it
  at 61%): drive the multi-round refactor loop + win-seeking exploration to **L1**. The codex agent IS the proposer.
  `verifier_is_oracle=true` (execution-grounded — ARC progress, NOT a moat headline).
- **exp4328** E3 **ka59** → L1.
- **exp4329** E3 **tr87 + ft09** (+1 level each, checkpoint per game, hard per-game wall-time cap).

### PHASE C — ARC NORTH STAR: adapter-free discovery sweep (shallow tail) + tn36 delta-RE
- **exp4330** training-free graph-explore discovery sweep (2512.24156) over the remaining unsolved SHALLOW-tail games
  (bp35/dc22/g50t/lf52/re86/s5i5/sb26/vc33) at the 12k discovery budget; reverse-engineer tn36's ACTION6 click-payload
  schema (its per-game delta); capture + reproduction-gate any new L1 solves. Monotonic +1+. `verifier_is_oracle=true`.

### PHASE D — CONTINUOUS SELF-LEARNING (mandatory; NEW mechanism)
- **exp4331** train a **LEARNED FRAME ENCODER** over the 11 solved games' traces; re-test leave-one-game-out cross-game
  value transfer for SEARCH EFFICIENCY (the gap exp4318 logged — generic features failed). `verifier_is_oracle=false`.

### PHASE E — INFRA + HYGIENE + CAPSTONE
- **exp4332** SOTA-ingestion → `.401 (reliable channel only; /deep-research banned in-loop).
- **exp4333** verifier registry/gaps hygiene + GAP-4 execution regression guard (robust aggregate-available helper).
- **exp4334** hardware continuity (opportunistic per north-star §3; KV260 SSH-only, never host SD-card).
- **exp4335** capstone `.400 — the verifier scorecard + the **DiffusionGemma gate decision** (did the oracle-distinct
  in-generation win replicate across ≥2 corpora with matched controls + CI95-excl-0?) + G1–G4 via `publication_gate.py`.

---

## 5. Dependency graph

```
exp4324 (transition)
   ├─ PHASE A ─ exp4325 (replicate in-gen, 2nd corpus) ─┐
   │            exp4326 (adaptive guided-gen scale-up) ─┤
   ├─ PHASE B ─ exp4327 (E3 ar25) ──────────────────────┤
   │            exp4328 (E3 ka59) ──────────────────────┤
   │            exp4329 (E3 tr87+ft09) ─────────────────┤
   ├─ PHASE C ─ exp4330 (adapter-free sweep + tn36 RE) ─┤
   ├─ PHASE D ─ exp4331 (learned frame encoder xfer) ───┤
   └─ PHASE E ─ exp4332 (sota→v401) ────────────────────┤
                exp4333 (registry/gaps hygiene) ────────┤
                exp4334 (hardware continuity) ──────────┤
                exp4335 (capstone .400) ◀───────────────┘  reads exp4325/4326 (gate decision) + exp4327-4331
```

No hard `gated_on` gates: all phases are independent DEPTH and the capstone aggregates whatever lands (robust
aggregate-available helper from exp4308 — a missing artifact for one axis is a per-axis gap, never an all-False).

---

## 6. Hardware requirements

- **PHASE A (DiffusionGemma)**: 1× RTX 3090 (Q4_K_M GGUF, 16 GB) via the llama.cpp PR-#24423 binary
  `~/.cache/llama.cpp-master/build/bin/llama-diffusion-gemma-eval` (NOT a standard GGUF loader — known-issues
  2026-06-15). Partial-state scorer on CPU.
- **PHASE B (E3)**: codex/gpt-5.5 induces the world model; offline ARC sim (`environment_files/<game>/`), zero quota.
- **PHASE C (sweep)**: offline ARC sim, CPU, zero quota.
- **PHASE D (self-learning)**: CPU (small frame-encoder + value-head over cached traces).
- **PHASE E**: CPU / aggregation; hardware-continuity probes KV260 (SSH), PolarFire (SSH), GateMate (USB) opportunistically.

---

## 7. Discipline compliance

- **Codex-Default v2 (2026-06-10):** every task `agent_type: codex` + `model: gpt-5.5`. Planner/retro/audits stay Opus
  (this doc). E3 tasks REQUIRE codex (the codex agent IS the world-model proposer — do NOT nest a `CodexProposer`).
- **ARC-AGI-3 Incremental-Progress Scoping:** every solve task targets +1 level (NOT all-levels); monotonic.
- **ARC Solve Reproducibility + Solver-Reuse:** every solve emits `offline_reproduced` + `reproduced_levels`; only
  reproduced levels count; reuse `arc_solver_kit` / `arc_executable_world_model` + the registry; update the registry.
- **Circularity / Oracle-Distinctness:** every verifier task declares `verifier_is_oracle`. The in-generation +
  self-learning tasks are oracle-distinct (`false`, matched control, CI95-excl-0); the E3 + sweep solves are
  execution-grounded (`true`, ARC progress NOT a moat headline).
- **Failed-Experiment Rerun + Exclusion-Manifest Cross-Check:** retired/failed-scope tasks carry `prior_failures:`
  (all four sub-fields) or `operator_override:`. The cross-domain selection scope is RETIRED and NOT re-proposed.
- **Pre-Launch Preconditions / Inference-Substrate / Principle-Annotated Fields / Verdict Terminal-Prefix:** honored
  per-task.
- **SOTA-Ingestion Cycle + Missing-Verifier Gap Logging + Hardware-Task Continuity + Overdue-Priority (2 infra slots):**
  reserved (exp4332/4333/4334; transition + hygiene + sota + hardware = 4 infra slots).
- **Operator-Only External Publication:** no leaderboard submission; no public-doc edits; drafts off main.

---

## 8. Success criteria

A `.400 milestone is a success if it advances the north star or the headline:
1. **Headline:** the in-generation oracle-distinct moat win REPLICATES on a 2nd corpus with CI95-excl-0 (exp4325), and
   the gate decision (exp4335) records whether the DiffusionGemma gate flips. A powered failure-to-replicate is also
   decision-grade (the win was corpus-specific).
2. **ARC accuracy:** ≥1 deep-tail E3 solve (ar25/ka59/tr87/ft09 → L1, offline-reproduced) OR ≥1 new shallow-tail
   adapter-free solve → `reproducible_total_levels ≥ 14`. An honest E3 partial (induced model + residual-mismatch gap)
   is progress.
3. **Self-learning:** the learned frame encoder either transfers cross-game (reduction >1.0, CI lower bound >1.0) or
   logs a sharper game-invariant-representation gap.
4. Cross-domain selection is NOT re-run (retired); efficiency is NOT re-measured (settled); off-ARC is NOT
   re-accumulated (settled).
