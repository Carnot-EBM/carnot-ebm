# Research Roadmap v403 — Milestone 2026.06.403

**Status:** PROPOSED (pre-staged roadmap, outer-loop Claude Opus 4.8, 2026-06-17)
**Prior milestone:** 2026.06.402 (CONVERT-the-moat / E3-deeper / action-cost-self-learning)
**Milestone doc convention:** v7/v8 format (what prior proved → gaps → phases → dependency graph → invariants).

---

## TL;DR — the one-line thesis of .403

> **.401 PROVED the in-generation oracle-distinct verifier-moat exists (leak-robust, replicated, CI95-excl-0).
> .402 tried to CONVERT it into a generation gain with S3 — and the instrument BROKE (the arms were framed as
> multiple-choice selection, so best-of-K / self-reward-SMC / unguided collapsed to "pick the max-logit option";
> the no-op guard fired `controls_not_differentiable` + a TAUTOLOGY flag). The conversion question — does putting
> the proven oracle-distinct scorer INSIDE the denoising loop Pareto-improve generation at fixed compute? — is
> therefore STILL OPEN, not nulled. .403 RE-ATTEMPTS the headline with a FIXED, Prism-hardened harness (real
> token-by-token denoising generation, hierarchical trajectory search + partial-remask branching, DIFFERENTIATED
> controls, and the sharpest oracle-distinct must-beat: Carnot's EXTERNAL leak-robust scorer vs the model's own
> intrinsic Self-Verified Feedback). In parallel it drives the ARC north star DEEPER (operator MANDATORY
> 2026-06-17) and COMPOUNDS the .402 self-learning win (the learned action-cost heuristic that cut held-out
> env-actions 25→16) into a standing, experience-scaling planner.**

`paper_ready` stays **TRUE** (G1–G4; FoVer 0.9131 frozen, never substituted). .403 adds the conversion +
ARC-depth + compounding-efficiency LENSES, not a new headline.

---

## 1. What .402 proved (the TRUE close-state)

Read via `scripts/summarize_artifact.py` + `results/experiment_4357_capstone_v402.json`
(`verifier_thesis_state: "moat_proven_leak_robust_but_s3_utility_open"`):

| Axis | Result | Verdict |
|---|---|---|
| **HEADLINE — convert the moat (S3 verifier-guided search, exp4348)** | **UNRESOLVED — harness failure.** The arms were framed as MCQ **selection** (pick option A/B/C/D from option-logits), so best-of-K / self-reward-SMC / unguided all collapse to "argmax logit" → the three deltas were bit-identical (`0.266667`) → `adversarial_verify` fired **CRITICAL TAUTOLOGY** and the no-op guard returned `controls_not_differentiable`. `s3_moat_utility: open`. | ⚠️ Re-attempt (instrument broke; question OPEN) |
| **HEADLINE validation — PAPO alignment (exp4349)** | Correctly **blocked** (gated on the S3 win, which did not land). | ✅ gate worked |
| **ARC north star (exp4350/4351/4352 + outer-loop)** | **ka59 L1 newly cracked (+1 game); tn36 L7 reproduced.** sc25 L2 blocked (spell-delta gap); ar25 L2 blocked (action7 undo-stack gap); tr87/ft09 still mechanic-limited (world-model accuracy ≈0). Authoritative registry after outer-loop work (tu93 L1→L3, lp85 L4, etc.): **26 reproducible levels / 15 games** (capstone snapshot was 23/14). The fresh_env sweep over the 10 unsolved games got **0 unlocks** (honest negative — the remaining unsolved games are the hard spatial-planning ones). | ✅ +progress (operator mandate) |
| **Continuous self-learning (exp4353)** | **CLEAN WIN.** A learned A* **action-cost heuristic** trained on the solved-level traces cut held-out **env-actions-to-solve 25 → 16** (−9, −36%), `positive_control_passed=true`, `reproduction_gated=true`, `verifier_is_oracle=false`. The working+compounding mechanism (cross-game value-transfer is RETIRED, 3 nulls). | ✅ win |
| **Publication gate** | `paper_ready=True`, G1–G4 all pass (FoVer 0.9131, exp2850). Unmet gates: none. | ✅ unchanged |
| **Infra (exp4355)** | The .401 capstone CIRCULAR_MOAT_OVERCLAIM stamping bug is **FIXED** (the .402 capstone scans **0 flags**); GAP-4 regression guard passed; registries reconciled. | ✅ durable |

**The decisive fact for .403:** the S3 failure was a **harness bug, not a science null.** The leak-robust scorer
is real and proven (.401). The question of whether it makes generation *better at equal compute* — the north-star
§5 "earns its place" win condition realized **in generation** — has **never been validly tested.** That is the #1
gap and the .403 headline.

---

## 2. The 3 biggest gaps between current state and the PRD vision

### Gap 1 (HEADLINE) — the proven oracle-distinct moat is a MEASUREMENT; its UTILITY in generation is UNTESTED
north-star §5: *"the verifier earns its place if it is equally effective at lower cost/latency — Pareto-improve
the base generator."* .401 proved the in-generation moat EXISTS (the scorer separates good/bad partial states,
leak-robust). But "exists" ≠ "useful." The only test of utility (S3, .402) broke as a harness. **Closing this gap
= putting the proven external leak-robust scorer inside a *real* denoising-generation loop and showing it
Pareto-improves the generation at fixed NFE — vs best-of-N AND vs the model's own intrinsic self-verification.**
The SOTA-ingestion (exp4354) flagged **Prism-hardened verifier-guided search** (arXiv:2602.01842) as the single
strongest .403 method for exactly this.

### Gap 2 (ARC NORTH STAR — operator MANDATORY 2026-06-17) — deep-tail mechanics + blocked next-levels
ARC-AGI-3 is the north star (accuracy = solve-rate, efficiency = actions/RHAE). We are at 26 levels / 15 games and
the operator's standing E3 mandate is to keep advancing it monotonically (+1..+n per game). The frontier has three
distinct shapes: (a) **high-headroom cracked games** with un-reproduced deeper levels (sc25 has 5 live-recorded
levels, only L1 reproduced = +4 upside; tu93→L4; lp85→L5; tn36→L8); (b) **blocked-mechanic next-levels** (ar25 L2 =
action7 undo-stack gap, ka59 L2); (c) **mechanic-limited tails** (tr87/ft09, world-model accuracy ≈0 — the
explore-verify-plan loop cannot yet induce their dynamics). The verifier is load-bearing throughout
(consistency-energy grounds each induced world model, no oracle).

### Gap 3 (CONTINUOUS SELF-LEARNING) — the action-cost win is a single linear fit; the PRD wants learning that COMPOUNDS at inference speed
research-program.md "Continuous Self-Learning" mandates every milestone advance the self-learning architecture
(Tiers 1–4: *"Carnot must get smarter over time. Every query should make the next one faster and more accurate."*).
.402 produced a working mechanism (the learned action-cost heuristic, 25→16). The gap is turning a one-off linear
fit into a **deployed, experience-scaling planner**: wire it into `arc_solver_kit` as the standing A* cost so every
future solve is action-minimal by default, and PROVE it **compounds** (held-out actions-to-solve falls as the
solved-trace corpus grows — a learning curve), with an LLM-generated-heuristic stronger-function-class arm
(arXiv:2503.18809) as the upper-bound probe. This serves the first-class north-star EFFICIENCY axis (lower RHAE).

---

## 3. Architecture — where .403 acts

```
            ┌──────────────────────── THE NORTH STAR: solve ARC-AGI-3 ─────────────────────────┐
            │              accurately (solve-rate)  AND  efficiently (actions / RHAE)            │
            └───────────────────────────────────────────────────────────────────────────────────┘
                         ▲                          ▲                           ▲
    ┌────────────────────┴───────┐   ┌──────────────┴───────────┐   ┌───────────┴───────────────────┐
    │  GAP 1 — HEADLINE          │   │  GAP 2 — ARC depth        │   │  GAP 3 — self-learning that    │
    │  CONVERT the proven moat   │   │  (operator MANDATORY)     │   │  COMPOUNDS (efficiency axis)   │
    │                            │   │                           │   │                                │
    │  Prism-hardened verifier-  │   │  E3 executable world-model│   │  deploy + compound the .402    │
    │  guided DENOISING SEARCH:  │   │  + consistency-energy     │   │  action-cost heuristic:        │
    │  Carnot EXTERNAL leak-     │   │  verifier (load-bearing): │   │   - wire into arc_solver_kit   │
    │  robust scorer (exp4337)   │   │   B1 deeper cracked       │   │     as the standing A* cost    │
    │  INSIDE the loop, HTS +    │   │     (sc25 L2 / tu93 L4 /  │   │   - learning curve: actions    │
    │  partial-remask branching, │   │      lp85 L5 / tn36 L8)   │   │     vs corpus size (compounds?)│
    │  at FIXED NFE, vs:         │   │   B2 blocked-mechanic L2s │   │   - LLM-gen heuristic arm      │
    │   - unguided               │   │     (ar25 L2 / ka59 L2)   │   │     (2503.18809) upper bound   │
    │   - best-of-N@matched-NFE  │   │   B3 mechanic-limited     │   │                                │
    │   - Prism intrinsic SVF    │   │     tails tr87/ft09       │   │  verifier_is_oracle=false      │
    │     (THE oracle-distinct   │   │     (active-data inducer) │   └────────────────────────────────┘
    │      must-beat)            │   │                           │
    │  verifier_is_oracle=false  │   │  verifier_is_oracle=TRUE  │
    │  → PAPO alignment (gated)  │   │  (execution-grounded;     │
    └────────────────────────────┘   │   NOT a moat headline)    │
                                     └───────────────────────────┘

   Generator = commodity (DiffusionGemma via the llama.cpp PR binary / codex world-model proposer).
   Energy/learned scorer = the VERIFICATION + guidance + efficiency layer (Carnot's whole value-add).
```

---

## 4. The phases (11 tasks, exp4358–exp4368)

### PHASE 0 — TRANSITION
- **exp4358** — archive .402 → activate .403; record the TRUE .402 close-state (S3 harness-failed → moat utility
  OPEN; ARC 26/15; action-cost heuristic WON; capstone-stamp fix durable; paper_ready=True). `codex`, infra.

### PHASE A — THE HEADLINE: convert the proven moat into a generation GAIN (Prism-hardened, FIXED harness)
- **exp4359 (A1, critical)** — **Prism-hardened verifier-guided DENOISING SEARCH.** RE-ATTEMPT the .402 conversion
  with the bug fixed: a **real token-by-token denoising-generation loop** on a free-form executable task (NOT
  MCQ-selection), Prism Hierarchical Trajectory Search + local branching with partial remasking (arXiv:2602.01842),
  the .401 leak-robust scorer (exp4337 `.pkl`) as the **external** in-loop guidance, at a **FIXED NFE** budget vs
  three **DIFFERENTIATED** controls: (a) unguided single-pass, (b) best-of-N@matched-NFE (compute-matched
  must-beat), (c) **Prism intrinsic Self-Verified Feedback / self-reward SMC** (the model's own self-verification —
  the **sharpest oracle-distinct must-beat**). Reuse the skeptic-proof harness: the no-op DEGENERATE_CONTROLS guard
  (`controls_differentiated`), a metric-TAUTOLOGY guard, an independent scorer leak re-check, **branch-diversity
  receipts**, and **scorer-disagreement rows**. Emit `s3_guided_beats_control` BARE bool at n≥80, ≥3 seeds.
  `verifier_is_oracle=false`. A win = the moat graduates *exists → useful* (Pareto-improves generation). A clean
  null (controls differentiated, scorer leak-free, but Carnot does NOT beat best-of-N / SVF) is decision-grade and
  retires the in-generation scale-up direction. `codex`, `live_llm_inference`, max_turns 100.
- **exp4360 (A2, GATED on A1 `s3_guided_beats_control==true`)** — **PAPO reward-state-alignment diagnostic**
  (arXiv:2606.08501). IF A1 shows a gain, skeptic-proof it: is it from rewards aligned to AUTHENTIC denoising
  states (vs randomly-remasked / position artifacts)? The operator was twice-burned on DiffusionGemma over-claims
  — a gain that fails alignment is QUARANTINED + the residual logged as a missing-verifier gap. `codex`,
  `live_llm_inference`, max_turns 100.

### PHASE B — ARC NORTH STAR (operator MANDATORY 2026-06-17; incremental +1..+n per game; `verifier_is_oracle=true`)
- **exp4361 (B1)** — **E3 DEEPER on the high-headroom cracked games:** sc25 L2 (toward its 5 live-recorded levels,
  the single biggest upside +4), tu93 L4, lp85 L5, tn36 L8 — +1 each via explore-verify-plan (AERA 2605.25931 +
  Agent2World 2512.22336). Loop with per-target checkpoint + wall-time cap. `codex`, `live_llm_inference`.
- **exp4362 (B2)** — **E3 the BLOCKED-mechanic next-levels:** ar25 L2 (close the action7 undo-stack gap) + ka59 L2
  — extend the existing world models against the named hidden-rule gaps. `codex`, `live_llm_inference`.
- **exp4363 (B3)** — **E3 the MECHANIC-LIMITED tails tr87 + ft09** (world-model accuracy ≈0): apply the
  active-data-collection lever (the M2-v4 disambiguation that cracked vc33) ON TOP OF explore-verify-plan to induce
  the missing dynamics. +1 each; an honest partial (refined model + sharper gap) is progress. `codex`,
  `live_llm_inference`.

### PHASE C — CONTINUOUS SELF-LEARNING (mandated; BUILD ON the .402 win; `verifier_is_oracle=false`)
- **exp4364** — **Deploy + COMPOUND the learned action-cost heuristic.** (1) Wire the .402 win into
  `arc_solver_kit` as the standing A* cost (every future solve action-minimal by default). (2) Measure the
  COMPOUNDING learning curve: held-out env-actions-to-solve as a function of the number of solved-level traces in
  the training corpus (does more experience → fewer actions?), with the positive control. (3) Optional stronger-arm
  probe: an LLM-generated heuristic program (arXiv:2503.18809) vs the linear heuristic (fresh held-out, static
  analysis, reproduction-gated). Emit `action_efficiency_compounds` BARE bool. NOT cross-game value transfer
  (RETIRED). `codex`, cpu/aggregation.

### PHASE E — INFRA + HYGIENE + CAPSTONE
- **exp4365** — SOTA-ingestion → .404 (mandatory per SOTA-Ingestion Cycle Discipline; reliable channel only;
  verified arXiv IDs; flag A2D2/SEPO out-of-band). `codex`, max_turns 60.
- **exp4366** — registry/gaps hygiene + GAP-4 regression guard (the capstone-stamp fix is durable per .402; this is
  routine reconciliation). `codex`, max_turns 60.
- **exp4367** — hardware continuity KV260 (opportunistic, SSH-reachability precondition only). `codex`, max_turns 60.
- **exp4368** — **capstone .403 + the HEADLINE DECISION:** did the Prism-hardened search CONVERT the proven moat
  into a fixed-NFE generation gain (`s3_moat_utility ∈ {useful_generation_gain / proven_but_not_useful / open}`),
  validated by PAPO alignment? + the new `reproducible_total_levels` + `action_efficiency_compounds` + G1–G4.
  `codex`, max_turns 100.

---

## 5. Dependency graph (cascade-proof)

```
exp4358 (transition)
  ├─► exp4359 (A1 Prism-S3 headline) ──gated(s3_guided_beats_control==true)──► exp4360 (A2 PAPO)
  ├─► exp4361 (B1 deeper cracked)    ┐
  ├─► exp4362 (B2 blocked-mechanic)  ├─ independent ARC tasks (no cross-gates; per-target checkpoints)
  ├─► exp4363 (B3 mechanic tails)    ┘
  ├─► exp4364 (C self-learning compounds)   ── independent (CPU/offline)
  ├─► exp4365 (SOTA-ingestion)              ── independent
  ├─► exp4366 (hygiene + GAP-4)             ── independent
  ├─► exp4367 (KV260 continuity)            ── independent
  └─► exp4368 (capstone) ── reads 4359/4360/4361/4362/4363/4364 (robust aggregate-available; SKIP flagged; HONOR verifier_is_oracle)
```

Only ONE structured gate (`exp4360` on `exp4359.s3_guided_beats_control==true`) — so a null/blocked A1 cleanly
skips the PAPO sidecar's Sonnet call without cascade-blocking the milestone. Everything else is independent; the
capstone uses the robust aggregate-available-report-gaps helper (NO hard-block-all-False) proven in .402.

---

## 6. Invariants carried (unchanged from .402)

- `paper_ready=True` (G1–G4; frozen FoVer 0.9131 **never** silently substituted — .403 adds the
  conversion/ARC-depth/compounding LENSES, not a new headline).
- **Oracle-distinct discipline:** every verifier-value task declares `verifier_is_oracle` honestly. An
  EXECUTABLE-oracle ARC solve is `execution_grounded` (`verifier_is_oracle=true`, ARC progress, NOT a moat
  headline). The moat is the oracle-DISTINCT learned result (the in-generation scorer; `verifier_is_oracle=false`)
  with a matched control + CI95-excl-0.
- The in-generation oracle-distinct moat **EXISTS** (proven leak-robust, .401 exp4338) — .403 tests its UTILITY.
- Conductor **STOOD-DOWN on TRM training** — NO task launches TRM training, runs pkill/kill against `train.py`, or
  writes `results/trm_runs/`. Qwen FORBIDDEN as the TRAINED base (Spurious-Rewards confound); Qwen/Gemma GGUF as an
  off-policy judge/generator is fine.
- **A2D2 (2606.13565) + SEPO (2502.01384)** verifier-as-reward GENERATOR-training are OUT-OF-BAND / operator-owned
  — flagged in SOTA-ingestion, NOT auto-run in-loop. The .403 headline is a NO-TRAINING test-time search.
- **Cross-game value TRANSFER (exp4342) and cross-domain SELECTION (exp4314) are RETIRED** — do NOT re-propose.
- DiffusionGemma MUST use the **llama.cpp PR binary** (`llama-diffusion-gemma-eval`), NOT a standard GGUF loader.
- NO autonomous edits to `docs/index.html` / README / paper prose. Online ARC play stays operator-gated (NO
  leaderboard submission; only offline-reproduced levels count).
- KV260 = THE sovereignty story (opportunistic continuity); GateMate/PolarFire opportunistic.

---

## 7. Hardware requirements

- **exp4359 / exp4360 (DiffusionGemma):** 1× RTX 3090 (the Q4_K_M GGUF is 16 GB) + the llama.cpp PR binary. The
  leak-robust scorer (exp4337 `.pkl`) loads CPU-side. PRECONDITIONS gate every run (PR binary + GGUF + scorer
  present, else `blocked_<resource>`).
- **exp4361 / exp4362 / exp4363 (ARC E3):** offline `environment_files/<game>/` + codex/gpt-5.5 as the world-model
  proposer. No GPU. PRECONDITIONS: offline env present, harness imports.
- **exp4364 (self-learning):** CPU-only, offline (heuristic learning on solve traces; NO model training).
- **exp4367 (KV260):** SSH-reachability precondition ONLY (`ssh kria 'true'`); NEVER a host SD-card device path.

---

## 8. References (the .403 planning sweep — all arXiv-verified 2026-06-17, reliable channel + WebSearch/WebFetch)

| ID | Title | .403 role |
|---|---|---|
| **2602.01842** | **Prism: Hierarchical Search + Self-Verification for Discrete Diffusion LMs (ICML 2026)** | **HEADLINE** — HTS + partial-remask branching = the efficient verifier-guided search mechanism; its intrinsic SVF is the oracle-distinct must-beat (exp4359). |
| 2604.06260 | S3 Stratified Scaling Search for diffusion LMs | the base search policy; .403 fixes its .402 control-collapse with differentiated controls + tautology guards. |
| 2602.22871 | Test-Time Scaling via Reward-Guided Stitching (external PRM) | external-PRM step-stitching architecture (the in-generation-guidance family). |
| 2602.01849 | Self-Rewarding Sequential Monte Carlo for masked diffusion LMs | the intrinsic-confidence particle filter = the must-beat self-reward SMC arm. |
| 2509.25604 | RFG: Reward-Free Guidance for dLLM reasoning | a training-free guidance contrast arm. |
| 2606.08501 | PAPO: reward-state alignment for diffusion-LLM reasoning | the headline skeptic-proofing diagnostic (exp4360). |
| 2605.05138 | Executable World Models for ARC-AGI-3 | the E3 world-model-induction SOTA (Phase B). |
| 2605.25931 / 2512.22336 | AERA explore-verify-plan / Agent2World adaptive testing | the explore-verify-plan harness (Phase B). |
| 2503.18809 | Classical planning with LLM-generated heuristics | the stronger-function-class arm for the compounding action-cost heuristic (exp4364). |
| 2606.13565 / 2502.01384 | A2D2 / SEPO (verifier-as-reward generator TRAINING) | **OUT-OF-BAND / operator-owned** — flagged, NOT auto-run. |

Cross-refs: north-star.md §1/§2/§5; CLAUDE.md "Circularity / Oracle-Distinctness Discipline",
"ARC-AGI-3 Incremental-Progress Scoping", "ARC Solve Reproducibility + Solver-Reuse Discipline",
"SOTA-Ingestion Cycle Discipline"; `results/experiment_4354_sota_ingestion_v403.json`
(`flagged_for_v403: prism_hardened_s3_verifier_guided_search_v403`).
