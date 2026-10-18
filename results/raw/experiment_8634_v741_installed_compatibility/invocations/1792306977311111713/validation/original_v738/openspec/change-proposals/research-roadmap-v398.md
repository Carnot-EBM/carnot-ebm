# Research Roadmap v398 — PROVE EFFICIENCY-PARITY (the §5 win condition) + ESTABLISH THE IN-GENERATION MOAT WITH DIFFERENTIATED CONTROLS + BROADEN THE SELECTION MOAT TO CROSS-DOMAIN

**Milestone:** 2026.06.398
**Planned:** 2026-06-16 (outer-loop, Claude Opus 4.8)
**Predecessor:** 2026.06.397 (`research-roadmap-v397.md`)
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3 accurately AND efficiently; the energy
verifier is the load-bearing value-add (router / pruner / scorer), the generator is commodity.
§5 win condition (operator 2026-06-06): the verifier earns its place if it is **equally effective
as the LM at lower cost/latency** (efficiency-parity / Pareto-dominate an LLM-as-judge).

---

## 1. What `.397 proved (the honest scorecard)

`.397 closed the last open SELECTION-moat axis and built the missing in-generation capability, then
ran into two harness bugs (a degenerate-controls guided run + a doomed-looped efficiency task) and one
capstone-aggregation robustness bug. Read via `scripts/summarize_artifact.py` + the outer-loop audit
docs (`exp4301-capstone-blocked-spurious-false`, `exp4293-in-generation-moat-degenerate-controls`):

| Phase | Result | State |
|---|---|---|
| **Cross-GENERATOR (exp4291)** | `cross_generator_delta` **+0.50**, CI95 **[0.29, 0.71]** (excludes 0), `vote@1=0.25`, `oracle@K=0.75` (real headroom), non-degenerate guards PASS, `verifier_is_oracle=false`, n=8 generators / 24 held-out tasks | ✅ **CLOSED** — the oracle-distinct selection win transfers to construction-disjoint ARC-GEN generators; the LAST open axis of the selection moat |
| **Partial-state scorer (exp4292)** | `partial_state_scorer_built=true`, `partial_state_leak_free=true`, `partial_state_auroc=0.966`, `leak_ablation_auroc=0.937` | ✅ **BUILT** leak-free — but 0.966 on *masked* states is a yellow flag warranting an independent leak re-check |
| **In-generation moat (exp4293)** | `diffusiongemma_guidance_moat=true` **but FLAGGED TAUTOLOGY** — `condition_accuracy: {carnot 0.867, rfg 0.3, unguided 0.3, entrgi 0.3}`: three distinct controls bit-identical = no-op signature; `carnot−rfg == carnot−unguided == 0.567` | ❌ **NOT held** — "Carnot beats a no-op," not "Carnot beats the model's self-guidance." Quarantined (correctly) |
| **Efficiency harden (exp4294)** | **no artifact** — C1 failed 3× (2h wall-clock cap + a `strong_judge.py` bug) and was 3-fail-skipped | ❓ **UNRESOLVED** — a failed task, NOT a measured null; the `.396 Pareto win (energy 0.654 vs judge 0.212) still stands un-hardened |
| **Self-learning (exp4295)** | `online_cross_family_delta=0.483` vs `static=0.417` (+0.067); tier-2 fixed (memory 0.428, retrieval 0.456 — no longer a no-op) | ✅ online adaptation helps; needs a powered CI to be decision-grade |
| **ARC (exp4296)** | +1 to **22 levels** (game r11l → L1) | ✅ monotonic progress |
| **INFRA (exp4297)** | DEGENERATE_SEPARATION check added to `adversarial_verify.py`, regression test passed | ✅ the safety net that would have caught the `.396 exp4282 |
| **Hygiene (exp4299) + Capstone (exp4301)** | both `blocked_v397_artifacts_missing` — one missing artifact (exp4294) hard-blocked the whole aggregation, defaulting ALL booleans to False | ⚠️ **SPURIOUS** — a capstone-aggregation robustness bug, NOT the real result |
| **Publication gate** | `paper_ready=True` (FoVer 0.9131, G1–G4) as last computed in `.395 | ✅ unchanged; the cross-generator close STRENGTHENS the selection claim |

**Net.** The SELECTION moat is now proven on BOTH axes (within-pool cross-family `.395 +0.40 AND
cross-generator `.397 +0.50, both oracle-distinct, both CI95-excl-0). The two unresolved §5 questions
are (a) **efficiency-parity** — the operator's stated win condition, never hardened (the task failed to
run), and (b) **does the verifier IMPROVE generation** — the scorer exists but the guided run's
controls were degenerate. Plus the selection moat is proven within ONE domain (ARC); cross-DOMAIN
generalization (escaping the math-domain-bound limitation) is unproven. `.398 attacks exactly these,
plus fixes the two harness bugs mechanically.

---

## 2. The 3 biggest gaps `.398 closes (Depth-Over-Breadth — north-star §1 / §5)

### GAP A — EFFICIENCY-PARITY is the operator's STATED win condition and it is UNRESOLVED — THE HEADLINE
North-star §5: "the verifier earns its place if it is equally effective as the LM at lower
cost/latency." The `.396 Pareto win (energy 0.654 vs Qwen3.6-35B judge 0.212, ~50M× cheaper) is real
but the judge scored *below random* (a weak-prompt confound), and the `.397 hardening (exp4294)
**never ran** — it doomed-looped on the 2h cap with two judges + strong CoT prompts. `.398 RE-SCOPES
the head-to-head to fit the window — single-synchronous-resume-accumulate, checkpoint after EVERY
judge call, per-task progress prints (the codex idle-timeout fix, memory
`powering-run-background-mechanism-fails`), and a PARTIAL (one well-prompted judge, n≥30) accepted.
**Protocol upgrade (new SOTA, 2026-06-16 sweep):** report the head-to-head as a **fixed-compute /
iso-FLOPs accuracy curve** (Budget-aware Discriminative Verification, arXiv:2510.14913), not "1
verifier call vs 1 judge call" — a discriminative verifier (one forward pass) vs a CoT judge is a clean
iso-compute comparison. Build the STRONG judge per "Thinking Small Judges" (arXiv:2509.13332: thinking
judges +~10pp at <2× FLOPs; few-shot >8× cost for modest gain — itself ammunition for the
Pareto-dominance claim). Emit `efficiency_pareto_holds` := (energy accuracy within/above the
best-prompted judge's CI AND `cost_ratio` ≤ 0.1). Oracle-distinct, the win condition itself.
(Precedent: CompassVerifier 2508.03686, Calibrated Reasoning 2509.19681.)

### GAP B — the §5 in-generation moat is OPEN: the capability exists but the controls were degenerate
"Does the external verifier IMPROVE generation, not just rank it?" The partial-state scorer is BUILT
leak-free (exp4292), but exp4293's guided run had three bit-identical no-op controls (rfg = unguided =
entrgi = 0.3), so "Carnot 0.867 beats RFG 0.3" was "Carnot beats a no-op." **The precise fix (2026-06-16
sweep, the key finding):** RFG (arXiv:2509.25604) is the log-likelihood ratio of an *enhanced* dLLM over
a *reference* dLLM, and it **no-ops to == unguided ONLY when enhanced == reference**. So the `.397 RFG
arm fell back to unguided because it had no distinct reference. The fix: construct RFG as a
strictly-weaker reference (base / un-post-trained / lower-temperature DiffusionGemma) vs the enhanced
generator, and **sweep guidance strength γ > 0** (γ=0 recovers unguided), using arXiv:2506.10971 to pick
γ in the provably-engaged regime. EntRGi (arXiv:2602.05000) is a SINGLE-model entropy-gated guidance
that engages without a 2nd checkpoint — so it is the guaranteed-engaged non-Carnot control even if the
RFG reference is unavailable. `.398 re-runs with (1) an INDEPENDENT leak re-check on the exp4292 scorer
(AUROC 0.966 yellow flag), (2) ≥1 GENUINELY-ENGAGED non-Carnot control + a mechanical no-op guard that
REJECTS the run if any two arms tie bit-identically, (3) only then is `carnot − {best engaged control}`
a valid moat test. (Architectural prior: EDLM, arXiv:2410.21357 — an EBM correcting a discrete diffusion
LM; Carnot's oracle-distinct external verifier is the differentiator.)

### GAP C — the selection moat is proven in ONE domain (ARC); cross-DOMAIN is unproven
The moat now holds across families and across generators — but all WITHIN ARC. The verifier is
domain-bound (math strong, code weak, facts earned-negative — memory `verifier-domain-bound-math-only`).
The `.397 SOTA ingestion (exp4298) flagged exactly this: now that same-substrate cross-generator
closed, BROADEN to cross-DOMAIN selector generalization — one router + per-domain set-encoder over
ARC + ARC-GEN + a 3rd domain (FoVer-step / math selection), trained on train-DOMAINS, tested beats-vote
on a HELD-OUT DOMAIN frozen. **Method (2026-06-16 sweep):** EEVEE interleaved router↔prompt
co-evolution (arXiv:2606.11182) is the anti-overfit template; DG-PRM (arXiv:2507.17849) — a
multi-dimensional reward tree with dynamic per-step signal selection — is the selector-side route to OOD
generalization (give the verifier multiple invariant dimensions, select which apply per domain), mapping
onto the `verifier_gaps.md` missing-discriminator program; GEPA (arXiv:2605.19633) is the optimization
engine. **Anti-leak (flagged by the sweep): EEVEE's abstract does NOT detail domain-identity-leak
prevention — so `.398 MUST add a label-ablation arm** (router sees only features vs router sees the
family/domain label) to prove the router is not just reading domain identity, plus the held-out-domain
freeze. A held-out-domain win is the strongest selection result yet and tests whether the moat escapes
the domain bound; a collapse is a genuine, decision-grade scope boundary. NEW axis (cross-domain), not
churn (cross-generator was within-domain). `verifier_is_oracle=false`.

Plus the mandated recurring work: continuous self-learning (powered CI, cross-domain regime), ARC +1,
hardware continuity, SOTA-ingestion → `.399, registry/gaps hygiene, and an INFRA safety net that fixes
BOTH `.397 harness bugs mechanically (a DEGENERATE_CONTROLS check + a robust capstone aggregator).

---

## 3. Architecture (where each phase plugs in)

```
                         ARC-AGI-3 (north-star §0: accuracy + efficiency)
                                          │
        ┌─────────────────────┬──────────┴───────────┬─────────────────────┐
        │                     │                       │                     │
  EFFICIENCY (§5 win)   IN-GENERATION (§5)      SELECTION moat        SELF-LEARNING
  Phase A               Phase B                 Phase C               Phase D
        │                     │                       │                     │
 ┌──────┴───────┐    ┌────────┴────────┐    ┌─────────┴─────────┐  ┌────────┴────────┐
 │ energy vs    │    │ DiffusionGemma  │    │ cross-DOMAIN      │  │ online + Tier-2 │
 │ WELL-PROMPTED│    │ energy-guided   │    │ router over       │  │ retrieval       │
 │ judge + 2nd  │    │ gen, ENGAGED    │    │ ARC+ARC-GEN+FoVer │  │ (Dynamic        │
 │ model;       │    │ controls (RFG   │    │ held-out DOMAIN   │  │ Cheatsheet/     │
 │ iso-FLOPs    │    │ enhanced≠ref,   │    │ frozen + label-   │  │ Decocted),      │
 │ curve, ≤0.1× │    │ γ>0) + no-op    │    │ ablation (EEVEE)  │  │ POWERED CI      │
 │              │    │ guard + leak    │    │                   │  │                 │
 │              │    │ re-check        │    │                   │  │                 │
 └──────────────┘    └─────────────────┘    └───────────────────┘  └─────────────────┘
        │                     │                       │                     │
        └──────────── Phase E: ARC +1 (new game) ── Phase F: infra/hygiene/capstone ──┘
```

All verifier tasks declare `verifier_is_oracle: false` (oracle-distinct frontier, P0 2026-06-14).
Every moat/headline claim carries a MATCHED no-verifier control with CI95-excl-0 (Circularity
Discipline). The two `.397 harness bugs (degenerate controls, hard-block-all-False capstone) are fixed
mechanically in Phase F so they cannot recur.

---

## 4. Phases & tasks (11 tasks, exp4302–exp4312)

**Phase 0 — transition**
- **exp4302** archive `.397 → activate `.398; record the TRUE `.397 scorecard (cross-generator CLOSED;
  partial-state scorer built leak-free; in-generation NOT held [degenerate controls]; efficiency
  UNRESOLVED [task failed]; self-learning online-helps; ARC 22). Frame `.398 as prove-efficiency-parity
  + establish-in-generation-with-differentiated-controls + broaden-to-cross-domain.

**Phase A — PROVE EFFICIENCY-PARITY (the §5 win condition; re-scope the failed C1; THE HEADLINE)**
- **exp4303** energy-verifier vs LLM-judge on cross-family ARC selection with a STRONGER judge prompt
  (few-shot + explicit grid-reasoning + CoT, per 2509.13332) + a 2nd judge model (gemma-4-31B alongside
  Qwen3.6-35B), reported as a fixed-compute / iso-FLOPs accuracy curve (2510.14913), RE-SCOPED to fit
  the 2h window: synchronous resume-accumulate, checkpoint after every judge call, per-task progress
  prints, a PARTIAL (one well-prompted judge, n≥30) accepted. Emit `efficiency_pareto_holds` := (energy
  accuracy within/above the BEST-prompted judge's CI AND `cost_ratio` ≤ 0.1). `verifier_is_oracle=false`.

**Phase B — ESTABLISH THE IN-GENERATION MOAT WITH DIFFERENTIATED CONTROLS (deepest §5; fix exp4293)**
- **exp4304** DiffusionGemma energy-guided run reusing the exp4292 partial-state scorer, with (1) an
  INDEPENDENT leak re-check on the scorer (AUROC 0.966 yellow flag — if it fails, emit
  `scorer_leaky_rebuild_needed`, in-generation stays open with a named blocker), (2) ≥1 GENUINELY-ENGAGED
  non-Carnot control — RFG with a strictly-weaker reference vs enhanced generator at γ>0 (2509.25604) OR
  EntRGi single-model entropy-gated guidance (2602.05000) — plus a mechanical no-op guard that REJECTS
  the run if any two arms tie bit-identically (the exp4293 signature), (3) the moat test. Emit
  `diffusiongemma_guidance_moat` := (Carnot−{best engaged control} > 0 AND CI95-excl-0 AND
  `controls_differentiated`==true). `verifier_is_oracle=false`. PRECONDITION: the exp4292 scorer module
  loads + its `.397 artifact has `partial_state_scorer_built==true`.

**Phase C — BROADEN THE SELECTION MOAT TO CROSS-DOMAIN (new axis; EEVEE; SOTA-ingestion flag)**
- **exp4305** a domain router + per-domain set-encoder over ≥3 domains (ARC families + ARC-GEN
  generators + FoVer-step/math selection), trained only on train-DOMAIN outcomes, tested beats-vote on a
  HELD-OUT DOMAIN frozen (EEVEE 2606.11182; DG-PRM multi-dimensional discriminators 2507.17849), with a
  LABEL-ABLATION arm (router sees only features vs sees the domain label) to prove it is not reading
  domain identity. Emit `cross_domain_selection_holds` := (held-out-domain delta>0 AND CI95-excl-0 AND
  non-degenerate guards: vote@1>0.05, oracle<1.0, delta<0.95). `verifier_is_oracle=false`. Implicitly
  replicates the cross-generator close (ARC-GEN is one train domain).

**Phase D — CONTINUOUS SELF-LEARNING (mandated; powered CI + cross-domain regime)**
- **exp4306** harden `online_adaptation_helps` with a POWERED bootstrap CI95 on (best-adaptive − static)
  in the cross-domain regime (more distribution shift), Tier-1 online reweighting + Tier-2 retrieval
  (Decocted 2604.04373 / Dynamic Cheatsheet 2504.07952: retrieval-only curated context, NO weight
  mutation, NO LoRA). Emit `online_adaptation_helps` := (best-adaptive − static > 0 AND CI95-excl-0).
  `verifier_is_oracle=false`.

**Phase E — ARC NORTH STAR (mandated monotonic +1)**
- **exp4307** ARC +1 on a NEW game (`total_levels >= 23`; NOT r11l/ls20/wa30/sc25), hardened
  set-encoder routing the offline solver's candidate ranking.

**Phase F — INFRA + HYGIENE + CAPSTONE**
- **exp4308** INFRA safety net (defense-in-depth; fixes BOTH `.397 harness bugs): (1) add a
  **DEGENERATE_CONTROLS** check to `scripts/adversarial_verify.py` (flag CRITICAL when ≥2 distinct
  control arms in a `condition_accuracy`/arms map are bit-identical — the no-op signature exp4293 hit) +
  a unit test that it flags exp4293 and NOT a genuinely differentiated run; (2) a reusable
  **aggregate-available-report-gaps** capstone helper (a missing artifact for one axis must NOT zero out
  a conclusive verdict on another — the exp4301/exp4299 bug) + a test. Do NOT modify
  `scripts/research_conductor.py`.
- **exp4309** SOTA-ingestion → `.399 (reliable channel; /deep-research banned in-loop).
- **exp4310** registry/gaps hygiene + GAP-4 execution regression guard (re-run; `.397's exp4299 was
  blocked by the aggregation bug, now reads available artifacts).
- **exp4311** hardware continuity (opportunistic per north-star §3: KV260 SSH-only, PolarFire, GateMate).
- **exp4312** capstone `.398 (UNGATED) — the verifier scorecard, using the robust aggregator from
  exp4308: did efficiency-parity harden? did the in-generation moat hold with differentiated controls?
  does cross-domain selection hold? G1–G4 via `publication_gate.py`.

---

## 5. Dependency graph

```
exp4302 (archive/activate)
   ├─► exp4303 (efficiency-parity hardened, re-scoped) ──────────────────┐
   ├─► exp4304 (in-generation, engaged controls + leak re-check) ────────┤
   ├─► exp4305 (cross-domain selector generalization) ───────────────────┤
   ├─► exp4306 (self-learning, powered CI, cross-domain) ─────────────────┤
   ├─► exp4307 (ARC +1, new game) ───────────────────────────────────────┤
   ├─► exp4308 (infra: DEGENERATE_CONTROLS + robust capstone aggregator) ─┤
   ├─► exp4310 (registry/gaps + GAP-4 guard) ────────────────────────────┤
   ├─► exp4311 (hardware continuity) ────────────────────────────────────┤
   └─► exp4309 (SOTA ingestion → .399)                                    │
                                                                          ▼
                                                          exp4312 (capstone .398, aggregates all)
```

No hard `gated_on` chains this milestone (the `.397 exp4292 scorer is a satisfied cross-milestone
prerequisite for exp4304 — checked as a PRECONDITION, not a gate). Everything is independent; the
capstone aggregates whatever lands via the robust exp4308 aggregator (so a single missing artifact can
no longer zero out the scorecard).

---

## 6. Hardware & substrate requirements

| Task | Substrate | Hardware |
|---|---|---|
| exp4303 (efficiency harden) | `live_llm_inference` | cached Qwen3.6-35B + gemma-4-31B GGUF (judges); CPU for the energy verifier |
| exp4304 (in-generation run) | `live_llm_inference` | 1× RTX 3090 (DiffusionGemma Q4_K_M 16GB) + the llama.cpp PR binary `llama-diffusion-gemma-eval` (the ONLY working loader) |
| exp4305 (cross-domain) | `verifier_ensemble_against_cached_candidates` | CPU (ARC + ARC-GEN + FoVer cached pools + set-encoder) |
| exp4306 (self-learning) | `verifier_ensemble_against_cached_candidates` | CPU |
| exp4307 (ARC +1) | `offline_arc_agi3_solver_incremental_progress` | CPU |
| exp4302/4308/4309/4310/4311/4312 | aggregation / infra / hardware_smoke | CPU (+ SSH boards for 4311) |

**HARD RULES (every task):** the TRM Sudoku checkpoint is DONE (val 0.8227) and the conductor stays
stood-down — NO task launches TRM training, runs pkill/kill against train.py, or writes
`results/trm_runs/`. Qwen is FORBIDDEN as a TRAINED base (Spurious-Rewards confound); Qwen GGUF as an
off-policy judge is fine. Every verifier-value task declares `verifier_is_oracle` honestly (Circularity
Discipline) — an execution-grounded win is `execution_grounded` (cheap, valid, NOT a moat); the moat is
the LEARNED result with a matched control, CI95-excl-0. NO autonomous edits to `docs/index.html` /
README / paper prose. Online ARC play stays operator-gated (NO leaderboard submission). DiffusionGemma
MUST use the PR binary, not a standard GGUF loader.

---

## 7. Success criteria (what `.398 must report)

- `efficiency_pareto_holds` (BARE bool) — does the verifier match a WELL-prompted frontier judge at
  ≤0.1× cost on an iso-compute curve? (the operator's §5 win condition, finally hardened)
- `diffusiongemma_guidance_moat` (BARE bool) — does the external verifier IMPROVE generation vs a
  GENUINELY-ENGAGED control (no two arms bit-identical)?
- `cross_domain_selection_holds` (BARE bool) — does the selection moat transfer to a HELD-OUT DOMAIN
  (with a label-ablation proving it is not reading domain identity)?
- `online_adaptation_helps` (BARE bool, powered) — does cheap Tier-1/Tier-2 adaptation beat static?
- `total_levels >= 23` — monotonic ARC progress.
- `paper_ready` (G1–G4) — the FoVer headline stays the publication target; a hardened efficiency-parity
  result, an established in-generation moat, or a cross-domain selection win would each be a new
  headline-grade SUPPORTING result.

Honest negatives are decision-grade: an efficiency parity-not-Pareto SCOPES the win condition to "as
good at lower cost" (still a win); an in-generation null with PROPERLY-ENGAGED controls retires the
"verifier improves generation" ambition (selection + efficiency remain the proven value); a cross-domain
collapse confirms the moat is domain-bound (a genuine boundary, consistent with the verifier-domain-bound
finding). None is churn — each moves a load-bearing question.

---

## 8. Literature ingested (2026-06-16 planning sweep — reliable channel, all arXiv-verified)

| Front | Method | arXiv | Maps to |
|---|---|---|---|
| Efficiency | Budget-aware Discriminative Verification | 2510.14913 | iso-FLOPs accuracy-curve protocol for exp4303 |
| Efficiency | Thinking Small Judges | 2509.13332 | how to build the genuinely-strong CoT judge + FLOPs accounting |
| Efficiency | CompassVerifier / Calibrated Reasoning | 2508.03686 / 2509.19681 | small-verifier-rivals-frontier precedent |
| In-generation | RFG (reward-free guidance) | 2509.25604 | the RFG-no-op fix: enhanced≠reference + γ>0 |
| In-generation | EntRGi | 2602.05000 | single-model entropy-gated control (guaranteed-engaged) |
| In-generation | Guidance in masked discrete diffusion | 2506.10971 | pick γ in the provably-engaged regime |
| In-generation | EDLM | 2410.21357 | architectural prior (EBM corrects diffusion; oracle-distinct is our diff) |
| Cross-domain | EEVEE router↔prompt co-evolution | 2606.11182 | the cross-domain router template (+ add label-ablation for anti-leak) |
| Cross-domain | DG-PRM multi-dimensional reward tree | 2507.17849 | multi-discriminator route to OOD selector generalization |
| Cross-domain | GEPA / optimize_anything | 2605.19633 | the router-config optimization engine |
| Self-learning | Decocted / Dynamic Cheatsheet | 2604.04373 / 2504.07952 | retrieval-only curated context, NO weight mutation |

(TTARAG 2601.11443 surfaced for self-learning but DOES weight updates — excluded as wrong-fit.)
