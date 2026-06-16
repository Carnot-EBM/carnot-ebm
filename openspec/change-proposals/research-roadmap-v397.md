# Research Roadmap v397 — CLOSE THE LAST OPEN MOAT AXIS (cross-GENERATOR) + UNBLOCK THE §5 IN-GENERATION THESIS (a learned PARTIAL-STATE diffusion scorer) + HARDEN THE EFFICIENCY PARETO WIN

**Milestone:** 2026.06.397
**Planned:** 2026-06-16 (outer-loop, Claude Opus 4.8)
**Predecessor:** 2026.06.396 (`research-roadmap-v396.md`)
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3 accurately AND efficiently; the energy
verifier is the load-bearing value-add (router / pruner / scorer), the generator is commodity.

---

## 1. What `.396 proved (the honest scorecard)

`.396 ran the now-unlocked DiffusionGemma full run, attempted to harden the cross-family win on a 2nd
substrate, paid the owed efficiency axis, re-powered self-learning, and advanced ARC. Read via
`scripts/summarize_artifact.py` + the outer-loop audit doc:

| Phase | Result | State |
|---|---|---|
| **EFFICIENCY (exp4284)** | energy verifier **0.654** vs Qwen3.6-35B LLM-judge **0.212**, accuracy delta CI95 **[0.308, 0.577]** (excludes 0 — a **Pareto win**), `cost_ratio` **1.95e-08** (~50M× cheaper), `verifier_is_oracle=false` | ✅ **WON** — but the judge scored *below random* (0.25 baseline), so the win needs hardening against the "weak-prompt" critique |
| **DiffusionGemma guidance (exp4281)** | `diffusiongemma_guidance_moat=false`; verdict `cannot_score_partial_states`; all deltas 0.0 | ❌ **BLOCKED** — no learned verifier can score PARTIAL (masked) denoising states; the §5 in-generation question is UNANSWERED |
| **Cross-GENERATOR (exp4282)** | `cross_family_delta=1.0` CI[1.0,1.0] — **FLAGGED degenerate** (wrong-majority-only pool + 4 candidates → trivially separable; vote@1=0.0 structural) | ⚠️ **STILL OPEN** — measures pool degeneracy, not transfer; the within-pool win (exp4271, +0.40) STANDS |
| **Self-learning (exp4283)** | `online_adaptation_helps=false`, powered "static is the ceiling" — but **FLAGGED TAUTOLOGY** (tier2==static==0.5, the tier-2 arm was a no-op); online=0.581 showed a sub-CI uplift | ⚠️ **BUGGY** — the powered null is suspect because the tier-2 arm did nothing |
| **ARC (exp4285)** | +1 to **21 levels** (game ls20) | ✅ monotonic progress |
| **Publication gate** | `paper_ready=True` (FoVer 0.9131, G1–G4 met) | ✅ unchanged |

**The capstone (exp4289) headline:** `diffusiongemma_thesis_state = partial_state_blocked`;
`guidance_moat_holds=False`; `cross_family_hardens_on_arcgen=False`; `verifier_efficiency_parity=True`.
Two of five science phases were excluded as flagged_adversarial (exp4282, exp4283).

**Net:** the verifier's value as a SELECTOR/RANKER is strong and now efficiency-proven; but the two
hardest open questions — (a) does the verifier transfer to a construction-disjoint GENERATOR, and (b)
can it IMPROVE generation (not just rank) — are both unanswered, the first by a degenerate test, the
second by a missing capability. `.397 attacks exactly those two, plus hardens the one clean win.

---

## 2. The 3 biggest gaps `.397 closes (Depth-Over-Breadth — north-star §1)

### GAP A — the cross-GENERATOR axis (the LAST open axis of the selection moat) — THE HEADLINE
The `.395 win generalized across families *we carved from one pool*. A skeptic's only remaining critique:
"that's your partition." ARC-GEN (arXiv:2511.00162, Google) is a construction-disjoint mimetic generator
with native family ids — the right 2nd substrate. The `.396 attempt FAILED because the pool was built
**wrong-majority-only with 4 candidates/task** → trivially separable → a degenerate +1.0. The fix is the
POOL CONSTRUCTION, not the substrate: a NON-degenerate ARC-GEN pool with realistic candidate counts,
`vote@1` well above 0, and an oracle ceiling below 1.0 — headroom the verifier must actually EARN.
Closing this makes the oracle-distinct moat cross-generator, not just cross-partition. (P0 #1.)

### GAP B — the §5 in-generation thesis is blocked on a MISSING CAPABILITY
"Does the external verifier IMPROVE generation, not just rank it?" cannot be tested until a learned
verifier can score PARTIAL (masked) DiffusionGemma canvases — and standard PRMs are "brittle or
ill-calibrated" on masked states (Prism, arXiv:2602.01842). `.397 BUILDS the missing partial-state scorer
(harness-first: a tested scorer + a LEAK audit), THEN — gated on a leak-free build — runs the
energy-guided generation with matched no-Carnot-verifier controls (unguided / RFG / EntRGi). The
deepest open §5 question, attacked at its real blocker.

### GAP C — the efficiency Pareto win needs hardening to headline-grade
The `.396 win is real and oracle-distinct, but the 35B judge scored below random — a skeptic blames the
prompt. `.397 replicates with a STRONGER judge prompt + a 2nd judge model, converting "beats a (possibly
badly-prompted) judge" into "matches/beats a *well-prompted* frontier judge at ~free cost" — the §5
efficiency headline (precedent: CompassVerifier 2508.03686, Calibrated Reasoning 2509.19681: small
verifiers rival frontier judges at 60–1000× lower cost).

Plus the mandated recurring work: continuous self-learning (fix the tier-2 bug), ARC +1, hardware
continuity, SOTA-ingestion, registry/gaps hygiene, and an INFRA safety net (the DEGENERATE_SEPARATION
check that would have caught exp4282 mechanically).

---

## 3. Architecture (where each phase plugs in)

```
                         ARC-AGI-3 (north-star §0: accuracy + efficiency)
                                          │
              ┌───────────────────────────┼───────────────────────────────┐
              │                           │                                │
       GENERATOR (commodity)      VERIFIER (Carnot value-add)        EFFICIENCY
       open LLM / TRM / codex      energy/learned, oracle-distinct    (the win condition)
              │                           │                                │
   ┌──────────┴───────────┐    ┌──────────┴───────────┐         ┌──────────┴─────────┐
   │ Phase B: DiffusionGemma│   │ Phase A: cross-GENERATOR│        │ Phase C: vs LLM-judge│
   │  energy-guided GEN     │   │  ARC-GEN non-degenerate │        │  stronger prompt +   │
   │  via a learned         │   │  pool → beats-vote on   │        │  2nd judge model     │
   │  PARTIAL-STATE scorer  │   │  held-out generators    │        │  (Pareto, ~free)     │
   │  (B1 build → B2 run)   │   │  (the LAST moat axis)   │        │                      │
   └────────────────────────┘   └─────────────────────────┘        └──────────────────────┘
              │                           │                                │
              └───────────── Phase D: continuous self-learning (Tier-1 + Tier-2 retrieval) ──┘
                                          │
                       Phase E: ARC +1 (monotonic, new game) ── Phase F: infra/hygiene/capstone
```

All verifier tasks declare `verifier_is_oracle: false` (oracle-distinct frontier, P0). Every moat/headline
claim carries a MATCHED no-verifier control with CI95-excl-0 (Circularity Discipline).

---

## 4. Phases & tasks (12 tasks, exp4290–exp4301)

**Phase 0 — transition**
- **exp4290** archive `.396 → activate `.397; record the `.396 scorecard truthfully (efficiency Pareto win;
  DiffusionGemma partial-state-blocked; cross-generator still-open/degenerate; self-learning tier-2 bug; ARC 21).

**Phase A — CLOSE THE CROSS-GENERATOR AXIS (the headline; P0 oracle-distinct)**
- **exp4291** ARC-GEN cross-generator, **NON-degenerate pool rebuild**. Realistic candidate counts (NOT
  wrong-majority-only), `vote@1 ∈ (0,1)`, oracle ceiling < 1.0. Train the set-encoder on train-generators,
  test beats-vote on HELD-OUT generators. Emit `cross_generator_holds` := (held-out delta>0 AND CI95-excl-0
  AND non-degenerate guards pass: vote@1>0.05, oracle<1.0, delta<0.95). `verifier_is_oracle=false`.

**Phase B — UNBLOCK THE §5 IN-GENERATION THESIS (harness-first)**
- **exp4292** BUILD + GATE a learned **partial-state diffusion scorer** for DiffusionGemma masked canvases
  (Prism 2602.01842 / Manta-LM 2605.14531 template), with a **LEAK audit** (does partial-state scoring read
  the final answer → circular?). Deliverable = a tested scorer that scores partial states non-degenerately on
  a held-out fixture. Emit `partial_state_scorer_built` (bare bool), `partial_state_leak_free` (bare bool),
  `partial_state_auroc`. `verifier_is_oracle=false`.
- **exp4293** *(GATED on exp4292 `partial_state_scorer_built==True` AND `partial_state_leak_free==True`)*
  DiffusionGemma energy-guided full run USING the learned partial-state scorer + matched controls (unguided /
  RFG 2509.25604 / EntRGi 2602.05000) + a guidance-dynamics diagnostic (2506.10971). Emit
  `diffusiongemma_guidance_moat` := (Carnot-guided − RFG > 0 AND CI95-excl-0). `verifier_is_oracle=false`.

**Phase C — HARDEN THE EFFICIENCY HEADLINE (§5 efficiency; oracle-distinct; skeptic-proof)**
- **exp4294** Replicate the energy-verifier vs LLM-judge head-to-head on cross-family ARC selection with a
  STRONGER judge prompt + a 2nd judge model (gemma-4-31B). Emit `efficiency_pareto_holds` := (energy accuracy
  within/above the BEST-prompted judge's CI AND `cost_ratio` ≤ 0.1). `verifier_is_oracle=false`.

**Phase D — CONTINUOUS SELF-LEARNING (mandated; fix the `.396 bug)**
- **exp4295** Re-run self-learning with the Tier-2 constraint-memory arm FIXED (the `.396 tier2==static==0.5
  tautology was a no-op) + a Tier-2 RETRIEVAL-augmented selector-context arm (Decocted 2604.04373:
  retrieval-only context, NO weight mutation, NO LoRA). Emit `online_adaptation_helps` := (best-adaptive −
  static > 0 AND CI95-excl-0). `verifier_is_oracle=false`.

**Phase E — ARC NORTH STAR (mandated monotonic +1)**
- **exp4296** ARC +1 on a NEW game (`total_levels >= 22`; NOT ls20/wa30/sc25), hardened set-encoder routing.

**Phase F — INFRA + HYGIENE + CAPSTONE**
- **exp4297** INFRA safety net: add a **DEGENERATE_SEPARATION** check to `scripts/adversarial_verify.py`
  (flag delta ≥ ~0.95 AND vote@1 ≤ ~0.05, or oracle@K==1.0 with a perfect selector) + a unit test that the
  check now flags exp4282. Prevents recurrence of the `.396 degenerate-pool waste.
- **exp4298** SOTA-ingestion → `.398 (reliable channel; /deep-research banned in-loop).
- **exp4299** registry/gaps hygiene + GAP-4 execution regression guard (no regression vs `.396).
- **exp4300** hardware continuity (opportunistic per north-star §3: KV260 SSH-only, PolarFire, GateMate).
- **exp4301** capstone `.397 (UNGATED) — the verifier scorecard: did the cross-generator moat close? did the
  in-generation thesis unblock? did the efficiency headline harden? G1–G4 via `publication_gate.py`.

---

## 5. Dependency graph

```
exp4290 (archive/activate)
   ├─► exp4291 (cross-generator, NON-degenerate pool) ──────────────────┐
   ├─► exp4292 (partial-state scorer BUILD + leak audit) ──► exp4293 (DiffusionGemma run, GATED on 4292)
   ├─► exp4294 (efficiency harden: stronger prompt + 2nd judge) ─────────┤
   ├─► exp4295 (self-learning: tier-2 fixed + retrieval) ────────────────┤
   ├─► exp4296 (ARC +1, new game) ───────────────────────────────────────┤
   ├─► exp4297 (infra: DEGENERATE_SEPARATION check) ─────────────────────┤
   ├─► exp4299 (registry/gaps + GAP-4 guard) ────────────────────────────┤
   ├─► exp4300 (hardware continuity) ────────────────────────────────────┤
   └─► exp4298 (SOTA ingestion → .398)                                    │
                                                                          ▼
                                                          exp4301 (capstone .397, aggregates all)
```

Only one hard gate: exp4293 ⟵ exp4292 (no point running a guided-generation benchmark without a leak-free
partial-state scorer). Everything else is independent; the capstone aggregates whatever lands.

---

## 6. Hardware & substrate requirements

| Task | Substrate | Hardware |
|---|---|---|
| exp4291 (cross-generator) | `verifier_ensemble_against_cached_candidates` | CPU (ARC-GEN clone + set-encoder) |
| exp4292 (partial-state scorer build) | `live_llm_inference` | 1× RTX 3090 (DiffusionGemma Q4_K_M 16GB) + the llama.cpp PR binary `llama-diffusion-gemma-eval` |
| exp4293 (DiffusionGemma run, gated) | `live_llm_inference` | 1× RTX 3090 + the PR binary (the ONLY working loader) |
| exp4294 (efficiency harden) | `live_llm_inference` | cached Qwen3.6-35B + gemma-4-31B GGUF (judges); CPU for the energy verifier |
| exp4295 (self-learning) | `verifier_ensemble_against_cached_candidates` | CPU |
| exp4296 (ARC +1) | `offline_arc_agi3_solver_incremental_progress` | CPU |
| exp4290/4297/4298/4299/4300/4301 | aggregation / infra / hardware_smoke | CPU (+ SSH boards for 4300) |

**HARD RULES (every task):** the TRM Sudoku checkpoint is DONE (val 0.8227) and the conductor stays
stood-down — NO task launches TRM training, runs pkill/kill against train.py, or writes `results/trm_runs/`.
Qwen is FORBIDDEN as a TRAINED base (Spurious-Rewards confound); Qwen GGUF as an off-policy judge is fine.
Every verifier-value task declares `verifier_is_oracle` honestly (Circularity Discipline) — an execution-grounded
win is `execution_grounded` (cheap, valid, NOT a moat); the moat is the LEARNED result with a matched control,
CI95-excl-0. NO autonomous edits to `docs/index.html` / README / paper prose. Online ARC play stays
operator-gated (NO leaderboard submission). DiffusionGemma MUST use the PR binary, not a standard GGUF loader.

---

## 7. Success criteria (what `.397 must report)

- `cross_generator_holds` (BARE bool) — did the oracle-distinct selection win transfer to held-out
  ARC-GEN *generators* on a NON-degenerate pool? (the LAST moat axis)
- `partial_state_scorer_built` + `partial_state_leak_free` — was the missing capability built, leak-free?
- `diffusiongemma_guidance_moat` (if the gate opened) — did the external verifier IMPROVE generation vs RFG?
- `efficiency_pareto_holds` — does the verifier match a well-prompted frontier judge at ≤0.1× cost?
- `online_adaptation_helps` — does cheap Tier-1/Tier-2 adaptation beat the static selector (powered, bug-fixed)?
- `total_levels >= 22` — monotonic ARC progress.
- `paper_ready` (G1–G4) — the FoVer headline stays the publication target; a verified cross-generator moat
  or in-generation win would be a new headline-grade supporting result.

Honest negatives are decision-grade: a cross-generator collapse SCOPES the moat to within-pool; a
partial-state leak or block keeps the §5 in-generation thesis open with a named, measured blocker; a powered
self-learning null retires the online-adaptation ask. None is churn — each moves a load-bearing question.
