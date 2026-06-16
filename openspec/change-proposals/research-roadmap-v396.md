# Research Roadmap v396 — SCALE the now-unlocked DiffusionGemma energy-guided full run (does the EXTERNAL verifier improve GENERATION, not just rank it?) + HARDEN the cross-family win on a 2nd substrate + pay the owed EFFICIENCY axis

**Milestone:** 2026.06.396
**Author:** `.396 planning sweep (Claude Opus 4.8, outer-loop), 2026-06-16
**Predecessor:** `openspec/change-proposals/research-roadmap-v395.md`
**North star:** `ops/north-star.md` §0 (solve ARC-AGI-3, accurately + efficiently) · §1 (FoVer headline) · §5 (energy VERIFIES, refinement GENERATES)

> **Architecture-freshness flag (operator action):** `_bmad/architecture.md` "Last Reconciled" is 2026-05-16 —
> now **31 days stale** (today 2026-06-16). The DiffusionGemma energy-guided full run is borderline *new
> capability* (the verifier shapes GENERATION in-loop, not just verifies post-hoc). Per CLAUDE.md "Architecture
> Freshness Check", the operator should reconcile `_bmad/architecture.md` before the next *new-capability*
> milestone. `.396 proceeds as DEPTH on an already-specced thesis
> (`docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md`), so it is not blocked — but the flag is
> raised here per the rule.

---

## 0. THE HEADLINE QUESTION

**Now that the verifier-moat is hardened for post-hoc SELECTION (cross-family generalizes), does the EXTERNAL
energy verifier improve GENERATION when wired into the loop — i.e. does Carnot's verifier-as-guidance-energy beat
both unguided DiffusionGemma AND the model's own reward-free self-guidance (RFG)?**

This is the §5 thesis — *energy VERIFIES, refinement GENERATES* — taken to its sharpest test. `.395 proved the
verifier can RANK candidates from unseen families (the post-hoc moat). `.396 asks whether it can SHAPE the
generative trajectory of a 26B discrete-diffusion LLM. The DiffusionGemma full run was DEFERRED from `.395 and is
now **activatable**: the `.395 capstone (exp4279) flipped `diffusiongemma_full_run_gate=True` because all three
hardening axes landed (provenance-blind + multi-seed + cross-family OOD), and the loader was repaired
(exp4274 `preflight_go=true`).

**The over-claim trap, structurally defused.** The operator was twice-burned by premature DiffusionGemma
over-claims. `.396 defuses this two ways: (1) matched **no-Carnot-verifier controls** — the run reports the
external verifier guidance *against* unguided AND against RFG (arXiv:2509.25604, the model's OWN log-likelihood-
ratio self-guidance) AND EntRGi (arXiv:2602.05000); a "win" requires beating these, CI95-excl-0, not just beating
unguided. (2) **Circularity discipline** — the HEADLINE arm uses a LEARNED energy verifier (`verifier_is_oracle=
false`) on a reasoning domain where the verifier is NOT the executable oracle; an execution-grounded arm
(code/Sudoku, `verifier_is_oracle=true`) is reported SEPARATELY as cheap-automatic-decentralized, NOT a moat.

Decision-grade either way:
- **GUIDANCE MOAT HOLDS** (learned-verifier-guided beats RFG, CI95-excl-0, `verifier_is_oracle=false`) → the
  external verifier earns its place IN the generative loop at LLM scale — the moat-scissor realized in generation,
  north-star-grade.
- **GUIDANCE MOAT FAILS** (ties RFG, or only the execution-grounded/circular arm wins) → honest: the verifier's
  value is post-hoc SELECTION + execution-grounded guidance, NOT learned in-generation steering. We scope the
  claim and the DiffusionGemma thesis is bounded (a real, publishable negative).

`.396 is DEPTH on the now-hardened thesis (north-star §1 Depth-Over-Breadth): it RUNS the deferred scale-up, then
HARDENS the cross-family win on an independent substrate, pays the owed EFFICIENCY axis, and advances ARC +1.

---

## 1. WHAT `.395 PROVED (the launch state for `.396)

| Axis | `.395 result | Status for `.396 |
|---|---|---|
| **Cross-family OOD (A2, exp4271)** | **GENERALIZES** — `cross_family_delta` **+0.4038** on held-out families, CI95 [0.25, 0.558] EXCLUDES 0; within-pool control +0.4423 → `within_minus_cross_gap` only **0.0385**; `held_out_family_n=52`; `verifier_is_oracle=false` | ✅ HARDENED on 1 substrate → `.396 STRESSES it on a 2nd (ARC-GEN) |
| Hardening (capstone exp4279) | `hardened_win=True` (provenance-blind +0.385 / multi-seed +0.458 / cross-family +0.404 — all 3 axes) | ✅ The first fully-hardened oracle-distinct moat |
| **DiffusionGemma gate (exp4279)** | **`diffusiongemma_full_run_gate=True`** — loader repaired (exp4274 `loader_repaired`/`preflight_go`/`guidance_changes_selection` all true), cost feasible | 🚀 **THE `.396 HEADLINE** — run the deferred full benchmark |
| Self-learning (A4, exp4273) | `static is the ceiling` — online reweight +0.096 but CI95 [0.0, 0.192] **touches 0** at n=52 (an n-limit, not a proven null) | ➕ `.396 RE-POWERS the test on the larger ARC-GEN family set |
| ARC north star (C1, exp4275) | +1 → **20 levels** (game wa30-ee6fef47 L1, real-env-confirmed) | ➕ `.396 targets +1 on ANOTHER new game (→ ≥21) |
| Publication gate | `paper_ready=True` (G1∧G2∧G3∧G4; FoVer 0.9131) | ✅ Unchanged — the cross-family ARC win is a NEW supporting result the operator may fold into the paper |
| Code oracle-distinct (exp4264) | CORPUS-SPECIFIC (−0.006) → **RETIRED** to exclusion manifest | ⛔ NOT re-proposed (`code_oracle_distinct_replication_retry` retired) |
| Verifier-as-reward in-loop (exp4263) | OUT-OF-BAND / operator-owned → **RETIRED** to exclusion manifest | ⛔ NOT an in-loop task |

**SOTA already ingested for `.396 (exp4276 `flagged_for_v396` + this sweep's WebSearch re-verification 2026-06-16):**
- **RFG (arXiv:2509.25604)** — reward-free (log-likelihood-ratio self-)guidance for diffusion-LLM reasoning, +9.2%
  math/code. → the **no-external-verifier control** the DiffusionGemma moat must beat.
- **EDLM (arXiv:2506.13759)** — energy-based diffusion LM (unnormalized energy reweights denoising). → the peer
  baseline + theoretical scaffolding for the Carnot verifier-as-energy guidance arm.
- **EntRGi (arXiv:2602.05000)** — entropy-aware reward guidance for dLLMs. → a 2nd guidance control.
- **ARC-GEN (arXiv:2511.00162; github.com/google/ARC-GEN)** — mimetic procedural generator, all 400 ARC-1 + 500
  ARC-2 tasks, native family ids. → the **2nd, construction-disjoint** family substrate for the cross-family
  stress (closes the single-partition critique). Report lift separately on original-ARC / ARC-TGI / ARC-GEN.
- **Paying Less Generalization Tax (arXiv:2601.18217)** — richer family metadata + distractor invariance drive
  transfer. → a randomized family-rich stress split (robustness-not-theater) on the cross-family test.
- **EFFICIENCY (north-star §5 owed axis): CompassVerifier (arXiv:2508.03686), Calibrated Reasoning explanatory
  verifier (arXiv:2509.19681)** + the small-verifier precedent (Luna-2 / DeBERTa: 60–1000× cheaper than a frontier
  judge). → the energy-verifier-vs-LLM-judge head-to-head: parity at 10–100× cheaper.

---

## 2. THE PLAN — 5 phases + archive (10 tasks, exp4280–exp4289)

### PHASE A — RUN THE DEFERRED DiffusionGemma ENERGY-GUIDED FULL BENCHMARK (the headline scale-up)

- **exp4281 (A1) — DiffusionGemma energy-guided full run, with matched no-Carnot-verifier controls.** Via the
  llama.cpp PR binary (`~/.cache/llama.cpp-master/build/bin/llama-diffusion-gemma-eval`; the ONLY working loader —
  NOT llama-cpp-python / transformers / vLLM, all of which crash on the diffusion-gemma arch). Wire the verifier
  ensemble as a per-step guidance energy reweighting denoising token selection. On n≥30 per arm:
  - **HEADLINE / oracle-distinct arm** (`verifier_is_oracle=false`): a LEARNED energy verifier guiding on a
    REASONING domain (FoVer-step / math) where the verifier is NOT the executable oracle. Compare **unguided** vs
    **RFG** (model self-guidance) vs **EntRGi** vs **Carnot-verifier-guided**. Emit `diffusiongemma_guidance_moat`
    BARE bool := (Carnot-guided − RFG > 0 AND CI95-excl-0).
  - **SUPPORTING / execution-grounded arm** (`verifier_is_oracle=true`): an executable-oracle verifier guiding on
    code/Sudoku — reported as `execution_grounded` (cheap/automatic/decentralized), explicitly NOT a moat.

  If the learned verifier cannot score partial (masked) denoising states, that is an honest finding (the moat arm
  blocks; the execution-grounded arm + the control comparison still land). NO TRM training. `max_turns: 100`.

### PHASE B — HARDEN THE CROSS-FAMILY WIN ON AN INDEPENDENT SUBSTRATE (ARC-GEN)

- **exp4282 (B1) — ARC-GEN cross-family STRESS.** Clone `github.com/google/ARC-GEN` (gitignored — no embedded-repo
  gitlink, per the CI-breaking incident). Build a family-disjoint candidate pool with native generator family ids
  + exact target hashes; rerun the set-encoder-beats-vote gate trained on a subset of generator-families, tested
  on held-out generator-families. Report `cross_family_delta` SEPARATELY on original-ARC / ARC-TGI / ARC-GEN
  families (the generators-become-their-own-distribution failure mode). Add a randomized family-rich stress split
  (arXiv:2601.18217). Emit `arcgen_cross_family_holds` BARE bool. `verifier_is_oracle=false`. *Decision-grade: a
  2nd-substrate survive closes the single-partition critique; a collapse scopes the `.395 win to its recovered
  manifold.*

### PHASE C — CONTINUOUS SELF-LEARNING, RE-POWERED (the mandated self-learning experiment)

- **exp4283 (C1) — online verifier-weight adaptation, re-powered on the larger family set.** `.395 found `static
  is the ceiling` but the CI TOUCHED 0 at n=52 — an n-limit, not a proven null. Re-run the Tier-1 online
  reweighting (CPU counter/weight updates — NOT the retired live-LoRA path) across the COMBINED family set (the
  `.395 recovered manifest + ARC-GEN's hundreds of families) so the (online − static) CI is POWERED. Add a Tier-2
  constraint-memory arm (cache per-family selection patterns, reuse on the nearest-neighbor family) as a distinct
  mechanism. Emit `online_adaptation_helps` BARE bool. `verifier_is_oracle=false`. `retire_if_same_verdict: true`
  — if static is STILL the ceiling WITH power, retire the online-adaptation-helps-the-ARC-selector ask.

### PHASE D — PAY THE OWED §5 EFFICIENCY AXIS

- **exp4284 (D1) — energy-verifier vs LLM-as-judge efficiency head-to-head (north-star §5 win condition).** On the
  ORACLE-DISTINCT cross-family ARC selection task (NOT code — code efficiency-vs-judge is retired/circular):
  measure the learned energy verifier's selection accuracy vs an LLM-as-judge (a cached SOTA GGUF) selecting from
  the same candidates. Report accuracy parity (within CI) AND the compute/latency/cost ratio. Target: "parity at
  10–100× cheaper." Emit `efficiency_parity_at_lower_cost` BARE bool. `verifier_is_oracle=false`. This is a NEW
  measurement (the §5 owed axis), not a re-run.

### PHASE E — ARC NORTH STAR (accuracy progress)

- **exp4285 (E1) — offline ARC incremental +1 on a NEW game.** Per ARC-AGI-3 Incremental-Progress Scoping (+1,
  not all-levels): target the best-headroom UNATTEMPTED game (NOT wa30 = exp4275's advance, NOT sc25 = exp4261's
  wall), hardened set-encoder routing the offline solver. Monotonic `total_levels ≥ 21`, `levels_completed ≥ 1`
  real-env-confirmed. NO TRM training; NO leaderboard submission.

### PHASE F — HYGIENE & CAPSTONE

- **exp4280 (archive)** — archive `.395 → activate `.396; record the `.395 landmark close-state (cross-family
  GENERALIZED, gate flipped open) truthfully.
- **exp4286 (F1) — SOTA-ingestion → `.397 forks.** Reliable channel only (`/deep-research` banned in-loop).
- **exp4287 (F2) — registry/gaps hygiene + GAP-4 regression guard + log new gaps.**
- **exp4288 (F3) — hardware continuity (opportunistic per north-star §3).**
- **exp4289 (capstone `.396)** — aggregate: `diffusiongemma_guidance_moat` + `arcgen_cross_family_holds` +
  `online_adaptation_helps` + `efficiency_parity_at_lower_cost` + ARC `total_levels` + G1–G4 publication gate.
  SKIP any `flagged_adversarial` artifact; HONOR `verifier_is_oracle` (a circular/leaked result may NOT headline a
  moat). NO `gated_on`.

---

## 3. DEPENDENCY GRAPH

```
exp4280 (archive .395 → activate .396)
   │
   ├─► exp4281 (A1 DiffusionGemma full run)  ──────────────┐  [PR-binary; matched RFG/EntRGi controls]
   │                                                        │
   ├─► exp4282 (B1 ARC-GEN cross-family stress) ─┐          │
   │                                             ▼          │
   ├─► exp4283 (C1 self-learning re-powered) ◄─ (reads B1's family set; soft input, no hard gate)
   │                                                        │
   ├─► exp4284 (D1 efficiency head-to-head)                │
   │                                                        │
   ├─► exp4285 (E1 ARC +1 new game)                         │
   │                                                        ▼
   ├─► exp4286 (F1 SOTA-ingestion → .397)                  │
   ├─► exp4287 (F2 registry/gaps hygiene)                  │
   ├─► exp4288 (F3 hardware continuity)                    │
   │                                                        │
   └─► exp4289 (capstone .396) ◄────────── aggregates A1/B1/C1/D1/E1 + F2/F3
```

No HARD `gated_on` chains (the `.395 fork-gate on `family_split_feasible` is resolved — the manifest exists on
disk). C1 reads B1's ARC-GEN family set as a SOFT input (no gate → no cascade-block risk; C1 has an honest
fallback to the `.395 manifest if B1 is thin).

---

## 4. THE THREE BIGGEST GAPS (current state → PRD vision) THIS MILESTONE ATTACKS

1. **The verifier-moat is proven for post-hoc SELECTION but NOT for in-generation GUIDANCE at LLM scale.** The §5
   thesis claims energy VERIFIES and *shapes* refinement — but `.395 only showed it RANKS. PHASE A closes this:
   does the external verifier improve a 26B diffusion LLM's GENERATION over the model's own self-guidance? This is
   the deepest open form of "does the verifier earn its place."
2. **The hardened cross-family win rests on ONE substrate (the recovered 52-task manifest).** A skeptic says "you
   generalized across families YOU carved from one pool." PHASE B replicates on ARC-GEN — an independent Google
   procedural generator — closing the single-partition critique (PRD: the verifier must be a GENERAL signal for
   the ARC-AGI-3 harness, which faces new games constantly).
3. **Continuous self-learning (PRD FR-11) hit a ceiling `.395 couldn't resolve** (`static is the ceiling`, CI
   touched 0 at n=52). PHASE C re-powers the test on ARC-GEN's larger family set + adds a Tier-2 mechanism — does
   Carnot actually get smarter over time, or is static genuinely the ceiling? Plus PHASE D pays the §5 EFFICIENCY
   axis (the verifier earns its place at LOWER cost), the other half of "does the verifier earn its place."

---

## 5. HARDWARE REQUIREMENTS

| Resource | Used by | Precondition |
|---|---|---|
| 1× RTX 3090 (24 GB) + the llama.cpp PR binary | A1 (DiffusionGemma Q4_K_M GGUF, 16 GB) | `ls ~/.cache/llama.cpp-master/build/bin/llama-diffusion-gemma-eval` + the GGUF cache (both confirmed 2026-06-16) |
| CPU only | B1, C1, D1, E1 (verifier scoring against cached candidates; ARC-GEN generation) | the cached pools + `github.com/google/ARC-GEN` clone (gitignored) |
| KV260 / PolarFire / GateMate (opportunistic) | F3 | SSH reachability (KV260 SSH-not-SD-card); no board blocks the milestone |

**HARD RULES (every task):** the TRM Sudoku checkpoint is DONE (val 0.8227) and the conductor stays stood-down —
NO task may launch TRM training, run `pkill`/`kill` against `train.py`, or WRITE `results/trm_runs/`. Qwen is
FORBIDDEN as the TRAINED base (Spurious-Rewards confound); Qwen GGUF as an off-policy teacher/judge is fine. Every
verifier-value task MUST declare `verifier_is_oracle` honestly (Circularity Discipline) — a guidance/selection win
driven by an EXECUTABLE oracle is `execution_grounded` (cheap, valid, NOT a moat); the moat is the LEARNED-energy
result with a matched control, CI95-excl-0. NO autonomous edits to `docs/index.html` / README / paper prose
(Public Documentation Discipline). Online ARC play stays operator-gated (NO leaderboard submission).
