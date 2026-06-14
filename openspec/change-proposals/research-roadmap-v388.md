# Research Roadmap — Milestone 2026.06.388

**Title:** HARDEN THE VERIFIER MOAT WHERE IT MATTERS — efficiency vs LLM-as-judge
(the owed north-star §5 win) + GAP-4 execution-verifier production-safety + a SOVEREIGN
local generator + the now-unlocked DiffusionGemma verifier-guided decoding

**Planned:** 2026-06-14 (UTC) · **Supersedes:** 2026.06.387 · **Planner:** Opus 4.8 (outer-loop)

---

## 0. One-paragraph thesis

Milestone .387 **PROVED the verifier moat** on an executable domain: on the HumanEval code pool
(exp4177), the Carnot executable verifier + V-STaR selector reached pass@1 **0.84 vs SC-vote 0.66**
(+0.18, CI95 [0.08, 0.30]), and a **matched no-verifier control** also sat at 0.66 — so the win is
the verifier, not extra compute, and `positive_control_confirmed=true`. That flipped the
**DiffusionGemma gate to MET**. But the .387 proof has three honest caveats that .388 must close,
because each is load-bearing for the north star (solve ARC-AGI-3 accurately AND **efficiently**):
**(1) the EFFICIENCY axis was NOT won** — `efficiency_parity=false`; the test compared the verifier
against *self-consistency vote* (nearly free), never against an *LLM-as-judge*, which is the actual
north-star §5 win condition ("equally effective as the LM at lower cost", target "parity at 10-100x
cheaper"); **(2) the code verifier is semi-circular** — on code the "verifier" ≈ the unit-test
oracle, so the harder claim (a verifier beats vote where the verifier is NOT the oracle) is the ARC
GAP-3 question, which came back **BOUNDED** (exp4178: latent AUROC 0.89 discriminates but selection
delta 0.0); **(3) the only ARC verifier that reaches headroom is GAP-4 execution** (vote 0.45 →
gated 0.58), and it is unhardened (precision decays to ~0.47 on uncontaminated tasks) and **not
sovereign** (the generator was codex — closed + contaminated). So .388 converts "proven on the easy
case" into the moat the north star needs: **(A)** the efficiency moat vs an LLM-as-judge with real
wall-clock cost; **(B)** GAP-4 made production-safe (graded gate) + a fully **local/open-weight**
generator arm (decentralization rule 1) that self-distills the verifier's accepted programs; **(C)**
the now-unlocked DiffusionGemma verifier-guided decoding + monotonic ARC +1 + a real ARC-AGI-3
LIVE-env grounding probe; **(D)** reserved infra/hardware/SOTA/capstone. **GAP-3 trained ARC content
energies stay RETIRED** (operator-authorization required) — .388 invests the ARC slot in the proven
execution path, and re-flags CEM (arXiv:2510.20607) to the operator rather than auto-activating it.

## 1. What .387 proved (the inputs to this plan)

| Result | Artifact | Reading |
|---|---|---|
| **Verifier moat PROVEN on code** — ARM-A 0.84 vs SC-vote 0.66 (+0.18, CI95 [0.08, 0.30]); matched no-verifier control also 0.66; positive_control_confirmed | exp4177 | The accuracy moat is real **where headroom exists and the verifier executes**; the win is the verifier (matched control rules out compute) |
| **EFFICIENCY axis NOT won** — `efficiency_parity=false`, verifier 2-4x the vote cost; comparison was vs vote, NOT vs an LLM-judge | exp4177 `accuracy_cost_pareto` | The north-star §5 win condition (parity at lower cost vs an LLM-judge) is **owed** — the central .388 question |
| **DiffusionGemma gate: MET** | exp4183 capstone | The verifier added value on a headroom-present executable domain → the spec gate condition is satisfied → .388 MAY activate verifier-guided diffusion |
| **GAP-3 Stage-1 (learned/model-native ARC energy) BOUNDED** — latent AUROC 0.89 but pass2-vs-vote delta 0.0; permutation control failed (pooling artifact) | exp4178 | Learned ARC *energies* discriminate per-candidate but do NOT beat vote on selection — consistent with the retired Stage-2 lineage |
| **GAP-4 execution verification is the only ARC positive** — vote 0.4516 → gated 0.5806 (+4/−0 on ARC-1); precision decays 0.90→0.47 on uncontaminated ARC-2; agreement-as-*selector* RETIRED (exp4023, confidence-label only) | `ops/verifier_gaps.md` | The proven ARC path is execution/program-synthesis; forward work = **graded gate** + **local generator**, NOT another agreement-selector or trained energy |
| ARC: `total_games_solved=13`, `total_levels_solved=14` (lp85 → L2, real-env-confirmed) | exp4179 | Monotonic +1 held; continue per the ARC Incremental-Progress Scoping discipline |
| SOTA ingestion flagged **CEM compositional ARC energy** (arXiv:2510.20607) for .388 | exp4180 | Strongest flagged method — but it is the RETIRED trained-content-energy lineage; .388 re-flags it to the operator, does not auto-run it |

## 2. The three biggest gaps (current state vs PRD vision)

1. **The EFFICIENCY moat is unmeasured — and it is the north-star §5 win condition.** .387 proved
   +18pp accuracy on code but `efficiency_parity=false` because it compared the verifier against
   self-consistency *vote* (≈free), not against an *LLM-as-judge*. The PRD/north-star claim is that
   Carnot's externally-grounded forward-pass verifier is "equally effective as the LM at lower
   cost/latency" — target "parity at 10-100x cheaper" (north-star §5; arXiv:2504.01005 sets the bar:
   SC beats a generative verifier on cost until ~8x). This is the single most load-bearing
   unmeasured moat claim, and the precondition for the RSI-scale verification story.
2. **The ARC verifier (Carnot's core product for the north star) reaches headroom only via
   execution, and that win is neither production-safe nor sovereign.** GAP-4 (program-induction +
   execution) is the only ARC positive (vote 0.45 → 0.58) but its precision decays to ~0.47 on
   uncontaminated tasks and the generator was **codex** (closed-weight + contaminated). The ledger's
   forward moves — a **graded min-hamming gate** (production-safe; τ=0.005 + vote-aware guard) and a
   **LOCAL open-weight generator arm** (decentralization rule 1) — are owed. GAP-3 learned energies
   are bounded/retired, so execution is the path.
3. **Verifier-guided GENERATION is unexercised and the real ARC-AGI-3 LIVE env is untouched.** The
   DiffusionGemma gate is now MET — the verifier can *shape* generation (not just rerank), the
   LLM-scale realization of the thesis — but it has never been tried. And the entire ARC-AGI-3 stack
   is offline/synthetic; a live-env random/greedy grounding baseline is the cheap, unblocked
   north-star step (§0: the SDK auto-issues an anonymous key + exposes 25 live environments).

## 3. Architecture of the milestone

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │  PHASE A — THE EFFICIENCY MOAT (the owed north-star §5 win)     │
  cached + fresh    │  A1 headroom re-census ──► executable headroom domain(s)        │
  candidate pools   │     + LLM-as-judge arm      + an LLM-judge baseline built       │
  (code proven 0.18)│       + REAL cost meter     + per-arm wall-clock/token cost     │
                    │            │ max_selectable_headroom (BARE)                     │
                    │            ▼ gated_on >= 0.10                                   │
                    │  A2 DECISIVE: Carnot exec/energy verifier  vs  LLM-as-judge     │
                    │     accuracy parity (CI95 on A−judge) AND cost ratio →          │
                    │     verifier_efficiency_win := within-CI AND >=10x cheaper      │
                    └───────────────────────────────┬──────────────────────────────┘
                                                     │
   ┌──────────────────────────────────────────┐     │     ┌──────────────────────────────────┐
   │ PHASE B — THE ARC VERIFIER (execution)    │     │     │ PHASE C — UNLOCKED GEN + ARC + LIVE│
   │ B1 GAP-4 graded gate (production-safe,    │     │     │ C1 DiffusionGemma verifier-guided  │
   │    NON-retired; not agreement-selector)   │     │     │    decoding (gate MET; feasibility)│
   │ B2 SOVEREIGN local generator + GAP-4-     │     │     │ C2 ARC incremental +1 (monotonic)  │
   │    verified self-distillation (Tier-2/3)  │     │     │ C3 ARC-AGI-3 LIVE-env grounding    │
   └──────────────────────────────────────────┘     │     └──────────────────────────────────┘
                                                     ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────────┐
   │ PHASE D — RESERVED SLOTS: D1 SOTA-ingestion · D2 registry+gaps hygiene · D3 hardware ·      │
   │ D4 capstone (is the efficiency moat won? is GAP-4 sovereign+safe? did DiffusionGemma fire?) │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 4. Phase descriptions

### Phase A — The efficiency moat (the decisive owed north-star §5 win)

.387 proved the accuracy moat but compared against the wrong baseline for cost. The win condition is
not "beats free vote" (it costs more than free vote) — it is **"matches an LLM-as-judge on accuracy
at a fraction of the cost."** 2026 best practice corroborates the thesis directly: lightweight
verifiers (DeBERTa-440M, sub-200ms; open-weight judges 60-1000x cheaper than frontier) fronted by
deterministic checks, frontier LLM-judge reserved for nuance — exactly Carnot's Meta-EBM Cascade
Router. So .388 measures the cheap-executable-verifier-beats-the-expensive-judge claim head-on.

- **A1 — Headroom re-census + LLM-as-judge baseline harness + real cost meter** (`exp4185`). Reuse
  `scripts/headroom_gate.py`; re-establish executable-headroom domain(s) (code is proven at 0.18;
  attempt a second executable pool — MBPP/EvalPlus or held-out Sudoku — generating candidates from a
  SOTA GGUF behind a PRECONDITIONS cache check if no cached pool exists). Build the **LLM-as-judge
  arm**: a SOTA GGUF (`gemma-4-26B-A4B-it-GGUF` / `Qwen3.6-35B-A3B-GGUF`) prompted to select the best
  candidate, with per-call wall-clock + token instrumentation. Emit `max_selectable_headroom` (BARE)
  + `llm_judge_ready` + the cost meter. Establishes the comparator and the positive control A2 needs.
- **A2 — THE decisive efficiency moat** (`exp4186`, gated_on A1 `max_selectable_headroom >= 0.10`).
  On the headroom-present executable domain: **ARM A** = Carnot executable/energy verifier + V-STaR
  selector; **ARM J** = LLM-as-judge (SOTA GGUF); **ARM B** = SC-vote; **oracle** ceiling. Report
  accuracy parity (bootstrap CI95 on pass@1(A) − pass@1(J)) **and** the real cost ratio (verifier
  wall-clock/tokens ÷ judge wall-clock/tokens). `verifier_efficiency_win := true` iff A is within-CI
  of J on accuracy **and** A is >=10x cheaper (or strictly Pareto-dominates: >= J accuracy at < J
  cost). This resolves the north-star §5 efficiency-parity question with REAL cost, fixing the .387
  abstract-cost-unit caveat. A clean "judge wins on accuracy" or "no cost advantage" is an honest,
  decision-grade COMPLETE verdict.

### Phase B — The ARC verifier via the proven path: execution, made production-safe + sovereign

`ops/verifier_gaps.md` is unambiguous: cheap hand-invariants are exhausted, GAP-3 learned energies
are bounded/retired, and **GAP-4 execution/program-synthesis verification is the only ARC positive**.
Its forward work (per the ledger's "Banked successor", explicitly NOT the retired agreement-selector
R&D) is a production-safe gate and a sovereign generator.

- **B1 — GAP-4 graded execution-energy gate (production-safe)** (`exp4187`). Implement + validate the
  **graded min-hamming gate** (τ=0.005 ONLY + a **vote-aware guard** to block the one measured
  exact-match mis-promotion over a high-vote gold — the 25094a63 case) on the **non-degenerate ARC-1
  venue first**. The agreement signal remains a CONFIDENCE LABEL only (per the exp4023 retirement —
  do NOT re-run agreement-as-a-precision-selector). Report pass@1/pass@2 vs vote, the gross +/−
  ledger, and the band precision at τ≤0.02. This makes the proven +4/−0 verifier safe for a richer
  candidate pool where near-misses (~0.5 precision) must not be promoted.
- **B2 — Sovereign local generator + GAP-4-verified self-distillation** (`exp4188`, SELF-LEARNING /
  decentralization). Replace codex with a **local open-weight generator** (`gemma-4-26B-A4B-it-GGUF`
  / `Qwen3.6-35B-A3B-GGUF`) for ARC program induction: does a fully-sovereign generator + GAP-4
  execution verifier recover ARC headroom without any closed-weight call (decentralization rule 1)?
  The verifier labels demo-perfect induced programs, and those verified programs form a
  **self-distillation corpus** (Tier-2 constraint memory / Tier-3 — the research-program.md
  continuous-self-learning requirement: the verifier teaches the local generator). Report local
  induction rate vs codex, pool-rerank pass@2, and the distillation-corpus size. prior_failures cites
  the open `GAP-DECENTRALIZATION-MOE-SYNC-4069` null (coverage 0.23); forward difference = the
  hardened B1 graded gate + active-data prompting + verified self-distillation.

### Phase C — Now-unlocked verifier-guided generation + ARC progress + live-env grounding

- **C1 — DiffusionGemma verifier-guided decoding (feasibility + smoke; gate MET)** (`exp4189`).
  The gate flipped MET (exp4177), so the verifier-shapes-generation mechanism is now in scope. HARD
  PRECONDITION (DiffusionGemma is NOT yet cached — verified 2026-06-14): the model must be
  cached/downloadable AND expose per-step token logits; else honest `blocked_diffusiongemma_*`. Use
  Carnot's executable verifier ensemble as the **guidance energy** reweighting per-step token
  selection during denoising on an executable domain (code/Sudoku); guided vs unguided vs AR-baseline
  on a small n with bootstrap CI. This is feasibility + smoke (per the "break large benchmarks into
  phases" lesson), not a headline benchmark — a clean blocked verdict (model unavailable / no
  per-step hook) is COMPLETE and de-risks the C2/C3 scale-up framing for .389. Fresh refs:
  arXiv:2602.22871 (reward-guided stitching), 2602.01849 (self-rewarding SMC), 2509.13866 (theory).
- **C2 — ARC-AGI-3 incremental +1 (monotonic)** (`exp4190`). Per the ARC Incremental-Progress Scoping
  discipline, advance the solved-LEVEL count by >=1 via explore → induce → **GAP-4-verify (the
  hardened B1 stack)** → act, on the next deeper level of an already-solved game (or a spatial L1).
  Real-env-confirm; STOP at the first level that fails; honest no-solve = COMPLETE.
- **C3 — ARC-AGI-3 LIVE-env grounding probe** (`exp4191`). The §0 unblocked north-star step: connect
  the EXISTING offline harness (`python/carnot/agentic/arc_agi3_world_model.py`) to the live SDK
  (`arc_agi`, anonymous key), enumerate the 25 live environments, and run a random/greedy baseline to
  establish the real-env metric pipeline (`EnvironmentScore.score` / `levels_completed` for ACCURACY;
  actions-taken vs `baseline_actions` for EFFICIENCY). **NO leaderboard submission** (operator-only
  external publication); **no scored online play** beyond a reachability/baseline probe (the online
  quota gate stays: only submit when an offline result beats the TRM baseline + best prior Carnot
  run — memory `arc3_online_gated_on_offline_beating_baselines`). Honest `blocked_arc_live_*` if
  access fails.

### Phase D — Reserved slots (infra / hardware / SOTA / capstone)

- **D1 — SOTA-ingestion** (`exp4192`, reserved bleeding-edge slot). Ingest the .388 sweep
  (2602.22871, 2604.06260, the 1/1000-cost-judge OpenReview work, the cascade-verification trend)
  mapped onto the efficiency-moat / GAP-4-sovereignty / DiffusionGemma stack; **re-flag CEM
  (2510.20607) explicitly as needing OPERATOR authorization** (the trained-content-energy SELECTOR
  lineage is retired) so the discover→ingest→plan loop surfaces it to the operator rather than
  silently dropping or auto-activating it. Flag the single strongest method for .389. Real arXiv IDs.
- **D2 — Verifier-registry + gaps hygiene** (`exp4193`, reserved infra slot). Bit-exact GAP-4 ARC-1
  regression replay (vote 0.4516 → gated 0.5806, zero codex/GGUF/GPU); record the .388 efficiency-
  moat verdict + GAP-4 graded-gate + sovereign-generator outcomes into `ops/verifier_gaps.md`
  (never-prune); promote `GAP-MOAT` toward `filled` only if the efficiency win lands.
- **D3 — Hardware continuity** (`exp4194`). GateMate (USB-detect) + PolarFire (ssh) drive-toward-
  terminal; KV260 (ssh) opportunistic/terminal-confirm. SSH/USB-detect preconditions ONLY (KV260
  SSH-Not-SD-Card Discipline); distinct wall-clock timers per board.
- **D4 — Capstone .388** (`exp4195`). Headline question: **is the verifier moat won on the axis the
  north star needs — efficiency-parity vs an LLM-as-judge (A2) — and is the ARC execution verifier
  now production-safe (B1) and sovereign (B2)?** Plus: did DiffusionGemma fire (C1), total ARC levels
  (C2), live-env reachable (C3). SKIP any `flagged_adversarial` artifact; cite upstream sha256.

## 5. The DiffusionGemma gate (now MET — explicitly respected)

The DiffusionGemma spec (`docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md`) gated
activation on a verifier-graft reporting `verifier_value_added == true` on an executable domain. The
.387 capstone (exp4183) recorded the gate **MET** on the basis of exp4177 (verifier_value_added=true,
positive_control_confirmed=true, on the code executable domain with selectable headroom). So .388 is
authorized to activate verifier-guided diffusion — but does so **conservatively** (C1 is a
feasibility + smoke task behind a hard cache/per-step-logit precondition, because DiffusionGemma is
not yet cached and the open release's per-step-logit exposure is unverified). The full headline
benchmark is deferred to .389 pending C1's feasibility verdict.

## 6. Dependency graph

```
exp4184 (archive/activate) ─► exp4185 (A1 headroom + LLM-judge harness) ─► exp4186 (A2 efficiency moat, gated_on A1)
                              exp4187 (B1 GAP-4 graded gate) ──► exp4188 (B2 sovereign generator + self-distill)
                              exp4189 (C1 DiffusionGemma) ─── independent (hard precondition)
                              exp4190 (C2 ARC +1) ──────────── uses the B1 hardened GAP-4 stack
                              exp4191 (C3 live-env grounding) ─ independent (SDK reachability)
                              exp4192 (D1 SOTA) ─┐
                              exp4193 (D2 registry) ─┤─► exp4195 (D4 capstone, ungated, aggregates all)
                              exp4194 (D3 hardware) ─┘
```

## 7. Hardware requirements

- **GPU optional, declared-not-assumed.** A1/A2 prefer cached candidate pools; the LLM-judge arm
  (A1/A2) and the sovereign generator (B2) load a SOTA GGUF (`gemma-4-26B-A4B-it-GGUF` /
  `Qwen3.6-35B-A3B-GGUF`, all cached) via the `.gguf` path (GGUF tokenizer rule) behind a PRECONDITIONS
  cache check. C1 (DiffusionGemma) needs CUDA (RTX 3090) IF the model becomes available — else blocks.
- **No write to the TRM stable checkpoint dir** (read-only; training is DONE, SIGTERM'd at val 0.8227;
  the conductor stays stood-down on TRM training). Phase-B ARC work reads cached pools / induces
  programs; it does not retrain TRM.
- **FPGA boards** (D3): GateMate (USB-detect), PolarFire (ssh), KV260 (ssh, terminal/opportunistic).

## 8. Self-learning coverage (research-program.md requirement)

The explicit self-learning task is **B2 — GAP-4-verified self-distillation**: the local open-weight
generator induces ARC programs, the GAP-4 execution verifier labels the demo-perfect ones, and those
verified programs become a self-distillation corpus (Tier-2 constraint memory → Tier-3 predictive
verification: the verifier teaches the generator). This is continuous self-learning where the ENERGY
FUNCTION (execution consistency) is the ground-truth reward — the FR-11 loop, made sovereign. The A2
efficiency moat also bears on self-learning: a cheap, externally-grounded verifier is the precondition
for verifying a machine-scale self-improvement loop without an LLM-judge per artifact.

## 9. Disciplines honored

Codex-Default v2 (all experiment tasks codex/gpt-5.5) · Verdict Terminal-Prefix · Principle-Annotated
artifact fields + gates · Pre-Launch Preconditions (GGUF cache, DiffusionGemma cache, ARC live-env
reachability, FPGA SSH/USB-detect) · Inference-Substrate Declaration · FALSE_NEGATIVE_RISK /
positive-control (the A1→A2 headroom gate) · Reading-Results Discipline · Exclusion-Manifest
cross-check (no task matches a retired id; **GAP-3 trained-content-energy + GAP-4 agreement-selector
lineages are RESPECTED as retired — not re-run; CEM re-flagged to operator**) · Adversarial Artifact
Verification + Sample-Size Rigor · ARC Incremental-Progress Scoping · Missing-Verifier Gap Logging ·
SOTA-Ingestion Cycle · Hardware-Task Continuity (KV260 SSH-Not-SD-Card) · Reserved infra slots (≥2:
D1, D2) · Decentralization rule 1 (the B2 sovereign-generator arm; local GGUF, no closed-weight
dependency) · Operator-Only External Publication (C3 does NOT submit to the leaderboard; no arxiv/
release steps).
