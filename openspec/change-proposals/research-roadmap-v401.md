# Research Roadmap v401 — SETTLE the in-generation oracle-distinct moat with a LEAK-ROBUST partial-state scorer (replicate-or-retire) + drive the FIRST E3 ARC solve with explore-verify-plan

**Milestone:** 2026.06.401
**Planned:** 2026-06-17 (Claude Opus 4.8, outer-loop planner)
**Supersedes:** v400 (2026.06.400)
**North star:** Solve ARC-AGI-3, accurately and efficiently (operator directive 2026-06-08; `ops/north-star.md` §0). The energy VERIFIER is Carnot's value-add; the generator is commodity (local LLM / TRM refiner / coding agent).

---

## 1. What .400 proved (the inputs to this milestone)

`.400 was DEPTH on every proven-or-mandated direction (north-star §1) and came back mostly NULL / PARTIAL — a
sobering, decision-grade scorecard (exp4335 capstone, `verifier_thesis_state = in_generation_moat_corpus_specific`):

| Axis | .400 result | Status for .401 |
|---|---|---|
| **In-generation moat** (exp4325) | **DID NOT REPLICATE — the scorer LEAKED on a 2nd corpus.** The exp4292 partial-state scorer that closed the moat on corpus-1 (.399 exp4315) FAILED its independent leak re-check on a 2nd oracle-distinct corpus (`scorer_leak_recheck_passed=false` → `scorer_leaky_on_second_corpus`, `in_generation_moat_replicates=false`). **DiffusionGemma gate: STILL_PENDING_second_corpus_scorer_leaky.** | **HEADLINE — SETTLE (replicate-or-retire).** The `.399 "first oracle-distinct win" is in DOUBT (the scorer may recover answer cells from position cues). The diagnosed root cause is the scorer leak → build it LEAK-ROBUST, then re-run the gate. |
| **Adaptive scale-up** (exp4326) | **bounded to post-hoc stitching.** Schedule-only adaptive guidance did NOT beat the no-adaptation control (`+0.15`, CI95 [-0.075, 0.35] INCLUDES 0); ARC-grid generation infeasible in-window (`reasoning_corpus_fallback`). | A real reward-OPTIMIZATION step (SEPO) is owed, but it = verifier-as-reward generator training → OUT-OF-BAND/operator-owned. Flag, do NOT auto-run. |
| **E3 deep tail** (exp4327-4329) | **0 levels reproduced; ar25 CLOSE.** Per-game verifier accuracy: **ar25 0.89**, ka59 0.56, ft09 0.10, tr87 0.00. All `offline_reproduced=false`, `plan_executed=false`. ar25 got ONE refactor round (timed out under live codex contention). | **NORTH STAR — drive the FIRST E3 solve.** ar25 is one good refactor loop + explore-before-plan away from L1. `verifier_is_oracle=true` (execution-grounded). |
| **Self-learning** (exp4331) | **NULL.** Learned-frame-encoder cross-game transfer `reduction=1.008`, CI95 [1.0,1.03], positive control passed. 2nd cross-game-transfer null (after exp4318 generic features). | **SELF-LEARNING — ACTION-ROLE features.** Raw-frame features carry no game-invariant search-value signal; disentangle action-role/interaction (ReactiveGWM). |
| **Shallow tail** (exp4330) | **no advance.** Adapter-free graph-explore is exhausted on the shallow tail. ARC stays at **13 reproducible / 11 games** (+ sc25 provisional 5 live-recorded, 0 reproduced). | **NORTH STAR — sc25 reproduction via E3** (the biggest reproducible-level upside, +1..+5; the BFS solver stalled on the win-mechanic). |
| **Hygiene / paper** | hygiene passed (7 gaps logged); `paper_ready=True` (FoVer 0.9131, G1–G4). | Carry forward; publication is operator-only. |

Cross-domain selection stays **RETIRED, domain-bound** (exp4314, `ops/exclusion_manifest.yaml`) — NOT re-proposed.

---

## 2. The three biggest gaps (current state vs PRD / north-star vision)

1. **The verifier moat in GENERATION is UNSETTLED and at risk of being a leak artifact.** The project's one
   oracle-distinct win (exp4315) failed to replicate because the partial-state scorer leaks on a 2nd corpus. This is
   the single highest-leverage question: is there a LEAK-ROBUST scorer (oracle-distinct under answer-cell masking) that
   replicates the moat — or was the win corpus-specific? Settling it (replicate-or-retire) either gives a hardened
   headline-grade result or forces an honest pivot. North-star §1: advance the headline or it is noise.

2. **ARC-AGI-3 solve DEPTH — 0 deep-tail E3 solves, ar25 is one loop away.** 13 reproducible levels are almost all L1
   first-contact solves. The deep tail resists graph-explore (mechanic-limited); the SOTA is the executable-world-model
   agent (arXiv:2605.05138). The E3 harness induces a partial model but plans BEFORE it has verified the mechanics
   (`plan_executed=false`) and ar25 only got one refactor round. Closing this (explore-verify-plan + adaptive testing)
   is the direct path to the FIRST E3 solve and a higher ARC score. **Operator MANDATORY (2026-06-17): E3 deep tail.**

3. **The verifier is DOMAIN-BOUND and does not transfer cross-game.** Cross-domain selection is retired; cross-game
   value transfer nulled twice (generic + learned-frame features). The constructive response (north-star §0 step 2:
   verifier domain expansion toward the grid/rule-induction domains ARC-AGI-3 needs) is a game-AGNOSTIC ACTION-ROLE
   representation that carries a search-value signal where raw-frame features failed — the self-learning frontier AND
   the ARC efficiency axis in one.

---

## 3. Architecture — where .401 acts

```
            ARC-AGI-3 (the north star: accuracy = solved levels, efficiency = actions + cost)
                                          │
         ┌────────────────────────────────┼───────────────────────────────────┐
         │ GENERATOR (commodity)           │ VERIFIER (Carnot's value-add)      │ SELF-LEARNING
         │                                 │                                    │
   ┌─────▼─────────────┐         ┌─────────▼──────────────┐          ┌──────────▼───────────┐
   │ coding agent       │ induces │ WorldModelVerifier      │ PHASE B  │ action-role encoder   │
   │ (codex/gpt-5.5)    │────────▶│ grounds the model;      │ /C       │ (ReactiveGWM-style)   │
   │ → Python world     │ EXPLORE │ explore-verify-PLAN     │◀─────────│ → cross-game value    │
   │   model            │ -verify │ [execution-grounded]    │ PHASE D  │ transfer (search)     │
   └────────────────────┘         └─────────────────────────┘          └───────────────────────┘
                                          │
   ┌────────────────────┐         ┌───────▼─────────────────┐
   │ DiffusionGemma 26B  │ guided  │ LEAK-ROBUST partial-     │  PHASE A (HEADLINE — SETTLE)
   │ (open, Apache-2.0)  │────────▶│ state scorer (DiNa-LRM): │  build leak-robust scorer →
   │ denoising           │ by      │ oracle-distinct UNDER    │  re-run 2nd-corpus moat gate
   └────────────────────┘         │ answer-cell masking      │  → replicate-or-RETIRE the moat
                                   └──────────────────────────┘
```

The energy/learned verifier appears in three load-bearing roles: (A) a LEAK-ROBUST external process reward steering
DiffusionGemma generation — oracle-distinct UNDER answer-cell masking, the headline being settled; (B) the
`WorldModelVerifier` grounding an induced executable world model with an explore-verify-PLAN loop — execution-grounded,
the ARC deep tail; (C) a game-agnostic action-role value head over solver state for search efficiency — oracle-distinct,
self-learning.

---

## 4. Phases

### PHASE 0 — TRANSITION
- **exp4336** archive `.400 → activate `.401; record the TRUE `.400 close-state (in-generation moat
  CORPUS-SPECIFIC / scorer leaked on 2nd corpus, gate STILL_PENDING; E3 deep tail 0 solves, ar25 0.89 closest;
  self-learning null; ARC 13 reproducible / 11 games; cross-domain RETIRED; `paper_ready=True`). `agent_type: codex`.

### PHASE A — HEADLINE: SETTLE the in-generation moat with a LEAK-ROBUST scorer (replicate-or-retire → DiffusionGemma gate)
- **exp4337** BUILD a **DiNa-LRM-style leak-robust partial-state reward scorer** (arXiv:2602.11146 — timestep-conditioned
  reward head trained on noisy/masked DiffusionGemma canvases, noise-calibrated uncertainty). HARD gate: it stays
  oracle-distinct UNDER ANSWER-CELL MASKING on **≥2 corpora** (signal does NOT collapse to chance with answer cells
  masked, AND does NOT survive when it should be the answer leaking) — the diagnosed fix for exp4325's leak. Deliverable:
  the scorer module + the leak-audit artifact. `scorer_leak_audit_passed` BARE bool. `verifier_is_oracle=false`.
- **exp4338** (gated on exp4337 `scorer_leak_audit_passed==true`) RE-RUN the in-generation moat **replication on the 2nd
  corpus** with the leak-robust scorer, EXACT exp4315 harness (unguided + EntRGi engaged control + self-reward SMC + the
  no-op guard + an independent leak re-check + Carnot reward-guided stitching) at n≥80/arm, ≥3 seeds. Emit
  `in_generation_moat_replicates` BARE bool. **DECIDES the DiffusionGemma gate.** `retire_if_same_verdict: true` — a
  powered failure-to-replicate even WITH a leak-robust scorer RETIRES the in-generation moat as corpus-specific.
  `verifier_is_oracle=false`.

### PHASE B — ARC NORTH STAR: the FIRST E3 solve via explore-verify-plan (operator MANDATORY 2026-06-17)
- **exp4339** E3 **ar25** (0.89, closest) with **AERA explore-verify-plan** (arXiv:2605.25931 — EXPLORE collects
  verifier-gated transition lemmas; PLAN only after the model passes mechanic checks) + **Agent2World adaptive
  world-model testing** (arXiv:2512.22336 — adaptive unit tests surface the hidden-rule gap). Drive to **L1**.
  The codex agent IS the proposer (NO nested CodexProposer). `verifier_is_oracle=true` (execution-grounded).
- **exp4340** E3 **ka59** (0.56) → L1, same upgraded explore-verify-plan harness. `verifier_is_oracle=true`.

### PHASE C — ARC NORTH STAR: sc25 reproduction via E3 (the biggest reproducible-level upside)
- **exp4341** E3 executable-world-model on **sc25** to crack the win-mechanic the BFS solver stalled on (spell-cast +
  3×3 cast-grid alignment + tank-controls; the registry has rich mechanic notes). Target: reproduce **L1 offline**
  (a +1, opening the path to the 5 live-recorded levels → up to +5). Honest partial (best verifier accuracy + the
  win-mechanic gap) is progress. `verifier_is_oracle=true`.

### PHASE D — CONTINUOUS SELF-LEARNING (mandatory; NEW mechanism)
- **exp4342** train a game-AGNOSTIC **ACTION-ROLE / object-interaction encoder** (ReactiveGWM, arXiv:2605.15256) over
  the solved games' traces; re-test leave-one-game-out cross-game value transfer for SEARCH EFFICIENCY (the gap exp4331
  logged — raw-frame features failed). HARD: report state-count reduction under leave-one-game-out, positive control.
  `retire_if_same_verdict: true` (a 3rd null across feature representations retires the cross-game-transfer direction).
  `verifier_is_oracle=false`.

### PHASE E — INFRA + HYGIENE + CAPSTONE
- **exp4343** SOTA-ingestion → `.402 (reliable channel only; `/deep-research` banned in-loop). Conditioned on the
  Phase-A outcome: if the moat is retired, surface the consequence-based oracle-free verifier (arXiv:2602.06291) as the
  lead `.402 candidate.
- **exp4344** verifier registry/gaps hygiene + GAP-4 execution regression guard (robust aggregate-available helper).
- **exp4345** hardware continuity (opportunistic per north-star §3; KV260 SSH-only, never host SD-card).
- **exp4346** capstone `.401 — the verifier scorecard + the **DiffusionGemma gate decision** (did the leak-robust
  scorer replicate the moat across ≥2 corpora with matched controls + CI95-excl-0 — or is the moat retired?) +
  ARC reproducible-total + G1–G4 via `publication_gate.py`.

---

## 5. Dependency graph

```
exp4336 (transition)
   ├─ PHASE A ─ exp4337 (build leak-robust scorer + masked leak audit) ─┐
   │            exp4338 (in-gen moat replication, 2nd corpus) ◀─ gated on exp4337 scorer_leak_audit_passed==true
   ├─ PHASE B ─ exp4339 (E3 ar25 explore-verify-plan) ─────────────────┤
   │            exp4340 (E3 ka59 explore-verify-plan) ─────────────────┤
   ├─ PHASE C ─ exp4341 (E3 sc25 reproduction) ────────────────────────┤
   ├─ PHASE D ─ exp4342 (action-role cross-game encoder) ──────────────┤
   └─ PHASE E ─ exp4343 (sota→v402) ───────────────────────────────────┤
                exp4344 (registry/gaps hygiene) ───────────────────────┤
                exp4345 (hardware continuity) ─────────────────────────┤
                exp4346 (capstone .401) ◀──────────────────────────────┘  reads exp4337/4338 (gate decision) + exp4339-4342
```

ONE structured `gated_on` gate: exp4338 depends on exp4337's `scorer_leak_audit_passed==true` (if no leak-robust scorer
can be built, the replication is gate-skipped and the gate is decided NO → moat retired). All other phases are
independent DEPTH; the capstone aggregates whatever lands (robust aggregate-available helper — a missing artifact for
one axis is a per-axis gap, never an all-False).

---

## 6. Hardware requirements

- **PHASE A (DiffusionGemma)**: 1× RTX 3090 (Q4_K_M GGUF, 16 GB) via the llama.cpp PR-#24423 binary
  `~/.cache/llama.cpp-master/build/bin/llama-diffusion-gemma-eval` (NOT a standard GGUF loader — known-issues
  2026-06-15). Partial-state scorer (reward head) on CPU/1×3090.
- **PHASE B/C (E3)**: codex/gpt-5.5 induces the world model; offline ARC sim (`environment_files/<game>/`), zero quota.
- **PHASE D (self-learning)**: CPU (small action-role encoder + value-head over cached traces).
- **PHASE E**: CPU / aggregation; hardware-continuity probes KV260 (SSH), PolarFire (SSH), GateMate (USB) opportunistically.

---

## 7. Discipline compliance

- **Codex-Default v2 (2026-06-10):** every task `agent_type: codex` + `model: gpt-5.5`. Planner/retro/audits stay Opus
  (this doc). E3 tasks REQUIRE codex (the codex agent IS the world-model proposer — do NOT nest a `CodexProposer`).
- **ARC-AGI-3 Incremental-Progress Scoping:** every solve task targets +1 level (NOT all-levels); monotonic.
- **ARC Solve Reproducibility + Solver-Reuse:** every solve emits `offline_reproduced` + `reproduced_levels`; only
  reproduced levels count; reuse `arc_solver_kit` / `arc_executable_world_model` + the registry; update the registry.
- **Circularity / Oracle-Distinctness:** every verifier task declares `verifier_is_oracle`. The in-generation +
  self-learning tasks are oracle-distinct (`false`, matched control, CI95-excl-0); the E3 solves are execution-grounded
  (`true`, ARC progress NOT a moat headline).
- **Failed-Experiment Rerun + Exclusion-Manifest Cross-Check:** every retired/failed-scope task carries `prior_failures:`
  (all four sub-fields) or `operator_override:`. The leak-robust scorer rebuild (exp4337) and the moat replication
  (exp4338) carry `retire_if_same_verdict: true` — the convergence/retirement mechanic. Cross-domain selection scope is
  RETIRED and NOT re-proposed.
- **Pre-Launch Preconditions / Inference-Substrate / Principle-Annotated Fields / Verdict Terminal-Prefix /
  gated-fields-must-be-bare:** honored per-task.
- **SOTA-Ingestion Cycle + Missing-Verifier Gap Logging + Hardware-Task Continuity + Overdue-Priority (2 infra slots):**
  reserved (exp4343/4344/4345; transition + hygiene + sota + hardware = 4 infra slots).
- **HARD RULE (carried from .400):** conductor STAYS STOOD-DOWN on TRM training — NO task launches TRM training, runs
  pkill/kill against train.py, or writes `results/trm_runs/`. Qwen FORBIDDEN as the trained base (Spurious-Rewards
  confound); Qwen/Gemma GGUF as an off-policy judge/generator is fine. SEPO (verifier-as-reward GENERATOR training) is
  OUT-OF-BAND/operator-owned — flagged, NOT auto-run. No autonomous edits to `docs/index.html`/README/paper prose. Online
  ARC play stays operator-gated (NO leaderboard submission). DiffusionGemma via the llama.cpp PR binary only.

---

## 8. Success criteria

A `.401 milestone is a success if it advances the north star or SETTLES the headline:
1. **Headline (replicate-or-retire):** either a leak-robust scorer replicates the in-generation oracle-distinct moat on
   a 2nd corpus with CI95-excl-0 (exp4338 → the gate flips, a hardened headline-grade result), OR a powered
   failure-to-replicate RETIRES the moat as corpus-specific (decision-grade convergence). The gate decision (exp4346)
   records which. A scorer that cannot be built leak-robust at all (exp4337) is itself decision-grade.
2. **ARC accuracy:** ≥1 deep-tail E3 solve (ar25/ka59 → L1, offline-reproduced) OR sc25 L1 reproduced via E3 →
   `reproducible_total_levels ≥ 14`. An honest E3 partial (induced model + residual-mismatch gap) is progress.
3. **Self-learning:** the action-role encoder either transfers cross-game (state-count reduction >1.0, CI lower bound
   >1.0) or logs a 3rd, sharper game-invariant-representation gap (and retires the direction).
4. Cross-domain selection is NOT re-run (retired); efficiency is NOT re-measured (settled); off-ARC is NOT
   re-accumulated (settled).
