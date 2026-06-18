# Research Roadmap v404 — 2026.06.404

**Status:** PROPOSED (pre-staged by the outer-loop planner, Claude Opus 4.8, 2026-06-18)
**Milestone:** 2026.06.404
**Prior milestone:** 2026.06.403 (`research-roadmap.yaml`)
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3 accurately AND efficiently; the energy
verifier is the load-bearing, oracle-distinct contribution to the system's solve-rate + action-efficiency.

---

## 1. What .403 proved (the honest scorecard)

Read from the `exp4368` capstone (`s3_moat_utility=open`, `verifier_thesis_state=harness_still_open`,
`reproducible_total_levels=33`, `action_efficiency_compounds=true`, `paper_ready=True`) via
`scripts/summarize_artifact.py`:

| Axis | .403 outcome | Status |
|---|---|---|
| **HEADLINE — DiffusionGemma moat→generation conversion** | exp4359 Prism-hardened verifier-guided denoising search halted at the independent leak re-check: `scorer_leaky_in_search_corpus`, `benchmark_n=0`, `controls_differentiated=false`. The .401 leak-robust scorer (exp4337) leaked on the NEW free-form generation corpus — leak-robustness is **corpus-specific**. PAPO (exp4360) correctly `blocked_gate_check_failed`. | ❌ **OPEN (3rd consecutive block:** .399 degenerate-controls → .402 MCQ-harness-bug → .403 scorer-leak) |
| **ARC north star (accuracy)** | 26 → **33** reproducible levels / 17 games (registry authoritative). tu93 +1 (exp4361); tr87/ft09 reconciled; ar25/ka59 partial (exp4362). | ✅ **+7 levels** |
| **Self-learning (efficiency)** | exp4364: learned A* action-cost heuristic **COMPOUNDS** (held-out env-actions 25→16), positive-control-passed, reproduction-gated, **DEPLOYED** into `arc_solver_kit`, `verifier_is_oracle=false`. The stronger LLM-generated-heuristic arm did **not** run. | ✅ **clean WIN (oracle-distinct)** |
| **Publication gate** | `publication_gate.py`: `paper_ready=True`, `unmet_gates=[]` (FoVer 0.9131, G1–G4). | ✅ **MET** (operator submission only) |

**The decisive lesson:** the project has chased the verifier-moat's UTILITY through the **DiffusionGemma
in-generation accuracy** vehicle for three milestones and it keeps breaking on **infra/methodology**, not
science. Meanwhile the **efficiency** vehicle (the learned action-cost heuristic) landed a **clean,
oracle-distinct, deployed, compounding WIN**. The SOTA-ingestion's own discover→ingest→plan output
(`exp4365.flagged_for_v404 = llm_generated_action_heuristics_compounding_v404`) points the .404 headline at
**building on the win**, not re-hammering the blocked path — exactly the north-star §1 / Depth-Over-Breadth
ethos.

---

## 2. The three biggest gaps (current state vs PRD / north-star vision)

1. **The verifier-moat's EFFICIENCY utility is proven-but-shallow (linear heuristic only).** exp4364
   proved a *linear* learned action-cost heuristic compounds. The PRD's continuous-self-learning vision
   ("Carnot must get smarter over time; every solve makes the next faster") and the north-star EFFICIENCY
   axis (ARC-AGI-3 scores action-efficiency vs a human baseline — arXiv:2603.24621) call for a **stronger
   function class**. arXiv:2503.18809 (LLM-generated Python heuristic programs) is the SOTA method and the
   flagged .404 headline. Gap: we have not tested whether a stronger learned heuristic class compounds
   *further* / generalizes better — the difference between "the verifier helps a little" and "the verifier
   is the efficiency engine."

2. **ARC-AGI-3 accuracy depth: 33 levels, but the deep tails are blocked on NAMED hidden-rule gaps.**
   sc25 (1 of 5 live-recorded; spell-delta gap), ar25 L2 (action7 undo-stack), ka59 L2 (step-counter HUD),
   tn36 L8 (program-editor control), tr87 L7, lp85/tu93 L5, ft09 L2. The north star is the LIVE benchmark;
   monotonic offline-reproducible-level growth is the de-risking metric (operator 2026-06-17 E3 MANDATORY).

3. **The in-generation ACCURACY utility cannot be MEASURED because the instrument keeps breaking.** The
   scorer-leak (corpus-specific), the MCQ-harness collapse, and the degenerate controls are all measurement
   failures. We need a **scorer-INDEPENDENT** control (CoDiLA, arXiv:2603.20216) so the measurement is not
   hostage to the leaky external scorer — OR a falsifiable RETIREMENT of the in-generation-via-this-scorer
   direction. Separately, the **oracle-distinct ACCURACY frontier** (the standing 2026-06-14 P0 MANDATORY)
   has a cheap complementary probe — verifier-as-DETECTOR AUROC where selection headroom is ~0 — that is
   NOT hostage to the DiffusionGemma infra and diversifies the "verifier earns its place" bet.

---

## 3. Milestone design — 11 experiments across 5 phases

```
PHASE 0  TRANSITION ───────────────────────────────────────────────────────────────────
  exp4369  archive .403 → activate .404; record the TRUE .403 close-state

PHASE A  HEADLINE — oracle-distinct EFFICIENCY moat, STRONGER function class (verifier_is_oracle=false)
  exp4370  A1  LLM-generated action-cost heuristics (2503.18809) — stronger arm vs the DEPLOYED linear
               heuristic; fresh held-out levels + static-leakage analysis + reproduction gate
  exp4371  A2  [GATED on A1 win] contamination/leakage skeptic-proofing (the twice-burned-operator gate)

PHASE B  ARC NORTH STAR — accuracy, +1..+n per game (operator 2026-06-17 E3 MANDATORY; verifier_is_oracle=true)
  exp4372  B1  E3 DEEPER high-headroom: tn36 L8 · tr87 L7 · lp85 L5 · tu93 L5 · sc25 L2
  exp4373  B2  E3 BLOCKED-mechanic: ar25 L2 (action7 undo-stack) · ka59 L2 (step-counter HUD) · ft09 L2

PHASE C  IN-GENERATION ACCURACY moat — REPAIR-OR-RETIRE (verifier_is_oracle=false)
  exp4374  C1  scorer leak-REPAIR (requalify ON the generation corpus) + scorer-INDEPENDENT CoDiLA control
               (2603.20216) + PAPO diagnostic (2606.08501) → fixed-NFE Prism search OR clean RETIREMENT

PHASE D  ORACLE-DISTINCT ACCURACY — complementary measurement (2026-06-14 P0 MANDATORY; verifier_is_oracle=false)
  exp4375  D1  verifier-as-DETECTOR AUROC where SELECTION headroom is ~0 (cheap, cached, infra-independent)

PHASE E  INFRA + HYGIENE + CAPSTONE ──────────────────────────────────────────────────────
  exp4376  SOTA-ingestion → .405 (reliable channel only; /deep-research BANNED in-loop)
  exp4377  registry/gaps hygiene + GAP-4 regression guard + durable verifier_is_oracle stamp
  exp4378  hardware continuity — KV260 SSH-reachability (opportunistic, north-star §3)
  exp4379  CAPSTONE .404 — scorecard + headline decision + G1-G4 publication gate
```

### Dependency graph

```
exp4369 (transition)
   │
   ├─► exp4370 (A1 LLM-heuristics) ──gated──► exp4371 (A2 skeptic-proof)
   ├─► exp4372 (B1 E3 deeper)        [independent]
   ├─► exp4373 (B2 E3 blocked-mech)  [independent]
   ├─► exp4374 (C1 DiffusionGemma repair-or-retire)  [independent]
   └─► exp4375 (D1 detector)         [independent]
            │
   exp4376 (SOTA-ingestion → .405)   [reads .404 outcomes]
   exp4377 (hygiene + GAP-4 guard)   [reads .404 outcomes]
   exp4378 (KV260 continuity)        [independent]
   exp4379 (CAPSTONE) ◄── aggregates exp4370,4371,4372,4373,4374,4375 (+ registry, publication_gate)
```

Only **A2 is gated** (on A1's `llm_heuristic_beats_linear==true`). Everything else runs independently so a
single-task block never cascades. The capstone aggregates available artifacts (robust
aggregate-available-report-gaps helper — NO hard-block-all-False, per the exp4301 lesson).

---

## 4. Phase detail

### PHASE A — HEADLINE: the oracle-distinct EFFICIENCY moat, stronger function class

**Thesis.** The north-star §5 win condition is "the verifier earns its place — equally effective at lower
cost." exp4364 realized this on ARC action-efficiency with a *linear* learned heuristic (a clean,
oracle-distinct, deployed, compounding win). The natural depth step (arXiv:2503.18809) is a **stronger
function class**: ask the LLM (the codex/gpt-5.5 proposer, with a SOTA GGUF declared as the reproducible
generator) to write several **domain-dependent Python heuristic programs** per ARC game, select the
strongest by greedy-best-first-search on TRAINING levels, then **evaluate on FRESH HELD-OUT levels** vs the
DEPLOYED linear cost. A win = the verifier-moat's efficiency utility deepens (a stronger learned heuristic
compounds further / generalizes better); a clean null = the linear cost is already near-optimal for our
solved games (decision-grade, the function class is settled).

- **A1 (exp4370)** is the headline. HARD gate: the best clean LLM-generated heuristic reduces held-out
  actions-to-solve **below** the deployed linear heuristic AND passes static-leakage analysis (no
  env-internal/answer-cell access) AND every counted plan still `arc_solver_kit.reproduce`s.
  `verifier_is_oracle=false`. The action-cost heuristic ESTIMATES cost-to-go; the executable env defines the
  win — the heuristic is oracle-DISTINCT, not the oracle (consistent with exp4364, which scanned clean).
- **A2 (exp4371, gated)** is the skeptic-proof a twice-burned operator requires: verify the A1 win is NOT
  from public-ARC-layout memorization, hidden game-specific shortcuts, or single-held-out-split overfit —
  via fresh games outside the likely training overlap + a static+dynamic leakage audit + the compounding
  curve on a held-out trace corpus. This is the efficiency-axis analog of the .403 PAPO reward-state
  diagnostic.

### PHASE B — ARC NORTH STAR (accuracy; operator 2026-06-17 E3 MANDATORY; +1..+n per game)

The executable-world-model coding agent (arXiv:2605.05138) is the SOTA for full ARC-AGI-3 solves and the
harness is built+validated (`python/carnot/agentic/arc_executable_world_model.py` + `scripts/arc_e3_solve.py`).
Per ARC-AGI-3 Incremental-Progress Scoping, every solve task targets +1..+n NEW levels on ONE game, NOT a
full solve. The codex agent IS the proposer (NO nested `CodexProposer`). `verifier_is_oracle=true` (the
solve is execution-grounded — ARC progress, NOT a moat headline).

- **B1 (exp4372)** — the high-headroom games with the most reproduced levels / biggest upside: tn36 L8
  (program-editor control), tr87 L7, lp85 L5, tu93 L5, sc25 L2 (1-of-5 live-recorded; +4 ceiling). Loop the
  five with a per-target checkpoint + hard wall-time cap (breadth-of-progress beats all-or-nothing).
- **B2 (exp4373)** — the blocked-mechanic next-levels on the named hidden-rule gaps: ar25 L2 (action7
  undo-stack), ka59 L2 (step-counter HUD), ft09 L2. Active-data collection where the inducer under-determines.

### PHASE C — IN-GENERATION ACCURACY moat: REPAIR-OR-RETIRE

The DiffusionGemma in-generation conversion has blocked 3×, each on a DIFFERENT measurement failure. .404
gives it **one disciplined attempt with a clean exit**, mapped by the SOTA-ingestion:

- **C1 (exp4374)** — (1) **requalify** the .401 leak-robust scorer's leak-robustness ON the free-form
  generation corpus (the .403 leak was corpus-specific); (2) add a **scorer-INDEPENDENT** local-coherence
  control (CoDiLA, arXiv:2603.20216 — a small local AR verifier / deterministic block-coherence penalty
  that needs NO external scorer) so the in-generation arms are measurable without the leaky scorer; (3) PAPO
  (arXiv:2606.08501) as the reward-state diagnostic; THEN the fixed-NFE Prism search vs differentiated
  controls. DiffusionGemma via the llama.cpp PR binary (NOT a standard GGUF loader). `verifier_is_oracle=false`.
  - `prior_failures`: exp4359 (`scorer_leaky_in_search_corpus`) + exp4348 (`controls_not_differentiable`).
    `retire_if_same_verdict: true` — if the scorer leaks AGAIN after corpus-specific requalification AND the
    CoDiLA scorer-independent control ALSO cannot differentiate the arms, the in-generation-conversion-via-this-scorer
    direction RETIRES (the 4th block = the falsifiable retirement gate). A CLEAN differentiated null (controls
    OK, scorer leak-free, Carnot does not beat best-of-N/SVF) is a DIFFERENT, decision-grade verdict and does
    NOT retire.

### PHASE D — ORACLE-DISTINCT ACCURACY: complementary detector measurement

The standing 2026-06-14 P0 MANDATORY: HEADLINE = an oracle-distinct learned/energy verifier; COMPLEMENTARY =
the verifier-as-DETECTOR measurement (cheap; cached data). The in-generation moat (Phase C) is the
oracle-distinct accuracy *headline* attempt; the detector measurement is the cheap complementary probe that
is NOT hostage to the DiffusionGemma infra — diversifying the "verifier earns its place" bet across THREE
vehicles (efficiency / in-generation / detection) instead of betting everything on the blocked path.

- **D1 (exp4375)** — verifier-as-DETECTOR AUROC where SELECTION headroom is ~0 (spec:
  `docs/research-notes/verifier-as-detector-measurement-spec.md`): does the ensemble DETECT step-errors
  (AUROC) even on corpora where it cannot SELECT a better answer? Cached candidates, no live LLM,
  `verifier_is_oracle=false`. If a prior detector measurement exists, extend it to a NEW corpus/condition
  (no churn).

### PHASE E — INFRA + HYGIENE + CAPSTONE

- **exp4376** SOTA-ingestion → .405 (reliable channel only — `sweep_clusters.py`/`sweep_semscholar.py` +
  low-concurrency WebSearch/WebFetch; /deep-research BANNED in-loop; every method a verified arXiv ID;
  A2D2/SEPO flagged out-of-band).
- **exp4377** registry/gaps hygiene + GAP-4 regression guard + confirm the durable `verifier_is_oracle`
  capstone stamp fix (audit-only; NO production verifier edits).
- **exp4378** KV260 SSH-reachability continuity (opportunistic per north-star §3; NEVER a host SD-card
  precondition).
- **exp4379** CAPSTONE: scorecard + headline decision (did the stronger learned-heuristic class deepen the
  efficiency moat? did the DiffusionGemma repair convert or retire? ARC reproducible-total; G1-G4) — robust
  aggregate-available helper, SKIP `flagged_adversarial`, HONOR `verifier_is_oracle`.

---

## 5. HARD RULES (carried into every .404 task)

- **Conductor STOOD-DOWN on TRM training.** NO task may launch TRM training, run `pkill/kill` against
  `train.py`, or WRITE `results/trm_runs/`. (`.404 is a no-training test-time / offline-search / induction milestone.)
- **Qwen FORBIDDEN as the TRAINED base** (Spurious-Rewards confound). Qwen/Gemma GGUF as an off-policy
  JUDGE/GENERATOR is fine.
- **SEPO (2502.01384) + A2D2 (2606.13565)** (verifier-as-reward GENERATOR training) are OUT-OF-BAND /
  operator-owned — flagged in SOTA-ingestion, NOT auto-run in-loop.
- **Every LEARNED-verifier value task declares `verifier_is_oracle` honestly** (Circularity Discipline). An
  execution-grounded ARC solve (E3) is `verifier_is_oracle=true` (progress, NOT a moat headline); the
  learned action-cost heuristic + the in-generation scorer + the detector are oracle-distinct
  (`verifier_is_oracle=false`).
- **NO autonomous edits to `docs/index.html` / README / paper prose.**
- **Online ARC play stays operator-gated** (NO leaderboard submission; only offline-reproduced levels count).
- **DiffusionGemma via the llama.cpp PR binary** (`~/.cache/llama.cpp-master/build/bin/llama-diffusion-gemma-eval`),
  NOT a standard GGUF loader (known-issues 2026-06-15).
- **Cross-game value TRANSFER (exp4342) and cross-domain SELECTION (exp4314) are RETIRED** — do NOT re-propose either.
- **SOTA models:** any task that generates/judges with an LLM declares a SOTA GGUF in `model_specs`
  (Qwen3.6-35B-A3B / gemma-4-31B / gemma-4-26B-A4B / gemma-4-12B); legacy small models are CPU-smoke only.

---

## 6. Hardware requirements

| Task | Hardware | Notes |
|---|---|---|
| exp4370/4371 (LLM-heuristics) | CPU (offline search) + codex/gpt-5.5 proposer | GGUF generator declared; reproduction-gated offline search is CPU |
| exp4372/4373 (E3) | CPU (offline env sim) + codex/gpt-5.5 proposer | `environment_files/<game>/`; zero quota for transition collection |
| exp4374 (DiffusionGemma) | 1× RTX 3090 (Q4_K_M GGUF, 16GB) via the PR binary | precondition-gated; CoDiLA control is a small local AR verifier |
| exp4375 (detector) | CPU (cached candidates) | `verifier_ensemble_against_cached_candidates` |
| exp4378 (KV260) | KV260 via `ssh kria` | opportunistic; clean documented skip if unreachable |

All other tasks (transition, SOTA-ingestion, hygiene, capstone) are CPU aggregation.

---

## 7. Success criteria (the .404 capstone reads these)

1. **`llm_heuristic_beats_linear`** (BARE bool) — did the stronger function class deepen the efficiency moat
   (clean, leakage-audited, reproduction-gated)? — the headline.
2. **`reproducible_total_levels`** (BARE int, ≥ 33) — the monotonic ARC accuracy signal.
3. **`s3_moat_utility`** ∈ {useful_generation_gain / proven_but_not_useful / retired / open} — the
   in-generation conversion decision (or a clean retirement).
4. **`detector_auroc`** + the oracle-distinct accuracy frontier reading.
5. **`publication_gate`** — G1-G4 (`paper_ready` + `unmet_gates`), the stable finish line.
