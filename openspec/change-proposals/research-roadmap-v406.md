# Research Roadmap v406 — 2026.06.406

**Headline:** The verifier-as-DETECTOR DETECTS step-errors well (AUROC 0.92–0.98)
and now COMPOUNDS + GENERALIZES cross-domain — but it CANNOT LOCALIZE the
earliest causal error (`.405 exp4381: first-error F1 **0.096**, bidirectional
fusion a measured no-op, 103/114 missed first-error traces). The `.405 BiPRM
localization null is a **DATA problem**, not a fusion-method problem. `.406 turns
detection into an actionable **cross-domain first-error LOCALIZER** by
**synthesizing verifiable process data** (arXiv:2605.02395, the `.405 SOTA-ingestion
`flagged_for_v406`): construct a correct symbolic chain → inject a template-aware
error at a KNOWN step → recompute the suffix under the corrupted state → VERIFY
the injected step is not derivable from its prefix → train a contrastive
earliest-error localizer on ground-truth labels. CALIBRATE cross-domain detection
across FoVer + ARC + code + GSM8K with leave-one-domain-out (arXiv:2102.10395) +
build the pools the `.405 run could not reach. PROVE the localizer COMPOUNDS (the
mandated self-learning slot — the live vehicle now has REAL headroom: F1 0.096,
not a saturated AUROC). Drive the ARC north star DEEPER by raising world-model
lookahead-fidelity past the planning threshold (operator MANDATORY 2026-06-17).
`paper_ready=True` (FoVer 0.9131, G1–G4) carried unchanged.

**Provenance:** outer-loop planner (Claude Opus 4.8), 2026-06-18, after all 11
`.405 tasks completed. The `.405 SOTA-ingestion (exp4387) mapped this fork:
`flagged_for_v406 = verifiable_process_data_cross_domain_localization_v406`
(arXiv:2605.02395, independently WebSearch/WebFetch-verified 2026-06-18 — the
paper's own headline finding is *"first-error localization remains substantially
more challenging than overall step classification, highlighting the need for
fine-grained and verifiable process supervision"*, exactly the `.405 result).

---

## 1. What .405 proved (the honest scorecard)

Read via `scripts/summarize_artifact.py` + the `.405 capstone (exp4390,
`verdict: v405_detector_detects_but_not_actionable_compounds_true_generalizes_true_arc_levels_34_publication_ready`).
`.405 deepened the one alive oracle-distinct vehicle — the verifier-as-DETECTOR —
and split it into three readings:

- **LOCALIZATION → clean powered NULL (exp4381 `clean_powered_null_bidirectional_not_actionable`).**
  The detector DETECTS trace-level error risk well (abstention-curve AUROC 0.980),
  but `localization_f1_by_direction` is **identical** across all three directions
  — unidirectional L2R, bidirectional fusion, and causal-online all give F1
  **0.096** (11/114 error traces), `localization_delta_ci95 = [0.0, 0.0]`.
  Bidirectional fusion is a measured no-op. The detector flags a *downstream
  consequence* rather than the *earliest causal error*. Logged as
  `GAP-FOVER-BIPRM-LOCALIZATION-untyped` (status open). A2 (exp4382) correctly
  gate-skipped (no win to validate).
- **ABSTENTION → works mechanically, thin in absolute value.** Risk-coverage is
  clean (retained-accuracy 0.999 at 90% coverage), but the base rate is already
  98.3% correct, so threshold-only abstention reports "no useful operating point"
  — the win needs a structured report/abstain signal, not a raw score threshold.
- **COMPOUNDS → TRUE but weak (exp4385 `detector_compounds_heldout_localization_f1`).**
  As the corpus grows 491 → 4,911 the held-out localization-F1 nudges 0.371 →
  0.387 (`compounding_delta_ci95 = [0.0034, 0.0328]`, positive control passed),
  while AUROC sits saturated at 0.986. Genuine compounding, but the saturated
  AUROC left little headroom to climb.
- **GENERALIZES cross-domain → TRUE (exp4386 `detector_generalizes_cross_domain_non_fover`).**
  Detection on the GAP-4 ARC pool: AUROC **0.963**, CI95 [0.922, 0.991],
  n=28,443, `verifier_is_oracle=false`, `domains_at_chance=[]`. But only ONE
  non-FoVer domain was reachable — code/GSM8K pools were "unavailable until
  labeled verifier-score pools exist."
- **ARC north star → STALLED at 34 reproducible levels / 17 games** (no new
  levels). Both exp4383 (deeper: lp85 L6 / tu93 L5 / tn36 L8 / tr87 L7) and
  exp4384 (blocked tails: ar25 / ka59 / ft09 L2) returned `new_levels_reproduced=0`.
  Root cause is visible in the artifacts: world-model **lookahead-fidelity
  0.80–0.875**, below the threshold needed to plan deeper. The outer-loop is
  independently probing ka59 (object-RELEVANCE, not clicks) + curious/directed
  exploration (first-contact solves 1→3).
- **`paper_ready=True`** (G1–G4, FoVer 0.9131, `unmet_gates: []`). Operator
  submission only.

**The SOTA-ingestion's forward pointer (exp4387):**
`flagged_for_v406 = verifiable_process_data_cross_domain_localization_v406`.
The strongest method (arXiv:2605.02395) reframes the localization null as a
supervision-data problem and supplies the fix — verifiable synthetic first-error
pairs. It also flagged: multi-domain calibration (2102.10395) to scale the
cross-domain win; Prover-Verifier Deliberation (2605.25133) and ThinkPRM
(2504.16828) as secondary abstention/label sources; and Mind-Studio (2606.16070)
to keep ARC E3 as *mechanic-gap repair*, NOT the detector headline. A2D2
(2606.13565) + SEPO (2502.01384) stay OUT-OF-BAND/operator-owned.

---

## 2. The three biggest gaps (current state vs PRD / north-star vision)

1. **The detector DETECTS but cannot LOCALIZE — and that blocks the actionable
   second-headline.** north-star §0 names the oracle-distinct frontier as THE
   open claim; §5 names the verifier as Carnot's entire value-add. Detection
   "beats chance" + compounds + generalizes, but a trace-level flag is not a
   usable capability — the defensible, actionable result is *which step is the
   first error*. The `.405 BiPRM fusion proved this is NOT a method-tuning problem
   (fusion was a literal no-op); it is a **supervision-data** problem — there are
   no ground-truth earliest-causal-error labels to learn from. **Gap: synthesize
   verifiable first-error supervision and train a localizer that beats the 0.096
   ensemble baseline, cross-domain.** → Phase A (headline).

2. **Cross-domain detection is real but THIN — one non-FoVer domain.** north-star
   §0 step 2 (verifier domain expansion) + memory `verifier-domain-bound-math-only`.
   exp4386 generalized to GAP-4 ARC only; code/GSM pools were unavailable. A
   two-domain result is not an OOD claim, and calibration (not just ranking) is
   what makes a detector a deployable contract. **Gap: build the missing pools
   and calibrate detection across FoVer + ARC + code + GSM with leave-one-domain-
   out.** → Phase D (complementary, cached, cheap).

3. **ARC-AGI-3 north star — deep levels blocked on world-model FIDELITY, not
   search.** 34/17 levels; exp4383/4384 both stalled at lookahead-fidelity
   0.80–0.875. The un-reproduced next levels need the induced world model's
   K-step rollout to MATCH the env before planning can reach them. Operator
   MANDATORY 2026-06-17 (incremental +1..+n per game). **Gap: gate planning on a
   fidelity threshold and target the specific mechanic that breaks the rollout.**
   → Phase B.

---

## 3. Architecture — where .406 acts

```
                 ┌───────────────────────────── the ONE alive oracle-distinct vehicle ─────────────────────────────┐
                 │                                  verifier-as-DETECTOR                                            │
   cached FoVer  │   DETECTS (trace-level)        LOCALIZES (first error)        ABSTAINS (selective prediction)    │
   + ARC + code  │   AUROC 0.92–0.98  ✅ .404/.405   F1 0.096  ❌ .405 NULL          works, thin  ◐ .405             │
   + GSM traces  │        │                            │                                  │                          │
                 └────────┼────────────────────────────┼──────────────────────────────────┼──────────────────────────┘
                          │                            │                                  │
                          │            PHASE A (HEADLINE): verifiable process data        │  PHASE A abstention report
                          │            synthesis (2605.02395) → contrastive               │  (structured, not raw threshold)
                          │            earliest-error localizer → beat F1 0.096           │
                          ▼                            ▼                                  ▼
              PHASE D: CALIBRATED multi-domain   PHASE C: does the LOCALIZER          A2 skeptic-proof:
              detection contract (2102.10395)    COMPOUND as verifiable + real        train-synthetic/test-real,
              FoVer+ARC+code+GSM, LODO,           labels accumulate? (real headroom    position-only control,
              build missing pools, log gaps       now: F1 0.096, not saturated)        template-ablation

   PHASE B (ARC north star, verifier_is_oracle=true): raise world-model lookahead-fidelity past the planning
   threshold (Mind-Studio 2606.16070) → lp85 L6 / tu93 L5 / tn36 L8 / tr87 L7 (B1) + ar25/ka59/ft09 L2 (B2).
```

Every Phase-A/C/D claim is **oracle-distinct** (`verifier_is_oracle=false` +
matched control + CI95-excl-0). ARC SOLVEs are execution-grounded
(`verifier_is_oracle=true`) — ARC progress, NOT a moat headline (Circularity
Discipline).

---

## 4. Phase descriptions

### Phase 0 — Transition (exp4391)
Archive `.405 → activate `.406; record the TRUE close-state (detector
detects-but-not-actionable; localization a DATA-bottlenecked null; compounds +
generalizes TRUE; ARC 34/17 fidelity-blocked; paper_ready=True). Codex,
mechanical.

### Phase A — HEADLINE: verifiable process data → actionable cross-domain localizer
- **A1 (exp4392)** — Synthesize verifiable first-error process data (2605.02395):
  symbolic chain → template-aware error injection at a known step → suffix
  recomputation → prefix-invalidity verification. Train a contrastive
  earliest-error localizer on the ground-truth labels; evaluate first-error
  localization F1 on a HELD-OUT real FoVer split AND the GAP-4 ARC split vs the
  `.405 ensemble baseline (0.096). Report a structured abstention signal
  (Prover-Verifier-Deliberation-style, 2605.25133) alongside the raw-threshold
  curve. Emit `localizer_beats_ensemble_baseline` BARE bool. Cached + CPU; the
  symbolic synthesis needs no live LLM. **The headline — directly attacks the
  one gap the `.405 fusion null left open.**
- **A2 (exp4393, GATED on A1 win)** — Skeptic-proof (twice-burned operator): is
  the localization gain genuine, or (a) template-artifact leakage (train-synthetic
  / test-REAL first-error labels), (b) position/length bias (content-blind
  position-only baseline), (c) synthetic-distribution overfit (held-out split +
  second seed)? Emit `localizer_win_is_genuine` BARE bool.

### Phase B — ARC north star (operator MANDATORY; verifier_is_oracle=true)
- **B1 (exp4394)** — E3 DEEPER with a **lookahead-fidelity GATE before planning**:
  the `.405 partials stalled at fidelity 0.80–0.875, so B1 targets raising fidelity
  to the planning threshold (Mind-Studio 2606.16070 K-step rollout matching)
  BEFORE attempting lp85 L6 / tu93 L5 / tn36 L8 / tr87 L7. prior_failures on
  exp4383.
- **B2 (exp4395)** — E3 BLOCKED-mechanic tails ar25 L2 (action7 undo-stack) /
  ka59 L2 (step-counter HUD; fold in the outer-loop's fresh object-RELEVANCE
  finding) / ft09 L2. prior_failures on exp4384/exp4373.

### Phase C — Continuous self-learning (mandated; verifier_is_oracle=false)
- **C (exp4396)** — Does the **localizer** COMPOUND as verifiable-synthetic + real
  first-error labels accumulate? Reuse the exp4364/exp4385 compounding-curve
  discipline, but on the live localizer where headroom is REAL (F1 0.096, not a
  saturated AUROC). Positive control (full-corpus ceiling) + no-learning baseline.
  Emit `localizer_compounds` BARE bool. If A1 produced no localizer, fall back to
  the ensemble-detector compounding reading on the held-out localization metric.

### Phase D — Cross-domain detection calibration + pool expansion (verifier_is_oracle=false)
- **D (exp4397)** — Scale the cross-domain win (2102.10395): score the detector
  against FoVer + GAP-4 ARC + **code (HumanEval, cached exp1999/2838/2839)** +
  **GSM8K (cached)** pools; build any pool that needs assembly from EXISTING
  cached candidates (no new live inference); apply leave-one-domain-out
  calibration (temperature/Platt) and report detection AUROC + risk-coverage +
  localization-F1 BY domain. Emit `detection_calibrated_multi_domain` BARE bool;
  log any chance-detection domain as a missing-verifier gap (the product backlog).
  This is cross-domain DETECTION, NOT the RETIRED cross-domain SELECTION axis
  (exp4314).

### Phase E — Infra + hygiene + capstone
- **E1 (exp4398)** SOTA-ingestion → `.407 (reliable channel only; verified arXiv
  IDs; flag A2D2/SEPO out-of-band).
- **E2 (exp4399)** registry/gaps hygiene + GAP-4 regression guard + durable
  capstone-stamp confirmation.
- **E3 (exp4400)** KV260 SSH-reachability continuity (opportunistic, north-star §3).
- **E4 (exp4401)** capstone `.406 + the headline decision (did the detector
  graduate from "detects but can't localize" to an actionable cross-domain
  localizer?) + G1–G4.

---

## 5. Dependency graph

```
exp4391 (archive .405→.406)
   │
   ├─ Phase A ─ exp4392 (A1 verifiable-data localizer) ──gated──▶ exp4393 (A2 skeptic-proof)
   │                       │
   │                       └────────────────────────────────────▶ exp4396 (C: localizer compounds; soft-reads A1)
   │
   ├─ Phase B ─ exp4394 (B1 E3 deeper + fidelity gate)
   │            exp4395 (B2 E3 blocked-mechanic tails)
   │
   ├─ Phase D ─ exp4397 (calibrated multi-domain detection; reads exp4386 + cached pools)
   │
   └─ Phase E ─ exp4398 (SOTA→.407) · exp4399 (hygiene+GAP-4) · exp4400 (KV260) · exp4401 (capstone)
                                                                                       ▲
                  exp4392/4393/4396/4397/4394/4395 all feed the capstone ─────────────┘
```

Only one hard `gated_on` edge: exp4393 ⟸ exp4392 `localizer_beats_ensemble_baseline==true`.
Phase C soft-reads A1 (falls back to the ensemble detector if A1 built no localizer).

---

## 6. Hardware requirements

- **Phase A / C / D (detector / localizer / cross-domain):** CPU, cached
  candidates, zero GPU/quota — deliberately infra-independent (the property that
  kept the detector vehicle alive while the GPU/GGUF in-generation path blocked 4×).
- **Phase B (ARC E3):** codex-as-proposer + Python world-model induction; no GGUF
  training (conductor STOOD-DOWN on TRM training). SOTA GGUF named in model_specs
  as the optional NL-realization path, precondition-gated.
- **Phase E:** aggregation (CPU) + KV260 SSH-reachability (no host SD card).

---

## 7. SOTA mapping (`.405 ingestion exp4387, verified 2026-06-18)

| arXiv | Method | `.406 use | Failure mode guarded |
|---|---|---|---|
| **2605.02395** | Controllable & Verifiable Process Data Synthesis for PRMs | **A1 headline** — synthesize prefix-invalid first-error pairs, train contrastive localizer | synthetic errors miss ARC mechanics / leak template artifacts → keep source labels + LODO + executable prefix checks |
| 2102.10395 | Multi-domain calibration for OOD detectors | **D** — calibrated multi-domain contract, explicit unavailable-domain gaps | two domains too few → build code/GSM pools, report by-domain |
| 2605.25133 | Prover-Verifier Deliberation (selective prediction) | **A1 abstention** — structured report/abstain over the localizer, not a raw threshold | collapses outside the verifier's region → abstention layer over a validated localizer only |
| 2504.16828 | ThinkPRM generative step-wise verifier | A1 fallback label source for the untyped first-error gap | hallucinated rationales → check every label against symbolic/executable prefix |
| 2606.16070 | Mind-Studio executable world models + lookahead | **B1/B2** — fidelity-gated ARC mechanic-gap repair (north star, NOT headline) | lookahead stayed partial/oracle-grounded → fidelity gate before plan |
| 2601.17223 / 2505.15960 | Verifiable / formally-verified process supervision (corroborators) | A1 design corroboration (verifiable process data → generalizable PRM) | — |
| ~~2606.13565 (A2D2)~~ / ~~2502.01384 (SEPO)~~ | verifier-as-reward GENERATOR training | **OUT-OF-BAND / operator-owned — NOT auto-run** | — |

---

## 8. Invariants carried from .405

`paper_ready=True` (G1–G4; FoVer 0.9131 is the FROZEN headline — `.406 adds the
localization / calibration / compounding LENSES, never a substitute). Oracle-
distinct discipline (`verifier_is_oracle=false` + matched control on every
verifier-value task; ARC SOLVEs `verifier_is_oracle=true`, NOT moat headlines).
Conductor STOOD-DOWN on TRM training (NO task launches TRM training / pkill
train.py / writes results/trm_runs/). Qwen FORBIDDEN as the TRAINED base
(Spurious-Rewards confound); Qwen/Gemma GGUF as an off-policy judge/generator is
fine. A2D2/SEPO verifier-as-reward GENERATOR-training OUT-OF-BAND/operator-owned.
Cross-game value TRANSFER (exp4342) + cross-domain SELECTION (exp4314) RETIRED —
NOT re-proposed (Phase A/D are DETECTION/LOCALIZATION, a distinct measurement).
DiffusionGemma in-generation conversion RETIRED (4th block, exp4374) — NOT
re-proposed. The LLM-generated-heuristic efficiency arm is SETTLED (exp4370
null) — control only. NO autonomous edits to docs/index.html / README / paper
prose. Online ARC play stays operator-gated (NO leaderboard submission; only
offline-reproduced levels count).
