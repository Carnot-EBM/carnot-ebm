# Research Roadmap v261 — Milestone 2026.05.261

**Proposed:** 2026-05-21
**Milestone:** 2026.05.261
**Previous milestone:** 2026.05.260 (exp2738–exp2750)
**Experiment ID range:** exp2751–exp2763
**Milestone title:** Verifier Live-GPU v3 + Phase 4 FEP Redesign v2 + Empirical Delta Audit + Ensemble v12 + Conformal Routing

---

## What Milestone .260 Proved

Milestone 2026.05.260 completed with 10 of 12 acceptance criteria met (exp2738–exp2750 produced
artifacts).

**Key outcomes from .260:**

1. **KV260 GRADUATED** (exp2742): kv260_terminal=true. Board-level Ising sampler latency transcript
   landed — 3.183μs mean UIO read latency, uio_count=5, bitstream loaded, n_cycles=100. KV260
   permanently exits per-milestone mandatory hardware tracking. All three FPGA boards have now
   reached terminal status (GateMate .247, PolarFire .241, KV260 .260).

2. **FR-11 Tier 4 learning loop closed** (exp2747): learning_loop_closed=true. auroc_cycle1=0.755 →
   auroc_cycle3=1.0, delta=+0.245. BUT auroc_cycle2=1.0 and auroc_cycle3=1.0 are both exactly 1.0
   — adversarial re-check needed (IMPLAUSIBLE_PERFECT risk).

3. **New verifiers viable** (exp2743, exp2746): Tier 0v set-consistency AUROC=0.818, viable.
   Tier 0w paraphrastic consistency module created.

4. **Phase 4 FEP FAILS** (exp2748): fep_auroc=0.489 ~ random; alpha values unscaled and signed
   ([-1.23, -1.27, 0.06, -1.23, ~0, 2.34, 25.9, 0.0]). Simple covariance/variance formula
   produces unstable, unscaled alpha estimates. ODAR heuristic vastly outperforms the FEP
   implementation (fep_vs_odar_delta = -0.484). Root cause: needs normalized/learned alpha weights.

5. **Verifier live-GPU BLOCKED** (exp2740): blocked_gguf_qwen36_not_cached. CUDA available but
   Qwen3.6-35B-A3B-GGUF not in ~/.cache/huggingface/hub. Fallback model gemma-4-26B-A4B-it-GGUF
   IS cached — use it for .261 live-GPU validation.

6. **Empirical delta = 0.000** (exp2744): sum_successes=0, sum_attempts=131. Zero repair success
   rate is suspicious — either measurement bug or the repair pipeline genuinely never succeeds on
   this corpus. Must diagnose before paper-v6 can cite the 4/delta convergence bound.

7. **Weak-strong policy threshold inversion** (exp2745): t_low=0.184 > t_high=0.107 (inverted!).
   82% accepted early with 0% FNR is likely an adversarial flag. Platt calibration orientation
   needs fixing.

8. **Tier 0g char-n-gram fix landed** (exp2741): root_cause=H1 TF-IDF collapse confirmed;
   char-n-gram TF-IDF fix applied. gguf_non_degenerate_post_fix=true. BUT adversarial-flagged
   (31.5s < 60s gate for real GGUF inference).

9. **arXiv package v2 compiled** (exp2749): 27pp, empirical delta updated (now 0.000), weak-strong
   + set-consistency cites added. paper_v6_submission_package_v2_ready=false because Phase 4 not
   validated.

---

## Three Biggest Gaps for .261

### Gap 1 (CRITICAL): Verifier Live-GPU claim never validated on real hardware

Two consecutive milestones have failed to land a non-adversarial verifier live-GPU artifact:
- exp2727 (.259): adversarial-flagged DURATION_TOO_SHORT (11.5s)
- exp2740 (.260): blocked_gguf_qwen36_not_cached

**Root cause of exp2740:** Qwen3.6-35B-A3B-GGUF not in HuggingFace cache. gemma-4-26B-A4B-it-GGUF
IS cached and was used by exp2741 as fallback (but exp2741 was also adversarial-flagged for
DURATION_TOO_SHORT=31.5s — the GGUF may not have been actually loaded for inference).

**Fix for .261:** exp2752 targets gemma-4-26B-A4B-it-GGUF explicitly in PRECONDITIONS step 0 (not
as a fallback — as the primary model). Strict duration gate: duration_s >= 60 (if model loads and
generates responses, this cannot be faked). N=30 live examples, random_seed=42.

**Addressed by:** exp2752 (Verifier Live-GPU v3: gemma-4-26B-A4B-it-GGUF direct GGUF inference).

### Gap 2 (CRITICAL): Phase 4 FEP aggregator produces random predictions

The fundamental Phase 4 hypothesis (verifier-as-alpha_t) is supported (alpha_t_nonzero=true), but
the specific aggregation formula is wrong: raw covariance/variance produces unscaled, sign-flipped
alpha values that lead to incoherent joint energy.

**Root cause:** `alpha_i = Cov(V_i_energy, label) / Var(V_i_energy)` produces:
- Negative alpha for verifiers negatively correlated with correctness (these should be INVERTED, not excluded)
- Wildly differing scales (alpha[6]=25.9 vs alpha[0]=-1.23) — sum is dominated by one verifier
- Zero alpha for verifiers with zero variance (degenerate verifiers)

**Fix for .261:** exp2753 tries multiple normalized pooling strategies:
1. Softmax-normalized |alpha| weights (discards sign, normalizes scale)
2. Learned logistic regression from calibration data (optimal linear combination)
3. Temperature-scaled geometric mean (robust to outlier alpha values)
4. Comparison against ODAR heuristic as baseline (fep must beat ODAR or match it)

**Gate:** fep_auroc >= 0.70 AND fep_vs_odar_delta >= 0.0 for FEP redesign to count as viable.

**Addressed by:** exp2753 (Phase 4 FEP Aggregator Redesign v2).

### Gap 3 (HIGH): Empirical delta = 0.000 contradicts paper-v6 headline claim

The 4/delta convergence bound (arXiv:2512.02080) requires delta > 0. delta=0 means E[n] → ∞.
If repair never succeeds (0/131 attempts), the paper's use of this bound is invalid.

**Root cause options:** (a) definitional mismatch — "n_repair_attempts" fields count something
different from what the delta formula expects; (b) repair pipeline regression since the metric was
first defined; (c) the FoVer subset used has no repairable examples (all violations are
irreparable for the models tested).

**Fix for .261:** exp2754 runs a diagnostic with verbose per-attempt logging on N=20 FoVer
violations, tracking: (a) whether ORCA TTT stops early (conformal stopping), (b) whether repair
attempts are attempted at all, (c) what "success" means in the repair loop.

**Addressed by:** exp2754 (Empirical Delta Root-Cause Audit).

---

## Architecture Snapshot

```
Live GGUF (gemma-4-26B-A4B-it-GGUF — confirmed cached)
        |
   [Conformal Selective Acting — anytime-valid routing (exp2757)]
   [Weak-Strong Policy v2 — fixed t_low < t_high (exp2758)]
        |
   [ODAR two-tier routing (exp2720, 65% savings)]
        |        |
     Accept   Full Ensemble v12 (exp2756)
              k=15 base
              + Tier 0g (char-n-gram TF-IDF, fixed exp2741)
              + Tier 0v (set-consistency, exp2743, AUROC=0.818)
              + Tier 0w (paraphrastic consistency, exp2746)
              + Tier 0y (differentiable conformal, exp2759)
              = k=18 ensemble
              |
         [Phase 4 FEP v2 joint energy (exp2753)]
         [ORCA-NEXUS FR-11 Tier 4 learning (exp2755)]
              |
         Repair / Accept
```

---

## Phase Structure

### Phase A — Archive + Admin (1 task)
- **exp2751**: Archive .260 + Activate .261

### Phase B — Critical Gap Fixes (3 tasks)
- **exp2752**: Verifier Live-GPU v3 — gemma-4-26B-A4B-it-GGUF direct GGUF inference, N=30,
  duration >= 60s gate, verifier_discriminative boolean primary gate
- **exp2753**: Phase 4 FEP Aggregator Redesign v2 — normalized alpha weights, 3 pooling
  strategies, target fep_auroc >= 0.70 AND fep_vs_odar_delta >= 0
- **exp2754**: Empirical Delta Root-Cause Audit — verbose repair logging on N=20 FoVer
  violations; diagnose why sum_successes=0/131; classify as bug vs negative result

### Phase C — Quality Checks + New Research (5 tasks)
- **exp2755**: FR-11 Tier 4 Adversarial Re-check v2 — independent test set, adversarial sanity
  for auroc_cycle2=1.0; FR-11 mandate (continuous_self_learning_task: true)
- **exp2756**: Ensemble v12 Integration — add Tier 0v + Tier 0g-char-ngram + Tier 0w to k=15
  base ensemble; measure AUROC lift over k=15 baseline
- **exp2757**: Conformal Selective Acting Tier 0x (arXiv:2605.20270) — anytime-valid risk
  control routing as principled replacement for threshold-calibrated weak-strong routing
- **exp2758**: Weak-Strong Policy Fix v2 — diagnose t_low > t_high inversion; fix calibration
  orientation; adversarial sanity check for 82% savings + 0% FNR
- **exp2759**: Differentiable Conformal Training Tier 0y (arXiv:2604.20098) — train Tier 0e
  calibration module end-to-end; target ECE < 0.005

### Phase D — Publication + Ship (3 tasks)
- **exp2760**: Phase 1 Ship Status v7 — confirm checklist from exp2730 is current; check if
  operator has tagged; if not, re-emit operator action list
- **exp2761**: Paper v6 Theory Update v4 — integrate FEP redesign result + ensemble v12 AUROC +
  empirical delta diagnosis; recompile; update arXiv package
- **exp2762**: arXiv Package v3 — compile final package with .261 research results (gated on
  exp2761.latex_compiles == true); produce operator checklist v3

### Phase E — Capstone (1 task)
- **exp2763**: Capstone v261 — cross-artifact synthesis, ops doc updates, gaps for .262
  (claude/opus, requires_claude: true)

---

## Dependency Graph

```
exp2751 (archive)
    |
    +-- exp2752 (verifier live-GPU v3)
    +-- exp2753 (FEP redesign v2)
    +-- exp2754 (empirical delta audit)
    +-- exp2755 (FR-11 Tier 4 re-check) [FR-11 mandate]
    +-- exp2756 (ensemble v12)
    +-- exp2757 (conformal selective acting)
    +-- exp2758 (weak-strong policy fix)
    +-- exp2759 (differentiable conformal)
    +-- exp2760 (Phase 1 ship status)
    |
    +-- exp2761 (paper v6 theory v4)
         |
         +-- exp2762 (arXiv package v3) [gated on exp2761.latex_compiles]
              |
              +-- exp2763 (capstone) [reads all]
```

---

## Hardware Continuity (CLAUDE.md Discipline Check)

**KV260:** GRADUATED in .260 (kv260_terminal=true, 3.183μs latency). Per CLAUDE.md Hardware-Task
Continuity Discipline terminal-state graduation clause: "once a board hits its terminal state, the
board CAN be dropped from per-milestone mandatory inclusion." KV260 exits mandatory tracking.

**GateMate:** GRADUATED .247. Already dropped from mandatory tracking.

**PolarFire:** GRADUATED .241. Already dropped from mandatory tracking.

**All three FPGA boards at terminal state.** No mandatory hardware task for .261.
Opportunistic: if Extropic Z1 SDK becomes available, queue a Phase-2 task in .262+.

---

## FR-11 Mandate

FR-11 (continuous self-learning) mandate: exp2755 (FR-11 Tier 4 Adversarial Re-check v2,
`continuous_self_learning_task: true`). This task re-validates the Tier 4 multi-cycle
learning loop with adversarial sanity checks on the IMPLAUSIBLE_PERFECT auroc_cycle2=1.0 result.

**FR-11 tier status:**
- FR-11 Tier 1 (online constraint weight learning): COMPLETED
- FR-11 Tier 2 (NEXUS constraint memory, real violations): COMPLETED .256
- FR-11 Tier 3 (ORCA conformal stopping): COMPLETED .258
- FR-11 Tier 3+ (ORCA-NEXUS integration, 17 rules): VIABLE .259
- FR-11 Tier 4 (multi-cycle learning delta): PENDING re-validation (adversarial re-check in exp2755)

---

## Agent Routing

| Task | Agent | Why |
|------|-------|-----|
| exp2751 (archive) | gemini | Mechanical admin |
| exp2752 (live-GPU v3) | gemini | GPU probe + inference — mechanical |
| exp2753 (FEP redesign) | gemini | Numerical optimization — mechanical |
| exp2754 (delta audit) | gemini | Diagnostic scan — mechanical |
| exp2755 (FR-11 re-check) | gemini | 3-cycle loop re-run — mechanical |
| exp2756 (ensemble v12) | gemini | Module wiring — mechanical |
| exp2757 (conformal routing) | gemini | Algorithm implementation — mechanical |
| exp2758 (weak-strong fix) | gemini | Calibration fix — mechanical |
| exp2759 (diff conformal) | gemini | Training module — mechanical |
| exp2760 (ship status) | gemini | Status check — mechanical |
| exp2761 (paper theory v4) | gemini | LaTeX splice — mechanical |
| exp2762 (arXiv package) | gemini | Package prep — mechanical |
| exp2763 (capstone) | claude/opus | Multi-artifact synthesis + doc updates — requires_claude |

**Routing summary:** 12 gemini + 1 claude/opus = 92.3% gemini. Within 2/13 ceiling for claude.

---

## Exclusion Manifest Cross-Check

Retired scopes checked against all .261 task titles and descriptions:
- `otv_kvcache_probe_retired` — no task proposes OTV one-token KV-cache probe. CLEAR.
- `diversity_maximizing_verifier_selection_retired` — no task proposes diversity-maximizing greedy
  selection. CLEAR.
- Integer exp IDs 260, 308, 309, 346, 380, 381, 382, 383, 410, 425, 491, 527, 603, 627: none of
  these ID scopes match any .261 task. CLEAR.
- HalluSAEGeometricProbe, discriminative JEPA (783, 799, 804, 809, 825, 887): not proposed. CLEAR.
- exp2091 (gemini CLI CSL grammar): not proposed. CLEAR.

**Result: 0 scope matches found.** All .261 tasks proceed without `operator_override:` requirement.

---

## Acceptance Criteria (12 total)

| # | Criterion | Experiment | Target |
|---|-----------|-----------|--------|
| 1 | Verifier discriminative on live GGUF | exp2752 | verifier_discriminative=true, duration_s>=60 |
| 2 | Phase 4 FEP redesign viable | exp2753 | fep_auroc>=0.70 AND fep_vs_odar_delta>=0 |
| 3 | Empirical delta diagnosed | exp2754 | delta_root_cause identified (bug or negative result) |
| 4 | FR-11 Tier 4 adversarially verified | exp2755 | learning_loop_revalidated=true |
| 5 | Ensemble v12 integrated | exp2756 | ensemble_v12_auroc >= ensemble_v11_auroc |
| 6 | Conformal routing viable | exp2757 | conformal_routing_viable=true, anytime_valid_guarantee=true |
| 7 | Weak-strong thresholds corrected | exp2758 | t_low < t_high (not inverted) |
| 8 | Differentiable conformal calibration improves ECE | exp2759 | tier0y_ece < 0.01 |
| 9 | Phase 1 ship checklist current | exp2760 | operator_ship_checklist_current=true |
| 10 | Paper v6 LaTeX compiles with .261 results | exp2761 | latex_compiles_v4=true |
| 11 | arXiv package v3 ready for operator | exp2762 | paper_v6_submission_package_v3_ready=true |
| 12 | Capstone synthesizes all artifacts | exp2763 | n_criteria_met >= 8 of 12 |

---

## New Research Papers for .261

From post-.260 arxiv sweep (2026-05-21):

1. **arXiv:2605.20270** — "Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained
   LLMs" (Khosravi, Huo). Anytime-valid conformal routing with formal coverage guarantees. Addresses
   threshold inversion in exp2745. **exp2757 targets this.**

2. **arXiv:2604.20098** — "Differentiable Conformal Training for LLM Reasoning Factuality"
   (Hittesdorf, Salzetta, Cheng). End-to-end training of calibration module for hallucination
   filtering. **exp2759 targets this** (already in research-references.md, carried from .260 sweep).
