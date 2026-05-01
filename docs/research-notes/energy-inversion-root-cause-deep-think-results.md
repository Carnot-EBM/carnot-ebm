# Deep Think Response — Energy Inversion Root Cause Diagnostic

**Status:** Response received 2026-05-01. Methodology-only response,
zero parameter prescriptions, all four hypotheses get a runnable
diagnostic test on *existing* artifacts. Strongest Deep Think
response of the day.
**Date received:** 2026-05-01
**Source prompt:** `energy-inversion-root-cause-deep-think-prompt.md`

---

## Sign convention (Deep Think established)

`ΔE = μ(E_incorrect) - μ(E_correct)`

- **Healthy:** ΔE > 0 (incorrect outputs have higher energy than correct).
- **Inverted:** ΔE < 0 (incorrect outputs have *lower* energy than correct — the bug we're diagnosing).

This avoids correlation-polarity ambiguity in the prior verbal description.

---

## The four diagnostic tests (verbatim from Deep Think)

### Test A — Corpus narrowness (pure distribution shift)

- **Required data:** exp1110 (ID train/val) + exp1100 (OOD SOTA eval). Keys: `energy`, `is_correct`.
- **Quantity:** ΔE_ID, ΔE_OOD.
- **Threshold (supports A):** ΔE_ID >> 0 AND ΔE_OOD < 0.
- **Threshold (rejects A):** ΔE_ID ≤ 0 (model inverted on its own training distribution).
- **Confidence caveat:** Less trustworthy if exp1110 validation is so deduplicated that ΔE_ID >> 0 reflects exact-string memorization.

### Test B — Loss-function geometry (structural margin sinkhole)

- **Required data:** exp1100 OOD eval JSON + exp1099/1110 training loss formula.
- **Quantity:** Per pair, compute theoretical loss L from frozen energies. Then `r(ΔE_pair, L_pair)`.
- **Threshold (supports B):** Pearson r STRICTLY POSITIVE (severely inverted pairs paradoxically yield LOWER loss under the formula). Loss geometry rewards inversion OOD.
- **Threshold (rejects B):** Inverted pairs yield massive loss penalties. Geometry fights inversion as intended; failure is in parameter generalization, not loss design.
- **Confidence caveat:** Drops if training loss has dynamic batch-level normalizations or contrastive denominators that can't be reconstructed per-pair.

### Test C — Lipschitz over-regularization (macroscopic smoothing)

- **Required data:** exp1100 eval JSON + text length per candidate.
- **Quantity:** R² (energy ~ length) AND partial correlation `r(energy, is_correct | length)`.
- **Threshold (supports C):** R²_length > 0.5 (length explains majority of variance) AND `r(energy, is_correct | length) ≈ 0` (no signal once length is controlled).
- **Threshold (rejects C):** R²_length < 0.1 AND inversion persists independent of length.
- **Confidence caveat:** Susceptible to omitted variable bias — could be reasoning-step count, code-block presence, or other low-frequency proxies, not just length.

### Test D — Verifier null-space contamination

- **Required data:** exp1118 eval JSON with per-candidate `verifier_scores` (6 floats).
- **Quantity:** Per question pair, ΔV = ‖V_correct - V_incorrect‖₂. Bottom 20% (low ΔV = ensemble blind spot) vs top 20% (high ΔV = ensemble distinguishes). Compute mean ΔE per subset.
- **Threshold (supports D):** ΔE << 0 in low-ΔV subset BUT ΔE > 0 in high-ΔV subset. Inversion is *strictly localized* to ensemble blind spots.
- **Threshold (rejects D):** Inversion uniform across ΔV deciles.
- **Confidence caveat:** Verifier scores must be CONTINUOUS pre-sigmoid logits; hard 0/1 binaries create artificial ΔV=0 clusters that misrepresent the gradient landscape.

---

## Decision tree

```
Run all 4 tests concurrently on existing artifacts.

A supports + B/C/D reject  →  CORPUS — proceed exp1121 production wiring,
                              exp1120 will resolve the inversion.

B supports                 →  LOSS GEOMETRY — CANCEL exp1121,
                              .88 needs EBM-head loss redesign task.

C supports                 →  LIPSCHITZ — exp1120 caps near zero,
                              .88 needs spectral-norm ablation.

D supports                 →  NULL SPACE — exp1120 caps near zero,
                              .88 needs null-space measurement on
                              the new k=5 ensemble (exp1121 itself).
```

## Partial-resolution interpretation (if exp1120 takes r from -0.13 → +0.05)

This *definitively rejects B*, confirms A as necessary, suggests **A + C** or **A + D**.

- **Why rejects A+B:** if loss geometry structurally forced inversion, broader SOTA corpus would just teach the inverted mapping more accurately on new data. r crossing zero proves the geometry is *capable* of correct monotonic ordering.
- **Why suggests A + (C or D):** corpus shift removed the primary inversion driver (confirming A), but only reaching +0.05 means the network structurally chokes the signal — either Lipschitz capacity-limited (C) or null-space-trapped (D).
- **How to distinguish C vs D without retraining:** rerun Test C and Test D on the freshly-generated exp1120 artifacts. R²_length dominance → C. Inversion-localized-to-low-ΔV → D.

---

## Synthesis (Carnot side)

### Drift check (per `feedback_carnot_prediction_pattern.md`)

- ✅ All four tests are **methodology**, no parameter prescriptions.
- ✅ Thresholds are **directional** (`> 0`, `>> 0`, `< 0`) not specific decimal values, except:
  - Test C uses `R² > 0.5` — borderline numerical, but reasonable as a "majority of variance" threshold (standard ML interpretation).
  - Test D uses bottom/top 20th percentile — a percentile choice, not a magic number.
- ✅ Each test has an **explicit confidence caveat** about when it'd be less reliable.
- ✅ Decision tree is unambiguous and traces to specific .87/.88 actions.

### Risks Deep Think flagged that we should respect

1. **Test C's omitted-variable risk** — length is one proxy; if Lipschitz smoothed onto a different low-frequency feature (step count, code-block presence), Test C will falsely reject. Mitigation: when running Test C, include 2-3 additional length-class proxies (token count, line count, step count) to make the test robust.
2. **Test D's binary-vs-continuous risk** — exp1118's `verifier_scores` field needs to be continuous logits, not 0/1 binaries. Mitigation: verify the schema before running Test D; if binary, re-extract logits from the model checkpoint.
3. **Test B's reconstruction risk** — if the training loss has batch-level contrastive normalization, per-pair reconstruction is approximate. Mitigation: read the exact loss formula from `scripts/experiment_1099_*.py` / `scripts/experiment_1110_*.py` first, decide if approximation is acceptable.

### Cross-validation of the partial-resolution claim

Deep Think's logic that "r crossing zero rejects B" is **load-bearing for our .87/.88 decision**. Reading the argument carefully:

> If the loss geometry STRUCTURALLY forced incorrect answers to the wrong side of the margin, retraining on a broader corpus would learn that inverted mapping more accurately on new data, leaving correlation strongly negative.

This is sound: a structurally-broken loss would exhibit *consistent* inversion regardless of corpus. A corpus-fixable inversion shows direction-flipping when corpus changes. Agreed; this is well-calibrated qualitative reasoning, not a parameter prescription.

---

## Recommended action

Execute Tests A, B, C, D on existing artifacts immediately. The four tests are cheap (~10 min each) and decide:

- **Whether exp1120 should be allowed to complete** (if Test A supports + others reject, yes; if Test B supports, abort exp1120 and pivot)
- **Whether exp1121 production wiring is the right next task** (only if Test A is the sole supported hypothesis)
- **What .88's first task should be** (depends on which of B/C/D supports, if any)

The conductor is currently running exp1120. We have a window of perhaps 20-40 min before exp1120 produces its verdict and exp1121 starts. **Now is the optimal time to run these diagnostics**, before the conductor commits a milestone direction based on exp1120 alone.
