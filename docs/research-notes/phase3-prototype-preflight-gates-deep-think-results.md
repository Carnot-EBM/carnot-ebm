# Deep Think Response — Phase-3 Prototype Pre-Flight Adversarial Gates

**Status:** Response received 2026-05-01. 7 hostile-reviewer attacks
specified, each with named failure mode, architectural-assumption
trace, diagnostic formula (precise enough to implement), abort
threshold (direction + order of magnitude), and confidence-calibration
caveat. Decision tree distinguishes 0/1/2+ attacks fired plus a fatal-
combination escalation. Strongest adversarial-design response of the
project to date.
**Date received:** 2026-05-01
**Source prompt:** `phase3-prototype-preflight-gates-deep-think-prompt.md`

---

## Summary table

| # | Attack | Failure mode | Diagnostic | Abort threshold |
|---|---|---|---|---|
| **1** | **Posterior Bypass (LM-Prior Masking)** | Decoder ignores latent bottleneck, uses LM prior alone | RLIG = CE(x|dec(z_rand)) − CE(x|dec(z_true)) | RLIG drops below ~10⁻¹ nats/token |
| **2** | **Deadband Evasion (STE Gaming)** | Encoder collapses |z| → 0 to escape sgn() boundary | M_dead = mean(I(|z| < 10⁻³)) | M_dead exceeds ~10⁻¹ (>10%) |
| **3** | **Dynamic Verifier Joint-Null Exploitation** | GRPO finds adversarial text that fools multiple verifiers simultaneously | r_live_max = max pairwise Pearson on live GRPO outputs | r_live_max > 0.5 over 100-step window |
| **4** | **GRPO Advantage Homogenization (Energy Flatlining)** | Intra-group energy variance collapses → noise-amplified advantages | σ_E = mean_prompts(std_dev_K(EBM(sgn(z_k)))) | σ_E drops below ~10⁻⁴ |
| **5** | **Dynamic Hopf Limit-Cycle Trapping** | Cross-Hessian spectral radius grows during training, breaching DDE bound | C_grad = cos(g_t, g_{t−1}) on raw gradients | C_grad < ~−10⁻¹ persistent |
| **6** | **SP-IWPER Priority Collapse** | Replay sampler skews to handful of "jackpot" transitions | H_w = −Σwᵢlog(wᵢ)/log(B) | H_w drops below ~10⁻¹ |
| **7** | **Ising Spin Mode Collapse** | All inputs map to single sgn(z) state | H_dist = mean_pairs(Hamming(sgn(zᵢ), sgn(zⱼ)))/d | H_dist drops below ~10⁻² |

## Decision tree (verbatim)

- **0 attacks fire:** continue scaling to .89+
- **1 attack fires:** PAUSE. Treat as substrate hyperparameter mismatch (GRPO temp, regularization, PER α). Tune relevant parameter, retry once. If violation persists for 1000 steps, abort.
- **2+ attacks fire:** ABORT immediately. Multiple coupled failures empirically prove an architectural pillar buckled. Return to Phase-2 / Phase-3 Deep Think.
- **Attack 1 + Attack 2 OR Attack 7 fires** (FATAL combination): the continuous-to-discrete substrate is structurally dead. DBAE requires fundamental redesign (e.g., explicit VQ-routing). Do not attempt hyperparameter fixes.

## Drift check (per `feedback_carnot_prediction_pattern.md`)

- ✅ **No parameter prescriptions** at the architecture level.
- ✅ Thresholds are **direction + order-of-magnitude** ("drops below ~10⁻¹"), not specific decimal values.
- ✅ Each attack traces to a **specific architectural assumption** quoted from the synthesis.
- ✅ Each diagnostic has **explicit confidence calibration** naming what would make it less reliable.
- ✅ FATAL-combination clause is honest about when hyperparameter tuning won't save the prototype.

## Critical operational notes

- **Attack 4 conflict with optimizer momentum:** Attack 5's diagnostic
  (gradient cosine auto-correlation) **MUST be computed on raw
  pre-momentum stochastic gradients**. Adam β1 ≥ 0.9 will mask the
  limit cycle. Implementation needs a hook before the optimizer step.
- **Attack 3's pass-rate floor:** correlation matrices are unstable
  when global pass rate is <5%. The diagnostic should be gated until
  the policy is generating non-gibberish (warm-up window TBD
  empirically).
- **Attack 6's α dependency:** if PER importance exponent is set near
  0, uniform sampling is forced and H_w sits artificially near 1.0.
  The threshold is meaningful only if α > 0 is actually configured.

## Operational implications for .88 prototype kickoff

The 7 attacks define a **mandatory diagnostic library** that the
.88 prototype must instrument FROM STEP 1, not retrofit later. This
matches the CLAUDE.md mandate that "empirical instrumentation IS
adversarial check at scale."

Implementation work for .88 (filed as candidate task):
- `python/carnot/diagnostics/phase3_attack_probes.py` — module
  computing all 7 diagnostics per training step.
- Test fixtures simulating each failure mode (so the diagnostics
  are themselves tested before training starts).
- Logging schema: per-step JSONL with all 7 quantities + metadata
  (step number, batch size, optimizer state, learning rates).
- Abort hook: when any diagnostic crosses its threshold, write
  `abort_artifact.json` and gracefully shut down training.

## Cross-validation status

This is the second Deep Think response of the day staying entirely
in the methodology/diagnostic-design lane. Combined with Q1's
energy-inversion response, the pattern is: **questions framed as
"what to measure" get well-calibrated answers; questions framed as
"what value to use" get systematically-wrong prescriptions**.

The .88 prototype kickoff now has a clear pre-flight checklist.
Blockers remaining:
1. Resolve Q1 (energy inversion) before .88 to know if the EBM
   substrate is sound.
2. Resolve Q3 (GRPO + SP-IWPER composition) before .88 to know if
   the Stage 3+4 joint design works at all.
3. Implement the 7-attack diagnostic library as a pre-task in .88
   itself.
