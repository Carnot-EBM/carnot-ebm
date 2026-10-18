# Topological-architecture follow-ups: depth-recurrent probe + external-scratchpad repair

**Status:** Draft change proposal.
**Origin:** User question 2026-04-26 about arXiv 2604.17121 ("The Topological
  Trouble With Transformers", Mozer / Siddiqui / Liu). The paper argues that
  pure-feedforward transformers cannot reliably track dynamic state because
  each new input pushes the evolving latent representation deeper into the
  layer stack. Two of Carnot's recent failure patterns are direct
  consequences of this:
    - DRIFTProbe Tier 0i has now failed three times across .69/.70/.72
      (`drift_probe_not_viable` → `tier0i_marginal` → `tier0i_no_improvement`).
      All three attempts read state from a *single* hidden layer.
    - MathIterativeSelfRepair (Exp 930) landed +0 pp on GSM8K despite code
      repair (Exps 905/906) landing +68 / +72 pp on HumanEval. The code-repair
      win came from execution traceback as an *external* state-feedback
      signal; math repair has no equivalent external signal.
**Target milestone:** 2026.04.73 — earliest practical milestone after the
  paper has been added to `research-references.md` (this proposal also
  appends it).
**Priority:** Medium-high. Both experiment lines have well-funded prior
  attempts; restructuring them along the paper's prescription is cheaper
  than a fresh research direction and the predictive power of the paper's
  framing makes the bet reasonable.
**Depends on:** nothing — both experiments reuse existing primitives
  (probe trainer, IterativeSelfRepair pipeline) with a clear delta.

## Summary

The Topological Trouble paper's prescription is "shift to recurrent /
continuous-thought architectures with implicit activation dynamics."
Two Carnot experiment lines have been blocked on what now reads as the
exact failure mode the paper predicts. Carnot doesn't need a new
foundation model to test the prescription — it needs two scoped
experiments that change the *interface* between existing components.

1. **DRIFTProbe v3 — depth-recurrent.** Replace the single-layer linear
   probe with a probe that pools or attends across *all* layers of the
   target model. The state is somewhere in the stack; we just don't
   know which layer. Pool/attend across the depth dimension and let
   the probe learn where to look.

2. **MathIterativeSelfRepair v2 — external scratchpad.** The successful
   code-repair pipeline reinjects execution tracebacks as input text
   (external feedback). Math repair currently reinjects the prior
   attempt's *latent state* (recurrent feedback through the same model).
   The paper says the latent path can't work; switch math repair to
   the same external-text path that code repair uses. The "scratchpad"
   is just the prior attempt's natural-language reasoning + the
   verifier's signed-diff output, fed as text into the next attempt.

## What this proposal IS NOT

- **Not a new architecture.** Both experiments reuse the existing
  TransformerEncoder for probe-target activations and the existing
  Qwen3.6-35B / Gemma4 GPU stack for repair. The delta is the
  *interface* between layers, not new neurons.
- **Not a refutation of single-layer probes for all tasks.** The
  paper's claim is specifically about *dynamic state tracking*. A
  static feature like "is this token a number?" can absolutely be
  recovered from a single layer. The probe restructure applies only
  to probes that track *evolving* state (hallucination drift,
  reasoning trajectory, etc.).
- **Not a rejection of internal recurrence.** Eventually Carnot wants
  to ship a recurrent EBT/EBM foundation model (Phase 3). This proposal
  is the cheapest near-term test of the recurrence hypothesis using
  existing models — it does not preclude or replace Phase 3.

## Proposed experiments

### Exp A — DRIFTProbe v3 (depth-recurrent)

**Deliverable:**
`python/carnot/verify/drift_probe_v3.py` (new module) +
`scripts/experiment_<N>_drift_probe_v3.py` +
`tests/python/test_drift_probe_v3.py` +
`results/experiment_<N>_drift_probe_v3.json`.

**What it does:**

Replace the single-layer linear probe with one of two
depth-recurrent designs (the experiment runs both for comparison):

1. **AttentionPool over layers.** Stack the target model's hidden
   states from all layers `[h_0, h_1, ..., h_L]` into a `(L+1, d)`
   tensor; learn a query vector `q` of dimension `d`; produce
   probe_input = softmax(`(h_l · q)_l`) · `h_l` summed over `l`.
   Single attention head, scalar output via a small MLP head.
2. **GRU over layers.** Treat the layer dimension as a sequence
   index and run a 1-layer GRU over `(L+1, d)`. Final hidden state
   feeds the scalar output head.

**Hypothesis:** if Mozer's framing is right, depth-pooled state
captures the hallucination signal that single-layer probes miss.
The .69/.70/.72 single-layer baseline gives us a clean
counterfactual.

**Acceptance:**
  - AUC on the same hallucination-detection eval used in Exps
    899/911/923 — must clear *both* the .69 baseline and the .72
    `tier0i_no_improvement` ceiling by ≥ 0.05 AUC for the technique
    to count as a pass.
  - If both depth-recurrent designs land below the ceiling, write
    `drift_topologically_constrained` verdict and retire DRIFT from
    the FR-11 cascade plan permanently — three independent attempts
    with the architecturally-aligned fix is enough.

**`prior_failures`** (mandatory per the discipline):

```yaml
prior_failures:
  - experiment_id: exp899-drift-hidden-state-probe
    verdict: drift_probe_not_viable
    addressed_by: >
      Single-layer linear probe could not track dynamic state. Per
      arXiv 2604.17121, transformers push evolving state deeper into
      the layer stack — single-layer probes miss it by construction.
      v3 pools/attends across all layers.
    retire_if_same_verdict: true
  - experiment_id: exp911-drift-probe-tier0i
    verdict: tier0i_marginal
    addressed_by: >
      Same single-layer architecture, different layers tried. Same
      topological constraint applies. v3 changes the architecture.
    retire_if_same_verdict: true
  - experiment_id: exp923-drift-probe-ensemble
    verdict: tier0i_no_improvement
    addressed_by: >
      Multi-layer ensemble was a step in the right direction but
      still treated each layer as independent. v3 uses a learned
      pooling/recurrence over the layer dimension.
    retire_if_same_verdict: true
```

### Exp B — MathIterativeSelfRepair v2 (external scratchpad)

**Deliverable:**
edits to `python/carnot/pipeline/iterative_self_repair.py` to support
a `feedback_mode: external_text | latent_recurrence` flag +
`scripts/experiment_<N>_math_repair_v2_scratchpad.py` +
`tests/python/test_math_repair_v2_scratchpad.py` +
`results/experiment_<N>_math_repair_v2_scratchpad.json`.

**What it does:**

The existing pipeline runs:

```
attempt_n_output = LLM(prompt + history_n_minus_1_latent_state)
```

v2 runs:

```
scratchpad = render_text(prior_attempt.cot, prior_attempt.error_diff)
attempt_n_output = LLM(prompt + scratchpad)
```

`render_text` formats the prior attempt's CoT and the verifier's
signed-diff output as *natural-language text* re-fed into the next
prompt. No latent recurrence — same channel as the successful code
repair pipeline.

**Hypothesis:** if the math-repair zero is topological (no external
state feedback channel), restoring an external text channel should
recover non-zero improvement. We're not predicting +72 pp parity
with code — math doesn't have execution traceback's binary fail
signal — but anything ≥ +5 pp on GSM8K-25 is a meaningful win and
proves the topological framing applies.

**Acceptance:**
  - signed_improvement on GSM8K-25 ≥ +0.05.
  - Live GPU run only — no E4B fallback (the .70 / .72 ambiguity
    about model identity is the second-largest source of noise in
    the comparison; pin the model in the prompt and verify in
    artifact).
  - If signed_improvement < 0.05, write `math_external_scratchpad_no_improvement`
    and re-evaluate whether GSM8K is the right benchmark
    (single-step problems may benefit less than multi-step).

**`prior_failures`:**

```yaml
prior_failures:
  - experiment_id: exp930-math-iterative-self-repair
    verdict: math_repair_zero
    addressed_by: >
      v1 reinjected prior attempt as latent state through the same
      forward pass. Per arXiv 2604.17121, the prior-attempt state
      gets pushed deep into the layer stack and is unreachable by
      the next attempt. v2 reinjects as natural-language text in
      the prompt, giving the next attempt explicit access — the
      same external-channel pattern that made code repair (Exp 906
      +72 pp HumanEval) work.
    retire_if_same_verdict: true
```

## Decentralization implications

- **Rule 1 (local-first):** both experiments use existing local
  models (Qwen3.6-35B-A3B-GGUF, target-model activations from any
  open-weight LLM in the stack). No closed-weight dependency.
- **Rule 7 (no vendor abstractions):** both deliverables live in
  existing modules (`python/carnot/verify/` and
  `python/carnot/pipeline/`). No new vendor-specific code paths.

## Why this is in change-proposals, not just two experiment YAMLs

- DRIFT has failed three times. Adding it back to the cascade
  without a written-down plan is exactly the kind of doomed-rerun
  pattern the failure-ledger discipline is designed to prevent.
  The proposal is the locus where the rationale lives.
- MathIterativeSelfRepair's failure was previously read as
  "technique doesn't generalize to math." The paper changes that
  reading — the technique didn't generalize because the
  *implementation* used the wrong feedback channel. Future
  Carnot-Claude needs the proposal to find this rationale, not
  just a YAML diff.

## Risks

- **The paper's framing is wrong.** The taxonomy in arXiv 2604.17121
  is theoretical with no benchmark numbers. If the topological
  argument doesn't actually predict the failure mode, both
  experiments will reproduce their priors. Mitigation: the
  retire-on-fail flag is set on both. We don't get to keep
  iterating on the framing if it doesn't predict.
- **Depth-recurrent probe is more expensive than linear probe.**
  The AttentionPool/GRU adds a few thousand parameters and a small
  forward pass. Negligible.
- **External scratchpad inflates prompt length.** Each retry
  appends prior CoT + error diff. After 3 retries the prompt could
  be 2-3× the original length. Most current local LLMs handle 8 K
  tokens fine; the experiment caps at 3 retries explicitly.

## Acceptance criteria (overall)

1. arXiv 2604.17121 is added to `research-references.md` as a
   first-class entry in the .72 arxiv scan section (with the
   relevance-to-Carnot bullets explicit).
2. `python/carnot/verify/drift_probe_v3.py` ships with both
   AttentionPool and GRU variants; either one beating the .72
   ceiling by ≥ 0.05 AUC counts as a pass.
3. `iterative_self_repair.py` grows a `feedback_mode` flag without
   breaking the existing `latent_recurrence` path used by code repair.
4. Both experiments carry the four-part `prior_failures:` discipline
   and respect the failure-ledger pre-launch check.
5. If both experiments land below their thresholds, DRIFT and the
   external-scratchpad approach to math repair are retired
   permanently. We earn the right to keep iterating only by clearing
   the topological-framing prediction.
