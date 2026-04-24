# Generative-time safety gate — public-facing product surface (FOLLOWS dogfood deployment)

**Status:** Draft change proposal.
**Origin:** 2026-04-24 design discussion.
**Target milestone:** 2026.04.62 or later.
**Priority:** **SECOND** — this is the public product follow-on to the internal
  dogfood deployment; see [`conductor-self-protection-safeguard.md`](./conductor-self-protection-safeguard.md)
  first.
**Depends on:** the same classifiers the dogfood proposal deploys internally.
  Calibration thresholds, false-positive audit data, and incident samples
  accumulate from internal use before product-surface exposure.
**Traces to:** PRD Phase 1 safety surface, `openspec/capabilities/safety/` (if
  it exists), and the research-references entry for UnsafeBench
  distribution-shift discipline.

## Why this exists and why it's not the first priority

The safety-classifier line (prompt-injection KAN v3, privacy-filter v2, jailbreak
KAN v1) has reached the point where every gate is passing in offline evaluation:

- Prompt-Injection KAN v3: AUROC 0.9078 (Exp 724) — first to clear the 0.90
  publication gate.
- Privacy-Filter KAN v2: AUROC 1.0 on 2/3 holdout datasets (Exp 743).
- Jailbreak Detection KAN v1: AUROC 1.0 on its evaluation split (Exp 775).

All three classifiers are **post-hoc**: they take a completed LLM output and score
it. A post-hoc classifier can only *reject* an output after the model has already
generated it, which is useful for filtering but does not prevent the unsafe text
from existing in the stream — and in streaming deployments, it is already visible
to the end user by the time the classifier fires. This is the gap that separates
"audit tool" from "safety gate".

**Why this proposal is the SECOND priority, not the first.** The same three
classifiers are more urgently needed as internal dogfood tooling to protect the
autoresearch conductor itself from MCP / arXiv / GitHub prompt-injection. The
companion proposal `conductor-self-protection-safeguard.md` deploys them at the
conductor's most vulnerable call sites: inbound MCP responses, outbound LLM
prompts, pre-execution script guards. That work is the correct first deployment
for four reasons:

1. **Adversarial surface is already live.** Gerolamo MCP (23 tools, network-
   fetched intelligence corpus) was integrated in this session. The conductor
   reads open-web content daily. The attack surface is not hypothetical.
2. **Real adversarial validation data.** Every conductor run is a natural
   adversarial experiment. Within a month of dogfood deployment we will have
   more real incident samples than any static benchmark produces.
3. **Threshold calibration is cheaper inside.** We control both ends of the
   conductor's loop and can iterate false-positive bounds without affecting
   external users.
4. **Failure modes surface earlier.** If the v3 KAN has a blind spot, we find
   it in internal logs before a customer does.

This proposal, the public-facing generative-time gate, should follow dogfood
deployment. When it lands, it builds on the same classifier weights but with
calibration thresholds informed by real incident logs from internal use.

This change proposes the cheapest viable step toward in-loop intervention: wire
the existing v3 Prompt-Injection KAN into the existing HERMES v2 sentence-boundary
monitor and let the generation loop abort early when the rolling energy exceeds a
calibrated threshold. No new classifier architecture, no new training, no new
dependencies.

## What already exists (nothing new to write)

Four pieces of infrastructure have been accumulating for exactly this purpose:

- **v3 Prompt-Injection KAN** (`python/carnot/models/prompt_injection_kan.py`,
  Exp 724). 5,016 parameters, 19.7 ms median inference on CPU, 0.9078 AUROC.
- **HERMES v2 sentence-by-sentence generation loop** (Exp 641). Sentence-level
  hook point in `python/carnot/pipeline/hermes_v2.py` that fires between token
  emissions; was originally used for a constraint-violation check that had recall
  0.0 and was retired. The plumbing is the valuable artifact.
- **InterWhen mid-generation monitor pattern** (Exp 627, arXiv 2602.11202). The
  "monitor runs at sentence boundaries, emits `verdict=violation` or
  `verdict=safe`, and the generator honours it" pattern is already codified.
- **JEPA-Reasoner Probe** (Exp 726, OOD AUC 1.0, 0.02 ms latency). Proof-of-
  concept that a classifier running on the model's *hidden state* (not on
  emitted tokens) can be fast enough for token-level decisions.

The proposal is to compose these: run the v3 KAN at HERMES v2 sentence boundaries
and use a hard stop or pivot-prompt on trigger.

## Proposed experiments

Pick the first one as an MVP. Experiments 2-4 are sequential follow-ups.

### Experiment A — MVP sentence-boundary safety gate (one week, existing hardware)

**Deliverable:** `results/experiment_<N>_generative_time_kan_gate.json` plus
`python/carnot/pipeline/safety_gate.py`.

**What it does:** Wraps the existing `VerifyRepairPipeline` with a
`SafetyGateMonitor` that fires after each sentence boundary. The monitor runs
the v3 Prompt-Injection KAN on a rolling prefix (accumulated text so far). When
rolling energy exceeds `gate_threshold` (initial value: 0.85, calibrated on a
held-out set so benign false-cutoff rate is ≤ 2%), the generation stream ends
with `finish_reason='safety_gate'` and an explicit `violation_reason` field
recording which classifier fired and at which sentence index.

**Acceptance gates (all three required for ship):**

1. On a held-out HackAPrompt live-generation set of 50 prompts known to trigger
   injection in an ungated run, the gate fires on ≥ 85% (detection rate).
2. On a benign GSM8K reference set of 100 prompts, the gate fires on ≤ 2%
   (false-cutoff rate).
3. Per-sentence overhead is ≤ 25 ms at p95 on the RTX 3090 pair. If this
   fails the hidden-state variant in Experiment C becomes the primary path.

**Honest-verdict enum:** `gate_shipped`, `detection_below_threshold`,
`false_cutoff_above_budget`, `latency_above_budget`, `blocked_no_live_gpu`.

**Why it's cheap:** the KAN exists, HERMES v2 exists, the held-out sets exist.
The work is plumbing and threshold calibration.

### Experiment B — Pivot-prompt instead of hard stop

**Deliverable:** `results/experiment_<N+1>_generative_time_pivot_prompt.json`.

**What it does:** Extends Experiment A. On gate trigger, instead of `finish_reason
='safety_gate'`, inject a `<safety_redirect>` system message and let the LLM
regenerate from the sentence that triggered. The redirect text is a simple
template: "The previous request triggered Carnot's safety gate (reason:
`{violation_reason}`). Respond with a refusal that explains what was blocked
and why, without reproducing the unsafe content." This recovers utility on
borderline prompts where a user's intent is legitimate but phrased ambiguously.

**Acceptance gates:**

1. On the same held-out HackAPrompt set, the pivoted response never reproduces
   the originally requested unsafe content (graded by LLM-as-judge over the
   200-prompt set, judge agreement ≥ 0.9).
2. On a mixed-ambiguity validation set (50 benign-but-borderline prompts), the
   pivoted response produces a coherent refusal with a reason at rate ≥ 80%.

**Honest-verdict enum:** `pivot_ships`, `pivot_leaks_unsafe_content`,
`pivot_refusal_incoherent`, `blocked_no_live_gpu`.

### Experiment C — Hidden-state classifier variant (speed optimisation)

**Deliverable:** `results/experiment_<N+2>_hidden_state_safety_gate.json`.

**What it does:** If Experiment A's per-sentence overhead is uncomfortable
(≥ 15 ms), swap the v3 KAN (which runs on emitted text) for a small classifier
that runs on the LLM's hidden state at sentence boundaries, modelled on the
JEPA-Reasoner Probe pattern (Exp 726). The probe trains on
`(hidden_state_at_sentence_boundary, safety_label)` pairs. Target: 0.02 ms
per fire, same AUROC bar as v3 KAN (0.90).

**Acceptance gates:**

1. Probe AUROC on a held-out set ≥ 0.90 (matches v3 KAN's bar).
2. Per-sentence overhead ≤ 2 ms at p95 (100x faster than Experiment A).
3. KL divergence vs the v3 KAN on a common eval set ≤ 0.05 (the two should
   agree on what is unsafe even though they look at different signals).

### Experiment D — Energy-guided decoding (COLD, arXiv 2602.11141)

**Deliverable:** `results/experiment_<N+3>_energy_guided_decode.json`.

**What it does:** Instead of stopping or pivoting on trigger, use the
classifier's gradient to bias next-token logits. The generation produces
safer sequences directly rather than rejecting unsafe ones. This is a bigger
rewrite — it interacts with the sampler's beam/top-k path — and the trade-off
is that gradient-guided sampling can reduce fluency when the bias is too
strong.

**Acceptance gates:**

1. On HackAPrompt, unsafe-output rate drops below the Experiment A gate
   without stream interruption.
2. Fluency on benign GSM8K does not regress by > 1 pp vs the ungated baseline
   (graded by perplexity on outputs, not by LLM judge).
3. The bias coefficient is tunable post-hoc so operators can move the
   fluency / safety trade-off without retraining.

**Honest-verdict enum:** `cold_ships`, `cold_fluency_regression`,
`cold_safety_regression`, `cold_gradient_unstable`, `blocked_no_live_gpu`.

## Why stage it this way

Experiments A and B ship utility. Experiment C is optimization, worth doing
only if A's latency proves uncomfortable in production. Experiment D is the
architecturally correct answer but is a bigger lift with a harder evaluation
story (fluency regressions are subjective). Each experiment has an honest-
verdict path that says "didn't pass, here's why" rather than a silent pass-
through. Each fits the existing conductor pattern and requires no new
dependencies.

**Important staging dependency:** Experiments A-D in this proposal assume the
classifier thresholds have been calibrated against real incident data from the
dogfood deployment (see `conductor-self-protection-safeguard.md` Experiments A/B).
Running this proposal's Experiment A without that calibration risks shipping a
gate whose 0.85 threshold is either too aggressive (user-visible false-cutoffs)
or too permissive (lets obvious injections through). Schedule this proposal only
after the dogfood deployment has accumulated ≥ 2 milestones of incident logs.

## Integration into the existing cascade

The safety gate belongs at the front of the cascade, before any Phase 1
verification work — both because it's cheap, and because it's the only way
to prevent unsafe text from reaching Phase 1's verify-repair loop (which
would otherwise produce an embarrassing "I verified the hate speech is
logically consistent" result).

New cascade ordering after this change:

```
input → [safety_gate]          ← Experiments A-D
      → [Phase 1 tier 0 pre-filter: KAN Tier 0b]
      → [Phase 1 tier 1 constraint extractor]
      → [Phase 1 tier 2 JEPA-Reasoner Probe]
      → [Phase 1 tier 3 Ising formal verifier]
      → output
```

## Risks we should record honestly

- **Threshold calibration will drift.** The 0.85 gate threshold is calibrated
  on a fixed held-out set; as attack distributions evolve, the threshold
  needs re-calibration. Mitigation: add a monthly re-calibration experiment
  to the conductor's schedule.
- **False cutoff on legitimate users.** 2% false-cutoff sounds acceptable
  until someone's perfectly reasonable prompt ends up in that 2%.
  Mitigation: log every cutoff with the triggering sentence and the KAN
  energy at fire time; audit quarterly.
- **Adversarial users who know about the gate.** Once this ships, attackers
  will craft prompts that stay below the energy threshold while still
  eliciting unsafe content. This is why the Garak integration (separate
  change proposal) is a near-term follow-up, not a long-term plan.
- **Classifier-generator coupling.** If the v3 KAN has a blind spot, the
  gate shares it. Mitigation: Experiment C's hidden-state classifier
  provides a second independent signal; both firing is a much stronger
  safety claim than either alone.
