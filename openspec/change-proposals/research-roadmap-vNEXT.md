# Research Roadmap — Milestone 2026.06.363

**Status:** Pre-staged by the outer-loop planner (Claude Opus 4.8), 2026-06-07.
**Supersedes:** 2026.06.362 (the offline-verifier-proof milestone).
**One-line:** `.362` finally LANDED the offline verifier proof — but its
EFFICIENCY axis was measured against a **below-chance LLM-judge strawman** and
its cascade **never escalated**. `.363` RE-PROVES efficiency against a
**competent judge**, makes the cascade non-degenerate, **replicates the moat on
a second corpus**, then takes the **first realistic agentic step** toward the
ARC-AGI-3 venue (north-star §5, sequenced second). Facts retires; the verifier
stays math-domain-bound; `paper_ready` stays TRUE; frozen 0.9131 unchanged.

---

## 1. What the previous milestone (.362) proved — and what it got wrong

`.362` recovered the `.361` infra wash (a poison-test cascade + a recurring
`blocked_llama_cpp_inference_failed`) by shipping a **robust, tested
`gguf_inference` harness** (exp3915) with a smoke-before-use + GGUF fallback
chain, then ran the long-blocked offline verifier proof. Read honestly via
`scripts/summarize_artifact.py`:

| Axis | Artifact | Verdict | Honest read |
|---|---|---|---|
| **ACCURACY (moat scissor)** | exp3916 | **MOAT_SURVIVES** | **GENUINE.** Energy ensemble AUROC **0.967** catches errors the reasoner self-verify (0.44 weak / **0.66** strong) misses. Residual-catch **0.914** (CI95 [0.843,0.971]), overlap **0.50**, n_res=70. Corroborates the Self-Correction Illusion (arXiv:2606.05976). |
| **EFFICIENCY (head-to-head)** | exp3917 | CHEAPER_NOT_PARITY @ 11512× | **INVALID COMPARATOR.** LLM-judge (gemma-4-26B 0-shot) AUROC **0.4423 = below chance**. Energy (0.81) *beats* the judge; "not parity" is a strawman artifact. 11512× wall-clock ratio is real (0.066 ms vs 763 ms) but worthless against a broken baseline. |
| **CASCADE (router)** | exp3918 | WINS @ 11512× | **DEGENERATE.** `escalation_fraction = 0.0` — never escalated; the cascade mechanism was never exercised. `auroc_gap = -0.39` only because the LLM baseline is sub-chance. |
| **FACTS (graph-grounding)** | exp3920 | FLAGGED (blocked, model not invoked) | **RETIRES.** Fourth facts fabrication/block (exp3862/3886/3896/3920). `retire_if_same_verdict` fired. |
| **ARC scaffold** | exp3919 | READY | Verifier-first harness skeleton + passing unit test on a synthetic env. |
| **FR-11 v25** | exp3921 | INVARIANT_HELD | Self-learning mandate held; AUROC in frozen CI, +0.0185 memory contribution preserved. |
| **Hardware** | exp3922 | FLAGGED (duration=0 tautology) | Needs one clean continuity re-run. |
| **Capstone** | exp3923 | paper_ready=TRUE, frozen 0.9131 | But reported the efficiency/cascade flaws at face value. |

**The load-bearing correction:** the capstone set `verifier_earns_its_place =
false` because it gated on the *pure* energy verifier reaching *parity* with a
judge that was below chance. That gate is meaningless against a broken
comparator. The verifier's accuracy value (MOAT_SURVIVES) is proven; its
**efficiency value is UNPROVEN** until measured against a competent judge. This
is exactly the FALSE_NEGATIVE_RISK / positive-control discipline (CLAUDE.md):
a comparison against a baseline that failed its own positive control is not
evidence.

## 2. The three biggest gaps (current state vs north-star §5 / PRD)

1. **The efficiency proof rests on a sub-chance comparator.** North-star §5's
   win condition is "equally effective as the LM at lower cost/latency
   (efficiency-parity), OR Pareto-dominate (cheaper at equal accuracy AND/OR
   more accurate at equal cost)." We cannot claim either against a 0.44-AUROC
   judge. **Gap: a COMPETENT LLM-judge comparator + a valid head-to-head + a
   non-degenerate cascade.**

2. **The moat is single-corpus.** MOAT_SURVIVES was measured only on the
   exp3884 in-distribution corpus. A second corpus (FoVer slice) is needed to
   show it is not a corpus artifact. **Gap: moat replication.**

3. **The agentic-proof venue has a scaffold but zero science.** North-star §5
   sequences the ARC-AGI-3 harness second, after the offline proof — which
   `.363` Phase 1 completes. The real ARC-AGI-3 benchmark scores frontier models
   <1% and scores by **action-efficiency**, explicitly prioritizing
   **domains with verifiers** (arXiv:2603.24621) — the perfect venue for the
   verifier-as-action-pruner. **Gap: the first realistic agentic run measuring
   action-efficiency with vs without the verifier.**

## 3. Milestone design — 10 experiments across 5 phases

```
PHASE 0 — hygiene + retire facts + record the comparator flaw
  exp3924  archive .362 / activate .363; RETIRE the facts route to future-work
           (exclusion manifest); quarantine any poison test; green-gate; record
           the below-chance-comparator finding as the .363 forward-bet.

PHASE 1 — HARDEN THE OFFLINE VERIFIER PROOF (the real .363 science)
  exp3925  DIAGNOSE + FIX the below-chance LLM-judge (polarity/parse bug vs
           genuine weakness); ship a COMPETENT judge config (AUROC >> chance,
           validated on a held-out check) backed by the robust harness.
  exp3926  VALID efficiency head-to-head with the competent judge + a defensible
           cost methodology (per-item wall-clock AND FLOP/token, amortized-load
           excluded). The real efficiency verdict (parity / Pareto / neither).
  exp3927  NON-DEGENERATE cascade router with the competent judge — escalation
           fraction MUST be > 0 (positive control); real matched-accuracy cost.
  exp3928  MOAT SCISSOR replication on a SECOND corpus (FoVer slice), multi-seed,
           strong self-verify arm — confirm MOAT_SURVIVES is not a corpus artifact.

PHASE 2 — AGENTIC PROOF VENUE first realistic run (north-star §5, second)
  exp3929  ARC-AGI-3 harness: first realistic verifier-as-router agentic run on
           a non-toy env — actions-to-solve WITH vs WITHOUT the verifier-pruner
           (the Exp1165 ~4x shape), + a real-benchmark access preflight.

PHASE 3 — standing mandates + hardware + literature
  exp3930  FR-11 v26 (self-learning MANDATE) — online learning of the cascade
           escalation band (Tier-1 self-learning tied to the deployable artifact).
  exp3931  Hardware continuity (CLEAN re-run; exp3922 was flagged) — KV260 + PolarFire
           + GateMate, distinct timers, no-fabric-claim.
  exp3932  Literature synthesis — agentic verification + LLM-judge calibration +
           cascade efficiency 2026; codex synthesis of the staged references.

PHASE 4 — capstone
  exp3933  Capstone .363 — the HARDENED verifier scorecard: efficiency now valid,
           cascade non-degenerate, moat replicated, agentic first-run. Answer
           "does the verifier earn its place" against a COMPETENT judge.
```

### Dependency graph (no hard `gated_on` on the critical path — disk-read fallback)

```
exp3924 (archive/retire-facts) ─── green-gate for all
exp3925 (competent judge) ──► exp3926 (efficiency) ──► exp3927 (cascade)
exp3928 (moat replication)  ── independent (reuses exp3915 harness)
exp3929 (agentic run)       ── reuses exp3919 scaffold + exp3925 verifier
exp3930 (FR-11 v26)         ── reuses exp3927 cascade band
exp3931 (hardware)          ── independent
exp3932 (literature)        ── independent
exp3933 (capstone)          ── disk-reads all above; skips flagged artifacts
```

Each downstream task **disk-reads** its upstream artifact and emits
`blocked_upstream_*` if absent — a skipped upstream costs ONE task, never a
cascade (the `.340/.361` lesson). No hard `gated_on`.

## 4. Architecture / what this milestone touches

- `python/carnot/verify/gguf_inference.py` — the `.362` robust harness; reused
  verbatim by every live-model task.
- `python/carnot/verify/reasoner_self_verification.py` — the LLM-judge; `.363`
  ships a competent-judge config (polarity-corrected + boosted prompt + stronger
  model option).
- `python/carnot/verify/cost_instrumented_verification.py` — the cost harness;
  `.363` adds amortized-load exclusion + FLOP/token accounting.
- `python/carnot/agentic/arc_agi3_harness.py` — the `.362` scaffold; `.363`
  adds a realistic env + the action-efficiency measurement.
- `ops/exclusion_manifest.yaml` — `.363` retires the facts/graph-grounding route.

## 5. Hardware requirements

- **2× RTX 3090 (CUDA)** — required for the live LLM-judge runs (exp3925/3926/
  3928). Every GPU task uses `{project_root}/.venv/bin/python` (bare `python`
  has no torch → silent CPU drop) and routes inference through the robust
  `gguf_inference` harness (smoke-before-use + GGUF fallback).
- **KV260 / PolarFire / GateMate** — opportunistic continuity (north-star §3);
  KV260 via SSH only (never host SD card). No board blocks a milestone.

## 6. Invariants (carried from .356-.362)

- `paper_ready` stays TRUE (G1-G4 met); FoVer **0.9131 frozen**, never silently
  substituted.
- Verifier stays **math-domain-bound** (facts retires this milestone; do not
  re-test generalization without a new architecture).
- Never aggregate `flagged_adversarial` artifacts into a headline or capstone.
- Energy-as-generator is closed-negative (EBT/Route-1/Route-2 bounded) — no
  generator experiments; the verifier is the surviving asset.
- No external publication; operator-only.
- Routing: all tasks `codex` + `requires_codex` + `gpt-5.5` (anti-wipeout;
  gemini crashes GPU workloads / 429-wiped `.333/.355`; standing operator
  gemini↔codex flip authority 2026-06-05). GPU tasks add `requires_gpu`.

## 7. Discipline notes specific to .363

- **Positive control on every comparison.** The `.362` efficiency flaw was a
  comparison against a sub-chance baseline. exp3925 ships a validated competent
  judge BEFORE exp3926/3927 use it; exp3926 asserts the judge AUROC is above a
  defensible floor before reporting any parity/cost verdict; exp3927 asserts
  `escalation_fraction > 0` before claiming the cascade works.
- **Defensible cost.** The 11512× wall-clock ratio is reported alongside a
  FLOP/token-based ratio with amortized model-load excluded, so the efficiency
  headline survives adversarial review.
- **No unit-test wall-clock floor on a fixture** (the `.361` poison-test lesson).
  The 60 s live-floor is asserted ONLY on full-corpus science artifacts.
- **Honest nulls are results.** If the competent judge BEATS the energy verifier
  on accuracy, that is the real finding — and the cascade (escalate close-calls)
  becomes the deployable story, which must then actually escalate.

## 8. New references integrated (see research-references.md, 2026-06-07 sweep)

- ARC-AGI-3 Technical Report (arXiv:2603.24621) — action-efficiency scoring,
  verifier-prioritized domains; THE agentic venue.
- Executable World Models for ARC-AGI-3 (arXiv:2605.05138).
- "Know When You're Wrong" (arXiv:2603.06604) — LLM error-detection calibration;
  the below-chance-judge diagnosis.
- JudgeRLVR (arXiv:2601.08468); ToolPRMBench (arXiv:2601.12294);
  CompassVerifier (arXiv:2508.03686).
