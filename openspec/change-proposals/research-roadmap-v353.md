# Research Roadmap v353 Change Proposal — Harden the Contamination-Free Formal Core

**Milestone:** 2026.06.353
**Planned:** 2026-06-05 (Claude Opus 4.8, outer-loop planning agent)
**Milestone doc:** this file

## What the previous milestones proved (the converged state)

The project is **converged**. After ~70 milestones of energy-foundation exploration:

- **The sole defensible headline is the FoVer 0.9131 math step-level error verifier** (G1–G4 all
  pass, `paper_ready=TRUE`, G2 independently reproduced on a clean CI runner 2026-05-31).
- **Both energy-foundation routes are BOUNDED.** Energy-as-selector (P0.1) does not beat AR/SC on
  math/CSP corpora where SC is near-optimal (and is bounded where SC is weak too, exp3672).
  Energy-as-generator / Thesis-A (EBT) is discriminative-PASS but generative-BOUNDED at scale
  (operator's direct runs; exp3766). Neither is added to the exclusion manifest — they are
  *findings*, not doomed-rerun ids.
- **The verifier is math-domain-bound** — earned-negative on facts (RAGTruth + NLI) and weak on
  code. It is a reasoning verifier, not a general fact-checker.
- **The loop will NOT self-seed the next paradigm** (the Verification Trap, Deep-Think P3,
  2026-06-03). The operator must SEED the next foundation-model route (EDLM is staged + preflighted
  GO) or explicitly FREEZE. The loop scaffolds; it does not commit.

The most recent milestones (.344–.352) banked the verifier product (certified abstention point +
3 integration surfaces, gaming-resistance mitigation, anomaly-escalation advisory) and re-probed
several now-closed/bounded threads (TRM latent-symbol bridge — non-decodable orbits; LDT-gap;
generalization to facts/code). **The one GENUINELY-NEW finding to come out of this window is
exp3826**: a faithful formal-vs-learned ablation showing a **contamination-free FORMAL core**
(`tier0r_curry_howard` + `tier0u_logical_consistency`, no trained weights) at AUROC **0.8947**,
versus the full ensemble **0.9131** and the learned-only probe **0.8699**.

## Why this finding is worth a milestone (and is NOT churn)

A process-reward-model paper's single biggest credibility threat is **training-data contamination**
of the learned components. exp3826 says most of Carnot's moat (0.8947 of 0.9131) survives in a
SAT/Z3/AST-checkable core that has **no trained weights and therefore cannot be contaminated**. If
that holds across seeds, it is the strongest, most defensible framing the verifier has — a
contamination-resistant headline, not a degraded variant (cf. FoVer arXiv:2505.15960; structured
formal intermediaries arXiv:2603.29500; PRM survey arXiv:2510.08049). exp3826 is a **single run**
reporting point AUROCs. Before it can be cited in the paper it needs the standing
**Adversarial-Confirmation Discipline** treatment: multi-seed replication with a CI, a positive
control, and an honest characterization of exactly what the learned probe adds.

This is depth on the one open, publishable refinement — not a re-grind of a bounded thread.

## The three biggest gaps between current state and the PRD vision

1. **The contamination-free core is asserted from one run.** It must be replicated (5-seed CI),
   turned into a *deployable* certified product (abstention point on the formal core alone), and its
   complement (the +0.0184 learned contribution) characterized honestly. (.353 Phase 1.)
2. **Continuous self-learning has reached Tier-1/2/3 but never Tier-4** (adaptive *structure* of the
   energy function — the research-program.md long-term Kona tier). Tiers 1–3 only reweight / cache /
   predict; Tier-4 adds/prunes constraint *structure*. (.353 Phase 2.)
3. **The next-paradigm decision is staged but the loop cannot pull the trigger.** EDLM is preflighted
   GO and staged for one operator command. .353 provides the *execution surface*: a precondition-gated
   kill-gate that RUNS the moment the operator has seeded, and blocks cleanly otherwise — respecting
   the Verification Trap (the loop never self-seeds). (.353 Phase 3.)

## Architecture (what .353 touches)

```
                 cached FoVer corpus (data/fover_test_v4.json, N>=1000, gold step labels)
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        │ FORMAL core              │ LEARNED probe              │ full ensemble (FROZEN 0.9131)
        │ 0.9*tier0r + 0.1*tier0u  │ fr11_session_memory        │ formal + memory
        │ (SAT/Z3/AST, NO weights) │ (traces / learned state)   │
        ▼                          ▼                            ▼
   exp3835 5-seed CI ──► exp3836 certified abstention   exp3837 learned-contribution
   (contamination-free   point on the FORMAL CORE        per-error-category breakdown
    moat floor + CI)      (deployable clean product)      (what +0.0184 buys, honestly)
        │
        ▼ formal_only_auroc_mean (BARE gate field)
   ─────────────────────────────────────────────────────────────────────────────────
   exp3838  FR-11 v22  Tier-4 ADAPTIVE-STRUCTURE self-learning (prune marginal verifiers /
            flag uncovered residual regions; hold frozen 0.9131 within CI)   [CPU]
   ─────────────────────────────────────────────────────────────────────────────────
   exp3839  EDLM minimal kill-gate EXECUTION — precondition: operator has cloned the seed
            repo AND a free GPU; else blocked_edlm_not_seeded (loop never self-seeds)  [GPU]
   ─────────────────────────────────────────────────────────────────────────────────
   exp3840 publication-gate regression (G1–G4, frozen 0.9131 unchanged) │ exp3841 refs │
   exp3842 KV260 opportunistic │ exp3843 capstone .353
```

## Phases

- **Phase 0 — Activation.** exp3834 archive .352 / activate .353.
- **Phase 1 — Harden the contamination-free formal core (DEPTH).** exp3835 (5-seed CI replication,
  the depth anchor), exp3836 (certified abstention point on the formal core only — gated on the core
  reproducing), exp3837 (characterize the learned-probe contribution honestly).
- **Phase 2 — Continuous self-learning, Tier-4.** exp3838 FR-11 v22 adaptive-structure (the mandated
  self-learning experiment; the one untried tier).
- **Phase 3 — Operator decision execution surface.** exp3839 EDLM kill-gate, precondition-gated so it
  runs iff the operator already seeded; never self-seeds.
- **Phase 4 — Invariants + housekeeping.** exp3840 publication-gate regression, exp3841 references
  refresh, exp3842 KV260 opportunistic audit, exp3843 capstone.

## Dependency graph

```
exp3834 ─► (everything)
exp3835 ─► exp3836 (gated_on formal_only_auroc_mean >= 0.85)
exp3835, exp3836, exp3837, exp3838, exp3839 ─► exp3840 ─► exp3843
exp3841, exp3842 ─► exp3843
```

## Hardware requirements

- Phase 1, Phase 2, Phase 4: **CPU only** — re-scoring cached candidates with the existing ensemble +
  aggregation. Loop-safe (no GPU contention, no gemini/429 exposure).
- Phase 3 (exp3839 EDLM): **GPU-preferred** (≥10 GB free on the internal 2×RTX 3090). Hard
  precondition; clean `blocked_no_free_gpu` / `blocked_edlm_not_seeded` exit otherwise. Runs via
  `.venv/bin/python` (the EBT venv/CUDA interpreter discipline — bare `python` has no torch).

## Routing

Per the recent operator practice (codex cheap-default; gemini crashes real GPU workloads and has
wiped milestones via 429 — do NOT route any task to gemini): formulaic/mechanical tasks → `codex`
+ `gpt-5.5`; synthesis/judgment tasks → `claude`; the one GPU foundation-model task (exp3839) →
`claude` + `opus` + `max_turns: 100`. Every Run command pins `.venv/bin/python`.

## Invariants (must hold at capstone)

- `paper_ready` stays **TRUE** (G1–G4).
- FoVer **0.9131** stays frozen and untouched (Phase 1 is a NEW lens on the SAME ensemble — a
  decomposition + a clean-core product — NOT a re-measurement of the headline).
- Both energy routes stay **bounded** (.353 runs no energy-as-selector / energy-as-generator
  experiment; exp3839 only EXECUTES the operator's EDLM seed if present, and reports an honest
  bounded/blocked outcome — it does not re-open Thesis-A).
- No operator-curated doc is auto-edited (Public Documentation Discipline). Doc changes are emitted
  as PROPOSALS.

## Risks / discipline notes

- exp3835 scope-matches exp3820 (`INCONCLUSIVE_ablation_harness_unfaithful`) → carries a
  `prior_failures:` block (root cause: exp3820 used the wrong 5-verifier AndComposition; exp3826
  fixed the harness; exp3835 adds the 5-seed CI). `retire_if_same_verdict: true` retires the
  formal-ablation scope if multi-seed comes back inconclusive again.
- exp3839 carries an `operator_override:` (it is operator-gated EXECUTION of the staged EDLM seed,
  not a Thesis-A rerun; EDLM is a distinct residual-corrector route per exp3781/3793/3815).
- The single structured `gated_on` (exp3836 on exp3835) targets a **bare** float field
  (`formal_only_auroc_mean`) per the gated-fields-must-be-bare rule.
