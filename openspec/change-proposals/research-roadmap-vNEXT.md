# Research Roadmap — Milestone 2026.06.340

**Status:** Pre-staged by outer-loop Claude (Opus 4.8), 2026-06-02.
**Predecessor:** 2026.06.339 (a CONVERGENCE milestone — every major open research thread reached a
terminal/settled state: re-freeze closed-negative, code = leak → math-only-with-abstain, selection
diagnosis formally closed, KV260 terminal, FR-11 v13 positive, paper_ready stayed TRUE).
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.340`)

---

## 1. What the previous milestone (.339) proved

`.339` validated the two PROVISIONAL `.338` wins under full rigor — and **both walked back to the
conservative outcome**. Read via `scripts/summarize_artifact.py`:

| Result | Finding | Artifact |
|---|---|---|
| **Re-freeze CLOSED-NEGATIVE** | Under the FROZEN 5-seed dual-condition protocol, dependency-aware (0.9249), the external de-entangled/CIG baseline (0.9287), and a FUSION (0.9285) **all fail to robustly beat frozen 0.9131** (the paired delta CI does not exclude 0). The .338 single-condition leads evaporated. Verdict `no_candidate_beats_frozen_headline_stays_0_9131`. **The headline stays 0.9131.** | exp3704 |
| **— but exp3704 is FLAGGED (benign)** | `flagged_adversarial=True`, live re-check CRITICAL: a **TAUTOLOGY** between `strongest_candidate_auroc` and `external_comparator_auroc` — which are equal **by construction** (the external candidate IS the strongest). A linter false-positive (a legitimate copy, not two distinct measurements coinciding), but the milestone record now carries a flagged headline-adjacent artifact. | exp3704 |
| **Code signal was a LEAK** | exp3695's code AUROC=1.0 reproduced in-corpus; on a held-out corpus it was **0.9932 (≥0.99 → still a leak red-flag)**. Verdict `one_point_zero_was_a_leak_code_claim_narrowed_earned`. | exp3705 |
| **Shipped detector made honest** | exp3706 NARROWED the shipped second-pair-of-eyes detector back to **math-only with an explicit abstain-on-code path**, math operating point (AUROC ~0.98, ECE ~0.009) preserved, E2E green. | exp3706 |
| **Selection diagnosis FORMALLY CLOSED** | exp3707 recorded the closure + an operator retirement recommendation. Energy SELECTION is settled-bounded (exp3672 earned-negative + arXiv:2512.23067). | exp3707 |
| **FR-11 v13** | Multi-session Tier-2 consolidation transferred to a fresh session without collapse, library bounded. | exp3708 |
| **KV260 TERMINAL** | Non-fabricated on-board Ising-sampler latency transcript captured (POC functional anchor, no speedup claim) → terminal candidate. | exp3709 |
| **Publication gate** | **paper_ready = TRUE.** G1 (FoVer 0.9131, exp2850), G2 (CI run 26725185125), G3, G4. Frozen headline unchanged. | exp3712 |
| **Backend** | gemini still crashes real workloads (exp3703, marked Failed/flagged). Keep codex+requires_codex. | exp3703 |

**Strategic position:** `.339` confirmed that the project has **converged**. The headline (FoVer 0.9131)
is frozen, proven, independently reproduced, and narrowing-clean — `paper_ready=true`. Every direction
the autonomous loop can self-generate is now settled, most as *trustworthy negatives*:

- energy-descent existential claim (P0.1): **honest-negative**, bounded (Route-1/Route-2)
- energy SELECTION beats SC: **settled-bounded**, diagnosis CLOSED (exp3707)
- headline re-freeze via reweighting: **closed-negative** (no candidate beats frozen, exp3704)
- code generalization: **leak**, narrowed to math-only-with-abstain (exp3705/3706)
- facts generalization: **RETIRED** (exp3670); trained-judge-OOD: **RETIRED** (exp3659)
- KV260 sovereignty story: **TERMINAL** (exp3709)

This is exactly the state the `project_energy_selection_thesis_bounded` memory anticipated:
> "Foundation-model alternatives need a human-seeded thesis; the loop won't self-initiate one."

`.340` therefore does **not manufacture breadth** (north-star §1: a milestone that produces a new
version of an existing artifact without moving the headline is *noise*). It is a **CONVERGENCE &
FINALIZATION** milestone.

---

## 2. The three biggest gaps (PRD vision vs current state)

1. **The publication gates are met but two of them rest on honor-discipline, not enforcement.** G3
   (prose narrowing-clean) is checked by an inline phrasing scan in `publication_gate.py`, but
   CLAUDE.md's Paper-v6 Narrowing Discipline says a `paper_v6_narrowing_lint.py` pre-commit hook
   *SHOULD* exist and does not. G4 (numbers trace to artifacts) has never been audited end-to-end for
   *every* headline number against *clean, non-flagged* artifacts. paper_ready=true is only as strong
   as its weakest enforcement. `.340` ships the G3 mechanical lint and runs a full G4 provenance audit —
   converting the gate from "true today" to "true and mechanically defended."

2. **The proven discriminator's DEPLOYABLE value is uncharacterized.** Carnot's 0.9131 step-error
   discrimination is real and frozen, but the SELECTION conversion (best-of-N argmax beats SC) is
   settled-negative. The OTHER conversion — discrimination → **selective-prediction / abstention** (flag
   or abstain on likely-wrong steps, the literal job of the shipped "second pair of eyes") — has never
   been characterized. This is the on-mission product framing (escape hallucinations = abstain when
   wrong). arXiv:2603.21172 ("Entropy Alone is Insufficient") shows heuristic uncertainty fails as a
   selective-prediction signal — motivating an ENERGY-based one. `.340` characterizes the risk-coverage
   curve (AURC, risk@coverage) of the proven discriminator. This is the single genuinely-new, defensible
   direction — and it is explicitly NOT the closed SELECTION question.

3. **The headline rests on a single corpus (FoVer).** G1 is strong but FoVer-specific. A defensible
   headline survives a *second, distinct* step-error corpus. `.340` replicates the 0.9131-class
   discrimination on a FRESH corpus (a different PRM/step-error set) — the north-star-sanctioned way to
   advance the headline ("replicates on a new seed/corpus"), as opposed to re-measuring FoVer (churn).

**The meta-gap (surfaced, not closed by the loop):** the next *substantive research* direction requires
a **human-seeded thesis**. `.340` includes a synthesis task that lays out the converged state and the
candidate next-theses for the OPERATOR to choose among — it does not pick one autonomously.

---

## 3. Milestone architecture (4 phases, 11 tasks)

```
Phase 0 — Transition + routing safety (exp3713, exp3714)
    archive .339 / activate .340  -->  backend-state diagnostic v6 (6th gemini probe; gates a .341 flip)

Phase 1 — Record hygiene + gate hardening (exp3715, exp3716, exp3717)
    exp3715  corrigendum: clean re-emit of the flagged exp3704 (benign TAUTOLOGY) — headline stays frozen
    exp3716  SHIP paper_v6_narrowing_lint.py (G3 honor-discipline -> mechanical pre-commit lint)
    exp3717  full G4 provenance audit: every headline number -> a clean, non-flagged primary artifact

Phase 2 — Deployable value + headline robustness (exp3718, exp3719)
    exp3718  RISK-COVERAGE abstention characterization of the proven 0.9131 discriminator (AURC,
             risk@coverage, energy-vs-entropy selective prediction) — the sanctioned NEW framing
    exp3719  REPLICATE the headline-class discrimination on a FRESH step-error corpus (G1 strengthening)

Phase 3 — Self-learning + hardware relaxation + convergence synthesis + capstone
    exp3720  FR-11 continuous self-learning v14 (mandatory continuous-self-learning task)
    exp3721  hardware: KV260 terminal CONFIRM + mandate-relaxation recommendation + PolarFire/GateMate
             opportunistic audit (one consolidated task, per north-star §3)
    exp3722  CONVERGENCE SYNTHESIS + operator next-thesis recommendation (the meta-gap surfacing)
    exp3723  capstone v340 + G1-G4 gate synthesis
```

**Dependency graph (no fragile gates — every task is independently runnable):**
- No task is `gated_on` another task's runtime field. exp3717 (provenance audit) and exp3723 (capstone)
  *read* upstream artifacts but use disk-presence preconditions + "not_measured" fallbacks (never a
  None-read cascade), so a skipped upstream degrades gracefully.
- exp3715 (corrigendum) is independent of exp3704's flagged state — it re-derives the conclusion cleanly.
- exp3718/3719 are independent science tasks. exp3722 synthesizes whatever landed; exp3723 capstones.

---

## 4. Why this is convergence-respecting, not churn

| Settled thread | `.340` does NOT | `.340` DOES |
|---|---|---|
| energy SELECTION (closed exp3707) | re-run a best-of-N selection experiment | characterize ABSTENTION (a different conversion of discrimination) |
| re-freeze (closed-negative exp3704) | run a 4th reweighting candidate | re-emit the flagged artifact CLEAN; keep headline frozen |
| code generalization (leak) | re-attempt a code-native AUROC | (left settled; detector is math-only-with-abstain) |
| FoVer headline (frozen) | re-measure FoVer for the Nth time | replicate on a DIFFERENT corpus (a new datapoint) |
| KV260 (terminal) | run another KV260 latency experiment | CONFIRM terminal + relax the per-milestone mandate |
| paper_ready (true) | declare victory and stop | HARDEN G3/G4 from honor-discipline to mechanical |

Every `.340` task either (a) advances the headline on a NEW datapoint, (b) closes a real enforcement
gap, (c) characterizes deployable value in a genuinely-new (non-settled) frame, (d) is mandatory
self-learning, or (e) surfaces the operator decision the loop cannot make. None re-grinds a settled
question.

---

## 5. Invariants (carried from .334–.339)

- **`paper_ready = true` (G1∧G2∧G3∧G4) MUST NOT regress.** Frozen FoVer headline **0.9131 stays frozen**;
  any candidate is an operator re-freeze CANDIDATE, never a silent swap. `ops/north-star.md` is
  operator-curated — no task edits it or triggers the CI reproducer.
- **P0.1 stays honest-negative**; energy-SELECTION settled-bounded; facts-generalization + trained-judge-OOD
  RETIRED; the selection diagnosis CLOSED (exp3707).
- **Backend: all tasks codex + `requires_codex`** (anti-wipeout — `.333` was a whole-milestone gemini
  crash wipeout, exp3703 re-confirmed real-workload crashes). exp3714 runs the 6th gemini probe that
  gates a possible `.341` flip. The operator may override to gemini-default at activation if quota
  preservation outweighs the wipeout risk.
- **Inference-substrate hygiene:** every aggregation / verifier-scoring task sets `inference_substrate`
  correctly and carries **NO GGUF/CUDA/live-model marker** in `model_specs`/`target_model`, then runs
  `adversarial_verify` and confirms clean before finishing (the .337 DURATION false-flag fix, kept fixed).
- **Leak-guard:** any AUROC ≥ 0.99 on n≥1000 is treated as a leak until proven leak-free.
- **No poison tests:** parametrize pytests over honest verdicts on realistic synthetic fixtures; never
  hard-assert one success string against a real corpus; never use Q/R/H-number placeholder tokens.
- **Every `gated_on` is a BARE scalar** (none used in this milestone — graceful disk-presence fallbacks
  instead).

---

## 6. Hardware

- **KV260:** terminal candidate reached (exp3709). `.340` exp3721 CONFIRMS the terminal state (overlay +
  non-fabricated latency transcript on disk) and recommends the operator LIFT the per-milestone KV260
  mandate (north-star §3). SSH-reachability precondition only; NO host SD-card check; NO speedup claim.
- **PolarFire / GateMate:** opportunistic only — folded into exp3721 (reachability/continuity audit,
  GateMate documentation-only because `openFPGALoader` is missing). No board blocks the milestone.

---

## 7. SOTA local models

Only exp3719 (fresh-corpus replication) may need live generation; it names a mandated SOTA GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF` or `unsloth/gemma-4-26B-A4B-it-GGUF`), loads via the `.gguf` path (NEVER
`AutoTokenizer` on a `-GGUF` repo id), and gates on cached + CUDA. If a fresh corpus can be assembled
from on-disk labeled candidates, it runs as `verifier_ensemble_against_cached_candidates` with no live
model. All other tasks are aggregation / verifier-scoring / hardware-smoke (no live model).

---

## 8. The operator decision this milestone surfaces

The loop has converged. Continuing to emit milestones that re-touch settled questions is the churn
north-star §1 names as noise. exp3722 lays out — for the OPERATOR — the converged state and the
candidate next-theses (e.g. energy-based selective prediction at scale; a genuinely different verifier
architecture for a domain where SC is weak AND abstention has headroom; or finalize-and-submit the paper
and shift to maintenance). The loop recommends but does not choose; per the project memory, the next
substantive thesis is the operator's to seed.
