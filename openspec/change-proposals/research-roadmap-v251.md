# Research Roadmap v251: Ensemble v9 Adversarial Validation + External Benchmarks + TTT + FJD Safety v2

**Milestone:** 2026.05.251
**Previous milestone:** 2026.05.250
**Date:** 2026-05-20
**Status:** PROPOSED

## Post-.250 Planning Sweep New Papers (2026-05-20)

Three new papers added to research-references.md from the .251 planning sweep:
- **arXiv:2605.17028** (PARALLAX) — Meta-evaluation of 22 hallucination detectors on 6 corpora; artifact-controlled leaderboard; queued as additional evaluation target in exp2623.
- **arXiv:2602.11364** (DiffuTruth / Energy of Falsehood) — Thermodynamic verification using NLI contradiction scores; validates Carnot's energy-basin hallucination framework; paper-v6 §2 citation.
- **arXiv:2603.22303** (AvgWD/EigenWD) — Training-free embedding-geometry hallucination signals; candidate Tier 0w verifier requiring only embedding access.

---

## What Milestone .250 Proved (7 wins, 3 gaps)

### Wins
1. **sklearn fix landed** (exp2609): scikit-learn installed in conductor venv; 25-retro execution drought broken; entire verifier retrain chain unblocked.
2. **tier0s real-corpus retrain** (exp2610): TF-IDF logistic regression on FoVer real corpus; AUROC carry-forward from .250 capstone.
3. **tier0u TF-IDF fix** (exp2611): NLI-proxy cosine overlap on real corpus; improved from 0.5360 near-random baseline.
4. **Tier 0z Semantic Energy prototyped** (exp2612): training-free Boltzmann verifier per arXiv:2508.14496; no synthetic-to-real distribution gap.
5. **Safety Tier B viable** (exp2613 + exp2616): safety corpus built, Shannon entropy features validated, Group F added to ensemble.
6. **Ensemble v9 built** (exp2615): new ensemble incorporating improved verifiers + Tier 0z; AUROC claim generated.
7. **Rule 3 distribution executed** (exp2614): HF model card citation stub updated; IPFS pin attempted.

### Gaps
1. **Ensemble v9 AUROC not adversarially validated** — exp2615 generated a single-seed AUROC claim. Per CLAUDE.md Adversarial Artifact Verification rule: "a claim about a benchmark accuracy requires N >= 30 examples AND multi-seed replication before it can be cited in paper-v6." No adversarial_verify.py pass has been confirmed clean for v9. This gap is the PRIMARY block on paper-v6 finalization.
2. **No external benchmark coverage** — all verifier evaluation is on FoVer synthetic corpus. HalluScan (hallucination-specific) and WildGuard (safety-specific) were not tested; headline claims are unvalidated on out-of-distribution corpora.
3. **TTT loop not demonstrated end-to-end** — JEPA online_update() was evaluated (exp2617) but the full Verifier-Driven TTT loop (arXiv:2505.19475: verifier scores → select examples → test-time fine-tune) has not been prototyped; FR-11 "Continuous Self-Learning" mandate is partially satisfied at Tier 3 (online_update) but Tier 3 TTT integration is pending.

---

## Three Biggest Gaps Between Current State and PRD Vision

### Gap 1: Adversarial validation of ensemble v9 (HIGHEST PRIORITY)
**State:** Ensemble v9 AUROC was computed in exp2615 but on a single seed and without adversarial_verify.py. The CLAUDE.md adversarial discipline requires 5-seed replication with AUROC mean ± std before paper-v6 §5 can cite the new headline number. Without this, the paper improvement claim over v7b (0.9857) is not headline-eligible.

**Why it matters for PRD:** The primary paper claim is that Carnot's ensemble AUROC exceeds the detection threshold for meaningful self-improvement (arXiv:2505.19475: AUROC > 0.65 needed). A non-adversarially-validated AUROC could be inflated by seed selection or eval set overlap. Per CLAUDE.md: "Cross-check surprising results... include a successful replication artifact (n>1 seeds, same direction)."

### Gap 2: External benchmark coverage — no OOD validation
**State:** FoVer is the sole evaluation corpus. The verifiers were trained on FoVer-derived features. A genuinely independent benchmark (HalluScan for hallucination, WildGuard for safety) would confirm that the AUROC improvements generalize. The tier0s real-retrain specifically targeted FoVer distribution; its OOD performance is unknown.

**Why it matters for PRD:** FR-12 (Verifiable Reasoning) requires that verification generalizes beyond the training distribution. HalluScan and WildGuard are independently curated; they constitute the "third-party reproducer" evidence needed for Phase 1 ship.

### Gap 3: FR-11 TTT loop not closed
**State:** FR-11 Tier 3 (JEPA online_update) is wired and evaluated. But arXiv:2505.19475 demonstrates that AUROC > 0.65 is sufficient for a verifier-driven TTT loop to improve model quality by 32% relative. Carnot's ensemble exceeds that threshold — yet no code exists to run the loop end-to-end: (verifier score responses → select high-confidence examples → test-time adapt LLM). Building this closes FR-11 at the prototype level.

**Why it matters for PRD:** Phase 1 ship requires all FR-* technical requirements implemented. FR-11 "Autonomous Self-Learning Loop" has four tiers. Tiers 1-3 are wired; Tier 3 TTT integration is the last missing piece to claim FR-11 complete at prototype fidelity.

---

## Architecture Snapshot (entering .251)

```
Verifier Ensemble v9 (AUROC = pending adversarial validation, single-seed from .250):
  Group A (logprob): tier0a, tier0b, tier0c
  Group B (semantic): tier0d, tier0e, tier0f
  Group C (type/logic): tier0g, tier0h, tier0i
  Group D (Curry-Howard): tier0r (AUROC=0.9123)
  Group E (hallucination-specific): tier0t (dynamical), tier0v (HalluField proxy)
  Group E-retrained: tier0s (HalluGuard retrained real-corpus via .250 exp2610)
                     tier0u (Logical Consistency TF-IDF via .250 exp2611)
  Group F (safety): tier0x (Shannon entropy safety classifier — .250 exp2613)
  Pending .251:
    Tier 0z (Semantic Energy, training-free) — integrated in .250; adversarial-validate in .251
    Tier 0x v2 (FJD logit-temperature scaling — arXiv:2509.14558) — prototype in .251

FR-11 Self-Learning Stack:
  Tier 1: Online weight updates — WIRED
  Tier 2: Constraint memory — WIRED
  Tier 3: JEPA online_update() — WIRED + evaluated on real data (.250 exp2617)
  Tier 3 TTT: Verifier-Driven TTT loop — NOT YET PROTOTYPED (pending exp2624)
  Tier 4: Adaptive energy (KAN structural) — PROTOTYPED

Hardware:
  GateMate A1-EVB-2M: TERMINAL (.247 capstone exp2580)
  KV260: NON-TERMINAL (SD card absent; synthesis succeeded; continuing)
  PolarFire SoC: TERMINAL (.241 exp2501)
  RTX 3090 x2: available

Publication:
  arxiv_ready_v4 = True (exp2558, .246)
  Phase 4 honest negative: documented §4.4 (.246)
  arXiv ID: pending operator submission (OPERATOR-ONLY action)
  HF model cards: citation stub added (.250 exp2614)
  IPFS mirror: pin attempted (.250 exp2614; verify in .251 exp2631)
  paper-v6 §7 (Safety) and §8 (Distribution): not yet updated with v9 results
```

---

## Dependency Graph

```
exp2621 (archive .250 + activate .251)
    │
    ├── exp2622 (ensemble v9 adversarial validation 5-seed) ────────────┐
    │       │                                                            │
    │       ├── exp2627 (BB-UCP conformal calibration) [gated]          │
    │       │                                                            │
    │       └── exp2629 (paper v6 §7+§8+v9 AUROC) [gated] ◄────────────┘
    │
    ├── exp2623 (external benchmark HalluScan+WildGuard) ─(indep)
    │
    ├── exp2624 (Verifier-Driven TTT prototype) ─(FR-11 mandate)
    │
    ├── exp2625 (FJD Safety Tier 0x v2) ─(indep)
    │
    ├── exp2626 (Multi-Exit KAN energy verifier) ─(indep)
    │
    ├── exp2628 (KV260 terminal attempt) ─(hardware mandate)
    │
    ├── exp2630 (live GGUF pipeline smoke test) ─(indep)
    │
    └── exp2631 (distribution final mile HF+IPFS post-v9) ─(indep)
                                                    │
                                        ┌───────────┘
                                        ▼
                                exp2632 (capstone v251, claude+opus)
                                        │
                                exp2633 (retro v251)
```

---

## Phase Descriptions

### Phase 0: Archive and Activation (exp2621)
Archive milestone .250 into `research-complete.yaml`. Activate .251 by copying `research-roadmap-next.yaml` → `research-roadmap.yaml`. Records .250 outcomes: sklearn fix landed, verifier chain unblocked, ensemble v9 built, safety Tier B viable.

### Phase 1: Adversarial Validation (exp2622)
**5-seed ensemble v9 replication on real FoVer corpus.** Runs `adversarial_verify.py` on each seed artifact. Records `ensemble_v9_auroc_mean`, `ensemble_v9_auroc_std`, `adversarially_verified: bool`. This is the gate for paper-v6 update (exp2629) and conformal calibration (exp2627). Without this, paper claims about v9 remain non-headline-eligible.

Concretely:
- Seeds: 42, 7, 99, 1234, 2026
- For each seed: load real FoVer corpus, run ensemble scoring, compute AUROC, run adversarial_verify.py
- Gate: mean AUROC ≥ 0.90 AND adversarial_verify.py no CRITICAL flags on majority of seeds

### Phase 2: External Benchmarks (exp2623)
**Out-of-distribution evaluation on HalluScan + WildGuard.** Uses publicly-available datasets to test whether tier0s/tier0u/tier0z generalize beyond FoVer. Records per-verifier AUROC on each external corpus. Documents distribution shift magnitude. No retraining — pure evaluation pass.

PRECONDITIONS: HuggingFace credentials available for WildGuard download.

### Phase 3: TTT Loop Prototype (exp2624)
**Verifier-Driven TTT loop per arXiv:2505.19475.** FR-11 Tier 3 TTT integration. Implements the selection→adapt→evaluate cycle:
1. Run verifier ensemble on a set of LLM responses
2. Select top-N high-confidence examples (AUROC > 0.65 threshold per the paper)
3. Use those examples as few-shot context for a test-time adapted inference run
4. Measure improvement on held-out examples

This does NOT require full gradient fine-tuning — the paper shows in-context TTT (few-shot selection) provides meaningful improvement when verifier quality exceeds 0.65. Closes FR-11 Tier 3 at prototype fidelity.

### Phase 4: Safety v2 + KAN Enhancement (exp2625, exp2626)
**exp2625 (FJD Safety Tier 0x v2):** Implements the logit-temperature scaling approach from arXiv:2509.14558 ("LLM Jailbreak Detection for (Almost) Free!"). Replaces Shannon entropy feature in Tier 0x with temperature-scaled logit divergence. No training required. Target: safety_auroc_v2 > 0.70 (improve over Tier 0x v1 from .250).

**exp2626 (Multi-Exit KAN):** Adds prediction heads at each KAN layer per arXiv:2506.03302. Reduces local Lipschitz constant, improves parsimony. Tests whether early-exit prediction from inner layers matches full-path AUROC — if yes, enables ~3x inference speedup on CPU.

### Phase 5: Conformal + Paper (exp2627, exp2629) — gated on exp2622
Both tasks are gated on `exp2622.adversarially_verified == true`. Running before adversarial validation would waste effort if the headline number changes.

**exp2627 (BB-UCP conformal bootstrapping):** Implements label-free conformal calibration per arXiv:2509.23002. Bootstraps calibration intervals over ensemble scores without requiring labeled test data. Records `coverage_at_90pct`, `bb_ucp_interval_width`. Strengthens deployment claims in paper-v6 §6.

**exp2629 (Paper v6 final polish):** Updates main.tex with adversarially-validated v9 AUROC, §7 Safety with Group F results + FJD v2 if viable, §8 Distribution with IPFS CID and HF mirror link. Final integrity check before operator submission.

### Phase 6: Infrastructure + Hardware (exp2628, exp2630, exp2631)
**exp2628 (KV260 terminal attempt):** Per Hardware-Task Continuity Discipline (CLAUDE.md), KV260 is NON-TERMINAL and mandates one task per milestone. Branch A: if SD card detected → PYNQ flash + latency smoke test. Branch B: SD absent → update prep script with exact SD-insertion instructions + verify synthesis bitstream still valid.

**exp2630 (live GGUF pipeline smoke):** End-to-end smoke: 20 examples through VerifyRepairPipeline with Qwen3.6-35B-A3B-GGUF. Measures `pipeline_e2e_latency_mean_s`, `verification_rate`, `repair_success_rate`. Validates that the ensemble (v9) works with real GGUF outputs, not just FoVer corpus.

**exp2631 (distribution final mile):** Post-v9 HF model card update + IPFS re-pin with new AUROC. Verifies IPFS CID is accessible. Updates README Distribution Channels section. Adds v9 weights hash to model card.

### Phase 7: Synthesis (exp2632, exp2633)
**exp2632 (capstone, claude+opus, requires_claude):** Synthesizes all .251 outcomes across 11 deliverables. Produces operator-action checklist: (a) is ensemble v9 adversarially validated? (b) does external benchmark coverage confirm OOD generalization? (c) is TTT prototype operational? (d) is FJD v2 safety improvement confirmed? (e) is paper-v6 ready for operator arXiv submission? (f) is KV260 TERMINAL? (g) operator recommendations. The claude requirement is justified: this is 11-artifact cross-synthesis with open-ended operator recommendation under ambiguity — meets all 3 positive-criterion conditions per CLAUDE.md.

**exp2633 (retro, codex):** Operational retrospective, changelog and status updates.

---

## Hardware Requirements

| Board | State | .251 Task | Next Gate |
|---|---|---|---|
| KV260 | NON-TERMINAL | exp2628 (Branch A: SD detected → flash; Branch B: prep update) | SD card insertion + PYNQ flash |
| GateMate A1-EVB-2M | TERMINAL | None | Graduated |
| PolarFire SoC | TERMINAL | None | Graduated |
| RTX 3090 x2 | Available | exp2630 (GGUF smoke, optional GPU) | None |

---

## Decentralization Compliance Check (CLAUDE.md Rules 1–7)

1. **Local-first open models** — exp2630 uses Qwen3.6-35B-A3B-GGUF (local). All verifier evals use local FoVer corpus. exp2624 TTT uses in-context (no gradient) — no external calls. ✓
2. **Closed-weight integration optional** — no closed-weight API calls in any .251 task. exp2630 uses GGUF local model. ✓
3. **Distribution mirroring** — exp2631 updates HF primary + IPFS secondary per Rule 3. ✓
4. **Multiple integration surfaces** — verifier pipeline exposed via Python API + CLI + MCP server; no single-surface drift. ✓
5. **Hardware portability** — all verifier tasks run on CPU (logistic regression, TF-IDF). GGUF runs on CPU with optional GPU. KV260 is sovereignty hardware. ✓
6. **Data minimization** — no closed-weight LLM calls means no data flows to external vendors. ✓
7. **No vendor abstractions in core** — all .251 code targets `python/carnot/verify/` via abstract protocols. ✓

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` against .251 task scopes. Retired IDs: 2091, 260, 308, 309, 346, 380, 381, 382, 383, 410, 425, 491, 527, 603, 627.

| .251 Task | Pattern checked | Result |
|---|---|---|
| exp2622 adversarial validation | No retired "ensemble adversarial validation" pattern | CLEAR |
| exp2623 external benchmark | No retired HalluScan/WildGuard experiment | CLEAR |
| exp2624 TTT prototype | arXiv:2505.19475 is NEW (not retired) | CLEAR |
| exp2625 FJD safety v2 | GRPO (527, 603, 627) — different domain (code, not safety logit); SelfGrader (separate paper) — different technique | CLEAR |
| exp2626 Multi-Exit KAN | KAN verifier lineage: no retired Multi-Exit variant | CLEAR |
| exp2627 BB-UCP conformal | No retired conformal calibration experiment | CLEAR |
| exp2628 KV260 | iCE40 PIMI (retired) — different (PIMI vs SD flash); KV260 SD flash is non-retired track | CLEAR |
| exp2629 paper polish | No retired "paper-v6 final polish" task | CLEAR |
| exp2630 GGUF smoke | WOPR puzzle cartridge (retired 425) — different domain; GGUF smoke is new | CLEAR |
| exp2631 distribution HF+IPFS | No retired distribution task (exp2614 completed in .250) | CLEAR |
| exp2632 capstone | No retired capstone pattern | CLEAR |
| exp2633 retro | No retired retro pattern | CLEAR |

Zero manifest matches. Milestone activation not blocked by exclusion manifest.

---

## Failed-Experiment Rerun Compliance Table

| Task | Prior failure(s) | Addressed by |
|---|---|---|
| exp2621 (archive .250) | exp2608 (archive .249) — different milestone | Different scope: archiving .250, not .249 |
| exp2622 (adversarial val) | exp2619 (capstone .250) — single-seed synthesis, no 5-seed replication | This task runs explicit 5-seed replication per sample-size rigor |
| exp2623 (external benchmark) | No prior HalluScan/WildGuard eval — first attempt | N/A — new scope |
| exp2624 (TTT prototype) | exp2617 (JEPA real-data eval .250) — evaluated online_update() but not TTT loop | Different scope: TTT loop (select → adapt → eval) vs. online_update() instrumentation |
| exp2625 (FJD safety v2) | exp2613 (safety corpus .250) — Shannon entropy features only | Different technique: logit-temperature scaling per arXiv:2509.14558 |
| exp2626 (Multi-Exit KAN) | KAN verifier prior iterations — no multi-exit variant attempted | New architecture: per-layer prediction heads |
| exp2627 (BB-UCP conformal) | No prior conformal bootstrapping attempt | N/A — new scope |
| exp2628 (KV260) | exp2618 (KV260 branch A/B in .250) | Same branch structure; .250 result determines which branch is relevant |
| exp2629 (paper polish) | exp2616 (safety Group F + §7 stub) — stub only | This is final polish with adversarially-validated v9 AUROC; different scope |
| exp2630 (GGUF smoke) | No prior GGUF end-to-end pipeline smoke test | N/A — new scope |
| exp2631 (distribution) | exp2614 (HF+IPFS .250) — citation stub + attempted IPFS pin | Post-v9 update: adds v9 AUROC + verifies IPFS CID accessible |
| exp2632 (capstone) | exp2619 (capstone .250) — 0 prior experiments → unclear synthesis | .251 has substantive results to synthesize across 11 deliverables |
| exp2633 (retro) | exp2620 (retro .250) | Standard retro continuation |

---

## Agent Routing

| Task | Agent | Why |
|---|---|---|
| exp2621–exp2631, exp2633 | codex + gpt-5.5 | Formulaic: archive, run AUROC, load corpora, implement verifier technique, hardware branch, YAML write. Each is single-scope with deterministic acceptance gates. |
| exp2632 (capstone) | claude + opus, requires_claude: true | Cross-artifact synthesis of 11 deliverables with open-ended operator recommendations. Meets all 3 positive-criterion conditions: (1) codex produced incomplete .250 capstone when experiments were gated; (2) 11 deliverables require multi-file cross-context synthesis; (3) operator recommendation is open-ended judgment under ambiguity, not a deterministic threshold check. |

**Routing distribution:** 12 codex/gpt-5.5 (92.3%), 1 claude+opus (7.7%) — codex-default discipline maintained.

---

## Critical Path

```
exp2622 (adversarial val) → exp2627 (BB-UCP) + exp2629 (paper polish)
                          → exp2632 (capstone)
```

Paper-v6 operator submission cannot proceed until exp2622 returns `adversarially_verified: true`. If exp2622 finds a CRITICAL adversarial flag on the v9 AUROC claim, exp2629 pivots to document the finding rather than update the claim.

---

## What Success Looks Like for .251

- `adversarially_verified: true` (exp2622) — v9 AUROC headline-eligible for paper-v6
- `ensemble_v9_auroc_mean >= 0.95` with `auroc_std < 0.02` (exp2622) — improvement over v7b (0.9857) confirmed multi-seed
- `halluscan_auroc > 0.70` AND `wildguard_auroc > 0.65` (exp2623) — OOD generalization confirmed
- `ttt_loop_ran: true` with `n_selections >= 5` (exp2624) — FR-11 Tier 3 TTT prototype operational
- `safety_auroc_v2 > 0.70` (exp2625) — FJD logit-temperature improves over v1
- `multi_exit_viable: bool` (exp2626) — early-exit speedup assessed
- `bb_ucp_coverage_at_90pct > 0.85` (exp2627) — conformal calibration tighter than naive
- `paper_updated_with_v9: true` AND `submission_package_ready: true` (exp2629) — ready for operator arXiv submit
- `pipeline_e2e_latency_mean_s < 30.0` (exp2630) — live GGUF smoke passes
- `ipfs_cid_verified: true` (exp2631) — content-addressed distribution confirmed
- `n_experiments_completed >= 7` (exp2632) — sustained execution above drought
