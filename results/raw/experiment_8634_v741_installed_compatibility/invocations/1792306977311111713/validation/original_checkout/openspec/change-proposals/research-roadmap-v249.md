# Research Roadmap v249: Real-Corpus Recovery + IPFS Distribution + Safety Tier B + KV260 Terminal

**Milestone:** 2026.05.249
**Previous Milestone:** 2026.05.248
**Date:** 2026-05-20
**Status:** PROPOSED
**ID allocation:** Milestone .248 used exp2582–exp2594; .249 starts at **exp2595**.

---

## What Milestones .247 and .248 Proved (and Did Not Execute)

Milestones .247 and .248 were **planning-only**: all 13 tasks in each milestone were validated and
pre-staged but zero experiments completed in either window (the 22nd and 23rd consecutive empty
retros). The structural execution gap — conductor fires retro before any tasks activate — is the
meta-bottleneck. The research-roadmap tasks planned but never run include:

- `exp2569`–`exp2581` (.247): tier0s retrain, tier0u fix, IPFS mirror, safety corpus, JEPA v3,
  GateMate continuity, KV260 continuity, ensemble v9, capstone.
- `exp2582`–`exp2594` (.248): identical critical-path scope re-proposed with proper prior_failures
  documentation; same zero-execution outcome.

**What we know from the last EXECUTED experiments (.246, capstone exp2567):**

1. **AUROC headline carry-forward**: 0.9857 (ensemble v7b, adversarially verified, cite-safe).
2. **Paper errata applied**: tier0s corrected 1.0 → 0.3758 real-corpus; tier0u 0.96 → 0.5360.
   The underlying verifiers remain near-random on natural text.
3. **arXiv ready**: `arxiv_ready_v4=True` — operator may submit at any time.
4. **GateMate TERMINAL**: bitstream flashed; chip alive (JTAG IDCODE confirmed). Graduated per
   Hardware-Task Continuity Discipline. Only on-board timing transcript remains as opportunistic
   follow-on, not mandatory.
5. **KV260 NON-TERMINAL**: SD card not inserted; `pynq_url_reachable=False`. Still requires operator
   action (insert SD media). One task per milestone mandatory until terminal.
6. **JEPA online**: `jepa_online_active=False` — JEPA v3 integration was planned but never ran.
7. **Safety classifier**: `safety_classifier_viable=False` — safety corpus task never ran.
8. **IPFS mirror**: `documented_operator_needed` — IPFS pin never executed.

---

## Three Biggest Gaps Entering .249

| # | Gap | .249 Fix |
|---|-----|---------|
| 1 | tier0s/tier0u near-random on natural text (real AUROC 0.3758/0.5360) — two verifiers wasting calibration signal | exp2596 (tier0s logistic regression on FoVer real pairs) + exp2597 (tier0u TF-IDF cosine NLI-proxy) → exp2604 (ensemble v9 gated on improvement) |
| 2 | IPFS mirror + HF model card citations not complete (Rule 3 compliance) | exp2598 (HF model card local update) + exp2599 (IPFS pin + CID) |
| 3 | KV260 non-terminal; safety classifier not viable; JEPA online not active | exp2603 (KV260 hardware), exp2600+exp2601 (safety corpus + ensemble), exp2602 (JEPA v3 online) |

---

## Architecture: .249 Phase Structure

```
Phase 0 (Admin):
  exp2595 — Archive .248 + activate .249

Phase 1 (Real-Corpus Verifier Recovery — CRITICAL PATH):
  exp2596 — tier0s retrain: logistic regression on FoVer real pairs (target AUROC > 0.65)
             prior_failure: exp2509 (synthetic NTK-proxy, real AUROC=0.3758)
  exp2597 — tier0u fix: TF-IDF cosine NLI-proxy on real text (target AUROC > 0.60)
             prior_failure: exp2535 (synthetic self-consistency, real AUROC=0.5360)

Phase 2 (Publication Distribution — Rule 3):
  exp2598 — HuggingFace model card citation update (local file edit; operator uploads)
  exp2599 — IPFS mirror: pin arXiv preprint package + generate CID

Phase 3 (Safety Classifier Tier B — Product Track):
  exp2600 — Safety corpus 200 pairs + Tier0xSafetyVerifier (Ising pattern energy)
  exp2601 — Group F safety ensemble + paper §7 stub [gated: exp2600.safety_verifier_viable==true]

Phase 4 (Continuous Self-Learning + New Paper):
  exp2602 — JEPA v3 online integration: session-level update in VerifyRepairPipeline
             (continuous_self_learning_task: true)
  exp2605 — Online CoT Verifier learnability (arXiv:2603.03538) Tier 0y prototype

Phase 5 (Hardware — MANDATORY until terminal):
  exp2603 — KV260 terminal attempt: branch A (SD card flash if inserted by operator),
             branch B (update automated prep script; document next operator step)

Phase 6 (Ensemble v9 — gated on real-corpus improvements):
  exp2604 — Ensemble v9: incorporate improved tier0s/tier0u [gated: exp2596 OR exp2597 improved]

Synthesis:
  exp2606 — Capstone synthesis (claude+opus, no hard gate)
  exp2607 — Milestone retro (codex)
```

---

## Dependency Graph

```
exp2595 (admin)
  ├─► exp2596 (tier0s retrain)──────────────────────────────► exp2604 (ensemble v9, gated OR)
  ├─► exp2597 (tier0u fix)──────────────────────────────────►/
  ├─► exp2598 (HF model card) — independent
  ├─► exp2599 (IPFS mirror) — independent
  ├─► exp2600 (safety corpus)───► exp2601 (group F, gated)
  ├─► exp2602 (JEPA v3 online) — independent
  ├─► exp2603 (KV260) — independent
  ├─► exp2605 (CoT Verifier) — independent
  └─► exp2606 (capstone) ◄── all above
      exp2607 (retro) — always last
```

---

## New Literature: Post-.248 Planning Sweep (2026-05-20)

Two new papers added to `research-references.md`:

1. **arXiv:2603.03538** — "Online Learnability of Chain-of-Thought Verifiers: Soundness and
   Completeness Trade-offs" (Balcan, Blum et al., ICML 2026 candidate). Characterizes optimal
   verifier accuracy via Littlestone dimension; addresses the soundness/completeness tension
   directly relevant to tier0s/tier0u gap. Motivates exp2605 (Tier 0y CoT verifier prototype
   using the soundness-completeness PAC bound as a selection gate).

2. **arXiv:2507.00075** — "Theoretical Modeling of LLM Self-Improvement Training Dynamics
   Through Solver-Verifier Gap" (Sun et al., 2026). Models LLM self-improvement as arising
   from the gap between solver and verifier capabilities; shows external data can be injected
   at any stage. Provides theoretical scaffolding for why tier0s/tier0u fail on natural text
   (verifier capability does not generalize across the synthetic→real gap). Added as theory
   cite in paper-v6 §3.

---

## Hardware Requirements

| Board | Current Status | .249 Task | Terminal State |
|-------|---------------|-----------|---------------|
| GateMate A1-EVB-2M | TERMINAL (bitstream flashed, JTAG alive) | None mandatory | Graduated |
| KV260 | NON-TERMINAL (SD media absent) | exp2603 | Board latency transcript + `kv260_synthesis_succeeded: true` |
| PolarFire SoC | TERMINAL (exp2501) | None | Graduated |

---

## Decentralization Compliance (CLAUDE.md Rules 1–7)

- **Rule 1** (local-first): All tasks use CPU-only sklearn/scipy; no closed-weight API calls.
- **Rule 2** (closed models optional): No closed-weight dependency in core.
- **Rule 3** (distribution mirroring): exp2598+exp2599 directly target this rule; IPFS as secondary.
- **Rule 4** (multiple integration surfaces): ensemble.py API, CLI, MCP server unchanged.
- **Rule 5** (hardware portability): KV260 track active (exp2603).
- **Rule 6** (data minimization): no closed-weight calls in any task.
- **Rule 7** (no vendor abstractions in core): no changes to core.

---

## Exclusion Manifest Cross-Check (2026-05-20)

Retired IDs in `ops/exclusion_manifest.yaml`: exp2091 (gemini bail-out), exp260, exp308, exp309,
exp346, exp380-383, exp410, exp425, exp491, exp527, exp603, exp627, and scope-retired entries
(HalluSAEGeometricProbe, discriminative JEPA).

None of the .249 task scopes match any retired experiment_id or scope:
- tier0s retrain (exp2596): different technique from exp2509 (real data vs synthetic NTK)
- tier0u fix (exp2597): different technique from exp2535 (TF-IDF cosine vs template self-consistency)
- IPFS mirror, HF model card, safety corpus, JEPA v3, CoT Verifier, ensemble v9, KV260: all new scope

Exclusion manifest cross-check: **0 conflicts found**.

---

## Failed-Experiment Rerun Compliance

| New Task | Prior Failure | Prior Verdict | Addressed By | Retire If Same? |
|----------|--------------|--------------|-------------|----------------|
| exp2596 tier0s retrain | exp2509 | real AUROC=0.3758 (near-random on natural text) | Switches from synthetic NTK-proxy training to logistic regression on 6548 real FoVer pairs | false (other approaches exist) |
| exp2597 tier0u fix | exp2535 | real AUROC=0.5360 (near-random on natural text) | Switches from template self-consistency to TF-IDF cosine overlap on natural (context, claim) pairs | false (other approaches exist) |

---

## Agent Routing Summary

| Agent | Count | Tasks |
|-------|-------|-------|
| codex (gpt-5.5) | 12 (92.3%) | exp2595–exp2605, exp2607 |
| claude+opus | 1 (7.7%) | exp2606 (capstone — cross-artifact synthesis) |

Codex-default discipline: 12/13 tasks use codex. The single claude+opus task (capstone) meets all
three positive-criterion conditions: (1) prior Sonnet capstones exhausted max-turns; (2) multi-
artifact synthesis across 11 experiments; (3) open-ended recommendation under ambiguity.
