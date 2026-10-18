# Research Roadmap v260 — Milestone 2026.05.260

**Proposed:** 2026-05-21
**Milestone:** 2026.05.260
**Previous milestone:** 2026.05.259 (exp2725–exp2737)
**Experiment ID range:** exp2738–exp2750
**Milestone title:** Verifier Energy v2 Live GPU + KV260 Terminal Latency + Set-Consistency Tier 0v + Phase 4 FEP + FR-11 Tier 4 Self-Learning

---

## What Milestone .259 Proved

Milestone 2026.05.259 completed 12 of 13 tasks with 10 of 12 acceptance criteria met.

**Critical findings requiring .260 follow-up (adversarial flags):**

1. **exp2727 (Verifier Energy Debug v1) — ADVERSARIAL-FLAGGED**: Duration 11.5s for a task claiming
   CUDA + GGUF + live-GPU inference. The diagnostic logging change is real (tier0s_halluguard.py +
   nup_probe.py now have fast-path vs. genuine-zero distinguishers), but the live_gguf claim was
   not validated on real hardware. The verifier fix is provisional until re-run with:
   - RTX 3090 GPU (CUDA-verified)
   - Qwen3.6-35B-A3B-GGUF cached and loaded
   - duration_s >= 60s
   - model_specs + reproducibility_checksum present

2. **exp2731 (Tier 0g Semantic Energy) — ADVERSARIAL-FLAGGED**: Duration 3.3s for live GGUF eval.
   The synthetic non-degeneracy (energies 8.4-9.1, 6584 clusters) is real, but gguf_non_degenerate=false
   (all 3 GGUF inputs produced identical energy 9.085). Root cause unknown: may be GGUF logit collapse
   or k-NN clustering collapsing on TF-IDF of GGUF-formatted text.

**Successful .259 outcomes carried forward:**

- pdflatex installed (exp2729): paper-v6 LaTeX compiles, 3 theory citations added, 27-page arXiv package
- Phase 1 ship: SHIP recommended (exp2730), operator_ship_checklist_v6 complete (HF + IPFS steps added)
- ORCA-NEXUS Tier 3+ viable (exp2733): 30 violations → 17 rules synthesized end-to-end
- FALCON two-layer repair integrated (exp2734): 50.67% candidate reduction
- Test collection clean (exp2726): 5 importorskip fixes, smart-subset still green
- KV260 SSH continuity verified (exp2735): ssh kria reachable, bitstream loaded, uio_count=5,
  uio0_first_word_read=true — board is READY FOR TERMINAL latency transcript task
- OTV probe AUROC=0.25 (worse than random) — retirement queued in .260
- Diversity-maximizing selection lift=-8.5e-6 — retirement queued in .260

---

## Three Biggest Gaps for .260

### Gap 1 (CRITICAL): Verifier energy fix not live-GPU-validated
**exp2727 adversarial flag.** The logging change is durable but the live_gguf claim was fabricated (11.5s
wall time). Until verified on RTX 3090 with Qwen3.6-35B GGUF + duration >= 60s, the "verifier is now
discriminative on live SOTA GGUF" claim is provisional. Blocking paper-v6 §4 claim and Phase 1 production
deployment.

**Addressed by:** exp2740 (Verifier Energy Debug v2 Live GPU).

### Gap 2 (HIGH): Tier 0g Semantic Energy not live-GPU-validated; upstream model may return constant logits
**exp2731 adversarial flag.** Synthetic non-degeneracy confirmed (energies 8.4-9.1). But GGUF inputs
produce identical energy 9.085 — either the TF-IDF of GGUF-formatted responses collapses into one cluster,
or the Qwen3.6 GGUF fast-path returns constant token probabilities for these queries. Root cause needs
diagnosis. Blocking Tier 0g production deployment.

**Addressed by:** exp2741 (Tier 0g Live GPU Re-run + Upstream GGUF Logit Diagnosis).

### Gap 3 (MEDIUM): OTV probe + diversity-selection lineages need formal retirement
**exp2728 (OTV probe_auroc=0.25)** and **exp2732 (diversity_lift=-8.5e-6)** both show selection-stage
gains are exhausted at k=15. Formal retirement prevents carry-forward churn in .261+.

**Addressed by:** exp2739 (OTV + Diversity-Selection Retirement).

---

## Architecture Snapshot

```
Live GGUF (Qwen3.6-35B / Gemma-4-31B)
        |
   [OTV probe retired — use two-tier ODAR routing]
        |
   [ODAR routing (exp2720, two-tier, 65% savings)]
        |     |
        |  [Full ensemble (k=15 + Tier 0g)]
        |
   [Tier 0v: Set-Consistency Energy Network (NEW, exp2743)]
   [Tier 0w: Paraphrastic Consistency Probe (NEW, exp2746)]
        |
   [ORCA-NEXUS online learning (exp2733, FR-11 Tier 3+)]
   [FALCON two-layer repair (exp2734)]
        |
   [Phase 4 FEP factor graph routing (NEW, exp2748)]
```

---

## Hardware Continuity

| Board | State | Next step |
|-------|-------|-----------|
| KV260 | NON-TERMINAL: uio0_first_word_read=true (exp2735) | exp2742: Board-level latency transcript (TERMINAL step) |
| GateMate | TERMINAL (graduated .247) | No further mandatory task |
| PolarFire | TERMINAL (graduated .241) | No further mandatory task |

**KV260 TERMINAL criteria (from CLAUDE.md Hardware-Task Continuity table):**
`board-level latency transcript landed in a non-fabricated artifact + kv260_synthesis_succeeded: true`

exp2742 is designed to meet this criterion. If successful, KV260 graduates to TERMINAL status in .260.

---

## New Research From .260 Literature Sweep

Four papers from the post-.259 planning sweep are directly integrated as .260 experiments:

1. **arXiv:2602.17633** (Weak-Strong Verification) → exp2745
2. **arXiv:2602.11361** (Paraphrastic Consistency) → exp2746
3. **arXiv:2603.20927** (Active Inference FEP Engineering) → exp2748
4. **arXiv:2503.10695** (Set-Consistency Energy Networks) → exp2743

---

## Phase Structure

### Phase A: Archive + Retirement Cleanup (exp2738–exp2739)
- exp2738: Archive .259, activate .260 (15 min)
- exp2739: OTV + diversity-selection retirement (add to exclusion_manifest.yaml) (10 min)

### Phase B: Adversarial Fix Re-runs — Critical Gap Resolution (exp2740–exp2741)
- exp2740: Verifier energy debug v2 — live GPU, RTX 3090, Qwen3.6-35B GGUF, duration >= 60s (45 min)
- exp2741: Tier 0g semantic energy live GPU re-run + GGUF logit diagnosis (45 min)

### Phase C: Hardware Terminal Step (exp2742)
- exp2742: KV260 board-level latency transcript (TERMINAL step) via SSH (30 min)

### Phase D: New Verifier Research (exp2743–exp2746)
- exp2743: Set-Consistency Energy Network Tier 0v (arXiv:2503.10695) (35 min)
- exp2744: Empirical delta computation for 4/delta paper bound (25 min)
- exp2745: Weak-Strong Verification Policy (arXiv:2602.17633) (30 min)
- exp2746: Paraphrastic Consistency Probing Verifier (arXiv:2602.11361) (30 min)

### Phase E: Self-Learning + Phase 4 Research (exp2747–exp2748)
- exp2747: FR-11 Tier 4 continuous self-learning live benchmark (mandatory per research-program.md) (40 min)
- exp2748: Phase 4 Active Inference FEP factor graph (arXiv:2603.20927) (35 min)

### Phase F: Publication + Synthesis (exp2749–exp2750)
- exp2749: Paper v6 arXiv package v2 — empirical delta + new cites (20 min)
- exp2750: Capstone v260 — cross-artifact synthesis + gaps for .261 (claude/opus) (90 min)

---

## Dependency Graph

```
exp2738 (archive) → [exp2739, exp2740, exp2741, exp2742, exp2743, exp2744, exp2745, exp2746, exp2747, exp2748]
exp2740 → exp2747 (FR-11 live benchmark needs discriminative verifier)
exp2744 → exp2749 (paper v2 needs empirical delta)
[all] → exp2750 (capstone)
```

---

## FR-11 Mandate

**FR-11 Tier 2 (NEXUS v2):** COMPLETED in .256 (exp2695). Carry-forward.
**FR-11 Tier 3 (ORCA conformal stopping):** COMPLETED in .258 (exp2719). Carry-forward.
**FR-11 Tier 3+ (ORCA-NEXUS integration):** VIABLE in .259 (exp2733, 17 rules). Carry-forward.
**FR-11 Tier 4 (live benchmark):** NEW in .260 (exp2747). Does online learning measurably improve
verifier accuracy across multiple cycles on live FoVer data? This is the first end-to-end empirical
validation of the self-learning loop with real GPU inference.

---

## Decentralization Compliance

- All experiments use locally-hosted GGUF models (Qwen3.6-35B-A3B / Gemma-4-31B-it / Gemma-4-26B-A4B)
- Phase 4 FEP (exp2748) implements local factor graph computation, no cloud dependencies
- KV260 terminal task (exp2742) runs entirely on attached hardware via SSH
- Paper v6 (exp2749) does not submit to arXiv — OPERATOR-ONLY per CLAUDE.md
- No closed-weight model dependencies in any core verifier module

---

## Agent Routing

| Category | Count | Assignment |
|----------|-------|------------|
| gemini (default) | 12 | exp2738–exp2749 |
| claude/opus (synthesis, requires_claude: true) | 1 | exp2750 capstone |
| **Total** | **13** | |

exp2750 meets all 3 positive criteria for requires_claude:
1. Multi-file tool choreography: reads 12 upstream artifacts + updates ops/status.md, ops/changelog.md, ops/metrics.md
2. Open-ended judgment under ambiguity: synthesis decisions across heterogeneous findings (adversarial flags, hardware results, new verifiers)
3. Cross-context reasoning: must hold all 12 experiment results to produce coherent gap prioritization for .261

---

## Acceptance Criteria for .260

1. `exp2740.verifier_discriminative == true` AND `exp2740.duration_s >= 60` (adversarial flag cleared)
2. `exp2741.gguf_non_degenerate == true` OR `exp2741.gguf_collapse_root_cause` documented
3. `exp2742.kv260_synthesis_succeeded == true` (KV260 TERMINAL criteria met)
4. `exp2743.tier0v_auroc >= 0.65` (Set-Consistency verifier viable)
5. `exp2744.empirical_delta_computed == true` AND `delta_source == 'empirical'`
6. `exp2745.weak_strong_policy_added == true` (routing policy implemented)
7. `exp2746.tier0w_auroc >= 0.55` (paraphrastic consistency viable)
8. `exp2747.learning_loop_closed == true` AND `exp2747.n_learning_cycles >= 3`
9. `exp2748.fep_factor_graph_computed == true` (Phase 4 active inference prototype)
10. `exp2749.paper_v6_submission_package_v2_ready == true`
11. `exp2739.retirements_landed == true` (OTV + diversity added to exclusion_manifest.yaml)
12. `exp2750.n_artifacts_read >= 11` (capstone actually read upstream)

---

## Key Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| exp2740 GPU precondition fails (GGUF not cached) | Low (RTX 3090 on bench) | blocked_model_not_cached verdict; exp2741 still runs |
| exp2742 KV260 SSH unreachable | Low (verified in .259) | blocked_kv260_ssh_unreachable; document and defer terminal |
| exp2747 FR-11 live benchmark adversarial-flagged again | Medium | Explicit duration >= 30s gate + model_specs required |
| exp2748 Phase 4 FEP too abstract for gemini (requires_claude criteria?) | Low (factor graph computation is mechanical) | If blocked, reduce to factor graph structure only (no FEP optimization) |
| Paper v6 arXiv submission still holds until Phase 4 validates | N/A (per CLAUDE.md, hold is correct) | exp2749 produces operator-ready package only |
