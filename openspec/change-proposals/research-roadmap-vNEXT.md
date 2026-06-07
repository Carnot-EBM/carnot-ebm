# Research Roadmap — Milestone 2026.06.362

**Recover the .361 infra wash and FINISH the offline verifier proof.**

**Author:** Outer-loop (Claude Opus 4.8), 2026-06-07.
**Milestone doc for:** `research-roadmap-next.yaml` (10 tasks, exp3914–exp3923).
**Status:** Pre-staged per the Pre-Staged Roadmap Convention.

---

## 1. What the previous milestone (.361) proved — and didn't

.361 was scoped exactly right ("Prove the verifier earns its place" — north-star §5,
the project's TOP priority) but **executed as a near-total infra wash: only 3 of ~11
tasks produced artifacts.** Two infrastructure failures — not science — caused it:

1. **The recurring live-35B-GGUF inference blocker.** exp3904 (the *critical* ACCURACY
   axis / moat scissor) returned `blocked_llama_cpp_inference_failed`. CUDA was up, the
   Qwen3.6-35B GGUF was cached, `carnot.verify` imported — but the inference **call
   itself** failed (a 35B Q4_K_M at `n_gpu_layers=-1` full offload under llama_cpp
   0.3.23). The moat scissor has now been BLOCKED/INCONCLUSIVE **three milestones
   running** (.359 exp3885, .360 exp3895, .361 exp3904), *always* on the live-model path.

2. **The agent-shipped poison-test cascade** (memory `incident_agent_shipped_test_cascade`).
   exp3905 (cost harness) shipped `tests/python/test_cost_instrumented_verification.py`
   with `assert artifact["duration_s"] >= 60` — but that test runs the harness on a
   **10-item fixture** that legitimately completes in ~35.8s. The assertion fails, so the
   conductor's PRE-TEST gate reported "1 failed, 105 passed" and **SKIP-cascaded every
   task after exp3905**: efficiency (exp3906), cascade (exp3907), ARC scaffold (exp3908),
   facts (exp3909), FR-11 v25 (exp3910), both hardware tasks, **and the capstone**. The
   60s adversarial-verify floor is for the *full-corpus science artifact*, never for a
   tiny unit-test fixture.

**Net:** the single most important question in the project — *does the cheap verifier
earn its place?* — still has no clean answer. The blockers are addressable infra, and
the prior tasks were SKIPPED (never ran), not failed-with-verdict.

**Strategic context that is settled (carried in, unchanged):** both energy-core theses
are bounded-negative — energy-as-*selection* (P0.1) and energy-as-*generation* (EBT) —
so the **VERIFIER is the surviving asset** (north-star §5). `paper_ready` is TRUE (G1–G4
met; FoVer 0.9131 frozen, independently reproduced via CI run 26725185125). This
milestone does not touch the headline; it finishes the lens that decides Carnot's value.

## 2. The three biggest gaps between current state and the PRD vision

1. **The verifier's value is UNPROVEN on both axes.** ACCURACY (the moat: does the
   external verifier catch what self-verification misses, in-distribution, vs *strong*
   self-verify?) and EFFICIENCY (the operator win condition: equally effective at lower
   cost/latency — "parity at Nx cheaper"). Neither has a clean landed number. **Phase 1
   closes this.**
2. **The live-model critical path is fragile.** The same inference call has blocked the
   decisive result three times. Until a robust, tested, reusable inference entrypoint
   exists, every live-model milestone is one runtime exception away from a wash. **Phase 0
   fixes this at the root (harness-first).**
3. **No agentic integration surface, and facts still math-bound.** The ARC-AGI-3
   agentic-proof venue (sequenced SECOND per north-star §5) has not been started beyond a
   toy pilot; and the verifier remains math-domain-bound (facts earned-negative, three
   fabricated attempts). **Phase 2 starts the scaffold (GPU-free); Phase 3 gives facts
   one last disciplined retry.**

## 3. Milestone design — 10 tasks across 5 phases

```
PHASE 0 — record + UNBLOCK (break the cascade, fix the root cause)
  exp3914  archive .361 -> activate .362; QUARANTINE the poison test; green-gate
  exp3915  BUILD+TEST robust gguf_inference harness (fallback chain + 1-token smoke)   [GPU]
PHASE 1 — the offline verifier proof (re-run on the hardened harness)
  exp3916  ACCURACY — moat scissor (weak+strong arms, decoupled gate)   [GPU]  prior_failures
  exp3917  EFFICIENCY — energy-verifier vs LLM-judge head-to-head        [GPU]
  exp3918  Meta-EBM cascade router prototype (reuses exp3917 per-item scores; no re-inference)
PHASE 2 — agentic proof venue scaffold (verifier-first, GPU-free)
  exp3919  ARC-AGI-3 harness scaffold — build+unit-test (infra-only, synthetic, no science)
PHASE 3 — facts (PRD Tier C; LAST disciplined retry)
  exp3920  facts graph-grounding — last retry via robust harness   [GPU]  prior_failures, retire_if_same
PHASE 4 — mandates + hardware + capstone
  exp3921  FR-11 v25 online independence-reweighting (research-program.md MANDATE)
  exp3922  hardware continuity consolidated (GateMate + PolarFire + KV260)
  exp3923  capstone .362 — the VERIFIER SCORECARD (answers "earns its place?")
```

### Dependency graph (all disk-read fallbacks — NO hard `gated_on` on the critical path)

```
exp3914 (green gate) ──> unblocks the whole milestone (pre-test gate green)
exp3915 (gguf harness) ──> exp3916, exp3917, exp3920   (each disk-reads exp3915; blocked_upstream_* if absent)
exp3917 (efficiency)   ──> exp3918 (cascade reuses per-item scores)
exp3916..exp3922 ──> exp3923 (capstone aggregates whatever landed, skipping flagged)
```

Per the .340/.358 lesson, **no downstream task hard-gates on an upstream**: each
disk-reads its prerequisite and emits `blocked_upstream_*` if absent, so a single
skipped upstream costs ONE task, never a cascade. The poison-test fix (exp3914) plus
the robust inference harness (exp3915) are the two structural changes that make this
milestone robust where .361 was not.

### Why harness-first, again

.360 and .361 both showed that the failure mode is **a fragile live-model build masked
as science.** The fix is the same one that worked for exp3894 (the tested reasoner
judge): ship a *passing unit test on a fixture* as the deliverable, with the science as
a separate, downstream step. exp3915 generalizes this to the inference layer itself so
the decisive results (exp3916/3917/3920) cannot block on a raw `llama_cpp.Llama` call
again. **Crucially, the fixture unit tests assert CORRECTNESS only — never a wall-clock
floor (the exp3905 poison-test root cause).** The 60s live floor is asserted only on the
full-corpus science artifacts (exp3916/3917/3920).

## 4. New literature folded in (2026-06-07 sweep → `research-references.md`)

- **arXiv:2510.14913** Budget-aware Discriminative Verification — cleanest precedent for
  the EFFICIENCY axis (single-forward-pass verifier beats generative at fixed compute).
  Cited in exp3917.
- **arXiv:2605.17609** Adaptive Generate-Rank-Verify (ADAP) — cost-optimality theory for
  the cascade router. Cited in exp3918.
- **arXiv:2605.30290** Self-Trained Verification — quantifies the self-verification blind
  spot AND the inflated-verifier-score failure mode (ACCURACY axis).
- **arXiv:2510.13744** Hard2Verify — a harder-than-FoVer step-level moat venue (future
  corpus extension).
- **arXiv:2603.04304** V₁ pairwise self-verify — the toughest o1-subsumption threat; a
  future strong-arm variant for the moat scissor.
- **arXiv:2604.13717** Cost-effective LLM-as-judge — the EFFICIENCY moving target (beat a
  cheap ensembled-small-judge, not only a single frontier judge). Cited in exp3917.
- **arXiv:2603.24621** ARC-AGI-3 official — canonical citation for the exp3919 scaffold.
- **arXiv:2602.19643** KGHaluBench — graded KG fact testbed for exp3920.

## 5. Self-learning coverage (research-program.md requirement)

exp3921 (FR-11 v25) is the continuous-self-learning task: Tier-1 online
independence-reweighting, loading the persisted v24 state (exp3888 landed clean,
INVARIANT_HELD), continuing on a fresh corpus slice, confirming the learned weighting
holds the +0.0185 memory-ablation contribution and the frozen 0.9131 across the
v24→v25 iteration. CPU counter updates (<1µs/update — the Tier-1 hardware path).

## 6. Hardware requirements

- **Phase 0/1/3 (exp3915/3916/3917/3920):** 1× RTX 3090 sufficient (the robust harness
  prefers gemma-4-26B-A4B-it, the cheaper headline-eligible MoE, exactly because the 35B
  full-offload failed in exp3904). `requires_gpu: true`, `{project_root}/.venv/bin/python`.
- **Phase 2/4 (exp3918/3919/3921/3923):** CPU-only (aggregation / synthetic env / verifier
  scoring). Robust by construction.
- **exp3922 hardware:** GateMate via DirtyJTAG; PolarFire + KV260 via SSH (KV260
  SSH-not-SD-card discipline). Per-board preconditions — one board down does not fail the task.

## 7. Discipline checklist (planner self-audit)

- [x] **Gemini-Default / routing:** all tasks `codex` + `requires_codex` + `gpt-5.5`
      (anti-wipeout; standing operator gemini↔codex flip authority 2026-06-05).
- [x] **Verdict Terminal-Prefix:** every `honest_verdict` starts `complete:`/`success:`/
      `blocked_<resource>`.
- [x] **Failed-Experiment Rerun + Exclusion-Manifest:** `prior_failures` (all 4 sub-fields)
      on the two scope-matched tasks (exp3916 moat → exp3904/exp3895; exp3920 facts →
      exp3896/exp3886 with `retire_if_same_verdict: true`). `operator_override` on every
      task (routine transitions / mandates / continuations).
- [x] **Pre-Launch Preconditions:** every compute-bound task has a step-0 PRECONDITIONS
      block with `blocked_<resource>` exits; disk-read fallbacks (no hard `gated_on`).
- [x] **Principle-Annotated Artifact Fields:** every REQUIRED ARTIFACT FIELD carries a
      `principle:` line; values emitted BARE (no `{value,principle}` wrapper — exp3871 bug).
- [x] **Adversarial-Verify / Sample-Size:** live-model artifacts assert `duration_s>=60`
      on the FULL-CORPUS run (never a fixture); seeds + checksum + model_specs required.
- [x] **Hardware-Task Continuity:** one consolidated hardware slot (north-star §3
      opportunistic relaxation).
- [x] **Self-learning:** exp3921 present (FR-11 v25).
- [x] **Invariants:** `paper_ready` stays TRUE; frozen 0.9131 never substituted; never
      aggregate `flagged_adversarial`; EBT replication superseded/dropped; no external
      publication.

## 8. Success criteria for .362

The milestone succeeds if it produces, on real data and non-flagged:
1. A clean **moat verdict** (MOAT_SURVIVES / SUBSUMED / INCONCLUSIVE) from a scissor whose
   inference path actually ran — ending the 3-milestone block.
2. A clean **efficiency verdict** with a measured `cost_ratio_walltime` — the first-ever
   "parity at Nx cheaper" number.
3. A **cascade-router** prototype number (WINS / MARGINAL).
4. A **green pre-test gate** throughout (poison test quarantined → no SKIP cascade).

A capstone that can finally state `verifier_earns_its_place: <bool>` from measured
numbers — rather than INCONCLUSIVE for a fourth time — is the milestone's purpose.
