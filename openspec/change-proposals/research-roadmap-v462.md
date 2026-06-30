# Research Roadmap v462 — PHASE D (SECOND execution): the FIRST REAL oracle-distinct verifier-moat test

**Milestone:** 2026.06.462
**Planner:** outer-loop Claude Opus 4.8, 2026-06-30 (UTC).
**Status of the program:** PHASE D = the off-ARC distributional-energy verifier moat, now the majority
lever (the ARC-AGI-3 Submission Sprint Forcing Function RETIRED 2026-06-30). `.461` was the first PHASE D
milestone; `.462` is the first milestone in which the decisive question is actually *measured*.

---

## 1. What the previous milestone (.461) proved — and why .462 is a re-execution, not a continuation

`.461` ran all 13 arms (exp5001→exp5013) but **did not test the decisive question.** All three principled
constructions failed to *execute* on attempt 1 — an execution failure, not a scientific null:

| Arm | `.461` outcome | Root cause | Was the moat measured? |
|---|---|---|---|
| **D1 LoRA-EBM** (exp5003) | `running_..._pretrain_skeleton`, n_pairs=0, train_loss=null, duration 0.99s, `flagged_adversarial` (DURATION_TOO_SHORT) | **Bootstrap-and-bail** — wrote a skeleton artifact and stopped *before loading/training the model*. Preconditions ALL passed (4B cached, CUDA up, candidates + FoVer cached). | **No** — never trained. |
| **D2 uPRM** (exp5004) | `blocked_uprm_logprob_candidate_cache`, `flagged_adversarial` | The uPRM scorer + first-error formula are **fully implemented**; it blocked only because the per-token-logprob candidate cache was empty and `CARNOT_UPRM_ENABLE_FRESH_GENERATION=''` (generation disabled). | **No** — never scored. |
| **D3 EBRM** (exp5005) | `complete_ebrm_no_win_musr_plus_0p000`, the SOLE clean arm | **Degenerate**: abstained **97.5%** to a **k=1 "tuned-SC"** strawman, over the weak registry quality ensemble (point-estimate 0.515 < SC 0.585). delta=+0.000 / McNemar p=1.0 is an artifact of always-abstaining-to-SC. | **No** — a real trained verifier was never exercised. |
| **D4 cross-corpus** (exp5006) | `running_..._mmlu_pro_hard_skeleton`, `flagged_adversarial` | Bootstrap-and-bail (same as D1). | **No.** |
| **D5 gate** (exp5007) | **MIXED-SCOPED** (not realized, not bounded-retired) | Correct read: no clean positive MuSR arm + no clean D4 confirmation; D1/D2 nulls not clean → cannot bound-retire either. | n/a |

**The honest state: zero real trained-verifier measurements have landed.** The fabrication gate worked
perfectly (D1/D2/D4 correctly skipped) — the problem was never detection, it was **execution**. The
capstone `next_milestone_pointer` says `tighten_strongest_arm` (EBRM); the deeper read is that "tightening
EBRM" requires the two prerequisites the degeneracy exposed — **a real base scorer (D1) and a genuine
tuned-SC baseline (not k=1)** — both of which `.462` builds. `.461` therefore proved the *harness wiring*
and surfaced *three concrete, individually-fixable execution defects*; `.462` fixes each and measures the
question for the first time.

`.461` E1 SOTA-ingestion (today, discover→ingest→plan→experiment) flagged five fresh papers as the `.462`
inputs (see §8). ARC stayed LOCKED (levels 69 + the publishable FoVer paper); the deepen well is dry
(flat at 69 for 7 milestones) — ARC is now opportunistic.

## 2. The three biggest gaps between current state and the PRD vision

1. **The verifier moat is UNPROVEN and now UNTESTED** (north-star §5 — the existential gap). With the
   generator commodity (open local LLM) and energy-as-generator closed-negative, the *verifier* is
   Carnot's entire value-add, and not one real trained oracle-distinct verifier has yet beaten a genuine
   tuned-SC on a headroom-present domain. `.462` measures it for the first time (D1/D2/D3).
2. **The EFFICIENCY axis is unmeasured** (north-star §5 — "equally effective as the LM at lower
   cost/latency"). Even if every accuracy arm ties SC, a *Pareto* win (equal accuracy at fewer judge calls)
   is still a moat. `.462` adds the uncertainty-routed cheap→judge cascade (2510.20369) to test it.
3. **Continuous self-learning** (PRD FR-11 / research-program Tier 3) must keep advancing — the learned
   verifier should improve across runs. `.462` keeps the ARC self-play checkpoint loop alive (E2).

## 3. Architecture — where PHASE D sits

```
            PHASE D — oracle-distinct verifier moat (OFF ARC, headroom-present reasoning corpora)
            ┌──────────────────────────────────────────────────────────────────────────────┐
 INFRA      │  B1 genuine tuned-SC baseline (K-way majority vote, K swept + reported;        │
 (run 1st)  │     always-abstain degeneracy guard)        B2 shared logprob-enriched          │
            │                                              candidate cache (MuSR + 2nd corpus) │
            └───────────────┬───────────────────────────────────────┬──────────────────────┘
                            │ (genuine SC baseline)                  │ (K candidates + per-token logprobs)
            ┌───────────────▼───────────────────────────────────────▼──────────────────────┐
 ACCURACY   │  D1 LoRA-EBM (TRAIN: Qwen3.5-1.7B + LoRA + energy head, contrastive on FoVer +  │
 arms       │      gold-labeled candidate pairs; EVAL min-energy select vs genuine tuned-SC)  │
            │  D2 uPRM (next-token-prob first-error process score; UNSUPERVISED selector)     │
            │  D3 EBRM (refine the D1 trained scorer; uncertainty head, NON-degenerate        │
            │      abstention) ── gated_on D1 scorer_trained                                  │
            └───────────────┬──────────────────────────────────────────────────────────────┘
                            │ best oracle-distinct verifier (by delta vs genuine tuned-SC on MuSR)
 EFFICIENCY ┌───────────────▼──────────────┐   CROSS-CORPUS ┌──────────────────────────────┐
 arm        │  CASCADE: cheap verifier      │   D4: best arm  │  on a 2nd headroom-present    │
            │  selects when confident, routes│   on a 2nd corpus│  oracle-distinct corpus       │
            │  uncertain → strong judge;     │   (GPQA /        │  (generalization confirmation)│
            │  charge judge-call budget      │   MMLU-Pro-hard) └──────────┬───────────────────┘
            └───────────────┬───────────────┘                            │
                            └──────────────────────┬─────────────────────┘
                                                   ▼
                                  D5 GATE: realize / bounded-retire / scoped
                                  + DiffusionGemma-gate status (operator-gated)
```

The verifier is **oracle-distinct** everywhere (`verifier_is_oracle=False`): it scores reasoning quality
and NEVER reads gold/answer_index/model_id at inference. This is the non-circular moat test the ARC
generation wall could not host (per CLAUDE.md "Circularity / Oracle-Distinctness Discipline").

## 4. The DiffusionGemma gate this thread resolves

`docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md` is **STILL-PENDING**. Its three
conditions (headroom present + a non-trivial oracle-distinct verifier + a matched control with CI95
excluding 0) are exactly the PHASE D gate. A POSITIVE PHASE D arm satisfies the conditions *on the tested
domain* — but **activation stays operator-gated**: `.462` records
`diffusiongemma_gate_conditions_satisfied_off_arc` honestly and never autonomously flips the gate to MET
(per the circularity discipline). ARC's ~13pp headroom remains the canonical un-captured target.

## 5. Phases and the falsifiable gate

- **PHASE 0 (transition):** archive `.461 → activate `.462; assert the active YAML parses + own-test
  pre-test gate (`--no-cov`); record the `.461 close-state (MIXED-SCOPED moat; the three execution defects).
- **PHASE B (infra, 2 reserved slots, run FIRST):** B1 genuine tuned-SC baseline + degeneracy guard;
  B2 shared logprob-enriched candidate cache (unblocks D2, feeds D1/D3/cascade; anti-churn).
- **PHASE D (verifier moat, the majority):** D1 train+eval LoRA-EBM; D2 uPRM (unblocked); D3 EBRM
  (non-degenerate, gated on D1); the EFFICIENCY cascade; D4 cross-corpus; D5 gate aggregation.
- **PHASE C (hardware):** KV260 SSH-only continuity.
- **PHASE E (standing):** E1 SOTA-ingestion (map onto `.463); E2 ARC self-play (continuous self-learning /
  FR-11); E3 opportunistic ARC level-up (honest no-bank if dry).
- **CAPSTONE:** aggregate the moat verdict + the `.463 pointer.

**The falsifiable gate (the only non-circular evidence):** on a headroom-present ORACLE-DISTINCT domain
(`verifier_is_oracle=False`, `headroom_present=True` = oracle@K − **genuine** tuned-SC ≥ 0.10 ∧ flips>0),
≥1 of {trained LoRA-EBM (D1), uPRM (D2), EBRM (D3)} beats **genuine tuned-SC** with paired CI95 excluding
0 (McNemar p<0.05), confirmed on MuSR ∧ ≥1 second corpus (D4). OR the EFFICIENCY arm achieves accuracy
parity (within CI) at a materially lower judge-call budget (north-star §5 Pareto win).
`retire_if_same_verdict: true` — if the **properly-executed** LoRA-EBM (D1) AND uPRM (D2) both null with
CI95 including 0 on every headroom-present oracle-distinct corpus, the off-ARC accuracy-moat retires as
bounded (a publishable null converging with the ARC tie). A degenerate or skeleton arm is a FAILED
execution, NOT a null — it does not trigger retirement (this is the `.461 lesson made mechanical).

## 6. Dependency graph

```
exp5014 (transition)
   └─> exp5015 (B1 genuine-SC baseline)  ─┐
   └─> exp5016 (B2 logprob cache)        ─┤ (infra, both feed the D arms)
                                          ├─> exp5017 (D1 LoRA-EBM train+eval)
                                          │       └─> exp5019 (D3 EBRM)  [gated_on D1 scorer_trained]
                                          ├─> exp5018 (D2 uPRM)          [uses B2 logprob cache; LC-ERD fallback]
                                          └─> exp5020 (cascade efficiency)
   exp5017 / exp5018 / exp5019 ──(best arm)──> exp5021 (D4 cross-corpus)
   exp5017..exp5021 ──────────────────────────> exp5022 (D5 gate) ──> exp5027 (capstone)
exp5023 (C KV260) · exp5024 (E1 SOTA) · exp5025 (E2 self-play) · exp5026 (E3 ARC) ──> exp5027 (capstone)
```

## 7. Hardware requirements

- **Offline training + scoring on the conductor's dedicated GPU-0 CUDA device** (2026-06-27 allocation;
  `CARNOT_ARC_GENERATOR_CUDA_GPU=0`, drop-in `40-arc-generator-3090-20260619.conf`). The D arms train a
  small QLoRA scorer + run live generation-with-logprobs; do NOT iGPU-pin (the iGPU constraint is for the
  *live ARC submission stack only*, not offline induction/training — CLAUDE.md ARC-sprint GPU rule).
- **KV260** via SSH-only (`ssh kria`), never the host SD-card path (KV260 SSH-Not-SD-Card Discipline).

## 8. Models (CLAUDE.md SOTA-models rule)

- **Generator (candidate + logprob source):** `unsloth/gemma-4-12B-it-GGUF` on the GPU-0 CUDA llama-server
  (the .461 D2 confirmed it returns `top_logprobs`). The lightweight SOTA option — fast enough for the
  per-question K-candidate generation B2 needs.
- **Trainable verifier base (D1 LoRA-EBM):** `Qwen/Qwen3.5-1.7B` **base** repo (NOT a `-GGUF` repo per the
  GGUF-tokenizer rule) + LoRA + a scalar energy head. Smaller than `.461's 4B choice → trains + evals in a
  single codex session (the structural fix for the D1 bail).
- **`.462 SOTA inputs (E1-ingested, see research-references.md V462 block):** 2606.19818 (UARM uncertainty
  head), 2606.09073 (distributional-pessimistic), 2602.24040 (RewardUQ calibration), 2510.20369
  (uncertainty-routed cascade — the efficiency arm), 2605.24005 (LC-ERD — the D2 unblock fallback).

## 9. What this milestone does NOT do (do-not-re-propose ledger)

- **NOT** energy-as-ARC (S0 program CONCLUDED 2026-06-26 — no live ARC value); NOT the ARC dynamics-engine
  L2 wall, macro/horizon-collapse, click-heatmap, trust-gate, MATM retrieval (NULLED .454), TTT-code-engine,
  local code inducers, decision-need, action-prefix latents, perception-from-grid, representation #5.
- **NOT** the verifier-as-REWARD path (exp4247/exp4263, retired): PHASE D trains the verifier/scorer and
  uses it as an oracle-distinct SELECTOR, never as an RFT reward to train a policy. Each D arm carries
  `operator_override` for the substring scope-match + a `prior_failures` block for its `.461 predecessor.
- **NOT** a new ARC majority — ARC is opportunistic post-sprint (E2 self-play + E3 one deepen).
- **NOT** an autonomous DiffusionGemma activation — a POSITIVE records the satisfied conditions; activation
  stays operator-gated.

## 10. Agent routing

All experiment tasks `agent_type: codex` / `gpt-5.5` (the 2026-06-30 PHASE D directive + standing
quota-conserve default; `CODEX_FORCE_EXPERIMENTS=1`). Planner + retro stay Claude Opus 4.8
(`AGENT_TYPE_PLANNER`/`AGENT_TYPE_RETRO`) — the operator's deliberate quality choice. The D1 training arm
is de-risked *structurally* (smaller 1.7B base + shared cache + train-first-not-skeleton-first + a gate
that REQUIRES train_loss non-null) rather than by routing off codex; if it bails a second time,
`retire_if_same_verdict` surfaces it to the operator (per the `.461 retro's own escalation note).
